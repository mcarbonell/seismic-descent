"""
anisotropic_hmc.py — Paso 3: Adaptación Anisótropa de Longitud de Onda y Dinámica Hamiltoniana (HMC).

Combines:
1. Anisotropic spatial metric adaptation: Estimates diagonal Riemannian metric / Hessian
   diagonal from swarm gradient history, adjusting effective lengthscales per dimension.
2. Hamiltonian momentum dynamics: Implements symplectic momentum integration with
   phase-modulated inertial mass, allowing particles to traverse ill-conditioned,
   tortuous ravines (like Rosenbrock) without zig-zag oscillations.
3. Full integration with phase-modulated swarm gravity and seismic wave fields.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from seismic_descent.rff import RandomFourierFeatures
from seismic_descent.lissajous import LissajousWaveField, OrthogonalLissajousWaveField


class SeismicAnisotropicHMC:
    """
    Seismic Descent with Anisotropic Metric Adaptation and Hamiltonian Dynamics.

    Parameters
    ----------
    bounds : array-like of shape (D, 2)
        Lower and upper bounds for each dimension in original search space.
    n_particles : int, default=10
        Number of parallel particles exploring the shared seismic landscape.
    n_steps : int, default=2000
        Number of optimization steps.
    n_cycles : int, default=10
        Number of complete seismic earthquake oscillation cycles.
    dt_base : float, default=0.2
        Maximum base step size in the normalized [-1, 1]^D domain.
    dt_floor : float, default=0.2
        Minimum step size scale floor in the cyclic schedule.
    noise_amplitude : float, default=0.5
        Noise gradient amplitude relative to unit directional gradient.
    noise_decay : float, default=1.0
        Global noise amplitude decay factor per step.
    dt_cycles_multiplier : float, default=5.0
        Frequency multiplier for cyclic learning rate relative to seismic phase.
    momentum_base : float, default=0.7
        Base momentum / inertial coefficient mu in [0, 1).
    anisotropic_power : float, default=0.5
        Strength of anisotropic preconditioning alpha in [0, 1].
        alpha=0 corresponds to isotropic; alpha=0.5 corresponds to RMSProp/HMC metric scaling.
    metric_beta : float, default=0.9
        Exponential moving average coefficient for diagonal metric estimation.
    gravity_strength : float, default=0.4
        Maximum attraction coefficient toward x_best.
    gravity_mode : str, default="phase_modulated"
        Modulation strategy: "phase_modulated", "constant", "none".
    noise_type : str, default="rff"
        Generator: "rff", "orthogonal_lissajous", "lissajous", or "none".
    noise_frequencies : int, default=64
        Number of features / rotation bases.
    base_lengthscale : float, default=0.4
        Base spatial lengthscale in normalized coordinates.
    seed : Optional[int], default=1
        Random seed for reproducibility.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        n_particles: int = 10,
        n_steps: int = 2000,
        n_cycles: int = 10,
        dt_base: float = 0.2,
        dt_floor: float = 0.2,
        noise_amplitude: float = 0.5,
        noise_decay: float = 1.0,
        dt_cycles_multiplier: float = 5.0,
        momentum_base: float = 0.7,
        anisotropic_power: float = 0.5,
        metric_beta: float = 0.9,
        gravity_strength: float = 0.4,
        gravity_mode: str = "phase_modulated",
        noise_type: str = "rff",
        noise_frequencies: int = 64,
        base_lengthscale: float = 0.4,
        seed: Optional[int] = 1,
    ):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be of shape (D, 2)")

        self.dim = self.bounds.shape[0]
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.n_cycles = n_cycles
        self.dt_base = dt_base
        self.dt_floor = dt_floor
        self.noise_amplitude = noise_amplitude
        self.noise_decay = noise_decay
        self.dt_cycles_multiplier = dt_cycles_multiplier
        self.momentum_base = momentum_base
        self.anisotropic_power = anisotropic_power
        self.metric_beta = metric_beta
        self.gravity_strength = gravity_strength
        self.gravity_mode = gravity_mode.lower()
        self.noise_type = noise_type.lower()
        self.base_lengthscale = base_lengthscale
        self.seed = seed

        self.center = (self.bounds[:, 1] + self.bounds[:, 0]) / 2.0
        self.half_range = (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0
        if np.any(self.half_range <= 0):
            raise ValueError("All upper bounds must be strictly greater than lower bounds.")

        # Perturbation field
        if self.noise_type == "rff":
            self.noise_field = RandomFourierFeatures(
                dim=self.dim,
                r=noise_frequencies,
                base_lengthscale=base_lengthscale,
                seed=seed,
            )
        elif self.noise_type in ("orthogonal_lissajous", "ortho_lissajous"):
            n_rot = max(2, min(noise_frequencies, 6))
            self.noise_field = OrthogonalLissajousWaveField(
                dim=self.dim,
                n_rotations=n_rot,
                base_lengthscale=base_lengthscale,
            )
        elif self.noise_type == "lissajous":
            self.noise_field = LissajousWaveField(
                dim=self.dim,
                coupled=True,
                base_lengthscale=base_lengthscale,
            )
        elif self.noise_type == "none":
            self.noise_field = None
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

    def _compute_gravity_factor(self, amp_ratio: float) -> float:
        if self.gravity_strength <= 0.0 or self.gravity_mode == "none":
            return 0.0
        if self.gravity_mode == "phase_modulated":
            return self.gravity_strength * max(0.0, 1.0 - abs(amp_ratio))
        elif self.gravity_mode == "constant":
            return self.gravity_strength
        return 0.0

    def optimize(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        fn_grad: Callable[[np.ndarray], np.ndarray],
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, List[float]]]:
        """
        Run Seismic Anisotropic HMC optimization.
        """
        rng = np.random.default_rng(self.seed)
        x_norm = rng.uniform(-1.0, 1.0, size=(self.n_particles, self.dim))

        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=np.float64)
            x0_norm = (x0_arr - self.center) / self.half_range
            x_norm[0] = np.clip(x0_norm, -1.0, 1.0)

        # Hamiltonian momentum velocities
        velocities = np.zeros((self.n_particles, self.dim), dtype=np.float64)

        # Diagonal metric tensor tracking (EMA of squared gradients)
        diag_metric = np.ones(self.dim, dtype=np.float64)

        t = 0.0
        dt_noise = (self.n_cycles * np.pi) / self.n_steps

        # Evaluate initial state
        x_real = self.center + x_norm * self.half_range
        real_vals = np.asarray(fn(x_real), dtype=np.float64)
        if real_vals.ndim == 0:
            real_vals = np.array([real_vals])

        best_idx = int(np.argmin(real_vals))
        best_val = float(real_vals[best_idx])
        best_x_real = x_real[best_idx].copy()
        best_x_norm = x_norm[best_idx].copy()

        best_per_step: List[float] = [best_val]

        for step in range(self.n_steps):
            decay = self.noise_decay ** step
            freq = 2.0 * decay
            sin_phase = np.sin(t * freq)
            amp = self.noise_amplitude * decay * sin_phase

            # 1. Real objective gradient
            x_real = self.center + x_norm * self.half_range
            f_grad_real = np.asarray(fn_grad(x_real), dtype=np.float64)
            if f_grad_real.ndim == 1:
                f_grad_real = f_grad_real.reshape(1, -1)

            # 2. Chain rule mapping
            f_grad_mapped = f_grad_real * self.half_range

            # 3. Update diagonal Riemannian metric from swarm gradients
            swarm_sq_grad = np.mean(f_grad_mapped ** 2, axis=0)  # Shape (D,)
            diag_metric = self.metric_beta * diag_metric + (1.0 - self.metric_beta) * swarm_sq_grad

            # 4. Compute Anisotropic Preconditioner D (rescaled to median 1.0)
            if self.anisotropic_power > 0.0:
                scale_metric = np.sqrt(diag_metric) + 1e-8
                med_scale = np.median(scale_metric)
                if med_scale > 1e-8:
                    rel_scale = scale_metric / med_scale
                else:
                    rel_scale = np.ones_like(scale_metric)
                # Preconditioning factor per dimension
                precond = 1.0 / (rel_scale ** self.anisotropic_power)
                precond = np.clip(precond, 0.1, 10.0)  # Safe guardrails against extreme skew
            else:
                precond = np.ones(self.dim, dtype=np.float64)

            # 5. Anisotropic Directional Gradient
            precond_grad = f_grad_mapped * precond  # Shape (N, D)
            norms = np.linalg.norm(precond_grad, axis=1, keepdims=True)
            f_grad_dir = np.where(norms > 1e-8, precond_grad / norms, 0.0)

            # 6. Perturbation field gradient
            if self.noise_field is not None:
                noise_grad = self.noise_field.grad(x_norm, t, amplitude=amp)
            else:
                noise_grad = 0.0

            # 7. Swarm Gravitational Attraction
            gamma = self._compute_gravity_factor(sin_phase)
            if gamma > 0.0:
                diff = best_x_norm - x_norm
                diff_norms = np.linalg.norm(diff, axis=1, keepdims=True)
                safe_norms = np.maximum(diff_norms, 1e-7)
                grav_dir = np.where(diff_norms > 1e-7, diff / safe_norms, 0.0)
                grav_pull = gamma * grav_dir
            else:
                grav_pull = 0.0

            # 8. Total force / gradient
            f_total = -(f_grad_dir + noise_grad - grav_pull)

            # 9. Hamiltonian Momentum Integration
            # Phase-modulated friction: during calm, friction dampens momentum for fine convergence;
            # during quake, momentum carries particles over high barriers.
            mu = self.momentum_base * (0.5 + 0.5 * abs(sin_phase))
            velocities = mu * velocities + (1.0 - mu) * f_total

            # 10. Cyclic step size
            cyclic_scale = self.dt_floor + (1.0 - self.dt_floor) * np.abs(np.sin(t * self.dt_cycles_multiplier))
            current_dt = self.dt_base * cyclic_scale

            # Update positions and project onto bounds
            x_norm += current_dt * velocities
            np.clip(x_norm, -1.0, 1.0, out=x_norm)

            t += dt_noise

            # Track global best point
            x_real = self.center + x_norm * self.half_range
            step_vals = np.asarray(fn(x_real), dtype=np.float64)
            if step_vals.ndim == 0:
                step_vals = np.array([step_vals])

            step_best_idx = int(np.argmin(step_vals))
            step_best_val = float(step_vals[step_best_idx])
            if step_best_val < best_val:
                best_val = step_best_val
                best_x_real = x_real[step_best_idx].copy()
                best_x_norm = x_norm[step_best_idx].copy()

            best_per_step.append(best_val)

        return best_x_real, best_val, {"best_per_step": best_per_step}


def seismic_anisotropic_hmc(
    fn: Callable[[np.ndarray], np.ndarray],
    fn_grad: Callable[[np.ndarray], np.ndarray],
    x0_real: np.ndarray,
    bounds: np.ndarray,
    n_steps: int = 2000,
    n_particles: int = 10,
    dt_base: float = 0.2,
    dt_floor: float = 0.2,
    noise_amplitude: float = 0.5,
    noise_decay: float = 1.0,
    dt_cycles_multiplier: float = 5.0,
    momentum_base: float = 0.7,
    anisotropic_power: float = 0.5,
    gravity_strength: float = 0.4,
    gravity_mode: str = "phase_modulated",
    noise_type: str = "rff",
    seed: Optional[int] = 1,
) -> Tuple[np.ndarray, float, Dict[str, List[float]]]:
    """Functional interface for Seismic Anisotropic HMC optimizer."""
    opt = SeismicAnisotropicHMC(
        bounds=bounds,
        n_particles=n_particles,
        n_steps=n_steps,
        dt_base=dt_base,
        dt_floor=dt_floor,
        noise_amplitude=noise_amplitude,
        noise_decay=noise_decay,
        dt_cycles_multiplier=dt_cycles_multiplier,
        momentum_base=momentum_base,
        anisotropic_power=anisotropic_power,
        gravity_strength=gravity_strength,
        gravity_mode=gravity_mode,
        noise_type=noise_type,
        seed=seed,
    )
    return opt.optimize(fn=fn, fn_grad=fn_grad, x0=x0_real)
