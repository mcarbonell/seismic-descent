"""
champion_v23.py — Seismic Descent Champion Architecture (v23).

Unified state-of-the-art synthesizer combining:
1. Orthogonal Random Features (ORF) or Orthogonal Lissajous Wave Fields.
2. Phase-Modulated Swarm Gravitational Coupling (cooperative basin exploitation).
3. Phase-Gated Symplectic Hamiltonian Momentum (valley navigation without wall-bouncing).
4. Adaptive Anisotropic Riemannian Metric Preconditioning (curved canyon adaptation).
5. Decoupled Cyclic Step Schedule with minimum floor.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from seismic_descent.rff import RandomFourierFeatures
from seismic_descent.orf import OrthogonalRandomFeatures
from seismic_descent.lissajous import LissajousWaveField, OrthogonalLissajousWaveField


class SeismicChampionV23:
    """
    Seismic Descent Champion v23 Optimizer.

    Parameters
    ----------
    bounds : array-like of shape (D, 2)
        Lower and upper bounds for each dimension.
    n_particles : int, default=10
        Number of parallel particles exploring the landscape.
    n_steps : int, default=2000
        Number of optimization steps.
    n_cycles : int, default=10
        Number of seismic earthquake cycles.
    dt_base : float, default=0.2
        Maximum base step size in normalized [-1, 1]^D coordinates.
    dt_floor : float, default=0.2
        Minimum scale floor for cyclic step schedule.
    noise_amplitude : float, default=0.5
        Noise gradient amplitude relative to unit directional gradient.
    noise_decay : float, default=1.0
        Decay factor for noise amplitude per step.
    dt_cycles_multiplier : float, default=5.0
        Frequency multiplier for cyclic learning rate schedule.
    noise_engine : str, default="orf"
        Landscape generator: "orf", "orthogonal_lissajous", "rff", "lissajous", or "none".
    noise_frequencies : int, default=64
        Number of random features or orthogonal rotation bases.
    base_lengthscale : float, default=0.4
        Base spatial lengthscale in normalized coordinates.
    gravity_strength : float, default=0.4
        Maximum swarm gravitational attraction coefficient gamma_0.
    momentum_base : float, default=0.7
        Base momentum / inertial coefficient mu in [0, 1).
    anisotropic_power : float, default=0.5
        Exponent alpha for Riemannian metric preconditioning.
    metric_beta : float, default=0.9
        Moving average factor for metric tracking.
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
        noise_engine: str = "orf",
        noise_frequencies: int = 64,
        base_lengthscale: float = 0.4,
        gravity_strength: float = 0.4,
        momentum_base: float = 0.7,
        anisotropic_power: float = 0.5,
        metric_beta: float = 0.9,
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
        self.noise_engine = noise_engine.lower()
        self.noise_frequencies = noise_frequencies
        self.base_lengthscale = base_lengthscale
        self.gravity_strength = gravity_strength
        self.momentum_base = momentum_base
        self.anisotropic_power = anisotropic_power
        self.metric_beta = metric_beta
        self.seed = seed

        self.center = (self.bounds[:, 1] + self.bounds[:, 0]) / 2.0
        self.half_range = (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0
        if np.any(self.half_range <= 0):
            raise ValueError("All upper bounds must be strictly greater than lower bounds.")

        # Initialize perturbation engine
        if self.noise_engine == "orf":
            self.noise_field = OrthogonalRandomFeatures(
                dim=self.dim,
                r=noise_frequencies,
                base_lengthscale=base_lengthscale,
                seed=seed,
            )
        elif self.noise_engine in ("orthogonal_lissajous", "ortho_lissajous"):
            n_rot = max(2, min(noise_frequencies, 6))
            self.noise_field = OrthogonalLissajousWaveField(
                dim=self.dim,
                n_rotations=n_rot,
                base_lengthscale=base_lengthscale,
            )
        elif self.noise_engine == "rff":
            self.noise_field = RandomFourierFeatures(
                dim=self.dim,
                r=noise_frequencies,
                base_lengthscale=base_lengthscale,
                seed=seed,
            )
        elif self.noise_engine == "lissajous":
            self.noise_field = LissajousWaveField(
                dim=self.dim,
                coupled=True,
                base_lengthscale=base_lengthscale,
            )
        elif self.noise_engine == "none":
            self.noise_field = None
        else:
            raise ValueError(f"Unknown noise_engine: {self.noise_engine}")

    def optimize(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        fn_grad: Callable[[np.ndarray], np.ndarray],
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, List[float]]]:
        """
        Run Seismic Champion v23 optimization.
        """
        rng = np.random.default_rng(self.seed)
        x_norm = rng.uniform(-1.0, 1.0, size=(self.n_particles, self.dim))

        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=np.float64)
            x0_norm = (x0_arr - self.center) / self.half_range
            x_norm[0] = np.clip(x0_norm, -1.0, 1.0)

        # Velocities and metric tracking
        velocities = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        diag_metric = np.ones(self.dim, dtype=np.float64)

        t = 0.0
        dt_noise = (self.n_cycles * np.pi) / self.n_steps

        # Initial state evaluation
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

            # 1. Objective gradient
            x_real = self.center + x_norm * self.half_range
            f_grad_real = np.asarray(fn_grad(x_real), dtype=np.float64)
            if f_grad_real.ndim == 1:
                f_grad_real = f_grad_real.reshape(1, -1)

            # 2. Chain rule
            f_grad_mapped = f_grad_real * self.half_range

            # 3. Update Riemannian metric tracking
            swarm_sq_grad = np.mean(f_grad_mapped ** 2, axis=0)
            diag_metric = self.metric_beta * diag_metric + (1.0 - self.metric_beta) * swarm_sq_grad

            # 4. Anisotropic Preconditioner
            if self.anisotropic_power > 0.0:
                scale_metric = np.sqrt(diag_metric) + 1e-8
                med_scale = np.median(scale_metric)
                rel_scale = scale_metric / med_scale if med_scale > 1e-8 else np.ones_like(scale_metric)
                precond = np.clip(1.0 / (rel_scale ** self.anisotropic_power), 0.1, 10.0)
            else:
                precond = np.ones(self.dim, dtype=np.float64)

            # 5. Directional gradient with preconditioning
            precond_grad = f_grad_mapped * precond
            norms = np.linalg.norm(precond_grad, axis=1, keepdims=True)
            f_grad_dir = np.where(norms > 1e-8, precond_grad / norms, 0.0)

            # 6. Perturbation gradient (ORF / Lissajous / RFF)
            if self.noise_field is not None:
                noise_grad = self.noise_field.grad(x_norm, t, amplitude=amp)
            else:
                noise_grad = 0.0

            # 7. Phase-modulated Swarm Gravity
            if self.gravity_strength > 0.0:
                gamma = self.gravity_strength * max(0.0, 1.0 - abs(sin_phase))
                diff = best_x_norm - x_norm
                diff_norms = np.linalg.norm(diff, axis=1, keepdims=True)
                safe_norms = np.maximum(diff_norms, 1e-7)
                grav_dir = np.where(diff_norms > 1e-7, diff / safe_norms, 0.0)
                grav_pull = gamma * grav_dir
            else:
                grav_pull = 0.0

            # 8. Total force
            f_total = -(f_grad_dir + noise_grad - grav_pull)

            # 9. Phase-gated symplectic momentum
            if self.momentum_base > 0.0:
                mu = self.momentum_base * (0.5 + 0.5 * abs(sin_phase))
                velocities = mu * velocities + (1.0 - mu) * f_total
                step_direction = velocities
            else:
                step_direction = f_total

            # 10. Cyclic step size
            cyclic_scale = self.dt_floor + (1.0 - self.dt_floor) * np.abs(np.sin(t * self.dt_cycles_multiplier))
            current_dt = self.dt_base * cyclic_scale

            # Update positions and project onto hypercube
            x_norm += current_dt * step_direction
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


def seismic_champion_v23(
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
    noise_engine: str = "orf",
    noise_frequencies: int = 64,
    gravity_strength: float = 0.4,
    momentum_base: float = 0.7,
    anisotropic_power: float = 0.5,
    seed: Optional[int] = 1,
) -> Tuple[np.ndarray, float, Dict[str, List[float]]]:
    """
    Functional interface for Seismic Champion v23.
    """
    opt = SeismicChampionV23(
        bounds=bounds,
        n_particles=n_particles,
        n_steps=n_steps,
        dt_base=dt_base,
        dt_floor=dt_floor,
        noise_amplitude=noise_amplitude,
        noise_decay=noise_decay,
        dt_cycles_multiplier=dt_cycles_multiplier,
        noise_engine=noise_engine,
        noise_frequencies=noise_frequencies,
        gravity_strength=gravity_strength,
        momentum_base=momentum_base,
        anisotropic_power=anisotropic_power,
        seed=seed,
    )
    return opt.optimize(fn=fn, fn_grad=fn_grad, x0=x0_real)
