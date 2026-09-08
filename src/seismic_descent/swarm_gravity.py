"""
swarm_gravity.py — Phase-Modulated Swarm Gravitational Coupling for Seismic Descent.

Paso 2: Acoplamiento gravitacional del enjambre.

Integrates dynamic gravitational/elastic attraction toward the global best particle
x_best into the Seismic Descent framework. The attraction strength gamma(t) is phase-modulated
with the seismic earthquake cycle:
- During seismic rupture (|A(t)| -> A_max): gamma(t) -> 0, allowing particles to scatter
  and explore freely without premature collapse.
- During seismic calm (|A(t)| -> 0): gamma(t) -> gamma_0, pulling the entire swarm into
  the most promising detected valley for rapid multi-particle exploitation.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from seismic_descent.rff import RandomFourierFeatures
from seismic_descent.lissajous import LissajousWaveField, OrthogonalLissajousWaveField


class SeismicSwarmGravity:
    """
    Seismic Swarm with Phase-Modulated Gravitational Attraction.

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
    gravity_strength : float, default=0.4
        Maximum attraction coefficient gamma_0 toward x_best.
    gravity_mode : str, default="phase_modulated"
        Modulation strategy for gravity:
        - "phase_modulated": gamma(t) = gamma_0 * (1 - |A(t)|/A_max)
        - "harmonic": gamma(t) = gamma_0 * cos^2(t)
        - "constant": gamma(t) = gamma_0
        - "none": gamma(t) = 0.0
    gravity_kernel : str, default="normalized"
        Attraction force formulation:
        - "normalized": unit direction vector toward x_best (constant step pull)
        - "linear": Hookean spring proportional to (x_best - x_i)
    noise_type : str, default="rff"
        Generator for landscape perturbation: "rff", "orthogonal_lissajous", "lissajous", or "none".
    noise_frequencies : int, default=64
        Features / frequency modes (RFF r, or orthogonal Lissajous rotations).
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
        gravity_strength: float = 0.4,
        gravity_mode: str = "phase_modulated",
        gravity_kernel: str = "normalized",
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
        self.gravity_strength = gravity_strength
        self.gravity_mode = gravity_mode.lower()
        self.gravity_kernel = gravity_kernel.lower()
        self.noise_type = noise_type.lower()
        self.base_lengthscale = base_lengthscale
        self.seed = seed

        self.center = (self.bounds[:, 1] + self.bounds[:, 0]) / 2.0
        self.half_range = (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0
        if np.any(self.half_range <= 0):
            raise ValueError("All upper bounds must be strictly greater than lower bounds.")

        # Build perturbation field
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

    def _compute_gravity_factor(self, t: float, amp_ratio: float) -> float:
        """Compute instantaneous attraction coefficient gamma(t)."""
        if self.gravity_strength <= 0.0 or self.gravity_mode == "none":
            return 0.0
        if self.gravity_mode == "phase_modulated":
            # Maximum attraction when amplitude is zero; zero attraction at peak quake
            return self.gravity_strength * max(0.0, 1.0 - abs(amp_ratio))
        elif self.gravity_mode == "harmonic":
            return self.gravity_strength * (np.cos(t) ** 2)
        elif self.gravity_mode == "constant":
            return self.gravity_strength
        else:
            raise ValueError(f"Unknown gravity_mode: {self.gravity_mode}")

    def optimize(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        fn_grad: Callable[[np.ndarray], np.ndarray],
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, List[float]]]:
        """
        Run seismic swarm optimization with gravitational cohesion.

        Parameters
        ----------
        fn : Callable[[np.ndarray], np.ndarray]
            Objective function accepting batch array of shape (N, D) or (D,) and returning (N,) or float.
        fn_grad : Callable[[np.ndarray], np.ndarray]
            Analytical gradient function accepting batch (N, D) and returning (N, D).
        x0 : Optional[np.ndarray], default=None
            Initial anchor point in real space (D,).

        Returns
        -------
        best_x : ndarray of shape (D,)
            Global best coordinates found in real space.
        best_val : float
            Objective value at best_x.
        info : dict
            Diagnostic metrics, including 'best_per_step'.
        """
        rng = np.random.default_rng(self.seed)
        x_norm = rng.uniform(-1.0, 1.0, size=(self.n_particles, self.dim))

        if x0 is not None:
            x0_arr = np.asarray(x0, dtype=np.float64)
            x0_norm = (x0_arr - self.center) / self.half_range
            x_norm[0] = np.clip(x0_norm, -1.0, 1.0)

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

            # 2. Chain rule mapping to normalized hypercube
            f_grad_mapped = f_grad_real * self.half_range

            # 3. L2 Gradient Normalization
            norms = np.linalg.norm(f_grad_mapped, axis=1, keepdims=True)
            f_grad_dir = np.where(norms > 1e-8, f_grad_mapped / norms, 0.0)

            # 4. Perturbation field gradient
            if self.noise_field is not None:
                noise_grad = self.noise_field.grad(x_norm, t, amplitude=amp)
            else:
                noise_grad = 0.0

            # 5. Gravitational attraction toward champion (best_x_norm)
            gamma = self._compute_gravity_factor(t, sin_phase)
            if gamma > 0.0:
                diff = best_x_norm - x_norm  # Shape: (n_particles, dim)
                if self.gravity_kernel == "normalized":
                    diff_norms = np.linalg.norm(diff, axis=1, keepdims=True)
                    safe_norms = np.maximum(diff_norms, 1e-7)
                    grav_dir = np.where(diff_norms > 1e-7, diff / safe_norms, 0.0)
                    grav_pull = gamma * grav_dir
                elif self.gravity_kernel == "linear":
                    # Hookean spring
                    grav_pull = gamma * diff
                else:
                    raise ValueError(f"Unknown gravity_kernel: {self.gravity_kernel}")
            else:
                grav_pull = 0.0

            # 6. Combined descent direction
            # Note: In x <- x - dt * grad, attraction towards x_best means displacement +dt*grav_pull,
            # so effective gradient contribution is -grav_pull.
            grad = f_grad_dir + noise_grad - grav_pull

            # 7. Decoupled cyclic learning rate schedule
            cyclic_scale = self.dt_floor + (1.0 - self.dt_floor) * np.abs(np.sin(t * self.dt_cycles_multiplier))
            current_dt = self.dt_base * cyclic_scale

            # Update positions and project onto bounds
            x_norm -= current_dt * grad
            np.clip(x_norm, -1.0, 1.0, out=x_norm)

            t += dt_noise

            # Track global best point strictly against unperturbed objective
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


def seismic_swarm_gravity(
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
    n_cycles: int = 10,
    dt_cycles_multiplier: float = 5.0,
    gravity_strength: float = 0.4,
    gravity_mode: str = "phase_modulated",
    gravity_kernel: str = "normalized",
    noise_type: str = "rff",
    seed: Optional[int] = 1,
) -> Tuple[np.ndarray, float, Dict[str, List[float]]]:
    """
    Functional interface to Seismic Swarm with Gravitational Coupling.
    """
    optimizer = SeismicSwarmGravity(
        bounds=bounds,
        n_particles=n_particles,
        n_steps=n_steps,
        n_cycles=n_cycles,
        dt_base=dt_base,
        dt_floor=dt_floor,
        noise_amplitude=noise_amplitude,
        noise_decay=noise_decay,
        dt_cycles_multiplier=dt_cycles_multiplier,
        gravity_strength=gravity_strength,
        gravity_mode=gravity_mode,
        gravity_kernel=gravity_kernel,
        noise_type=noise_type,
        seed=seed,
    )
    return optimizer.optimize(fn=fn, fn_grad=fn_grad, x0=x0_real)
