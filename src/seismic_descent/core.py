"""
Core implementation of Seismic Descent (v20 Champion Architecture).

Key architectural features:
1. Isotropic domain normalization to [-1, 1]^D.
2. L2 gradient normalization to decouple step scale from target objective scale.
3. Decoupled cyclic learning rate schedule dt(t) for multiscale exploration/exploitation.
4. Spatially correlated, analytic Random Fourier Features (RFF) ground deformation.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from seismic_descent.rff import RandomFourierFeatures


class SeismicSwarm:
    """
    Seismic Swarm Optimizer (v20 Architecture).

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
    noise_amplitude : float, default=0.5
        Noise gradient amplitude relative to unit directional gradient.
    noise_decay : float, default=1.0
        Global noise amplitude decay factor per step.
    dt_cycles_multiplier : float, default=5.0
        Frequency multiplier for cyclic learning rate relative to seismic phase.
    rff_frequencies : int, default=64
        Number of random Fourier features.
    rff_octaves : int, default=1
        Number of spatial noise octaves.
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
        noise_amplitude: float = 0.5,
        noise_decay: float = 1.0,
        dt_cycles_multiplier: float = 5.0,
        rff_frequencies: int = 64,
        rff_octaves: int = 1,
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
        self.noise_amplitude = noise_amplitude
        self.noise_decay = noise_decay
        self.dt_cycles_multiplier = dt_cycles_multiplier
        self.seed = seed

        self.center = (self.bounds[:, 1] + self.bounds[:, 0]) / 2.0
        self.half_range = (self.bounds[:, 1] - self.bounds[:, 0]) / 2.0
        if np.any(self.half_range <= 0):
            raise ValueError("All upper bounds must be strictly greater than lower bounds.")

        self.rff = RandomFourierFeatures(
            dim=self.dim,
            r=rff_frequencies,
            n_octaves=rff_octaves,
            base_lengthscale=base_lengthscale,
            seed=seed,
        )

    def optimize(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        fn_grad: Callable[[np.ndarray], np.ndarray],
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, List[float]]]:
        """
        Run seismic swarm optimization.

        Parameters
        ----------
        fn : Callable[[np.ndarray], np.ndarray]
            Objective function accepting batch array of shape (N, D) or (D,) and returning (N,) or float.
        fn_grad : Callable[[np.ndarray], np.ndarray]
            Analytical gradient function accepting batch (N, D) and returning (N, D).
        x0 : Optional[np.ndarray], default=None
            Initial anchor point in real space (D,). If provided, particle 0 is initialized here.

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

        best_idx = np.argmin(real_vals)
        best_val = float(real_vals[best_idx])
        best_x_real = x_real[best_idx].copy()

        best_per_step: List[float] = [best_val]

        for step in range(self.n_steps):
            decay = self.noise_decay ** step
            freq = 2.0 * decay
            amp = self.noise_amplitude * decay * np.sin(t * freq)

            # 1. Real objective gradient
            x_real = self.center + x_norm * self.half_range
            f_grad_real = np.asarray(fn_grad(x_real), dtype=np.float64)
            if f_grad_real.ndim == 1:
                f_grad_real = f_grad_real.reshape(1, -1)

            # 2. Chain rule mapping to normalized hypercube
            f_grad_mapped = f_grad_real * self.half_range

            # 3. L2 Gradient Normalization (directional focus)
            norms = np.linalg.norm(f_grad_mapped, axis=1, keepdims=True)
            f_grad_dir = np.where(norms > 1e-8, f_grad_mapped / norms, 0.0)

            # 4. RFF spatially correlated noise gradient
            noise_grad = self.rff.grad(x_norm, t, amplitude=amp)

            # 5. Combined landscape gradient
            grad = f_grad_dir + noise_grad

            # 6. Decoupled cyclic learning rate schedule
            current_dt = self.dt_base * np.abs(np.sin(t * self.dt_cycles_multiplier))

            # Update positions and project onto bounds
            x_norm -= current_dt * grad
            np.clip(x_norm, -1.0, 1.0, out=x_norm)

            t += dt_noise

            # Track global best point strictly against unperturbed objective
            x_real = self.center + x_norm * self.half_range
            step_vals = np.asarray(fn(x_real), dtype=np.float64)
            if step_vals.ndim == 0:
                step_vals = np.array([step_vals])

            step_best_idx = np.argmin(step_vals)
            step_best_val = float(step_vals[step_best_idx])
            if step_best_val < best_val:
                best_val = step_best_val
                best_x_real = x_real[step_best_idx].copy()

            best_per_step.append(best_val)

        return best_x_real, best_val, {"best_per_step": best_per_step}


def seismic_swarm(
    fn: Callable[[np.ndarray], np.ndarray],
    fn_grad: Callable[[np.ndarray], np.ndarray],
    x0_real: np.ndarray,
    bounds: np.ndarray,
    n_steps: int = 2000,
    n_particles: int = 10,
    dt_base: float = 0.2,
    noise_amplitude: float = 0.5,
    noise_decay: float = 1.0,
    n_cycles: int = 10,
    dt_cycles_multiplier: float = 5.0,
    seed: Optional[int] = 1,
) -> Tuple[np.ndarray, float, Dict[str, List[float]]]:
    """
    Functional interface to Seismic Swarm optimization (compatible with v20 signature).
    """
    optimizer = SeismicSwarm(
        bounds=bounds,
        n_particles=n_particles,
        n_steps=n_steps,
        n_cycles=n_cycles,
        dt_base=dt_base,
        noise_amplitude=noise_amplitude,
        noise_decay=noise_decay,
        dt_cycles_multiplier=dt_cycles_multiplier,
        seed=seed,
    )
    return optimizer.optimize(fn=fn, fn_grad=fn_grad, x0=x0_real)
