"""
Lissajous Wave Field for deterministic ergodic landscape deformation.

Replaces stochastic Random Fourier Features (RFF) with incommensurate harmonic
Lissajous wave modes governed by square roots of primes (Kronecker-Weyl Theorem).
Computes exact analytical gradients in O(D) without matrix multiplications.
"""

from typing import Optional, Tuple, Union
import numpy as np


def _get_primes(n: int) -> np.ndarray:
    """Generate the first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return np.array(primes, dtype=np.float64)


class LissajousWaveField:
    """
    Deterministic continuous landscape deformation field using incommensurate Lissajous waves.

    Parameters
    ----------
    dim : int
        Dimension of the search space (D).
    n_harmonics : int, default=2
        Number of spatial frequency harmonics (octaves).
    base_lengthscale : float, default=0.4
        Spatial lengthscale for the fundamental mode in [-1, 1]^D.
    coupled : bool, default=True
        Whether to include adjacent coordinate cross-coupling (toroidal entanglement).
    time_speed : float, default=1.0
        Temporal frequency scaling multiplier.
    """

    def __init__(
        self,
        dim: int,
        n_harmonics: int = 2,
        base_lengthscale: float = 0.4,
        coupled: bool = True,
        time_speed: float = 1.0,
    ):
        self.dim = dim
        self.n_harmonics = n_harmonics
        self.base_lengthscale = base_lengthscale
        self.coupled = coupled
        self.time_speed = time_speed

        # Incommensurate temporal frequencies from square roots of distinct primes
        primes = _get_primes(dim * 2 + 10)
        self.temporal_freqs_direct = np.sqrt(primes[:dim]) * time_speed
        self.temporal_freqs_coupled = np.sqrt(primes[dim : 2 * dim]) * time_speed

        # Spatial wavenumbers: k = pi / lengthscale
        self.spatial_k = np.pi / base_lengthscale

    def _ensure_2d(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Ensure input array is (N, D) shaped."""
        x = np.asarray(x, dtype=np.float64)
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x.reshape(1, -1)
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {x.shape[1]}")
        return x, is_1d

    def eval(
        self,
        x_norm: np.ndarray,
        t: float,
        amplitude: float = 1.0,
    ) -> Union[float, np.ndarray]:
        """Evaluate Lissajous perturbation field at x_norm in [-1, 1]^D."""
        x_arr, is_1d = self._ensure_2d(x_norm)
        n_points, d = x_arr.shape
        vals = np.zeros(n_points, dtype=np.float64)
        amp = amplitude
        norm_factor = 1.0 / np.sqrt(d * (2 if self.coupled else 1))

        for h in range(self.n_harmonics):
            freq_mult = 2.0 ** h
            k = self.spatial_k * freq_mult

            # Direct axis-aligned Lissajous waves
            angles_direct = k * x_arr + (t * self.temporal_freqs_direct * freq_mult)
            vals += amp * norm_factor * np.sum(np.cos(angles_direct), axis=1)

            # Toroidal cross-coupled Lissajous waves
            if self.coupled and d > 1:
                x_next = np.roll(x_arr, -1, axis=1)
                angles_coupled = k * (x_arr + x_next) * 0.5 + (t * self.temporal_freqs_coupled * freq_mult)
                vals += amp * norm_factor * np.sum(np.cos(angles_coupled), axis=1)

            amp *= 0.5

        return float(vals[0]) if is_1d else vals

    def grad(
        self,
        x_norm: np.ndarray,
        t: float,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """
        Compute exact analytic gradient of Lissajous wave field in O(D).
        """
        x_arr, is_1d = self._ensure_2d(x_norm)
        n_points, d = x_arr.shape
        grad = np.zeros((n_points, d), dtype=np.float64)
        amp = amplitude
        norm_factor = 1.0 / np.sqrt(d * (2 if self.coupled else 1))

        for h in range(self.n_harmonics):
            freq_mult = 2.0 ** h
            k = self.spatial_k * freq_mult

            # 1. Direct gradient: d/dx_i cos(k x_i + w_i t) = -k sin(...)
            angles_direct = k * x_arr + (t * self.temporal_freqs_direct * freq_mult)
            grad -= amp * norm_factor * k * np.sin(angles_direct)

            # 2. Coupled gradient: x_i appears in term i and term (i - 1)
            if self.coupled and d > 1:
                x_next = np.roll(x_arr, -1, axis=1)
                angles_coupled = k * (x_arr + x_next) * 0.5 + (t * self.temporal_freqs_coupled * freq_mult)
                sines_coupled = np.sin(angles_coupled)
                sines_prev = np.roll(sines_coupled, 1, axis=1)
                grad -= amp * norm_factor * (k * 0.5) * (sines_coupled + sines_prev)

            amp *= 0.5

        return grad[0] if is_1d else grad

    def eval_and_grad(
        self,
        x_norm: np.ndarray,
        t: float,
        amplitude: float = 1.0,
    ) -> Tuple[Union[float, np.ndarray], np.ndarray]:
        """Simultaneously evaluate noise value and analytic gradient."""
        return self.eval(x_norm, t, amplitude), self.grad(x_norm, t, amplitude)
