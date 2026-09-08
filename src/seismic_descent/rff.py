"""
Random Fourier Features (RFF) representation for Gaussian Random Fields.

Approximates continuous, spatially correlated noise fields across arbitrary N-dimensional
continuous domains and computes exact analytical gradients in O(1) w.r.t evaluation points.
"""

from typing import Optional, Tuple, Union
import numpy as np


class RandomFourierFeatures:
    """
    Gaussian Random Field generator approximated via Random Fourier Features (RFF).

    Parameters
    ----------
    dim : int
        Dimension of the search space (D).
    r : int, default=64
        Number of random Fourier frequencies.
    n_octaves : int, default=1
        Number of frequency octaves.
    base_lengthscale : float, default=0.4
        Spatial lengthscale for the base octave in the normalized space [-1, 1]^D.
    seed : Optional[int], default=1
        Random seed for generating omegas, phis, and drifts.
    """

    def __init__(
        self,
        dim: int,
        r: int = 64,
        n_octaves: int = 1,
        base_lengthscale: float = 0.4,
        seed: Optional[int] = 1,
    ):
        self.dim = dim
        self.r = r
        self.n_octaves = n_octaves
        self.base_lengthscale = base_lengthscale
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.phis = rng.uniform(0, 2 * np.pi, size=(n_octaves, r))
        self.drifts = rng.uniform(0.1, 0.5, size=(n_octaves, r))
        self.z = [rng.normal(0, 1.0, size=(r, dim)) for _ in range(n_octaves)]

    def _ensure_2d(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Ensure input array is (N, D) shaped."""
        x = np.asarray(x, dtype=np.float64)
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x.reshape(1, -1)
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected dimension {self.dim}, got input with dimension {x.shape[1]}"
            )
        return x, is_1d

    def eval(
        self,
        x_norm: np.ndarray,
        t: float,
        amplitude: float = 1.0,
    ) -> Union[float, np.ndarray]:
        """
        Evaluate noise field value at normalized coordinates x_norm in [-1, 1]^D.

        Parameters
        ----------
        x_norm : array-like of shape (D,) or (N, D)
            Coordinates in normalized domain.
        t : float
            Simulation time / phase.
        amplitude : float, default=1.0
            Field amplitude multiplier.

        Returns
        -------
        noise_vals : float or ndarray of shape (N,)
            Evaluated noise values.
        """
        x_arr, is_1d = self._ensure_2d(x_norm)
        n_points = x_arr.shape[0]
        vals = np.zeros(n_points, dtype=np.float64)
        amp = amplitude
        sqrt_2_r = np.sqrt(2.0 / self.r)

        for o in range(self.n_octaves):
            lengthscale = self.base_lengthscale * (2.0 ** o)
            omegas = self.z[o] / lengthscale
            phis = self.phis[o][:, None]
            drifts = self.drifts[o][:, None]

            projections = omegas @ x_arr.T  # (r, N)
            angles = projections + t * drifts + phis
            vals += amp * sqrt_2_r * np.sum(np.cos(angles), axis=0)
            amp *= 0.5

        return vals[0] if is_1d else vals

    def grad(
        self,
        x_norm: np.ndarray,
        t: float,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """
        Compute analytic gradient of the noise field at x_norm in [-1, 1]^D.

        Returns
        -------
        grad : ndarray of shape (D,) or (N, D)
            Analytic gradient vector(s) of the noise field.
        """
        x_arr, is_1d = self._ensure_2d(x_norm)
        n_points, d = x_arr.shape
        grad = np.zeros((n_points, d), dtype=np.float64)
        amp = amplitude
        sqrt_2_r = np.sqrt(2.0 / self.r)

        for o in range(self.n_octaves):
            lengthscale = self.base_lengthscale * (2.0 ** o)
            omegas = self.z[o] / lengthscale
            phis = self.phis[o][:, None]
            drifts = self.drifts[o][:, None]

            projections = omegas @ x_arr.T  # (r, N)
            angles = projections + t * drifts + phis
            sines = np.sin(angles)  # (r, N)

            grad_contrib = (omegas.T @ sines).T  # (N, D)
            grad -= amp * sqrt_2_r * grad_contrib
            amp *= 0.5

        return grad[0] if is_1d else grad

    def eval_and_grad(
        self,
        x_norm: np.ndarray,
        t: float,
        amplitude: float = 1.0,
    ) -> Tuple[Union[float, np.ndarray], np.ndarray]:
        """
        Simultaneously evaluate noise and analytic gradient in a single pass.
        """
        x_arr, is_1d = self._ensure_2d(x_norm)
        n_points, d = x_arr.shape
        vals = np.zeros(n_points, dtype=np.float64)
        grad = np.zeros((n_points, d), dtype=np.float64)
        amp = amplitude
        sqrt_2_r = np.sqrt(2.0 / self.r)

        for o in range(self.n_octaves):
            lengthscale = self.base_lengthscale * (2.0 ** o)
            omegas = self.z[o] / lengthscale
            phis = self.phis[o][:, None]
            drifts = self.drifts[o][:, None]

            projections = omegas @ x_arr.T
            angles = projections + t * drifts + phis

            vals += amp * sqrt_2_r * np.sum(np.cos(angles), axis=0)
            sines = np.sin(angles)
            grad_contrib = (omegas.T @ sines).T
            grad -= amp * sqrt_2_r * grad_contrib
            amp *= 0.5

        return (vals[0] if is_1d else vals), (grad[0] if is_1d else grad)
