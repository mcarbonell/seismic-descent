"""
orf.py — Orthogonal Random Features (ORF) representation for Gaussian Random Fields.

Implements structured orthogonal random features (Yu et al., NeurIPS 2016).
Replaces standard independent Gaussian projection vectors with blocks of mutually
orthogonal random vectors via Gram-Schmidt orthogonalization (QR-equivalent,
deterministic across BLAS builds) with Chi-distributed radial lengths.

This drastically reduces kernel approximation variance and eliminates redundant
directional sampling in high dimensions (D >= 10, 20, 50).
"""

from typing import Optional, Tuple, Union
import numpy as np


def _modified_gram_schmidt(g: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the columns of ``g`` (modified Gram-Schmidt).

    Mathematically equivalent to ``np.linalg.qr(g)`` with the sign convention
    ``diag(R) > 0`` (Haar sampling convention), but built exclusively from
    elementwise arithmetic and ``np.sum`` (pairwise summation) — no BLAS or
    LAPACK calls. ``np.linalg.qr`` returns values that differ in the last ulp
    (±1e-15) across OpenBLAS/LAPACK builds, and the optimizer amplifies that
    noise into O(1) trajectory differences, which made ORF golden regression
    tests pass only on the reference build (see tests/test_golden_regression.py).
    """
    d = g.shape[0]
    q = np.empty_like(g)
    for j in range(d):
        v = g[:, j].copy()
        for i in range(j):
            v -= np.sum(q[:, i] * v) * q[:, i]
        nrm = float(np.sqrt(np.sum(v * v)))
        if nrm == 0.0:
            # Degenerate draw (probability ~2^(-52*d) for Gaussian input):
            # keep the historical LAPACK behaviour instead of failing.
            return np.linalg.qr(g)[0]
        q[:, j] = v / nrm
    return q


class OrthogonalRandomFeatures:
    """
    Orthogonal Random Features (ORF) Generator for Seismic Landscapes.

    Parameters
    ----------
    dim : int
        Dimension of the search space (D).
    r : int, default=64
        Number of random Fourier frequencies (rounded up to nearest multiple of D internally).
    n_octaves : int, default=1
        Number of frequency octaves.
    base_lengthscale : float, default=0.4
        Spatial lengthscale for the base octave in the normalized space [-1, 1]^D.
    seed : Optional[int], default=1
        Random seed for reproducibility.
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
        self.n_octaves = n_octaves
        self.base_lengthscale = base_lengthscale
        self.seed = seed

        # Number of orthogonal DxD blocks needed
        n_blocks = max(1, int(np.ceil(r / dim)))
        self.r = n_blocks * dim

        rng = np.random.default_rng(seed)
        self.phis = rng.uniform(0, 2 * np.pi, size=(n_octaves, self.r))
        self.drifts = rng.uniform(0.1, 0.5, size=(n_octaves, self.r))

        self.z = []
        for _ in range(n_octaves):
            blocks = []
            for _ in range(n_blocks):
                # 1. Sample Gaussian matrix
                g = rng.normal(0, 1.0, size=(dim, dim))
                # 2. Orthonormalize to sample uniformly from Haar measure O(D).
                #    Deterministic MGS instead of np.linalg.qr: bit-identical
                #    across platforms/numpy/BLAS builds (see _modified_gram_schmidt).
                #    MGS yields diag(R) > 0 by construction, i.e. the same sign
                #    convention as the historical sign-fixed QR.
                q = _modified_gram_schmidt(g)

                # 3. Chi-distributed radial scaling (norm of D-dimensional Gaussian)
                s = np.sqrt(rng.chisquare(df=dim, size=dim))
                block_w = s[:, None] * q
                blocks.append(block_w)

            octave_w = np.vstack(blocks)  # Shape (r, dim)
            self.z.append(octave_w)

    def _ensure_2d(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
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
        Evaluate ORF noise field value at normalized coordinates x_norm in [-1, 1]^D.
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

        return float(vals[0]) if is_1d else vals

    def grad(
        self,
        x_norm: np.ndarray,
        t: float,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """
        Compute exact analytic gradient of ORF field w.r.t x_norm.
        """
        x_arr, is_1d = self._ensure_2d(x_norm)
        n_points = x_arr.shape[0]
        grads = np.zeros((n_points, self.dim), dtype=np.float64)
        amp = amplitude
        sqrt_2_r = np.sqrt(2.0 / self.r)

        for o in range(self.n_octaves):
            lengthscale = self.base_lengthscale * (2.0 ** o)
            omegas = self.z[o] / lengthscale
            phis = self.phis[o][:, None]
            drifts = self.drifts[o][:, None]

            projections = omegas @ x_arr.T  # (r, N)
            angles = projections + t * drifts + phis
            sin_vals = np.sin(angles)  # (r, N)

            grads -= amp * sqrt_2_r * (sin_vals.T @ omegas)
            amp *= 0.5

        return grads[0] if is_1d else grads

    def eval_and_grad(
        self,
        x_norm: np.ndarray,
        t: float,
        amplitude: float = 1.0,
    ) -> Tuple[Union[float, np.ndarray], np.ndarray]:
        return self.eval(x_norm, t, amplitude), self.grad(x_norm, t, amplitude)
