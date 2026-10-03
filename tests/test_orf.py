"""Unit tests for Orthogonal Random Features (ORF) module."""

import hashlib
import sys

import numpy as np
import pytest
from seismic_descent.orf import OrthogonalRandomFeatures, _modified_gram_schmidt


def test_orf_shapes():
    dim = 7
    orf = OrthogonalRandomFeatures(dim=dim, r=64, seed=42)
    x = np.random.uniform(-1, 1, size=(10, dim))
    val = orf.eval(x, t=1.5)
    assert val.shape == (10,)

    grad = orf.grad(x, t=1.5)
    assert grad.shape == (10, dim)


def test_orf_analytic_grad_vs_finite_difference():
    dim = 5
    orf = OrthogonalRandomFeatures(dim=dim, r=32, base_lengthscale=0.5, seed=123)
    x = np.array([[0.2, -0.4, 0.1, 0.35, -0.25]])
    t = 2.0
    amp = 0.8

    analytic_grad = orf.grad(x, t=t, amplitude=amp)[0]

    eps = 1e-6
    numerical_grad = np.zeros(dim)
    for i in range(dim):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[0, i] += eps
        x_minus[0, i] -= eps
        f_plus = float(orf.eval(x_plus, t=t, amplitude=amp)[0])
        f_minus = float(orf.eval(x_minus, t=t, amplitude=amp)[0])
        numerical_grad[i] = (f_plus - f_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytic_grad, numerical_grad, rtol=1e-4, atol=1e-4)


def test_orf_eval_and_grad_consistency():
    dim = 4
    orf = OrthogonalRandomFeatures(dim=dim, r=32, seed=7)
    x = np.random.uniform(-1, 1, size=(6, dim))
    t = 0.9

    v_sep = orf.eval(x, t=t)
    g_sep = orf.grad(x, t=t)
    v_comb, g_comb = orf.eval_and_grad(x, t=t)

    np.testing.assert_allclose(v_sep, v_comb, atol=1e-12)
    np.testing.assert_allclose(g_sep, g_comb, atol=1e-12)


def test_mgs_matches_sign_fixed_qr():
    """The deterministic MGS must equal the historical sign-fixed QR up to rounding."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=(8, 8))

    q = _modified_gram_schmidt(g)
    np.testing.assert_allclose(q.T @ q, np.eye(8), atol=1e-12)

    q_ref, r_ref = np.linalg.qr(g)
    ph = np.sign(np.diag(r_ref))
    ph[ph == 0] = 1.0
    np.testing.assert_allclose(q, q_ref * ph[None, :], atol=1e-12)


def test_orf_construction_hash():
    """The ORF basis must be bit-identical across platforms/numpy/BLAS builds.

    The construction is BLAS-free (deterministic modified Gram-Schmidt). Any
    reintroduction of np.linalg.qr — whose ±1e-15 ulp noise across OpenBLAS
    builds the optimizer amplifies into O(1) trajectory differences — changes
    this digest and breaks the ORF golden test on part of the CI matrix.
    """
    if sys.byteorder != "little":
        pytest.skip("digest pinned for little-endian platforms (CI: x86-64)")
    orf = OrthogonalRandomFeatures(dim=5, r=64, seed=42)
    digest = hashlib.sha256(np.ascontiguousarray(orf.z[0]).tobytes()).hexdigest()
    assert digest == "a60904b8548b4f5f4be8d23c1fdf1a42255ad9ce4580a57811d4638b5e392e34"
