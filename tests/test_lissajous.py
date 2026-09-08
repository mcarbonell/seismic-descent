"""Unit tests for Lissajous Wave Field module."""

import numpy as np
import pytest
from seismic_descent.lissajous import LissajousWaveField


def test_lissajous_shapes():
    dim = 5
    field = LissajousWaveField(dim=dim, n_harmonics=2, coupled=True)
    x = np.random.uniform(-1, 1, size=(8, dim))
    val = field.eval(x, t=1.5)
    assert val.shape == (8,)

    grad = field.grad(x, t=1.5)
    assert grad.shape == (8, dim)


def test_lissajous_analytic_grad_vs_finite_difference():
    dim = 4
    for coupled in [False, True]:
        field = LissajousWaveField(dim=dim, n_harmonics=2, coupled=coupled, base_lengthscale=0.5)
        x = np.array([[0.25, -0.3, 0.45, -0.15]])
        t = 2.1
        amp = 0.7

        analytic_grad = field.grad(x, t=t, amplitude=amp)[0]

        eps = 1e-6
        numerical_grad = np.zeros(dim)
        for i in range(dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[0, i] += eps
            x_minus[0, i] -= eps
            f_plus = float(field.eval(x_plus, t=t, amplitude=amp)[0])
            f_minus = float(field.eval(x_minus, t=t, amplitude=amp)[0])
            numerical_grad[i] = (f_plus - f_minus) / (2.0 * eps)

        np.testing.assert_allclose(analytic_grad, numerical_grad, rtol=1e-4, atol=1e-4)


def test_lissajous_eval_and_grad_consistency():
    dim = 3
    field = LissajousWaveField(dim=dim, coupled=True)
    x = np.random.uniform(-1, 1, size=(6, dim))
    t = 0.85

    v_sep = field.eval(x, t=t)
    g_sep = field.grad(x, t=t)
    v_comb, g_comb = field.eval_and_grad(x, t=t)

    np.testing.assert_allclose(v_sep, v_comb, atol=1e-12)
    np.testing.assert_allclose(g_sep, g_comb, atol=1e-12)
