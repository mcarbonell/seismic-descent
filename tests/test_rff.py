"""Unit tests for Random Fourier Features (RFF) module."""

import numpy as np
import pytest
from seismic_descent.rff import RandomFourierFeatures


def test_rff_shapes_and_reproducibility():
    dim = 5
    rff1 = RandomFourierFeatures(dim=dim, r=32, n_octaves=2, seed=42)
    rff2 = RandomFourierFeatures(dim=dim, r=32, n_octaves=2, seed=42)

    x = np.random.uniform(-1, 1, size=(10, dim))
    val1 = rff1.eval(x, t=0.5)
    val2 = rff2.eval(x, t=0.5)
    assert val1.shape == (10,)
    np.testing.assert_allclose(val1, val2, atol=1e-12)

    grad1 = rff1.grad(x, t=0.5)
    grad2 = rff2.grad(x, t=0.5)
    assert grad1.shape == (10, dim)
    np.testing.assert_allclose(grad1, grad2, atol=1e-12)


def test_rff_analytic_gradient_vs_finite_difference():
    dim = 4
    rff = RandomFourierFeatures(dim=dim, r=64, n_octaves=2, seed=123)
    x = np.array([[0.1, -0.2, 0.5, -0.4]])
    t = 1.2
    amp = 0.8

    # Analytic gradient
    analytic_grad = rff.grad(x, t=t, amplitude=amp)[0]

    # Finite difference gradient
    eps = 1e-6
    numerical_grad = np.zeros(dim)
    for i in range(dim):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[0, i] += eps
        x_minus[0, i] -= eps
        f_plus = float(rff.eval(x_plus, t=t, amplitude=amp)[0])
        f_minus = float(rff.eval(x_minus, t=t, amplitude=amp)[0])
        numerical_grad[i] = (f_plus - f_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytic_grad, numerical_grad, rtol=1e-4, atol=1e-4)


def test_rff_eval_and_grad_consistency():
    dim = 3
    rff = RandomFourierFeatures(dim=dim, r=32, seed=99)
    x = np.random.uniform(-1, 1, size=(5, dim))
    t = 0.7

    val_separate = rff.eval(x, t=t)
    grad_separate = rff.grad(x, t=t)
    val_combined, grad_combined = rff.eval_and_grad(x, t=t)

    np.testing.assert_allclose(val_separate, val_combined, atol=1e-12)
    np.testing.assert_allclose(grad_separate, grad_combined, atol=1e-12)
