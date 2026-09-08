"""Unit tests for Seismic Anisotropic HMC module."""

import numpy as np
import pytest
from seismic_descent.anisotropic_hmc import SeismicAnisotropicHMC, seismic_anisotropic_hmc


def test_anisotropic_hmc_initialization():
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    opt = SeismicAnisotropicHMC(
        bounds=bounds,
        n_particles=10,
        momentum_base=0.8,
        anisotropic_power=0.5,
        gravity_strength=0.3,
    )
    assert opt.dim == 2
    assert opt.momentum_base == 0.8
    assert opt.anisotropic_power == 0.5


def test_anisotropic_hmc_ill_conditioned_quadratic():
    # Anisotropic ellipsoid with condition number 100
    # f(x) = x_0^2 + 100 * x_1^2
    def ellipsoid(x):
        x = np.atleast_2d(x)
        return x[:, 0]**2 + 100.0 * x[:, 1]**2

    def ellipsoid_grad(x):
        x = np.atleast_2d(x)
        grad = np.zeros_like(x)
        grad[:, 0] = 2.0 * x[:, 0]
        grad[:, 1] = 200.0 * x[:, 1]
        return grad

    bounds = np.array([[-10.0, 10.0], [-10.0, 10.0]])
    x0 = np.array([8.0, 8.0])

    best_x, best_val, info = seismic_anisotropic_hmc(
        fn=ellipsoid,
        fn_grad=ellipsoid_grad,
        x0_real=x0,
        bounds=bounds,
        n_steps=200,
        n_particles=10,
        momentum_base=0.7,
        anisotropic_power=0.5,
        gravity_strength=0.4,
        noise_type="rff",
        seed=42,
    )

    assert best_val < 0.05
    assert best_x.shape == (2,)


def test_anisotropic_hmc_with_orthogonal_lissajous():
    def rosenbrock(x):
        x = np.atleast_2d(x)
        return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1.0 - x[:, :-1])**2, axis=1)

    def rosenbrock_grad(x):
        x = np.atleast_2d(x)
        grad = np.zeros_like(x)
        grad[:, :-1] += -400.0 * x[:, :-1] * (x[:, 1:] - x[:, :-1]**2) - 2.0 * (1.0 - x[:, :-1])
        grad[:, 1:] += 200.0 * (x[:, 1:] - x[:, :-1]**2)
        return grad

    bounds = np.array([[-2.0, 2.0]] * 4)
    best_x, best_val, _ = seismic_anisotropic_hmc(
        fn=rosenbrock,
        fn_grad=rosenbrock_grad,
        x0_real=np.zeros(4),
        bounds=bounds,
        n_steps=150,
        n_particles=8,
        momentum_base=0.7,
        anisotropic_power=0.4,
        gravity_strength=0.3,
        noise_type="orthogonal_lissajous",
        seed=1,
    )
    assert np.isfinite(best_val)
    assert best_val < 15.0
