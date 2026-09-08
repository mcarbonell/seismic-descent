"""Unit tests for Phase-Modulated Swarm Gravity module."""

import numpy as np
import pytest
from seismic_descent.swarm_gravity import SeismicSwarmGravity, seismic_swarm_gravity


def test_swarm_gravity_initialization():
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    opt = SeismicSwarmGravity(
        bounds=bounds,
        n_particles=12,
        gravity_strength=0.3,
        gravity_mode="phase_modulated",
        gravity_kernel="normalized",
        noise_type="rff",
    )
    assert opt.dim == 2
    assert opt.n_particles == 12
    assert opt.gravity_strength == 0.3


def test_swarm_gravity_modulation_factor():
    bounds = np.array([[-1.0, 1.0]])
    opt = SeismicSwarmGravity(bounds=bounds, gravity_strength=0.5, gravity_mode="phase_modulated")
    # Peak amplitude (amp_ratio = 1.0) -> gravity should drop to 0
    assert opt._compute_gravity_factor(t=np.pi/4, amp_ratio=1.0) == 0.0
    # Zero amplitude (amp_ratio = 0.0) -> gravity should be maximum (0.5)
    assert opt._compute_gravity_factor(t=0.0, amp_ratio=0.0) == 0.5
    # Partial amplitude (amp_ratio = 0.4) -> gravity should be 0.5 * (1 - 0.4) = 0.3
    np.testing.assert_allclose(opt._compute_gravity_factor(t=0.0, amp_ratio=0.4), 0.3)


def test_swarm_gravity_sphere_optimization():
    # Simple Sphere function: f(x) = sum(x^2), min at 0
    def sphere(x):
        return np.sum(x ** 2, axis=-1)

    def sphere_grad(x):
        return 2.0 * x

    bounds = np.array([[-5.0, 5.0]] * 4)
    x0 = np.array([3.0, -3.0, 2.5, -2.5])

    best_x, best_val, info = seismic_swarm_gravity(
        fn=sphere,
        fn_grad=sphere_grad,
        x0_real=x0,
        bounds=bounds,
        n_steps=200,
        n_particles=10,
        gravity_strength=0.4,
        gravity_mode="phase_modulated",
        noise_type="rff",
        seed=42,
    )

    assert best_val < 0.1
    assert best_x.shape == (4,)
    assert len(info["best_per_step"]) == 201


def test_swarm_gravity_orthogonal_lissajous_support():
    def rosenbrock(x):
        x = np.atleast_2d(x)
        return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1.0 - x[:, :-1])**2, axis=1)

    def rosenbrock_grad(x):
        x = np.atleast_2d(x)
        grad = np.zeros_like(x)
        grad[:, :-1] += -400.0 * x[:, :-1] * (x[:, 1:] - x[:, :-1]**2) - 2.0 * (1.0 - x[:, :-1])
        grad[:, 1:] += 200.0 * (x[:, 1:] - x[:, :-1]**2)
        return grad

    bounds = np.array([[-2.0, 2.0]] * 3)
    best_x, best_val, _ = seismic_swarm_gravity(
        fn=rosenbrock,
        fn_grad=rosenbrock_grad,
        x0_real=np.zeros(3),
        bounds=bounds,
        n_steps=100,
        n_particles=8,
        gravity_strength=0.3,
        noise_type="orthogonal_lissajous",
        seed=1,
    )
    assert np.isfinite(best_val)
