"""Unit tests for Seismic Swarm core optimizer."""

import numpy as np
import pytest
from seismic_descent.core import SeismicSwarm, seismic_swarm
from seismic_descent.functions import SPHERE, RASTRIGIN


def test_sphere_convergence():
    bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])
    x0 = np.array([4.0, -4.0])

    best_x, best_val, info = seismic_swarm(
        fn=SPHERE["fn"],
        fn_grad=SPHERE["grad"],
        x0_real=x0,
        bounds=bounds,
        n_steps=500,
        n_particles=10,
        seed=42,
    )

    assert best_val < 0.05
    assert len(best_x) == 2
    # Check that tracking is monotonic
    tracking = info["best_per_step"]
    assert all(tracking[i] >= tracking[i + 1] for i in range(len(tracking) - 1))


def test_bounds_enforcement():
    bounds = np.array([[-2.0, 3.0], [1.0, 5.0], [-10.0, -5.0]])

    def dummy_fn(x):
        if x.ndim == 1:
            return np.sum(x)
        return np.sum(x, axis=1)

    def dummy_grad(x):
        if x.ndim == 1:
            return np.ones_like(x)
        return np.ones_like(x)

    optimizer = SeismicSwarm(bounds=bounds, n_particles=8, n_steps=200, seed=1)
    best_x, _, _ = optimizer.optimize(dummy_fn, dummy_grad)

    for i in range(3):
        assert bounds[i, 0] <= best_x[i] <= bounds[i, 1]


def test_rastrigin_multimodal_escape():
    bounds = np.array([[-5.12, 5.12]] * 5)
    x0 = np.full(5, 3.0)

    best_x, best_val, _ = seismic_swarm(
        fn=RASTRIGIN["fn"],
        fn_grad=RASTRIGIN["grad"],
        x0_real=x0,
        bounds=bounds,
        n_steps=1000,
        n_particles=10,
        seed=42,
    )

    # In 5D Rastrigin, an initial point at x=(3,3,3,3,3) starts with f(x0) ~ 115.
    # A standard gradient descent gets stuck near f(x) ~ 40-70.
    # Seismic swarm escapes local minima into much deeper basins.
    assert best_val < 20.0


def test_dt_floor_parameter():
    bounds = np.array([[-5.12, 5.12]] * 2)
    # Validate initialization with custom dt_floor
    opt = SeismicSwarm(bounds=bounds, dt_floor=0.35)
    assert opt.dt_floor == 0.35

    _, val, _ = opt.optimize(SPHERE["fn"], SPHERE["grad"])
    assert val < 0.1

