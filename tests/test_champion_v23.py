"""Unit tests for SeismicChampionV23 module."""

import numpy as np
import pytest
from seismic_descent.champion_v23 import SeismicChampionV23, seismic_champion_v23


def test_champion_v23_initialization():
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
    opt = SeismicChampionV23(
        bounds=bounds,
        n_particles=10,
        noise_engine="orf",
        gravity_strength=0.4,
        momentum_base=0.7,
        anisotropic_power=0.5,
    )
    assert opt.dim == 3
    assert opt.noise_engine == "orf"
    assert opt.gravity_strength == 0.4


def test_champion_v23_all_engines():
    def sphere(x):
        return np.sum(x ** 2, axis=-1)

    def sphere_grad(x):
        return 2.0 * x

    bounds = np.array([[-5.0, 5.0]] * 3)
    x0 = np.array([3.0, 3.0, 3.0])

    for engine in ["orf", "orthogonal_lissajous", "rff"]:
        best_x, best_val, info = seismic_champion_v23(
            fn=sphere,
            fn_grad=sphere_grad,
            x0_real=x0,
            bounds=bounds,
            n_steps=100,
            n_particles=8,
            noise_engine=engine,
            seed=42,
        )
        assert best_val < 0.1
        assert best_x.shape == (3,)
