"""Edge-case and contract tests: inputs at the boundary of validity must not crash
or produce invalid outputs (NaN, out-of-bounds, silent empty runs)."""
import numpy as np
import pytest

from seismic_descent import seismic_swarm
from seismic_descent.champion_v23 import SeismicChampionV23, seismic_champion_v23
from seismic_descent.functions import SPHERE, RASTRIGIN

BOUNDS_2D = np.array([[-5.0, 5.0], [-5.0, 5.0]])


class TestCoreAPI:
    def test_single_particle(self):
        _, val, _ = seismic_swarm(fn=SPHERE["fn"], fn_grad=SPHERE["grad"],
                                  x0_real=np.array([3.0, 3.0]), bounds=BOUNDS_2D,
                                  n_steps=50, n_particles=1, seed=0)
        assert np.isfinite(val)

    def test_single_step(self):
        _, val, _ = seismic_swarm(fn=SPHERE["fn"], fn_grad=SPHERE["grad"],
                                  x0_real=np.array([3.0, 3.0]), bounds=BOUNDS_2D,
                                  n_steps=1, n_particles=5, seed=0)
        assert np.isfinite(val)

    def test_start_at_optimum(self):
        _, val, _ = seismic_swarm(fn=SPHERE["fn"], fn_grad=SPHERE["grad"],
                                  x0_real=np.array([0.0, 0.0]), bounds=BOUNDS_2D,
                                  n_steps=100, n_particles=5, seed=0)
        assert np.isfinite(val) and val >= 0.0

    def test_dimension_1(self):
        _, val, _ = seismic_swarm(fn=SPHERE["fn"], fn_grad=SPHERE["grad"],
                                  x0_real=np.array([2.0]), bounds=np.array([[-5.0, 5.0]]),
                                  n_steps=100, n_particles=10, seed=0)
        assert np.isfinite(val)

    def test_results_within_bounds(self):
        x, val, _ = seismic_swarm(fn=SPHERE["fn"], fn_grad=SPHERE["grad"],
                                  x0_real=np.array([3.0, 3.0]), bounds=BOUNDS_2D,
                                  n_steps=100, n_particles=10, seed=0)
        assert np.all(x >= BOUNDS_2D[:, 0] - 1e-12) and np.all(x <= BOUNDS_2D[:, 1] + 1e-12)
        assert np.isfinite(val)

    def test_zero_noise_amplitude_is_pure_gradient(self):
        """noise_amplitude=0 must disable the earthquake channel: strictly better
        than the same run with noise on the unimodal Sphere."""
        args = dict(fn=SPHERE["fn"], fn_grad=SPHERE["grad"], x0_real=np.array([3.0, 3.0]),
                    bounds=BOUNDS_2D, n_steps=200, n_particles=10, seed=3)
        _, v_no, _ = seismic_swarm(**args, noise_amplitude=0.0)
        _, v_on, _ = seismic_swarm(**args, noise_amplitude=0.5)
        assert v_no < v_on + 1e-12


class TestChampionV23:
    def test_single_particle(self):
        opt = SeismicChampionV23(bounds=BOUNDS_2D, n_particles=1, n_steps=50, seed=0)
        _, val, _ = opt.optimize(fn=SPHERE["fn"], fn_grad=SPHERE["grad"])
        assert np.isfinite(val)

    def test_results_finite_and_in_bounds_all_engines(self):
        for engine in ("rff", "orf", "lissajous"):
            opt = SeismicChampionV23(bounds=BOUNDS_2D, n_particles=5, n_steps=60,
                                     noise_engine=engine, seed=0)
            x, val, _ = opt.optimize(fn=SPHERE["fn"], fn_grad=SPHERE["grad"])
            assert np.isfinite(val), engine
            assert np.all(x >= BOUNDS_2D[:, 0] - 1e-12) and np.all(x <= BOUNDS_2D[:, 1] + 1e-12), engine

    def test_history_contract(self):
        opt = SeismicChampionV23(bounds=BOUNDS_2D, n_particles=5, n_steps=30, seed=0)
        _, _, hist = opt.optimize(fn=SPHERE["fn"], fn_grad=SPHERE["grad"])
        assert "best_per_step" in hist
        assert len(hist["best_per_step"]) == 31  # initial value + one per step
        assert all(np.isfinite(v) for v in hist["best_per_step"])

    def test_functional_interface(self):
        x, val, _ = seismic_champion_v23(fn=SPHERE["fn"], fn_grad=SPHERE["grad"],
                                         x0_real=np.array([3.0, 3.0]), bounds=BOUNDS_2D,
                                         n_steps=30, n_particles=5, seed=0)
        assert np.isfinite(val) and x.shape == (2,)

    def test_sanity_improves_over_x0(self):
        """With x0 far from optimum, 300 steps must improve the value by >10x on Sphere."""
        x0 = np.array([4.5, -4.5])
        v0 = float(SPHERE["fn"](x0))
        _, val, _ = seismic_champion_v23(fn=SPHERE["fn"], fn_grad=SPHERE["grad"],
                                         x0_real=x0, bounds=BOUNDS_2D,
                                         n_steps=300, n_particles=10, seed=42)
        assert val < v0 / 10
