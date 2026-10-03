"""Golden regression tests: freeze the historical trajectory of reproducible configs.

The constants below are generated from the code at HEAD (2026-10-03, post-audit) with
the exact configs shown. If a refactor or dependency upgrade changes the dynamics, these
tests fail — which is exactly what the repository's traceability rule demands: changes to
the algorithm's reproducible behavior must never be silent.

Note (2026-10-03): the ORF golden was regenerated when `np.linalg.qr` was replaced by the
deterministic modified Gram-Schmidt in `seismic_descent.orf` (CI fix). LAPACK's QR output
varies in the last ulp (±1e-15) across BLAS builds, which used to break this test on part
of the CI matrix; the construction is now BLAS-free and bit-identical across platforms
(guarded by `test_orf_construction_hash`).

The ORF golden also moved from Rastrigin to Sphere: ORF-on-Rastrigin is chaotic and
amplifies the remaining cross-platform ulp noise (libm/BLAS, Win vs Linux) into O(1)
final-value differences, whereas ORF-on-Sphere stays within ~1e-14 relative after
300 steps (measured Win vs Linux), comfortably under `_RTOL`.
"""
import numpy as np
import pytest

from seismic_descent.champion_v23 import SeismicChampionV23, seismic_champion_v23
from seismic_descent.core import seismic_swarm
from seismic_descent.functions import RASTRIGIN, ACKLEY, SPHERE

GOLDENS = {
    ("v23", "sphere5d", "orf"): 0.003993997465953818,
    ("v23", "ackley10d", "lissajous"): 19.983915274686087,
    ("v23", "sphere5d", "rff"): 0.019441366205003953,
    ("v20", "sphere2d", "rff"): 0.0005662477185043842,
}

_RTOL = 1e-9  # exact on every tested platform; tolerance guards against BLAS noise in the pipeline


def test_v23_sphere_5d_orf_golden():
    # Sphere instead of Rastrigin: the ORF-on-Rastrigin trajectory is chaotic
    # and amplifies cross-platform ulp noise (libm/BLAS, Win vs Linux) into O(1)
    # differences (measured: 1e-16 at step ~50 -> 8.2 at step 300), so it cannot
    # be pinned at rtol=1e-9. On Sphere the trajectory stays at ulp level
    # (measured Win vs Linux: rel diff ~1e-14 at n_steps=300).
    opt = SeismicChampionV23(
        bounds=np.array([[-10.0, 10.0]] * 5), n_particles=10, n_steps=300,
        noise_engine="orf", seed=42,
    )
    _, val, _ = opt.optimize(fn=SPHERE["fn"], fn_grad=SPHERE["grad"])
    assert val == pytest.approx(GOLDENS[("v23", "sphere5d", "orf")], rel=_RTOL)


def test_v23_ackley_10d_lissajous_golden():
    opt = SeismicChampionV23(
        bounds=np.array([[-32.768, 32.768]] * 10), n_particles=10, n_steps=300,
        noise_engine="lissajous", seed=7,
    )
    _, val, _ = opt.optimize(fn=ACKLEY["fn"], fn_grad=ACKLEY["grad"])
    assert val == pytest.approx(GOLDENS[("v23", "ackley10d", "lissajous")], rel=_RTOL)


def test_v23_sphere_5d_rff_golden():
    opt = SeismicChampionV23(
        bounds=np.array([[-10.0, 10.0]] * 5), n_particles=10, n_steps=300,
        noise_engine="rff", seed=123,
    )
    _, val, _ = opt.optimize(fn=SPHERE["fn"], fn_grad=SPHERE["grad"])
    assert val == pytest.approx(GOLDENS[("v23", "sphere5d", "rff")], rel=_RTOL)


def test_v20_sphere_2d_golden():
    bounds = np.array([[-5.0, 5.0]] * 2)
    _, val, _ = seismic_swarm(
        fn=SPHERE["fn"], fn_grad=SPHERE["grad"], x0_real=np.array([4.0, -4.0]),
        bounds=bounds, n_steps=500, n_particles=10, seed=42,
    )
    assert val == pytest.approx(GOLDENS[("v20", "sphere2d", "rff")], rel=_RTOL)


@pytest.mark.parametrize("engine", ["rff", "orf", "lissajous"])
def test_determinism_same_seed_identical_result(engine):
    """Two independent runs with the same seed must be bit-identical."""
    bounds = np.array([[-5.12, 5.12]] * 5)
    results = []
    for _ in range(2):
        opt = SeismicChampionV23(bounds=bounds, n_particles=10, n_steps=100,
                                 noise_engine=engine, seed=1)
        _, val, _ = opt.optimize(fn=RASTRIGIN["fn"], fn_grad=RASTRIGIN["grad"])
        results.append(val)
    assert results[0] == results[1]


class TestV24DimNormalizedAmplitude:
    """v24 candidate flag: default must not change v23 dynamics; enabled applies
    the measured sqrt-D law (informe_baselines_v24.md: v24 wins 21/24
    differentiating cells at the champion level)."""

    def test_default_off_preserves_v23_behavior(self):
        from seismic_descent.champion_v23 import seismic_champion_v23
        bounds = np.array([[-5.12, 5.12]] * 5)
        vals = []
        for flag in (None, False):
            kw = {} if flag is None else {"noise_amp_dim_normalized": False}
            _, v, _ = seismic_champion_v23(
                fn=RASTRIGIN["fn"], fn_grad=RASTRIGIN["grad"], x0_real=np.full(5, 3.0),
                bounds=bounds, n_steps=500, n_particles=10, noise_engine="orf",
                seed=42, **kw,
            )
            vals.append(v)
        assert vals[0] == vals[1]

    def test_amp_scale_exactly_sqrt_refdim_over_D(self):
        for dim in (5, 20, 45):
            opt = SeismicChampionV23(bounds=np.array([[-1.0, 1.0]] * dim),
                                     noise_amp_dim_normalized=True, n_steps=1)
            assert np.sqrt(5.0 / dim) == pytest.approx(np.sqrt(opt.ref_dim / dim))
            assert opt.ref_dim == 5.0  # v24 definition preserved by default

    def test_custom_ref_dim_parameterized(self):
        opt = SeismicChampionV23(bounds=np.array([[-1.0, 1.0]] * 20),
                                 noise_amp_dim_normalized=True, ref_dim=2.0, n_steps=1)
        assert np.sqrt(opt.ref_dim / 20) == pytest.approx(np.sqrt(0.1))

    def test_correction_improves_trid_20d_on_median(self):
        """Paired-seed check (config of the evidence, budget 3000): the median
        improves with the flag, and at least 4/5 paired seeds do (the effect
        is large but not universal — seed-level flips exist, ~20%)."""
        from seismic_descent.champion_v23 import seismic_champion_v23
        from seismic_descent.functions_extended import TRID
        bounds = np.array([[-400.0, 400.0]] * 20)
        diffs = []
        for seed in range(1, 6):
            _, v_abs, _ = seismic_champion_v23(
                fn=TRID["fn"], fn_grad=TRID["grad"], x0_real=None, bounds=bounds,
                n_steps=300, n_particles=10, noise_engine="orf", seed=seed,
            )
            _, v_cor, _ = seismic_champion_v23(
                fn=TRID["fn"], fn_grad=TRID["grad"], x0_real=None, bounds=bounds,
                n_steps=300, n_particles=10, noise_engine="orf", seed=seed,
                noise_amp_dim_normalized=True,
            )
            diffs.append(v_cor - v_abs)
        assert float(np.median(diffs)) < 0.0
        assert sum(d < 0 for d in diffs) >= 4
