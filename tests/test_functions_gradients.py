"""Numeric verification of every objective gradient shipped in the package.

The optimizer consumes these gradients on every step; an error here silently
contaminates every benchmark. Prior to 0.23.1 no test covered them (only the
noise-field gradients were tested).
"""

import numpy as np
import pytest

from seismic_descent.functions import ALL_FUNCTIONS
from seismic_descent.functions_extended import EXTENDED_FUNCTIONS, get_search_range

ALL_SUITE = {**ALL_FUNCTIONS, **EXTENDED_FUNCTIONS}


def _fd_grad(fn, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (fn(xp) - fn(xm)) / (2.0 * eps)
    return g


@pytest.mark.parametrize("name", sorted(ALL_SUITE))
def test_objective_gradient_matches_finite_differences(name):
    fcfg = ALL_SUITE[name]
    fn, grad_fn = fcfg["fn"], fcfg["grad"]
    rng = np.random.default_rng(1234)
    for dim in (1, 2, 5, 9):
        r = get_search_range(fcfg, dim)
        # Avoid the extreme edges (some functions have steep walls / kinks there,
        # and Trid's canonical range grows quadratically with D).
        x = rng.uniform(-0.8 * r, 0.8 * r, size=dim)
        fd = _fd_grad(fn, x)
        an = np.asarray(grad_fn(x), dtype=np.float64)
        denom = max(1.0, float(np.linalg.norm(fd)))
        rel_err = float(np.linalg.norm(an - fd)) / denom
        assert rel_err < 1e-5, f"{name} D={dim}: relative gradient error {rel_err:.2e}"


def test_gradients_batch_consistency():
    """Batch (N, D) and single-row (D,) calls must agree."""
    rng = np.random.default_rng(7)
    for name, fcfg in ALL_SUITE.items():
        x = rng.uniform(-1.0, 1.0, size=(4, 3)) * 0.5
        batch = np.asarray(fcfg["grad"](x), dtype=np.float64)
        singles = np.stack([np.asarray(fcfg["grad"](row)) for row in x])
        np.testing.assert_allclose(batch, singles, rtol=1e-10, atol=1e-12,
                                   err_msg=f"batch/single mismatch in {name}")


def test_functions_values_at_known_optima():
    """Spot-check function values at documented optima."""
    assert abs(ALL_FUNCTIONS["rastrigin"]["fn"](np.zeros(5))) < 1e-12
    assert abs(ALL_FUNCTIONS["sphere"]["fn"](np.zeros(3))) < 1e-12
    assert abs(ALL_FUNCTIONS["rosenbrock"]["fn"](np.ones(4))) < 1e-12
    assert abs(EXTENDED_FUNCTIONS["levy"]["fn"](np.ones(6))) < 1e-10
    assert abs(EXTENDED_FUNCTIONS["zakharov"]["fn"](np.zeros(7))) < 1e-12
    d = 3
    st = EXTENDED_FUNCTIONS["styblinski_tang"]
    val_st = st["fn"](np.full(d, -2.903534))
    assert abs(val_st - (-39.16599 * d)) < 1e-2
    td = EXTENDED_FUNCTIONS["trid"]
    assert abs(td["min_val_fn"](10) - (-10 * 14 * 9 / 6)) < 1e-12
