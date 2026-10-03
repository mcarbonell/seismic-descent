"""Reference baselines for rigorous comparison — all fully seeded and with
explicit, documented configurations.

Baselines provided:
- ``run_random_search``: uniform sampling (sanity floor).
- ``run_cmaes``: baseline CMA-ES via pycma (seeded).
- ``run_ipop_cmaes``: IPOP restart strategy (population doubling on stop).
  This is the *fair* version of CMA-ES for multimodal landscapes: plain
  CMA-ES without restarts is known to stall on Rastrigin-type functions.
- ``run_pso``: best-of-swarm PSO (default coefficients documented below).
- ``run_bfgs_multistart``: L-BFGS-B from multiple random starts using the
  same analytic gradients the seismic optimizers receive (first-order control).

Every function returns ``{"best_val": float, "n_evals": int}`` and uses an
explicit :class:`numpy.random.Generator` so results are bit-reproducible.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import cma
    _CMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CMA_AVAILABLE = False

try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False


@dataclass
class BaselineResult:
    best_val: float
    best_x: np.ndarray
    n_evals: int
    meta: Dict = field(default_factory=dict)


def _as_batch(fn: Callable, x: np.ndarray) -> np.ndarray:
    v = np.asarray(fn(x), dtype=np.float64)
    return np.atleast_1d(v)


# --------------------------------------------------------------------------- #

def run_random_search(
    fn: Callable,
    bounds: np.ndarray,
    budget: int,
    seed: int = 1,
) -> BaselineResult:
    """Uniform random search over the box. Sanity floor for any study."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    best_val, best_x, evals = np.inf, lo.copy(), 0
    chunk = 1000
    while evals < budget:
        n = min(chunk, budget - evals)
        X = rng.uniform(lo, hi, size=(n, len(lo)))
        vals = _as_batch(fn, X)
        i = int(np.argmin(vals))
        if float(vals[i]) < best_val:
            best_val, best_x = float(vals[i]), X[i].copy()
        evals += n
    return BaselineResult(best_val, best_x, evals)


# --------------------------------------------------------------------------- #

def run_cmaes(
    fn: Callable,
    x0: np.ndarray,
    bounds: np.ndarray,
    budget: int,
    seed: int = 1,
    sigma0: Optional[float] = None,
) -> BaselineResult:
    """Plain (1-time) CMA-ES. Documented config: sigma0 = 20% of mean range."""
    if not _CMA_AVAILABLE:
        raise ImportError("pycma is required for the CMA-ES baseline (pip install cma).")
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = float(np.mean(hi - lo))
    sigma0 = float(sigma0) if sigma0 is not None else 0.2 * span
    opts = {
        "bounds": [lo.tolist(), hi.tolist()],
        "maxfevals": int(budget),
        "verbose": -9,
        "seed": int(seed),
    }
    es = cma.CMAEvolutionStrategy(list(np.asarray(x0, dtype=float)), sigma0, opts)
    evals = 1
    best_val = float(fn(np.asarray(x0, dtype=float)))
    best_x = np.asarray(x0, dtype=float).copy()
    while not es.stop() and evals < budget:
        X = es.ask()
        V = [float(fn(np.array(s))) for s in X]
        es.tell(X, V)
        evals += len(X)
        i = int(np.argmin(V))
        if V[i] < best_val:
            best_val, best_x = V[i], np.array(X[i])
    return BaselineResult(best_val, best_x, evals, {"sigma0": sigma0})


def run_ipop_cmaes(
    fn: Callable,
    x0: np.ndarray,
    bounds: np.ndarray,
    budget: int,
    seed: int = 1,
    max_restarts: int = 20,
) -> BaselineResult:
    """IPOP-CMA-ES: restart with doubled population whenever CMA-ES stops.

    Reference: Auger & Hansen (2005), "A Restart CMA Evolution Strategy With
    Increasing Population Size". This is the fair CMA-ES for multimodal
    benchmarks; plain CMA-ES is known to prematurely converge on them.
    """
    if not _CMA_AVAILABLE:
        raise ImportError("pycma is required for the IPOP-CMA-ES baseline.")
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    dim = len(lo)
    span = float(np.mean(hi - lo))
    sigma0 = 0.2 * span
    pop_mult = 1.0

    best_val, best_x, evals = np.inf, np.asarray(x0, dtype=float).copy(), 0
    restarts = 0
    x_start = np.asarray(x0, dtype=float)

    while evals < budget and restarts <= max_restarts:
        popsize = int(round((4 + int(3 * np.log(dim))) * pop_mult))
        popsize = max(4, popsize)
        opts = {
            "bounds": [lo.tolist(), hi.tolist()],
            "maxfevals": int(budget - evals),
            "verbose": -9,
            "seed": int(seed * 1000 + restarts + 1),
            "popsize": popsize,
        }
        es = cma.CMAEvolutionStrategy(list(x_start), sigma0, opts)
        while not es.stop() and evals < budget:
            X = es.ask()
            V = [float(fn(np.array(s))) for s in X]
            es.tell(X, V)
            evals += len(X)
            i = int(np.argmin(V))
            if V[i] < best_val:
                best_val, best_x = V[i], np.array(X[i])
        if evals >= budget:
            break
        # Prepare next restart: double population, new random start
        pop_mult *= 2.0
        restarts += 1
        x_start = rng.uniform(lo, hi, size=dim)

    return BaselineResult(best_val, best_x, evals,
                          {"restarts_used": restarts, "sigma0": sigma0})


# --------------------------------------------------------------------------- #

def run_pso(
    fn: Callable,
    bounds: np.ndarray,
    budget: int,
    seed: int = 1,
    n_particles: int = 40,
    w: float = 0.7298,
    c1: float = 1.49618,
    c2: float = 1.49618,
    max_steps: int = 10_000,
) -> BaselineResult:
    """Global-best PSO with constriction-style defaults.

    Defaults are the classic Clerc & Kennedy (2002) constriction coefficients
    (w=0.7298, c1=c2=1.49618). Swarm of 40 particles; each step consumes
    ``n_particles`` evaluations of ``fn``.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    dim = len(lo)
    span = hi - lo

    X = rng.uniform(lo, hi, size=(n_particles, dim))
    V = rng.uniform(-0.1, 0.1, size=(n_particles, dim)) * span
    vals = _as_batch(fn, X)
    evals = n_particles

    pbest_x, pbest_v = X.copy(), vals.copy()
    i = int(np.argmin(vals))
    gbest_x, gbest_v = X[i].copy(), float(vals[i])

    steps = 0
    while evals + n_particles <= budget and steps < max_steps:
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        V = w * V + c1 * r1 * (pbest_x - X) + c2 * r2 * (gbest_x - X)
        X = np.clip(X + V, lo, hi)
        vals = _as_batch(fn, X)
        evals += n_particles
        improved = vals < pbest_v
        pbest_x[improved] = X[improved]
        pbest_v[improved] = vals[improved]
        i = int(np.argmin(pbest_v))
        if float(pbest_v[i]) < gbest_v:
            gbest_v, gbest_x = float(pbest_v[i]), pbest_x[i].copy()
        steps += 1

    return BaselineResult(gbest_v, gbest_x, evals,
                          {"n_particles": n_particles, "w": w, "c1": c1, "c2": c2})


# --------------------------------------------------------------------------- #

def run_bfgs_multistart(
    fn: Callable,
    fn_grad: Callable,
    bounds: np.ndarray,
    budget: int,
    seed: int = 1,
) -> BaselineResult:
    """L-BFGS-B multistart with analytic gradients (first-order control).

    Evaluation accounting is exact: every objective evaluation and every
    gradient evaluation consumes one budget unit each (the standard,
    conservative 1:1 rate for first-order methods), tracked by wrapping
    ``fn``/``fn_grad``. Restarts until the budget is exhausted.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError("scipy is required for the BFGS multistart baseline.")
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    counter = {"evals": 0}

    def counted_fn(xx):
        counter["evals"] += 1
        return float(fn(xx))

    def counted_grad(xx):
        counter["evals"] += 1
        return np.asarray(fn_grad(xx), dtype=np.float64)

    best_val, best_x, = float(counted_fn(rng.uniform(lo, hi))), None
    starts = 0
    while counter["evals"] < budget:
        x0 = rng.uniform(lo, hi)

        def fg(xx):
            return counted_fn(xx), counted_grad(xx)

        try:
            res = _scipy_minimize(
                fg, x0, method="L-BFGS-B", jac=True,
                bounds=list(zip(lo, hi)),
                options={"maxiter": budget - counter["evals"]},
            )
            cand_v, cand_x = float(res.fun), np.asarray(res.x)
        except Exception:  # pragma: no cover - numerical corner
            cand_v, cand_x = float(counted_fn(x0)), x0
        if best_x is None or cand_v < best_val:
            best_val, best_x = cand_v, cand_x
        starts += 1
        if budget - counter["evals"] < 20:
            break

    return BaselineResult(best_val, best_x, counter["evals"], {"starts": starts})
