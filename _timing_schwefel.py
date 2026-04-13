import time, numpy as np
from benchmark_ackley import seismic_descent, simulated_annealing, cmaes
from benchmark_schwefel import schwefel_nd
from perlin_opt_nd_grf import EVAL_BUDGET_BASE

np.random.seed(0)
for d in [2, 5, 10, 20]:
    budget = EVAL_BUDGET_BASE * (d + 1)
    x0 = np.random.uniform(-500, 500, size=d)
    t0 = time.time(); seismic_descent(schwefel_nd, x0, n_steps=EVAL_BUDGET_BASE, search_range=500.0); tp = round(time.time()-t0, 2)
    t0 = time.time(); simulated_annealing(schwefel_nd, x0, n_steps=budget, search_range=500.0);       ts = round(time.time()-t0, 2)
    t0 = time.time(); cmaes(schwefel_nd, list(x0), eval_budget=budget, search_range=500.0);           tc = round(time.time()-t0, 2)
    print(f"{d}D budget={budget}: Seismic={tp}s SA={ts}s CMA={tc}s -> ~{round((tp+ts+tc)*30)}s total")
