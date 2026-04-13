import time, numpy as np
from perlin_opt_nd_grf import seismic_descent_rff, cmaes_nd, EVAL_BUDGET_BASE
from perlin_opt_nd import simulated_annealing_nd

np.random.seed(0)
for d in [5, 10, 20]:
    budget = EVAL_BUDGET_BASE * (d + 1)
    x0 = np.random.uniform(-5.12, 5.12, size=d)
    t0 = time.time(); seismic_descent_rff(x0, n_steps=EVAL_BUDGET_BASE);  tp = round(time.time()-t0, 2)
    t0 = time.time(); simulated_annealing_nd(x0, n_steps=budget);          ts = round(time.time()-t0, 2)
    t0 = time.time(); cmaes_nd(list(x0), eval_budget=budget);              tc = round(time.time()-t0, 2)
    print(f"{d}D budget={budget}: Seismic={tp}s SA={ts}s CMA={tc}s -> ~{round((tp+ts+tc)*30)}s total")
