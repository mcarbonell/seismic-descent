"""
perlin_opt_nd_fairbench.py — Benchmark con presupuesto de evaluaciones igualado

Seismic Descent usa D+1 evaluaciones por paso (gradiente numérico).
Para comparar honestamente, todos los algoritmos reciben el mismo
presupuesto de evaluaciones de función reales: EVAL_BUDGET = 2000 * (D+1)

- Seismic Descent: EVAL_BUDGET / (D+1) pasos
- SA:              EVAL_BUDGET pasos (1 eval/paso)
- CMA-ES:          EVAL_BUDGET evaluaciones totales
"""

import numpy as np
import cma
from perlin_opt_nd import rastrigin_nd, value_noise_nd, perlin_optimization_nd, simulated_annealing_nd

EVAL_BUDGET_BASE = 2000  # evaluaciones base (equivale a 2000 pasos de Seismic en 1D)


def cmaes_nd_budget(x0, eval_budget, search_range=5.12):
    opts = cma.CMAOptions()
    opts['bounds'] = [[-search_range] * len(x0), [search_range] * len(x0)]
    opts['maxfevals'] = eval_budget
    opts['verbose'] = -9
    es = cma.CMAEvolutionStrategy(x0, 4.0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [rastrigin_nd(s) for s in solutions])
    return es.result.xbest, es.result.fbest


def run_fair_benchmark(dims, n_trials=30, threshold=None):
    if threshold is None:
        threshold = 1.0 * dims

    eval_budget = EVAL_BUDGET_BASE * (dims + 1)
    seismic_steps = EVAL_BUDGET_BASE          # = eval_budget / (dims+1)
    sa_steps      = eval_budget               # 1 eval/paso
    cma_evals     = eval_budget

    np.random.seed(42)
    p_ok, sa_ok, cma_ok = 0, 0, 0
    p_vals, sa_vals, cma_vals = [], [], []

    for _ in range(n_trials):
        x0 = np.random.uniform(-5.12, 5.12, size=dims)

        _, pval, _ = perlin_optimization_nd(x0, n_steps=seismic_steps)
        _, sval, _ = simulated_annealing_nd(x0, n_steps=sa_steps)
        _, cval    = cmaes_nd_budget(list(x0), eval_budget=cma_evals)

        p_vals.append(pval);   p_ok  += pval < threshold
        sa_vals.append(sval);  sa_ok += sval < threshold
        cma_vals.append(cval); cma_ok += cval < threshold

    print(f"\n=== Rastrigin {dims}D — {n_trials} pruebas | umbral f<{threshold} | budget={eval_budget} evals ===")
    print(f"Seismic Descent: {p_ok:2}/{n_trials}  media={np.mean(p_vals):7.3f}  mediana={np.median(p_vals):7.3f}  ({seismic_steps} pasos)")
    print(f"SA:              {sa_ok:2}/{n_trials}  media={np.mean(sa_vals):7.3f}  mediana={np.median(sa_vals):7.3f}  ({sa_steps} pasos)")
    print(f"CMA-ES:          {cma_ok:2}/{n_trials}  media={np.mean(cma_vals):7.3f}  mediana={np.median(cma_vals):7.3f}  ({cma_evals} evals)")
    return p_vals, sa_vals, cma_vals


if __name__ == '__main__':
    import time
    for d, trials in [(5, 30), (10, 30), (20, 10)]:
        t0 = time.time()
        run_fair_benchmark(dims=d, n_trials=trials)
        print(f"  ({round(time.time()-t0)}s)")
