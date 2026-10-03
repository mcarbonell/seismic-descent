"""Verificación 3: replicación reducida de las tablas del README.

README afirma (15 trials, budget 3000):
  Rastrigin 5D:  v20=9.206 | v23-ORF=6.452 | v23-OrthoLiss=30.63 | CMA-ES=2.984
  Rosenbrock 5D: v20=4.602 | v23-ORF=3.122 | v23-OrthoLiss=8.032 | CMA-ES~9.4e-15
  Runtimes: 0.28s / 0.51s / 0.87s / 3.22s
Y (budget 25k, Rastrigin 5D): Seismic=4.85 vs CMA-ES=6.96 -> "Seismic gana a 25k".

Aquí: 5 trials para budget 3000; 3 trials para 25k. Mismos seeds que el suite (trial+1).
"""
import time
import numpy as np
import sys
sys.path.insert(0, "/home/user/seismic-descent/src")
import cma

from seismic_descent.functions import ALL_FUNCTIONS
from seismic_descent.core import seismic_swarm
from seismic_descent.champion_v23 import seismic_champion_v23


def trials(fn_name, dim, budget, n_trials):
    f = ALL_FUNCTIONS[fn_name]
    r = f["search_range"]
    bounds = np.array([[-r, r]] * dim)
    n_steps = budget // 10  # 10 partículas -> evals/step = 10 (+10 inicial); aprox budget
    res = {"v20": [], "v23orf": [], "v23oliss": [], "cma": []}
    times = {"v20": 0.0, "v23orf": 0.0, "v23oliss": 0.0, "cma": 0.0}
    for trial in range(n_trials):
        seed = trial + 1
        x0 = np.full(dim, r * 0.6)
        t0 = time.perf_counter()
        _, v, _ = seismic_swarm(fn=f["fn"], fn_grad=f["grad"], x0_real=x0, bounds=bounds,
                                n_steps=n_steps, n_particles=10, seed=seed)
        res["v20"].append(v); times["v20"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        _, v, _ = seismic_champion_v23(fn=f["fn"], fn_grad=f["grad"], x0_real=x0, bounds=bounds,
                                       n_steps=n_steps, n_particles=10, noise_engine="orf", seed=seed)
        res["v23orf"].append(v); times["v23orf"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        _, v, _ = seismic_champion_v23(fn=f["fn"], fn_grad=f["grad"], x0_real=x0, bounds=bounds,
                                       n_steps=n_steps, n_particles=10, noise_engine="orthogonal_lissajous", seed=seed)
        res["v23oliss"].append(v); times["v23oliss"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        x0c = rng.uniform(-r, r, size=dim).tolist()
        es = cma.CMAEvolutionStrategy(x0c, 0.2 * 2 * r, {"maxfevals": budget, "verbose": -9,
                                                         "bounds": [-r, r]})
        es.optimize(f["fn"])
        res["cma"].append(es.result.fbest); times["cma"] += time.perf_counter() - t0
    for k in res:
        times[k] /= n_trials
    return {k: float(np.median(v)) for k, v in res.items()}, times


for fn_name, dim in (("rastrigin", 5), ("rosenbrock", 5), ("rastrigin", 10)):
    med, tms = trials(fn_name, dim, 3000, 5)
    print(f"\n{fn_name} {dim}D budget=3000 (5 trials, mediana):")
    for k in ("v20", "v23orf", "v23oliss", "cma"):
        print(f"  {k:10s} mediana={med[k]:>10.4g}   tiempo={tms[k]:.2f}s")

print("\n--- Spot check budget=25000 Rastrigin 5D (3 trials) ---")
med, tms = trials("rastrigin", 5, 25000, 3)
for k in ("v20", "v23orf", "v23oliss", "cma"):
    print(f"  {k:10s} mediana={med[k]:>10.4g}   tiempo={tms[k]:.2f}s")
