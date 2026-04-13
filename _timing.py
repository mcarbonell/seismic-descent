import time
import numpy as np
from perlin_opt_nd import perlin_optimization_nd, simulated_annealing_nd, cmaes_nd

np.random.seed(0)
for d in [5, 10, 20]:
    x0 = np.random.uniform(-5.12, 5.12, size=d)

    t0 = time.time()
    perlin_optimization_nd(x0, n_steps=2000)
    tp = round(time.time() - t0, 2)

    t0 = time.time()
    simulated_annealing_nd(x0, n_steps=2000)
    ts = round(time.time() - t0, 2)

    t0 = time.time()
    cmaes_nd(list(x0), n_steps=20000)
    tc = round(time.time() - t0, 2)

    total = round((tp + ts + tc) * 30)
    print(f"{d}D: Perlin={tp}s  SA={ts}s  CMA={tc}s  -> ~{total}s para 30 trials")
