"""
benchmark_schwefel.py — Seismic Descent vs SA vs CMA-ES en Schwefel ND

Schwefel: f(x) = 418.9829*D - sum(x_i * sin(sqrt(|x_i|)))
Dominio: [-500, 500]^D
Optimo global: x_i = 420.9687 para todo i, f = 0

Caracteristicas:
- Optimo global en una esquina del dominio (lejos del centro)
- Gradiente informativo (no hay meseta como en Ackley)
- Muchos minimos locales profundos y engañosos
- Penaliza exploracion simetrica centrada en el origen

Reutiliza seismic_descent, simulated_annealing y cmaes de benchmark_ackley.py.
"""

import numpy as np
from benchmark_ackley_no_abs import seismic_descent, simulated_annealing, cmaes, run_benchmark

def schwefel_nd(x):
    x = np.asarray(x)
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


if __name__ == '__main__':
    import time
    # Schwefel: dominio [-500, 500], optimo ~0, umbral 50*D es generoso
    # pero razonable dado lo dificil que es encontrar x_i=420.9687 en cada dim
    for d, trials in [(2, 50), (5, 30), (10, 30), (20, 10)]:
        t0 = time.time()
        run_benchmark(schwefel_nd, 'Schwefel', 500.0, dims=d,
                      n_trials=trials, threshold=50.0 * d)
        print(f"  ({round(time.time()-t0)}s)")
