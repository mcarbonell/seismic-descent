"""
benchmark_ackley.py — Seismic Descent vs SA vs CMA-ES en Ackley ND

Ackley tiene un único valle global suave en el origen rodeado de una
superficie casi plana con muchos mínimos locales pequeños. Estructura
muy diferente a Rastrigin — buen test de generalización.

Reutiliza rff_noise_nd y cmaes_nd de perlin_opt_nd_grf.py.
Refactoriza seismic_descent y SA para aceptar f como parámetro.
"""

import numpy as np
import cma
from perlin_opt_nd_grf import rff_noise_nd, cmaes_nd, EVAL_BUDGET_BASE

# ------------------------------------------------------------------ #
# Funciones objetivo
# ------------------------------------------------------------------ #

def ackley_nd(x, a=20, b=0.2, c=2*np.pi):
    x = np.asarray(x)
    D = len(x)
    return (-a * np.exp(-b * np.sqrt(np.sum(x**2) / D))
            - np.exp(np.sum(np.cos(c * x)) / D)
            + a + np.e)


# ------------------------------------------------------------------ #
# Seismic Descent genérico (f como parámetro)
# ------------------------------------------------------------------ #

def seismic_descent(f, x0, n_steps=2000, dt=0.1, noise_amplitude=10.0,
                    noise_decay=1.0, search_range=32.768):
    x = np.array(x0, dtype=float)
    D = len(x)
    t = 0.0
    best_x = x.copy()
    best_val = f(x)
    best_history = [best_val]
    eps = 1e-4

    for step in range(n_steps):
        decay = noise_decay ** step
        freq  = 2.0 * decay
        amp   = noise_amplitude * decay * np.sin(t * freq)

        noise_c = rff_noise_nd(x, t, amp)
        f_center = f(x) + noise_c

        x_mat = np.tile(x, (D, 1))
        np.fill_diagonal(x_mat, x_mat.diagonal() + eps)
        f_eps = np.array([f(x_mat[i]) + rff_noise_nd(x_mat[i], t, amp)
                          for i in range(D)])
        grad = (f_eps - f_center) / eps

        x -= dt * grad
        x = np.clip(x, -search_range, search_range)
        t += 0.05

        real_val = f(x)
        if real_val < best_val:
            best_val = real_val
            best_x = x.copy()
        best_history.append(best_val)

    return best_x, best_val, best_history


def simulated_annealing(f, x0, n_steps=5000, T0=10.0, cooling=0.999,
                        step_size=0.3, search_range=32.768):
    x = np.array(x0, dtype=float)
    current_val = f(x)
    best_x = x.copy()
    best_val = current_val
    T = T0

    for _ in range(n_steps):
        x_new = np.clip(x + np.random.normal(0, step_size, size=len(x)),
                        -search_range, search_range)
        new_val = f(x_new)
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            x, current_val = x_new, new_val
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
        T *= cooling

    return best_x, best_val


def cmaes(f, x0, eval_budget, search_range=32.768):
    opts = cma.CMAOptions()
    opts['bounds'] = [[-search_range] * len(x0), [search_range] * len(x0)]
    opts['maxfevals'] = eval_budget
    opts['verbose'] = -9
    es = cma.CMAEvolutionStrategy(x0, 4.0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [f(s) for s in solutions])
    return es.result.xbest, es.result.fbest


# ------------------------------------------------------------------ #
# Benchmark fair
# ------------------------------------------------------------------ #

def run_benchmark(f, fname, search_range, dims, n_trials=30, threshold=None):
    if threshold is None:
        threshold = 1.0

    eval_budget   = EVAL_BUDGET_BASE * (dims + 1)
    seismic_steps = EVAL_BUDGET_BASE
    sa_steps      = eval_budget

    np.random.seed(42)
    p_ok, sa_ok, cma_ok = 0, 0, 0
    p_vals, sa_vals, cma_vals = [], [], []

    for _ in range(n_trials):
        x0 = np.random.uniform(-search_range, search_range, size=dims)

        _, pval, _ = seismic_descent(f, x0, n_steps=seismic_steps,
                                     search_range=search_range)
        _, sval    = simulated_annealing(f, x0, n_steps=sa_steps,
                                         search_range=search_range)
        _, cval    = cmaes(f, list(x0), eval_budget=eval_budget,
                           search_range=search_range)

        p_vals.append(pval);   p_ok  += pval < threshold
        sa_vals.append(sval);  sa_ok += sval < threshold
        cma_vals.append(cval); cma_ok += cval < threshold

    print(f"\n=== {fname} {dims}D — {n_trials} pruebas | umbral f<{threshold} | budget={eval_budget} evals ===")
    print(f"Seismic+RFF: {p_ok:2}/{n_trials}  media={np.mean(p_vals):7.3f}  mediana={np.median(p_vals):7.3f}  ({seismic_steps} pasos)")
    print(f"SA:          {sa_ok:2}/{n_trials}  media={np.mean(sa_vals):7.3f}  mediana={np.median(sa_vals):7.3f}  ({sa_steps} pasos)")
    print(f"CMA-ES:      {cma_ok:2}/{n_trials}  media={np.mean(cma_vals):7.3f}  mediana={np.median(cma_vals):7.3f}  ({eval_budget} evals)")
    return p_vals, sa_vals, cma_vals


if __name__ == '__main__':
    import time
    # Ackley: dominio [-32.768, 32.768], óptimo global f(0,...,0) = 0
    # umbral 1.0 es razonable (Ackley en el óptimo = 0, mínimos locales ~3-10)
    for d, trials in [(2, 50), (5, 30), (10, 30), (20, 10)]:
        t0 = time.time()
        run_benchmark(ackley_nd, 'Ackley', 32.768, dims=d, n_trials=trials)
        print(f"  ({round(time.time()-t0)}s)")
