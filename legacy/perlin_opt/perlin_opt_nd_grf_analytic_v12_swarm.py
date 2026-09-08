"""
perlin_opt_nd_grf_analytic_v12_swarm.py — Seismic Swarm con RFF y Gradiente Analítico

Esta versión evalúa un "enjambre" de partículas simultáneamente sobre el mismo
campo de ruido RFF, paralelizando la exploración costosa del ruido RFF.
Las partículas aprovechan las mismas features, amortizando el cálculo.
"""

import numpy as np
import cma
from perlin_opt_nd import simulated_annealing_nd

# ------------------------------------------------------------------ #
# Funciones Objetivo Vectorizadas (N_particles, D)
# ------------------------------------------------------------------ #

def rastrigin_nd_vec(X):
    """Evalúa Rastrigin para un bloque de partículas X de tamaño (N, D)."""
    if X.ndim == 1:
        return 10.0 * len(X) + np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X))
    D = X.shape[1]
    return 10.0 * D + np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1)

def rastrigin_grad_nd_vec(X):
    """Gradiente analítico de Rastrigin N-dimensional vectorizado."""
    return 2.0 * X + 20.0 * np.pi * np.sin(2.0 * np.pi * X)


# ------------------------------------------------------------------ #
# Random Fourier Features — Vectorizado
# ------------------------------------------------------------------ #

_R = 64
_rng = np.random.default_rng(seed=1)

_N_OCTAVES = 4
_OMEGAS = []
_PHIS   = _rng.uniform(0, 2 * np.pi, size=(_N_OCTAVES, _R))
_DRIFTS = _rng.uniform(0.1, 0.5,     size=(_N_OCTAVES, _R))

_D_MAX = 100
for o in range(_N_OCTAVES):
    lengthscale = 2.0 * (2.0 ** o)
    omegas = _rng.normal(0, 1.0 / lengthscale, size=(_R, _D_MAX))
    _OMEGAS.append(omegas)


def rff_noise_and_grad_nd_vec(X, t, amplitude=15.0, octaves=4):
    """
    Calcula simultáneamente el gradiente RFF para N partículas (shape de X: N, D).
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    N, D = X.shape
    grad = np.zeros((N, D))
    amp = amplitude
    sqrt_2_R = np.sqrt(2.0 / _R)

    for o in range(octaves):
        omegas = _OMEGAS[o][:, :D]           # (R, D)
        phis   = _PHIS[o][:, None]           # (R, 1)
        drifts = _DRIFTS[o][:, None]         # (R, 1)
        
        # Proyecciones: (R, D) @ (D, N) -> (R, N)
        projections = omegas @ X.T
        angles = projections + t * drifts + phis
        
        sines = np.sin(angles)               # (R, N)
        
        # Gradiente: (D, R) @ (R, N) -> (D, N). Trasponer a (N, D)
        grad_contrib = (omegas.T @ sines).T
        grad -= amp * sqrt_2_R * grad_contrib
        
        amp *= 0.5

    return grad


# ------------------------------------------------------------------ #
# Seismic Swarm RFF
# ------------------------------------------------------------------ #

def seismic_swarm_rff_analytic(x0, n_steps=2000, n_particles=10, dt=0.01, 
                               noise_amplitude=15.0, noise_decay=1.0, search_range=5.12):
    D = len(x0)
    
    # Inicializar enjambre (1 partícula en x0, resto aleatorias)
    X = np.random.uniform(-search_range, search_range, size=(n_particles, D))
    X[0] = np.array(x0, dtype=float)
    
    t = 0.0
    
    # Eval inicial
    real_vals = rastrigin_nd_vec(X)
    best_idx = np.argmin(real_vals)
    best_val = real_vals[best_idx]
    best_x = X[best_idx].copy()

    for step in range(n_steps):
        decay = noise_decay ** step
        freq  = 2.0 * decay
        amp   = noise_amplitude * decay * np.sin(t * freq)

        f_grad = rastrigin_grad_nd_vec(X)      # (N, D)
        noise_grad = rff_noise_and_grad_nd_vec(X, t, amp)  # (N, D)
        
        grad = f_grad + noise_grad

        X -= dt * grad
        np.clip(X, -search_range, search_range, out=X)
        t += 0.05

        # 1 evaluación real en cada paso por partícula
        real_vals = rastrigin_nd_vec(X)
        step_best_idx = np.argmin(real_vals)
        if real_vals[step_best_idx] < best_val:
            best_val = real_vals[step_best_idx]
            best_x = X[step_best_idx].copy()

    return best_x, best_val, None


# ------------------------------------------------------------------ #
# CMA-ES (baseline)
# ------------------------------------------------------------------ #

def cmaes_nd(x0, eval_budget, search_range=5.12):
    opts = cma.CMAOptions()
    opts['bounds'] = [[-search_range] * len(x0), [search_range] * len(x0)]
    opts['maxfevals'] = eval_budget
    opts['verbose'] = -9
    es = cma.CMAEvolutionStrategy(x0, 4.0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [rastrigin_nd_vec(s) for s in solutions])
    return es.result.xbest, es.result.fbest


# ------------------------------------------------------------------ #
# Benchmark
# ------------------------------------------------------------------ #

EVAL_BUDGET_BASE = 500

def run_benchmark(dims, n_trials=30, threshold=None, n_particles=10):
    if threshold is None:
        threshold = 1.0 * dims

    eval_budget = EVAL_BUDGET_BASE * (dims + 1)
    
    # El enjambre hace n_particles evals por paso.
    seismic_steps = eval_budget // n_particles
    # SA es 1 partícula que hace 1 eval por paso.
    sa_steps = eval_budget

    np.random.seed(42)
    p_ok, sa_ok, cma_ok = 0, 0, 0
    p_vals, sa_vals, cma_vals = [], [], []

    import time
    t_seismic = 0
    t_sa = 0
    t_cma = 0

    for _ in range(n_trials):
        x0 = np.random.uniform(-5.12, 5.12, size=dims)

        t0 = time.time()
        _, pval, _ = seismic_swarm_rff_analytic(x0, n_steps=seismic_steps, n_particles=n_particles)
        t_seismic += time.time() - t0

        t0 = time.time()
        _, sval, _ = simulated_annealing_nd(x0, n_steps=sa_steps)
        t_sa += time.time() - t0

        t0 = time.time()
        _, cval = cmaes_nd(list(x0), eval_budget=eval_budget)
        t_cma += time.time() - t0

        p_vals.append(pval);   p_ok  += pval < threshold
        sa_vals.append(sval);  sa_ok += sval < threshold
        cma_vals.append(cval); cma_ok += cval < threshold

    print(f"\n=== Rastrigin {dims}D — {n_trials} pruebas | umbral f<{threshold} | budget={eval_budget} evals ===")
    print(f"Seismic+RFF Swarm: {p_ok:2}/{n_trials}  media={np.mean(p_vals):7.3f}  mediana={np.median(p_vals):7.3f}  ({n_particles} part, {seismic_steps} pasos) t={t_seismic:.2f}s")
    print(f"SA:                {sa_ok:2}/{n_trials}  media={np.mean(sa_vals):7.3f}  mediana={np.median(sa_vals):7.3f}  ({sa_steps} pasos)  t={t_sa:.2f}s")
    print(f"CMA-ES:            {cma_ok:2}/{n_trials}  media={np.mean(cma_vals):7.3f}  mediana={np.median(cma_vals):7.3f}  ({eval_budget} evals)  t={t_cma:.2f}s")
    return p_vals, sa_vals, cma_vals


if __name__ == '__main__':
    for d, trials in [(5, 30), (10, 30), (20, 10), (50, 5)]:
        run_benchmark(dims=d, n_trials=trials)
