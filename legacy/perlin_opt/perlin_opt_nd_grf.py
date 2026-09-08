"""
perlin_opt_nd_grf.py — Seismic Descent con Random Fourier Features (RFF)

Sustituye el value noise (independiente por dimensión) por una aproximación
de Gaussian Random Field via Random Fourier Features (Rahimi & Recht, 2007).

El kernel squared-exponential tiene distribución espectral gaussiana, lo que
permite aproximarlo con R vectores de frecuencia en R^D:

  noise(x) ≈ sqrt(2/R) * A * sum_r cos(ω_r · x + t*drift_r + φ_r)

donde:
  ω_r ~ N(0, 1/l²·I)   — frecuencias sampleadas del espectro del kernel
  φ_r ~ Uniform(0, 2π) — fases aleatorias
  drift_r ~ Uniform     — hace evolucionar el campo con t (el "terremoto")
  l                     — lengthscale: controla el ancho de correlación espacial

A diferencia del ruido sinusoidal anterior, las ω_r son vectores en R^D,
lo que crea interferencia entre dimensiones y correlación espacial real en ND.
"""

import numpy as np
import cma
from perlin_opt_nd import rastrigin_nd, simulated_annealing_nd

# ------------------------------------------------------------------ #
# Random Fourier Features — Gaussian Random Field
# ------------------------------------------------------------------ #

_R = 64       # número de features (más = mejor aproximación, más coste)
_rng = np.random.default_rng(seed=1)

# Precomputar frecuencias y fases para hasta 100 dimensiones y 4 octavas
# Cada octava tiene su propio set de frecuencias con distinto lengthscale
_N_OCTAVES = 4
_OMEGAS = []  # lista de arrays (R, D_max) por octava
_PHIS   = _rng.uniform(0, 2 * np.pi, size=(_N_OCTAVES, _R))
_DRIFTS = _rng.uniform(0.1, 0.5,     size=(_N_OCTAVES, _R))

_D_MAX = 100
for o in range(_N_OCTAVES):
    lengthscale = 2.0 * (2.0 ** o)  # octava 0: l=2, octava 1: l=4, etc.
    omegas = _rng.normal(0, 1.0 / lengthscale, size=(_R, _D_MAX))
    _OMEGAS.append(omegas)


def rff_noise_nd(x, t, amplitude=15.0, octaves=4):
    """
    Ruido correlacionado espacialmente via Random Fourier Features.
    Correlación real en ND: dos puntos cercanos tienen ruido similar.
    """
    x = np.asarray(x)
    D = len(x)
    val = 0.0
    amp = amplitude

    for o in range(octaves):
        omegas = _OMEGAS[o][:, :D]          # (R, D)
        phis   = _PHIS[o]                    # (R,)
        drifts = _DRIFTS[o]                  # (R,)
        # producto punto: (R, D) · (D,) = (R,)
        projections = omegas @ x
        val += amp * np.sqrt(2.0 / _R) * np.sum(np.cos(projections + t * drifts + phis))
        amp *= 0.5

    return val


# ------------------------------------------------------------------ #
# Seismic Descent ND con RFF
# ------------------------------------------------------------------ #

def seismic_descent_rff(x0, n_steps=2000, dt=0.01, noise_amplitude=15.0,
                        noise_decay=1.0, search_range=5.12):
    x = np.array(x0, dtype=float)
    D = len(x)
    t = 0.0
    best_x = x.copy()
    best_val = rastrigin_nd(x)
    best_history = [best_val]
    eps = 1e-4

    for step in range(n_steps):
        decay = noise_decay ** step
        freq  = 2.0 * decay
        amp   = noise_amplitude * decay * abs(np.sin(t * freq))

        noise_center = rff_noise_nd(x, t, amp)
        f_center = rastrigin_nd(x) + noise_center

        # Gradiente numérico vectorizado
        x_mat = np.tile(x, (D, 1))
        np.fill_diagonal(x_mat, x_mat.diagonal() + eps)
        f_eps = np.array([rastrigin_nd(x_mat[i]) + rff_noise_nd(x_mat[i], t, amp)
                          for i in range(D)])
        grad = (f_eps - f_center) / eps

        x -= dt * grad
        x = np.clip(x, -search_range, search_range)
        t += 0.05

        real_val = rastrigin_nd(x)
        if real_val < best_val:
            best_val = real_val
            best_x = x.copy()
        best_history.append(best_val)

    return best_x, best_val, best_history


# ------------------------------------------------------------------ #
# CMA-ES (igual que fairbench)
# ------------------------------------------------------------------ #

def cmaes_nd(x0, eval_budget, search_range=5.12):
    opts = cma.CMAOptions()
    opts['bounds'] = [[-search_range] * len(x0), [search_range] * len(x0)]
    opts['maxfevals'] = eval_budget
    opts['verbose'] = -9
    es = cma.CMAEvolutionStrategy(x0, 4.0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [rastrigin_nd(s) for s in solutions])
    return es.result.xbest, es.result.fbest


# ------------------------------------------------------------------ #
# Benchmark fair (mismo presupuesto de evaluaciones)
# ------------------------------------------------------------------ #

EVAL_BUDGET_BASE = 2000

def run_benchmark(dims, n_trials=30, threshold=None):
    if threshold is None:
        threshold = 1.0 * dims

    eval_budget   = EVAL_BUDGET_BASE * (dims + 1)
    seismic_steps = EVAL_BUDGET_BASE
    sa_steps      = eval_budget

    np.random.seed(42)
    p_ok, sa_ok, cma_ok = 0, 0, 0
    p_vals, sa_vals, cma_vals = [], [], []

    for _ in range(n_trials):
        x0 = np.random.uniform(-5.12, 5.12, size=dims)

        _, pval, _ = seismic_descent_rff(x0, n_steps=seismic_steps)
        _, sval, _ = simulated_annealing_nd(x0, n_steps=sa_steps)
        _, cval    = cmaes_nd(list(x0), eval_budget=eval_budget)

        p_vals.append(pval);   p_ok  += pval < threshold
        sa_vals.append(sval);  sa_ok += sval < threshold
        cma_vals.append(cval); cma_ok += cval < threshold

    print(f"\n=== Rastrigin {dims}D — {n_trials} pruebas | umbral f<{threshold} | budget={eval_budget} evals ===")
    print(f"Seismic+RFF: {p_ok:2}/{n_trials}  media={np.mean(p_vals):7.3f}  mediana={np.median(p_vals):7.3f}  ({seismic_steps} pasos)")
    print(f"SA:          {sa_ok:2}/{n_trials}  media={np.mean(sa_vals):7.3f}  mediana={np.median(sa_vals):7.3f}  ({sa_steps} pasos)")
    print(f"CMA-ES:      {cma_ok:2}/{n_trials}  media={np.mean(cma_vals):7.3f}  mediana={np.median(cma_vals):7.3f}  ({eval_budget} evals)")
    return p_vals, sa_vals, cma_vals


if __name__ == '__main__':
    import time
    for d, trials in [(5, 30), (10, 30), (20, 10)]:
        t0 = time.time()
        run_benchmark(dims=d, n_trials=trials)
        print(f"  ({round(time.time()-t0)}s)")
