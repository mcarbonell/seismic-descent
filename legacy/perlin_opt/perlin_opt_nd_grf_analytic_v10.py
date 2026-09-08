"""
perlin_opt_nd_grf_analytic.py — Seismic Descent con RFF y Gradiente Analítico

Esta versión elimina el cuello de botella computacional del gradiente numérico.
Al usar gradientes analíticos tanto para la función objetivo (Rastrigin)
como para el campo de ruido RFF, el coste por iteración pasa de O(D) a O(1)
evaluaciones de función.
"""

import numpy as np
import cma
from perlin_opt_nd import rastrigin_nd, simulated_annealing_nd

# ------------------------------------------------------------------ #
# Gradiente Analítico de Rastrigin
# ------------------------------------------------------------------ #

def rastrigin_grad_nd(x):
    """
    Gradiente analítico de Rastrigin N-dimensional.
    f(x) = 10D + sum(x_i^2 - 10*cos(2*pi*x_i))
    df/dx_i = 2*x_i + 20*pi*sin(2*pi*x_i)
    """
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


# ------------------------------------------------------------------ #
# Random Fourier Features — Gaussian Random Field
# ------------------------------------------------------------------ #

_R = 64       # número de features (más = mejor aproximación, más coste)

_N_OCTAVES = 4
_PHIS   = None
_DRIFTS = None
_OMEGAS = None
_CURRENT_D = None

def init_rff(D):
    global _PHIS, _DRIFTS, _OMEGAS, _CURRENT_D
    if _CURRENT_D == D:
        return
    _CURRENT_D = D
    rng = np.random.default_rng(seed=1)
    
    _PHIS   = rng.uniform(0, 2 * np.pi, size=(_N_OCTAVES, _R))
    _DRIFTS = rng.uniform(0.1, 0.5,     size=(_N_OCTAVES, _R))
    _OMEGAS = []

    # Escalar lengthscales con la raíz cuadrada de la dimensión
    l_base = 5.0 * np.sqrt(D) / 4.0
    
    for o in range(_N_OCTAVES):
        # Resultando en rumbos de l_base/4, l_base/2, l_base, l_base*2
        lengthscale = l_base * (2.0 ** (o - 2))
        omegas = rng.normal(0, 1.0 / lengthscale, size=(_R, D))
        _OMEGAS.append(omegas)


def rff_noise_and_grad_nd(x, t, amplitude=15.0, octaves=4):
    """
    Calcula simultáneamente el valor del ruido RFF y su gradiente analítico.
    dV/dx = sum_o amp_o * sqrt(2/R) * (-sin(omegas @ x + t*drifts + phis) @ omegas)
    """
    x = np.asarray(x)
    D = len(x)
    init_rff(D)
    val = 0.0
    grad = np.zeros(D)
    amp = amplitude
    sqrt_2_R = np.sqrt(2.0 / _R)

    for o in range(octaves):
        omegas = _OMEGAS[o][:, :D]           # (R, D)
        phis   = _PHIS[o]                    # (R,)
        drifts = _DRIFTS[o]                  # (R,)
        
        # Producto punto: (R, D) @ (D,) = (R,)
        projections = omegas @ x
        angles = projections + t * drifts + phis
        
        # Valor del ruido
        val += amp * sqrt_2_R * np.sum(np.cos(angles))
        
        # Gradiente del ruido
        # Derivada de cos(w*x + ...) es -sin(w*x + ...) * w
        sines = np.sin(angles)  # (R,)
        
        # Vectorización: multiplicamos cada seno por el vector omega correspondiente y sumamos
        # (D, R) @ (R,) = (D,)
        grad -= amp * sqrt_2_R * (omegas.T @ sines)
        
        amp *= 0.5

    return val, grad


# ------------------------------------------------------------------ #
# Seismic Descent ND con RFF y Gradiente Analítico
# ------------------------------------------------------------------ #

def seismic_descent_rff_analytic(x0, n_steps=2000, dt=0.01, noise_amplitude=15.0,
                                 noise_decay=1.0, search_range=5.12):
    x = np.array(x0, dtype=float)
    t = 0.0
    best_x = x.copy()
    
    # 1 evaluación real
    best_val = rastrigin_nd(x)
    best_history = [best_val]

    for step in range(n_steps):
        decay = noise_decay ** step
        freq  = 2.0 * decay
        amp   = noise_amplitude * decay * np.sin(t * freq)

        # Gradientes analíticos (0 evaluaciones de la función objetivo reales aquí, solo matemáticas)
        f_grad = rastrigin_grad_nd(x)
        noise_val, noise_grad = rff_noise_and_grad_nd(x, t, amp)
        
        grad = f_grad + noise_grad

        x -= dt * grad
        x = np.clip(x, -search_range, search_range)
        t += 0.05

        # 1 evaluación real en cada paso para registrar la mejor encontrada
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
# Benchmark fair analítico vs numérico vs baseline
# ------------------------------------------------------------------ #

EVAL_BUDGET_BASE = 500

def run_benchmark(dims, n_trials=30, threshold=None):
    if threshold is None:
        threshold = 1.0 * dims

    # En la versión analítica, cada paso de Seismic Descent hace:
    # EXACTAMENTE 1 evaluación para rastrigin_nd (para rastrear el mejor valor)
    # En la numérico hacía (D + 1) evaluaciones por paso.
    # Por tanto, ahora podemos usar la totalidad del presupuesto en "pasos"
    eval_budget   = EVAL_BUDGET_BASE * (dims + 1)
    
    seismic_steps = eval_budget  # Todo el budget se puede usar para explorar
    sa_steps      = eval_budget

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
        _, pval, _ = seismic_descent_rff_analytic(x0, n_steps=seismic_steps)
        t_seismic += time.time() - t0

        t0 = time.time()
        _, sval, _ = simulated_annealing_nd(x0, n_steps=sa_steps)
        t_sa += time.time() - t0

        t0 = time.time()
        _, cval    = cmaes_nd(list(x0), eval_budget=eval_budget)
        t_cma += time.time() - t0

        p_vals.append(pval);   p_ok  += pval < threshold
        sa_vals.append(sval);  sa_ok += sval < threshold
        cma_vals.append(cval); cma_ok += cval < threshold

    print(f"\n=== Rastrigin {dims}D — {n_trials} pruebas | umbral f<{threshold} | budget={eval_budget} evals ===")
    print(f"Seismic+RFF (Analytic): {p_ok:2}/{n_trials}  media={np.mean(p_vals):7.3f}  mediana={np.median(p_vals):7.3f}  ({seismic_steps} pasos)  t={t_seismic:.2f}s")
    print(f"SA:                     {sa_ok:2}/{n_trials}  media={np.mean(sa_vals):7.3f}  mediana={np.median(sa_vals):7.3f}  ({sa_steps} pasos)  t={t_sa:.2f}s")
    print(f"CMA-ES:                 {cma_ok:2}/{n_trials}  media={np.mean(cma_vals):7.3f}  mediana={np.median(cma_vals):7.3f}  ({eval_budget} evals)  t={t_cma:.2f}s")
    return p_vals, sa_vals, cma_vals


if __name__ == '__main__':
    # Probamos hasta 50 dimensiones ahora que somos eficientes
    for d, trials in [(5, 30), (10, 30), (20, 10), (50, 5)]:
        run_benchmark(dims=d, n_trials=trials)
