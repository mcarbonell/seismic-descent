"""
perlin_opt_nd.py — Seismic Descent, extensión N-dimensional

Sustituye pnoise2 por opensimplex, que escala a dimensiones arbitrarias.
El algoritmo base (2D) está en perlin_opt.py, este archivo no lo modifica.
"""

import numpy as np
import cma

# ------------------------------------------------------------------ #
# Funciones objetivo N-D
# ------------------------------------------------------------------ #

def rastrigin_nd(x):
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)


# ------------------------------------------------------------------ #
# Value noise N-D
#
# Grid de valores aleatorios fija con interpolacion suave (smoothstep).
# Sin deriva sistematica, isotropico, multiescala con octavas.
# ------------------------------------------------------------------ #

_GRID_SIZE = 64
_rng = np.random.default_rng(seed=0)
# Grid 3D: [dim, grid_x, grid_t] — cada dimension tiene su propia grid
_GRIDS = _rng.uniform(-1, 1, size=(100, _GRID_SIZE, _GRID_SIZE))

def _smoothstep(t):
    return t * t * (3 - 2 * t)

def _value_noise_1d(xi, t, grid):
    """Interpolacion bilineal suave en la grid (xi, t)."""
    # mapear a coordenadas de grid
    gx = (xi / 5.12 * 0.5 + 0.5) * (_GRID_SIZE - 1)
    gt = (t % (2 * np.pi)) / (2 * np.pi) * (_GRID_SIZE - 1)
    x0, t0 = int(gx) % _GRID_SIZE, int(gt) % _GRID_SIZE
    x1, t1 = (x0 + 1) % _GRID_SIZE, (t0 + 1) % _GRID_SIZE
    fx, ft = _smoothstep(gx - int(gx)), _smoothstep(gt - int(gt))
    return (grid[x0, t0] * (1-fx) * (1-ft) +
            grid[x1, t0] *    fx  * (1-ft) +
            grid[x0, t1] * (1-fx) *    ft  +
            grid[x1, t1] *    fx  *    ft)

def value_noise_nd(x, t, amplitude=15.0, octaves=4):
    """Value noise multiescala por dimension, sin deriva sistematica."""
    val = 0.0
    amp = amplitude
    t_scaled = t
    for o in range(octaves):
        for i, xi in enumerate(x):
            val += (amp / len(x)) * _value_noise_1d(xi / (2.0 ** o), t_scaled, _GRIDS[i + o * 10])
        amp *= 0.5
        t_scaled *= 2.0
    return val


# ------------------------------------------------------------------ #
# Perlin Descent N-D
# ------------------------------------------------------------------ #

def perlin_optimization_nd(x0, n_steps=5000, dt=0.01, noise_amplitude=15.0,
                            noise_decay=1.0, search_range=5.12):
    x = np.array(x0, dtype=float)
    t = 0.0
    best_x = x.copy()
    best_val = rastrigin_nd(x)
    best_history = [best_val]
    eps = 1e-4

    for step in range(n_steps):
        decay = noise_decay ** step
        freq = 2.0 * decay
        amp = noise_amplitude * decay * abs(np.sin(t * freq))

        # Gradiente numérico: Rastrigin tiene gradiente analítico,
        # el ruido se evalúa solo en x (no por dimensión) para ahorrar coste
        noise_center = value_noise_nd(x, t, amp)
        f_center = rastrigin_nd(x) + noise_center
        x_mat = np.tile(x, (len(x), 1))
        np.fill_diagonal(x_mat, x_mat.diagonal() + eps)
        f_eps = np.array([rastrigin_nd(x_mat[i]) + value_noise_nd(x_mat[i], t, amp)
                          for i in range(len(x))])
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
# SA N-D
# ------------------------------------------------------------------ #

def simulated_annealing_nd(x0, n_steps=5000, T0=10.0, cooling=0.999,
                            step_size=0.3, search_range=5.12):
    x = np.array(x0, dtype=float)
    current_val = rastrigin_nd(x)
    best_x = x.copy()
    best_val = current_val
    best_history = [best_val]
    T = T0

    for _ in range(n_steps):
        x_new = np.clip(x + np.random.normal(0, step_size, size=len(x)),
                        -search_range, search_range)
        new_val = rastrigin_nd(x_new)
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            x, current_val = x_new, new_val
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
        T *= cooling
        best_history.append(best_val)

    return best_x, best_val, best_history


# ------------------------------------------------------------------ #
# CMA-ES N-D
# ------------------------------------------------------------------ #

def cmaes_nd(x0, n_steps=5000, search_range=5.12):
    opts = cma.CMAOptions()
    opts['bounds'] = [[-search_range] * len(x0), [search_range] * len(x0)]
    opts['maxfevals'] = n_steps
    opts['verbose'] = -9
    es = cma.CMAEvolutionStrategy(x0, 4.0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [rastrigin_nd(s) for s in solutions])
    return es.result.xbest, es.result.fbest


# ------------------------------------------------------------------ #
# Benchmark N-D
# ------------------------------------------------------------------ #

def run_benchmark(dims, n_trials=30, n_steps=5000, threshold=None):
    """threshold por defecto: 1.0 * dims (heurística razonable)"""
    if threshold is None:
        threshold = 1.0 * dims

    np.random.seed(42)
    p_ok, sa_ok, cma_ok = 0, 0, 0
    p_vals, sa_vals, cma_vals = [], [], []

    for _ in range(n_trials):
        x0 = np.random.uniform(-5.12, 5.12, size=dims)

        _, pval, _ = perlin_optimization_nd(x0, n_steps=n_steps)
        _, sval, _ = simulated_annealing_nd(x0, n_steps=n_steps)
        _, cval   = cmaes_nd(list(x0), n_steps=n_steps)

        p_vals.append(pval);   p_ok  += pval  < threshold
        sa_vals.append(sval);  sa_ok += sval  < threshold
        cma_vals.append(cval); cma_ok += cval < threshold

    print(f"\n=== Rastrigin {dims}D — {n_trials} pruebas | umbral f<{threshold} ===")
    print(f"Seismic Descent:{p_ok}/{n_trials}  media={np.mean(p_vals):.3f}  mediana={np.median(p_vals):.3f}  (evals={n_steps})")
    print(f"SA:             {sa_ok}/{n_trials}  media={np.mean(sa_vals):.3f}  mediana={np.median(sa_vals):.3f}  (evals={n_steps})")
    print(f"CMA-ES:         {cma_ok}/{n_trials}  media={np.mean(cma_vals):.3f}  mediana={np.median(cma_vals):.3f}  (evals={n_steps})")
    return p_vals, sa_vals, cma_vals


if __name__ == '__main__':
    run_benchmark(dims=5,  n_trials=30, n_steps=2000)
    run_benchmark(dims=10, n_trials=30, n_steps=2000)
    run_benchmark(dims=20, n_trials=10, n_steps=2000)
