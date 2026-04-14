"""
benchmark_budgets.py - Análisis a gran escala de tiempos y presupuestos frente a CMA-ES
"""

import argparse
import time
import numpy as np
import cma
from perlin_opt_nd import simulated_annealing_nd

# Importamos las funciones necesarias del script v14 (Swarm puro con ciclos temporalmente precisos)
from perlin_opt_nd_grf_analytic_v14_cycles import (
    seismic_swarm_rff_analytic,
    rastrigin_nd_vec
)

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

def run_benchmark(dims, n_trials=30, threshold=None, eval_budget_base=500):
    if threshold is None:
        threshold = 1.0 * dims

    eval_budget = eval_budget_base * (dims + 1)
    n_particles = dims
    
    seismic_steps = eval_budget // n_particles
    sa_steps = eval_budget

    print(f"\n--- Dimensiones: {dims}D | Presupuesto: {eval_budget} eval | Enjambre: {n_particles} partículas, {seismic_steps} pasos temporales ---")

    np.random.seed(42)
    p_ok, sa_ok, cma_ok = 0, 0, 0
    p_vals, sa_vals, cma_vals = [], [], []

    t_seismic = 0
    t_sa = 0
    t_cma = 0

    print(">>> Simulando Swarm [Iniciando...]")
    for i in range(n_trials):
        x0 = np.random.uniform(-5.12, 5.12, size=dims)
        t0 = time.time()
        _, pval, _ = seismic_swarm_rff_analytic(x0, n_steps=seismic_steps, n_particles=n_particles)
        t_seismic += time.time() - t0
        p_vals.append(pval)
        p_ok += pval < threshold
        if (i+1) % max(1, n_trials//3) == 0:
            print(f"    Swarm {i+1}/{n_trials} completados...")
    
    print(">>> Simulando SA [Iniciando...]")
    for i in range(n_trials):
        x0 = np.random.uniform(-5.12, 5.12, size=dims)
        t0 = time.time()
        _, sval, _ = simulated_annealing_nd(x0, n_steps=sa_steps)
        t_sa += time.time() - t0
        sa_vals.append(sval)
        sa_ok += sval < threshold
        if (i+1) % max(1, n_trials//3) == 0:
            print(f"    SA {i+1}/{n_trials} completados...")

    print(">>> Simulando CMA-ES [Iniciando...]")
    for i in range(n_trials):
        x0 = np.random.uniform(-5.12, 5.12, size=dims)
        t0 = time.time()
        _, cval = cmaes_nd(list(x0), eval_budget=eval_budget)
        t_cma += time.time() - t0
        cma_vals.append(cval)
        cma_ok += cval < threshold
        if (i+1) % max(1, n_trials//3) == 0:
            print(f"    CMA-ES {i+1}/{n_trials} completados...")

    print(f"\n=== Resultados Rastrigin {dims}D ({n_trials} pruebas) | umbral f<{threshold} ===")
    print(f"Seismic+RFF Swarm: {p_ok:2}/{n_trials}  media={np.mean(p_vals):7.3f}  mediana={np.median(p_vals):7.3f}  t={t_seismic:.2f}s")
    print(f"SA:                {sa_ok:2}/{n_trials}  media={np.mean(sa_vals):7.3f}  mediana={np.median(sa_vals):7.3f}  t={t_sa:.2f}s")
    print(f"CMA-ES:            {cma_ok:2}/{n_trials}  media={np.mean(cma_vals):7.3f}  mediana={np.median(cma_vals):7.3f}  t={t_cma:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', choices=['low', 'med', 'high'], default='low', help="Preset de presupuesto")
    parser.add_argument('--custom_base', type=int, default=None, help="EVAL_BUDGET_BASE manual")
    parser.add_argument('--dims', type=str, default="5,10,20,50", help="Dimensiones separadas por coma")
    args = parser.parse_args()

    presets = {'low': 500, 'med': 2500, 'high': 10000}
    base = presets[args.preset] if args.custom_base is None else args.custom_base
    dims_to_run = [int(d) for d in args.dims.split(',')]

    print(f"======= INICIANDO BENCHMARK MULTI-PRESUPUESTO =======")
    print(f">>> Preset activado: {args.preset.upper()}")
    print(f">>> EVAL_BUDGET_BASE = {base}")
    print(f">>> Presupuesto total esperado paramétrico (ej. en 10D): {base * 11} evaluaciones\n")
    
    ratio = base / 500.0
    estimated_s = 110.0 * ratio
    print(f"[ESTIMACIÓN TIEMPO] Si evaluamos las 4 dimensiones completas (5D a 50D): ~ {estimated_s/60.0:.1f} minutos.")
    print("=====================================================\n")

    trials_map = {5: 30, 10: 30, 20: 10, 50: 5}

    for d in dims_to_run:
        trials = trials_map.get(d, 10)
        run_benchmark(dims=d, n_trials=trials, eval_budget_base=base)
