"""
benchmark_suite_v18.py — Suite de benchmark multi-función para Seismic Descent.

Evalúa Seismic Swarm (v18) contra SA y CMA-ES en todas las funciones
del registro benchmark_functions.py.

USO:
    python benchmark_suite_v18.py                          # todas las funciones, preset low
    python benchmark_suite_v18.py --functions rastrigin schwefel  # solo esas
    python benchmark_suite_v18.py --preset high            # presupuesto alto
    python benchmark_suite_v18.py --dims 5 10              # solo esas dimensiones
"""

import argparse
import time
import json
import os
import numpy as np
import cma

from seismic_descent_v18 import seismic_swarm
from benchmark_functions import ALL_FUNCTIONS


def cmaes_run(fn, x0, eval_budget, search_range):
    """Wrapper genérico de CMA-ES."""
    opts = cma.CMAOptions()
    opts['bounds'] = [[-search_range] * len(x0), [search_range] * len(x0)]
    opts['maxfevals'] = eval_budget
    opts['verbose'] = -9
    # sigma0 proporcional al rango de búsqueda
    sigma0 = search_range * 0.8
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [float(fn(np.array(s))) for s in solutions])
    return es.result.xbest, es.result.fbest


def sa_generic(fn, x0, n_steps=5000, T0=10.0, cooling=0.999,
               step_size=0.3, search_range=5.12):
    """Simulated Annealing genérico (no hardcodeado a Rastrigin)."""
    x = np.array(x0, dtype=float)
    current_val = float(fn(x))
    best_x = x.copy()
    best_val = current_val
    T = T0

    for _ in range(n_steps):
        x_new = np.clip(
            x + np.random.normal(0, step_size, size=len(x)),
            -search_range, search_range
        )
        new_val = float(fn(x_new))
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            x, current_val = x_new, new_val
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
        T *= cooling

    return best_x, best_val


def run_single_benchmark(func_config, dims, n_trials, eval_budget_base):
    """Ejecuta benchmark para 1 función x 1 dimensión."""
    fn = func_config['fn']
    fn_grad = func_config['grad']
    search_range = func_config['search_range']
    fname = func_config['name']
    
    # Heurísticas de amplitud
    if fname == 'Schwefel':
        noise_amp = 1500.0
    elif fname == 'Griewank':
        noise_amp = 5.0
    elif fname == 'Rosenbrock':
        noise_amp = 10.0
    else:
        noise_amp = 15.0

    eval_budget = eval_budget_base * (dims + 1)
    n_particles = max(dims, 5)  # mínimo 5 partículas
    seismic_steps = eval_budget // n_particles
    sa_steps = eval_budget
    
    # Threshold: usar un porcentaje del valor típico aleatorio
    threshold = dims * 1.0  # heurístico, se puede refinar
    if fname == 'Schwefel':
        threshold = dims * 50.0

    results = {'seismic': [], 'sa': [], 'cmaes': []}
    times = {'seismic': 0.0, 'sa': 0.0, 'cmaes': 0.0}

    np.random.seed(42)

    print(f"Running {fname} {dims}D ({n_trials} trials, budget={eval_budget})...")

    for trial in range(n_trials):
        x0 = np.random.uniform(-search_range, search_range, size=dims)

        # --- Seismic ---
        t0 = time.time()
        _, sval, _ = seismic_swarm(
            fn, fn_grad, x0,
            n_steps=seismic_steps,
            n_particles=n_particles,
            search_range=search_range,
            noise_amplitude=noise_amp
        )
        times['seismic'] += time.time() - t0
        results['seismic'].append(float(sval))

        # --- SA ---
        t0 = time.time()
        # step size proportional to search range
        sa_step_size = search_range * 0.05
        _, sa_val = sa_generic(fn, x0, n_steps=sa_steps, search_range=search_range, step_size=sa_step_size)
        times['sa'] += time.time() - t0
        results['sa'].append(float(sa_val))

        # --- CMA-ES ---
        t0 = time.time()
        try:
            _, cma_val = cmaes_run(fn, x0, eval_budget, search_range)
        except Exception as e:
            print(f"CMA-ES failed on trial {trial}: {e}")
            cma_val = float('inf')
        times['cmaes'] += time.time() - t0
        results['cmaes'].append(float(cma_val))

    res_dict = {
        'function': fname,
        'dims': dims,
        'n_trials': n_trials,
        'eval_budget': eval_budget,
        'results': {
            algo: {
                'values': vals,
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'successes': int(sum(1 for v in vals if v < threshold)),
                'time_s': float(times[algo]),
            }
            for algo, vals in results.items()
        },
        'threshold': threshold,
    }
    
    print(f"  Seismic: {res_dict['results']['seismic']['successes']}/{n_trials} success, median={res_dict['results']['seismic']['median']:.2f}, time={times['seismic']:.2f}s")
    print(f"  SA:      {res_dict['results']['sa']['successes']}/{n_trials} success, median={res_dict['results']['sa']['median']:.2f}, time={times['sa']:.2f}s")
    print(f"  CMA-ES:  {res_dict['results']['cmaes']['successes']}/{n_trials} success, median={res_dict['results']['cmaes']['median']:.2f}, time={times['cmaes']:.2f}s")

    return res_dict

def generate_markdown(all_results, output_file):
    md = "# Benchmark Suite v18 Results\n\n"
    for res in all_results:
        md += f"## {res['function']} {res['dims']}D — {res['n_trials']} trials | budget={res['eval_budget']} evals (threshold: {res['threshold']})\n\n"
        md += "| Algoritmo | Éxitos | Media | Mediana | Std | Tiempo |\n"
        md += "|---|---|---|---|---|---|\n"
        
        for algo in ['seismic', 'sa', 'cmaes']:
            data = res['results'][algo]
            name = "Seismic Swarm" if algo == 'seismic' else ("SA" if algo == 'sa' else "CMA-ES")
            md += f"| {name} | {data['successes']}/{res['n_trials']} | {data['mean']:.2f} | {data['median']:.2f} | {data['std']:.2f} | {data['time_s']:.1f}s |\n"
        md += "\n"
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Markdown written to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--functions', nargs='+', default=list(ALL_FUNCTIONS.keys()))
    parser.add_argument('--dims', nargs='+', type=int, default=[5, 10, 20])
    parser.add_argument('--preset', choices=['low', 'medium', 'high'], default='low')
    args = parser.parse_args()

    presets = {
        'low': {'trials': 10, 'budget_base': 500},
        'medium': {'trials': 30, 'budget_base': 1000},
        'high': {'trials': 30, 'budget_base': 5000},
    }
    
    config = presets[args.preset]
    
    all_results = []
    
    os.makedirs('results', exist_ok=True)
    
    for fname in args.functions:
        if fname not in ALL_FUNCTIONS:
            print(f"Unknown function: {fname}")
            continue
            
        fconfig = ALL_FUNCTIONS[fname]
        for d in args.dims:
            res = run_single_benchmark(fconfig, d, config['trials'], config['budget_base'])
            all_results.append(res)
            
    with open('results/benchmark_suite_v18.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(all_results, indent=2))
        
    generate_markdown(all_results, 'docs/findings_v18_suite.md')