"""
analysis_convergence_v18.py — Estudio empírico de convergencia de Seismic Descent.

Genera:
  1. Curvas de convergencia budget -> best_val para múltiples dimensiones
  2. Análisis del techo dimensional D* 
  3. Gráficas PNG guardadas en results/
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from seismic_descent_v18 import seismic_swarm
from benchmark_functions import RASTRIGIN

def convergence_vs_budget(dims_list, budgets, n_trials=20):
    """
    Para cada dimensión y budget, ejecutar n_trials y recoger mediana de best_val.
    """
    results = {}  # {dims: {budget: median_val}}
    
    for dims in dims_list:
        results[dims] = {}
        fn_config = RASTRIGIN
        fn, fn_grad = fn_config['fn'], fn_config['grad']
        search_range = fn_config['search_range']
        
        for budget in budgets:
            n_particles = max(dims, 5)
            n_steps = budget // n_particles
            
            vals = []
            for trial in range(n_trials):
                np.random.seed(trial * 1000 + dims)
                x0 = np.random.uniform(-search_range, search_range, size=dims)
                _, bval, _ = seismic_swarm(
                    fn, fn_grad, x0,
                    n_steps=n_steps, n_particles=n_particles,
                    search_range=search_range,
                )
                vals.append(bval)
            
            results[dims][budget] = {
                'median': float(np.median(vals)),
                'mean': float(np.mean(vals)),
                'q25': float(np.percentile(vals, 25)),
                'q75': float(np.percentile(vals, 75)),
            }
            print(f"  {dims}D, budget={budget}: median={results[dims][budget]['median']:.4f}")
    
    return results


def plot_convergence(results, budgets, filename='results/convergence_curves.png'):
    """Graficar curvas de convergencia."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {5: '#2ecc71', 10: '#3498db', 20: '#e74c3c', 50: '#9b59b6'}
    
    for dims, budget_results in results.items():
        medians = [budget_results[b]['median'] for b in budgets]
        q25s = [budget_results[b]['q25'] for b in budgets]
        q75s = [budget_results[b]['q75'] for b in budgets]
        
        color = colors.get(dims, '#333')
        ax.plot(budgets, medians, 'o-', color=color, label=f'{dims}D', linewidth=2)
        ax.fill_between(budgets, q25s, q75s, alpha=0.15, color=color)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Evaluation Budget', fontsize=12)
    ax.set_ylabel('Best f(x) — Median over trials', fontsize=12)
    ax.set_title('Seismic Descent: Convergence vs Budget (Rastrigin)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)
    print(f"Guardado: {filename}")


if __name__ == '__main__':
    # Usando presupuestos y dimensiones reducidas para validación rápida
    dims_list = [5, 10, 20]
    budgets = [1000, 5000, 25000, 100000]
    
    print("=== Convergencia vs Budget ===")
    results = convergence_vs_budget(dims_list, budgets, n_trials=5)
    plot_convergence(results, budgets)
    
    md = "# Estudio Empírico de Convergencia v18\n\n"
    md += "## Curvas de convergencia vs Budget\n\n"
    md += "![Convergencia](../results/convergence_curves.png)\n\n"
    
    for dims in dims_list:
        md += f"### {dims}D\n"
        for b in budgets:
            md += f"- Budget {b}: Mediana = {results[dims][b]['median']:.4f}\n"
        md += "\n"
        
    os.makedirs('docs', exist_ok=True)
    with open('docs/findings_v18_convergence.md', 'w', encoding='utf-8') as f:
        f.write(md)