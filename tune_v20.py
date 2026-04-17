"""
tune_v20.py — Script de ajuste de hiperparámetros para Seismic Descent v20.

Busca la mejor combinación de dt_base, noise_amplitude y dt_cycles_multiplier.
"""

import numpy as np
import time
from seismic_descent_v20 import seismic_swarm
from benchmark_functions import ALL_FUNCTIONS

def score_hyperparams(dt_base, noise_amp, multiplier, n_trials=3):
    """
    Evalúa una combinación de hiperparámetros en 5D para todas las funciones.
    Retorna un score global (menor es mejor).
    El score se normaliza usando el valor mínimo global de la función.
    """
    total_score = 0.0
    dims = 5
    eval_budget = 3000
    n_particles = 5
    seismic_steps = eval_budget // n_particles
    
    # Para poder sumar peras con manzanas (Rastrigin vs Schwefel),
    # calcularemos el % de éxito para llegar cerca del mínimo, o normalizaremos el valor final.
    # Vamos a usar un enfoque de "Rango Normalizado": log10(val + 1e-8) puede ser útil.
    
    for fname, fconfig in ALL_FUNCTIONS.items():
        fn = fconfig['fn']
        fn_grad = fconfig['grad']
        search_range = fconfig['search_range']
        global_min = fconfig['global_min_val']
        
        bounds = np.zeros((dims, 2))
        bounds[:, 0] = -search_range
        bounds[:, 1] = search_range
        
        func_score = 0.0
        np.random.seed(42) # Semilla fija por función para comparar justo
        
        for _ in range(n_trials):
            x0 = np.random.uniform(-search_range, search_range, size=dims)
            _, val, _ = seismic_swarm(
                fn, fn_grad, x0, bounds=bounds,
                n_steps=seismic_steps, n_particles=n_particles,
                dt_base=dt_base, noise_amplitude=noise_amp,
                dt_cycles_multiplier=multiplier
            )
            
            # Usamos log error para que grandes fallos en Schwefel no eclipsen pequeños fallos en Rastrigin
            error = max(val - global_min, 0.0)
            func_score += np.log10(error + 1.0)
            
        total_score += func_score / n_trials
        
    return total_score

if __name__ == "__main__":
    dt_bases = [0.05, 0.1, 0.2]
    noise_amps = [0.5, 1.0, 1.5]
    multipliers = [5.0, 10.0, 50.0, 100.0]
    
    print("Iniciando Grid Search para V20 (dt_base, noise_amp, dt_multiplier)...")
    print("El score es sum(log10(error + 1)) promedio por función. Menor es mejor.")
    print("-" * 60)
    
    best_score = float('inf')
    best_params = None
    
    results = []
    
    for dt in dt_bases:
        for amp in noise_amps:
            for mult in multipliers:
                score = score_hyperparams(dt, amp, mult)
                print(f"dt_base={dt:.2f}, amp={amp:.1f}, mult={mult:5.1f} | Score: {score:.4f}")
                results.append((score, dt, amp, mult))
                
                if score < best_score:
                    best_score = score
                    best_params = (dt, amp, mult)
                    
    print("-" * 60)
    print(f"MEJOR COMBINACIÓN: dt_base={best_params[0]}, amp={best_params[1]}, mult={best_params[2]}")
    print(f"Mejor Score: {best_score:.4f}")
    
    # Top 3
    results.sort(key=lambda x: x[0])
    print("\nTop 3 configuraciones:")
    for i in range(3):
        s, d, a, m = results[i]
        print(f"{i+1}. dt_base={d:.2f}, amp={a:.1f}, mult={m:5.1f} (Score: {s:.4f})")
