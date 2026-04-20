import numpy as np
import time
import sys
import os

# Asegurar importación
sys.path.append(os.path.join(os.getcwd(), 'seismic-descent'))
from seismic_descent_v22 import seismic_swarm
from scratch.seismic_v23_vector_wave import SeismicOptimizerV23

def rastrigin(x):
    # Soporta tanto un vector (dim,) como un enjambre (N, dim)
    if x.ndim == 1:
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    else:
        return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

def rastrigin_grad(x):
    if x.ndim == 1:
        return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    else:
        return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

def run_v22(dim, steps, seed=42):
    bounds = [[-5.12, 5.12]] * dim
    x0 = np.zeros(dim)
    
    start = time.perf_counter()
    # Usamos 1 partícula para comparar con V23 que es single-point
    best_x, best_val, info = seismic_swarm(
        rastrigin, rastrigin_grad, x0, bounds, 
        n_steps=steps, n_particles=1, noise_amplitude=1.0
    )
    end = time.perf_counter()
    return best_val, (end - start)

def run_v23(dim, steps, seed=42):
    x = np.zeros(dim)
    opt = SeismicOptimizerV23(dim=dim, total_steps=steps, seed=seed)
    
    start = time.perf_counter()
    for _ in range(steps):
        x, _ = opt.step(rastrigin, x)
    end = time.perf_counter()
    return rastrigin(x), (end - start)

if __name__ == "__main__":
    dims = [100, 1000]
    steps = 500
    
    print(f"{'Dim':<6} | {'Algo':<8} | {'Loss':<12} | {'Time(s)':<10} | {'Speedup'}")
    print("-" * 60)
    
    for dim in dims:
        loss22, t22 = run_v22(dim, steps)
        loss23, t23 = run_v23(dim, steps)
        
        speedup = t22 / t23 if t23 > 0 else 0
        
        print(f"{dim:<6} | {'V22':<8} | {loss22:<12.4f} | {t22:<10.4f} | ---")
        print(f"{'':<6} | {'V23':<8} | {loss23:<12.4f} | {t23:<10.4f} | {speedup:.2f}x")
        print("-" * 60)
