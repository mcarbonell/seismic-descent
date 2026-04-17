"""
seismic_descent_v22.py — Función Oscilante y Ruido Constante Normalizado

MEJORA CRÍTICA (v22):
1. El gradiente del ruido se normaliza a magnitud 1.0 (igual que el gradiente de la función).
2. El ruido mantiene una fuerza constante (no oscila con el tiempo).
3. La "barrera" de la función objetivo es la que oscila: se multiplica por (sin(t) + 1) / 2.
   Esto hace que la función original "aparezca y desaparezca", atrapando y soltando a la partícula.
"""

import numpy as np

# ------------------------------------------------------------------ #
# Random Fourier Features — Dominio Normalizado [-1, 1]
# ------------------------------------------------------------------ #

_R = 64
_rng = np.random.default_rng(seed=1)

_N_OCTAVES = 1
_Z = [] 
_PHIS   = _rng.uniform(0, 2 * np.pi, size=(_N_OCTAVES, _R))
_DRIFTS = _rng.uniform(0.1, 0.5,     size=(_N_OCTAVES, _R))

_D_MAX = 100
for o in range(_N_OCTAVES):
    z = _rng.normal(0, 1.0, size=(_R, _D_MAX))
    _Z.append(z)

def rff_noise_grad_vec(X_norm, t, amplitude=1.0, octaves=_N_OCTAVES):
    if X_norm.ndim == 1:
        X_norm = X_norm.reshape(1, -1)
    N, D = X_norm.shape
    grad = np.zeros((N, D))
    amp = amplitude
    sqrt_2_R = np.sqrt(2.0 / _R)
    BASE_LENGTHSCALE = 0.4

    for o in range(octaves):
        lengthscale = BASE_LENGTHSCALE * (2.0 ** o)
        omegas = _Z[o][:, :D] / lengthscale
        phis   = _PHIS[o][:, None]
        drifts = _DRIFTS[o][:, None]
        
        projections = omegas @ X_norm.T
        angles = projections + t * drifts + phis
        sines = np.sin(angles)
        
        grad_contrib = (omegas.T @ sines).T
        grad -= amp * sqrt_2_R * grad_contrib
        amp *= 0.5

    return grad


# ------------------------------------------------------------------ #
# Seismic Swarm — Función Oscilante
# ------------------------------------------------------------------ #

def seismic_swarm(
    fn,                    
    fn_grad,               
    x0_real,               
    bounds,                
    n_steps=2000,
    n_particles=10,
    dt_base=0.1,          
    noise_amplitude=1.0,   
    noise_decay=1.0,
    n_cycles=10,
):
    bounds = np.array(bounds)
    D = len(x0_real)
    
    center = (bounds[:, 1] + bounds[:, 0]) / 2.0
    half_range = (bounds[:, 1] - bounds[:, 0]) / 2.0
    
    x0_norm = (np.array(x0_real) - center) / half_range
    X_norm = np.random.uniform(-1.0, 1.0, size=(n_particles, D))
    X_norm[0] = np.clip(x0_norm, -1.0, 1.0)
    
    t = 0.0
    dt_noise = (n_cycles * np.pi) / n_steps

    X_real = center + X_norm * half_range
    real_vals = fn(X_real)
    best_idx = np.argmin(real_vals)
    best_val = real_vals[best_idx]
    best_x_real = X_real[best_idx].copy()
    
    best_per_step = [float(best_val)]

    for step in range(n_steps):
        decay = noise_decay ** step
        freq  = 2.0 * decay

        # 1. Gradiente de la Función Objetivo (Normalizado a [0, 1])
        X_real = center + X_norm * half_range
        f_grad_real = fn_grad(X_real)
        f_grad_mapped = f_grad_real * half_range
        
        norms_f = np.linalg.norm(f_grad_mapped, axis=1, keepdims=True)
        f_grad_dir = np.where(norms_f > 1e-8, f_grad_mapped / norms_f, 0.0)
        
        # 2. Gradiente del Ruido (Normalizado a [0, 1])
        noise_grad = rff_noise_grad_vec(X_norm, t, amplitude=1.0)
        norms_noise = np.linalg.norm(noise_grad, axis=1, keepdims=True)
        noise_grad_dir = np.where(norms_noise > 1e-8, noise_grad / norms_noise, 0.0)
        
        # 3. La MAGIA de la v22: La Función Original Aparece y Desaparece
        # (sin(t * freq) + 1) / 2 genera una onda suave entre 0.0 y 1.0
        obj_weight = (np.sin(t * freq) + 1.0) / 2.0
        
        # El ruido es constante en el tiempo (multiplicado por su peso estático)
        noise_weight = noise_amplitude * decay
        
        # 4. Combinamos: Ruido constante + Función intermitente
        grad = f_grad_dir * obj_weight + noise_grad_dir * noise_weight
        
        # 5. Aplicar paso (dt se mantiene constante)
        X_norm -= dt_base * grad
        np.clip(X_norm, -1.0, 1.0, out=X_norm)
        
        t += dt_noise

        # Evaluación
        X_real = center + X_norm * half_range
        real_vals = fn(X_real)
        step_best_idx = np.argmin(real_vals)
        if real_vals[step_best_idx] < best_val:
            best_val = real_vals[step_best_idx]
            best_x_real = X_real[step_best_idx].copy()
        
        best_per_step.append(float(best_val))

    return best_x_real, best_val, {'best_per_step': best_per_step}
