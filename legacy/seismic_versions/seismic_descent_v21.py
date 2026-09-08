"""
seismic_descent_v21.py — Seismic Swarm con Paisaje Subrogado Dinámico y Aceptación Discreta.

MEJORA CRÍTICA (v21):
1. Registro histórico de f_min y f_max para normalizar la función objetivo a [0, 1].
2. El campo RFF ahora computa su valor escalar (además de su gradiente).
3. Función Subrogada: S(X, t) = f_norm(X) + RFF_val(X, t).
4. Criterio de Aceptación: Un paso solo se acepta si mejora el valor de la función subrogada.
"""

import numpy as np

# ------------------------------------------------------------------ #
# Random Fourier Features — Valor y Gradiente
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

def rff_noise_val_and_grad(X_norm, t, amplitude=1.0, octaves=_N_OCTAVES):
    """
    Retorna el valor escalar y el gradiente del campo de ruido RFF.
    Si el gradiente se basa en -sin(ang)*w, el valor escalar es cos(ang).
    """
    if X_norm.ndim == 1:
        X_norm = X_norm.reshape(1, -1)
    N, D = X_norm.shape
    
    grad = np.zeros((N, D))
    val = np.zeros(N)
    
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
        
        # 1. Gradiente: derivamos cos -> -sin
        sines = np.sin(angles)
        grad_contrib = (omegas.T @ sines).T
        grad -= amp * sqrt_2_R * grad_contrib
        
        # 2. Valor: la primitiva de -sin es cos
        cosines = np.cos(angles)
        val_contrib = np.sum(cosines, axis=0) # Suma a través de las R features
        val += amp * sqrt_2_R * val_contrib
        
        amp *= 0.5

    return val, grad

# ------------------------------------------------------------------ #
# Seismic Swarm — Subrogado Dinámico
# ------------------------------------------------------------------ #

def seismic_swarm(
    fn,                    
    fn_grad,               
    x0_real,               
    bounds,                
    n_steps=2000,
    n_particles=10,
    dt_base=0.2,          
    noise_amplitude=0.5,   
    noise_decay=1.0,
    n_cycles=10,
    dt_cycles_multiplier=5.0,
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

    # Evaluación inicial
    X_real = center + X_norm * half_range
    real_vals = fn(X_real)
    f_grad_real = fn_grad(X_real)
    
    # Registro histórico de f_min y f_max
    global_f_min = np.min(real_vals)
    global_f_max = np.max(real_vals)
    
    best_idx = np.argmin(real_vals)
    best_val = real_vals[best_idx]
    best_x_real = X_real[best_idx].copy()
    
    best_per_step = [float(best_val)]

    for step in range(n_steps):
        decay = noise_decay ** step
        freq  = 2.0 * decay
        amp   = noise_amplitude * decay * np.sin(t * freq)

        f_range = max(global_f_max - global_f_min, 1e-8)
        
        # 1. Gradiente de la función normalizada
        # d/dX_norm [ (f(X_real) - min) / range ] = (df/dX_real * half_range) / range
        f_grad_norm = (f_grad_real * half_range) / f_range
        
        # 2. Obtener ruido (valor y gradiente) en la posición actual
        noise_val, noise_grad = rff_noise_val_and_grad(X_norm, t, amp)
        
        # 3. Gradiente de la función subrogada
        S_grad = f_grad_norm + noise_grad
        
        # 4. Proponer nuevo paso
        current_dt = dt_base * np.abs(np.sin(t * dt_cycles_multiplier))
        X_norm_new = X_norm - current_dt * S_grad
        np.clip(X_norm_new, -1.0, 1.0, out=X_norm_new)
        
        # 5. Evaluar el nuevo paso en el mundo real
        X_real_new = center + X_norm_new * half_range
        new_real_vals = fn(X_real_new)
        new_f_grad_real = fn_grad(X_real_new)
        
        # 6. Actualizar registro histórico con lo que acabamos de ver
        global_f_min = min(global_f_min, np.min(new_real_vals))
        global_f_max = max(global_f_max, np.max(new_real_vals))
        f_range = max(global_f_max - global_f_min, 1e-8)
        
        # 7. CRITERIO DE ACEPTACIÓN
        # Calculamos el valor subrogado en el punto VIEJO con el t ACTUAL
        S_val_old = (real_vals - global_f_min) / f_range + noise_val
        
        # Calculamos el valor subrogado en el punto NUEVO con el t ACTUAL
        noise_val_new, _ = rff_noise_val_and_grad(X_norm_new, t, amp)
        S_val_new = (new_real_vals - global_f_min) / f_range + noise_val_new
        
        # Solo aceptamos si el paisaje subrogado MEJORA (es menor)
        accept = S_val_new < S_val_old
        
        # 8. Aplicar los movimientos aceptados
        X_norm = np.where(accept[:, None], X_norm_new, X_norm)
        real_vals = np.where(accept, new_real_vals, real_vals)
        f_grad_real = np.where(accept[:, None], new_f_grad_real, f_grad_real)
        
        t += dt_noise

        # Tracking del mejor real
        step_best_idx = np.argmin(real_vals)
        if real_vals[step_best_idx] < best_val:
            best_val = real_vals[step_best_idx]
            best_x_real = (center + X_norm[step_best_idx] * half_range).copy()
        
        best_per_step.append(float(best_val))

    return best_x_real, best_val, {'best_per_step': best_per_step}
