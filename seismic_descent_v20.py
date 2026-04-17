"""
seismic_descent_v20.py — Seismic Swarm con Normalización de Gradiente y dt Cíclico.

MEJORA CRÍTICA:
1. Normalización L2 del gradiente objetivo para independizar el tamaño del paso
   de la escala de la función (eje Y).
2. dt Cíclico usando abs(sin(t)) para compensar la pérdida de magnitud natural
   y permitir fases de exploración larga y explotación fina en cada ciclo.
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
    """
    Gradiente RFF en espacio normalizado [-1, 1].
    """
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
# Seismic Swarm — Normalizado Bidireccional (Dominio y Gradiente)
# ------------------------------------------------------------------ #

def seismic_swarm(
    fn,                    # callable(X_real) -> array(N,)
    fn_grad,               # callable(X_real) -> array(N, D)
    x0_real,               # array(D,)
    bounds,                # array(D, 2)
    n_steps=2000,
    n_particles=10,
    dt_base=0.05,          # dt máximo base (ej. 5% de la caja)
    noise_amplitude=1.0,   # Amplitud relativa al gradiente normalizado (que tiene norma 1)
    noise_decay=1.0,
    n_cycles=10,
    dt_cycles_multiplier=10.0, # El dt oscilará X veces más rápido que el sismo
):
    """
    Seismic Swarm v20 con Gradient Normalization y dt cíclico desacoplado.
    """
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
        amp   = noise_amplitude * decay * np.sin(t * freq)

        # 1. Evaluar gradiente real
        X_real = center + X_norm * half_range
        f_grad_real = fn_grad(X_real)
        
        # 2. Mapear al espacio normalizado (Regla de la Cadena)
        f_grad_mapped = f_grad_real * half_range
        
        # 3. GRADIENT NORMALIZATION (Desacoplar de la magnitud de Y)
        # Extraemos solo la dirección. Si la norma es casi cero, evitamos dividir por cero.
        norms = np.linalg.norm(f_grad_mapped, axis=1, keepdims=True)
        f_grad_dir = np.where(norms > 1e-8, f_grad_mapped / norms, 0.0)
        
        # 4. Obtener gradiente de ruido (amplitud comparable a f_grad_dir)
        noise_grad = rff_noise_grad_vec(X_norm, t, amp)
        
        # 5. Combinar
        grad = f_grad_dir + noise_grad
        
        # 6. DT CÍCLICO DESACOPLADO
        # Multiplicamos 't' por el factor para que el ciclo sea mucho más rápido
        current_dt = dt_base * np.abs(np.sin(t * dt_cycles_multiplier))
        
        X_norm -= current_dt * grad
        np.clip(X_norm, -1.0, 1.0, out=X_norm)
        
        t += dt_noise

        # Evaluación y tracking
        X_real = center + X_norm * half_range
        real_vals = fn(X_real)
        step_best_idx = np.argmin(real_vals)
        if real_vals[step_best_idx] < best_val:
            best_val = real_vals[step_best_idx]
            best_x_real = X_real[step_best_idx].copy()
        
        best_per_step.append(float(best_val))

    return best_x_real, best_val, {'best_per_step': best_per_step}
