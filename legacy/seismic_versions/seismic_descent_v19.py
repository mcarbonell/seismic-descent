"""
seismic_descent_v19.py — Seismic Swarm con Dominio Isotrópico Normalizado.

MEJORA CRÍTICA: El optimizador trabaja internamente en el rango [-1, 1]^D.
Mapea automáticamente las coordenadas y gradientes desde/hacia el espacio real
definido por 'bounds'. Esto permite un dt y una escala de ruido universales.
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
    # Base aleatoria z ~ N(0, 1) para las frecuencias
    z = _rng.normal(0, 1.0, size=(_R, _D_MAX))
    _Z.append(z)


def rff_noise_grad_vec(X_norm, t, amplitude=15.0, octaves=_N_OCTAVES):
    """
    Gradiente RFF en espacio normalizado [-1, 1].
    Lengthscale fijo para este dominio.
    """
    if X_norm.ndim == 1:
        X_norm = X_norm.reshape(1, -1)
    N, D = X_norm.shape
    grad = np.zeros((N, D))
    amp = amplitude
    sqrt_2_R = np.sqrt(2.0 / _R)

    # En [-5.12, 5.12] el range es 10.24 y el lengthscale era 2.0.
    # En [-1.0, 1.0] el range es 2.0, por regla de 3: 2.0 * (2.0 / 10.24) approx 0.39
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
# Seismic Swarm — Normalizado
# ------------------------------------------------------------------ #

def seismic_swarm(
    fn,                    # callable(X_real) -> array(N,)
    fn_grad,               # callable(X_real) -> array(N, D)
    x0_real,               # array(D,) punto inicial en espacio real
    bounds,                # array(D, 2) con (min, max) por dimensión
    n_steps=2000,
    n_particles=10,
    dt=0.01,
    noise_amplitude=15.0,
    noise_decay=1.0,
    n_cycles=10,
):
    """
    Seismic Swarm con normalización interna a [-1, 1].
    """
    bounds = np.array(bounds)
    D = len(x0_real)
    
    # Parámetros de transformación
    # X_real = center + X_norm * half_range
    # Grad_norm = Grad_real * half_range
    center = (bounds[:, 1] + bounds[:, 0]) / 2.0
    half_range = (bounds[:, 1] - bounds[:, 0]) / 2.0
    
    # Inicialización en espacio normalizado
    # Mapeamos x0_real a normalized
    x0_norm = (np.array(x0_real) - center) / half_range
    
    X_norm = np.random.uniform(-1.0, 1.0, size=(n_particles, D))
    X_norm[0] = np.clip(x0_norm, -1.0, 1.0)
    
    t = 0.0
    dt_noise = (n_cycles * np.pi) / n_steps

    # Evaluación inicial
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

        # 1. Mapear a espacio real para evaluar gradiente
        X_real = center + X_norm * half_range
        
        # 2. Obtener gradiente real
        f_grad_real = fn_grad(X_real)
        
        # 3. Mapear gradiente real a normalizado (Regla de la cadena)
        # dF/dX_norm = dF/dX_real * dX_real/dX_norm = f_grad_real * half_range
        f_grad_norm = f_grad_real * half_range
        
        # 4. Obtener gradiente de ruido (ya en espacio normalizado)
        noise_grad = rff_noise_grad_vec(X_norm, t, amp)
        
        # 5. Update en espacio normalizado
        grad = f_grad_norm + noise_grad
        X_norm -= dt * grad
        
        # Clip en espacio normalizado
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
