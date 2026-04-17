"""
seismic_descent_v18.py — Seismic Swarm Genérico (Function-Agnostic)

Basado en v14_cycles. No modifica ningún archivo existente.
Desacopla la función objetivo del optimizador para validación multi-función.

MEJORA CRÍTICA: Normalización del lengthscale de RFF según el search_range
para evitar problemas de altísima/bajosísima frecuencia en distintos dominios.
"""

import numpy as np

# ------------------------------------------------------------------ #
# Random Fourier Features — Vectorizado y Escalado
# ------------------------------------------------------------------ #

_R = 64
_rng = np.random.default_rng(seed=1)

_N_OCTAVES = 1
_Z = []  # Normal estándar generada 1 vez, escalada dinámicamente según el dominio
_PHIS   = _rng.uniform(0, 2 * np.pi, size=(_N_OCTAVES, _R))
_DRIFTS = _rng.uniform(0.1, 0.5,     size=(_N_OCTAVES, _R))

_D_MAX = 100
for o in range(_N_OCTAVES):
    # Base aleatoria z ~ N(0, 1) para las frecuencias
    z = _rng.normal(0, 1.0, size=(_R, _D_MAX))
    _Z.append(z)


def rff_noise_grad_vec(X, t, amplitude=15.0, octaves=_N_OCTAVES, search_range=5.12):
    """
    Gradiente RFF vectorizado para N partículas. Shape X: (N, D).
    El scale_factor ajusta la frecuencia del ruido al tamaño del dominio
    para garantizar que el 'terremoto' tenga la misma proporción física
    en Rastrigin (5.12) que en Schwefel (500.0).
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N, D = X.shape
    grad = np.zeros((N, D))
    amp = amplitude
    sqrt_2_R = np.sqrt(2.0 / _R)

    # Escalamos el lengthscale usando el rango de Rastrigin como baseline
    scale_factor = search_range / 5.12

    for o in range(octaves):
        lengthscale = scale_factor * 2.0 * (2.0 ** o)
        # omegas reales = z / lengthscale
        omegas = _Z[o][:, :D] / lengthscale
        phis   = _PHIS[o][:, None]
        drifts = _DRIFTS[o][:, None]
        
        projections = omegas @ X.T
        angles = projections + t * drifts + phis
        sines = np.sin(angles)
        
        grad_contrib = (omegas.T @ sines).T
        grad -= amp * sqrt_2_R * grad_contrib
        amp *= 0.5

    return grad


# ------------------------------------------------------------------ #
# Seismic Swarm — Genérico
# ------------------------------------------------------------------ #

def seismic_swarm(
    fn,                    # callable(X) -> array(N,) — función objetivo vectorizada
    fn_grad,               # callable(X) -> array(N, D) — gradiente analítico vectorizado
    x0,                    # array(D,) — punto inicial
    n_steps=2000,
    n_particles=10,
    dt=0.01,
    noise_amplitude=15.0,
    noise_decay=1.0,
    search_range=5.12,
    n_cycles=10,
):
    """
    Seismic Swarm genérico con RFF y gradiente analítico adaptativo.
    
    Parámetros
    ----------
    fn : callable
        Función objetivo. Debe aceptar X de shape (N, D) y devolver array (N,).
        También debe aceptar X de shape (D,) y devolver un escalar.
    fn_grad : callable
        Gradiente analítico de fn. Misma convención de shapes.
    x0 : array_like de shape (D,)
        Punto de inicio para la primera partícula del enjambre.
    n_steps : int
        Número de pasos de optimización.
    n_particles : int
        Número de partículas en el enjambre.
    dt : float
        Paso de gradiente descendente.
    noise_amplitude : float
        Amplitud base del campo RFF.
    noise_decay : float
        Factor de decay exponencial. Usar 1.0 para sin decay (recomendado).
    search_range : float
        El dominio es [-search_range, search_range]^D.
    n_cycles : int
        Número exacto de ciclos sísmicos completos a ejecutar.
    
    Retorna
    -------
    best_x : array(D,)
    best_val : float
    history : dict con 'best_per_step' (array de n_steps+1 floats)
    """
    D = len(x0)
    
    X = np.random.uniform(-search_range, search_range, size=(n_particles, D))
    X[0] = np.array(x0, dtype=float)
    
    t = 0.0
    dt_noise = (n_cycles * np.pi) / n_steps

    real_vals = fn(X)
    best_idx = np.argmin(real_vals)
    best_val = real_vals[best_idx]
    best_x = X[best_idx].copy()
    
    best_per_step = [float(best_val)]

    for step in range(n_steps):
        decay = noise_decay ** step
        freq  = 2.0 * decay
        amp   = noise_amplitude * decay * np.sin(t * freq)

        f_grad = fn_grad(X)
        noise_grad = rff_noise_grad_vec(X, t, amp, search_range=search_range)
        
        grad = f_grad + noise_grad
        X -= dt * grad
        np.clip(X, -search_range, search_range, out=X)
        t += dt_noise

        real_vals = fn(X)
        step_best_idx = np.argmin(real_vals)
        if real_vals[step_best_idx] < best_val:
            best_val = real_vals[step_best_idx]
            best_x = X[step_best_idx].copy()
        
        best_per_step.append(float(best_val))

    return best_x, best_val, {'best_per_step': best_per_step}
