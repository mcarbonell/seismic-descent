"""
seismic_descent_v19.py — Ergodic Noise Morphing (Discreto-Continuo)

Basado en v18 genérico. Reemplaza el ruido dependiente del tiempo estricto 
(que causaba alta frecuencia destructiva en altos budgets) por una 
interpolación de varianza constante entre dos paisajes estáticos aleatorios (A y B).

Al realizar un *morphing* suave entre dos campos RFF coherentes, aseguramos
la propiedad de **Ergodicidad**: el sistema fluye orgánicamente por todo 
el espacio sin vibraciones espasmódicas.
"""

import numpy as np

# ------------------------------------------------------------------ #
# Ergodic RFF Field
# ------------------------------------------------------------------ #

class RFFField:
    """Campo de Ruido Gaussiano (GRF) estático y coherente."""
    def __init__(self, octaves=4, seed=1, search_range=5.12, R=64):
        self.R = R
        self.octaves = octaves
        rng = np.random.default_rng(seed=seed)
        
        self.omegas = []
        self.phis = []
        
        scale_factor = search_range / 5.12
        for o in range(octaves):
            lengthscale = scale_factor * 2.0 * (2.0 ** o)
            # z ~ N(0, 1), ω = z / lengthscale
            omegas = rng.normal(0, 1.0, size=(R, 100)) / lengthscale
            phis = rng.uniform(0, 2 * np.pi, size=(R, 1))
            self.omegas.append(omegas)
            self.phis.append(phis)
            
    def grad(self, X, amplitude):
        """Devuelve el gradiente del campo estático para X (N, D)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N, D = X.shape
        grad = np.zeros((N, D))
        amp = amplitude
        sqrt_2_R = np.sqrt(2.0 / self.R)
        
        for o in range(self.octaves):
            omegas = self.omegas[o][:, :D]  # (R, D)
            phis = self.phis[o]             # (R, 1)
            
            # Ángulo estático (no hay drift de tiempo aquí)
            angles = omegas @ X.T + phis    # (R, N)
            sines = np.sin(angles)          # (R, N)
            
            grad_contrib = (omegas.T @ sines).T # (N, D)
            grad -= amp * sqrt_2_R * grad_contrib
            amp *= 0.5
            
        return grad

# ------------------------------------------------------------------ #
# Seismic Swarm — Ergodic Morphing
# ------------------------------------------------------------------ #

def seismic_swarm(
    fn,                    
    fn_grad,               
    x0,                    
    n_steps=2000,
    n_particles=10,
    dt=0.01,
    noise_amplitude=15.0,
    search_range=5.12,
    morph_steps=100,       # Pasos para transicionar del Paisaje A al B
    octaves=4
):
    """
    Seismic Swarm v19: Ergodic Morphing.
    La partícula navega un paisaje que muta suavemente, permitiendo una 
    exploración fluida del dominio guiada por la topología RFF.
    """
    D = len(x0)
    
    X = np.random.uniform(-search_range, search_range, size=(n_particles, D))
    X[0] = np.array(x0, dtype=float)
    
    real_vals = fn(X)
    best_idx = np.argmin(real_vals)
    best_val = real_vals[best_idx]
    best_x = X[best_idx].copy()
    
    best_per_step = [float(best_val)]
    
    # Inicializar paisajes base
    seed_A = 1
    seed_B = 2
    field_A = RFFField(octaves, seed_A, search_range)
    field_B = RFFField(octaves, seed_B, search_range)
    
    step = 0
    for i in range(n_steps):
        # Interpolación de Varianza Constante [0, 1]
        u = (step % morph_steps) / morph_steps
        weight_A = np.cos(u * np.pi / 2)
        weight_B = np.sin(u * np.pi / 2)
        
        # En v19, el ruido no oscila cíclicamente en amplitud, 
        # sino que decae suavemente hacia el final para permitir asentar (refinamiento).
        # decay_factor va de 1.0 a 0.0
        decay_factor = 1.0 - (i / n_steps)
        amp = noise_amplitude * decay_factor
        
        grad_A = field_A.grad(X, amp)
        grad_B = field_B.grad(X, amp)
        
        noise_grad = weight_A * grad_A + weight_B * grad_B
        f_grad = fn_grad(X)
        
        # Descenso por la suma de ambos gradientes
        X -= dt * (f_grad + noise_grad)
        np.clip(X, -search_range, search_range, out=X)
        
        # Evaluar
        real_vals = fn(X)
        step_best_idx = np.argmin(real_vals)
        if real_vals[step_best_idx] < best_val:
            best_val = real_vals[step_best_idx]
            best_x = X[step_best_idx].copy()
            
        best_per_step.append(float(best_val))
        
        # Avanzar el morphing
        step += 1
        if step % morph_steps == 0:
            seed_A = seed_B
            seed_B += 1
            field_A = field_B
            field_B = RFFField(octaves, seed_B, search_range)
            
    return best_x, best_val, {'best_per_step': best_per_step}
