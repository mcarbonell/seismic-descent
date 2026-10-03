"""Verificación 5: ABLACIÓN CLAVE — ¿el ruido espacialmente correlacionado (RFF)
supera al ruido BLANCO i.i.d. de potencia equiparada?

Hipótesis central del paper: la correlación espacial es lo que distingue a Seismic
Descent de ruido i.i.d. (SA/SGLD). Control: misma dinámica, mismo schedule,
misma potencia media del gradiente de ruido, pero direcciones i.i.d. por paso.
"""
import numpy as np
import sys
sys.path.insert(0, "/home/user/seismic-descent/src")

from seismic_descent.functions import RASTRIGIN, ROSENBROCK
from seismic_descent.core import SeismicSwarm


def run_variant(fdict, dim, seed, n_steps=300, n_particles=10, mode="rff"):
    """v20 con motor de ruido intercambiable: 'rff' (correlacionado) o 'white' (i.i.d. equiparado)."""
    r = fdict["search_range"]
    bounds = np.array([[-r, r]] * dim)
    opt = SeismicSwarm(bounds=bounds, n_particles=n_particles, n_steps=n_steps, seed=seed)
    rng = np.random.default_rng(seed)
    rng_white = np.random.default_rng(seed + 10_000)
    x_norm = rng.uniform(-1.0, 1.0, size=(n_particles, dim))
    t, dt_noise = 0.0, (opt.n_cycles * np.pi) / n_steps
    # Potencia equiparada: norma media del gradiente RFF a |amp|=1 (medida empíricamente)
    X = rng.uniform(-1, 1, size=(400, dim))
    ref = np.linalg.norm(opt.rff.grad(X, 0.7, amplitude=1.0), axis=1).mean()

    x_real = opt.center + x_norm * opt.half_range
    vals = np.atleast_1d(np.asarray(fdict["fn"](x_real), dtype=np.float64))
    best_val = float(vals.min())
    for step in range(n_steps):
        decay = opt.noise_decay ** step
        amp = opt.noise_amplitude * decay * np.sin(t * 2.0 * decay)
        x_real = opt.center + x_norm * opt.half_range
        g = np.asarray(fdict["grad"](x_real), dtype=np.float64)
        if g.ndim == 1:
            g = g.reshape(1, -1)
        g = g * opt.half_range
        nrm = np.linalg.norm(g, axis=1, keepdims=True)
        g_dir = np.where(nrm > 1e-8, g / nrm, 0.0)
        if mode == "rff":
            g_noise = opt.rff.grad(x_norm, t, amplitude=amp)
        elif mode == "white":  # ruido blanco: dirección uniforme en la esfera, misma potencia media
            w = rng_white.normal(0, 1, size=(n_particles, dim))
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            g_noise = w * (ref * amp)
        else:  # "none": sin ruido
            g_noise = 0.0
        grad = g_dir + g_noise
        cs = opt.dt_floor + (1.0 - opt.dt_floor) * np.abs(np.sin(t * opt.dt_cycles_multiplier))
        x_norm -= opt.dt_base * cs * grad
        np.clip(x_norm, -1.0, 1.0, out=x_norm)
        t += dt_noise
        x_real = opt.center + x_norm * opt.half_range
        vals = np.atleast_1d(np.asarray(fdict["fn"](x_real), dtype=np.float64))
        best_val = min(best_val, float(vals.min()))
    return best_val


for fname, fdict in (("rastrigin", RASTRIGIN), ("rosenbrock", ROSENBROCK)):
    for dim in (5, 10):
        meds = {}
        for mode in ("rff", "white", "none"):
            vals = [run_variant(fdict, dim, s, mode=mode) for s in range(1, 6)]
            meds[mode] = float(np.median(vals))
        print(f"{fname} {dim}D (budget 3000, 5 trials, mediana): "
              f"RFF correlacionado={meds['rff']:.3f} | BLANCO equiparado={meds['white']:.3f} | SIN ruido={meds['none']:.3f}")
