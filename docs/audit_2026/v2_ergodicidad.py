"""Verificación 2: ¿La 'ergodicidad Laplaciana' se reproduce con el paquete Python real?

Metodología:
- Rastrigin 1D, SeismicSwarm v20 (paquete real), n_particles=1 y 10, 50k pasos.
- Registramos la posición de cada partícula en cada paso (versión instrumentada idéntica a core.py).
- Ajuste MLE: Laplace vs Normal sobre residuos respecto al mínimo visitado más cercano.
- Tests: Kolmogorov-Smirnov para ambas distribuciones + razón de verosimilitud (AIC).
- Cobertura: fracción de rejilla visitada (métrica de ergodicidad).
"""
import numpy as np
import sys
sys.path.insert(0, "/home/user/seismic-descent/src")
from scipy import stats

from seismic_descent.functions import RASTRIGIN
from seismic_descent.core import SeismicSwarm


def run_instrumented(n_particles, n_steps, seed):
    """Copia instrumentada de SeismicSwarm.optimize que devuelve trayectorias."""
    bounds = np.array([[-5.12, 5.12]])
    opt = SeismicSwarm(bounds=bounds, n_particles=n_particles, n_steps=n_steps, seed=seed)
    rng = np.random.default_rng(opt.seed)
    x_norm = rng.uniform(-1.0, 1.0, size=(opt.n_particles, opt.dim))
    t = 0.0
    dt_noise = (opt.n_cycles * np.pi) / opt.n_steps
    traj = []
    for step in range(opt.n_steps):
        decay = opt.noise_decay ** step
        freq = 2.0 * decay
        amp = opt.noise_amplitude * decay * np.sin(t * freq)
        x_real = opt.center + x_norm * opt.half_range
        f_grad_real = np.asarray(RASTRIGIN["grad"](x_real), dtype=np.float64)
        if f_grad_real.ndim == 1:
            f_grad_real = f_grad_real.reshape(1, -1)
        f_grad_mapped = f_grad_real * opt.half_range
        norms = np.linalg.norm(f_grad_mapped, axis=1, keepdims=True)
        f_grad_dir = np.where(norms > 1e-8, f_grad_mapped / norms, 0.0)
        noise_grad = opt.rff.grad(x_norm, t, amplitude=amp)
        grad = f_grad_dir + noise_grad
        cyclic_scale = opt.dt_floor + (1.0 - opt.dt_floor) * np.abs(np.sin(t * opt.dt_cycles_multiplier))
        current_dt = opt.dt_base * cyclic_scale
        x_norm -= current_dt * grad
        np.clip(x_norm, -1.0, 1.0, out=x_norm)
        t += dt_noise
        traj.append(x_norm.copy())
    return opt.center + np.array(traj) * opt.half_range  # (steps, N, 1)


def analyze(traj, label, domain=(-5.12, 5.12), cell=0.04):
    X = traj[:, :, 0].ravel()
    # Cobertura de la rejilla
    grid = np.arange(domain[0], domain[1], cell)
    visited = np.zeros(len(grid), dtype=bool)
    idx = np.clip(((X - domain[0]) / cell).astype(int), 0, len(grid) - 1)
    visited[idx] = True
    coverage = visited.mean()

    # Mínimos locales de Rastrigin 1D: enteros (f=0 en 0, mínimos locales en ±1, ±2, ...)
    resid = X - np.round(X)          # distancia al entero (cuenca) más cercano
    # Ajustes MLE
    mu, sig = stats.norm.fit(resid)
    loc, b = stats.laplace.fit(resid)
    ll_n = stats.norm.logpdf(resid, mu, sig).sum()
    ll_l = stats.laplace.logpdf(resid, loc, b).sum()
    aic_n, aic_l = 2 * 2 - 2 * ll_n, 2 * 2 - 2 * ll_l
    ks_n = stats.kstest(resid, 'norm', args=(mu, sig)).statistic
    ks_l = stats.kstest(resid, 'laplace', args=(loc, b)).statistic

    # Kurtosis y comparación de colas
    kurt = stats.kurtosis(resid)  # exceso: Normal=0, Laplace=3
    print(f"\n--- {label} ---")
    print(f"  muestras={len(X):,}  cobertura rejilla={coverage*100:.1f}%")
    print(f"  residuo |x - entero más cercano|: std={resid.std():.3f}, kurtosis exceso={kurt:.2f} (Norm=0, Laplace=3)")
    print(f"  KS Normal={ks_n:.4f} | KS Laplace={ks_l:.4f}  -> mejor: {'LAPLACE' if ks_l < ks_n else 'NORMAL'}")
    print(f"  AIC Normal={aic_n:.0f} | AIC Laplace={aic_l:.0f}  -> mejor: {'LAPLACE' if aic_l < aic_n else 'NORMAL'} (Δ={abs(aic_l-aic_n):.0f})")
    # Firma temporal: autocorrelación de la posición (¿recorrido ergódico o atrapamiento?)
    x0 = traj[:, 0, 0]
    for lag in (100, 1000, 5000):
        if len(x0) > lag + 10:
            ac = np.corrcoef(x0[:-lag], x0[lag:])[0, 1]
            print(f"  autocorrelación posición p0 (lag={lag}): {ac:+.3f}")


for n in (1, 10):
    for seed in (1, 2):
        traj = run_instrumented(n, 20000, seed)
        analyze(traj, f"Rastrigin 1D, N={n} partículas, seed={seed}, 20k pasos")
