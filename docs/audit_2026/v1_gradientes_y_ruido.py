"""Verificación 1: gradientes analíticos de functions.py y ratio ruido/señal."""
import numpy as np
import sys
sys.path.insert(0, "/home/user/seismic-descent/src")

from seismic_descent.functions import ALL_FUNCTIONS
from seismic_descent.rff import RandomFourierFeatures
from seismic_descent.orf import OrthogonalRandomFeatures
from seismic_descent.lissajous import LissajousWaveField, OrthogonalLissajousWaveField

print("=" * 70)
print("1A. GRADIENTES DE functions.py vs DIFERENCIAS FINITAS (no cubierto por tests)")
print("=" * 70)
rng = np.random.default_rng(7)
for name, fdict in ALL_FUNCTIONS.items():
    fn, grad_fn = fdict["fn"], fdict["grad"]
    r = fdict["search_range"]
    max_err = 0.0
    for _ in range(50):
        d = int(rng.integers(2, 8))
        x = rng.uniform(-0.9 * r, 0.9 * r, size=d)
        eps = 1e-6
        fd = np.zeros(d)
        for i in range(d):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            fd[i] = (fn(xp) - fn(xm)) / (2 * eps)
        an = grad_fn(x)
        denom = max(1.0, np.linalg.norm(fd))
        max_err = max(max_err, np.linalg.norm(an - fd) / denom)
    status = "OK " if max_err < 1e-4 else "¡FALLO!"
    print(f"  {name:12s} error relativo máx: {max_err:.2e}  [{status}]")

# Esquina especial: Schwefel con x negativo (valor absoluto)
x = np.array([-123.4567, -0.001])
eps = 1e-6
fd = np.zeros(2)
for i in range(2):
    xp, xm = x.copy(), x.copy()
    xp[i] += eps; xm[i] -= eps
    fd[i] = (ALL_FUNCTIONS["schwefel"]["fn"](xp) - ALL_FUNCTIONS["schwefel"]["fn"](xm)) / (2 * eps)
an = ALL_FUNCTIONS["schwefel"]["grad"](x)
print(f"  schwefel(x<0): analítico={an}, dif.finitas={fd}, err={np.linalg.norm(an-fd):.2e}")

print()
print("=" * 70)
print("1B. MAGNITUD DEL GRADIENTE DE RUIDO (amp=0.5) vs gradiente objetivo normalizado (=1.0)")
print("=" * 70)
for dim in (2, 5, 20):
    for cls, cname in ((RandomFourierFeatures, "RFF"), (OrthogonalRandomFeatures, "ORF"),
                       (LissajousWaveField, "Lissajous(coupled)"), (OrthogonalLissajousWaveField, "OrthoLissajous")):
        if cls is LissajousWaveField:
            field = cls(dim=dim, coupled=True)
        elif cls is OrthogonalLissajousWaveField:
            field = cls(dim=dim, n_rotations=3)
        else:
            field = cls(dim=dim, r=64, seed=1)
        X = rng.uniform(-1, 1, size=(200, dim))
        norms = []
        for t in (0.25, 1.0, 3.0):
            G = field.grad(X, t, amplitude=0.5)
            norms.append(np.linalg.norm(G, axis=1).mean())
        print(f"  D={dim:2d} {cname:20s} ||grad_ruido|| medio (amp=0.5): {np.mean(norms):.2f}  -> ratio vs señal: {np.mean(norms):.2f}x")
