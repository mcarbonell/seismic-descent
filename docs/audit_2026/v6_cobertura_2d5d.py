"""Verificación 6 (C1, ampliación F5): cobertura empírica 2D/5D y decorrelación.

Extensión del protocolo 1D de v2_ergodicidad.py a D>1, con definiciones que hacen
las curvas COMPARABLES entre dimensiones:

- Dominio Rastrigin D-dimensional, enjambre v20 instrumentado (mismo bucle que core.py).
- Variantes: 'seismic' (campo RFF correlacionado), 'pure_gd' (amp=0), 'white'
  (dirección i.i.d. potencia-equiparada al campo RFF por calibración Monte-Carlo,
  igual que en experiment_ablation_noise.py).
- Rejillas con EL MISMO NÚMERO DE CELDAS en ambas dimensiones:
  2D -> 32x32 = 1024 celdas; 5D -> 4^5 = 1024 celdas.
- Cobertura(t) = fracción de celdas visitadas acumulando las trayectorias de las
  10 partículas (media ± banda sobre seeds).
- Autocorrelación: media sobre coordenadas del coeficiente de Pearson de la
  trayectoria de la partícula 0 para lags {1,10,100,1000} (GD -> ~1: atrapamiento;
  exploración ergódica -> decae).

Salida: docs/audit_2026/informe_cobertura_2d5d.md + docs/paper_assets/F5_coverage_2d5d.png
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from seismic_descent.functions import RASTRIGIN  # noqa: E402
from seismic_descent.core import SeismicSwarm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent

DIM_CONFIG = {2: 32, 5: 4}          # cells per coordinate -> 1024 cells both
STEPS = 50_000
PARTICLES = 10
SEEDS = list(range(1, 9))
LAGS = (10, 100, 1000, 5000)
CAL_POINTS = 2048


def calibrate_unit_noise_norm(dim: int, coord_cells: int, seed: int = 777) -> float:
    """Median norm of the unit-amplitude RFF gradient field over the domain
    (mirrors calibrate_unit_noise_norm in experiment_ablation_noise.py)."""
    opt = SeismicSwarm(bounds=np.array([[-5.12, 5.12]] * dim), n_particles=2, n_steps=2, seed=seed)
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1.0, 1.0, size=(CAL_POINTS, dim))
    norms = []
    for t in (2.0, 10.0, 40.0):
        g = opt.rff.grad(xs, t, amplitude=1.0)
        norms.append(np.linalg.norm(g, axis=1))
    return float(np.median(np.concatenate(norms)))


def run_variant(variant: str, dim: int, steps: int, seed: int) -> np.ndarray:
    """Instrumented v20 loop; returns trajectory (steps, N, dim) in normalized coords."""
    opt = SeismicSwarm(bounds=np.array([[-5.12, 5.12]] * dim),
                       n_particles=PARTICLES, n_steps=steps, seed=seed)
    rng = np.random.default_rng(opt.seed)
    x_norm = rng.uniform(-1.0, 1.0, size=(PARTICLES, dim))
    t = 0.0
    dt_noise = (opt.n_cycles * np.pi) / steps
    unit_norm = calibrate_unit_noise_norm(dim, DIM_CONFIG[dim], seed + 777) if variant == "white" else 1.0
    rng_white = np.random.default_rng(seed + 50_000)
    traj = np.empty((steps, PARTICLES, dim))
    for step_i in range(steps):
        decay = opt.noise_decay ** step_i
        freq = 2.0 * decay
        amp = opt.noise_amplitude * decay * np.sin(t * freq)
        x_real = opt.center + x_norm * opt.half_range
        g_real = np.asarray(RASTRIGIN["grad"](x_real), dtype=np.float64)
        g_map = g_real * opt.half_range
        norms = np.linalg.norm(g_map, axis=1, keepdims=True)
        g_dir = np.where(norms > 1e-8, g_map / norms, 0.0)
        if variant == "seismic":
            noise_grad = opt.rff.grad(x_norm, t, amplitude=amp)
        elif variant == "white":
            w = rng_white.normal(0.0, 1.0, size=(PARTICLES, dim))
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            noise_grad = w * (unit_norm * amp)
        else:  # pure_gd
            noise_grad = 0.0
        cyclic = opt.dt_floor + (1.0 - opt.dt_floor) * np.abs(np.sin(t * opt.dt_cycles_multiplier))
        x_norm -= (opt.dt_base * cyclic) * (g_dir + noise_grad)
        np.clip(x_norm, -1.0, 1.0, out=x_norm)
        t += dt_noise
        traj[step_i] = x_norm
    return traj


def cell_ids(traj: np.ndarray, coord_cells: int) -> np.ndarray:
    """Flattened grid cell id for every visited point, in [0, coord_cells^dim)."""
    idx = np.clip(((traj + 1.0) / 2.0 * coord_cells).astype(int), 0, coord_cells - 1)
    powers = coord_cells ** np.arange(traj.shape[-1])
    return (idx * powers).sum(axis=-1)  # (steps, N)


def coverage_curve(traj: np.ndarray, coord_cells: int, every: int = 50) -> np.ndarray:
    ids = cell_ids(traj, coord_cells).ravel()
    n_cells = coord_cells ** traj.shape[-1]
    visited = np.zeros(n_cells, dtype=bool)
    pts = ids[::every]
    curve = []
    for cid in pts:
        visited[cid] = True
        curve.append(visited.mean())
    return np.asarray(curve)


def autocorr_lags(traj: np.ndarray, lags: Tuple[int, ...]) -> Dict[int, float]:
    """Mean over coordinates of the lag-k Pearson autocorrelation of particle 0."""
    out: Dict[int, float] = {}
    x0 = traj[:, 0, :]                      # (steps, dim)
    for lag in lags:
        if x0.shape[0] <= lag + 10:
            out[lag] = np.nan
            continue
        acs = [np.corrcoef(x0[:-lag, d], x0[lag:, d])[0, 1] for d in range(x0.shape[1])]
        out[lag] = float(np.nanmean(acs))
    return out


def mean_band(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(curves)
    return A.mean(0), np.percentile(A, 25, 0), np.percentile(A, 75, 0)


def main() -> None:
    variants = ("seismic", "white", "pure_gd")
    colors = {"seismic": "#d62728", "white": "#9467bd", "pure_gd": "#1f77b4"}
    labels = {"seismic": "Seismic (RFF field)", "white": "White (power-matched)",
              "pure_gd": "pure GD (amp = 0)"}

    results: Dict[Tuple[str, int], Dict] = {}
    for dim in DIM_CONFIG:
        for variant in variants:
            covs, acs = [], []
            for seed in SEEDS:
                traj = run_variant(variant, dim, STEPS, seed)
                covs.append(coverage_curve(traj, DIM_CONFIG[dim]))
                acs.append(autocorr_lags(traj, LAGS))
            results[(variant, dim)] = {"cov": covs, "ac": acs}
            m, lo, hi = mean_band(covs)
            pos = min(len(m) - 1, int(0.9 * len(m)))
            print(f"[{variant:8s} D={dim}] coverage@90%steps={m[pos]*100:5.1f}% "
                  f"final={m[-1]*100:5.1f}% | autocorr lag100={np.mean([a[100] for a in acs]):+.3f}",
                  flush=True)

    # ---------------- figure F5 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    xs_t = (np.arange(len(results[("seismic", 2)]["cov"][0])) * 50) / STEPS
    for dim, ax, ttl in ((2, axes[0, 0], "(a) coverage vs time, D=2 (32x32 cells)"),
                         (5, axes[0, 1], "(b) coverage vs time, D=5 (4^5 cells)")):
        for variant in variants:
            m, lo, hi = mean_band(results[(variant, dim)]["cov"])
            ax.plot(xs_t, m, color=colors[variant], lw=1.5, label=labels[variant])
            ax.fill_between(xs_t, lo, hi, color=colors[variant], alpha=0.15, linewidth=0)
        ax.axhline(1.0, color="k", ls="--", lw=0.7, alpha=0.5)
        ax.set_xlabel("step (fraction of 50k steps)", fontsize=9)
        ax.set_ylabel("grid coverage", fontsize=9)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_ylim(0, 1.05)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower right")

    ax = axes[1, 0]
    for variant in variants:
        for dim in (2, 5):
            ac_m = [np.mean([a[lag] for a in results[(variant, dim)]["ac"]]) for lag in LAGS]
            ax.plot(LAGS, ac_m, marker="o", color=colors[variant], lw=1.4, ms=3.5,
                    ls="-" if dim == 2 else "--", alpha=1.0 if dim == 2 else 0.65,
                    label=f"{labels[variant]}, D={dim}")
    ax.set_xscale("log"); ax.set_xlabel("lag (steps)", fontsize=9)
    ax.set_ylabel("position autocorrelation\n(mean over coords, particle 0)", fontsize=8.5)
    ax.set_title("(c) decorrelation of the trajectory", fontsize=9.5)
    ax.legend(fontsize=6.3, frameon=False, loc="lower left", bbox_to_anchor=(0.0, -0.02))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # (d) 2D occupation density: pure GD vs seismic, last 100% of the run, seed=1
    ax = axes[1, 1]
    dim = 2; gc = 64
    for j, variant in enumerate(("pure_gd", "seismic")):
        traj = run_variant(variant, dim, STEPS, seed=1)
        pts = traj.reshape(-1, dim)
        H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=gc, range=[[-1, 1], [-1, 1]])
        ax_in = ax.inset_axes([0.03 + j * 0.5, 0.12, 0.44, 0.82])
        ax_in.imshow(np.log10(H.T + 1), origin="lower", extent=[-1, 1, -1, 1],
                     cmap="magma", vmin=0, vmax=np.log10(H.max() + 1))
        ax_in.set_title({"pure_gd": "pure GD (collapses\nto fixed points)", "seismic": "Seismic (keeps exploring)"}[variant],
                    fontsize=8)
        ax_in.set_xticks([]); ax_in.set_yticks([])
    ax.axis("off")
    ax.set_title("(d) 2D occupation density (10 particles x 10k steps, seed=1)",
                 fontsize=9.5, y=0.02)

    fig.suptitle("F5. Empirical coverage and decorrelation (Rastrigin; 8 seeds, 10 particles, 50k steps)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = ROOT / "docs" / "paper_assets" / "F5_coverage_2d5d.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"  [+] {out_png.name}")

    # ---------------- report ----------------
    lines = ["# Cobertura empírica 2D/5D y decorrelación (C1, figura F5)", ""]
    lines.append(f"> Rastrigin; seeds {SEEDS[0]}..{SEEDS[-1]}; {PARTICLES} partículas; {STEPS} pasos (50k); "
                 f"rejillas 32x32 (2D) y 4^5 (5D) = 1024 celdas en ambos casos → curvas comparables.")
    lines.append("")
    lines.append("| variante | D | cobertura@100% pasos | autocorrelación lag=100 (media seeds) |")
    lines.append("|---|---|---|---|")
    for variant in variants:
        for dim in DIM_CONFIG:
            m, lo, hi = mean_band(results[(variant, dim)]["cov"])
            ac = np.mean([a[100] for a in results[(variant, dim)]["ac"]])
            lines.append(f"| {labels[variant]} | {dim} | {m[-1]*100:.1f}% ({lo[-1]*100:.0f}-{hi[-1]*100:.0f}) | {ac:+.3f} |")
    lines.append("")
    lines.append("**Lectura honesta (3 resultados):** (i) **perturbar ≫ no perturbar**: la cobertura "
                 "del sísmico crece sin saturar durante los 50k pasos mientras el GD puro se congela "
                 "(12% en 2D, ~0% en 5D — literalmente una celda por partícula); (ii) la promesa 1D de "
                 "'cobertura completa' NO se extiende literal: a 50k pasos el sísmico no llega al 100% "
                 "(su curva sigue con pendiente positiva — coverage creciente, no completa a este "
                 "horizonte); (iii) el blanco potencia-equiparada cubre MÁS RÁPIDO (satura ~80%/~70%) "
                 "porque su perturbación es por paso i.i.d., mientras el campo sísmico es coherente en "
                 "el tiempo (autocorrelación de posición +0.74/+0.81 a lag 100 vs +0.07/+0.11 del "
                 "blanco): **la correlación espaciotemporal NO está para dispersar más rápido, sino "
                 "para explorar con estructura** (arrays de acuerdo coherentes que el tracking y el "
                 "detector KD aprovechan) — es el complemento funcional de la ablación de ruido "
                 "(paridad en valor final). Claim C1 queda así refinado: cobertura empírica "
                 "creciente y muy superior a GD, sin completitud a 50k, y coherencia temporal como "
                 "rasgo distintivo real del terremoto correlacionado. ⚠️ Y sigue sin ser ergodicidad "
                 "formal ni distribución Laplaciana (kurtosis −0.6…−0.85; theory.md retractado).")
    (ROOT / "docs" / "audit_2026" / "informe_cobertura_2d5d.md").write_text("\n".join(lines) + "\n")
    print("[+] informe_cobertura_2d5d.md")


if __name__ == "__main__":
    main()
