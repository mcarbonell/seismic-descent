"""experiment_v25_adaptive_amp.py — v25: "cuándo temblar y cuánto" hecho política.

Evidencia acumulada (auditoría 2026):
- ablación de ruido: perturbar ayuda SOLO en multimodal deceptivo, perjudica en valle/unimodal.
- ley √D + grid ref_dim: amplitud pequeña (ref≈1) gana en casi todo; Levy/Schwefel
  invierten el orden → ninguna constante global es óptima.

v25 = champion v23 + amplitud que SE ADAPTA al régimen detectado online, sin tocar v23/v24
(copia instrumentada fiel del bucle, con un único cambio: el cálculo de `amp`).

Dos mecanismos candidatos, ambos O(D) por paso y sin hiperparámetros nuevos relevantes:

- **v25c coherencia**: coh_t = ‖media_i(ĝ_i)‖ ∈ [0,1] sobre los gradientes objetivo
  normalizados y precondicionados (señal gratis, ya calculada en el paso 5).
  En valle compartido, las partículas "empujan" en la misma dirección (coh→1: camina,
  no tiembles); en multimodal deceptivo, las direcciones son discordantes
  (coh→0: tiembla fuerte).  amp_t = amp_lo + (amp_hi − amp_lo)·(1 − EMA_coh).
- **v25s estancamiento**: s_t = EMA de 1[mejora relativa del best < tol];
  amp_t = amp_lo + (amp_hi − amp_lo)·s_t². Sin mejoras → tiembla; progresando → refina.

Límites: amp_lo = 0.5·√(1/D) (ganador del grid ref_dim≈1), amp_hi = 0.5 absoluto
(histórico; ganador de la clase Levy). Comparados contra v23_abs, v24(ref5) y ref1.

Protocolo: 12 funciones × {5,10,20} × 15 trials pareados, presupuesto 3000, ORF.
Salida: results/v25_adaptive.json + docs/audit_2026/informe_v25_adaptativa.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, "src")
from seismic_descent.champion_v23 import SeismicChampionV23  # noqa: E402
from seismic_descent.functions import ALL_FUNCTIONS  # noqa: E402
from seismic_descent.functions_extended import EXTENDED_FUNCTIONS, get_search_range  # noqa: E402
from benchmarks.stats import wilcoxon_paired, holm_adjust, significance_stars  # noqa: E402

SUITE = {**ALL_FUNCTIONS, **EXTENDED_FUNCTIONS}
N_PARTICLES = 10
AMP_HI = 0.5           # amplitud absoluta histórica (clase Levy)
REF_LO = 1.0           # amp_lo = 0.5·sqrt(1/D) (ganador del grid)
MODESTY = 2            # exponente del gate de estancamiento


class SeismicChampionV25(SeismicChampionV23):
    """Champion v23 cuya amplitud efectiva se adapta al régimen detectado.

    Copia fiel del bucle optimize() de champion_v23 con un único cambio: el cálculo
    de `amp` se sustituye por una política adaptativa ('coherence' o 'stagnation').
    NO forma parte del paquete: es el candidato experimental v25.
    """

    def __init__(self, *args, mode: str = "coherence", ema_rate: float = 0.92,
                 stagnation_tol: float = 1e-12, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.ema_rate = ema_rate
        self.stagnation_tol = stagnation_tol

    def optimize(self, fn, fn_grad, x0=None):  # noqa: C901 — bucle instrumentado
        rng = np.random.default_rng(self.seed)
        x_norm = rng.uniform(-1.0, 1.0, size=(self.n_particles, self.dim))
        if x0 is not None:
            x_norm[0] = np.clip((np.asarray(x0, dtype=np.float64) - self.center) / self.half_range, -1.0, 1.0)

        velocities = np.zeros((self.n_particles, self.dim))
        diag_metric = np.ones(self.dim)
        t = 0.0
        dt_noise = (self.n_cycles * np.pi) / max(1, self.n_steps)

        amp_lo = 0.5 * np.sqrt(REF_LO / self.dim)
        amp_hi = AMP_HI
        coh_ema = 0.5        # neutral start
        stag_ema = 0.0

        x_real = self.center + x_norm * self.half_range
        real_vals = np.atleast_1d(np.asarray(fn(x_real), dtype=np.float64))
        best_idx = int(np.argmin(real_vals))
        best_val = float(real_vals[best_idx])
        best_x_real = x_real[best_idx].copy()
        best_x_norm = x_norm[best_idx].copy()
        best_per_step = [best_val]

        for step_i in range(self.n_steps):
            decay = self.noise_decay ** step_i
            freq = 2.0 * decay
            sin_phase = np.sin(t * freq)

            # --- ÚNICO CAMBIO vs v23: amplitud adaptativa por régimen ---
            if self.mode == "coherence":
                gate = 1.0 - coh_ema          # coherencia baja -> temblor fuerte
            else:  # "stagnation"
                gate = stag_ema ** MODESTY
            amp = (amp_lo + (amp_hi - amp_lo) * gate) * decay * sin_phase
            # ------------------------------------------------------------

            x_real = self.center + x_norm * self.half_range
            f_grad_real = np.asarray(fn_grad(x_real), dtype=np.float64)
            if f_grad_real.ndim == 1:
                f_grad_real = f_grad_real.reshape(1, -1)
            f_grad_mapped = f_grad_real * self.half_range

            swarm_sq_grad = np.mean(f_grad_mapped ** 2, axis=0)
            diag_metric = self.metric_beta * diag_metric + (1.0 - self.metric_beta) * swarm_sq_grad

            if self.anisotropic_power > 0.0:
                scale_metric = np.sqrt(diag_metric) + 1e-8
                med_scale = np.median(scale_metric)
                rel_scale = scale_metric / med_scale if med_scale > 1e-8 else np.ones_like(scale_metric)
                precond = np.clip(1.0 / (rel_scale ** self.anisotropic_power), 0.1, 10.0)
            else:
                precond = np.ones(self.dim)
            precond_grad = f_grad_mapped * precond
            norms = np.linalg.norm(precond_grad, axis=1, keepdims=True)
            f_grad_dir = np.divide(precond_grad, norms, out=np.zeros_like(precond_grad), where=norms > 1e-8)

            # señal de régimen 1: coherencia de direcciones de descenso del enjambre
            coh_raw = float(np.linalg.norm(np.mean(f_grad_dir, axis=0)))   # ∈ [0,1]
            coh_ema = self.ema_rate * coh_ema + (1.0 - self.ema_rate) * coh_raw

            noise_grad = self.noise_field.grad(x_norm, t, amplitude=amp) if self.noise_field is not None else 0.0

            if self.gravity_strength > 0.0:
                gamma = self.gravity_strength * max(0.0, 1.0 - abs(sin_phase))
                diff = best_x_norm - x_norm
                diff_norms = np.linalg.norm(diff, axis=1, keepdims=True)
                grav_pull = gamma * np.where(diff_norms > 1e-7, diff / np.maximum(diff_norms, 1e-7), 0.0)
            else:
                grav_pull = 0.0

            f_total = -(f_grad_dir + noise_grad - grav_pull)

            if self.momentum_base > 0.0:
                mu = self.momentum_base * (0.5 + 0.5 * abs(sin_phase))
                velocities = mu * velocities + (1.0 - mu) * f_total
                step_direction = velocities
            else:
                step_direction = f_total

            cyclic_scale = self.dt_floor + (1.0 - self.dt_floor) * np.abs(np.sin(t * self.dt_cycles_multiplier))
            x_norm += (self.dt_base * cyclic_scale) * step_direction
            np.clip(x_norm, -1.0, 1.0, out=x_norm)
            t += dt_noise

            x_real = self.center + x_norm * self.half_range
            step_vals = np.atleast_1d(np.asarray(fn(x_real), dtype=np.float64))
            step_best_idx = int(np.argmin(step_vals))
            step_best_val = float(step_vals[step_best_idx])

            # señal de régimen 2: estancamiento de la mejora del best
            improvement = (best_val - step_best_val) / max(1.0, abs(best_val))
            stag_ema = self.ema_rate * stag_ema + (1.0 - self.ema_rate) * float(improvement <= self.stagnation_tol)

            if step_best_val < best_val:
                best_val = step_best_val
                best_x_real = x_real[step_best_idx].copy()
                best_x_norm = x_norm[step_best_idx].copy()
            best_per_step.append(best_val)

        return best_x_real, best_val, {"best_per_step": best_per_step}


def amp_label(ref: float) -> str:
    return "v23_abs" if ref < 0 else f"v24_ref{ref:g}"


def run_cell(fname: str, dim: int, budget: int, seed: int, variant: str, x0=None) -> float:
    fcfg = SUITE[fname]
    r = get_search_range(fcfg, dim)
    bounds = np.array([[-r, r]] * dim)
    n_steps = max(20, budget // N_PARTICLES)
    common = dict(bounds=bounds, n_particles=N_PARTICLES, n_steps=n_steps,
                  noise_engine="orf", seed=seed)
    if variant == "v23_abs":
        opt = SeismicChampionV23(**common)
    elif variant.startswith("v24_ref"):
        ref = float(variant.split("ref")[1])
        opt = SeismicChampionV23(noise_amplitude=0.5 * np.sqrt(ref / 5.0),
                                 noise_amp_dim_normalized=True, **common)
    elif variant == "v25c_coherence":
        opt = SeismicChampionV25(mode="coherence", **common)
    elif variant == "v25s_stagnation":
        opt = SeismicChampionV25(mode="stagnation", **common)
    else:
        raise ValueError(variant)
    _, val, _ = opt.optimize(fn=fcfg["fn"], fn_grad=fcfg["grad"], x0=x0)
    return float(val)


VARIANTS = ["v23_abs", "v24_ref5", "v24_ref1", "v25c_coherence", "v25s_stagnation"]
SMOOTH = {"sphere", "trid", "zakharov", "dixon_price", "griewank", "rosenbrock"}
DECEPTIVE = {"levy", "michalewicz", "schwefel", "rastrigin", "styblinski_tang", "ackley"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    fnames = sorted(SUITE)
    if args.quick:
        fnames, args.dims, args.trials = ["levy", "sphere", "trid", "ackley"], [5, 20], 3

    cells: Dict[str, Dict[str, List[float]]] = {}
    t0 = time.time()
    for fname in fnames:
        for dim in args.dims:
            key = f"{fname}_{dim}d"
            cells[key] = {v: [] for v in VARIANTS}
            for trial in range(args.trials):
                # Paired across variants via the optimizer's own seeded init (x0=None);
                # protocol identical to experiment_refdim_train_test.py.
                for v in VARIANTS:
                    cells[key][v].append(run_cell(fname, dim, args.budget, trial + 1, v, None))
            meds = {v: float(np.median(cells[key][v])) for v in VARIANTS}
            print(f"[{key}] " + " | ".join(f"{v}={m:.4g}" for v, m in meds.items())
                  + f" ({time.time()-t0:.0f}s)", flush=True)

    keys = sorted(cells)
    # ranks across variants per cell
    rank_acc = {v: [] for v in VARIANTS}
    for k in keys:
        order = sorted(VARIANTS, key=lambda v: np.median(cells[k][v]))
        for i, v in enumerate(order, 1):
            rank_acc[v].append(i)
    ranks = {v: float(np.mean(rs)) for v, rs in rank_acc.items()}

    # Wilcoxon v25* vs v24_ref1 (its direct rival), Holm over all cells
    for v25 in ("v25c_coherence", "v25s_stagnation"):
        pvals = [wilcoxon_paired(cells[k][v25], cells[k]["v24_ref1"])[0] for k in keys]
        for k, pa in zip(keys, holm_adjust(pvals)):
            cells[k][f"holm_{v25}_vs_ref1"] = float(pa) if pa == pa else None

    Path("results").mkdir(exist_ok=True)
    with open("results/v25_adaptive.json", "w") as f:
        json.dump({"meta": vars(args), "ranks": ranks,
                   "cells": {k: {v: a for v, a in c.items() if isinstance(a, list)}
                             for k, c in cells.items()}}, f, indent=1)

    # ---------------- report ----------------
    lines = ["# v25: amplitud adaptativa por régimen detectado (coherencia vs estancamiento)", ""]
    lines.append(f"> Trials: {args.trials} | Budget: {args.budget} | dims {args.dims} | ORF | init interna pareada "
                 f"(x0=None, misma seed por trial). amp_lo=0.5·√(1/D), amp_hi=0.5 abs. "
                 f"v25 = champion v23, único cambio: política de amplitud.")
    lines.append("")
    lines.append("## Rank medio por celda (1 = mejor en todas)")
    lines.append("")
    lines.append("| variante | rank |")
    lines.append("|---|---|")
    for v in sorted(ranks, key=lambda v: ranks[v]):
        lines.append(f"| {v} | {ranks[v]:.2f} |")
    lines.append("")
    lines.append("## Medianas por celda (★ = Wilcoxon Holm del v25 indicado vs v24_ref1)")
    lines.append("")
    header = "| celda | clase | " + " | ".join(VARIANTS) + " |"
    lines.append(header); lines.append("|---" * (len(VARIANTS) + 2) + "|")
    for k in keys:
        fname = k.rsplit("_", 1)[0]
        cls = "valle" if fname in SMOOTH else "decept."
        row = f"| {k} | {cls} |"
        for v in VARIANTS:
            m = float(np.median(cells[k][v]))
            star = ""
            if v in ("v25c_coherence", "v25s_stagnation"):
                pv = cells[k].get(f"holm_{v}_vs_ref1")
                star = significance_stars(pv) if pv else ""
            row += f" {m:.4g}{star} |"
        lines.append(row)
    Path("docs/audit_2026").mkdir(exist_ok=True)
    Path("docs/audit_2026/informe_v25_adaptativa.md").write_text("\n".join(lines) + "\n")
    print(f"\n[+] results/v25_adaptive.json + docs/audit_2026/informe_v25_adaptativa.md ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
