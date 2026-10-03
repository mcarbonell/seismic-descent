"""make_paper_figures.py — Publication figures F1-F4 for the Seismic Descent paper.

Parses the *committed* audit reports (docs/audit_2026/*.md) — the raw results/*.json
are gitignored and regenerable, but the reports are the canonical, versioned source.
Everything plotted here traces to a specific table in a specific report (see
docs/paper_assets/README.md). F5 is generated separately by
docs/audit_2026/v6_cobertura_2d5d.py.

Usage: python -m benchmarks.make_paper_figures
Outputs: docs/paper_assets/F{1..4}_*.png (300 dpi)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = DOCS / "paper_assets"

FUNCS = ["ackley", "dixon_price", "griewank", "levy", "michalewicz", "rastrigin",
         "rosenbrock", "schwefel", "sphere", "styblinski_tang", "trid", "zakharov"]
DIMS4 = [2, 5, 10, 20]

BASELINE_LABELS = {
    "seismic_v20": "Seismic v20 (RFF)",
    "seismic_v23_orf": "Seismic v23 (Champion ORF)",
    "seismic_v24_orf": "Seismic v24 (ref_dim=5)",
    "seismic_v24r1_orf": "Seismic v24 (ref_dim=1) ★",
    "seismic_v25c_orf": "Seismic v25c (adaptive)",
    "cmaes": "CMA-ES (basic)",
    "ipop_cmaes": "IPOP-CMA-ES",
    "pso": "PSO (Clerc-Kennedy)",
    "bfgs_ms": "L-BFGS-B multistart",
    "random": "Random search",
}


# ---------------------------------------------------------------- MD parsing
def parse_md_table(lines: List[str], header_marker: str) -> List[List[str]]:
    """Return rows (list of cell strings) of the first table whose header row
    contains header_marker. Strips **bold** markers."""
    rows: List[List[str]] = []
    in_table = False
    for ln in lines:
        if ln.strip().startswith("|") and header_marker in ln:
            in_table = True
            continue
        if in_table:
            if not ln.strip().startswith("|"):
                break
            if re.fullmatch(r"\|[-| :]+\|[ :]*", ln.strip()):  # alignment row
                continue
            cells = [c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")]
            rows.append(cells)
    return rows


def fnum(s: str) -> Optional[float]:
    m = re.match(r"[-+0-9.eE]+", s.strip())
    return float(m.group(0)) if m else None


def load_phase1_baselines() -> Dict[str, Dict[str, float]]:
    txt = (DOCS / "audit_2026" / "informe_baselines_v24.md").read_text().splitlines()
    rows = parse_md_table(txt, header_marker="seismic_v23_orf")
    data: Dict[str, Dict[str, float]] = {}
    for cells in rows:
        key = cells[0]
        data[key] = {}
        for i, algo in enumerate(list(BASELINE_LABELS)):
            if i + 1 < len(cells):
                v = fnum(cells[i + 1])
                if v is not None:
                    data[key][algo] = v
        if len(cells) > len(BASELINE_LABELS) + 1:
            pv = fnum(cells[len(BASELINE_LABELS) + 2]) if len(cells) > len(BASELINE_LABELS) + 2 else None
            data[key]["holm_v24_vs_ipop"] = pv if pv is not None else 1.0
    return data


def load_phase2_baselines() -> Dict[str, Dict[str, float]]:
    txt = (DOCS / "audit_2026" / "informe_baselines_v24.md").read_text()
    section = txt.split("## Fase 2")[1] if "## Fase 2" in txt else txt
    rows = parse_md_table(section.splitlines(), header_marker="seismic_v23_orf")
    p2 = {}
    for cells in rows:
        parts = cells[0].rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 100:  # *_budget rows
            func_dim = parts[0]
            budget = int(parts[1])
            p2.setdefault(func_dim, {}).setdefault(budget, {})
            for i, algo in enumerate(("seismic_v20", "seismic_v23_orf", "cmaes", "ipop_cmaes")):
                v = fnum(cells[i + 1])
                if v is not None:
                    p2[func_dim][budget][algo] = v
    return p2


def load_noise_ablation() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """cell -> {'adv_perturb': (r, p), 'adv_corr': (r, p)} with r = (A-B)/(|A|+|B|)."""
    txt = (DOCS / "audit_2026" / "informe_ablacion_ruido.md").read_text().splitlines()
    rows = parse_md_table(txt, header_marker="Blanco (mediana)")
    out = {}
    for cells in rows:
        key = cells[0]
        rff = fnum(cells[1]); white = fnum(cells[2]); none = fnum(cells[3])
        p_corr = fnum(cells[4].split("(")[0] + " " + cells[4]) or fnum(cells[4])
        p_none = fnum(cells[5].split("(")[0] + " " + cells[5]) or fnum(cells[5])
        if None in (rff, white, none):
            continue
        def nrm(a, b):
            den = abs(a) + abs(b)
            return (a - b) / den if den else 0.0
        pc = fnum(cells[4].replace("v23 vs ipop:", ""))
        pv = fnum(cells[5])
        out[key] = {
            "adv_perturb": (nrm(none, rff), pv if pv is not None else 1.0),
            "adv_corr": (nrm(white, rff), pc if pc is not None else 1.0),
        }
    return out


def load_amp_sensitivity() -> Dict[str, Tuple[float, float, float]]:
    """cell_20/10 -> (abs_med, dnorm_med, p_holm) for a=0.5 columns (Parte B)."""
    txt = (DOCS / "audit_2026" / "informe_ablacion_componentes.md").read_text()
    part_b = txt.split("## Parte B")[1] if "## Parte B" in txt else txt
    rows = parse_md_table(part_b.splitlines(), header_marker="a=0.25 abs")
    out = {}
    for cells in rows:
        key = cells[0]
        abs_v = fnum(cells[3])                       # a=0.5 abs
        dn_cell = cells[4]                           # "val (p stars)"
        dn_v = fnum(dn_cell)
        pm = re.search(r"\(([-+0-9.eE]+)", dn_cell)
        p = float(pm.group(1)) if pm else (0.0 if "n/a" in dn_cell else 1.0)
        if abs_v is not None and dn_v is not None:
            out[key] = (abs_v, dn_v, p)
    return out


# ---------------------------------------------------------------- helpers
def style_ax(ax, grid=True):
    ax.tick_params(labelsize=8)
    if grid:
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] {name}")


# ---------------------------------------------------------------- figures
def fig_f1_budget_crossover(p2: Dict):
    """F1: median vs evaluation budget (phase 2, Rastrigin/Griewank 5D)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
    colors = {"seismic_v20": "#999999", "seismic_v23_orf": "#d62728",
              "cmaes": "#1f77b4", "ipop_cmaes": "#2ca02c"}
    for ax, funcdim in zip(axes, ["rastrigin_5d", "griewank_5d"]):
        budgets = sorted(p2[funcdim])
        for algo, lab in (("seismic_v20", "Seismic v20"), ("seismic_v23_orf", "Seismic v23"),
                          ("cmaes", "CMA-ES basic"), ("ipop_cmaes", "IPOP-CMA-ES")):
            ys = [p2[funcdim][b][algo] for b in budgets]
            ax.plot(budgets, ys, "o-", color=colors[algo], ms=4, lw=1.4, label=lab)
        ax.set_yscale("log")
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, NullFormatter
        ax.xaxis.set_major_locator(FixedLocator(budgets))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([f"{b // 1000}k" for b in budgets])
        ax.set_xlabel("evaluation budget", fontsize=9)
        ax.set_title(funcdim.replace("_", " "), fontsize=10)
        style_ax(ax)
    axes[0].set_ylabel("median best f (15 trials)", fontsize=9)
    axes[1].legend(fontsize=7.5, frameon=False, loc="lower left")
    save(fig, "F1_budget_crossover.png")


def fig_f2_baselines(p1: Dict):
    """F2: (a) cells where each algorithm attains the best median;
    (b) paired scatter v24 vs IPOP-CMA-ES (log-log), colored by Holm p."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    algos = list(BASELINE_LABELS)
    counts = {a: 0 for a in algos}
    for key, meds in p1.items():
        best = min((a for a in algos if a in meds), key=lambda a: meds[a])
        counts[best] += 1
    order = sorted(algos, key=lambda a: -counts[a])
    axA.barh([BASELINE_LABELS[a] for a in order][::-1],
             [counts[a] for a in order][::-1], color="#4C72B0")
    axA.set_xlabel("cells with best median (of 36)", fontsize=9)
    axA.set_title("(a) best-median count", fontsize=9.5)
    style_ax(axA, grid=False)
    axA.xaxis.grid(True, alpha=0.3)

    lim = [1e-16, 1e5]
    axB.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    cats = {  # category -> (marker color, label, points list)
        "win": ("#2ca02c", "v24r1 better (>10%)", []),
        "tie": ("#888888", "tie (±10%)", []),
        "loss": ("#d62728", "v24r1 worse (>10%)", []),
    }
    for key, m in sorted(p1.items()):
        ip = m["ipop_cmaes"]; v24 = m["seismic_v24r1_orf"]
        den = abs(ip) + abs(v24)
        rel = (v24 - ip) / den if den else 0.0
        cat = "tie" if abs(rel) <= 0.052 else ("win" if rel < 0 else "loss")
        if ip < v24 * 0.9:
            cat = "loss" if cat != "tie" else cat
        holm = m.get("holm_v24r1_vs_ipop", 1.0)
        sig = holm is not None and holm == holm and holm < 0.05
        cats[cat][2].append((ip, v24, sig, key))
    for cat, (color, label, pts) in cats.items():
        if not pts:
            continue
        xr = [p[0] for p in pts]; yr = [p[1] for p in pts]
        sz = [42 if p[2] else 16 for p in pts]
        axB.scatter(xr, yr, c=color, s=sz, edgecolor="k",
                    linewidth=0.5, zorder=3, label=f"{label} (n={len(pts)})")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.set_xlabel("IPOP-CMA-ES median f", fontsize=9)
    axB.set_ylabel("Seismic v24 (ref_dim=1) median f", fontsize=9)
    axB.set_title("(b) v24(ref1) vs IPOP (bigger dot = Holm p<0.05)", fontsize=9.5)
    axB.legend(fontsize=7.5, frameon=False, loc="lower right")
    style_ax(axB)
    save(fig, "F2_baselines_panels.png")


def fig_f3_regime_map(ab: Dict):
    """F3: regime-map heatmaps — perturbation advantage (none−rff) and
    correlation advantage (white−rff), normalized to [-1, 1], Holm stars."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, kind, title in ((axes[0], "adv_perturb", "(a) perturbing vs pure GD\n(none − RFF, normalized)"),
                            (axes[1], "adv_corr", "(b) correlated vs power-matched white\n(white − RFF, normalized)")):
        M = np.zeros((len(FUNCS), len(DIMS4)))
        P = np.ones_like(M)
        for i, f in enumerate(FUNCS):
            for j, d in enumerate(DIMS4):
                r, p = ab[f"{f}_{d}d"][kind]
                M[i, j] = r; P[i, j] = p if p == p else 1.0
        im = ax.imshow(M, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        for i in range(len(FUNCS)):
            for j in range(len(DIMS4)):
                star = "*" if P[i, j] < 0.05 else ""
                ax.text(j, i, star, ha="center", va="center", fontsize=11, color="k")
        ax.set_xticks(range(len(DIMS4)), [f"D={d}" for d in DIMS4], fontsize=8)
        if ax is axes[0]:
            ax.set_yticks(range(len(FUNCS)), [f.replace("_", " ") for f in FUNCS], fontsize=8)
        else:
            ax.set_yticks(range(len(FUNCS)), [])  # shared y labels on the left only
        ax.set_title(title, fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Regime map (medians, 15 paired trials, budget 3000; * = Holm p < 0.05)",
                 fontsize=10, y=1.02)
    save(fig, "F3_regime_map.png")


def fig_f4_amplitude_law(amp: Dict):
    """F4: a=0.5 absolute vs √D-normalized, normalized advantage per cell
    (v20 core, Parte B) with Holm stars."""
    dims = [5, 10, 20]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.8 / len(dims)
    xlabels = []
    for i, f in enumerate(FUNCS):
        for j, d in enumerate(dims):
            key = f"{f}_{d}d"
            if key not in amp:
                continue
            abs_v, dn_v, p = amp[key]
            den = abs(abs_v) + abs(dn_v)
            r = (abs_v - dn_v) / den if den else 0.0
            x = i + (j - 1) * width
            color = {5: "#bbbbbb", 10: "#4C72B0", 20: "#d62728"}[d]
            ax.bar(x, r, width=width * 0.92, color=color,
                   edgecolor="k" if p < 0.05 else "none", linewidth=0.9)
        xlabels.append(f.replace("_", " "))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(FUNCS)), xlabels, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("(abs − √D-norm) / (|abs|+|dnorm|)", fontsize=9)
    ax.set_title("F4. Effect of the √D amplitude law (v20 core, a = 0.5; black edge = Holm p < 0.05)",
                 fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#bbbbbb", "#4C72B0", "#d62728")]
    ax.legend(handles, ["D=5 (≡)", "D=10", "D=20"], fontsize=8, frameon=False, ncol=1,
              loc="center left", bbox_to_anchor=(1.005, 0.5), borderaxespad=0.0)
    ax.set_ylim(-0.15, 1.1)
    style_ax(ax)
    save(fig, "F4_amplitude_law.png")


def main():
    print("[make_paper_figures] parsing committed reports…")
    p1 = load_phase1_baselines()
    p2 = load_phase2_baselines()
    ab = load_noise_ablation()
    amp = load_amp_sensitivity()
    print(f"  cells: phase1={len(p1)} phase2={list(p2)} ablation={len(ab)} amp={len(amp)}")
    fig_f1_budget_crossover(p2)
    fig_f2_baselines(p1)
    fig_f3_regime_map(ab)
    fig_f4_amplitude_law(amp)
    print(f"[done] figures in {OUT.relative_to(ROOT)}/ (F5_coverage_2d5d.png via docs/audit_2026/v6_cobertura_2d5d.py)")


if __name__ == "__main__":
    main()
