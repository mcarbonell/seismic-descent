"""experiment_refdim_train_test.py — Is ref_dim=5 a tuned choice or a robust law?

The v24 amplitude correction is amp·sqrt(ref_dim/D) with ref_dim = 5, chosen a priori
(D=5 was the historical reference configuration — it keeps v24 ≡ v23 at D=5).
A reviewer will ask: was 5 *selected on the benchmark*? This experiment answers with a
train/test split by function family (mirroring COCO/BBOB methodology):

- Candidate values ref_dim ∈ {2, 3.5, 5, 7.5, 10} → selection ONLY on TRAIN functions.
- Frozen winner evaluated on TEST functions (never used for selection).
- Report: mean log10-ratio vs ref_dim=5 per candidate on train and test, per-cell
  medians, paired Wilcoxon (winner vs ref_dim=5) with Holm over the test cells.

Amplitude parameterization: SeismicChampionV23(noise_amp_dim_normalized=True,
noise_amplitude=0.5·sqrt(ref/5))  ⇒  effective amplitude 0.5·sqrt(ref/D).

Outputs: results/refdim_train_test.json + docs/audit_2026/informe_refdim_train_test.md
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
TRAIN_FUNCS = ["ackley", "dixon_price", "griewank", "michalewicz", "styblinski_tang", "zakharov"]
TEST_FUNCS = ["levy", "rastrigin", "rosenbrock", "schwefel", "sphere", "trid"]
REFS = [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0]
N_PARTICLES = 10


def variant_name(ref: float) -> str:
    return "v23_abs" if ref < 0 else f"ref{ref:g}"


def run_cell(fname: str, dim: int, budget: int, seed: int, ref: float) -> float:
    fcfg = SUITE[fname]
    r = get_search_range(fcfg, dim)
    bounds = np.array([[-r, r]] * dim)
    n_steps = max(20, budget // N_PARTICLES)
    if ref < 0:  # historical v23: absolute amplitude 0.5
        opt = SeismicChampionV23(bounds=bounds, n_particles=N_PARTICLES, n_steps=n_steps,
                                 noise_engine="orf", seed=seed)
    else:
        opt = SeismicChampionV23(
            bounds=bounds, n_particles=N_PARTICLES, n_steps=n_steps, noise_engine="orf",
            noise_amplitude=0.5 * np.sqrt(ref / 5.0),
            noise_amp_dim_normalized=True, seed=seed)
    _, val, _ = opt.optimize(fn=fcfg["fn"], fn_grad=fcfg["grad"])
    return float(val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--quick", action="store_true", help="2 funcs/set, dims 5/10, 3 trials")
    args = ap.parse_args()

    train, test = TRAIN_FUNCS, TEST_FUNCS
    if args.quick:
        train, test, args.dims, args.trials = train[:2], test[:2], [5, 10], 3

    variants = [-1.0] + REFS  # v23 absolute amplitude + ref candidates
    cells: Dict[str, Dict[str, List[float]]] = {}
    t0 = time.time()
    for split, fnames in (("train", train), ("test", test)):
        for fname in fnames:
            for dim in args.dims:
                key = f"{split}/{fname}_{dim}d"
                cells[key] = {variant_name(v): [] for v in variants}
                for trial in range(args.trials):
                    x0rel = np.random.default_rng(5000 + trial).uniform(-0.8, 0.8, size=dim)
                    for v in variants:
                        cells[key][variant_name(v)].append(
                            run_cell(fname, dim, args.budget, trial + 1, v))
                meds = {vn: float(np.median(vals)) for vn, vals in cells[key].items()}
                elapsed = time.time() - t0
                print(f"[{key}] " + " | ".join(f"{k}={m:.4g}" for k, m in meds.items())
                      + f"  ({elapsed:.0f}s)", flush=True)

    # ---- selection on TRAIN: mean log10 ratio vs ref5
    def medians(split: str) -> Dict[str, Dict[str, float]]:
        return {k: {vn: float(np.median(v)) for vn, v in cells[k].items() if isinstance(v, list)}
                for k in cells if k.startswith(split + "/")}

    def ranks(split: str) -> Dict[str, float]:
        """Mean rank of each variant across cells (1 = best in that cell).
        Scale-free and robust to negative objective values (Trid, Styblinski)."""
        meds = medians(split)
        acc = {variant_name(v): [] for v in variants}
        for k in meds:
            order = sorted(meds[k], key=lambda vn: meds[k][vn])
            for rank, vn in enumerate(order, 1):
                acc[vn].append(rank)
        return {vn: float(np.mean(rs)) for vn, rs in acc.items()}

    def score(split: str, vn: str) -> float:
        """Mean normalized advantage vs ref5: (ref5 - ref)/(|ref5|+|ref|), in [-1, 1]."""
        meds = medians(split)
        ds = []
        for k in meds:
            a, b = meds[k]["ref5"], meds[k][vn]
            den = abs(a) + abs(b)
            ds.append((a - b) / den if den else 0.0)
        return float(np.mean(ds))

    train_ranks = ranks("train")
    train_scores = {variant_name(v): score("train", variant_name(v)) for v in variants}
    # selection: ONLY the v24-family candidates (refs), by mean rank on train
    winner = min(REFS, key=lambda r: train_ranks[f"ref{r:g}"])
    print(f"\n[train] mean log10 ratios vs ref5: {train_scores}")
    print(f"[train] selected ref_dim* = {winner:g}")

    # ---- paired Wilcoxon on TEST: winner vs ref5 and winner vs v23_abs
    test_keys = [k for k in cells if k.startswith("test/")]
    for ref_tag in (f"ref{winner:g}",):
        for other in ("ref5", "v23_abs"):
            pvals = []
            for k in test_keys:
                pv, _ = wilcoxon_paired(cells[k][ref_tag], cells[k][other])
                pvals.append(pv)
            adj = holm_adjust(pvals)
            for k, pv, pa in zip(test_keys, pvals, adj):
                cells[k][f"holm_{ref_tag}_vs_{other}"] = (float(pa) if pa == pa else None)

    out = {"meta": {**vars(args), "train": train, "test": test, "refs": REFS,
                    "winner_refdim": winner, "train_scores": train_scores},
           "cells": {k: {vn: vals for vn, vals in v.items() if isinstance(vals, list)}
                     for k, v in cells.items()}}
    Path("results").mkdir(exist_ok=True)
    with open("results/refdim_train_test.json", "w") as f:
        json.dump(out, f, indent=1)

    # ---- report
    lines = ["# ref_dim train/test split (¿es 5 un valor tunado o una ley robusta?)", ""]
    lines.append(f"> Trials: {args.trials} | Budget: {args.budget} | dims {args.dims} | "
                 f"selección SOLO en train {args.__dict__ and train}.")
    lines.append("")
    lines.append("## Selección en TRAIN (rank medio por celda — scale-free, robusto a valores negativos; "
                 "1 = mejor en todas)")
    lines.append("")
    lines.append("| variante | rank medio train | ventaja norm. vs ref5 |")
    lines.append("|---|---|---|")
    for vn in sorted(train_ranks, key=lambda v: train_ranks[v]):
        lines.append(f"| {vn} | {train_ranks[vn]:.2f} | {train_scores[vn]:+.4f} |")
    lines.append(f"\n**ref_dim\\* seleccionado en train (por rank): {winner:g}**\n")
    lines.append("## Evaluación en TEST (medianas; estrellas = Wilcoxon pareado Holm vs ref5)")
    lines.append("")
    header = "| celda | v23 abs | " + " | ".join(f"ref{r:g}" for r in REFS) + " |"
    lines.append(header); lines.append("|---" * (len(REFS) + 2) + "|")
    for k in test_keys:
        meds = medians("test")[k]
        short = k.split("/")[1]
        row = f"| {short} | {meds['v23_abs']:.4g} |"
        for r in REFS:
            vn = f"ref{r:g}"
            pv = cells[k].get(f"holm_ref{winner:g}_vs_ref5") if r == winner else None
            star = significance_stars(pv) if pv else ""
            row += f" {meds[vn]:.4g}{star} |"
        lines.append(row)
    lines.append("")
    test_scores = None  # computed below with ranks
    lines.append("## Scores TEST (no usados para selección)")
    test_ranks = ranks("test")
    test_scores = {variant_name(v): score("test", variant_name(v)) for v in variants}
    lines.append("| variante | rank medio test | ventaja norm. vs ref5 |")
    lines.append("|---|---|---|")
    for vn in sorted(test_ranks, key=lambda v: test_ranks[v]):
        lines.append(f"| {vn} | {test_ranks[vn]:.2f} | {test_scores[vn]:+.4f} |")
    Path("docs/audit_2026").mkdir(exist_ok=True)
    Path("docs/audit_2026/informe_refdim_train_test.md").write_text("\n".join(lines) + "\n")
    print(f"\n[+] results/refdim_train_test.json + docs/audit_2026/informe_refdim_train_test.md ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
