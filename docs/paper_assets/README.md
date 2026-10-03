# Assets del paper — figuras F1–F5 y su procedencia

Generadas con `python -m benchmarks.make_paper_figures`, que **parsea los informes commiteados**
en `docs/audit_2026/*.md` (los `results/*.json` crudos son regenerables pero están gitignored).
Cada figura es por tanto trazable a una tabla versionada — si los informes cambian, la figura se
regenera con el mismo comando. Etiquetas en inglés (el paper va a venue internacional).

| Figura | Fichero | Claim (PAPER_BLUEPRINT) | Fuente (tabla) | Lectura |
|---|---|---|---|---|
| **F1** | `F1_budget_crossover.png` | C6/C11 (crossover de presupuestos) | `informe_baselines_v24.md` §Fase 2 | CMA-ES básico se congela ("infarto"), Seismic mejora lento, **IPOP converge siempre** — la figura del lenguaje preciso |
| **F2** | `F2_baselines_panels.png` | mapa honesto vs baselines | `informe_baselines_v24.md` §Fase 1 | (a) BFGS-MS/IPOP dominan a budget 3000; (b) v24 vs IPOP celda a celda con significancia (puntos grandes = Holm p<0.05) |
| **F3** | `F3_regime_map.png` | **C3/C4 — la figura central** | `informe_ablacion_ruido.md` | (a) perturbar ayuda (verde) SOLO en multimodales deceptivos, perjudica (rojo) en uni/valle; (b) correlacionado vs blanco equiparado: verde en no-separable D-alta — motiva el detector de régimen |
| **F4** | `F4_amplitude_law.png` | **C5 — ley √D** | `informe_ablacion_componentes.md` §Parte B | La corrección √D nunca empeora en el núcleo v20 y gana con significancia en 10+ celdas D≥10 (borde negro) | **nueva: `../appendix_A_ley_amplitud_raiz_d.md` (derivación)**
| **F5** (apéndice) | `F5_coverage_2d5d.png` (generada con `docs/audit_2026/v6_cobertura_2d5d.py`) | C1 **refinado** | `informe_cobertura_2d5d.md` | (a)(b) cobertura vs tiempo 2D/5D: sísmico **crece sin saturar** (81%/43% a 50k) ≫ GD congelado (16%/0.2%); blanco satura ~99% antes (cubrimiento bruto no es el rasgo); (c) autocorrelación: GD≈1 (atrapamiento), sísmico decae a 0 en ~5000 pasos, blanco en ~100 (el terremoto coherente explora con estructura); (d) densidad de ocupación 2D: GD = puntos fijos brillantes |

## Estado

- F1–F4: ✅ generadas 2026-10-03 y verificadas visualmente.
- F5: ✅ generada 2026-10-03 con `docs/audit_2026/v6_cobertura_2d5d.py` (8 seeds, 50k pasos,
  rejillas comparables 32×32 / 4^5; veredicto honesto: cobertura creciente no completa,
  coherencia temporal como rasgo distintivo). ⚠️ NO usar `assets/laplacian_ergodicity.png`
  (retractado).
