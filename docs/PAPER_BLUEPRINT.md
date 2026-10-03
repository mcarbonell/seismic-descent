# Blueprint del paper — Seismic Descent

**Estado:** 2026-10-03, tras la remediación P0/P1. Documento de planificación: qué se puede afirmar hoy con evidencia reproducible y qué queda por demostrar antes de enviar.

---

## 1. Contribución defendible (claims) y su evidencia actual

| # | Claim del paper | Evidencia | Estado |
|---|---|---|---|
| C1 | Deformar el paisaje con ruido espacialmente correlacionado + tracking produce **cobertura empírica creciente y muy superior a GD puro** (2D/5D: 81%/43% a 50k, sigue subiendo; GD congela en 16%/0.2%) y **coherencia temporal** del recorrido (autocorr a lag100 +0.92 vs +0.12 del blanco) — el rasgo distintivo real del terremoto; la "cobertura completa" 1D NO se extiende literal, y el blanco satura antes (cubrir rápido ≠ cubrir con estructura) | `v6_cobertura_2d5d.py` + `informe_cobertura_2d5d.md` (8 seeds, 50k pasos) | ✅ Verificado con refinamiento honesto (F5) — sin pretensión de ergodicidad formal |
| C2 | El método es un **optimizador anytime** O(N·D·r) por paso vs O(λ·D²) de CMA-ES → mucho menor coste por evaluación | Complejidad + tiempos medidos | ✅ Sencillo de escribir; medir tiempos por eval fijando hardware |
| C3 | El ruido **correlacionado no es peor que el blanco equiparado** en la gran mayoría de celdas, y lo supera significativamente en un subconjunto no separable + D alta (dixon_price_20d **, griewank_20d **, rosenbrock_20d *, zakharov_10d *; rastrigin_20d ns favorable) | `informe_ablacion_ruido.md` (Wilcoxon+Holm, 48 celdas) | ✅ Con matices: pierde marginalmente en celdas triviales de baja D (sphere_2d, michalewicz_2d). Dirección: correlación=estructura de valles útil; en trivial, da igual o estorba |
| C4 | **Mapa de regímenes**: perturbar (de cualquier forma) perjudica en paisajes casi unimodales (sphere/dixon/griewank-baja/trid: "nada" gana con ** ) y ayuda en multimodales deceptivos (levy, michalewicz, schwefel_5d, ackley_5d) — justifica a posteriori el detector de régimen de v23 | Misma ablación, columna rff vs nada | ✅ Hallazgo honesto que fortalece el paper: "saber cuándo temblar" es la contribución real |
| C5 | La **amplitud óptima escala como ~1/√D** (ley E‖∇ruido‖ ∝ a·√D/ℓ medida; corrección `a·√(ref/D)` mejora drásticamente en D=20; constante óptima ref≈1 validada con split train/test) | `informe_ablacion_componentes.md` Part B + `informe_refdim_train_test.md` | ✅ Fuerte + refinada: ref5 gana 21/24 vs v23; ref≈1 aún mejor (Trid 20D → óptimo global); inversión Levy/Schwefel → amplitud adaptativa (v25). Derivación formalizada ✅ → `appendix_A_ley_amplitud_raiz_d.md` |
| C6 | A presupuestos altos (≥25k evals) Seismic supera a **CMA-ES básico** en multimodales D-baja | `v3` y fase 2 de `benchmark_v24_baselines.py` | ⚠️ Parcial: IPOP-CMA-ES compite; reportar contra ambos (fase 2 lo incluye) |
| C7 | La arquitectura champion v23 mejora a la base v20; cada componente se justifica por ablación | `informe_ablacion_componentes.md` Part A | ✅ Con sorpresa honesta: `no_noise` gana en Ackley — explicar interacción ruido×gravedad |
| C8 | Determinismo reproducible: campos Lissajous inconmensurables (quasi-periodicidad) | Implementación + refs Kronecker-Weyl | ✅ Teórico menor; aplicar SOLO a Lissajous (nunca a RFF) |
| C9 | ~~Ergodicidad Laplaciana / Boltzmann-Gibbs / colas pesadas~~ | — | ❌ **Retractado** (KS/AIC: kurtosis negativa). No citar |
| C10 | ~~Ventaja universal del ruido correlacionado sobre blanco~~ | Ablación | ❌ Falso en general; sustituir por C3/C4 (mapa de regímenes) |
| C11 | ~~"CMA-ES Infarction" general~~ | Fase 2 | ⚠️ Solo vs CMA-ES básico sin reinicios; IPOP se defiende. Usar lenguaje preciso |
| C12 | Integración PyTorch / Deep Learning | `v4` | ❌ Prototipo (12 GB ResNet-18; sin best-tracking; schedule descalibrado). Futuro trabajo |
| C13 | MNIST supera a Adam | — | ❌ Ruido estadístico (una semilla). Retirar |
| C14 | **v25 (amplitud adaptativa por régimen)**: el gate de coherencia de gradientes detecta valles (apaga el temblor como ref1) sin perder competitividad en el resto; en esta suite el constante ref1 informado por split sigue ganando (rank 1.58 vs 2.36/3.22). v25c = robusta segunda; v25s (estancamiento) rechazada | `informe_v25_adaptativa.md` + veredicto en `findings_v25_amplitud_adaptativa.md` | ✅ Probado honestamente: mecanismo correcto, ventaja aún no demostrada en suite homogénea; su caso de uso es paisaje desconocido/mezclado (landscapes compuestos, estrés de régimen) |

## 2. Venue y formato recomendados

1. **Objetivo primario (realista):** workshop de optimización NeurIPS/ICML (p. ej. *OPT — Optimization for Machine Learning*) o **GECCO** (workshop o track de algoritmos). 6–8 páginas + apéndice.
2. **Stretch:** IEEE CEC (track de single-objective continuous optimization) exigiendo suite más amplia y posiblemente BBOB/COCO.
3. **Journal (a más largo plazo):** IEEE TEVC / Swarm and Evolutionary Computation — requiere COCO completo (24 funciones BBOB, `{2,3,5,10,20,40}D`) y análisis teórico de tiempos de escape.

## 3. Estructura propuesta (paper workshop, 8 pp)

1. **Introduction** — problema (escape de mínimos locales en primer orden), idea (terremoto correlacionado), contribuciones (C1, C3, C4, C5 + anytime O(N·D)).
2. **Related Work** — Energy Landscape Paving (Hansmann & Wille 2002); homotopía/suavizado Gaussiano y continuación (Moré & Wu; Mobahi & Fisher 2015); SGLD/Welling & Teh; PSO (Kennedy & Eberhart — la "gravedad" es best-only PSO modulado); RMSProp/Adam (precondicionado diagonal); CMA-ES/IPOP (baselines); RFF (Rahimi & Recht 2007), ORF (Yu et al. 2016); Kronecker-Weyl (solo Lissajous).
3. **Method** — normalización isotrópica, gradiente L2, campo RFF/ORF con gradiente analítico, schedule oscilatorio con `dt_floor`, tracking del mejor punto; pseudocódigo (Algoritmo 1); complejidad.
4. **Theory (modesta y verificable)** — cobertura/ergodicidad de cobertura; ley de escala √D del ratio ruido/señal con su corrección; quasi-periodicidad del campo Lissajous. *Sin termodinámica.*
5. **Experiments** — (a) ablación central correlacionado/blanco/apagado (12 funciones, 4 dims, Wilcoxon+Holm) → mapa de regímenes; (b) leave-one-out v23; (c) sensibilidad y corrección de amplitud; (d) comparativa vs baselines sembrados (CMA-ES, IPOP-CMA-ES, PSO, L-BFGS-B multistart); (e) crossover de presupuestos vs CMA-ES básico **e** IPOP.
6. **Limitations** — gradiente analítico, penalización en unimodales, degrado en D alta (motivando C5), sensibilidad a hiperparámetros.
7. **Conclusion** — + reproducibilidad (repo, CI, golden tests, DOI de Zenodo si se archiva).

### Figuras propuestas (todas generables hoy)

> **✅ F1–F4 YA GENERADAS** (2026-10-03) en `docs/paper_assets/` vía
> `python -m benchmarks.make_paper_figures` (parsea los informes commiteados — ver
> `docs/paper_assets/README.md` para trazabilidad figura↔tabla). F5 reutiliza el asset 1D.
- F1: ilustración 1D de la deformación del paisaje (visualizador) — *solo* como intuición, con caption que aclare que es pedagógica.
- F2: curvas de convergencia champion vs baselines (ya existe: `experiment_champion_v23`).
- F3: **heatmap del mapa de regímenes** (función × dimensión × ganancia rff−blanco y rff−nada, con estrellas de significancia) — la figura firma del paper.
- F4: curva de sensibilidad amp×D con/sin corrección √D.
- F5 (apéndice): cobertura empírica y decaimiento de autocorrelación (C1).

## 4. Trabajo pendiente antes de enviar (checklist)

- [x] Benchmark v24 completo (36 celdas, incluye seismic_v24) + fase 2 con IPOP a 25k → `informe_baselines_v24.md` (✅ 2026-10-03).
- [x] Ampliar C1 a 2D/5D (✅ 2026-10-03: `v6_cobertura_2d5d.py`, 8 seeds × 50k pasos, rejillas comparables; cobertura creciente ≫ GD, coherencia temporal cuantificada → F5).
- [x] **v24 = v23 + corrección √D** implementada (flag opt-in en `SeismicChampionV23` + `seismic_champion_v24`; v23 intacto bit a bit) y validada (21/24) → `findings_v24_amplitud_raiz_d.md` (✅ 2026-10-03).
- [x] Grid de `ref_dim` con split train/test → `informe_refdim_train_test.md` (✅ 2026-10-03: óptimo ≈1, inversión Levy). Pendiente opcional: grid conjunto con `dt_base`/`gravity_strength`.
- [x] ≥31 trials en las tablas finales (✅ 2026-10-03: las 5 tablas canónicas — baselines, v25, ref_dim, ablaciones ruido y componentes — regeneradas a 31 trials; figuras F1–F4 regeneradas; ley √D reforzada, p≈9.3e-10 en Parte B).
- [x] Derivación en apéndice de la ley de amplitud √D → `docs/appendix_A_ley_amplitud_raiz_d.md` (✅ 2026-10-03: lema E‖∇n‖² = amp²D/ℓ² exacto para RFF y ORF, unicidad del exponente 1/2 por la dinámica direccional, verificación numérica con el motor real; figura asociada F4).
- [ ] Si se apunta a CEC/journal: adaptador COCO/BBOB.
- [ ] Package de reproducibilidad: `uv.lock`/Dockerfile + tag `v0.24.0` + Zenodo DOI.

## 5. Riesgos residuales y mitigaciones

| Riesgo | Mitigación |
|---|---|
| "No es mejor que IPOP-CMA-ES" en multimodales | Posicionar el valor en *simplicidad/coste* (O(ND) sin matrices), anytime y robustez de escape, no en superioridad universal; el mapa de regímenes es la contribución |
| Sobreajuste de hiperparámetros a la suite | Tuning con split funciones train/test (ver checklist) |
| Revisores piden COCO | Tener el adaptador BBOB listo (objetivo stretch) |
| Independencia de seeds entre modos | Ya pareado por semilla en todos los experimentos nuevos |
