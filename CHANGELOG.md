# Changelog

Todos los cambios relevantes del proyecto se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [0.24.1] - 2026-10-03

### Corregido
- **CI rojo en 4 jobs** (push a `main`, run #2):
  - `benchmark smoke`: el workflow instalaba `cma scipy` pero
    `benchmarks/experiment_champion_v23.py` importa `matplotlib` →
    `ModuleNotFoundError`. Ahora instala `cma scipy matplotlib`.
  - `pytest py3.11 / py3.12 (no torch) / py3.12 (torch CPU)` (y posteriormente
    toda la matriz): fallaba `test_v23_rastrigin_5d_orf_golden`. Causas y arreglo:
    1. `seismic_descent.orf` usaba `np.linalg.qr`, cuyo resultado varía ±1e-15
       (ulp) según la build de OpenBLAS/LAPACK, y la dinámica amplifica ese ruido
       a O(1) (medido: Δz=1.3e-15 → Δfinal≈3.4). Sustituido por una Gram-Schmidt
       modificada determinista (solo aritmética elementwise + `np.sum`, sin
       BLAS/LAPACK; equivalente al QR con `diag(R) > 0`, misma distribución de
       Haar). Con ello los 4 jobs de la matriz CI pasaron a dar un valor idéntico
       entre numpy 2.0.2/2.2/2.5 (verificado en CI).
    2. Queda ruido de ulp propio de plataforma (libm/BLAS Win vs Linux, medido
       con un contenedor que reproducía el CI bit a bit). La trayectoria
       ORF+Rastrigin es caótica y lo amplifica a O(1) (Win 2.1586 vs Linux
       10.4040), por lo que el golden ORF se movió a **Sphere 5D** (mismo patrón
       que el golden RFF): divergencia Win↔Linux de solo ~1e-14 relativa tras
       300 pasos, con 5 órdenes de margen bajo `_RTOL=1e-9`.
  - Tests nuevos: `test_orf_construction_hash` (fija el digest SHA-256 de la
    construcción ORF, pasó en toda la matriz CI) y
    `test_mgs_matches_sign_fixed_qr` (equivalencia con el QR histórico).

## [0.24.0] - 2026-10-03

### Añadido
- **Configuración candidata v24** (`seismic_champion_v24` / flag opt-in
  `noise_amp_dim_normalized=True` en `SeismicChampionV23`): amplitud normalizada por dimensión
  `amp·√(5/D)` derivada de la ley medida E‖∇ruido‖∝amp·√D/ℓ. Default `False`: comportamiento
  v23 preservado bit a bit (verificado por golden tests). Ver `docs/findings_v24_amplitud_raiz_d.md`.
- `benchmarks/benchmark_v24_baselines.py`: benchmark comparativo (10 algoritmos: v20, v23,
  v24 ref5, **v24 ref1**, v25c experimental, CMA-ES, IPOP-CMA-ES, PSO, L-BFGS-B multistart,
  random) en 12 funciones × {5,10,20}D × 31 trials pareados, Wilcoxon+Holm por celda y
  fase de crossover a 25k (informe: `docs/audit_2026/informe_baselines_v24.md`).
- Tests V24: equivalencia default-off, escala exacta √(5/D), mejora en mediana en Trid 20D.
- `docs/PAPER_BLUEPRINT.md`: matriz claims↔evidencia y estructura del paper.
- Figuras del paper F1–F5 en `docs/paper_assets/` (generadores: `benchmarks/make_paper_figures.py`
  para F1–F4; `docs/audit_2026/v6_cobertura_2d5d.py` para F5) con matriz figura↔claim↔tabla en
  `docs/paper_assets/README.md`.
- Experimento **v25 (amplitud adaptativa por régimen)**: `benchmarks/experiment_v25_adaptive_amp.py`
  (coherencia de gradientes / estancamiento; fuera del paquete). Veredicto en
  `docs/findings_v25_amplitud_adaptativa.md`: recomendado v24 con `ref_dim=1`
  (parámetro opt-in de `SeismicChampionV23`), v25c robusta segunda, v25s rechazada.
- Parámetro opt-in `ref_dim` (default 5.0) en `SeismicChampionV23`, validado por split
  train/test (`benchmarks/experiment_refdim_train_test.py` → `informe_refdim_train_test.md`).
- `docs/appendix_A_ley_amplitud_raiz_d.md`: derivación formal de la ley √D (lema
  E‖∇n‖² = amp²·D/ℓ² exacto para RFF/ORF e independiente de r; ratio ruido/señal
  ρ = (amp/ℓ)√D sobre la dinámica direccional; unicidad del exponente 1/2; caso Lissajous;
  verificación numérica reproducible con el motor real).

### Corregido
- Divisiones por cero silenciosas al normalizar gradientes nulos (`np.divide(..., where=...)`
  en los 4 optimizadores; comportamiento numérico idéntico, sin RuntimeWarnings).

## [0.23.1] - 2026-10-03

### Corregido (crítico)
- `functions.py`: el paquete no se podía importar (`NameError: Union` no importado). Rompía la instalación y toda la suite de tests en cualquier entorno.
- `torch_optimizer.py`: el fallback sin PyTorch fallaba con `NameError` en `@torch.no_grad()` (debía ser `ImportError` para que el `try/except` de `__init__.py` lo capturara). Nuevo decorador `_NO_GRAD` con fallback idempotente.

### Añadido
- `LICENSE` (MIT), `CITATION.cff`.
- CI con GitHub Actions (tests en Python 3.9–3.12, con y sin torch CPU).
- Tests: gradientes numéricos de las funciones objetivo, golden tests de regresión (v20/v23 con semilla fija), casos borde (D=1, N=1, bounds asimétricos).
- `benchmarks/baselines.py`: baselines con semilla: CMA-ES básico, IPOP-CMA-ES (reinicios con población creciente), PSO, BFGS multistart (L-BFGS-B con gradiente analítico), búsqueda aleatoria.
- `benchmarks/stats.py`: Wilcoxon signed-rank pareado + corrección de Holm.
- `benchmarks/experiment_ablation_noise.py`: ablación formal de la hipótesis central (ruido correlacionado vs blanco equiparado vs apagado) con estadística.
- `benchmarks/experiment_component_ablation.py`: leave-one-out de los componentes del champion v23 + sensibilidad a hiperparámetros + corrección de escala √D.
- `docs/audit_2026/`: auditoría verificada empíricamente + scripts de verificación reproducibles.
- `docs/PROVENANCE.md`: declaración de desarrollo asistido por agentes.
- `functions_extended.py`: Levy, Michalewicz, Zakharov, Styblinski-Tang, Dixon-Price, Trid
  (gradientes verificados numéricamente en `tests/test_functions_gradients.py`).

### Cambiado
- Versión unificada a `0.23.1` en `pyproject.toml` y `__init__.py` (antes 0.20.0 / 0.23.0).
- `GEMINI.md`: matizada la "Regla de Oro" (protege trazabilidad experimental, no prohíbe mantenimiento).
- `README.md`, `docs/theory.md`: retiradas o matizadas las afirmaciones no verificadas empíricamente (ergodicidad Laplaciana, tabla MNIST preliminar); añadida sección de limitaciones y de reproducción.
- Benchmark scripts: baselines con semilla explícita; reportes con desviación y tests de significancia.

## [0.23.0] - 2026-10-03
- Arquitectura Champion v23 (ORF/Lissajous + gravedad + momentum + métrica anisotrópa).

## [0.20.0] - 2026-10-03
- Arquitectura base v20 (normalización a [-1,1]^D, gradiente L2, dt cíclico).
