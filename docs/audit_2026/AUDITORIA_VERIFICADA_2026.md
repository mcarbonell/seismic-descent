# Auditoría Verificada del Repositorio Seismic Descent

**Fecha:** 2026-10-03 · **Alcance:** código, tests, benchmarks, documentación, idea, novedad y viabilidad de publicación.
**Método:** auditoría con **verificación empírica** — todas las afirmaciones comprobables se ejecutaron contra el código real (scripts reproducibles en `docs/audit_2026/v1–v5`).
**Complementa:** a `docs/AUDIT.md` (auditoría previa estática, misma fecha). Aquí se confirman, corrigen y amplían sus hallazgos con evidencia experimental.

---

## 0. Veredicto ejecutivo

**La idea es original, publicable y con una metáfora potente; el repositorio NO está listo para sustentar un paper hoy.**

- ✅ **El algoritmo funciona y sus números del README son aproximadamente reproducibles** (ver §7).
- ✅ Los gradientes analíticos de las 6 funciones de benchmark son **correctos** (verificados numéricamente, y *no* estaban cubiertos por tests).
- 🔴 **En HEAD el paquete no se podía importar** (`NameError`) y los "26 tests 100% passing" eran falsos en cualquier entorno. *(Corregido en esta sesión: 2 líneas.)*
- 🔴 **La "Ergodicidad Laplaciana" —el pilar teórico del proyecto— está refutada empíricamente** por el propio código del repo (tests KS/AIC, §3.2).
- 🔴 **La ventaja del ingrediente diferencial del paper (ruido espacialmente correlacionado) es inconsistente** frente a ruido blanco de potencia equiparada (§3.3) y en Rastrigin 10D "sin ruido" gana al ruido sísmico.
- 🟠 El benchmark no es estadísticamente riguroso ni libre de sesgos (CMA-ES/SA sin semilla, coste de gradiente ignorado, CMA-ES básico sin reinicios).
- 🟠 La teoría de `docs/theory.md` es cualitativa y contiene la analogía termodinámica incorrecta (ya señalado en AUDIT.md, ahora además con refutación empírica).

**Recomendación:** el venue realista es un workshop (NeurIPS/ICML optimization & generalization) o GECCO/IEEE CEC tras el plan de §9. Un journal tipo IEEE TEVC exigiría benchmark COCO completo + teoría. La prioridad absoluta es **rehacer la narrativa central**: el paper no puede defenderse sobre la "ergodicidad laplaciana" (falsa empíricamente) sino sobre lo que sí es verificable: *un método ligero O(N·D) que mantiene escape continuo y cruza a CMA-ES básico en presupuestos altos*.

---

## 1. Metodología y entorno de esta auditoría

| Ítem | Detalle |
|---|---|
| Entorno | Python 3.11.2, venv limpio, `numpy 2.4.6`, `cma 4.5.0`, `scipy 1.17.1`. Sin torch (no disponible para esta plataforma; su análisis es estático/cuántico). |
| Instalación | `pip install -e .` sobre HEAD (`214f555`) + 2 fixes aplicados en esta sesión (§2.1). |
| Scripts de verificación | `docs/audit_2026/v1_gradientes_y_ruido.py` (gradientes + ratio ruido/señal) · `v2_ergodicidad.py` (histogramas + KS/AIC + cobertura) · `v3_replicacion_benchmark.py` (tablas README) · `v4_torch_estatico.py` (escalabilidad) · `v5_ablacion_correlado_vs_blanco.py` (hipótesis central). |
| Estado de tests tras fixes | **25 passed + 1 skipped** (el de torch, correctamente saltado). Concuerda con los "26" del README solo tras los fixes. |

---

## 2. Hallazgos bloqueantes (bloquean publicación / releases)

### 2.1 🔴 En HEAD el paquete no importa. Los "26 tests passing" eran falsos

Verificado: `python -c "import seismic_descent"` en HEAD falla — en **cualquier** entorno:

1. `src/seismic_descent/functions.py:7` usa `Union[...]` en anotaciones sin importarlo (`from typing import Dict, Any`). Las anotaciones se evalúan en tiempo de definición → `NameError: name 'Union' is not defined`. Rompe todo con o sin torch.
2. `src/seismic_descent/torch_optimizer.py:101`: sin torch instalado, el fallback `Optimizer = object` permite definir la clase, pero el decorador `@torch.no_grad()` explota con `NameError` (no `ImportError`, así que el `try/except ImportError` de `__init__.py` no lo atrapa).

**Fix aplicado en esta sesión (2 cambios mínimos, sin tocar lógica algorítmica):**
- `functions.py`: añadido `Union` al import de typing.
- `torch_optimizer.py`: decorador `_NO_GRAD` con fallback idempotente cuando torch no está; con torch el comportamiento es idéntico (`torch.no_grad()`).

Resultado: `25 passed, 1 skipped`. Nota para el proceso: esto demuestra que **ningún CI corre los tests** — no existe `.github/workflows`. Para un repo que sustenta un paper es inaceptable; añadir CI es P0 (coste: una tarde).

⚠️ *Señal de proceso:* la "Regla de Oro" de `GEMINI.md` ("jamás tocar un .py existente") es la raíz de este tipo de fallos: 42 archivos legacy duplicados sin refactor, y un bug trivial rompiendo el paquete sin que nadie lo detecte. Recomiendo eliminar esa regla para el código del paquete (`src/`) y reservarla, si se quiere, como convención histórica para `legacy/` únicamente.

### 2.2 🟠 Versión incoherente entre `pyproject.toml` y el paquete

`pyproject.toml` dice `0.20.0`; `__init__.py` dice `0.23.0`. `pip show` reporta 0.20.0 mientras `seismic_descent.__version__` reporta 0.23.0. Para un artefacto citado en un paper, `pip show` y `__version__` deben coincidir. (Ya señalado en AUDIT.md; sigue sin resolver.)

---

## 3. Verificación empírica de las afirmaciones centrales

### 3.1 ✅ Los gradientes analíticos son correctos (y no estaban testeados)

`v1_gradientes_y_ruido.py` compara los 6 gradientes de `functions.py` contra diferencias finitas centrales en 50 puntos aleatorios por función (D=2..7):

| Función | Error relativo máx. | Veredicto |
|---|---|---|
| sphere, rastrigin, rosenbrock | ≤ 5·10⁻¹⁰ | Correcto |
| ackley | 2.7·10⁻⁹ | Correcto |
| schwefel | 5.9·10⁻⁸ (incl. esquina x<0) | Correcto |
| griewank (truco cumprod) | 7.3·10⁻⁸ | Correcto |

Los tests actuales solo verifican los gradientes de los campos de ruido (RFF/ORF/Lissajous), **nunca los de las funciones objetivo** — que es donde un error contamina todos los benchmarks. Añadir estos tests es P0 (es gratis: ya están escritos en v1).

### 3.2 🔴 "Ergodicidad Laplaciana": REFUTADA empíricamente con el código del repo

`docs/theory.md` y el README afirman que el histograma ergódico converge a picos **Laplacianos** `e^{-|x|}` en los mínimos, con "colas pesadas" y "emulación Boltzmann-Gibbs". `v2_ergodicidad.py` ejecuta la dinámica **real del paquete** (`SeismicSwarm` v20, Rastrigin 1D, 20k pasos, N=1 y N=10, 2 semillas) plegando la trayectoria sobre la cuenca más cercana (los mínimos de Rastrigin 1D están en los enteros):

| Config | Kurtosis exceso | KS Normal | KS Laplace | ΔAIC (menor=mejor) | Cobertura rejilla |
|---|---|---|---|---|---|
| N=1, seed=1 | **−0.85** | 0.061 | 0.101 | Normal gana por 3758 | 100% |
| N=1, seed=2 | −0.61 | 0.088 | 0.094 | Normal gana por 2872 | 100% |
| N=10, seed=1 | −0.82 | 0.070 | 0.104 | Normal gana por 36709 | 100% |
| N=10, seed=2 | −0.58 | 0.091 | 0.095 | Normal gana por 27656 | 100% |

- Una Laplaciana requiere **kurtosis exceso = +3**; medimos **kurtosis negativa** (−0.6…−0.85), propia de distribuciones de cola *ligera* (sub-gaussianas). La dirección del efecto es la **opuesta** a la afirmada.
- La ajuste Normal domina a Laplace en KS y AIC en las 4 configuraciones.
- **Lo que SÍ se confirma:** cobertura del dominio 100% y autocorrelación decreciente — es decir, ergodicidad en el sentido débil de *coverage* (el enjambre barre el espacio), que es la propiedad que el algoritmo realmente explota.

**¿De dónde salió la figura laplaciana?** Del visualizador JS (`visualizer/js/explorer_1d.js`), que **no implementa el algoritmo del paquete**: usa amplitud auto-adaptativa con control discreto (×1.02/paso, ×2/÷2 cada 50 pasos, inexistente en Python), opción *greedy* de aceptación tipo Metropolis, clipping de paso al 1% del dominio, gradiente sin normalizar L2 sumado antes de escalar, y un heatmap con *splat* Gaussiano. La figura `assets/laplacian_ergodicity.png` evidencia la dinámica del *visualizador*, no la del optimizador que se publica. Usarla en el paper sería mala praxis.

**Consecuencia para el paper:** retirar la teoría Laplaciana/Boltzmann-Gibbs/colas pesadas (o demostrarla primero: Fokker-Planck del proceso + bondad de ajuste). La historia defendible es: *"deformación coherente del paisaje + seguimiento del mejor punto real = ergodicidad de cobertura y escape continuo"*, verificable con métricas de cobertura y tiempos de escape como los de arriba.

### 3.3 🔴 Ablación de la hipótesis central: ¿importa que el ruido esté correlacionado?

La tesis del paper es que la **correlación espacial** diferencia al método de SA/SGLD. `v5_ablacion_correlado_vs_blanco.py` corre la dinámica v20 idéntica con tres perturbaciones: RFF (correlacionado), **ruido blanco** (dirección uniforme en la esfera, **misma potencia media** que el RFF, mismo schedule) y **sin ruido**. 5 trials, presupuesto 3000 evals:

| Caso | RFF correlacionado | Blanco equiparado | Sin ruido |
|---|---|---|---|
| Rastrigin 5D | **11.54** | 12.96 | 17.92 |
| Rastrigin 10D | 55.63 | 61.37 | **53.94** ⚠️ |
| Rosenbrock 5D | 4.60 | 3.45 | **3.06** ⚠️ |
| Rosenbrock 10D | 35.96 | 44.89 | **7.79** ⚠️ |

Lectura honesta (muestra pequeña, sin significancia estadística aún):
- En Rastrigin 5D la hipótesis **se sostiene**: correlacionado > blanco > nada.
- En Rastrigin 10D el mejor es **sin ruido**; el RFF apenas distingue del blanco.
- En Rosenbrock (unimodal con valle curvo) el terremoto **es contraproducente**: sin ruido gana por ~4.6× en 10D. Lo que rescata al champion v23 en Rosenbrock es la gravedad+momentum+precondicionador, no el ruido.

**Consecuencia:** la ablación "correlacionado vs blanco equiparado vs apagado" es el experimento más importante del paper y hoy **no existe en el repo** (las ablaciones documentadas quitan componentes, pero nunca comparan contra ruido blanco equiparado). Hasta que esa ablación amplia (≥15 funciones, ≥30 trials, Wilcoxon) demuestre cuándo y por qué gana la correlación, la afirmación diferencial del paper es vulnerable. Hay una respuesta probable en los datos: la ventaja de la correlación depende de la **multimodalidad del paisaje** y decrece con D (ver §3.4). Conjetura verificable y elegante si se demuestra.

### 3.4 🟠 Ley de escala no documentada: el ratio ruido/señal crece como √D

El README presenta `noise_amplitude=0.5` como hiperparámetro "universal" relativo al gradiente (norma 1 por construcción). Medición (`v1`, 200 puntos × 3 tiempos, RFF r=64, ℓ=0.4, amp=0.5):

| D | ‖∇ruido‖ medio (RFF) | ratio vs señal |
|---|---|---|
| 2 | 1.37 | 1.4× |
| 5 | 2.36 | 2.4× |
| 20 | 5.58 | 5.6× |

El ratio crece como **√D** (teóricamente `E‖∇n‖² = amp²·D/ℓ²` para el kernel Gaussiano; las medidas encajan con √D: ×1.72=√(5/2), ×2.37≈√(20/5)). Es decir, la perturbación **no es adimensional**: en 20D el terremoto domina 5.6× la señal objetivo, lo que explica la degradación en alta dimensión documentada en `summary_of_experiments.md` ("CMA-ES triunfa sobradamente en >10D/50D"). Si esta ley se formaliza (1 página de cálculo del paper) y se corrige (p. ej., `amp_eff = amp/√D`), se convierte en una contribución teórica *favorable* al método — hoy es un bug conceptual silencioso.

---

## 4. Replicación de los resultados del README

`v3_replicacion_benchmark.py` reproduce filas clave con el protocolo del repo (seeds `trial+1`, presupuesto contado en fn-evals, 10 partículas; 5 trials en vez de 15):

| Fila del README | README (15 tr) | Esta auditoría (5 tr) | ¿Se sostiene? |
|---|---|---|---|
| Rastrigin 5D — v20 | 9.206 | 13.75 | mismo orden |
| Rastrigin 5D — v23 ORF | 6.452 | 7.53 | ✅ sí |
| Rastrigin 5D — v23 Ortho-Lissajous | 30.63 | 35.40 | ✅ sí (malo, como dice el README) |
| Rastrigin 5D — CMA-ES | 2.984 | 5.97 | mismo orden (CMA gana a 3000) |
| Rosenbrock 5D — v20 | 4.602 | 4.602 | ✅ exacto |
| Rosenbrock 5D — v23 ORF | 3.122 | 4.68 | aproximado |
| Rosenbrock 5D — CMA-ES | 9.4e-15 | 1.0e-14 | ✅ |
| Rastrigin 10D — v23 ORF | 31.087 | 33.32 | ✅ |
| Rastrigin 10D — CMA-ES | 13.929 | 14.92 | ✅ |
| **Crossover a 25k** Rastrigin 5D | Sísmico 4.85 < CMA 6.96 | v23-ORF 4.52 < CMA 5.97 | ✅ **cualitativamente confirmado** |

**Conclusión:** las cifras del README son honestas (no infladas) y el evento más interesante —el crossover a presupuestos altos— se reproduce. Los "speedups CPU" (6–17×) dependen de máquina/versión (medí 4× a 3k; CMA-ES 4.5 mejora mucho en 2026); usar rangos, no cifras absolutas.

**Sesgos que un revisor detectará (del protocolo actual, corregibles):**
1. **CMA-ES y SA van con el RNG global sin semilla** (`run_cmaes` ignora el argumento `s`; `run_sa` usa `np.random` global) → las corridas de los baselines *no son reproducibles* mientras las de Seismic sí lo están.
2. **El presupuesto cuenta fn-evals pero ignora el coste del gradiente**: Seismic consume 1 gradiente analítico por partícula y paso (3000 gradientes en el champion); los baselines, cero. Hay que declararlo o cargar el coste como k·fn-evals.
3. **CMA-ES básico, sin reinicios.** La narrativa del "CMA-ES Infarction" colapsa contra IPOP/BIPOP-CMA-ES, que son el estándar real. Comparar contra `pycma` con reinicios y contra SHADE/L-SHADE es obligatorio.
4. `benchmark_budget_scaling.py` usa **v20** (no el champion v23) y el README no lo indica; el lector asume v23.
5. Tabla MNIST (97.90% vs Adam 97.79%): margen de +0.11 pp, **sin trials ni desviación** (un MLP MNIST típico varía ±0.1–0.2 pp entre semillas) y **no verificable aquí** (sin torch). Como está, es ruido estadístico presentado como victoria.

---

## 5. Auditoría de código (módulo por módulo)

| Módulo | Estado | Hallazgos |
|---|---|---|
| `functions.py` | ✅ tras fix | Gradientes correctos (§3.1). Bug de import (§2.1). Metadatos `global_min*` incompletos/no usados. |
| `rff.py` | ✅ | Correcto y limpio. Verificar en el paper que `phis~U(0,2π)`, `drifts~U(0.1,0.5)` son parámetros *ad hoc* (sin ablación). `n_octaves=1` por defecto contradice la narrativa "multiescala" del README (las octavas se implementan pero no se usan en el champion). |
| `orf.py` | ✅ | ORF de Yu et al. 2016 bien implementado (QR Haar + radios χ). Documentar que `r` se redondea a múltiplos de D (64→65 en D=5, 64→80 en D=20). Ojo: ORF solo reduce varianza de la *aproximación del kernel*; no mejora "exploración" per se (cosa que el README insinúa). |
| `lissajous.py` | ✅ | Gradientes correctos. Campo **determinista** (Kronecker-Weyl aplica aquí, no al RFF — la AUDIT.md tiene razón). La normalización `1/√(d·2)` es arbitraria (sin justificar). |
| `core.py` (v20) | ✅ | Lógica correcta (normalización, chain rule, tracking del mejor frente a f_original). Sin criterio de parada; `noise_decay=1.0` ⇒ sin convergencia por diseño (documentar como *anytime algorithm*, no como bug). |
| `swarm_gravity.py` / `anisotropic_hmc.py` | ✅ | Misma lógica que champion, duplicada 3 veces con convenios de signo distintos (frágil; refactor tras el paper). "HMC" es **heavy-ball amortiguado por fase**, no integración simpléctica hamiltoniana — renombrar ("phase-gated momentum") o un revisor lo marcará. El precondicionador anisótropo es funcionalmente RMSProp diagonal: citar Tieleman & Hinton / Adam, no presentarlo como novedad. |
| `champion_v23.py` | ✅ | Síntesis correcta. **Ningún hiperparámetro proviene de búsqueda formal** (0.5, 0.2, 0.4, 0.7, 0.5, β=0.9, 10 ciclos): el paper necesita (a) grid/Bayes search, (b) análisis de sensibilidad por componente. `n_cycles` además se fija **en pasos** (`dt_noise = n·π/n_steps`), no en tiempo físico. |
| `torch_optimizer.py` | ⚠️ | Ver §6. |

### 5.1 Contradicciones internas entre `findings` y la arquitectura final

El diario de investigación contradice al champion sin reconciliación:
- `findings_v11_adam`: "Adam fue un desastre; la normalización RMS amortigua los terremotos… la partícula debe someterse al terreno de forma pura y estúpida" → v23 incorpora precondicionador **EMA de ∇² tipo RMSProp**.
- `findings_v16_momentum`: "el momentum heavy-ball fracasa por efecto honda; cero memoria es imperativo" → v23 incorpora **momentum** (aunque amortiguado por fase).
Puede que ambas reintroducciones estén justificadas (la amortiguación por fase resuelve lo que fallaba), pero el paper debe *explicar* por qué lo que fracasó funciona ahora.

### 5.2 Tests: saneados pero superficiales

25 tests pasan. Cubren shapes, reproducibilidad, gradientes de campos, y 1 test de "optimización" real (esfera convexa — trivial). Faltan (P1): gradientes de funciones objetivo (ya escrito en `v1`), tests de regresión del benchmark con semillas fijas, casos borde (D=1, N=1, `n_steps=0`, bounds asimétricos), y un test que fije el resultado exacto esperado de una corrida corta conocida (golden test anti-regresión).

### 5.3 Higiene del repo

- `.gitignore` contiene `_*.py` (cualquier archivo que empiece por `_`) — patrón trampa que silenciará ficheros legítimos en el futuro.
- `results/` está gitignored pero 3 ficheros quedaron trackeados (estado incoherente).
- `scratch/` está trackeado pese a ser zona de trabajo (incluye `test_cma_1d_*.py` sueltos).
- `docs/chat_*.md` (~2000 líneas de logs de conversación con agentes) y `docs/ACollection…xml` comentan la procedencia asistida del proyecto. Decidir conscientemente: se mantienen como historia (legítimo y honesto) o se archivan fuera del repo para la versión "paper". Recomiendo: mantener en repo un `docs/PROVENANCE.md` que declare el uso de agentes (transparencia que los venues empiezan a exigir) y mover los logs crudos a un release/archivo aparte.
- No hay CI, ni `LICENSE` en el árbol (pyproject declara MIT pero falta el fichero), ni `CITATION.cff`, ni `CONTRIBUTING.md`. Los tres últimos son P1 baratos para un repo de paper.

---

## 6. Optimizador PyTorch: juguete, no producto

Cálculo estático (`v4_torch_estatico.py`), matriz `OMEGAS` de tamaño `(octavas, R, #params)` materializada en fp32:

| Modelo | #params | Memoria del campo de ruido | Viabilidad |
|---|---|---|---|
| MLP MNIST (784-32-10) | 25,450 | 0.03 GB | OK |
| MLP medio (784-256-256-10) | 269,066 | 0.28 GB | Justo |
| ResNet-18 | 11.7 M | **12 GB** | Inviable |
| Red 124M | 124 M | **127 GB** | Inviable |

Además: `dt_noise = n_cycles·π / 2000` tiene el **2000 hardcodeado**: en 20 épocas de MNIST (~9380 pasos) se ejecutan 46.9 ciclos sísmicos en lugar de los 10 de diseño — el schedule temporal está mal calibrado para horizontes de entrenamiento reales. Y a diferencia del núcleo NumPy, **no hay tracking del mejor punto**: el ruido se inyecta directamente en los pesos, con degradación garantizada en fase de convergencia fina.

Veredicto: la sección "PyTorch integration" del README promete deep learning; la implementación actual solo es viable en modelos ≤ ~1M parámetros. Para el paper: o se escala (baja dimensión del campo, ruido por capas con ⊗ Kronecker, o el "Lissajous sweep" O(M) que ya apunta `ideas.md`), o se retira la afirmación.

---

## 7. Revisión de la idea y novedad real

**La idea** (deformar el paisaje con ruido espacialmente correlacionado y dejar que el gradiente haga el resto) es limpia, memorable y correctamente implementada. Tras revisar literatura:

| Antecedente | Relación con Seismic Descent | Diferencia clave |
|---|---|---|
| **Energy Landscape Paving** (Hansmann & Wille 2002) | Deforma el paisaje para escapar de mínimos | ELP deforma *localmente* con penalización tipo tabú (memoria); Seismic deforma *globalmente* con un campo coherente sin memoria. El repo **debe citarlo**: es el pariente más cercano conceptualmente. |
| **Homotopía/suavizado Gaussiano + continuación** (Mobahi & Fisher 2015, entre otros) | Familia de paisajes deformados que se recorren con gradiente | El suavizado es determinista y annealed (de grosero a fino); Seismic es *oscilatorio perpetuo* (no annealed) y estocástico-espacial. La conexión merece una sección: Seismic ≈ homotopía "cíclica" con kernel aleatorio. |
| **SGLD / Langevin** (Welling & Teh 2011) | Gradiente + ruido | Ruido blanco isótropo vs campo correlacionado; SGLD es muestreo, Seismic es optimización con tracking. Mi ablación (§3.3) sugiere que la diferencia de rendimiento es real pero modesta y dependiente de la multimodalidad — hay que demostrarla. |
| **Basin hopping / métodos de deformación de cuencas** | Escape de mínimos por sacudida | BH sacude la partícula; Seismic sacude el suelo. La metáfora es original; la mecánica tiene familia. |
| **CMA-ES/ES** | Baseline | Seismic no aprende geometría (no hay N³/N²): es su ventaja (coste) y su límite (no infiere anisotropía en D alta, §3.4–§4). |

**Qué es genuinamente novedoso y defendible:**
1. Uso de **RFF/ORF como deformación de paisaje con gradiente analítico O(N·D)** — no he encontrado precedente en optimización global.
2. El **schedule oscilatorio perpetuo con seguimiento externo del mejor punto** (ni annealing ni aceptación Metropolítena; "anytime").
3. La evidencia del **crossover a presupuestos altos** contra CMA-ES básico (verificada aquí) en multimodales duros.
4. Campos deterministas de Lissajous inconmensurables como generador reproducible de perturbaciones (tocable por teoría de quasi-periodicidad).

**Qué NO es novedoso (no vender como tal):** deformar paisajes (ELP, homotopías), momentum (heavy-ball/Nesterov), precondicionado diagonal (RMSProp/Adam), atracción al mejor del enjambre (PSO/Kennedy-Eberhart — **cítalo**: la "swarm gravity" es PSO reducido a best-only con modulación por fase), y optimización Bayesiana con GRF (parentesco conceptual que conviene reconocer).

---

## 8. Análisis de riesgos de revisión ("qué diría el referee #2")

1. *"Laplacian ergodicity is not proven"* — peor: **es refutable** con el propio código (§3.2). P0: retirar o demostrar.
2. *"Where is the ablation against power-matched white noise?"* — no existe; mi mini-ablación da resultados mixtos (§3.3). P0.
3. *"Baselines are unseeded, untuned and convenience-implemented (SA casera, CMA-ES sin reinicios)"* — P0: usar implementaciones de referencia (Nevergrad / pycma / pygad) con configuraciones documentadas.
4. *"No statistical significance, small sample (15 trials), 4 funciones"* — P0: COCO/BBOB o, mínimo, 12–15 funciones × {2,5,10,20,40}D × 31 trials + Wilcoxon/Friedman + Holm.
5. *"Gradient cost is free in the budget"* — sesgo del protocolo (§4). Declarar o cargar.
6. *"The v23 champion is 5 mechanisms tuned by hand"* — P1: ablaciones formales + búsqueda de hiperparámetros; identificar el subconjunto mínimo necesario (mi apuesta con los datos: gradiente normalizado + RFF + dt_floor + gravedad; momentum y métrica solo ayudan en valles curvos).
7. *"PyTorch claims are not scalable"* — P1 (§6).
8. *"Kronecker-Weyl/HMC/Boltzmann used loosely"* — precisar nomenclatura (P1 barato).

---

## 9. Plan de remediación priorizado (orientado a paper)

**Estado de ejecución (2026-10-03, misma sesión que esta auditoría):**

**P0 — Integridad:** ✅ **completo.**
- [x] Fix de imports (2 bugs) y 25→50 tests verdes (gradientes de 12 funciones, golden tests de regresión, casos borde; se encontró y corrigió además un `ZeroDivisionError` con `n_steps=0`).
- [x] CI con GitHub Actions (pytest en 3.9–3.12, con y sin torch CPU, + benchmark smoke).
- [x] Semillas en TODOS los baselines (CMA-ES y SA de los scripts existentes; baselines de referencia nuevos en `benchmarks/baselines.py` con config documentada); reportes con IQR/dispersión y Wilcoxon + corrección de Holm (`benchmarks/stats.py`). Pendiente solo para el envío final: subir trials a ≥31.
- [x] README/theory retractan la ergodicidad laplaciana y la tabla MNIST queda marcada como preliminar; sustituida la narrativa por cobertura 100% + decaimiento de autocorrelación (verificado en `v2`).
- [x] LICENSE (MIT), CITATION.cff, versión unificada 0.23.1, CHANGELOG.md, PROVENANCE.md.

**P1 — Núcleo científico:** ✅ **mayoritariamente ejecutado** (resultados en `docs/audit_2026/informe_*.md`).
- [x] **Ablación formal correlado-vs-blanco-vs-apagado**: 12 funciones (6 nuevas con gradientes verificados) × {2,5,10,20}D × 15 trials pareados, Wilcoxon+Holm → `informe_ablacion_ruido.md`. Resultado: ventaja del correlacionado SOLO en multimodales y creciente con D; penalización clara en unimodales (mapa de regímenes — nueva historia central del paper).
- [x] Ley √D del ratio ruido/señal medida y corrección `amp·√(5/D)` evaluada → `informe_ablacion_componentes.md` Parte B (mejoras drásticas en D=20: e.g. Trid abs=1.0e4 → dnorm=−75; Sphere 0.81 → 0.043).
- [x] Benchmark ampliado a 12 funciones × {5,10,20}D con baselines sembrados: CMA-ES, IPOP-CMA-ES (reinicios), PSO, L-BFGS-B multistart, random → `benchmark_v24_baselines.py` + fase 2 de crossover a 25k con IPOP incluida. (COCO queda para la fija de journal.)
- [x] Ablación leave-one-out de los componentes del champion v23 → `informe_ablacion_componentes.md` Parte A.
- [x] Nomenclatura "HMC" corregida en documentación (README/README.es) sin romper API.
- [ ] Grid/Bayes search de hiperparámetros con split train/test de funciones → planificado en `docs/PAPER_BLUEPRINT.md` §4.

**P1.5 — Hallazgo posterior a la auditoría (misma tarde):** la ley de amplitud detectada en el §3.4
(E‖∇ruido‖ ∝ amp·√D/ℓ) se convirtió en la **candidata v24** (`seismic_champion_v24` = v23 + flag opt-in
`noise_amp_dim_normalized=True`, default-off que preserva v23 bit a bit). Spot-check D=20 (15 trials):
Trid 2034→7.65 (**~266×**), Zakharov 3.7×, Rosenbrock 2.1×, Ackley 1.5×, Rastrigin 1.2×. Efecto sobre
la mediana, con volteos por semilla (~1 de 5 a presupuesto corto) — fijado así en los tests.
Ver `docs/findings_v24_amplitud_raiz_d.md`.

**P2 — Escritura y empaquetado:** 🔄 **iniciado.**
- [x] `Makefile` con objetivos `test/ablation/components/champion`; `docs/PAPER_BLUEPRINT.md` con la matriz claims↔evidencia y la estructura del paper de 8pp.
- [ ] Package inmutable (Dockerfile/lock) + tag + Zenodo DOI → pendiente del cierre del benchmark v24.

**Estimación de viabilidad tras el plan:** workshop NeurIPS/ICML o GECCO/CEC: **alta** si la ablación central resulta favorable y los baselines son justos. Journal (TEVC/Swara&EC): media — requeriría COCO completo y teoría de convergencia/escape (probabilidad de escape por cuenca derívable del campo RFF; atacable con teoría de *exit times* de procesos Gaussianos).

---

## 10. Diferencias con `docs/AUDIT.md` (auditoría previa)

| Tema | AUDIT.md decía | Esta auditoría verificó |
|---|---|---|
| Tests | "26 pruebas pasando (100%)" | **Falso en HEAD** (NameError en colección); tras fix: 25 pass + 1 skip. |
| Datos COCO | "el repositorio contiene datos COCO en exdata/" | **No existen en el árbol** (`exdata/` está gitignored y ausente); hay que regenerarlos o descartar esa línea. |
| ‖∇ruido‖ ~2.3 | "medido" | Confirmado: 2.36 en D=5… **y** ley √D hasta 5.6× en D=20 (nuevo). |
| Laplacian ergodicity | "no demostrada" (crítica teórica) | **Refutada empíricamente**: kurtosis −0.6…−0.85, Normal > Laplace en KS/AIC ×4 configs; figura proviene del visualizador JS ≠ algoritmo (nuevo y decisivo). |
| Kronecker-Weyl aplicado a RFF | incorrecto | Matiz: incorrecto para RFF; para Lissajous sí es defensible (quasi-periodicidad densa en el toro), pero "ergodicidad del campo" ≠ "ergodicidad de la trayectoria de la partícula". |
| HMC | nombre engañoso | Confirmado (heavy-ball amortiguado; además `velocities = μv + (1−μ)F` es EMA, no integrador simpléctico). |
| SA sin documentar | schedule no especificado | Está en código (T₀=10, cooling=0.999, paso 5% rango) pero sin semilla ni justificación → mismo veredicto práctico. |
| Lissajous determinista | "misma semilla, misma trayectoria" | Matiz: la trayectoria sí varía por seed (los *particles* se inicializan aleatoriamente); lo idéntico es el campo. |
| Escalabilidad torch | "inviable para modelos grandes" | Cuantificado: 12 GB ResNet-18 / 127 GB en 124M; + schedule descalibrado (hardcode 2000). |
| Benchmark champion | budget 3000 | Además: `run_cmaes` ignora la semilla; budget-scaling usa v20 sin decirlo; gradiente no contabilizado. |

---

## Anexo A. Cómo reproducir esta auditoría

```bash
python -m venv venv && venv/bin/pip install -e . pytest scipy cma
venv/bin/python -m pytest -q                                    # 25 passed, 1 skipped (tras fixes)
for s in docs/audit_2026/v*.py; do venv/bin/python "$s"; done   # ~2 min total
```

Los scripts no modifican el paquete; `v2` y `v5` re-implementan el bucle de `core.py` instrumentado (idéntica línea a línea, añadiendo registro y motores de ruido intercambiables).

## Anexo B. Cambios aplicados en esta sesión (2 fixes críticos)

1. `src/seismic_descent/functions.py:7` — `from typing import Any, Dict, Union`.
2. `src/seismic_descent/torch_optimizer.py:12–21,100` — decorador `_NO_GRAD` con fallback; sin torch el módulo importa y `SeismicOptimizer` lanza su `ImportError` intencional al instanciarse.

Ambos son correcciones de infraestructura (no alteran la dinámica del algoritmo) y eran **imprescindibles para instalar, importar y auditar el paquete**.
