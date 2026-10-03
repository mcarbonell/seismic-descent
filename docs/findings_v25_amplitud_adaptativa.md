# Findings v25 — Amplitud adaptativa por régimen detectado

**Fecha:** 2026-10-03 · **Estado:** experimento completado (fuera del paquete, en
`benchmarks/experiment_v25_adaptive_amp.py`). **Veredicto global: ref1 constante gana; la
coherencia funciona mecánicamente y queda robusta segunda; el estancamiento, rechazado.**

## Hipótesis

La evidencia acumulada (ablación de ruido: perturbar ayuda solo en multimodal deceptivo;
grid ref_dim: óptimo ≈1 salvo Levy/Schwefel, que invierten) sugiere que ninguna amplitud
constante es óptima. **v25 = champion v23 + política de amplitud que se adapta al régimen
detectado online**, con un único cambio respecto a v23:

- **v25c coherencia**: `coh_t = ‖media_i(ĝᵢ)‖ ∈ [0,1]` sobre los gradientes objetivo
  precondicionados y normalizados (señal gratis del paso 5 del bucle). Valle compartido
  (coh→1) → camina; paisaje discordante (coh→0) → tiembla fuerte.
  `amp = amp_lo + (amp_hi − amp_lo)·(1 − EMA_coh)`. 
- **v25s estancamiento**: `EMA[1(mejora rel. del best ≤ 1e-12)]`, `amp = amp_lo + (amp_hi − amp_lo)·s²`.

Límites: `amp_lo = 0.5·√(1/D)` (ganador del grid), `amp_hi = 0.5` absoluto (ganador Levy).
Protocolo: 12 funciones × {5,10,20} × 15 trials **pareados con init interna compartida
(x0=None)** — idéntico al del grid ref_dim — ORF, presupuesto 3000.

## Resultados (informe completo: `docs/audit_2026/informe_v25_adaptativa.md`)

| Variante | Rank medio | Victorias estrictas | A ≤10% del mejor |
|---|---|---|---|
| **v24_ref1** (constante informada) | **1.58** | **28/36** (valle 16, decept. 12) | **32/36** |
| **v25c_coherence** | **2.36** | 4 (dixon_5d, sphere_5d, styblinski×3) | 11/36 |
| v24_ref5 (=v24 oficial) | 3.67 | 2 (rastrigin_20d, styblinski_10d) | 6/36 |
| v25s_stagnation | 3.22 | 0 | 7/36 |
| v23_abs (histórico) | 4.17 | 2 (levy_10d, levy_20d) | 6/36 |

**Lectura por clases:**

- **Valle** (sphere/trid/zakharov/dixon/griewank/rosenbrock): v25c se comporta casi como
  ref1 (compite o empata en trid, rosenbrock_20d; gana dixon_5d y sphere_5d) — el gate de
  coherencia **detecta correctamente el valle** y apaga el temblor.
- **Deceptiva** (levy/michalewicz/schwefel/rastrigin/styblinski/ackley): v25c gana las 3
  celdas de styblinski_tang y empata en schwefel_5d/michalewicz_10d; en Levy siguen ganando
  amplitudes **altas constantes** (v23_abs) — ref1 pierde ahí, como en el grid.
- **v25s no gana nada**: sus victorias preliminares en Levy (quick diagonal-x0, 3.23) eran
  un artefacto del protocolo de inicialización, no del mecanismo. **Lección registrada:**
  los experimentos deben usar siempre init interna pareada (x0=None) — ya corregido en el
  script; el informe refleja el protocolo limpio.

## Veredicto

1. **Promoción:** la configuración recomendada pasa a ser **v24 con `ref_dim = 1`**
   (`SeismicChampionV23(noise_amp_dim_normalized=True, ref_dim=1.0)`) — el "constante
   informado por split" domina la comparativa intra-champion (rank 1.58, nunca peor que
   ref5/abs salvo la clase Levy≥10D, excepción conocida y documentada).
2. **v25c queda como candidata "robusta"**: nunca catastrófica, segunda en rank, gana su
   nicho (styblinski, dixon/sphere baja D), y es la opción natural cuando la clase de
   paisaje es **desconocida o mezclada** (landscapes compuestos CEC, cambio de régimen
   intra-ejecución) — contexto donde una constante Informada-de-esta-suite no tiene por qué
   transferirse.
3. **v25s rechazada** para el paper salvo como negativo honesto (apéndice).
4. Por qué adaptar no paga *aquí*: ref1 ya es casi-óptima por régimen en esta suite (la
   desviación de su nicho se concentra en una sola clase), así que el gate añade latencia y
   varianza sin margen que recuperar. Predicción testable: v25c sí pagaría en baterías
   heterogéneas por instancia (COCO mixto) y en problemas con régimen no estacionario.

## Benchmark externo (10 algoritmos, baseline sembrados; `informe_baselines_v24.md`)

Añadidas `seismic_v24r1_orf` y `seismic_v25c_orf` al benchmark canónico y repetidas las
36 celdas (**31 trials**, presupuesto 3000):

| Sísmico | vs IPOP-CMA-ES (victoria/empate±10%/derrota) |
|---|---|
| v20 | 1 / 5 / 30 |
| v23 | 2 / 5 / 29 |
| v24 (ref5) | 2 / 5 / 29 |
| **v24 ref1** | **3 / 11 / 22** — mejor sísmico en **30/36** celdas |
| v25c | 2 / 7 / 27 — mejor sísmico en 5 celdas (schwefel_5d 254 vs ref1 337, rastrigin_5d, levy_10d, styblinski×2) |

Nicho cuantificado @31 trials: **victorias estrictas vs IPOP en Schwefel 5D (337 vs 454), Schwefel 10D
(977 vs 1125) y Zakharov 20D (7.4 vs 124, rel −0.89, p_Holm=7.4e-05)**; la familia Styblinski-Tang
empata exactamente; Rosenbrock 20D empatada (15.8/15.8); Trid 20D −1445 del −1516 de IPOP (aunque
las distribuciones difieren, p_Holm=1.8e-07); Michalewicz 20D a nivel PSO/BFGS. En la comparativa
intra-suite (`informe_v25_adaptativa.md`, 31 trials): ref1 rank 1.69 (29 victorias), v25c rank 2.22
(4 victorias, única variante adaptativa no catastrófica; ≤10% del mejor en 9/36).

La lectura externa se mantiene (IPOP/BFGS dominan el multimodal clásico); la punta sísmica (ref1) es
la recomendación oficial, y v25c conserva su identidad de nicho deceptivo profundo de D baja/media.


## Pendiente (paper)

- [x] Evaluar **v24_ref1 y v25c en el benchmark externo** (✅ 2026-10-03 — arriba).
- [ ] Test de estrés de régimen: función mixta (p. ej. mitad de coords Sphere, mitad
      Rastrigin) donde v25c debería ganar estructuralmente.
- [ ] Apéndice con el negativo v25s + la lección de protocolo (init pareada).
