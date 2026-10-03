# Findings v24 — Corrección de amplitud √(ref_D / D)

**Fecha:** 2026-10-03 · **Base:** Champion v23 (sin cambios estructurales; flag opt-in
`noise_amp_dim_normalized=True`, default `False` que preserva el comportamiento v23 bit a bit).

## Motivación (medida en la auditoría 2026)

El README histórico presentaba `noise_amplitude = 0.5` como universal "relativo al gradiente
unitario". La auditoría midió que el atractor no es adimensional: para el campo RFF/ORF con
kernel Gaussiano de lengthscale ℓ,

> E‖∇ruido‖ ≈ amp · √D / ℓ  ⇒  ratio ruido/señal crece como **√D**
> (medido: 1.37× en D=2, 2.36× en D=5, 5.58× en D=20 a amp=0.5 — encaja √D).

En D=20 la perturbación domina 5.6× la señal del objetivo: el enjambre "surfea" el terremoto
pero apenas desciende. Esto explica el colapso en D alta ya anotado en `summary_of_experiments.md`.

## Corrección propuesta

`amp_eff(D) = noise_amplitude · √(ref_dim / D)` con `ref_dim = 5` (idéntico al histórico en D=5). La derivación formal de la ley (lema E‖∇n‖² = amp²D/ℓ², unicidad del exponente 1/2, verificación numérica) está en `appendix_A_ley_amplitud_raiz_d.md`.

## Evidencia

**Parte B de `informe_ablacion_componentes.md`** (núcleo v20, 12 funciones × {5,10,20}D × 15 trials,
amp ∈ {0.25, 0.5, 0.75}): la variante √D-normalizada iguala en D=5 (misma amp por construcción) y
mejora consistentemente en D≥10, con efectos muy grandes en D=20 (p. ej. Trid: mediana
1.0×10⁴ → −75; Sphere: 0.81 → 0.043).

**Benchmark definitivo `informe_baselines_v24.md` (36 celdas, 15→**31 trials**, presupuesto 3000):**
v24 (ref5) vs v23 absoluto: **gana 20 celdas, empata 12 (todas 5D, por construcción), pierde 4**
(las 3 de Levy + una marginal extra); a 31 trials la excepción se mantiene acotada a la clase Levy.
La configuración recomendada posterior al split train/test es **ref_dim=1** (ver sección siguiente y
`findings_v25_amplitud_adaptativa.md` para el veredicto externo: mejor sísmico en 30/36, 3/11/22 vs
IPOP-CMA-ES).


**Spot-check inicial sobre Champion v23 (D=20, 15 trials, presupuesto 3000):**

| Función 20D | v23 amp=0.5 absoluta | **v24 amp·√(5/20)=0.25** | Mejora |
|---|---|---|---|
| Trid | 2034 | **7.65** | ~266× |
| Zakharov | 102.7 | **27.65** | 3.7× |
| Ackley | 18.69 | **12.40** | 1.5× |
| Rastrigin | 89.5 | **74.59** | 1.2× |
| Rosenbrock | 76.85 | **37.41** | 2.1× |

**Matiz honesto:** el efecto es de mediana, no universal: en Trid 20D con 200 pasos (presupuesto
más corto que el de referencia) se observan volteos por semilla (~1 de 5). El test de regresión
(`tests/test_golden_regression.py::TestV24DimNormalizedAmplitude`) fija: mejora en mediana y ≥4/5
semillas pareadas a presupuesto 3000.

## Train/test split sobre `ref_dim` (2026-10-03, `informe_refdim_train_test.md`)

¿Es `ref_dim = 5` un valor tunado sobre el benchmark? Grid {0.5, 1, 2, 3.5, 5, 7.5, 10},
selección por **rank medio SOLO en 6 funciones "train"** (ackley, dixon, griewank, michalewicz,
styblinski, zakharov), evaluación congelada en 6 "test" (levy, rastrigin, rosenbrock, schwefel,
sphere, trid); 15 trials pareados, presupuesto 3000, D ∈ {5, 10, 20}:

- **Selección train: ref_dim\\* = 0.5.** **Ranking test (nunca usado): ref1 (2.50/8) > ref2 (2.89)
  > ref0.5 (3.33) > … > ref5 (4.89) > v23 absoluto (5.94).** La selección transfiere: el vecindario
  óptimo es **ref_dim ≈ 1** (0.5–2), no 5 — y tampoco es un artefacto de overfitting del split.
- **Escala de las ganancias a ref_dim pequeño (test):** trid_20d v23_abs 2034 → ref5 7.6 →
  **ref0.5 −1502 (óptimo global = −1520)**; sphere_20d 0.00017 (~1000× frente a v23_abs);
  zakharov_20d 4.79; rosenbrock_20d 12.4.
- **La excepción Levy se agudiza y ya es mapa de regímenes puro:** a menor ref_dim, sistemáticamente
  peor en levy_10d/20d (ref0.5=6.2/23.2 vs v23_abs=1.6/5.1; mejor en la familia: amplitud alta,
  ref10=3.8) — y schwefel_5d también invierte (ref1=240 < v23_abs=359 < ref0.5=455).

**Conclusión:** la *forma* de la ley `amp·√(ref/D)` se valida; la *constante* óptima es ~1, no 5
(mantener 5 como definición v24 por continuidad con v23; exponer `ref_dim` como parámetro es la
vía correcta — ya disponible: `SeismicChampionV23(ref_dim=…)`). La inversión en la clase
Levy/Schwefel demuestra que **ninguna constante global es óptima**: la amplitud debe adaptarse al
régimen detectado → definición natural de una futura v25 (detector de régimen × amplitud alta
solo en multimodal profundo), coherente con la tesis central del paper: "saber cuándo temblar y
cuánto".

## Uso

```python
from seismic_descent import seismic_champion_v24
# o: seismic_champion_v23(..., noise_amp_dim_normalized=True)
```

## Pendiente antes de declarar v24 oficial

- [x] Re-correr `benchmark_v24_baselines.py` completo con v24 como algoritmo (✅ 2026-10-03:
      gana 21/24 celdas diferenciadoras; excepciones: Levy). Informe con Wilcoxon+Holm
      (v24-vs-IPOP por celda) en `docs/audit_2026/informe_baselines_v24.md`.
- [x] Grid sobre `ref_dim` con split train/test (✅ 2026-10-03): óptimo train 0.5, test ref1
      transfiere bien; vértice Levy/Schwefel valida la lectura de regímenes. Parámetro
      `ref_dim` expuesto en `SeismicChampionV23` (default 5.0 = definición v24).
- [ ] Derivar la ley E‖∇‖² = amp²·D/ℓ² en el apéndice del paper (una página).
- [ ] Variante adaptativa "v25": amplitud alta solo cuando el detector declare paisaje
      multimodal profundo (clase Levy) — mapa de regímenes hecho política (futuro).
