# Seismic Descent — Hallazgos sesión N-dimensional (v2)

## Contexto

Extensión del algoritmo a N dimensiones. El ruido Perlin 2D (pnoise2) no escala
a dimensiones arbitrarias, por lo que se probaron varias alternativas:

- **opensimplex**: correcto conceptualmente pero inutilizable en ND — usa numba
  con `@njit(cache=True)` y la primera ejecución compila JIT en silencio durante
  ~1 hora. Una vez cacheado es rápido, pero sigue siendo una llamada Python por
  dimensión por octava, sin vectorizar.
- **Ruido sinusoidal**: rápido y vectorizable, pero tiene el problema del
  "arrastre de ola" — `sin(freq*x + phase + t)` tiene gradiente sistemático que
  empuja la pelota siempre en la dirección de la ola. No es ruido isotrópico.
- **Value noise con interpolación suave (smoothstep)**: solución adoptada. Grid
  de valores aleatorios fija, interpolación bilineal con smoothstep. Sin deriva
  sistemática, isótropo, multiescala con octavas.

---

## Coste de evaluaciones — hallazgo importante

Seismic Descent usa gradiente numérico: **D+1 evaluaciones de función por paso**
(1 en posición actual + 1 por cada dimensión). SA usa 1 evaluación por paso.

Con 2000 pasos en 10D:
- Seismic Descent: 2000 pasos × 11 evals = **22.000 evaluaciones reales**
- SA: 2000 pasos × 1 eval = **2.000 evaluaciones reales**
- CMA-ES: 2000 evaluaciones totales (~200 generaciones de población ~10)

El benchmark con "pasos igualados" es injusto para SA. El siguiente experimento
normalizará por evaluaciones de función reales.

---

## Resultados (pasos igualados a 2000, evals reales distintas)

### Rastrigin 5D — 30 pruebas | umbral f < 5.0

| Algoritmo | Éxitos | Media | Mediana | Pasos | Evals reales |
|---|---|---|---|---|---|
| Seismic Descent | 0/30 | 11.127 | 10.702 | 2.000 | ~12.000 |
| SA | 0/30 | 15.801 | 13.223 | 2.000 | 2.000 |
| CMA-ES | 18/30 | 5.837 | 4.975 | — | 2.000 |

Seismic supera a SA en media (11 vs 15) usando 6x más evaluaciones.

### Rastrigin 10D — 30 pruebas | umbral f < 10.0

| Algoritmo | Éxitos | Media | Mediana | Pasos | Evals reales |
|---|---|---|---|---|---|
| Seismic Descent | 0/30 | 56.094 | 56.246 | 2.000 | ~22.000 |
| SA | 0/30 | 65.325 | 61.675 | 2.000 | 2.000 |
| CMA-ES | 6/30 | 15.269 | 13.435 | — | 2.000 |

### Rastrigin 20D — 10 pruebas | umbral f < 20.0

| Algoritmo | Éxitos | Media | Mediana | Pasos | Evals reales |
|---|---|---|---|---|---|
| Seismic Descent | 0/10 | 196.660 | 204.780 | 2.000 | ~42.000 |
| SA | 0/10 | 200.692 | 206.926 | 2.000 | 2.000 |
| CMA-ES | 0/10 | 77.484 | 61.398 | — | 2.000 |

---

## Observaciones

**CMA-ES domina en ND con presupuesto igualado.** Es su terreno natural: aprende
la geometría del paisaje y adapta los saltos. Con 2000 evaluaciones y población
~8 hace ~250 generaciones, suficiente para orientarse en 5D-10D.

**Seismic vs SA:** Seismic supera a SA en media en 5D y 10D, pero está usando
muchas más evaluaciones reales. No es una comparación justa todavía.

**Rastrigin ND es brutalmente difícil.** El número de mínimos locales crece
exponencialmente con la dimensión (~10^D en el dominio [-5.12, 5.12]). Ningún
algoritmo consigue éxitos consistentes en 20D con solo 2000 evaluaciones.

**El gradiente numérico es el cuello de botella de Seismic en ND.** Dos opciones:
1. Gradiente analítico de Rastrigin — elimina el coste O(D) completamente
2. Aceptar el coste y normalizar el presupuesto en el benchmark

---

## Próximo experimento

Benchmark con **presupuesto de evaluaciones de función igualado**:
- SA: 2000 * (D+1) pasos
- CMA-ES: 2000 * (D+1) evaluaciones
- Seismic: 2000 pasos (= 2000*(D+1) evaluaciones reales)

Esto pondrá a los tres en igualdad real y veremos si la calidad del gradiente
de Seismic justifica su coste por paso.
