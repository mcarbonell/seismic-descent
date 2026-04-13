# Seismic Descent — Benchmark Ackley (v5)

## Función Ackley

Dominio: [-32.768, 32.768]^D, óptimo global f(0,...,0) = 0

Estructura muy diferente a Rastrigin:
- Superficie exterior casi plana con gradiente ~0
- Descenso suave hacia el centro
- Mínimos locales pequeños (~3-10) distribuidos por la superficie

## Resultados

### Ackley 2D — 50 pruebas | umbral f<1.0 | budget=6.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic+RFF | 7/50 | 15.119 | 18.562 |
| SA | 20/50 | 11.407 | 18.365 |
| CMA-ES | 42/50 | 2.778 | 0.000 |

### Ackley 5D — 30 pruebas | umbral f<1.0 | budget=12.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic+RFF | 0/30 | 19.188 | 19.455 |
| SA | 5/30 | 15.772 | 19.227 |
| CMA-ES | 26/30 | 1.430 | 0.000 |

### Ackley 10D — 30 pruebas | umbral f<1.0 | budget=22.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic+RFF | 0/30 | 19.340 | 19.423 |
| SA | 1/30 | 18.782 | 19.664 |
| CMA-ES | 29/30 | 0.039 | 0.000 |

### Ackley 20D — 10 pruebas | umbral f<1.0 | budget=42.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic+RFF | 0/10 | 19.730 | 19.768 |
| SA | 0/10 | 19.991 | 20.005 |
| CMA-ES | 10/10 | 0.000 | 0.000 |

## Diagnóstico

**Ackley es el peor caso para métodos basados en gradiente.**

La media de ~19-20 en todas las dimensiones indica que Seismic (y SA) quedan
atrapados en la meseta exterior de Ackley. El gradiente en esa región es
prácticamente cero — el gradiente numérico con eps=1e-4 no detecta la pendiente
suave hacia el centro, y la pelota no sabe hacia dónde moverse.

Esto no es un fallo del ruido RFF ni del schedule de amplitud. Es una limitación
**fundamental de los métodos de primer orden** en funciones con mesetas:

- Seismic Descent: gradiente numérico → cero en la meseta → sin movimiento útil
- SA: saltos aleatorios → eventualmente cae al centro por azar → funciona mejor
- CMA-ES: no usa gradiente, evalúa población → detecta la pendiente global → domina

**SA supera a Seismic en Ackley** precisamente porque los saltos aleatorios
son más útiles que el gradiente cuando el gradiente es cero.

## Conclusión general

| Función | Seismic vs SA | Seismic vs CMA-ES | Razón |
|---|---|---|---|
| Rastrigin 2D | ✅ Seismic gana | ✅ Seismic gana | Gradiente útil, Perlin real |
| Rastrigin 10D | ✅ Seismic gana | ❌ pierde | RFF correlacionado |
| Rastrigin 20D | ✅ Seismic gana | ❌ pierde | RFF correlacionado |
| Ackley (todas dims) | ❌ SA gana | ❌ pierde | Meseta con gradiente ~0 |

Seismic Descent es fuerte en funciones **multimodales con gradiente informativo**
(Rastrigin). Falla en funciones con **mesetas o gradiente casi cero** (Ackley).

## Próximo paso: Schwefel

Schwefel tiene el óptimo global en una esquina del dominio, lejos del centro.
Interesante porque penaliza la exploración simétrica. Seismic podría funcionar
bien si el gradiente es informativo, o mal si hay mesetas similares a Ackley.
