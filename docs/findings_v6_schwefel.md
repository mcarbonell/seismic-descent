# Seismic Descent — Benchmark Schwefel (v6)

## Función Schwefel

```
f(x) = 418.9829*D - sum(x_i * sin(sqrt(|x_i|)))
```

Dominio: [-500, 500]^D, óptimo global f(420.9687, ..., 420.9687) = 0

Características:
- Óptimo global en una región específica del dominio (no en el origen)
- Gradiente informativo localmente, pero con mínimos locales profundos engañosos
- Mínimos locales cerca de x_i ≈ -302 y x_i ≈ 500 casi tan buenos como el global
- El valor en la "meseta" es ~418.9829*D — todos los algoritmos quedan ahí

## Resultados

### Schwefel 2D — 50 pruebas | umbral f<100 | budget=6.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic+RFF | 2/50 | 388.1 | 414.6 |
| SA | 1/50 | 399.9 | 414.5 |
| CMA-ES | 2/50 | 380.1 | 414.5 |

### Schwefel 5D — 30 pruebas | umbral f<250 | budget=12.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic+RFF | 0/30 | 1051 | 1097 |
| SA | 0/30 | 1041 | 1096 |
| CMA-ES | 0/30 | 1039 | 1056 |

### Schwefel 10D — 30 pruebas | umbral f<500 | budget=22.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic+RFF | 0/30 | 2039 | 2044 |
| SA | 0/30 | 2037 | 2026 |
| CMA-ES | 0/30 | 2022 | 2074 |

### Schwefel 20D — 10 pruebas | umbral f<1000 | budget=42.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic+RFF | 0/10 | 4365 | 4271 |
| SA | 0/10 | 4344 | 4287 |
| CMA-ES | 0/10 | 4304 | 4207 |

## Diagnóstico

**Los tres algoritmos fallan por igual — es un problema de presupuesto, no de algoritmo.**

Las medias son casi idénticas entre sí en cada dimensión. El valor típico
~418.9829*D indica que todos quedan atrapados en mínimos locales profundos
(cerca de x_i ≈ -302 o x_i ≈ 500) que son casi tan buenos como el óptimo global.

Schwefel requiere un presupuesto mucho mayor para distinguir el óptimo global
de los mínimos locales engañosos. Con 2000*(D+1) evaluaciones ningún algoritmo
tiene suficiente información para orientarse en el dominio [-500, 500]^D.

A diferencia de Ackley (donde Seismic falla por gradiente nulo), aquí el
gradiente es informativo — simplemente no hay suficientes evaluaciones para
explorar un dominio 1000x más grande que Rastrigin.

## Cuadro comparativo completo

| Función | Característica | Seismic vs SA | Seismic vs CMA-ES |
|---|---|---|---|
| Rastrigin 2D | Multimodal, gradiente útil | ✅ gana | ✅ gana |
| Rastrigin 10D | Multimodal ND, RFF | ✅ gana | ❌ pierde |
| Rastrigin 20D | Multimodal ND, RFF | ✅ gana | ❌ pierde |
| Ackley | Meseta con gradiente ~0 | ❌ pierde | ❌ pierde |
| Schwefel | Dominio enorme, engañoso | empate | empate |

## Conclusión del perfil de Seismic Descent

**Funciona bien cuando:**
- El gradiente es informativo (no hay mesetas)
- El dominio es manejable con el presupuesto disponible
- La función es multimodal con estructura espacial (Rastrigin)

**Falla cuando:**
- El gradiente es ~0 (Ackley, mesetas)
- El dominio es muy grande y los mínimos locales son engañosos (Schwefel)
- Estas son limitaciones compartidas con todos los métodos de primer orden

**La ventaja real de Seismic** es que el ruido RFF correlacionado guía la
exploración de forma más inteligente que los saltos aleatorios de SA cuando
el gradiente tiene información. En funciones donde el gradiente no ayuda,
Seismic no tiene ventaja sobre SA.
