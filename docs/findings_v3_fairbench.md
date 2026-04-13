# Seismic Descent — Hallazgos benchmark fair (v3)

## Contexto

Benchmark con presupuesto de evaluaciones de función igualado para los tres
algoritmos. Todos reciben `2000 * (D+1)` evaluaciones reales:

- Seismic Descent: 2000 pasos × (D+1) evals/paso = 2000*(D+1) evals
- SA: 2000*(D+1) pasos × 1 eval/paso = 2000*(D+1) evals
- CMA-ES: 2000*(D+1) evaluaciones totales

## Resultados

### Rastrigin 5D — 30 pruebas | umbral f<5.0 | budget=12.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic Descent | 3/30 | 10.934 | 11.416 |
| SA | 1/30 | 13.654 | 14.585 |
| CMA-ES | 13/30 | 7.348 | 5.970 |

### Rastrigin 10D — 30 pruebas | umbral f<10.0 | budget=22.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic Descent | 0/30 | 52.763 | 47.442 |
| SA | 0/30 | 54.817 | 53.654 |
| CMA-ES | 7/30 | 15.157 | 13.929 |

### Rastrigin 20D — 10 pruebas | umbral f<20.0 | budget=42.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic Descent | 0/10 | 197.402 | 198.057 |
| SA | 0/10 | 164.772 | 165.532 |
| CMA-ES | 2/10 | 34.525 | 37.808 |

## Conclusiones

**Seismic supera a SA en 5D y 10D** con presupuesto igualado. La calidad del
gradiente compensa el coste extra por paso. En 20D SA empieza a ganar — con
42.000 pasos aleatorios cubre más espacio que 2000 pasos de gradiente en un
paisaje con ~10^20 mínimos locales.

**CMA-ES domina en todas las dimensiones.** Aprende la geometría del paisaje
y adapta los saltos. Es el algoritmo correcto para optimización continua ND
con presupuesto moderado.

**El problema de Seismic en ND es el ruido, no el gradiente.** El value noise
usado es independiente por dimensión — pierde la correlación espacial que hace
útil al Perlin 2D original. En 2D el Perlin crea valles y crestas coherentes
que guían la pelota entre cuencas. En ND el ruido actual es básicamente
perturbación isotrópica sin estructura, lo que reduce Seismic a algo parecido
a un descenso de gradiente con ruido gaussiano.

## Resumen comparativo global (2D vs ND)

| Dimensión | Seismic vs SA | Seismic vs CMA-ES |
|---|---|---|
| 2D (5000 pasos, mismo budget) | **Seismic gana** (47/50 vs 34/50) | **Seismic gana** (47/50 vs 11/50) |
| 5D (fair budget) | **Seismic gana** en media (10.9 vs 13.7) | Pierde (CMA 7.3) |
| 10D (fair budget) | Empate técnico (52.8 vs 54.8) | Pierde (CMA 15.2) |
| 20D (fair budget) | Pierde (197 vs 165) | Pierde (CMA 34.5) |

La ventaja de Seismic se concentra en dimensiones bajas donde el ruido Perlin
2D tiene correlación espacial real. Escalar correctamente a ND requiere un
ruido genuinamente N-dimensional correlacionado.

## Próximos pasos

- Investigar ruido correlacionado ND real (e.g. campos gaussianos aleatorios,
  o Perlin ND via proyecciones aleatorias)
- Probar gradiente analítico para eliminar el coste O(D) y dar más pasos reales
- Benchmark en otras funciones (Ackley, Schwefel) donde Rastrigin puede no ser
  representativo
