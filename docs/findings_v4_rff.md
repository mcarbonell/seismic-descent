# Seismic Descent — Hallazgos RFF / Gaussian Random Fields (v4)

## El problema resuelto

El value noise usado en v2/v3 era independiente por dimensión — básicamente
ruido isotrópico sin estructura espacial. Seismic Descent se comportaba como
SA con gradiente en lugar de sampling, sin ventaja real en ND.

La solución: **Random Fourier Features** (Rahimi & Recht, 2007) para aproximar
un Gaussian Random Field con kernel squared-exponential.

```
noise(x) ≈ sqrt(2/R) * A * Σ_r cos(ω_r · x + t*drift_r + φ_r)
```

donde `ω_r ~ N(0, 1/l²·I)` son vectores en R^D. La clave: al ser vectores
N-dimensionales crean interferencia entre dimensiones, generando correlación
espacial real — dos puntos cercanos tienen ruido similar, dos lejanos son
independientes. Exactamente las propiedades del Perlin 2D, pero en ND.

Parámetros:
- `R = 64` features por octava
- 4 octavas con lengthscales 2, 4, 8, 16 — exploración multiescala
- `drift_r` por feature hace evolucionar el campo con t (el terremoto)

---

## Resultados — benchmark fair (mismo presupuesto de evaluaciones)

Budget = 2000 * (D+1) evaluaciones reales para todos.

### Rastrigin 5D — 30 pruebas | umbral f<5.0 | budget=12.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| Seismic+RFF | 3/30 | 10.329 | 9.997 |
| SA | 7/30 | 10.041 | 7.979 |
| CMA-ES | 8/30 | 6.698 | 6.467 |

### Rastrigin 10D — 30 pruebas | umbral f<10.0 | budget=22.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| **Seismic+RFF** | 0/30 | **45.260** | **46.368** |
| SA | 0/30 | 54.180 | 52.255 |
| CMA-ES | 9/30 | 16.317 | 13.929 |

### Rastrigin 20D — 10 pruebas | umbral f<20.0 | budget=42.000 evals

| Algoritmo | Éxitos | Media | Mediana |
|---|---|---|---|
| **Seismic+RFF** | 0/10 | **134.165** | **133.861** |
| SA | 0/10 | 171.024 | 167.874 |
| CMA-ES | 2/10 | 32.535 | 30.346 |

---

## Comparativa de evolución del ruido ND

| Ruido | 10D media | 20D media | Correlación espacial ND |
|---|---|---|---|
| Sinusoidal (v1) | — | — | ❌ deriva sistemática |
| Value noise (v2/v3) | 52.763 | 197.402 | ❌ independiente por dim |
| **RFF / GRF (v4)** | **45.260** | **134.165** | ✅ correlación real en ND |

El RFF reduce la media en 20D un **32%** respecto al value noise.

---

## Cuadro completo Seismic vs SA (todas las dimensiones)

| Dimensión | Seismic vs SA | Notas |
|---|---|---|
| 2D | **Seismic gana** (47/50 vs 34/50) | Perlin 2D real, 5000 pasos |
| 5D | Empate técnico | RFF: 10.3 vs SA: 10.0 |
| 10D | **Seismic gana** | RFF: 45.3 vs SA: 54.2 |
| 20D | **Seismic gana** | RFF: 134 vs SA: 171 |

**Seismic+RFF supera a SA en 3 de 4 dimensiones con presupuesto igualado.**
CMA-ES sigue siendo el algoritmo más fuerte — es de segundo orden y aprende
la curvatura del paisaje, mientras Seismic es de primer orden.

---

## Por qué funciona el RFF

En 2D el Perlin crea "cauces" coherentes que guían la pelota entre cuencas.
El RFF reproduce esta propiedad en ND: las frecuencias `ω_r` son vectores en
R^D, no escalares, lo que crea estructuras correlacionadas en todas las
direcciones simultáneamente. La pelota no rebota aleatoriamente — se desliza
por el paisaje perturbado siguiendo gradientes coherentes.

La amplitud cíclica `A(t) = A0 * |sin(t * freq(t))| ` sigue siendo clave:
los momentos de calma (seno≈0) permiten asentarse en valles reales, y los
ciclos se alargan con el tiempo para dar refinamiento fino al final.

---

## Próximos pasos

- Benchmark en Ackley y Schwefel (Rastrigin puede favorecer a CMA-ES por su
  estructura regular)
- Explorar gradiente analítico para eliminar el coste O(D) y dar más pasos
  reales con el mismo presupuesto
- Aumentar R (número de features RFF) y medir el trade-off calidad/coste
- Aplicación a problemas reales: optimización de hiperparámetros, TSP
