# Perlin Optimization — Hallazgos sesión v1

## La idea

Algoritmo de optimización basado en descenso de gradiente sobre un paisaje dinámico:

```
f_total(x, t) = f_original(x) + A(t) * Perlin(x, t)
```

- `f_original` es la función objetivo real (Rastrigin 2D en este experimento)
- `Perlin(x, t)` es ruido Perlin con 4 octavas, que evoluciona en el tiempo
- `A(t)` controla la amplitud del ruido — el "terremoto"
- La pelota siempre desciende por gradiente del paisaje **combinado**, pero el mejor punto se registra contra `f_original` sola

La clave conceptual: no se le dan piernas a la pelota para saltar, se le da un terremoto al suelo. La pelota sigue haciendo lo único que sabe: rodar cuesta abajo.

---

## Benchmark

- Función: Rastrigin 2D (dominio [-5.12, 5.12], óptimo global en (0,0) con f=0)
- 50 pruebas con puntos de inicio aleatorios
- 5000 pasos por prueba
- Umbral de éxito: f < 0.5
- Comparación contra Simulated Annealing clásico con parámetros equivalentes

---

## Evolución del algoritmo

### v1 — Baseline (decay=0.999, sin seno)
`A(t) = A0 * 0.999^step`

| | Éxitos | Media | Mediana |
|---|---|---|---|
| Perlin | 26/50 | 0.550 | 0.475 |
| SA | 34/50 | 0.510 | 0.094 |

Problema: la pelota pasa por el óptimo global pero no se asienta. El gradiente sigue siendo ruidoso cuando la amplitud ya es pequeña.

### v2 — Amplitud cíclica, frecuencia fija
`A(t) = A0 * 0.999^step * |sin(t * 2.0)|`

| | Éxitos | Media | Mediana |
|---|---|---|---|
| Perlin | 27/50 | 0.522 | 0.437 |
| SA | 34/50 | 0.510 | 0.094 |

Mejora ligera. Los momentos de calma (seno≈0) permiten asentarse, pero la frecuencia fija es demasiado rápida para aprovecharlos bien.

### v3 — Amplitud cíclica, frecuencia decreciente
`A(t) = A0 * decay^step * |sin(t * decay)|`

La frecuencia del seno también decae: terremotos frecuentes al principio, ciclos más lentos al final para dar tiempo de refinamiento.

| | Éxitos | Media | Mediana |
|---|---|---|---|
| Perlin | 30/50 | 0.448 | 0.223 |
| SA | 34/50 | 0.510 | 0.094 |

Mejora clara en mediana. La convergencia ya no se congela pronto.

### v4 — Sin decay global (decay=1.0) ⭐
`A(t) = A0 * |sin(t * 1.0^step)|`

El decay global se elimina. La amplitud cíclica del seno ya hace el trabajo: cuando el seno pasa por cero la pelota se asienta, y como la frecuencia decrece, los períodos de calma se hacen más largos con el tiempo. El decay encima era redundante y recortaba la exploración.

| | Éxitos | Media | Mediana |
|---|---|---|---|
| **Perlin** | **50/50** | **0.122** | **0.081** |
| SA | 34/50 | 0.510 | 0.094 |

**Perlin gana en todas las métricas. 100% de éxitos vs 68% de SA. Mediana prácticamente igual (0.081 vs 0.094).**

---

## Configuración final (v4)

```python
def perlin_optimization(x0, y0, n_steps=5000, dt=0.01, noise_amplitude=15.0,
                         noise_decay=1.0, search_range=5.12):
    ...
    for step in range(n_steps):
        decay = noise_decay ** step          # = 1.0, sin efecto
        freq = 2.0 * decay                   # frecuencia decrece con el tiempo
        amp = noise_amplitude * decay * abs(np.sin(t * freq))
        ...
        t += 0.05
```

Parámetros clave:
- `noise_amplitude = 15.0` — suficiente para superar las barreras de Rastrigin (~10)
- `noise_decay = 1.0` — sin decay global
- `dt = 0.01` — paso de gradiente
- `octaves = 4` — exploración gruesa + refinamiento fino simultáneos
- `t += 0.05` por paso — velocidad de evolución del paisaje Perlin

---

## Observaciones cualitativas

- La trayectoria de Perlin cubre más espacio que SA, explorando sistemáticamente la región central
- SA hace saltos erráticos y refinamiento fino tardío; Perlin barre el espacio de forma más orgánica
- El mejor punto se registra siempre contra `f_original`, no contra el paisaje combinado — la pelota puede alejarse del mejor punto encontrado, pero ese queda guardado

---

## Benchmark vs CMA-ES (50.000 evaluaciones para CMA-ES)

CMA-ES añadido con 10x más evaluaciones (50.000 vs 5.000) y sigma=4.0:

| Algoritmo | Éxitos | Media | Mediana | Evaluaciones |
|---|---|---|---|---|
| **Perlin** | **47/50** | **0.139** | **0.100** | 5.000 |
| SA | 34/50 | 0.410 | 0.117 | 5.000 |
| CMA-ES | 11/50 | 1.498 | 0.995 | 50.000 |

CMA-ES no despega en 2D con Rastrigin — converge prematuramente a mínimos locales. Su ventaja real está en dimensiones altas (>10D). Perlin gana con 10x menos evaluaciones.

## Pendiente / próximos experimentos

- **Escalar a N dimensiones** (Simplex Noise) y comparar con CMA-ES en su terreno real (>10D)
- Probar en otras funciones multimodal: Ackley, Schwefel, Griewank
- Comparar coste computacional por iteración vs SA
- Explorar aplicación a problemas discretos (TSP con offsets en distancias)
- Estudiar sensibilidad a `noise_amplitude` y `dt`
