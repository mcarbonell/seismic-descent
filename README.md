# Earthquake Optimization (Perlin Descent)

Algoritmo de optimización basado en descenso de gradiente sobre un paisaje dinámico perturbado con ruido Perlin.

## La idea

En lugar de dar saltos aleatorios para escapar de mínimos locales (como Simulated Annealing), se le da un **terremoto al suelo**. La pelota sigue haciendo lo único que sabe: rodar cuesta abajo. Pero como el suelo tiembla de forma coherente y multiescala, los mínimos locales se convierten temporalmente en cuestas y la pelota escapa sola.

```
f_total(x, t) = f_original(x) + A(t) * Perlin(x, t)
```

- `f_original` — función objetivo real
- `Perlin(x, t)` — ruido Perlin con 4 octavas, evoluciona en el tiempo
- `A(t) = A0 * |sin(t * freq(t))|` — amplitud cíclica con frecuencia decreciente
- El mejor punto se registra siempre contra `f_original`, no contra el paisaje combinado

## Ventajas sobre SA

- Ruido **correlacionado espacialmente** (Perlin vs gaussiano blanco): la pelota se desliza hacia otro valle, no se teletransporta
- **Multiescala** por las octavas: exploración gruesa + refinamiento fino en un solo mecanismo
- Trayectorias suaves y continuas
- Favorece naturalmente los **valles anchos** (flat minima), propiedad deseable en redes neuronales

## Resultados (Rastrigin 2D, 50 pruebas, 5000 pasos)

| Algoritmo | Éxitos (f<0.5) | Media | Mediana | Evaluaciones |
|---|---|---|---|---|
| **Seismic Descent** | **47/50** | **0.139** | **0.100** | 5.000 |
| Simulated Annealing | 34/50 | 0.410 | 0.117 | 5.000 |
| CMA-ES | 11/50 | 1.498 | 0.995 | 50.000 |

## Resultados N-dimensional (Rastrigin, presupuesto igualado, Seismic+RFF)

| Dimensión | Seismic media | SA media | CMA-ES media | Seismic vs SA |
|---|---|---|---|---|
| 2D | 0.139 | 0.410 | 1.498 | ✅ gana |
| 5D | 10.3 | 10.0 | 6.7 | empate |
| 10D | 45.3 | 54.2 | 16.3 | ✅ gana |
| 20D | 134 | 171 | 32.5 | ✅ gana |

## Instalación

```bash
pip install numpy noise matplotlib cma
```

## Uso

```bash
python perlin_opt.py
```

## Estructura

```
docs/
  findings_v1.md          — hallazgos sesión 2D
  findings_v2_nd.md       — extensión ND, diagnóstico del ruido
  findings_v3_fairbench.md — benchmark con presupuesto igualado
  findings_v4_rff.md      — Random Fourier Features, resultados finales
  chat_arena*.md          — conversación original con la idea
  chat_opus4.6.md         — prototipo inicial
perlin_opt.py             — implementación 2D original
perlin_opt_nd.py          — extensión ND con value noise
perlin_opt_nd_fairbench.py — benchmark fair
perlin_opt_nd_grf.py      — Seismic Descent con RFF (mejor versión ND)
```

## Próximos experimentos

- Benchmark en Ackley y Schwefel
- Gradiente analítico para eliminar coste O(D)
- Aumentar R (features RFF) y medir trade-off calidad/coste
- Aplicación a TSP con offsets en distancias
