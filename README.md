# Seismic Descent

Algoritmo de optimización basado en descenso de gradiente sobre un paisaje
dinámico perturbado con ruido correlacionado espacialmente.

## La idea

En lugar de dar saltos aleatorios para escapar de mínimos locales (como
Simulated Annealing), se le da un **terremoto al suelo**. La pelota sigue
haciendo lo único que sabe: rodar cuesta abajo. Pero como el suelo tiembla
de forma coherente y multiescala, los mínimos locales se convierten
temporalmente en cuestas y la pelota escapa sola.

```
f_total(x, t) = f_original(x) + A(t) * noise(x, t)
```

- `f_original` — función objetivo real
- `noise(x, t)` — ruido correlacionado espacialmente (Perlin en 2D, RFF en ND)
- `A(t) = A0 * sin(t * freq(t))` — amplitud cíclica con frecuencia decreciente
- El mejor punto se registra siempre contra `f_original`, no contra el paisaje combinado

La clave frente a SA: el ruido es **correlacionado espacialmente** — dos puntos
cercanos tienen perturbaciones similares. La pelota se desliza hacia otro valle,
no se teletransporta. Las octavas del ruido dan exploración gruesa y refinamiento
fino en un solo mecanismo.

## Ruido en N dimensiones: Random Fourier Features

En 2D se usa Perlin noise directamente. En ND se aproxima un Gaussian Random
Field via Random Fourier Features (Rahimi & Recht, 2007):

```
noise(x) ≈ sqrt(2/R) * A * Σ_r cos(ω_r · x + t*drift_r + φ_r)
```

donde `ω_r ~ N(0, 1/l²·I)` son vectores en R^D. Al ser vectores N-dimensionales
crean interferencia entre dimensiones y correlación espacial real en ND.

## Resultados — Rastrigin (presupuesto de evaluaciones igualado)

| Dimensión | Seismic media | SA media | CMA-ES media | Seismic vs SA |
|---|---|---|---|---|
| 2D (5k pasos) | 0.139 | 0.410 | 1.498 | ✅ gana |
| 5D | 10.3 | 10.0 | 6.7 | empate |
| 10D | 45.3 | 54.2 | 16.3 | ✅ gana |
| 20D | 134 | 171 | 32.5 | ✅ gana |

## Resultados — Ackley y Schwefel

| Función | Seismic vs SA | Diagnóstico |
|---|---|---|
| Ackley | ❌ SA gana | Meseta exterior con gradiente ~0 paraliza el descenso |
| Schwefel | empate | Dominio enorme, todos los algoritmos fallan por igual |

## Perfil del algoritmo

**Funciona bien cuando:**
- El gradiente es informativo (funciones multimodales tipo Rastrigin)
- El ruido correlacionado puede guiar la exploración entre cuencas

**Falla cuando:**
- El gradiente es ~0 (Ackley, mesetas) — limitación compartida con todos los métodos de primer orden
- El dominio es muy grande y el presupuesto insuficiente (Schwefel)

## Optimizaciones Recientes (v7 - v14)

El desarrollo del algoritmo ha evolucionado superando importantes cuellos de botella:
- **Gradientes Analíticos ($\mathcal{O}(1)$)**: Calculamos matemáticamente el gradiente del campo RFF, haciendo que evaluar el paso sea casi gratis.
- **Inversión de Polaridad**: Eliminar la función `abs()` en la amplitud permitió que las montañas mutaran bruscamente en valles durante el ciclo, mejorando el escape radicalmente.
- **Seismic Swarm (Enjambre vectorizado)**: Usando `numpy`, evaluamos $N$ partículas paralelamente bajo un mismo campo RFF común. Reduce el tiempo de simulación un ~85% preservando resultados comparables.
- **Parametrización por Ciclos Exactos**: Independización matemática del problema de iteraciones asegurando que el *schedule* siempre decaiga a lo largo de 10 terremotos puros (v14).

## Instalación

```bash
pip install numpy noise matplotlib cma
```

## Uso

```bash
# Benchmark 2D original
python perlin_opt.py

# Benchmark N-dimensional con RFF (mejor versión)
python perlin_opt_nd_grf.py

# Benchmarks en otras funciones
python benchmark_ackley.py
python benchmark_schwefel.py
```

## Estructura

```
docs/
  findings_v1.md           — hallazgos sesión 2D, evolución del schedule A(t)
  findings_v2_nd.md        — extensión ND, diagnóstico del ruido
  findings_v3_fairbench.md — benchmark con presupuesto igualado
  findings_v4_rff.md       — Random Fourier Features, resultados Rastrigin ND
  findings_v5_ackley.md    — benchmark Ackley, limitación en mesetas
  findings_v6_schwefel.md  — benchmark Schwefel, perfil completo del algoritmo
  findings_v7_rastrigin_analytic.md  — Benchmark Rastrigin con gradientes analíticos O(1)
  findings_v8_no_abs.md              — Efecto contundente de permitir amplitud negativa
  findings_v9_no_abs_ackley_schwefel.md — Validación en mesetas y en hiper-espacios de Schwefel
  findings_v10_lengthscale.md        — Diagnóstico sobre la compresión de lengthscales
  findings_v11_adam.md               — Demostrando porqué Adam Optimizer asfixia el "terremoto"
  findings_v12_swarm.md              — Seismic Swarm: paralelización del muestreo sobre RFF analítico
  findings_v13_swarm_d.md            — Enjambre estricto N=D y descubrimiento de trade-off de presupuesto
  findings_v14_cycles.md             — Parametrización estricta de la oscilación sísmica a 10 ciclos
  findings_v15_reactive.md           — Terremotos desencadenados por estancamiento (Bang-Bang Control)
  findings_v16_momentum.md           — Momentum Crudo (Heavy-Ball) causando efecto honda y pérdida de octavas
  findings_v17_temporal_octaves.md   — Sismos Fractales: aplicación de Series de Fourier a las octavas de amplitud temporales
  chat_arena*.md           — conversación original con la idea
  chat_opus4.6.md          — prototipo inicial
perlin_opt.py              — implementación 2D original (Perlin noise)
perlin_opt_nd.py           — extensión ND con value noise
perlin_opt_nd_fairbench.py — benchmark fair (presupuesto igualado)
perlin_opt_nd_grf.py       — Seismic Descent con RFF (versión ND definitiva base)
perlin_opt_nd_grf_analytic*.py — Repositorio incremental de las versiones Analíticas y de Enjambre (v7 a v14)
benchmark_ackley.py        — benchmark genérico, función Ackley
benchmark_schwefel.py      — benchmark función Schwefel
```

## Próximos experimentos

- Búsqueda en grilla (Hyperparameter Sweeping) para equilibrar `noise_decay`, `noise_amplitude`, cantidad de ciclos ($K$) según la dimensión.
- Aplicación de un Momentum ingenuo (Heavy Ball sin regularizador RMS) combinado con lengthscales agrandadas dinámicamente $\propto \sqrt{D}$.
- Aumentar $R$ (features RFF) y medir trade-off calidad/coste en el rendimiento residual en Altas Dimensiones usando matriz de varianza.
- Aplicación teórica al problema del Viajante de Comercio (TSP) con adiciones aleatorias de costo-matriz.
