# Hallazgos v19: Seismic Descent con Ergodic Morphing

## Resumen

v19 implementa el concepto de **Ergodic Morphing**: en lugar de usar un único campo RFF con drift temporal, el algoritmo hace morphing suave entre dos campos estáticos (A y B) usando interpolación de varianza constante.

## Correcciones aplicadas

v19 fue desarrollado por un agente automático siguiendo `improvement_plan.md`, pero se introdujeron errores que fueron corregidos:

| Problema | Origen | Corrección |
|---|---|---|
| `decay_factor = 1.0 - (i/n_steps)` | Agente wrong | Volver a `noise_decay=1.0` sin decay global (v1) |
| Usaba `abs()` del seno | Agente wrong | Scheduling sinusoidal puro sin abs (v8) |
| Faltaba `n_cycles` param | Agente wrong | Añadido `n_cycles=10` (v14) |
| Drift temporal eliminado | Agente wrong | Restaurado `t * drifts` en ángulo RFF |

## Decisiones de diseño en v19

- **Morphing entre campos RFF estáticos**: `weight_A = cos(u*π/2)`, `weight_B = sin(u*π/2)` con `u = (step % morph_steps) / morph_steps`
- **Drift temporal restaurado**: `angles = omegas @ X.T + t * drifts + phis`
- **Sin decay global**: `noise_decay=1.0`
- **Scheduling sinusoidal sin abs**: `amp = noise_amplitude * np.sin(t * freq)`

## Resultados Benchmark (preset low, 10 trials)

| Función | Dims | Seismic v19 | SA | CMA-ES |
|---------|------|-------------|----|----|
| Rastrigin | 5D | 2/10 (7.39) | 1/10 (16.00) | 6/10 (4.48) |
| Rastrigin | 10D | 0/10 (31.56) | 0/10 (66.80) | 4/10 (13.93) |
| Rastrigin | 20D | 0/10 (104.50) | 0/10 (189.51) | 1/10 (24.87) |
| Schwefel | 5D | 0/10 (1377) | 0/10 (827) | 1/10 (623) |
| Schwefel | 10D | 0/10 (3044) | 0/10 (2040) | 1/10 (1323) |
| Schwefel | 20D | 0/10 (6142) | 0/10 (4188) | 0/10 (2329) |
| Ackley | 5D | 0/10 (18.83) | 9/10 (3.24) | 10/10 (0.00) |
| Ackley | 10D | 0/10 (18.88) | 10/10 (4.00) | 10/10 (0.00) |
| Ackley | 20D | 10/10 (19.04) | 10/10 (5.11) | 10/10 (0.00) |
| Griewank | 5D | 0/10 (67.28) | 10/10 (0.77) | 10/10 (0.05) |
| Griewank | 10D | 0/10 (200.01) | 10/10 (1.32) | 10/10 (0.01) |
| Griewank | 20D | 0/10 (392.07) | 10/10 (2.57) | 10/10 (0.01) |
| Rosenbrock | 5D | 0/10 (17323) | 10/10 (2.16) | 10/10 (0.00) |
| Rosenbrock | 10D | 0/10 (46382) | 0/10 (14.03) | 10/10 (0.00) |
| Rosenbrock | 20D | 0/10 (105177) | 0/10 (53.37) | 10/10 (4.13) |

## Análisis

**Seismic v19 tiene peor rendimiento que SA y CMA-ES en casi todas las funciones y dimensiones.**

El único caso donde Seismic supera a SA es Rastrigin 5D, pero CMA-ES sigue ganando. En Ackley 20D, Seismic tiene 10/10 éxitos (vs 10/10 SA y CMA-ES) pero con mediana 19.04 vs 5.11 y 0.00.

## Comparación con versiones anteriores

| Versión | Rastrigin 5D (30 trials) |
|---------|--------------------------|
| v4 original (2D) | 50/50 success |
| v14 original | 11/30 success |
| v18 genérico | ~10/30 success |
| v19 con morph | ~2-4/10 (low budget) |

**El algoritmo degrade en alta dimensionalidad**, confirmando la hipótesis del "techo dimensional D*" documentada en `improvement_plan.md`.

## Hipótesis sobre la degradación dimensional

1. **Lengthscale fijo**: `lengthscale = scale_factor * 2.0 * (2.0 ** o)` puede ser suboptimal en alta dimensionalidad
2. **Amplitud fija**: `noise_amplitude` no escala con D
3. **Escala del ruido**: En alta D, la superficie RFF tiene más direcciones a explorar, diluyendo la señal

## Conclusiones

1. El morphing entre campos RFF estáticos **no mejora** el rendimiento vs drift temporal continuo
2. El algoritmo funciona bien en 2D (v4: 100% éxito) pero degrada significativamente en 5D+
3. Seismic Descent requiere ajustas sus hiperparámetros (amplitude, lengthscale) según la dimensionalidad
4. CMA-ES domina en 5D+ en este régimen de budget bajo-medio

## Siguientes pasos sugeridos

- Investigar escalado de `noise_amplitude` con la dimensionalidad
- Estudiar lengthscales adaptativos por dimensión
- Explorar swarm size proporcional a D