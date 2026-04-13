# Resultados de Ejecución Analítica: Rastrigin

Estos son los resultados de ejecutar la versión analítica con RFF (`perlin_opt_nd_grf_analytic.py`) sobre la función de prueba **Rastrigin** en 5D, 10D, 20D y 50D.

```bash
python perlin_opt_nd_grf_analytic.py
```

## Resultados

### Rastrigin 5D — 30 pruebas | Umbral $f<5.0$ | Budget = 3000 evals

| Algoritmo | Éxitos | Media | Mediana | Presupuesto | Tiempo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seismic+RFF (Analytic)** | 0/30 | 9.298 | 9.481 | 3000 pasos | 12.52s |
| **SA** | 2/30 | 14.236 | 13.863 | 3000 pasos | 2.01s |
| **CMA-ES** | 18/30 | 5.936 | 4.975 | 3000 evals | 11.72s |

### Rastrigin 10D — 30 pruebas | Umbral $f<10.0$ | Budget = 5500 evals

| Algoritmo | Éxitos | Media | Mediana | Presupuesto | Tiempo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seismic+RFF (Analytic)** | 0/30 | 40.319 | 41.704 | 5500 pasos | 24.86s |
| **SA** | 0/30 | 66.774 | 69.806 | 5500 pasos | 5.00s |
| **CMA-ES** | 9/30 | 14.029 | 13.432 | 5500 evals | 25.64s |

### Rastrigin 20D — 10 pruebas | Umbral $f<20.0$ | Budget = 10500 evals

| Algoritmo | Éxitos | Media | Mediana | Presupuesto | Tiempo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seismic+RFF (Analytic)** | 0/10 | 112.983 | 114.024 | 10500 pasos | 20.04s |
| **SA** | 0/10 | 194.744 | 193.978 | 10500 pasos | 5.58s |
| **CMA-ES** | 1/10 | 31.441 | 31.341 | 10500 evals | 19.74s |

### Rastrigin 50D — 5 pruebas | Umbral $f<50.0$ | Budget = 25500 evals

| Algoritmo | Éxitos | Media | Mediana | Presupuesto | Tiempo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Seismic+RFF (Analytic)** | 0/5 | 324.560 | 327.919 | 25500 pasos | 33.21s |
| **SA** | 0/5 | 607.274 | 614.777 | 25500 pasos | 13.82s |
| **CMA-ES** | 0/5 | 122.977 | 110.440 | 25500 evals | 39.54s |
