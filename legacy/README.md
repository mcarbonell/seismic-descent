# Archivo Histórico de Iteraciones (Legacy)

Este directorio contiene las implementaciones cronológicas desarrolladas durante las fases de investigación y descubrimiento de **Seismic Descent**.

> [!NOTE]
> De acuerdo con la **Regla de Oro** de [GEMINI.md](../../GEMINI.md), todos estos archivos históricos se conservan íntegros y sin modificaciones, sirviendo como registro experimental reproducible. Para el uso en producción y desarrollo actual, utiliza el paquete modular [`src/seismic_descent`](../../src/seismic_descent).

---

## Estructura del Histórico

### 1. `perlin_opt/` — Fase Inicial y Escalado ND (v1 – v17)
Contiene las primeras versiones del algoritmo antes de la normalización universal:

- `perlin_opt.py`: Algoritmo original 2D basado en Perlin Noise clásico.
- `perlin_opt_nd.py`: Primera extensión a N-Dimensiones mediante value noise.
- `perlin_opt_nd_fairbench.py`: Benchmarks comparativos con presupuesto de evaluaciones igualado.
- `perlin_opt_nd_grf.py`: Introducción de campos Gaussianos mediante Random Fourier Features (RFF).
- `perlin_opt_nd_grf_analytic*.py`: Salto cualitativo a gradientes analíticos $\mathcal{O}(1)$:
  - `analytic.py` (v7): Gradiente analítico del campo RFF.
  - `no_abs.py` (v8–v9): Descubrimiento del beneficio de amplitudes negativas (inversión de montañas en embudos gravitacionales).
  - `v10`: Lengthscales dinámicas.
  - `v11_adam`: Experimento fallido con Adam Optimizer (la normalización RMS amortigua los sismos).
  - `v12_swarm`, `v13_swarm_D`: Vectorización masiva del enjambre ($N$ partículas compartiendo el terreno RFF).
  - `v14_cycles`: Parametrización asintótica estricta a 10 ciclos sísmicos (código base standard).
  - `v15_reactive`: Experimento con disparo sísmico condicionado por estancamiento (Bang-Bang).
  - `v16_momentum`: Experimento con Heavy-Ball momentum (efecto honda perjudicial).
  - `v17_temporal_octaves`: Sismos fractales con series de Fourier temporales.
- `benchmark_ackley.py`, `benchmark_schwefel.py`, `benchmark_budgets.py`: Scripts de evaluación de esta era.

### 2. `seismic_versions/` — Normalización y Arquitectura Universal (v18 – v22)
Contiene la evolución hacia la arquitectura actual y la integración con Deep Learning:

- `seismic_descent_v18.py`: Introducción del hipercubo normalizado $[-1, 1]^D$.
- `seismic_descent_v19.py`: Refinamiento del scheduling y validación de ergodicidad.
- `seismic_descent_v20.py`: **Arquitectura campeona actual**. Introduce la normalización L2 del vector gradiente y el tamaño de paso cíclico desacoplado `dt(t) = dt_base * abs(sin(t * mult))`, permitiendo hiperparámetros universales para todas las funciones.
- `seismic_descent_v21.py`: Experimento con aceptación subrogada (fracasado por rigidez).
- `seismic_descent_v22.py`: Experimento con función oscilante (vanishing objective).
- `seismic_descent_vmorph.py`: Variante con morphing topológico continuo para benchmarks COCO.
- `seismic_optimizer.py`: Primera versión del optimizador para redes neuronales en PyTorch.
- `benchmark_suite_v*.py`: Suites de evaluación correspondientes a cada hito.
- `benchmark_mnist.py`, `benchmark_adaptive_mnist.py`, `final_benchmark_*.py`: Pruebas del optimizador en clasificación de dígitos MNIST.
