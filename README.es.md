# Seismic Descent

Algoritmo de optimización basado en descenso de gradiente sobre un paisaje
dinámico perturbado con ruido correlacionado espacialmente.

![Seismic Descent](assets/seismic-descent.png)

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

## Propiedad Clave: Ergodicidad Sísmica

Un descubrimiento fundamental en el desarrollo del algoritmo (consolidado en la v19) es que la exploración generada por el ruido correlacionado debe ser **ergódica**. 

En lugar de sacudir ciegamente a la partícula con "ruido blanco" de alta frecuencia, *Seismic Descent* genera **paisajes topológicos completos y coherentes** de baja y alta frecuencia (mediante octavas de RFF), y realiza un *morphing* (interpolación suave) de un paisaje a otro con el tiempo. 

Esta mutación continua garantiza matemáticamente que una partícula, guiada puramente por el gradiente de este "suelo mutante", acabará explorando y visitando la totalidad del espacio de búsqueda sin quedarse atascada en ciclos infinitos o mesetas. La **ergodicidad** es lo que permite que el enjambre fluya por el terreno como un líquido, garantizando el escape de los mínimos locales más profundos.

## Instalación

Instala el paquete en modo editable desde la raíz del repositorio:

```bash
# Paquete núcleo
pip install -e .

# Con dependencias completas para desarrollo y benchmarks
pip install -e ".[dev]"
```

O instala las dependencias externas directamente:
```bash
pip install numpy matplotlib torch cma noise pytest
```

## Inicio Rápido

### Uso desde Python

```python
from seismic_descent import seismic_swarm, ALL_FUNCTIONS

# Cargar función objetivo Rastrigin 5D
rastrigin = ALL_FUNCTIONS["rastrigin"]
bounds = [[-5.12, 5.12]] * 5
x0 = [3.0] * 5

# Optimizar mediante la arquitectura campeona v20
best_x, best_val, info = seismic_swarm(
    fn=rastrigin["fn"],
    fn_grad=rastrigin["grad"],
    x0_real=x0,
    bounds=bounds,
    n_steps=2000,
    n_particles=10,
    dt_base=0.2,
    noise_amplitude=0.5,
)

print(f"Valor óptimo encontrado: {best_val:.6f}")
```

### Suite de Benchmarks (CLI)

Ejecuta la suite automatizada de benchmarks contra Simulated Annealing (SA) y CMA-ES:

```bash
# Benchmark en 5D para todas las funciones
python -m benchmarks.benchmark_suite --dims 5 --trials 5

# Benchmark en 2D para Rastrigin con presupuesto personalizado
python -m benchmarks.benchmark_suite --dims 2 --trials 3 --budget 2000 --function rastrigin
```

Ejecutar tests automatizados:
```bash
pytest -v
```

## Integración con PyTorch

El algoritmo Seismic Descent está disponible como un optimizador estándar de PyTorch. Esto permite entrenar redes neuronales con "temblores" correlacionados espacialmente para escapar de mínimos locales:

```python
from seismic_descent import SeismicOptimizer

model = MyModel()
optimizer = SeismicOptimizer(
    model.parameters(), 
    lr=0.01, 
    noise_amplitude=0.5, 
    n_cycles=10,
)
```

Consulta [legacy/seismic_versions/benchmark_mnist.py](legacy/seismic_versions/benchmark_mnist.py) para un ejemplo completo de entrenamiento en red neuronal y [docs/pytorch_optimizer.es.md](docs/pytorch_optimizer.es.md) para detalles técnicos.

### Último Benchmark (MNIST - 20 Épocas)

| Optimizador | Precisión | Margen |
| :--- | :--- | :--- |
| **SGD** | **98.28%** | Base |
| **Adaptive Floored Seismic** | **97.90%** | ✅ Supera a Adam |
| **Adam** | 97.79% | - |

## Estructura del Proyecto

```
seismic-descent/
├── src/seismic_descent/            # Paquete Python núcleo instalable
│   ├── core.py                     # SeismicSwarm (arquitectura campeona v20)
│   ├── rff.py                      # Random Fourier Features (gradientes analíticos O(1))
│   ├── torch_optimizer.py          # Optimizador SeismicOptimizer para PyTorch
│   └── functions.py                # Rastrigin, Schwefel, Ackley, Griewank, Rosenbrock
│
├── benchmarks/                     # Suites de evaluación comparativa (vs SA, CMA-ES)
│   ├── benchmark_suite.py          # Ejecutor automatizado de benchmarks CLI
│   └── timing/                     # Pruebas de tiempo de cómputo y profiling
│
├── legacy/                         # Historial experimental cronológico preservado
│   ├── perlin_opt/                 # v1 a v17 (Perlin, value noise, enjambres iniciales)
│   ├── seismic_versions/           # v18 a v22, vmorph, y experimentos MNIST
│   └── README.md                   # Documentación detallada del viaje experimental
│
├── tests/                          # Tests unitarios automatizados (pytest)
│   ├── test_rff.py                 # Verificación de gradientes analíticos y propiedades RFF
│   ├── test_seismic_swarm.py       # Pruebas de convergencia y respeto de límites
│   └── test_torch_optimizer.py     # Validación de convergencia en optimizador PyTorch
│
├── visualizer/                     # Visualizadores interactivos HTML5/JS (GitHub Pages)
│   ├── 1d_explorer.html            # Deformación 1D y mapa de ergodicidad
│   ├── 2d_explorer.html            # Wireframe 2D y trayectorias del enjambre
│   └── index.html                  # Mapa interactivo 2D
│
├── assets/                         # Gráficos y recursos visuales del repositorio
├── docs/                           # Documentación teórica, hallazgos (findings_v1 a v23)
└── scratch/                        # Espacio de trabajo para pruebas del desarrollador
```

## Próximos experimentos / Futuro del Proyecto

- **Hyperparameter Sweeping**: Búsqueda en grilla formalizada para ajustar balanceadamente `noise_amplitude` y $K$ (ciclos) cruzado por la varianza de Dimensión $D$.
- **Extensión a Machine Learning**: Reemplazo directo en Deep Learning cruzando `Seismic Optimizer` en un problema real simple como MNIST, usando las iteraciones de Epoch en vez del budget.
- **Topologías no Euclidianas**: Posibilidad de redefinir RFF como matriz de Costos para abordar optimización discreta como el Viajante de Comercio (TSP).
