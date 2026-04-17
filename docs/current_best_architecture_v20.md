# Arquitectura Actual Óptima: Seismic Descent v20

Tras múltiples iteraciones y experimentos (v1 hasta v22), la versión **v20** se ha consolidado como la arquitectura más robusta, estable y de propósito general para la optimización matemática usando enjambres sísmicos.

## Principios de la Arquitectura Campeona

1. **Dominio Isotrópico Normalizado (Eje X):**
   El optimizador opera internamente en un hipercubo estricto `[-1, 1]^D`. Las coordenadas se mapean dinámicamente al dominio real (definido por `bounds`) justo antes de evaluar la función objetivo y su gradiente. Esto permite un `lengthscale` universal para el ruido RFF.

2. **Normalización de Gradiente L2 (Eje Y):**
   Desacoplamos el tamaño del paso de la magnitud de la función. Se extrae únicamente la **dirección** del gradiente ($\nabla_{dir} = \nabla / ||\nabla||$). Esto evita que funciones con paredes extremas (como Rosenbrock, con gradientes de $10^5$) hagan explotar el algoritmo dentro del espacio normalizado.

3. **dt Cíclico Desacoplado (Cyclic Learning Rate):**
   Para recuperar la capacidad de explotación fina (que se pierde al normalizar el gradiente), el tamaño del paso oscila mediante `dt = dt_base * abs(sin(t * multiplier))`. 

## Hiperparámetros Universales (Tuned)
Gracias a la doble normalización (dominio y gradiente), logramos por primera vez utilizar una **configuración única de hiperparámetros para todas las funciones** (Griewank, Rosenbrock, Rastrigin, Schwefel, Ackley), sin lógica condicional:

*   **`dt_base = 0.2`**: Paso máximo bastante agresivo (20% del hipercubo normalizado).
*   **`noise_amplitude = 0.5`**: Fuerza del ruido relativa al gradiente direccional (que tiene norma 1.0).
*   **`dt_cycles_multiplier = 5.0`**: La oscilación del tamaño del paso ocurre 5 veces más rápido que la evolución del terreno sísmico, permitiendo múltiples fases de explotación profunda en cada conformación del ruido.

## Implementación de Referencia
*   **Motor Optimizado:** `seismic_descent_v20.py`
*   **Script de Benchmark:** `benchmark_suite_v20.py`
*   **Script de Tuning:** `tune_v20.py`

## Siguientes Pasos Futuros
La arquitectura v20 servirá como base para futuras investigaciones. Las áreas potenciales de mejora incluyen:
*   Mecanismos de momentum direccional (basados en v16) integrados en el espacio de gradiente normalizado.
*   Interacción topológica de enjambre (Swarm Intelligence basada en v12/v13) aplicada en el hipercubo normalizado.