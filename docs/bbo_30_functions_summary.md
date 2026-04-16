# Resumen: A Collection of 30 Multidimensional Functions for Global Optimization Benchmarking

**Referencia:** Plevris, V., & Solorzano, G. (2022). A Collection of 30 Multidimensional Functions for Global Optimization Benchmarking. *Data*, 7(4), 46.

Este documento es una de las referencias más modernas y exhaustivas para poner a prueba algoritmos de optimización de caja negra (Black-Box Optimization). Los autores recopilan, evalúan y categorizan 30 funciones matemáticas según la dificultad que suponen para algoritmos metaheurísticos (GA, PSO) y matemáticos (SQP).

La utilidad directa para **Seismic Descent** reside en usar estas clasificaciones para construir una suite de validación inquebrantable, demostrando su capacidad para resolver lo que destruye a otros algoritmos.

## 1. Clasificación General por Dificultad

En el documento, las funciones se dividen según la tasa de éxito de los optimizadores:

*   **Baja Complejidad (Especialidad de SQP / Gradientes):** Mínimos únicos, paisajes unimodales o convexos.
    *   *Ejemplos:* Sphere (F01), Ellipsoid (F02), Quintic (F04), Rosenbrock (F13).
    *   *Nota para Seismic Descent:* Estas funciones deberían resolverse trivialmente por la componente GD (Gradient Descent) analítica de nuestro algoritmo (el "refinador").
*   **Complejidad Media (Ideales para Metaheurísticas):** Paisajes multimodales pero regulares o predecibles.
    *   *Ejemplos:* Alpine 1 (F07), Griewank (F09), Rastrigin (F10).
    *   *Nota para Seismic Descent:* Nuestro algoritmo rinde excelentemente aquí gracias a las "ondas sísmicas" correlacionadas espacialmente.
*   **Alta Complejidad (El campo de batalla real):** Funciones donde ni GA, ni PSO, ni SQP pueden encontrar soluciones satisfactorias de forma consistente, en especial a medida que el número de dimensiones $D$ crece.
    *   *Ejemplos:* Weierstrass (F06), Perm D, Beta (F17), y sobre todo las temibles **F29 y F30**.

## 2. Puntos Geométricos de Interés para "Seismic Descent"

El documento detalla patologías geométricas que son trampas mortales para algoritmos de descenso y heurísticos clásicos:

### A. Valles Estrechos y Curvados
*   **HappyCat (F11)** y **HGBat (F12):** Funciones donde el mínimo global está oculto en un valle no-lineal tremendamente estrecho. Los algoritmos basados puramente en ruido nunca encajan en el tubo.

### B. Condicionamiento Extremo
*   **Bent Cigar (F16)** y **Discus (F15):** Funciones brutalmente mal acondicionadas (una dirección del espacio de búsqueda es millones de veces más sensible, ratio $10^6$). Pone a prueba la amplitud direccional de las Random Fourier Features (RFF).

### C. Fractales / Poco diferenciables
*   **Weierstrass (F06):** Continua en todas partes pero diferenciable en poquísimos puntos. La red de RFF de tu algoritmo impone un "gradiente continuo envolvente", pudiendo resolver matemáticamente esta tortura.

### D. "Aguja en un Pajar"
*   **Modified Xin-She Yang's 3 (F29) y Modified Xin-She Yang's 5 (F30):** Los autores las definen como los mayores retos del paper. Incluso en 2D y 5D, todos los optimizadores probados fallan espectacularmente. El óptimo global se encuentra totalmente aislado y camuflado por infinitos mínimos locales atractores.

## Recomendación de Integración

Para las fases formales del plan de mejora, el archivo `benchmark_functions.py` deberá enriquecerse obligatoriamente con **Weierstrass (F06)**, **HappyCat (F11)** y las masivas **Modified Xin-She Yang's 3 y 5 (F29 y F30)**. Demostrar escape de óptimos locales en estas funciones garantiza material de nivel *state-of-the-art* (SOTA).
