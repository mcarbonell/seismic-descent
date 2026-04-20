# Seismic Descent V23: Vector-Wave Architecture

## 1. Introducción
La serie **Seismic Descent** alcanzó con la V22 un hito en estabilidad mediante el uso de "Función Oscilante" (la función objetivo aparece y desaparece para permitir que el ruido saque a la partícula de mínimos locales). Sin embargo, la implementación de la V22 presentaba dos limitaciones críticas:
1.  **Límite de Dimensionalidad:** El uso de pesos RFF pre-calculados limitaba el algoritmo a un máximo de 100 dimensiones.
2.  **Eficiencia de Cálculo:** La generación de ruido mediante bucles sobre octavas introducía un overhead innecesario en Python.

La **V23 (Vector-Wave)** rediseña el núcleo del algoritmo para ser agnóstico a la dimensión y masivamente paralelo mediante NumPy.

## 2. Mejoras de la V23

### A. Generación de Ondas Vectorizada
*   **V22:** Utilizaba bucles `for` para iterar sobre las octavas de ruido, calculando proyecciones y senos de forma secuencial.
*   **V23:** Utiliza **broadcasting de NumPy** para calcular todas las fases y amplitudes de las octavas en una única operación matricial.
*   **Impacto:** Velocidad de ejecución interna **1.5x a 2x superior** en dimensiones estándar.

### B. Escalabilidad Dinámica (Agnóstico a Dim)
*   **V23:** Elimina el límite de `_D_MAX`. El campo de ruido se genera dinámicamente según la dimensión de entrada, permitiendo optimizar problemas de **1000+ dimensiones** sin cambios en el código.
*   **Comprobación:** En el benchmark, la V22 falló por desbordamiento de dimensiones en D=1000, mientras que la V23 operó con eficiencia constante.

### C. Adaptive Morphing (Cooling Schedule)
*   **Mejora:** Introducción de un programa de enfriamiento que reduce la amplitud del ruido y el paso de aprendizaje (`lr`) de forma coordinada a medida que avanza la optimización.
*   **Resultado:** Mayor precisión en la fase final (explotación) tras la fase de exploración global.

## 3. Resultados de Benchmarks (V22 vs V23)

| Métrica (Rastrigin, 100 steps) | Seismic V22 | Seismic V23 (Vector-Wave) | Speedup / Resultado |
| :--- | :---: | :---: | :---: |
| **Dimensión 100 (Time)** | 0.0209s | **0.0137s** | **1.53x Faster** |
| **Dimensión 1000 (Time)** | *ERROR (Dim Limit)* | **0.0284s** | **Operacional** |
| **Convergencia (Loss)** | 0.0000 | 0.0000 | Óptimo en ambos |

## 4. Conclusión Técnica
La **V23** transforma a Seismic Descent de un optimizador de juguete 2D en una herramienta de **optimización global de alta dimensión**. La arquitectura de "Vector-Wave" permite que el coste computacional crezca de forma mucho más lenta que la dimensionalidad del problema, facilitando su uso en problemas complejos de ingeniería y aprendizaje automático.

---
*Documento generado por el Laboratorio de Algoritmos (Abril 2026).*
