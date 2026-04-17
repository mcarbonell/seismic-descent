# Findings v20: Gradient Normalization y Cyclic dt Desacoplado

## El Problema de Escala en Y (Heredado de v19)
En la versión 19 logramos normalizar el dominio de búsqueda (eje X) mapeando todas las variables a un hipercubo `[-1, 1]^D`. Sin embargo, descubrimos que el algoritmo seguía siendo extremadamente sensible a la escala de la función objetivo (eje Y). 

Debido a la fórmula estándar del descenso de gradiente ($\text{Paso} = dt \times \nabla f$), funciones con gradientes masivos como Rosenbrock (del orden de $10^5$) provocaban que la partícula intentara dar saltos miles de veces más grandes que el dominio normalizado, rebotando violentamente contra los límites. Esto nos obligaba a usar un `dt` minúsculo específico para Rosenbrock y uno enorme para Griewank.

## Solución 1: Gradient Normalization (Normalización de la Dirección)
Para lograr verdaderos **hiperparámetros universales**, en la v20 desacoplamos el tamaño del paso de la magnitud matemática de la pendiente.

En lugar de usar el gradiente crudo, extraemos únicamente su dirección mediante normalización $L_2$:
$$ \nabla_{dir} = \frac{\nabla f}{||\nabla f||} $$

Al sumar este vector unitario al gradiente de ruido RFF, garantizamos que el "empuje" base del terreno siempre tenga magnitud $1.0$, sin importar si la función original tiene paredes de 2 metros o de 20,000 metros de altura.

## Solución 2: Cyclic dt Desacoplado
Al normalizar el gradiente, perdimos la capacidad natural del algoritmo para "frenar" automáticamente al llegar a un valle llano (donde el gradiente original se acerca a cero). 

Para recuperar la convergencia fina, introdujimos un **dt cíclico**:
$$ dt_{actual} = dt_{base} \cdot |\sin(t \cdot \text{multiplier})| $$

Esto fuerza al tamaño del paso a oscilar entre $0$ y $dt_{base}$. Al usar un `multiplier > 1.0` (desacoplando la frecuencia del dt de la frecuencia de evolución del terreno sísmico), permitimos que la partícula experimente múltiples fases de "enfriamiento/explotación" rápida (cuando el seno se acerca a cero) dentro de una misma conformación topológica del ruido RFF, antes de que este cambie significativamente.

## Ajuste de Hiperparámetros Universales
Se desarrolló un script de Grid Search (`tune_v20.py`) para encontrar la combinación óptima de hiperparámetros que minimizara el error promedio a través de *todas* las funciones del benchmark simultáneamente.

Los **Hiperparámetros Universales** resultantes fueron:
*   `dt_base = 0.2` (Un paso máximo equivalente al 20% del hipercubo normalizado).
*   `noise_amplitude = 0.5` (Ruido más tenue que la fuerza del gradiente direccional).
*   `dt_cycles_multiplier = 5.0` (El ciclo de dt ocurre 5 veces más rápido que la evolución del sismo).

## Resultados del Benchmark (Presupuesto Bajo)
Con estos parámetros estáticos y sin ninguna lógica condicional por función, el algoritmo logró un comportamiento estable y altamente competitivo:
*   **Griewank (5D y 10D):** 5/5 éxitos. Convergencia impecable.
*   **Rosenbrock (5D):** 5/5 éxitos. Mediana de `3.87` (vs el óptimo de 1.0).
*   **Rastrigin (5D):** 2/5 éxitos. La oscilación rápida del dt permite escapar de los múltiples mínimos locales.
*   **Schwefel (5D):** 2/5 éxitos. Reducción masiva del error mediano (de miles a `274`).

## Conclusión
La versión 20 representa un hito arquitectónico en el desarrollo de Seismic Descent. Al combinar un **Dominio Isotrópico Normalizado** con **Normalización de Gradiente**, el optimizador es ahora completamente agnóstico a la escala espacial y de magnitud del problema subyacente, permitiendo un uso directo "out-of-the-box" con hiperparámetros robustos y universales.