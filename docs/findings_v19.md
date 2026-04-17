# Findings v19: Normalización de Dominio Isotrópico (Domain Normalization)

## Objetivo de la v19
El objetivo fue independizar los hiperparámetros del optimizador (`dt`, `noise_amplitude`) de los límites del dominio de búsqueda (eje X). Se implementó un mapeo donde el enjambre opera internamente en un hipercubo inmaculado `[-1, 1]^D`, traduciendo las coordenadas al dominio real justo antes de evaluar la función objetivo, y escalando los gradientes de vuelta mediante la Regla de la Cadena ($\nabla_{norm} = \nabla_{real} \cdot \text{half\_range}$).

## Resultados y Problemas Descubiertos
*   **Éxito Parcial:** El tamaño del "terremoto" (RFF) se estandarizó exitosamente. El lengthscale base de `0.4` funciona de forma idéntica sin importar si el dominio original es pequeño (Rosenbrock, `[-5, 5]`) o masivo (Schwefel, `[-500, 500]`).
*   **Problema Crítico de Escala en Y (Magnitud del Gradiente):** Al intentar usar un `dt` universal, los resultados fueron desastrosos. Descubrimos que normalizar el eje X no es suficiente si el eje Y (los valores de la función) tiene escalas radicalmente distintas.
    *   En Griewank, los gradientes son moderados, por lo que `dt = 0.01` genera un avance razonable.
    *   En Rosenbrock, las paredes del "valle" tienen pendientes extremas (gradientes del orden de $10^5$). Al multiplicar ese gradiente inmenso por el `dt`, el paso intentado resulta ser miles de veces más grande que la caja `[-1, 1]`. Las partículas chocan violentamente contra los límites del espacio normalizado, provocando un fallo total en la convergencia.

## Conclusión y Siguientes Pasos
Normalizar el "ancho" (X) resolvió el problema del ruido, pero evidenció la necesidad de normalizar la "altura" (Y) o, más bien, la "fuerza" del gradiente.
Para lograr verdaderos hiperparámetros universales, el tamaño del paso debe desacoplarse de la magnitud matemática de la pendiente. 

**Propuesta para v20:**
1.  **Gradient Normalization:** Extraer solo la dirección del gradiente ($\frac{\nabla}{||\nabla||}$) ignorando su magnitud, para que la altura de la función no destruya la escala de movimiento.
2.  **Cyclic dt:** Dado que el paso ahora sería discreto y constante, introducir una variación cíclica multiplicando el paso base por `abs(sin(t))`, permitiendo exploración dinámica y explotación detallada dentro de cada ciclo sísmico.