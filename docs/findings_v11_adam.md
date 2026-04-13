# Hallazgos V11: El fracaso matemático de Adam Optimizer en Descenso Sísmico

Hemos probado a sustituir la actualización de Descenso de Gradiente pelado por **Adam Optimizer** (con parámetros estándar $\beta_1=0.9$, $\beta_2=0.999$, $\text{lr}=0.1$) en `perlin_opt_nd_grf_analytic_v11_adam.py`, asumiendo que el *Momentum* ayudaría a la partícula a fluir mejor a través de la topografía que tiembla. Se utilizaron las lengthscales base (estáticas).

## Resultados (Rastrigin) vs Baseline "crudo" (v8)

| Dimensión | Mediana v8 (GD Crudo) | Mediana v11 (Adam) | Diagnóstico |
|-----------|-----------------------|--------------------|-------------|
| **5D**    | **5.946**             | 46.763             | Desastroso  |
| **10D**   | **30.708**            | 82.084             | Desastroso  |
| **20D**   | **95.020**            | 171.132            | Desastroso  |
| **50D**   | **329.051**           | 455.688            | Desastroso  |

## Diagnóstico: La "Venganza" de Adam Limitando los Terremotos

Los resultados con Adam no solo son peores, sino absolutamente destructivos respecto a la versión de Descenso de Gradiente (GD) estándar. ¿Qué ha fallado?

La respuesta reside en la naturaleza misma de lo que estamos intentando lograr y cómo Adam bloquea precisamente ese comportamiento numérico:

1. **El Terremoto se basa en Gradientes Masivos:** En los primeros pasos de nuestra simulación, la amplitud es violentamente grande (hasta 15 veces el tamaño del gradiente normal). En un GD pelado (`x -= dt * grad`), un gradiente masivo provoca un paso masivo. La partícula es literalmente lanzada a otro "continente" de la función, logrando una enorme exploración global.
2. **Adam Normaliza la Variancia:** La fórmula de Adam hace que los pasos sean de la forma: `x -= dt * (m_hat / sqrt(v_hat))`. El término `sqrt(v_hat)` es esencialmente una normalización RMS del gradiente. **Adam normaliza la magnitud del vector paso**.
3. **El Terremoto es Anulado:** Cuando nuestro motor de ruido RFF genera un tsunami de gradiente de magnitud $1000$ intentando lanzar la partícula, Adam lo ve, calcula su RMS enorme, divide el salto entre ese RMS y condena a la partícula a dar el mismo paso pasito aburrido de tamaño rígido `~dt` (0.1).

Adam fue diseñado precisamente para absorber vibraciones esporádicas y resistir explosiones de gradientes en el entrenamiento de redes neuronales ("Gradient Clipping / Normalization intrínseca"). ¡Pero en Seismic Descent nosotros **queremos** esas explosiones temporales! Adam funciona como unos "amortiguadores de choque" perfectos, asfixiando por completo nuestro mecanismo estocástico de salto temporal.

## Conclusión

El éxito de "Seismic Descent" y de las perturbaciones con grandes varianzas de amplitud asume y requiere un método que sea estricta e ingenuamente proporcional (lineal) a la magnitud actual del paisaje sumado, de la forma $x_{n+1} = x_n - \alpha \nabla f$. 

Quizás un Momentum muy sencillo sin normalización RMS (como Heavy-Ball sin ADAM) ayudaría, pero el componente autorregresivo del denominador en Adam o RMSProp es tóxico para la arquitectura del algoritmo generador de terremotos.
