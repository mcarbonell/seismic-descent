# Hallazgos V10: Análisis del escalamiento de Lengthscales ($\sqrt{D}$)

He consolidado los cambios anteriores de `abs()` en git y he procedido a implementar el escalamiento de la constante de longitud de los features de RFF en `perlin_opt_nd_grf_analytic_v10.py`. 

La regla introducida dictaba adaptar las octavas dinámicamente según la dimensión para evitar que el ruido se vuelva prácticamente ruido blanco:
```python
l_base = 5.0 * np.sqrt(D) / 4.0
# Octavas: l_base/4, l_base/2, l_base, l_base*2
```

## Resultados vs Versión Base sin *abs* (v8)

| Dimensión | Métrica | Seismic (Sin *abs*, l fijo) | Seismic (v10 - $\sqrt{D}$) | Diagnóstico |
|-----------|---------|-----------------------------|----------------------------|-------------|
| **5D**    | Mediana | **5.946**                   | 9.495                      | Empeora     |
| **10D**   | Mediana | **30.708**                  | 37.923                     | Empeora     |
| **20D**   | Mediana | **95.020**                  | 105.055                    | Empeora     |
| **50D**   | Mediana | 329.051                     | **300.744**                | Mejora      |

## Diagnóstico Matemático ¡Hemos hecho lo contrario!

Los resultados fueron peores en dimensiones bajas-medias (5D a 20D) y solo mejoraron ligeramente en 50D. ¿Por qué? ¡La matemática de la propuesta nos falló!

En la versión original no escalada usábamos `lengthscales` predeterminados de **`[2, 4, 8, 16]`**.
Con la fórmula que hemos introducido `l_base = 5.0 * sqrt(D) / 4.0`:
- En **5D**: `l_base ≈ 2.79`. Nuevas octavas: `[0.69, 1.39, 2.79, 5.59]`
- En **20D**: `l_base ≈ 5.59`. Nuevas octavas: `[1.39, 2.79, 5.59, 11.18]`
- En **50D**: `l_base ≈ 8.83`. Nuevas octavas: `[2.20, 4.41, 8.83, 17.67]`

El problema es evidente: **Las nuevas reglas hicieron que los lengthscales fuesen MÁS PEQUEÑOS que antes en casi todas las dimensiones probadas**. Un `lengthscale` más pequeño se traduce en una frecuencia más alta, lo que significa que irónicamente **hicimos el ruido más caótico y blanco**, contraviniendo el propósito original de la mejora ("hacer que las montañas fuesen de continentes enteros"). Solo a partir de D=50 logramos rozar el perfil original, y por ende, es ahí donde único ha mejorado.

Si queremos ampliar las ondas con respecto al baseline original de `[2, 4, 8, 16]` (cuyo `l_base` asumido es 8), deberíamos usar quizá un escalado como `l_base = 8.0 * np.sqrt(D) / np.sqrt(10)` (donde 10D es el punto de control donde empieza a fallar antes). De esa forma `20D` daría un base de `~11.3` (octavas de `[2.8, 5.6, 11.3, 22]`) — efectivamente más vastas.

¿Te parece bien si para la próxima versión (**Adam Momentum**) simplemente escalamos los lengthscales usando una raíz cuadrada pero basándonos en tu configuración original `[2, 4, 8, 16]` multiplicada enteramente por un factor `sqrt(D/10)`, o prefieres abandonar el escalado por ahora y solo aplicar Adam al v8 estático?
