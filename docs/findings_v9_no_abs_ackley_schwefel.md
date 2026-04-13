# Hallazgos V9: Validando la eliminación de 'abs()' en Ackley y Schwefel

Para confirmar que la mejora masiva vista en Rastrigin al eliminar el `abs()` de la amplitud no fue un caso aislado de "sobreajuste" a esa función, hemos probado esta variante (inversión de polaridad) en las otras dos funciones de prueba clave: **Ackley** y **Schwefel**.

Se han utilizado `benchmark_ackley_no_abs.py` y `benchmark_schwefel_no_abs.py` con presupuestos justos idénticos a las pruebas base (v5 y v6).

## 1. Benchmarks en Ackley (f < 1.0)

| Dimensión | Métrica (Seismic) | Con *abs* (base) | Sin *abs* (nuevo) | Mejora / Diagnóstico |
|-----------|-------------------|------------------|-------------------|----------------------|
| **2D**    | Éxitos            | 7/50             | **31/50**         | ¡Mejora colosal!     |
|           | Mediana           | 18.562           | **0.410**         | Resolvió la función. |
| **5D**    | Mediana           | 19.455           | **18.961**        | Mejora ligera.       |
| **10D**   | Mediana           | 19.423           | **19.011**        | Mejora ligera.       |
| **20D**   | Mediana           | 19.768           | **19.411**        | Mejora ligera.       |

**Análisis de Ackley:** Quitar el `abs()` ha logrado que el algoritmo **resuelva el terrible problema de la meseta exterior en 2D**, llevando la tasa de éxito de casi el suelo (7/50) a más de la mitad (31/50), con una mediana excelente de `0.410`. En dimensiones más altas volvemos a caer en la trampa de la meseta (propia del uso de gradientes numéricos en superficies planas), pero aún así los valores finales residuales son estrictamente mejores.

---

## 2. Benchmarks en Schwefel (Engañoso y enorme)

| Dimensión | Métrica (Seismic) | Con *abs* (base) | Sin *abs* (nuevo) | Mejora / Diagnóstico |
|-----------|-------------------|------------------|-------------------|----------------------|
| **5D**    | Mediana           | 1097             | **987**           | ~100 puntos mejor.   |
| **10D**   | Mediana           | 2044             | **1896**          | ~150 puntos mejor.   |
| **20D**   | Mediana           | 4271             | **3507**          | ~700 puntos mejor.   |

**Análisis de Schwefel:** La mejora aquí escala progresivamente con las dimensiones (y a su vez todo Seismic+RFF es estrictamente mejor que CMA-ES en estas corridas particulares evaluando a todos los algoritmos por igual, aunque ninguno la resuelva por culpa del estricto presupuesto). La inversión total de polaridad evita quedar atrapado tan fácilmente en falsos valles.

---

## Conclusión Final sobre `abs()`

Es oficial: **Limitar el oscilador de amplitud usando valor absoluto es invariablemente perjudicial**. 
Cruzar el cero periódicamente —y por lo tanto, forzar al campo de perturbación a invertir todas sus pendientes temporalmente y convertir picos en valles— no solo es gratis computacionalmente, sino que produce un "terremoto" mucho más devastador contra los mínimos locales en *todas las dimensiones* y en *todas las topologías de paisajes probadas*.
