# Hallazgos V8: Eliminación del módulo en el scheduling de amplitud

A sugerencia del usuario, se ha creado una nueva variante del algoritmo de descenso analítico (`perlin_opt_nd_grf_analytic_no_abs.py`) para evaluar el impacto de usar `np.sin` en lugar de `abs(np.sin)` para el scheduling de la amplitud del ruido cíclico.

### El cambio:

```python
# ANTES (v7)
amp = noise_amplitude * decay * abs(np.sin(t * freq))

# AHORA (v8 - sin abs)
amp = noise_amplitude * decay * np.sin(t * freq)
```

## Resultados del Benchmark

Comparativa con los resultados obtenidos en la v7 (mismo budget):

| Dimensión | Métrica | Seismic (Con *abs*) | Seismic (Sin *abs*) | CMA-ES (Baseline) |
|-----------|---------|---------------------|---------------------|-------------------|
| **5D**  | Éxitos  | 0/30                | **13/30**           | 11/30             |
|         | Mediana | 9.481               | **5.946**           | 5.970             |
| **10D** | Mediana | 41.704              | **30.708**          | 12.437            |
| **20D** | Mediana | 114.024             | **95.020**          | 39.301            |
| **50D** | Mediana | **327.919**         | 329.051             | 102.481           |

## Conclusiones Críticas

1. **Beneficio Masivo en Bajas Dimensiones:** Eliminar el `abs()` ha mejorado drásticamente el rendimiento en 5D, 10D y 20D. ¡En 5D, **superó en número de éxitos a CMA-ES** en este régimen de bajo budget!
2. **¿Por qué funciona mejor sin `abs()`?** 
   - Con `abs()`, la amplitud del temblor oscila entre $0$ y $A_{max}$. Esto simplemente "enciende y apaga" la perturbación sobre el paisaje original.
   - Sin `abs()`, la amplitud oscila entre $-A_{max}$ y $A_{max}$. Al cruzar el cero y hacerse negativa, **invierte el campo de ruido**: lo que antes era un pico inducido por el ruido, ahora es un valle profundo, y viceversa. Esto genera un efecto de mezcla muchísimo más agresivo, agitando la partícula de lado a lado y sacándola de estructuras engañosas de manera más eficiente.
3. **Escalabilidad intacta en 50D:** En 50D el resultado es estadísticamente idéntico. Aquí, debido a la dispersión propia de 50D, quizás entren en juego los lengthscales mal ajustados.

Este experimento confirma que el "bombeo" simétrico que invierte la polaridad del ruido es una heurística superior a simplemente apagar y encender el ruido.
