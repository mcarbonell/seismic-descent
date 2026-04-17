# Experimentos Arquitectónicos: v21 y v22 (Aceptación Discreta y Función Oscilante)

Tras consolidar la **v20** (Gradient Normalization L2 + dt Cíclico) como la arquitectura más robusta y universal hasta la fecha, intentamos dos variaciones conceptualmente elegantes pero que resultaron empíricamente fallidas. Documentar estos fallos es crucial para entender la dinámica del enjambre sísmico.

## Experimento 1: v21 - Paisaje Subrogado y Aceptación Discreta (Greedy)
**Concepto:** En lugar de mover la partícula ciegamente sumando gradientes, creamos una función "subrogada" combinando la función objetivo normalizada históricamente al rango `[0, 1]` y el valor escalar del campo RFF. La partícula solo aceptaba un paso si el nuevo punto mejoraba (disminuía) el valor de esta función combinada.

**Por qué falló:**
1. **Pérdida de Inercia Sísmica:** El algoritmo se volvió rígido. Al rechazar pasos temporalmente "malos" dictados por la ola del sismo, las partículas dejaron de "surfear" el terreno y se quedaron congeladas en mínimos locales minúsculos del ruido.
2. **Flattening por Rango Histórico:** Al usar un registro global de $f_{max}$ y $f_{min}$, las diferencias sutiles en el fondo de los valles se aplastaron a valores cercanos a cero frente a las alturas masivas del inicio de la exploración, haciendo que el ruido dominara completamente la decisión de aceptación al final de la búsqueda.

## Experimento 2: v22 - Función Oscilante (Vanishing Objective)
**Concepto:** Normalizamos ambos gradientes (función y ruido) a magnitud $1.0$. Hicimos el ruido constante en el tiempo, y en su lugar, hicimos que la influencia de la función objetivo oscilara multiplicándola por `(sin(t) + 1) / 2`. La idea era que la función "desapareciera" cíclicamente para liberar a la partícula y luego la "atrapara" de nuevo.

**Por qué falló:**
Cuando el peso de la función se acercaba a $0.0$, la partícula quedaba a merced de un campo de ruido aleatorio estático. Sin la gravedad de la función objetivo guiando la exploración macroscópica, la partícula caía en el mínimo local más cercano del mapa de ruido. Al volver a "aparecer" la función, la partícula se encontraba perdida en un valle incorrecto. Este ciclo de "ceguera periódica" destruyó la convergencia.

## Conclusión General
Estos experimentos demostraron dos reglas de oro para el diseño de Seismic Descent:
1. **La Función Objetivo debe ser el faro inmutable:** Su gradiente direccional debe guiar a la partícula el 100% del tiempo con peso constante.
2. **El Ruido debe ser el elemento dinámico (vibración):** El ruido RFF no debe ser un mapa estático ni un criterio estricto de aceptación, sino una fuerza oscilatoria que empuja incondicionalmente a la partícula, permitiéndole fluir cuesta arriba temporalmente para escapar de trampas locales.

La **v20** sigue siendo la arquitectura campeona.