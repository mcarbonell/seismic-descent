# Hallazgos V15: Terremotos Reactivos (Triggered Earthquakes)

En esta iteración pusimos a prueba la hipótesis de aplicar "control Bang-Bang" al sismo: el terremoto solo se detona si las partículas se han estancado (la mejora del `best_val` en una ventana temporal cae por debajo de `1e-3`). 
Una vez estancadas, se dispara exactamente **1 ciclo sinusoidal completo** de perturbaciones, tras el cual el sismo se apaga (`amp=0`) permitiendo que las partículas caigan libremente hacia el fondo empujadas únicamente por el gradiente real del terreno, hasta volver a estancarse.

## Resultados vs V14 (Ciclos Continuos)

| Dimensión | Mediana v14 (Continuo 10 ciclos) | Mediana v15 (Trigger Reactivo) | 
|-----------|----------------------------------|--------------------------------|
| **5D**    | **6.174**                        | 6.582                          | 
| **10D**   | **32.709**                       | 33.382                         | 
| **20D**   | **94.410**                       | 96.480                         | 
| **50D**   | **323.640**                      | 327.075                        | 

## Diagnóstico del Fracaso del Modo Reactivo

El enfoque reactivo rindió marginalmente peor en absolutamente todas las dimensiones. De nuevo, las métricas son similares porque el tiempo de evaluación es tan corto que los algoritmos terminan ejecutando cantidades de perturbación parecidas, pero la ligera penalización es consistente. 

¿Por qué dejar asentar la pelota no fue mejor que sacudirla ininterrumpidamente?

1. **Pérdida de Presupuesto en Sincronía (Greedy Waste)**: Cuando apagamos el ruido (`amp=0`) creyendo ser eficientes, las partículas aplican *Descenso de Gradiente Puro* en su cuenca local. Al ser un método de primer orden con gradiente analítico, las partículas tocan el fondo exacto de la cuenca en 2 o 3 pasos y luego se quedan ahí dando saltitos minúsculos inútiles. Todo el lapso de tiempo que desperdician hasta que la "Ventana de Estancamiento" se da cuenta de que están atascadas y enciende el sismo, es un **presupuesto computacional tirado a la basura**.
2. **Morfo-inercia (Fluidez Topológica)**: Cuando el suelo experimenta un ciclo constante y armónico (V14), el paisaje cambia de forma *mientras* la pelota aún está cayendo. Al no detenerse nunca el sismo, la partícula siempre tiene "inercia virtual" inyectada por los gradientes de la ladera cambiante. Forzar una pausa para que se estanque mata esa inercia matemática natural, requiriendo un pico de fuerza gigante para volver a desatascarla de un foso estricto en reposo absoluto.
3. **Falta de Micro-sacudidas:** Al hacer 1 ciclo de sismo y luego `0` puro, la pelota cruza cordilleras, pero cuando cae en el valle de destino final carece de "micro-ruido" vibratorio (aquella amplitud de bajísima escala en los nodos 0 del seno) para poder pulir y evadir piedras chiquititas (`roughness`).

## Conclusión

El mecanismo reactivo, intuitivo para heurísticas como Tabu Search o Restart SA, es subóptimo cuando simulamos campos físicos de RFF. **La exploración óptima asume que el universo estocástico siempre está vibrando**: es la propia sinusoide, al cruzar lenta y orgánicamente sus nodos neutros matemáticos (donde `sin(t) ≈ 0`), la que ya actúa como un "asentador natural perfecto" sin necesidad de apagar manualmente el motor de ruido (como tú mismo brillante observaste en la etapa anterior sobre por qué funcionaba sin decay!).
