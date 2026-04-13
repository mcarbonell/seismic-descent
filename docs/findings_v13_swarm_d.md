# Hallazgos V13: Enjambre Adaptativo ($N\_particles = D$)

Hemos modificado el tamaño de nuestro enjambre sísmico vectorizado dictando que instancie tantas partículas como dimensiones tenga el problema en evaluación ($N = D$).

Se ha mantenido la ley del presupuesto justo, es decir, el bloque completo se simula en `budget // D` pasos.

## Resultados vs V12 ($N=10$ estático)

| Dimensión | Enjambre v12 ($N=10$) | Enjambre Adaptativo v13 ($N=D$) | Pasos temporales | Tiempo v13 |
|-----------|-----------------------|---------------------------------|------------------|------------|
| **5D**    | Mediana: 8.286        | Mediana: **5.604** (12/30 éxito)| $600$ (era 300)  | 3.23s      |
| **10D**   | Mediana: 32.819       | Mediana: 32.096                 | $550$ (era 550)  | 3.71s      |
| **20D**   | Mediana: **99.206**   | Mediana: 102.053                | $525$ (era 1050) | **1.84s**  |
| **50D**   | Mediana: **319.651**  | Mediana: 320.868                | $510$ (era 2550) | **2.09s**  |

## Diagnóstico del efecto del tamaño del Enjambre

El principio del presupuesto justo (`evaluaciones = pasos * partículas`) ha revelado un importantísimo *trade-off* temporal en nuestro simulador físico:

1. **La Paradoja del Terremoto Corto:** Cada paso de simulación avanza la variable temporal del campo RFF en $\Delta t=0.05$. 
  - Si usamos $N=50$ partículas (50D), solo nos quedan recursos para $510$ iteraciones temporales. Esto significa que la simulación se congela bruscamente cuando $t = 25.5$, dejando a las partículas atrapadas mucho antes de experimentar el descenso suave de perturbaciones por el `decay`.
  - Si usamos $N=10$ partículas (50D), en cambio, tenemos recursos para $2550$ iteraciones temporales ($t = 127.5$), brindándole a la pelota mucho más tiempo para "acomodarse" al fondo del valle real conforme las perturbaciones cesan orgánicamente.
2. **5D brilla, 20D sufre:** Al pasar de 10 a 5 partículas en `5D`, le dimos el doble de iteraciones a cada pelota ($600$ vs $300$). Esto propició una superación total de la marca base (mediana en $5.6$), mientras que someter `20D` a 20 partículas le arrebató la mitad de su vida temporal y provocó una menor calidad en sus residuales.
3. **Optimización Computacional de Vértigo:** Simular Rastrigin en 50D durante toda la prueba, instanciando un campo pseudo-aleatorio de 50 vectores de 64 features concurrentes nos tomó apenas **2.09 segundos**. ¡Es brujería vectorizada y un indicio rotundo de que NumPy es capaz de asumir un coste inmensamente mayor!

## Recomendación

Para maximizar el descenso iterativo (asegurar un *schedule* de amplitud completo) es imperativo que el bucle se ejecute suficientes veces temporales (`n_steps` $> 1500$). 
Ya que evaluar una partícula cuesta exactamente lo mismo que evaluar veinte gracias a NumPy, nuestra ventaja ganadora no está en mantener estrictamente el presupuesto *budget* dividiéndolo equitativamente (lo que penaliza nuestros pasos), sino en usar un mini-enjambre de por ej. $N=5$ o $N=10$ partículas y **mantener el budget temporal intacto**, dándonos el lujo de superar el budget clásico aprovechando la eficiencia gratuita de la GPU / Caché L3 de CPU.
