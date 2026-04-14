# Hallazgos V17: Octavas Temporales (Fractal Time Tremors)

Esta iteración prueba la brillante hipótesis de aplicar una Superposición de Fourier a la amplitud base del terremoto. En lugar de una simple onda sinusoidal continua, componemos ondas armónicas menores, construyendo un "terremoto fractal" (un gran sismo sobre el cual cabalgan réplicas de mayor frecuencia y menor amplitud):

`amp = A * sin(t*f) + (A/2) * sin(t*2f) + (A/4) * sin(t*4f)`

## Resultados vs V14 (Onda Simple `10 ciclos`)

| Dimensión | Mediana v14 (Onda Pura simple)| Mediana v17 (Octavas Temporales) |
|-----------|-------------------------------|----------------------------------|
| **5D**    | 6.174                         | **5.394** (Nuevo récord Swarm)   |
| **10D**   | **32.709**                    | 33.721                           |
| **20D**   | **94.410**                    | 98.387                           |
| **50D**   | **323.640**                   | 326.929                          |

## Diagnóstico Matemático 

Al analizar la métrica, el modelo nos revela otro umbral súper interesante:

1. **Magia en Baja Dimensión (5D):** En 5D, el enjambre logró romper su propio récord bajando a un `5.39`. Aquí, el espacio no es hiper-denso y las partículas pueden aprovechar las "réplicas rápidas" (las octavas $2f$ y $4f$) para sacudirse piedrecillas finas fuera de los hoyos pequeños antes de que termine el macro-cliclo principal.
2. **El "Flicker" en Alta Dimensión:** ¿Por qué empeora en $\geq 10D$? Por la ley del Teorema de Muestreo (Nyquist) sobre nuestro "budget".
Como en las altas dimensiones tenemos apenas $\sim 500$ iteraciones temporales en total, para alcanzar los $10$ ciclos base, la onda principal $f$ toma unos $50$ pasos computacionales en completarse. 
Si metemos una onda a $4f$, ¡su ciclo dura sólo **12.5 pasos** computacionales! Un temblor cuyas montañas de $A/4$ suben y bajan en 6 pasos pasa de ser un "terremoto fluido para surfear" a convertirse en mero **ruido blanco temporal** (parpadeo o *Flicker* indescifrable iteración a iteración). Al igual que el ruido espacial blanco, el ruido temporal blanco daña la capacidad direccional de las partículas, saboteando levemente el deslizamiento natural en las grandes laderas de $50D$.

## Conclusión

La arquitectura de "réplicas temporales" es mecánicamente válida y triunfa cuando la discretización temporal lo permite (como en 5D, o si tuviéramos un budget de simulación $10$ veces mayor). Pero para simulaciones súper-apretadas (escasas en budget temporal de Python), componer altas frecuencias rompe la elegancia de nuestra suposición de "superficie lisa topológica" convirtiéndolo fatalmente en ruido estroboscópico. ¡Para resolver simulaciones bajo escasez evaluacional, la onda pura asintótica del sismo base (V14) reina de forma impoluta!
