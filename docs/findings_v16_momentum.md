# Hallazgos V16: Momentum Crudo (Heavy-Ball)

En esta prueba hemos reintroducido inercia en la bajada, pero esta vez evadiendo el error cometido en Adam (v11): en lugar de normalizar la varianza del gradiente, usamos un Momentum Clásico "Heavy-Ball" simple.
`V = mu * V + grad`
`X -= dt * V`
Para este control utilizamos el estándar $\mu = 0.9$ sobre la estructura ganadora de nuestro Enjambre de $10$ ciclos (v14).

## Resultados vs V14 (GD Puro)

| Dimensión | Mediana v14 (Descenso estándar) | Mediana v16 (Momentum $\mu=0.9$) |
|-----------|---------------------------------|----------------------------------|
| **5D**    | **6.174**                       | 38.895                           |
| **10D**   | **32.709**                      | 95.753                           |
| **20D**   | **94.410**                      | 231.935                          |
| **50D**   | **323.640**                     | 628.423                          |

## Diagnóstico Matemático del Comportamiento Adverso

El resultado fue, al igual que con Adam pero por motivos opuestos, un **desastre balístico**. Los peores números vistos en el test.

1. **Amplificación por Acumulación ($10\times$ magnitud):** Cuando aplicamos un arrastre $\mu=0.9$, la velocidad estacionaria de un gradiente constante escala con factor `1 / (1 - \mu) = 10`. Si el gradiente del ruido ya era masivo artificialmente para expulsar la partícula (`15.0`), el momentum lo integra y lo amplifica, convirtiendo el `dt=0.01` efectivo en un tamaño de paso brutal de `0.1`.
2. **Efecto Honda (Sloshing):** Rastrigin está poblado de hoyos profundos consecutivos. Un Momentum crudo alto hace que la partícula descienda un valle, almacene velocidad, y cuando el gradiente apunta levemente en contra para indicar la subida o el próximo valle, la inercia aplasta ese cambio y escupe a la partícula no uno, sino **múltiples hoyos más allá**. La pelota se pasa la vida rebotando y estrellándose literalmente contra las paredes del dominio topológico delimitado por `[-5.12, 5.12]` (siendo atajada sistemáticamente por `np.clip`).
3. **Filtro de Paso Bajo (Destrucción de Octavas):** Matemáticamente, la velocidad $V$ actúa como una integral temporal ($V \approx \int \nabla \mathrm{d}t$). La integral de una onda seno es un coseno (desfase de $90^\circ$), pero más crítico aún: **integrar atenúa fuertemente las frecuencias altas**. Las octavas refinadas (microvibraciones del ruido Perlin que lijan la superficie al final del proceso) son absorbidas por la inercia, cegando la capacidad del algoritmo para descender suavemente en el milímetro final.

## Conclusión

El éxito superlativo de *Seismic Descent* recae en que las partículas obedezcan al suelo **instantánea e ingenuamente**, sin recuerdos matemáticos del pasado. 
Si queremos que salten valles, no debemos darles velocidad a las partículas; para eso ya les construimos enormes olas continentales al suelo que las tiran de una cuenca a otra a la fuerza. ¡Cualquier memoria sobre la partícula (sea la normalización de Adam o la integral del Momentum) interfiere destructivamente con las dinámicas acústicas del campo RFF!
