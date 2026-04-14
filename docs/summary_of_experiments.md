# Seismic Descent: Evolución y Resumen de Experimentos (v1 - v17)

Este documento condensa menos de 24 horas de intensa experimentación, en las que el **Descenso Sísmico** evolucionó desde una idea conceptual, cruda y empírica hasta convertirse en un algoritmo analítico $\mathcal{O}(1)$ capaz de batir sistemáticamente a Simulated Annealing y de vencer definitivamente a CMA-ES en largas escalas presupuestarias para topologías de baja dimensión.

## Fase 1: El Nacimiento y Escalado Multidimensional (v1 - v6)

- **v1 a v3:** Nacimiento matemático usando "Perlin Noise" crudo en 2D. La premisa principal rompe la tradición del descenso estocástico: en lugar de arrojar ruido blanco ciego a la partícula, se deforma *el suelo* en base a patrones espacialmente correlacionados.
- **v4 (RFF):** Emigración a N-Dimensiones mediante *Random Fourier Features* (RFF). Esto simuló un *Gaussian Random Field* (GRF) permitiendo interferencias acústicas reales en multi-dimensionalidad y demostrando una topología infinitamente maleable.
- **v5 & v6:** Benchmarks sobre *Ackley* y *Schwefel*. Se demostró que Seismic Descent es magistral en paisajes ruidosos multimodales (Rastrigin), pero fracasa como cualquier optimizador de primer orden sobre amplias "mesetas planas" (Ackley), dado que el gradiente subyacente $\nabla \approx 0$ bloquea el deslizamiento de la partícula.

## Fase 2: Rompiendo Cuellos de Botella (v7 - v9)

- **v7 (Gradientes Analíticos O(1)):** Quizás el salto técnico más fundamental. Abandonamos la aproximación numérica de diferencias finitas en $D$ dimensiones. Implementamos las derivadas de senos/cosenos vectoriales, haciendo que simular 50 dimensiones tomara segundos en lugar de horas.
- **v8 & v9 (Polaridad Negativa - Sin `abs()`):** Descubrimiento de oro. Quitar el límite positivo del ruido generó "espejos topológicos". A medida que el seno oscilaba a negativo, las montañas insalvables se convertían activamente en embudos gravitacionales gigantes, expulsando a las partículas atrapadas hacia zonas fértiles.

## Fase 3: Hiper-Aceleración y Dinámica de Fluidos (v10 - v14)

- **v10 (Lengthscales dinámicas):** Detectamos que escalar las ondas de Perlin con base en la dimensión causaba asfixia. Esto probó matemáticamente cuánto confía el enjambre en encontrar gradientes de inclinación coherentes y no caos absoluto.
- **v11 (Fracaso de Adam):** Implementamos *Adam Optimizer*. Acabó siendo un desastre rotundo. El denominador RMS regularizaba la explosión colosal del ruido empobreciendo y neutralizando los terremotos salvadores. Conclusión vital: **Seismic necesita que la sumisión de la partícula al terreno sea pura y estúpida.**
- **v12 & v13 (Seismic Swarm):** Convertimos una simple bolita cayendo en todo un ecosistema de $N$ partículas gracias al cálculo vectorial gigante sobre el Campo RFF. Computacionalmente casi "gratis" ($\mathcal{O}(ND)$), esto otorgó mapeo geográfico brutal sin acudir a metadatos pesados (como matrices de Covarianza).
- **v14 (El Reloj Asintótico):** Desvinculamos el avance del ruido del tiempo iterativo base. Se estableció con exactitud estricta que todo presupuesto corra siempre, de inicio a fin, a través de  **$10$ Ciclos Sísmicos Puros**. Este se convirtió en el código base *Standard Gold*. 

## Fase 4: Experimentos Estocásticos Especializados (v15 - v17)

- **v15 (Bang-Bang Reactivo):** Condicionamos el sismo a dispararse **solamente** si el enjambre se atascaba. Sorprendentemente rindió peor. El continuo "deslizamiento" (*Morpho-Inercia*) es indispensable; frenar motores y esperar atascos mata la fluidez de la partícula. Las micro-ondas cercanas al vector $\sim 0$ del seno temporal sirven como un *settling natural* perfecto.
- **v16 (Momentum Heavy-Ball):** Fracaso por "Efecto Honda". El arrastre de inercia se tragó los valles consecutivos y filtró (como un *Low-Pass Filter*) el refinamiento fino de la señal acústica. Nuevamente: cero memoria en las partículas es imperativo.
- **v17 (Sismos Fractales - Fourier Temporal):** Implementamos un tiempo fractal ($f + 2f + 4f$). El resultado batió los registros estadísticos históricos en baja dimensión ($5D \rightarrow 5.39$). Pero en altos espacios ($50D$), el acotado "presupuesto" transformaba el rápido parpadeo de ondas ultra-frecuentes en destructivo *Ruido Blanco Temporal*.

## Mención Especial: Benchmarking a Larga Escala Multi-Presupuesto (Scale-Up)

Coronando la fase de experimentación cruzamos Seismic Descent contra CMA-ES con sobrealimentación de presupuesto computacional (Base x10.000). 
- **En $5D$:** Seismic alcanzó un $100\%$ de triunfo ciego `(30/30 cruces)`, doblegando y eclipsando por completo a la todopoderosa metaheurística de matriz de covarianza gracias a sus continuas recapacidades sísmicas anti-estancamiento (el infarto prematuro de CMA).
- **En $>10D$:** La ley física de las dimensiones astronómicas expone sin embargo el talón de aquiles topográfico: CMA-ES triunfa sobradamente porque infiere la dimensionalidad cruzada matemáticamente, la cual es imposible de mapear puramente con embudos gravitacionales en 50 dimensiones hiper-densas.

---
> *Creado espontáneamente y documentado desde cero en una épica *hackathon* colaborativa de $\sim 24$ horas de Pura Pasión y Descubrimiento Cibernético.*
