https://www.sfu.ca/~ssurjano/optimization.html


Optimization Test Problems

The functions listed below are some of the common functions and datasets used for testing optimization algorithms. They are grouped according to similarities in their significant physical properties and shapes. Each page contains information about the corresponding function or dataset, as well as MATLAB and R implementations.

    Many Local Minima
    Ackley Function
    Bukin Function N. 6
    Cross-in-Tray Function
    Drop-Wave Function
    Eggholder Function
    Gramacy & Lee (2012) Function
    Griewank Function
    Holder Table Function
    Langermann Function
    Levy Function
    Levy Function N. 13
    Rastrigin Function
    Schaffer Function N. 2
    Schaffer Function N. 4
    Schwefel Function
    Shubert Function

    Bowl-Shaped
    Bohachevsky Functions
    Perm Function 0, d, β
    Rotated Hyper-Ellipsoid Function
    Sphere Function
    Sum of Different Powers Function
    Sum Squares Function
    Trid Function

    Plate-Shaped
    Booth Function
    Matyas Function
    McCormick Function
    Power Sum Function
    Zakharov Function

    Valley-Shaped
    Three-Hump Camel Function
    Six-Hump Camel Function
    Dixon-Price Function
    Rosenbrock Function
    
    Steep Ridges/Drops
    De Jong Function N. 5
    Easom Function
    Michalewicz Function

    Other
    Beale Function
    Branin Function
    Colville Function
    Forrester et al. (2008) Function
    Goldstein-Price Function
    Hartmann 3-D Function
    Hartmann 4-D Function
    Hartmann 6-D Function
    Perm Function d, β
    Powell Function
    Shekel Function
    Styblinski-Tang Function

---

Hola, existen competiciones de algoritmos de optimización caja negra?

---

# Competiciones de Optimización Caja Negra

Sí, existen varias competiciones importantes en este campo:

## Competiciones Principales

**1. BBOB (Black-Box Optimization Benchmarking)**
- Parte de las conferencias GECCO
- Benchmark estándar en la comunidad
- Funciones de prueba bien establecidas
- Permite comparación rigurosa de algoritmos

**2. CEC (Congress on Evolutionary Computation)**
- Competiciones anuales con diferentes tracks
- Problemas de optimización continua, discreta, multiobjetivo
- Gran participación de la comunidad científica

**3. COCO Platform (Comparing Continuous Optimizers)**
- Plataforma para evaluación automatizada
- Usa el benchmark BBOB
- Permite publicar resultados comparables

**4. Nevergrad (Meta)**
- Plataforma de benchmarking desarrollada por Meta
- Orientada a optimización de hiperparámetros
- Código abierto y activamente mantenida

## Áreas Específicas

- **AutoML**: competiciones como AutoML Challenge
- **Hyperparameter Optimization**: NeurIPS competitions
- **Real-world problems**: competiciones industriales ocasionales

¿Te interesa participar en alguna específica o estás buscando benchmarks para evaluar tu propio algoritmo?

---

**Sí, existen y son muy importantes dentro del campo de la informática y la inteligencia artificial.**

Las competiciones de algoritmos de optimización "caja negra" (Black-Box Optimization, BBO) son eventos fundamentales para que los investigadores comparen sus métodos de forma justa y estandarizada. Dado que en una caja negra no se conoce la función matemática subyacente (ni sus derivadas), estos concursos evalúan qué algoritmo encuentra la mejor solución con el menor número de evaluaciones de la función.

Aquí te detallo las competiciones y entornos más relevantes:

### 1. GECCO (Genetic and Evolutionary Computation Conference)
Es la conferencia más grande sobre computación evolutiva y alberga las competiciones más prestigiosas.
*   **Competición BBOB (Black-Box Optimization Benchmarking):** Es el estándar de oro. Los organizadores proponen un conjunto de funciones de prueba (benchmarks) secretas o estandarizadas (ruido, multimodalidad, condicionamiento, etc.). Los participantes envían sus algoritmos (código) y estos se ejecutan en un entorno controlado.
    *   Se mide la calidad de la solución y la velocidad de convergencia.
    *   Categorías típicas: Optimización de parámetros reales, optimización con ruido, optimización multiobjetivo.

### 2. IEEE CEC (Congress on Evolutionary Computation)
La CEC es otra conferencia gigante que organiza competencias anuales muy técnicas.
*   **Competiciones Anuales:** Cada año lanzan "Special Sessions" o competencias específicas. Por ejemplo, han habido competencias sobre:
    *   Optimización en entornos dinámicos (donde la función cambia con el tiempo).
    *   Optimización con restricciones.
    *   Optimización de gran escala (muchas variables).
    *   Optimización costosa (donde cada evaluación de la función cuesta mucho dinero o tiempo).

### 3. Plataforma COCO (Comparing Continuous Optimizers)
Aunque no es una competición "en vivo" con premios en metálico, es la infraestructura sobre la que se basan muchas competiciones (como la BBOB de GECCO).
*   Los investigadores pueden descargar el software de COCO, ejecutar sus algoritmos contra las funciones de prueba estándar y comparar sus resultados automáticamente con los del estado del arte (como CMA-ES, PSO, etc.). Es una forma de "competir" contra la literatura científica.

### 4. Competencias de Optimización Bayesiana y AutoML
En los últimos años, la optimización de hiperparámetros en Machine Learning ha popularizado un tipo específico de optimización caja negra.
*   **NeurIPS / ICML Competitions:** A menudo hay "challenges" de AutoML (Automated Machine Learning). Aquí la "caja negra" es el rendimiento de un modelo de IA. El objetivo es encontrar la mejor arquitectura o hiperparámetros en un tiempo limitado.
*   **HPOlib / HPOBench:** Son librerías que actúan como competiciones para ver qué algoritmo de optimización (como SMAC, Hyperopt o Optuna) configura mejor los modelos de ML.

### ¿Cómo funcionan estas competiciones?
A diferencia de un concurso de programación tradicional (como Codeforces), aquí el proceso suele ser:
1.  **Definición del problema:** Los organizadores definen una suite de problemas (funciones matemáticas complejas o simulaciones reales).
2.  **Caja Negra:** El participante no conoce la fórmula, solo puede enviar un vector de entrada $x$ y recibir un valor $f(x)$.
3.  **Presupuesto:** Se limita el número de evaluaciones (ej. máximo 10,000 llamadas a la función).
4.  **Ejecución:** Los organizadores ejecutan el código de todos los participantes en la misma máquina para garantizar igualdad de condiciones hardware.
5.  **Ranking:** Se comparan los resultados estadísticamente (mediana, media, desviación estándar) sobre múltiples ejecuciones.

### Resumen
Si estás interesado en participar o investigar en esto, te recomiendo buscar las **"GECCO BBOB Workshops"** y las **"IEEE CEC Competition Sessions"**. Son el lugar donde se definen los algoritmos más potentes del momento (como variantes de CMA-ES, Algoritmos Genéticos o Estrategias de Evolución Diferencial).