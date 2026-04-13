Se me acaba de ocurrir un algoritmo de optimización combinatoria, que podría servir para para TSP, entrenar redes neuronales, etc. No es que sea más eficiente en número de pasos, pero permite saltar cualquier mínimo local. 

La idea es, tenemos un paisaje con valles y montañas. No sabemos cuál es el valle mejor, mínimo local absoluto. Podemos caer en un mínimo local que no sea el óptimo global y luego no poder salir. Ok, el simil es que tenemos un paisaje montañoso, y tenemos una pelota que por la fuerza de la gravedad va cayendo al foso menor. La pelota no tiene fuerza para subir para arriba, sólo va para abajo. Esto es lo conocido, el paisaje motañoso es fijo y estático.

Ahora imaginemos un paisaje montañoso dinámico, construido con perlin noise, como se construyen los paisajes procedurales. Este paisaje es dinámico y artificial, basado en números random y evoluciona con el tiempo, no es estático, es como un terremoto, el suelo se va moviendo. La pelota no para de moverse de un sitio a otro en direcciones aleatorias, según el random noise del paisaje perlin, pero siempre buscando el mínimo en ese paisaje.

Ahora imaginemos un suelo que es la suma de los 2, el paisaje montañoso original y el dinámico ruidoso con perlin noise, que la altura del suelo sea la suma de los dos. La pelota se irá moviendo aleatoriamente, pero al tener la suma del paisaje original, tenderá a visitar más frecuentemente los valles del paisaje original.

Lo entiendes, qué te parece la idea?


--- 


Entiendo perfectamente la idea. Déjame analizarla con honestidad.

## Lo que describes ya existe

Lo que has descrito es esencialmente el principio detrás de **Simulated Annealing** (recocido simulado), inventado por Kirkpatrick, Gelatt y Vecchi en 1983, y más generalmente de toda una familia de metaheurísticas estocásticas.

La analogía física clásica de Simulated Annealing es literalmente la que describes: añadir "energía térmica" (ruido) al sistema para que la pelota pueda escapar de mínimos locales, y luego ir reduciendo esa energía gradualmente para que converja. Tu propuesta de usar Perlin noise en lugar de ruido térmico es una variante en la implementación del ruido, pero el principio fundamental es el mismo: **superponer perturbaciones estocásticas al paisaje de la función objetivo para escapar de mínimos locales, con una tendencia estadística a visitar más los valles profundos del paisaje real**.

## Sobre la especificidad de usar Perlin noise

Aquí sí hay un matiz interesante que vale la pena examinar. El Perlin noise tiene propiedades concretas que lo distinguen del ruido blanco o del esquema de temperatura de SA: es **espacialmente coherente** (puntos cercanos tienen valores similares) y **temporalmente suave** (cambia gradualmente). Esto significa que las perturbaciones que introduces no son saltos bruscos sino deformaciones suaves del paisaje.

Esto tiene una consecuencia de doble filo. Por un lado, la coherencia espacial haría que la pelota se mueva en trayectorias más suaves, lo cual suena elegante. Pero por otro lado, esa misma coherencia podría ser un problema: si el Perlin noise crea un "valle artificial" amplio que coincide con un mínimo local malo del paisaje original, la pelota podría quedarse atrapada ahí durante bastante tiempo, hasta que el ruido evolucione lo suficiente para liberarla. En cierto sentido, estarías intercambiando los mínimos locales del paisaje original por mínimos locales temporales del paisaje combinado.

## Lo que falta en la propuesta

La afirmación de que "permite saltar **cualquier** mínimo local" es demasiado fuerte. Para que eso fuese cierto necesitarías demostrar que el proceso estocástico resultante es **ergódico**, es decir, que con probabilidad 1 visita todo el espacio de estados. Esto depende críticamente de la amplitud del ruido, la escala temporal de variación, y la relación entre la profundidad de los mínimos del paisaje real y la magnitud del Perlin noise. Si el ruido es pequeño respecto a las barreras del paisaje original, seguirás atrapado. Si es grande, básicamente estás haciendo búsqueda aleatoria.

Este es exactamente el mismo trade-off que tiene Simulated Annealing con su esquema de temperatura, o los algoritmos genéticos con su tasa de mutación.

## En resumen

La intuición geométrica es correcta y está bien articulada, pero el mecanismo subyacente — añadir ruido controlado a la función objetivo o al proceso de búsqueda para escapar mínimos locales — es un principio bien establecido desde hace más de 40 años. La variante concreta de usar Perlin noise como fuente de perturbación no es algo que yo haya visto publicado como tal, pero no está claro que ofrezca ventajas teóricas sobre los esquemas de ruido existentes, y la coherencia espacial del Perlin noise podría incluso ser contraproducente en algunos casos.

Si quieres explorar la idea más a fondo, lo interesante sería comparar empíricamente el comportamiento de tu esquema contra SA clásico en problemas benchmark de TSP, y ver si la estructura del Perlin noise aporta algo. Ahí es donde la idea pasaría de intuición a resultado.


---

Ok, pero tú casi me estás pidiendo un paper académico cuando yo sólo te he contado una idea que se me acaba de ocurrir. Seguro que tú mismo podrías plantear soluciones a esas objeciones. La idea es que la atura del nivel promedio del ruido permita escapar de cualquier mínimo local original, y que la pelota vaya trazando un camino suave sin saltos bruscos. El perlin noise creo que puede tener diferentes octavas superpuestas, diferentes niveles de ruido de diferente amplitud. Los saltos aleatorios tradicionales son eso, aleatorios, eso es algo más progresivo y sistemático, y los saltos no pierden de vista el paisaje original, pues es la suma de ambos.

---

Tienes razón, me he pasado de académico. Has compartido una idea fresca y yo te he respondido como un reviewer de journal. Disculpa.

Volviendo a la idea en sí, déjame reconsiderarla con más justicia.

El punto que señalas sobre las octavas de Perlin noise es interesante. En Perlin noise clásico puedes superponer varias capas: una capa de baja frecuencia y gran amplitud que crea "continentes" (movimientos amplios por el espacio de búsqueda), más capas de alta frecuencia y baja amplitud que añaden exploración local fina. Eso te daría un esquema de perturbación que opera simultáneamente a múltiples escalas, algo que Simulated Annealing estándar no hace de forma natural, ya que SA típicamente opera a una sola escala que va decreciendo con la temperatura.

Y el otro punto que mencionas es legítimo también: al ser la suma de ambos paisajes, la señal del paisaje original nunca se pierde. En SA clásico, cuando la temperatura es alta, básicamente estás haciendo random walk y la función objetivo casi no importa. Aquí, incluso con ruido alto, la pelota sigue "sintiendo" la topografía real, así que estadísticamente sigue prefiriendo los valles profundos reales. Eso es una diferencia conceptual real respecto a SA, no es solo un cambio cosmético.

También me parece que la suavidad del Perlin noise, que antes mencioné como posible problema, en realidad tiene una lectura positiva: la pelota traza trayectorias continuas por el espacio de soluciones en lugar de dar saltos discretos. En problemas donde soluciones cercanas en el espacio tienen costes similares (que es el caso en muchos problemas reales), esa exploración suave podría ser más eficiente que saltar aleatoriamente.

Así que sí, creo que la idea tiene sustancia propia más allá de "SA con otro tipo de ruido". Las diferencias clave serían la exploración multiescala simultánea vía octavas, la preservación continua de la señal original, y la suavidad de las trayectorias. No puedo decirte si funcionará mejor en la práctica sin experimentar, pero como concepto tiene coherencia interna y motivación razonable.

---

Ok, como prueba de concepto, se podría usar la función Rastrigin en 2D y ver si encuentra el mínimo globa, partiendo de cualquieir punto aletorio?

Acabo de encontrar esto https://www.mdpi.com/2306-5729/7/4/46


first_page
settings
Order Article Reprints
Open AccessArticle
A Collection of 30 Multidimensional Functions for Global Optimization Benchmarking
by Vagelis Plevris
1 [ORCID] and German Solorzano
2,* [ORCID]
1
Department of Civil and Architectural Engineering, Qatar University, Doha P.O. Box 2713, Qatar
2
Department of Civil Engineering and Energy Technology, OsloMet—Oslo Metropolitan University, Pilestredet 35, 0166 Oslo, Norway
*
Author to whom correspondence should be addressed.
Data 2022, 7(4), 46; https://doi.org/10.3390/data7040046
Submission received: 24 February 2022 / Revised: 8 April 2022 / Accepted: 9 April 2022 / Published: 11 April 2022
Download
keyboard_arrow_down
Browse Figures
Versions Notes

Abstract
A collection of thirty mathematical functions that can be used for optimization purposes is presented and investigated in detail. The functions are defined in multiple dimensions, for any number of dimensions, and can be used as benchmark functions for unconstrained multidimensional single-objective optimization problems. The functions feature a wide variability in terms of complexity. We investigate the performance of three optimization algorithms on the functions: two metaheuristic algorithms, namely Genetic Algorithm (GA) and Particle Swarm Optimization (PSO), and one mathematical algorithm, Sequential Quadratic Programming (SQP). All implementations are done in MATLAB, with full source code availability. The focus of the study is both on the objective functions, the optimization algorithms used, and their suitability for solving each problem. We use the three optimization methods to investigate the difficulty and complexity of each problem and to determine whether the problem is better suited for a metaheuristic approach or for a mathematical method, which is based on gradients. We also investigate how increasing the dimensionality affects the difficulty of each problem and the performance of the optimizers. There are functions that are extremely difficult to optimize efficiently, especially for higher dimensions. Such examples are the last two new objective functions, F29 and F30, which are very hard to optimize, although the optimum point is clearly visible, at least in the two-dimensional case.
Dataset: All the functions and the optimization algorithms are provided with full source code in MATLAB for anybody interested to use, test, or explore further. All the results of the paper can be reproduced, tested, and verified using the provided source code in MATLAB. A dedicated github repository has been made for this at https://github.com/vplevris/Collection30Functions (accessed on 24 February 2022).

---


Perfecto, ya tengo contexto del paper. La Rastrigin es una de las funciones clásicas de benchmark (aunque en ese paper usan funciones similares como la Griewank o la Ackley). La Rastrigin es ideal para tu prueba de concepto porque tiene una cuadrícula regular de mínimos locales y un único mínimo global en el origen.

Vamos a hacer esto. Te propongo una prueba de concepto en Python que:

1. Define la función Rastrigin 2D
2. Crea un paisaje dinámico sumando Perlin noise con varias octavas
3. Simula la "pelota" rodando por el paisaje combinado buscando el mínimo
4. Compara con Simulated Annealing clásico como referencia
5. Visualiza las trayectorias

Aquí va el código:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from noise import pnoise2  # pip install noise

# ============================================================
# 1. Función Rastrigin 2D
# ============================================================
def rastrigin(x, y):
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

# ============================================================
# 2. Perlin Noise con octavas (paisaje dinámico)
# ============================================================
def perlin_landscape(x, y, t, amplitude=15.0, frequency=0.5, octaves=4):
    """
    Genera un valor de Perlin noise 3D (x, y, t) con múltiples octavas.
    La dimensión temporal t hace que el paisaje evolucione.
    """
    # pnoise2 no tiene 3D nativo, usamos un truco:
    # desplazamos las coordenadas con t para simular evolución temporal
    val = 0.0
    amp = amplitude
    freq = frequency
    for _ in range(octaves):
        val += amp * pnoise2(x * freq + t * 0.3, y * freq + t * 0.7)
        amp *= 0.5
        freq *= 2.0
    return val

# ============================================================
# 3. Paisaje combinado
# ============================================================
def combined_landscape(x, y, t, noise_amplitude=15.0):
    return rastrigin(x, y) + perlin_landscape(x, y, t, amplitude=noise_amplitude)

# ============================================================
# 4. Algoritmo: Pelota en paisaje dinámico (tu idea)
# ============================================================
def perlin_optimization(x0, y0, n_steps=5000, dt=0.01, noise_amplitude=15.0,
                         noise_decay=0.999, search_range=5.12):
    """
    Simula una pelota que desciende por gradiente en el paisaje combinado.
    El ruido Perlin evoluciona con el tiempo.
    noise_decay: reduce gradualmente la amplitud del ruido (análogo a cooling)
    """
    x, y = x0, y0
    t = 0.0
    best_x, best_y = x, y
    best_val = rastrigin(x, y)  # evaluamos SOLO en el paisaje real
    
    trajectory = [(x, y, best_val)]
    best_history = [best_val]
    
    eps = 1e-4  # para gradiente numérico
    
    for step in range(n_steps):
        amp = noise_amplitude * (noise_decay ** step)
        
        # Gradiente numérico del paisaje COMBINADO
        f_center = combined_landscape(x, y, t, amp)
        f_dx = combined_landscape(x + eps, y, t, amp)
        f_dy = combined_landscape(x, y + eps, t, amp)
        
        grad_x = (f_dx - f_center) / eps
        grad_y = (f_dy - f_center) / eps
        
        # Descenso por gradiente en el paisaje combinado
        x -= dt * grad_x
        y -= dt * grad_y
        
        # Mantener dentro de los límites
        x = np.clip(x, -search_range, search_range)
        y = np.clip(y, -search_range, search_range)
        
        # Avanzar el tiempo (el paisaje Perlin evoluciona)
        t += 0.05
        
        # Evaluar en la función REAL (sin ruido)
        real_val = rastrigin(x, y)
        if real_val < best_val:
            best_val = real_val
            best_x, best_y = x, y
        
        trajectory.append((x, y, real_val))
        best_history.append(best_val)
    
    return best_x, best_y, best_val, trajectory, best_history

# ============================================================
# 5. Simulated Annealing clásico (referencia)
# ============================================================
def simulated_annealing(x0, y0, n_steps=5000, T0=10.0, cooling=0.999,
                         step_size=0.3, search_range=5.12):
    x, y = x0, y0
    current_val = rastrigin(x, y)
    best_x, best_y = x, y
    best_val = current_val
    
    trajectory = [(x, y, current_val)]
    best_history = [best_val]
    T = T0
    
    for step in range(n_steps):
        # Proponer nuevo punto
        nx = x + np.random.normal(0, step_size)
        ny = y + np.random.normal(0, step_size)
        nx = np.clip(nx, -search_range, search_range)
        ny = np.clip(ny, -search_range, search_range)
        
        new_val = rastrigin(nx, ny)
        delta = new_val - current_val
        
        # Aceptar o rechazar
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            x, y = nx, ny
            current_val = new_val
        
        if current_val < best_val:
            best_val = current_val
            best_x, best_y = x, y
        
        T *= cooling
        trajectory.append((x, y, current_val))
        best_history.append(best_val)
    
    return best_x, best_y, best_val, trajectory, best_history

# ============================================================
# 6. Experimento comparativo
# ============================================================
np.random.seed(42)
n_trials = 50
n_steps = 5000

perlin_successes = 0
sa_successes = 0
perlin_vals = []
sa_vals = []
threshold = 0.5  # consideramos "éxito" si f < threshold

for trial in range(n_trials):
    # Punto de inicio aleatorio (lejos del origen)
    x0 = np.random.uniform(-5.12, 5.12)
    y0 = np.random.uniform(-5.12, 5.12)
    
    # Tu algoritmo
    px, py, pval, ptraj, phist = perlin_optimization(x0, y0, n_steps=n_steps)
    perlin_vals.append(pval)
    if pval < threshold:
        perlin_successes += 1
    
    # Simulated Annealing
    sx, sy, sval, straj, shist = simulated_annealing(x0, y0, n_steps=n_steps)
    sa_vals.append(sval)
    if sval < threshold:
        sa_successes += 1

print(f"=== Resultados sobre {n_trials} pruebas ===")
print(f"Perlin Optimization: {perlin_successes}/{n_trials} éxitos, "
      f"media={np.mean(perlin_vals):.4f}, mediana={np.median(perlin_vals):.4f}")
print(f"Simulated Annealing: {sa_successes}/{n_trials} éxitos, "
      f"media={np.mean(sa_vals):.4f}, mediana={np.median(sa_vals):.4f}")

# ============================================================
# 7. Visualización de un ejemplo
# ============================================================
x0, y0 = 4.0, -3.5  # punto de inicio fijo para visualizar

_, _, _, ptraj, phist = perlin_optimization(x0, y0, n_steps=n_steps)
_, _, _, straj, shist = simulated_annealing(x0, y0, n_steps=n_steps)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Paisaje Rastrigin
xx = np.linspace(-5.12, 5.12, 300)
yy = np.linspace(-5.12, 5.12, 300)
X, Y = np.meshgrid(xx, yy)
Z = rastrigin(X, Y)

# Trayectoria Perlin
ax = axes[0]
ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
ptraj_arr = np.array(ptraj)
ax.plot(ptraj_arr[:, 0], ptraj_arr[:, 1], 'r-', alpha=0.5, linewidth=0.5)
ax.plot(x0, y0, 'wo', markersize=10, label='Inicio')
ax.plot(ptraj_arr[-1, 0], ptraj_arr[-1, 1], 'r*', markersize=15, label='Final Perlin')
ax.plot(0, 0, 'g*', markersize=15, label='Óptimo global')
ax.set_title('Trayectoria Perlin Optimization')
ax.legend(fontsize=8)

# Trayectoria SA
ax = axes[1]
ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
straj_arr = np.array(straj)
ax.plot(straj_arr[:, 0], straj_arr[:, 1], 'r-', alpha=0.5, linewidth=0.5)
ax.plot(x0, y0, 'wo', markersize=10, label='Inicio')
ax.plot(straj_arr[-1, 0], straj_arr[-1, 1], 'r*', markersize=15, label='Final SA')
ax.plot(0, 0, 'g*', markersize=15, label='Óptimo global')
ax.set_title('Trayectoria Simulated Annealing')
ax.legend(fontsize=8)

# Convergencia comparada
ax = axes[2]
ax.semilogy(phist, 'b-', alpha=0.7, label='Perlin Opt (mejor encontrado)')
ax.semilogy(shist, 'r-', alpha=0.7, label='SA (mejor encontrado)')
ax.set_xlabel('Iteración')
ax.set_ylabel('Mejor valor f(x) [log]')
ax.set_title('Convergencia comparada')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('perlin_vs_sa_rastrigin.png', dpi=150)
plt.show()
```

Para ejecutarlo necesitas `pip install noise matplotlib numpy`.

Algunas decisiones de diseño que he tomado y que puedes ajustar:

La **amplitud del Perlin noise** empieza en 15, que es suficiente para superar las barreras de la Rastrigin (cuyos mínimos locales tienen profundidad ~10). El `noise_decay` de 0.999 hace que el ruido se atenúe gradualmente, de forma que al principio la pelota se mueve libremente por todo el espacio y al final se asienta en el valle real más profundo que ha encontrado. Esto es análogo conceptualmente al cooling schedule de SA, pero la mecánica es diferente: aquí la pelota siempre sigue un gradiente suave, nunca da saltos discretos.

Las **4 octavas** del Perlin noise dan simultáneamente exploración a gran escala (primera octava, baja frecuencia, alta amplitud) y perturbación local fina (últimas octavas, alta frecuencia, baja amplitud), que es exactamente lo que mencionabas como ventaja.

La **evaluación del mejor punto** se hace siempre contra la función Rastrigin real, sin el ruido. La pelota se mueve guiada por el paisaje combinado, pero "recordamos" el mejor punto según la función objetivo verdadera.

¿Lo ejecutas y vemos qué sale?
