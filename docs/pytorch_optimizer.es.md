# Seismic Optimizer para PyTorch

Este documento explica la implementación del algoritmo Seismic Descent como un optimizador estándar de PyTorch.

## Descripción General

Los optimizadores tradicionales como SGD o Adam utilizan la estocasticidad en cada paso (muestreo de mini-lotes o ruido de pesos). **SeismicOptimizer** introduce un campo de ruido dinámico y correlacionado espacialmente sobre todo el espacio de parámetros del modelo.

En cada paso, el optimizador suma el gradiente analítico de este campo de ruido al gradiente de la pérdida. El campo de ruido vibra con una frecuencia y un esquema de amplitud específicos ("temblores"), lo que ayuda a que los parámetros del modelo se "deslicen" fuera de los mínimos locales agudos hacia valles más amplios y estables.

## Fundamentación Matemática

El campo de ruido $\eta(w, t)$ se aproxima utilizando **Random Fourier Features (RFF)**:

$$ \eta(w, t) = \sqrt{\frac{2}{R}} \cdot A(t) \cdot \sum_{r=1}^R \cos(\omega_r \cdot w + t \cdot \text{drift}_r + \phi_r) $$

Donde:
- $w$ es el vector de parámetros aplanado de la red neuronal.
- $\omega_r$ representa vectores de frecuencia aleatorios.
- $A(t)$ es la amplitud actual del temblor.

La regla de actualización es:
$$ w_{t+1} = w_t - \gamma \cdot (\nabla_w L + \nabla_w \eta) $$

El gradiente $\nabla_w \eta$ se calcula analíticamente:
$$ \nabla_w \eta = -\sqrt{\frac{2}{R}} \cdot A(t) \cdot \sum_{r=1}^R \sin(\omega_r \cdot w + \text{offset}) \cdot \omega_r $$

## Parámetros de Configuración

- `lr`: Tasa de aprendizaje estándar.
- `noise_amplitude`: La escala inicial de los temblores del terremoto.
- `noise_decay`: Qué tan rápido disminuyen los temblores a lo largo del tiempo.
- `n_cycles`: El número de ciclos sísmicos completos (ondas senoidales) a realizar durante el entrenamiento.
- `n_octaves`: Número de escalas de ruido espacial (ruido fractal).

## Escalabilidad y Memoria

La implementación actual utiliza un **Campo de Características Global**. Concatena todos los parámetros del modelo en un solo vector y aplica una única proyección RFF.
- **Coste de Memoria**: $O(R \cdot M)$ donde $R$ es el número de características (por defecto 64) y $M$ es el número de parámetros del modelo.
- **Uso de VRAM**: Para modelos pequeños como MNIST, es despreciable. Para modelos de lenguaje masivos (LLMs), esto requeriría una optimización significativa de VRAM (por ejemplo, RFF por bloques o proyecciones dispersas).

## Implementación

El optimizador está definido en [seismic_optimizer.py](../seismic_optimizer.py). 

### Ejemplo de uso:
```python
from seismic_optimizer import SeismicOptimizer

optimizer = SeismicOptimizer(
    model.parameters(), 
    lr=0.01, 
    noise_amplitude=0.5, 
    n_cycles=10
)
```
