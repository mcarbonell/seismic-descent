# Earthquake Optimization (Perlin Descent)

Algoritmo de optimización basado en descenso de gradiente sobre un paisaje dinámico perturbado con ruido Perlin.

## La idea

En lugar de dar saltos aleatorios para escapar de mínimos locales (como Simulated Annealing), se le da un **terremoto al suelo**. La pelota sigue haciendo lo único que sabe: rodar cuesta abajo. Pero como el suelo tiembla de forma coherente y multiescala, los mínimos locales se convierten temporalmente en cuestas y la pelota escapa sola.

```
f_total(x, t) = f_original(x) + A(t) * Perlin(x, t)
```

- `f_original` — función objetivo real
- `Perlin(x, t)` — ruido Perlin con 4 octavas, evoluciona en el tiempo
- `A(t) = A0 * |sin(t * freq(t))|` — amplitud cíclica con frecuencia decreciente
- El mejor punto se registra siempre contra `f_original`, no contra el paisaje combinado

## Ventajas sobre SA

- Ruido **correlacionado espacialmente** (Perlin vs gaussiano blanco): la pelota se desliza hacia otro valle, no se teletransporta
- **Multiescala** por las octavas: exploración gruesa + refinamiento fino en un solo mecanismo
- Trayectorias suaves y continuas
- Favorece naturalmente los **valles anchos** (flat minima), propiedad deseable en redes neuronales

## Resultados (Rastrigin 2D, 50 pruebas, 5000 pasos)

| Algoritmo | Éxitos (f<0.5) | Media | Mediana | Evaluaciones |
|---|---|---|---|---|
| **Perlin Descent** | **47/50** | **0.139** | **0.100** | 5.000 |
| Simulated Annealing | 34/50 | 0.410 | 0.117 | 5.000 |
| CMA-ES | 11/50 | 1.498 | 0.995 | 50.000 |

## Instalación

```bash
pip install numpy noise matplotlib cma
```

## Uso

```bash
python perlin_opt.py
```

## Estructura

```
docs/
  findings_v1.md     — hallazgos sesión inicial
  chat_arena.md      — conversación original con la idea
  chat_opus4.6.md    — prototipo generado por Claude
perlin_opt.py        — implementación y benchmark
```

## Próximos experimentos

- Escalar a N dimensiones (Simplex Noise) y comparar con CMA-ES en su terreno
- Probar en Ackley, Schwefel, Griewank
- Aplicación a TSP con offsets en distancias
