# DGE — Hallazgos v8: MNIST (Iteración 1)

**Fecha:** 2026-04-18  
**Estado:** En iteración activa — NO conclusivo todavía  
**Archivo:** `scratch/dge_prototype_v8_mnist.py`

---

## Setup

Red: **784→32→10** (ReLU + CrossEntropy), D=25,450 parámetros  
Dataset: subconjunto MNIST — 3,000 train / 600 test  
Presupuesto: 100,000 evaluaciones del minibatch-loss  
Minibatch: 128 muestras por evaluación

| Algoritmo | k | Evals/paso | Pasos totales | Runtime |
|-----------|---|------------|---------------|---------|
| DGE v7 | 15 | 30 | 3,333 | 42 s |
| SPSA | 1 | 2 | 50,000 | 31 s |

---

## Resultados v8 (iteración 1)

| Evals | DGE train | DGE test | SPSA train | SPSA test |
|-------|-----------|----------|------------|-----------|
| 10k | 10.5% | 11.5% | 8.2% | 8.3% |
| 30k | **17.4%** | **16.8%** | 5.3% | 6.7% |
| 60k | 15.3% | 17.3% | 7.1% | 8.0% |
| 100k | 10.4% | 10.3% | 13.8% | 14.3% |
| **Random** | **10%** | **10%** | **10%** | **10%** |

**DGE alcanza 18% y luego oscila. SPSA no supera el 14%.**  
Ninguno converge con 100k evals — ambos están cerca del baseline random al final.

---

## Diagnóstico: por qué no converge todavía

### Problema 1: Delta demasiado grande
Con `delta=1e-2` y grupos de ~1,697 variables, la perturbación de un bloque entero
desplaza la red fuera del régimen lineal. Los pesos iniciales (He init) tienen
magnitud ~0.05, y un delta de 0.01 sobre 1697 pesos a la vez crea un cambio
masivo en la salida. El gradiente estimado es inaccurato.

### Problema 2: lr demasiado grande relativo a los pesos
Con Adam clipeado a `clip_norm=0.5`, el paso efectivo en el espacio de pesos
es de orden 0.5 cuando los pesos típicos son ~0.05 — es decir, un paso que
es **10x la magnitud de los propios pesos**. Causa oscilación explosiva.

### Problema 3: Presupuesto insuficiente
DGE hace solo 3,333 pasos. Adam backprop típicamente necesita ~500-2000 pasos
en un subset de 3k de MNIST. DGE con señal ruidosa necesita más.

---

## Plan de iteración progresiva

### v8b — Fix de hiperparámetros: delta pequeño + clip agresivo
- `delta = 1e-3` (10x menor, gradiente más preciso)
- `clip_norm = 0.05` (pasos conservadores ~0.001 en peso)
- Batch size 512 (señal de gradiente más limpia)
- 1M evals → 33k pasos DGE

### v8c — Más datos + más budget
- N_TRAIN=10k, N_TEST=2k
- 5M evals si v8b muestra tendencia positiva

### v8d — Fine-tuning sobre pesos pre-entrenados
- Inicializar con pesos de una red entrenada con Adam (10 epochs)
- DGE para el último tramo de afinado
- Replica el caso de uso real de MeZO

### v8e — Red más pequeña
- 784→16→10 (D≈12.5k) — menos parámetros, convergencia más rápida
- Valida si el problema es la escala de D

---

## Conclusión preliminar (no definitiva)

> MNIST desde cero sin backprop es un problema difícil incluso para MeZO
> (que lo hace sobre pesos pre-entrenados). La señal de gradiente es ruidosa
> a esta escala. Sin embargo, **DGE ya supera a SPSA en el tramo inicial**
> (18% vs 6% en el mismo presupuesto) y la oscilación sugiere un problema
> de hiperparámetros, no de convergencia estructural.
> **La iteración continúa.**

---

*Documento generado tras v8 iteración 1 — 2026-04-18. Pendiente de actualización.*
