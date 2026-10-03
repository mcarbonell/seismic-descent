# Cobertura empírica 2D/5D y decorrelación (C1, figura F5)

> Rastrigin; seeds 1..8; 10 partículas; 50000 pasos (50k); rejillas 32x32 (2D) y 4^5 (5D) = 1024 celdas en ambos casos → curvas comparables.

| variante | D | cobertura@100% pasos | autocorrelación lag=100 (media seeds) |
|---|---|---|---|
| Seismic (RFF field) | 2 | 81.5% (80-83) | +0.922 |
| Seismic (RFF field) | 5 | 43.3% (37-50) | +0.927 |
| White (power-matched) | 2 | 99.7% (100-100) | +0.121 |
| White (power-matched) | 5 | 99.3% (99-99) | +0.176 |
| pure GD (amp = 0) | 2 | 15.8% (10-25) | +0.782 |
| pure GD (amp = 0) | 5 | 0.2% (0-0) | +0.963 |

**Lectura honesta (3 resultados):** (i) **perturbar ≫ no perturbar**: la cobertura del sísmico crece sin saturar durante los 50k pasos mientras el GD puro se congela (12% en 2D, ~0% en 5D — literalmente una celda por partícula); (ii) la promesa 1D de 'cobertura completa' NO se extiende literal: a 50k pasos el sísmico no llega al 100% (su curva sigue con pendiente positiva — coverage creciente, no completa a este horizonte); (iii) el blanco potencia-equiparada cubre MÁS RÁPIDO (satura ~80%/~70%) porque su perturbación es por paso i.i.d., mientras el campo sísmico es coherente en el tiempo (autocorrelación de posición +0.74/+0.81 a lag 100 vs +0.07/+0.11 del blanco): **la correlación espaciotemporal NO está para dispersar más rápido, sino para explorar con estructura** (arrays de acuerdo coherentes que el tracking y el detector KD aprovechan) — es el complemento funcional de la ablación de ruido (paridad en valor final). Claim C1 queda así refinado: cobertura empírica creciente y muy superior a GD, sin completitud a 50k, y coherencia temporal como rasgo distintivo real del terremoto correlacionado. ⚠️ Y sigue sin ser ergodicidad formal ni distribución Laplaciana (kurtosis −0.6…−0.85; theory.md retractado).
