# Apéndice A — Derivación de la ley de escala √D del ratio ruido/señal

**Estado:** ✅ derivada y verificada numéricamente (2026-10-03).
**Contexto:** la auditoría (`docs/audit_2026/AUDITORIA_VERIFICADA_2026.md` §3.4) midió que la
norma del gradiente de perturbación crece como √D (2.36 en D=5 → 5.6 en D=20, cociente ≈ √(20/5))
y pidió "1 página de cálculo" que la formalizara. Este documento es esa página, más la justificación
de la corrección v24 `amp_eff = amp·√(ref_D/D)` (`findings_v24_amplitud_raiz_d.md`) y la conexión
con el ajuste train/test de `ref_D` (`informe_refdim_train_test.md`).

---

## A.1 Marco: dinámica direccional (la señal tiene norma 1)

Seismic Descent opera en el dominio normalizado `[-1,1]^D`. En cada paso, el gradiente del objetivo
se proyecta a su **dirección** (`f_grad_dir = precond_grad / ‖precond_grad‖`,
`champion_v23.py` §5), de modo que el canal de señal entra en la integración con norma **exactamente 1**
en todo punto, dimensión y función. El terremoto, en cambio, añade el gradiente del campo de
perturbación **en crudo**, `∇n(x,t)`.

Por tanto el ratio ruido/señal instantáneo es simplemente

```
ρ(x,t) = ‖∇n(x,t)‖ / 1 = ‖∇n(x,t)‖.
```

que el balance exploración/explotación dependa solo de la dimensión (y no de la función) es lo que
permite una corrección **universal**: basta controlar cómo crece E‖∇n‖ con D.

## A.2 El campo sísmico RFF/ORF es un GP Gaussiano aproximado

El motor canónico (`noise_engine="orf"`, también `rff`) genera el campo como suma de r rasgos de
Fourier aleatorios (Bochner; Rahimi–Recht 2007), con fases uniformes y deriva temporal:

```
n(x,t) = amp · √(2/r) · Σ_{j=1..r} cos(ω_j·x + φ_j + δ_j t),
   ω_j = z_j / ℓ,   z_j ~ N(0, I_D)          (RFF)
   o filas z_j con E‖z_j‖² = D y direcciones Haar-ortogonales  (ORF)
```

Esta construcción aproxima el GP de núcleo Gaussiano (SE) `k(x,x') = amp²·exp(−‖x−x'‖²/(2ℓ²))`,
exacto cuando r→∞. En ORF las filas de z se ortogonalizan por bloques (QR con corrección de signo,
distribución Haar) con radios χ_D: **E‖z_j‖² = D en ambos motores** — exactamente lo único que usa
la derivación. La ortogonalización solo reduce la varianza muestral, no la media.

## A.3 Lema (potencia del gradiente del terremoto)

**Lema.** Para todo `x`, `t`, `r ≥ 1` y `D ≥ 1`:

```
E‖∇n(x,t)‖² = amp² · D / ℓ²        ⇒        E‖∇n‖ = amp · √D / ℓ  (en RMS)
```

**Demostración.** Derivando, `∇n = −amp·√(2/r)·Σ_j sin(θ_j)·ω_j` con `θ_j = ω_j·x + φ_j + δ_j t`.
Desarrollando el cuadrado:

```
E‖∇n‖² = amp²·(2/r)·Σ_{i,j} E[sin θ_i sin θ_j]·E[ω_i·ω_j].
```

Las fases `φ_j ~ U(0,2π)` iid eliminan todo término cruzado (E sin θ_j = 0), y los `ω_j`
independientes con E ω_j = 0 hacen lo propio por el otro factor. Solo sobrevive la diagonal:

```
E‖∇n‖² = amp²·(2/r)·Σ_j E[sin²θ_j]·E‖ω_j‖²
       = amp²·(2/r)·r·(1/2)·(D/ℓ²) = amp²·D/ℓ².      ∎
```

**Corolarios.**

1. **Independencia de r.** La normalización √(2/r) hace la potencia exacta para cualquier r; no es un
   artefacto de "pocas frecuencias". En el límite r→∞ es, además, el resultado exacto del GP:
   derivando el núcleo SE, `Var(∂_α n) = amp²/ℓ²` por coordenada.
2. **Distribución completa (límite GP).** `∇n(x,t) ~ N(0, (amp²/ℓ²)·I_D)`, luego `‖∇n‖ ~ (amp/ℓ)·χ_D`
   y la norma *típica* es `E‖∇n‖ = (amp/ℓ)·√2·Γ((D+1)/2)/Γ(D/2) ≈ (amp/ℓ)·√(D−1/2)` — el mismo
   escalado √D con la corrección χ usual.
3. **Octavas.** Con amplitud ×1/2 y escala ×2 por octava, la octava o aporta `amp²·16^{-o}·D/ℓ²`:
   serie geométrica acotada (prefactor ≤ 16/15). La ley √D sobrevive intacta.
4. **Puntual en el tiempo.** La envolvente del terremoto (decaimiento y fase senoidal) modula `amp`
   punto a punto; el lema vale para la amplitud instantánea: `E‖∇n(x,t)‖² = amp(t)²·D/ℓ²`.

## A.4 Consecuencia de diseño: ρ(D) = (amp/ℓ)·√D

Combinando A.1 y A.3, con los parámetros históricos `amp = 0.5`, `ℓ = 0.4`:

| D | 2 | 5 | 10 | 20 | 30 | 50 |
|---|---|---|----|----|----|----|
| ρ(D) = (amp/ℓ)·√D | 1.77 | **2.80** | 3.95 | **5.59** | 6.85 | 8.84 |

El crecimiento ×1.58 (5→10D) y ×2.0 (5→20D) es el medido en la auditoría (2.36 → 5.6; el valor
absoluto medido es algo menor porque promedia la envolvente temporal del terremoto). Geométricamente,
el terremoto desvía la trayectoria un ángulo típico `atan(ρ)`: en D=5 el balance quedó calibrado por
la historia del proyecto en ≈ 70°, pero en D=20 supera los 79° — el factor de movilidad pasa a
dominar 5.6× la señal, que es la degradación en alta dimensión documentada en la v20–v23.

## A.5 La corrección v24: `amp_eff(D) = amp·√(ref_D/D)`

Imponer que el ratio se mantenga en su valor de referencia calibrado en D = ref_D,

```
ρ(D) ≜ amp_eff(D)·√D/ℓ ≡ amp·√ref_D/ℓ  =  ρ(ref_D)     para todo D,
```

determina de forma única la ley de corrección

```
amp_eff(D) = amp · √(ref_D / D).      (Ley √D)
```

Observaciones:

- **Unicidad del exponente 1/2.** Como la señal entra con norma 1 (†A.1), cualquier otra potencia
  `amp·D^{-α}` deja un residuo `D^{1/2−α}` en el ratio. La corrección no es un ajuste empírico de
  forma funcional: es la única escala que hace el ratio adimensional.
- **Un solo grado de libertad queda: `ref_D`.** Es la amplitud efectiva (vía `amp·√ref_D`) que
  hereda la calibración histórica. ref_D = 5 reproduce bit a bit la v23 en 5D (golden tests); el
  split train/test (`informe_refdim_train_test.md`) mueve el óptimo a ref* ≈ 1–2 con ref1 como
  configuración recomendada (flagship v24r1: mejor sísmico en 30/36 a 31 trials).
- **Universalidad.** La ley depende solo de (D, ℓ, la normalización direccional), no de la función
  objetivo: por eso una misma constante funciona en las 12 familias del benchmark, con las
  excepciones de régimen documentadas (Levy/Schwefel → v25 adaptativa).

## A.6 Otros motores de perturbación

- **Lissajous / Orthogonal-Lissajous** ya incorporan una normalización de valor `1/√(c·D)`
  (resp. `1/√(M·D)` con M rotaciones). Promediando sobre fases, su potencia de gradiente resulta
  O(1) en D (las rotaciones ortogonales preservan la norma): la ley √D no se les aplica, y la
  corrección es conservadora innecesaria allí. La flag `noise_amp_dim_normalized` se recomienda con
  motores espectrales (RFF/ORF), que son los del campeón por defecto y el caso medido en la auditoría.
- **Ruido blanco** (ablación C3): por construcción se equiparó norma; la ley explica además por qué
  sin equiparación el blanco haría falta reescalar igualmente.

## A.7 Verificación numérica (reproducible)

Estimación de `E‖∇n‖²` con el código real (N = 4000 puntos × 3 tiempos, r = 256, ℓ = 0.4, amp = 1;
predicción D/ℓ²):

| D | predicho D/ℓ² | RFF medido | RFF/predicho | ORF medido | ORF/predicho |
|---|---|---|---|---|---|
| 2 | 12.50 | 12.40 | 0.992 | 14.83 | 1.186 |
| 5 | 31.25 | 29.44 | 0.942 | 32.96 | 1.055 |
| 10 | 62.50 | 61.46 | 0.983 | 64.20 | 1.027 |
| 20 | 125.00 | 122.74 | 0.982 | 127.04 | 1.016 |
| 50 | 312.50 | 309.99 | 0.992 | 307.90 | 0.985 |

Acuerdo a pocos puntos porcentuales (error de muestreo finito en r, N; ORF con algo más de varianza
en D baja, como predice la ortogonalización por bloques). Snippet:

```python
from seismic_descent.rff import RandomFourierFeatures   # o .orf import OrthogonalRandomFeatures
import numpy as np
f = RandomFourierFeatures(dim=D, r=256, base_lengthscale=0.4, seed=0)
X = np.random.default_rng(0).uniform(-1, 1, size=(4000, D))
E = np.mean([(np.linalg.norm(f.grad(X, t, 1.0), axis=1)**2).mean() for t in (0.0, 0.73, 1.9)])
# E ≈ D / 0.4**2
```

## A.8 Conexión con la evidencia empírica

La derivación fija la **forma** `amp ∝ D^{-1/2}`; la experimentación fija la **constante**:

- Parte B de `informe_ablacion_componentes.md` (31 trials): activar la ley mejora prácticamente todas
  las celdas ≥10D con p_Holm ≈ 9.3e-10 (mínimo Wilcoxon a n=31) — la predicción más fuerte del paper.
- `informe_refdim_train_test.md`: selección honesta de `ref_D` en train (7 funciones) → ref* = 0.5;
  en test ref1 queda nº1 (rank 2.61). La teoría dice "escala como 1/√D"; los datos dicen "centrada
  en ref ≈ 1–2" — y juntas producen la configuración flagship v24r1.
- Figura del paper: **F4** (`docs/paper_assets/F4_amplitude_law.png`).

**Citas sugeridas (al pasar a LaTeX):** Rahimi & Recht (2007) para RFF/Bochner; Yu et al. (2016)
para ORF; Mobahi & Fisher para el encuadre de deformación del paisaje (ver `RELATED_WORK.md`).
