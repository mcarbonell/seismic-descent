"""Verificación 4 (estática): escalabilidad del SeismicOptimizer de PyTorch.

SIN importar torch: replicamos la lógica de dimensionado de OMEGAS.
"""
p_configs = {
    "Perceptrón MNIST del paper (784->32->10)": 784 * 32 + 32 + 32 * 10 + 10,
    "MLP mediano (784->256->256->10)": 784 * 256 + 256 + 256 * 256 + 256 * 10 + 10,
    "ResNet-18 típica": 11_689_512,
    "LLM pequeño (124M params)": 124_000_000,
}
R, n_octaves, nbytes = 64, 4, 4  # float32
print("Memoria SOLO del campo RFF del optimizador (matriz OMEGAS + gradiente de ruido):")
for name, P in p_configs.items():
    omegas = R * P * n_octaves * nbytes
    noise_grad = P * nbytes
    total_gb = (omegas + noise_grad) / 1e9
    feasible = "OK" if total_gb < 1 else ("AJUSTADO" if total_gb < 8 else "INVIABLE")
    print(f"  {name:42s} params={P:>13,}  mem={total_gb:8.2f} GB  [{feasible}]")

print()
print("Análisis del schedule temporal (hardcode):")
n_cycles, trained_steps = 10, 20 * 469  # 20 épocas MNIST con batch 128 (~469 iters)
dt_noise = (n_cycles * 3.14159265) / 2000.0
cycles_real = trained_steps * dt_noise / 3.14159265
print(f"  dt_noise = (n_cycles * pi) / 2000  <- 2000 está HARDCODEADO")
print(f"  Con {trained_steps} pasos reales (20 épocas MNIST): ciclos reales = {cycles_real:.1f} (esperados: {n_cycles})")
print(f"  -> La frecuencia sísmica escala mal con el horizonte de entrenamiento real.")
