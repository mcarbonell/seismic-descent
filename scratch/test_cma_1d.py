import cma
print("cma version:", cma.__version__)

try:
    es = cma.CMAEvolutionStrategy([0.0], 1.0)
    print("1D init ok")
except Exception as e:
    print("1D init failed:", e)

try:
    es = cma.CMAEvolutionStrategy([0.0, 0.0], 1.0)
    print("2D init ok")
except Exception as e:
    print("2D init failed:", e)
