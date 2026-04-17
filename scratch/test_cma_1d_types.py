import cma
import numpy as np

opts = cma.CMAOptions()
opts['verbose'] = -9
x0 = np.array([2.5])
try:
    es = cma.CMAEvolutionStrategy(list(x0), 1.0, opts)
    print("Passed with list(np.array([2.5]))")
except Exception as e:
    print("Failed with list(np.array([2.5])):", e)
    
try:
    es = cma.CMAEvolutionStrategy(x0.tolist(), 1.0, opts)
    print("Passed with x0.tolist()")
except Exception as e:
    print("Failed with x0.tolist():", e)
