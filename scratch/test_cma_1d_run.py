import cma
import numpy as np

def fn(x):
    return (x[0] - 2)**2

opts = cma.CMAOptions()
opts['bounds'] = [[-5.12], [5.12]]
opts['maxfevals'] = 100
opts['verbose'] = -9

try:
    es = cma.CMAEvolutionStrategy([0.0], 1.0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [float(fn(np.array(s))) for s in solutions])
    print("Result ok:", es.result.fbest)
except Exception as e:
    print("Run failed:", e)
    
try:
    print("xbest inside except block?", es.result.xbest)
except Exception as e:
    import traceback
    traceback.print_exc()
