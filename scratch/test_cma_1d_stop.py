import cma
import numpy as np

opts = cma.CMAOptions()
opts['verbose'] = -9
opts['bounds'] = [[-5.12], [5.12]]
opts['maxfevals'] = 1000

es = cma.CMAEvolutionStrategy([0.0], 1.0, opts)
print("es.stop() immediately?", es.stop())
