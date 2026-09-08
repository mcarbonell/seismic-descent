import numpy as np

try:
    import cocoex
except ImportError:
    print("Por favor instala cocoex: pip install cocoex cocopp")
    exit(1)

from seismic_descent_vmorph import seismic_swarm

def fn_wrapper_factory(problem):
    """Convierte el evaluador 1D de COCO a un evaluador ND (N_particles, D)"""
    def fn(X):
        vals = []
        for x in X:
            if problem.final_target_hit:
                # Si ya llegamos, devolvemos un valor dummy para no consumir más evaluaciones
                vals.append(0.0)
            else:
                try:
                    vals.append(problem(x))
                except Exception:
                    vals.append(np.inf)
        return np.array(vals)
    return fn

def fn_grad_factory(problem):
    """Aproximación de gradiente por diferencias finitas.
    En problemas 'Caja Negra' no hay gradientes analíticos, así que los debemos aproximar.
    NOTA: Esto consume 'D' evaluaciones extra por cada partícula.
    """
    def fn_grad(X):
        eps = 1e-6
        N, D = X.shape
        grads = np.zeros((N, D))
        for i in range(N):
            x = X[i].copy()
            if problem.final_target_hit:
                return grads # Devolvemos 0s si ya terminamos
                
            try:
                y0 = problem(x)
                for d in range(D):
                    x_eps = x.copy()
                    x_eps[d] += eps
                    y1 = problem(x_eps)
                    grads[i, d] = (y1 - y0) / eps
            except Exception:
                pass # Si nos pasamos de evaluaciones, el gradiente queda a 0
        return grads
    return fn_grad

def run_experiment():
    suite_name = "bbob"
    # Tomamos solo las funciones 1 (Sphere) y 2 (Ellipsoid), dimensión 2, instancia 1.
    # Esto es ideal para una prueba de concepto súper rápida de < 2 segundos.
    suite = cocoex.Suite(suite_name, "year: 2024", "dimensions: 2 function_indices: 1,2 instance_indices: 1")
    
    # Configuramos el observador que escribirá los resultados al disco
    output_folder = "Seismic_Descent_COCO_Results"
    observer = cocoex.Observer(suite_name, f"result_folder: {output_folder}")
    
    # Parámetros del experimento
    budget_multiplier = 1000  # Multiplicador de presupuesto
    n_particles = 3
    
    print("=" * 60)
    print(f" Iniciando prueba de concepto de COCO con Seismic Descent v19")
    print("=" * 60)
    
    for problem in suite:
        # Vinculamos el problema al observador para que este grabe las llamadas a problem()
        problem.observe_with(observer)
        
        D = problem.dimension
        max_evaluations = budget_multiplier * D
        
        # Inicializamos en un punto aleatorio dentro de los límites
        lower_bounds = problem.lower_bounds
        upper_bounds = problem.upper_bounds
        search_range = max(abs(lower_bounds[0]), abs(upper_bounds[0]))
        
        x0 = lower_bounds + np.random.rand(D) * (upper_bounds - lower_bounds)
        
        fn = fn_wrapper_factory(problem)
        fn_grad = fn_grad_factory(problem)
        
        # En caja negra con diferencias finitas, cada step cuesta (1 + D) evals por partícula
        cost_per_step = n_particles + (n_particles * (D + 1)) 
        n_steps = max_evaluations // cost_per_step
        if n_steps < 1:
            n_steps = 1
            
        print(f"\n---> Resolviendo {problem.name}")
        print(f"     Presupuesto max: {max_evaluations} eval | Pasos estimados: {n_steps}")
        
        try:
            seismic_swarm(
                fn=fn,
                fn_grad=fn_grad,
                x0=x0,
                n_steps=n_steps,
                n_particles=n_particles,
                dt=0.01,
                noise_amplitude=5.0, # Amplitud de ruido general
                search_range=search_range,
                morph_steps=10
            )
        except Exception as e:
            # Capturamos cualquier error de límite de evaluaciones interno de COCO
            pass
            
        print(f"     [+] Eval ejecutadas: {problem.evaluations}")
        print(f"     [+] ¿Llegó a la meta perfecta?: {'Sí' if problem.final_target_hit else 'No'}")

    print("\n" + "=" * 60)
    print(" ¡Experimento completado!")
    print(f" Ahora puedes usar cocopp para analizar la carpeta '{output_folder}':")
    print(f"     python -m cocopp {output_folder}")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()
