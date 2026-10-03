# Seismic Descent — reproducibilidad en un comando por objetivo.
# Requisitos: python + venv con `pip install -e ".[dev]" scipy` (scipy: estadística y L-BFGS-B).

PY ?= python

.PHONY: install test audit ablation components champion quick-benchmark clean-results

install:          ## Instala el paquete con dependencias de desarrollo
	$(PY) -m pip install -e ".[dev]" scipy

test:             ## Suite de tests (unitarios + golden regression)
	$(PY) -m pytest -q

ablation:         ## Ablación de la hipótesis central: correlacionado vs blanco vs nada (12 func × 4 dims, ~2 min)
	$(PY) -m benchmarks.experiment_ablation_noise --trials 15 --budget 3000

components:       ## Leave-one-out v23 + sensibilidad de amplitud / corrección sqrt(D) (~5 min)
	$(PY) -m benchmarks.experiment_component_ablation --trials 15 --budget 3000

champion:         ## Tabla champion del README (15 trials, 5D/10D/20D; ~10 min)
	$(PY) -m benchmarks.experiment_champion_v23 --dims 5 10 20 --trials 15

quick-benchmark:  ## Humo rápido del pipeline completo (1 trial, presupuesto 300)
	$(PY) -m benchmarks.experiment_ablation_noise --trials 1 --budget 300 --dims 2 --functions rastrigin
	$(PY) -m benchmarks.experiment_component_ablation --trials 1 --budget 300 --dims 5 --functions rastrigin

audit:            ## Re-ejecuta los scripts de verificación de la auditoría 2026
	$(PY) docs/audit_2026/v1_gradientes_y_ruido.py
	$(PY) docs/audit_2026/v2_ergodicidad.py

clean-results:    ## Borra los artefactos generados en results/ (git-ignored)
	rm -rf results/*.json results/*.png results/*.md

help:             ## Lista objetivos
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-16s %s\n", $$1, $$2}'

bootstrap: ## Recreate the external venv (survive sandbox resets) and install the package
	python3 -m venv /home/user/venv-seismic
	/home/user/venv-seismic/bin/pip install -q numpy scipy cma pytest matplotlib
	/home/user/venv-seismic/bin/pip install -q -e .
	@echo "[bootstrap] venv listo en /home/user/venv-seismic (usa /home/user/venv-seismic/bin/python)"
