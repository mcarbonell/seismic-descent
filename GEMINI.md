# Reglas estables del Repositorio Seismic Descent

## Principio rector: trazabilidad y reproducibilidad experimental

**REGLA DE ORO:** Nunca se pierde un algoritmo o experimento que funcione. Toda iteración del algoritmo que se haya usado para producir resultados (un `findings_v*.md`, una tabla del README, una figura) debe quedar **congelada y reproducible**: su código no se reescribe de forma que cambie su comportamiento, porque eso destruye la trazabilidad de los experimentos anteriores.

Las iteraciones algorítmicas nuevas se crean en **archivos o módulos nuevos** (ver `legacy/` y la convención v1→v23). Es preferible duplicar código entre versiones antes que arriesgar un algoritmo prometedor reescribiéndolo en una versión posterior.

## Lo que SÍ se puede (y se debe) hacer

La regla anterior **no prohíbe** el mantenimiento del código. Está permitido y es deseable:

1. **Corregir bugs** que impiden que el código funcione (imports rotos, errores de empaquetado, crashes), siempre que la corrección no altere la dinámica numérica de una versión congelada que produjo resultados publicados.
2. **Reestructurar y mejorar** el código (refactors de legibilidad, tipado, docstrings, unificación de duplicados) **siempre que el comportamiento observable quede idéntico** y quede protegido por tests: los golden tests de `tests/` (resultados exactos con semilla fija) son la red de seguridad que certifica que un refactor no cambió la dinámica.
3. **Mejorar infraestructura**: empaquetado (`pyproject.toml`), CI, `.gitignore`, documentación, scripts de benchmark y de reproducibilidad. Los scripts de benchmark pueden ganar opciones nuevas (semillas, métricas, reportes) si sus parámetros por defecto documentados no cambian de significado.
4. **Añadir parámetros opt-in** al paquete, siempre que el valor por defecto preserve el comportamiento histórico de esa versión.

Cuando una corrección **sí** cambie el comportamiento numérico de una versión con resultados publicados: no se modifica esa versión; se crea la siguiente (v24, v25…) y se documenta la corrección en el `findings` correspondiente y en `CHANGELOG.md`.

## Resumen

> No es "no tocar jamás un `.py` existente": es **"no cambies el comportamiento de nada que haya producido un resultado que queramos poder reproducir"**. Codear sobre una base sana (tests, CI, refactors seguros) es parte del cuidado del experimento, no una violación de la regla.
