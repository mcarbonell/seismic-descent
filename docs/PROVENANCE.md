# Provenance & Authorship Disclosure

In the interest of transparency (which an increasing number of venues require), this document states how the Seismic Descent project was produced.

- **Concept and research direction (human):** The core idea — deforming the optimization landscape with a time-varying, spatially correlated noise field ("an earthquake under the particle's feet") so that plain gradient descent escapes local minima — was conceived by **Mario R. Carbonell**, who also directed the research agenda, chose the experimental questions, and decided which directions to pursue or abandon (see `docs/ideas.md`, `docs/chat_arena1.md`).
- **Implementation and experimentation (agent-assisted):** Most of the code, benchmarks, plots, and findings documents (v1–v23) were drafted and iterated by AI coding agents under human direction, across many short sessions. Raw conversation logs are preserved in `docs/chat_*.md` for traceability.
- **Independent verification (2026-10-03):** A full audit with empirical verification (`docs/audit_2026/`) was carried out, which (a) fixed critical packaging bugs, (b) retracted the "Laplacian ergodicity" claim after statistical refutation, and (c) installed the testing/CI/statistics infrastructure now present in the repository.

Anyone using this repository should feel confident citing it under the MIT license; authorship of the scientific contribution follows the venue's own authorship policies regarding AI-assisted work.
