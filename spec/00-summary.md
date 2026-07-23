## 0. Summary of the decision

The problem is parallel-machine scheduling with release dates and a weighted
sum-of-start-times objective: \(Pm \mid r_j \mid \sum w_j C_j\) (strongly NP-hard).
We implement three interchangeable solvers ("tiers") behind one uniform interface:

| Tier | Solver | Role | When used |
|------|--------|------|-----------|
| **Tier 1** | **OR-Tools CP-SAT** (exact) | Proven optimum | Small instances (`M+N ≤ τ`) |
| **Tier 2** | **ALNS+Tabu** (memetic metaheuristic) | Near-optimal, scalable | Large instances (`M+N > τ`) |
| **Tier 3** | **Greedy ERD/SPT** (list scheduler) | Baseline **and** warm start | Always (internally) |

Plain **ALNS** (Section 5) remains UI-/`force_tier`-selectable. Auto large-K uses
**ALNS+Tabu** (Section 4B.4).

The **dispatcher** selects Tier 1 vs Tier 2 using a threshold `τ` (and optionally `T`)
that is fixed **once, offline** by a profiling study (Section 7). At solve time the
selection is O(1) — no wasted timeout budget.

**Compare solvers (not part of Auto):** Tabu Search, Genetic Algorithm, and a memetic
**GA+Tabu hybrid** implement the same `Solver` interface and the same
permutation + greedy earliest-start decode as ALNS (Section 5.1). They are selectable
via the UI and `force_tier` for side-by-side comparison; **Auto never selects them**.

**Language:** Python 3.10+. **Exact solver:** `ortools` (CP-SAT). **Rationale:** solvers are
Python-first with C++ cores; ALNS is glue code; analysis/plotting ecosystem is best in Python.

---
