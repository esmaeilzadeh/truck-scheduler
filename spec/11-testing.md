## 11. Testing (`tests/`)

- `test_validate.py`: hand-built feasible & infeasible solutions → validator accepts/rejects
  (overlap, release violation, horizon violation).
- `test_greedy.py`: tiny instances with known schedules; determinism; feasibility.
- `test_cpsat_small.py`: instances small enough to enumerate/verify optimum; check
  `proven_optimal` and objective equals brute-force optimum for `K ≤ 8`; fractional
  weights; stop_event.
- `test_alns.py`: ALNS ≤ greedy objective (never worse than warm start); reproducible with
  fixed seed; feasibility on many random instances.
- `test_alns_tabu.py` / `test_tabu.py` / `test_ga.py` / `test_ga_tabu.py`: never worse than
  greedy; seed reproducibility; feasibility; time-limit respect; small-instance obj ≥ CP-SAT.
- `test_dispatch.py`: dispatcher picks `alns_tabu` for large K under Auto; `force_tier` for
  greedy/cpsat/alns/alns_tabu/tabu/ga/ga_tabu; forced CP-SAT never returns `alns_fallback`.
- **Cross-check:** on small instances, metaheuristic objs ≥ `cpsat.obj` (optimum is a
  lower bound) and `greedy.obj ≥ cpsat.obj`.

---
