## 12. Deliverables checklist

- [x] `src/` with all modules and the uniform `Solver` interface.
- [x] Feasibility precheck + shared validator.
- [x] Tier 3 greedy (ERD/SPT) + variants.
- [x] Tier 1 CP-SAT (optional-interval model, weight scaling, warm start, proven-optimal,
      stop/timeout).
- [x] Tier 2 ALNS (destroy/repair operators, adaptive weights, SA acceptance, seeds).
- [x] Tier 2 Auto path: ALNS+Tabu hybrid (Section 4B.4) + `tune_alns_tabu.py`.
- [x] Compare solvers: Tabu Search, Genetic Algorithm, GA+Tabu hybrid (Section 4B.1–3).
- [x] `config/alns_params.json` from tuning (Section 6H) + tuning log.
- [x] `config/alns_tabu_params.json` from light ALNS+Tabu tune + tuning log.
- [x] `config/switch_policy.json` from profiling (Section 7) + `cpsat_time_limit_sec`.
- [x] `config/tabu_params.json`, `ga_params.json`, `ga_tabu_params.json`.
- [x] Dispatcher with `force_tier` including `alns_tabu`; Auto large-K → `alns_tabu` (Section 8).
- [x] Benchmark + plots (Section 9).
- [x] Streamlit UI with Gantt, algorithm select, compare-all, Stop, downloads (Section 10).
- [x] Tests (Section 11), all passing.
- [x] README: how to generate instances, tune, profile, run UI, reproduce results.

---
