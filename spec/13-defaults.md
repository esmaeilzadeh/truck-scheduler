## 13. Defaults if you skip tuning/profiling (safe fallbacks)

- ALNS params: literature defaults in Section 5.9 (or tuned file if present).
- ALNS+Tabu params: Section 4B.4 defaults (or tuned `config/alns_tabu_params.json`).
- Switch threshold: `threshold_K = 25`, `T_cap = null`, `budget_sec` for ALNS/ALNS+Tabu,
  **`cpsat_time_limit_sec = 10`** for forced/exact CP-SAT, with Auto ALNS fall-back when
  CP-SAT does not prove optimality (small-K path). Auto large-K uses ALNS+Tabu.
- Tabu / GA / GA+Tabu: defaults in Section 4B.1–4B.3 config files (hand-chosen starting
  points; offline-tuned when a tune run has been performed).
- Weights: `w1 = w2 = 1`.

These make the system runnable end-to-end before the offline studies are done; the studies
then replace the fallbacks with tuned values and give you report material.
