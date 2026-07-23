## 7. Tier-switch determination (offline profiling, one-time)

Goal: fix threshold(s) so that **at test time the tier choice is O(1)** with no wasted
solve budget. We do **not** run CP-SAT-until-timeout in production.

### 7.1 What we measure (on PROFILE suite)
For each instance (grouped by `K = M+N`, and secondarily by `T`):
- CP-SAT: `time_to_proven_optimal` (with a generous cap `C`, e.g. 300 s) and whether it
  proved optimality within `C`.
- ALNS: objective and runtime under the production time budget `B`.

### 7.2 Choosing the threshold `τ`
- Define the **acceptable test-time budget** `B` for exact solving (e.g. `B = 5 s`).
- For each size `K`, compute the **95th percentile** of CP-SAT solve time across seeds
  (use a high percentile, not the mean, because same-size instances vary in hardness).
- `τ = largest K such that P95(CP-SAT solve time at K) ≤ B` **and** CP-SAT proved
  optimality for (essentially) all instances at that K.
- Apply a **safety margin**: set the deployed threshold to `τ_deploy = τ − Δ` (e.g. one grid
  step below the crossover) to reduce risk of a hard small instance blowing the budget.

### 7.3 Optional second feature `T`
If profiling shows solve time depends strongly on `T` as well as `K`, store a small rule
table, e.g. `use_exact = (K ≤ τ_deploy) and (T ≤ T_cap)`, choosing `T_cap` the same
percentile way. Otherwise keep the single-`K` rule (simplest).

### 7.4 Output
Write to `config/switch_policy.json`:
```json
{
  "budget_sec": 10.0,
  "cpsat_time_limit_sec": 10.0,
  "threshold_K": 25,
  "T_cap": null,
  "safety_margin_steps": 1,
  "notes": "Manual Auto policy: CP-SAT when K=M+N<=25, ALNS+Tabu when K>25; cpsat_time_limit_sec=10s"
}
```
- `budget_sec` — Auto ALNS+Tabu / shared UI default for metaheuristics.
- `cpsat_time_limit_sec` — default wall budget for forced / exact CP-SAT (UI default when
  Algorithm = CP-SAT).
Also emit the **crossover plot** (CP-SAT vs ALNS runtime & quality vs `K`) to
`data/results/` for the report.

---
