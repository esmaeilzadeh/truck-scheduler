## 6H. ALNS hyperparameter tuning procedure (offline, one-time)

**This is NOT machine-learning training** — there is no learned model, no labels, and no
online adaptation persisted. It is offline **algorithm configuration / parameter tuning**.
ALNS's own operator-weight adaptation (5.5) is ephemeral per run and is not "training".

### 6H.1 Objective of tuning
Find the hyperparameter vector `θ` (Section 5.9) minimizing the **mean relative gap** over
the TUNE suite:
```
gap(inst, θ, seed) = (obj_ALNS(inst, θ, seed) − ref(inst)) / ref(inst)
```
where `ref(inst)` = CP-SAT optimum if the instance is small enough to solve exactly,
else the best objective found across **all** ALNS configs/seeds on that instance
(best-known proxy). Score of `θ` = `mean over (inst, seed) of gap`, minimized.

### 6H.2 Method (choose one; both acceptable)
- **Preferred: automatic configurator** — `irace` (or `SMAC`/`ParamILS`). Provide the
  parameter space of Section 5.9, the TUNE instances, a per-run time/iteration budget,
  and let it search. Fix a total tuning budget (e.g. 2000 target runs).
- **Fallback: random search** — sample `≥200` configs from the ranges, evaluate each on a
  fixed TUNE subset with `≥3` seeds, keep the best mean gap. (Grid search only if the space
  is reduced to 2–3 params, e.g. `ρ_max`, `cooling`.)

### 6H.3 Protocol details
- **Stochasticity:** evaluate every config with `≥3` distinct seeds; average.
- **Fixed budget per run:** identical `time_limit_sec` (or `max_iterations`) for all configs
  so comparisons are fair.
- **Never tune on TEST.** Report final numbers on TEST with the single chosen `θ`.
- **Output:** write winning `θ` to `config/alns_params.json`; log the full search to
  `data/results/alns_tuning.csv` for the report.
- **Effort guidance:** for a course project, seeding with literature defaults (5.9) plus a
  light random search over `{ρ_max, cooling, seg, λ}` is sufficient; full irace is optional
  polish and good report material.

---
