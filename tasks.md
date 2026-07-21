# Implementation Tasks — External Truck Gate Scheduling

Derived from [SPEC.md](SPEC.md). Tasks are ordered by dependency: foundations first,
then the three solver tiers, then offline studies, dispatcher, experiments, UI, and docs.

Legend: each task lists its **spec ref**, **produces** (files), and **depends on**.

---

## Phase 0 — Project scaffold

- [ ] **T0.1 Repo skeleton & dependencies** — Spec 3, 3.1
  - Create directory tree: `src/`, `src/solvers/`, `src/tuning/`, `src/experiments/`,
    `src/ui/`, `config/`, `data/instances/`, `data/results/`, `tests/`.
  - Write `requirements.txt`: `ortools`, `numpy`, `pandas`, `matplotlib` (and/or `plotly`),
    `streamlit`, `pytest`; optional `numba`, `irace`/`smac`.
  - Add `__init__.py` files as needed; empty `README.md` placeholder.
  - Produces: dir tree, `requirements.txt`.
  - Depends on: none.

---

## Phase 1 — Core model & shared utilities

- [ ] **T1.1 Data structures** — Spec 2.3
  - Implement `Operation` (frozen), `Instance`, `Solution` dataclasses in `src/model.py`.
  - Produces: `src/model.py`.
  - Depends on: T0.1.

- [ ] **T1.2 Objective function** — Spec 1.5, 2.4
  - `objective(instance, solution) = sum(op.w * starts[op.uid])`.
  - Produces: `src/model.py` (objective fn).
  - Depends on: T1.1.

- [ ] **T1.3 Feasibility precheck** — Spec 1.6
  - `feasibility_precheck(inst)`: verify `T > max(p_k)`, `sum(p_k) <= G*T`,
    and per-op `r_k + p_k - 1 <= T`. Raise `Infeasible` on violation.
  - Produces: `src/model.py` (precheck fn + `Infeasible` exception).
  - Depends on: T1.1.

- [ ] **T1.4 Validator** — Spec 1.4, 2.4
  - `validate(instance, solution) -> None`: raise on release, horizon, or gate-overlap
    violation (half-open intervals `[start, start+p)`).
  - Produces: `src/validate.py`.
  - Depends on: T1.1.

- [ ] **T1.5 I/O utilities** — Spec 2.1, 2.2
  - Read instance JSON -> `Instance` (build unified ops from `deliveries`/`pickups`,
    `rdt` default 1, weights `w1`/`w2`). Write `Solution` -> solution JSON with
    `solver`, `objective`, `proven_optimal`, `runtime_sec`, `assignments`, `meta`.
  - Produces: `src/io_utils.py`.
  - Depends on: T1.1.

- [ ] **T1.6 Solver interface** — Spec 3.1
  - `Solver` Protocol: `name`, `solve(inst, *, time_limit_sec, seed, warm_start) -> Solution`.
  - Produces: `src/solvers/base.py`.
  - Depends on: T1.1.

---

## Phase 2 — Tier 3: Greedy list scheduler (baseline + warm start)

- [ ] **T2.1 `earliest_start` helper** — Spec 4.2
  - Given sorted non-overlapping intervals on a gate, return smallest `t >= r` fitting
    `[t, t+p)` in a gap with `t <= T-p+1`, else `None`. Scan gaps (before/between/after).
  - Produces: `src/solvers/greedy.py` (helper) or a shared placement module.
  - Depends on: T1.1.

- [ ] **T2.2 Greedy ERD/SPT scheduler** — Spec 4.1, 4.3
  - Sort by `(r asc, p asc, w desc, uid)`; place each op at min-start gate (tie -> lowest
    gate). Expose `greedy_erd_spt` (default). Implement `Solver` interface.
  - Produces: `src/solvers/greedy.py`.
  - Depends on: T2.1, T1.6.

- [ ] **T2.3 Greedy variants** — Spec 4.3
  - `greedy_erd` (sort by `r, uid`) and `greedy_spt_ready` (event-driven: on gate free,
    pick shortest ready job) for ablation.
  - Produces: `src/solvers/greedy.py`.
  - Depends on: T2.2.

---

## Phase 3 — Tier 1: CP-SAT exact solver

- [ ] **T3.1 CP-SAT model** — Spec 4A.1, 4A.2
  - Per op: shared `s_k` int var over `[r_k, T-p_k+1]`; per gate optional interval
    `iv[k,g]` with presence `x[k,g]`. `AddExactlyOne` gate per op; `AddNoOverlap` per gate.
  - Produces: `src/solvers/cpsat.py`.
  - Depends on: T1.6.

- [ ] **T3.2 Objective, settings, warm start & status mapping** — Spec 4A.3, 4A.4
  - `Minimize(sum round(w_k*SCALE)*s_k)`; set `max_time_in_seconds`, `num_search_workers`;
    optional `AddHint` from greedy warm start; map status to `proven_optimal`
    (OPTIMAL=True, FEASIBLE=best+False, INFEASIBLE=raise).
  - Produces: `src/solvers/cpsat.py`.
  - Depends on: T3.1, T2.2 (warm start).

---

## Phase 4 — Tier 2: ALNS metaheuristic

- [ ] **T4.1 Permutation decoder** — Spec 5.1, 5.2
  - `decode(π) -> Solution` via greedy earliest-start over gates in order `π`; init
    `π0` from Tier-3 greedy order.
  - Produces: `src/solvers/alns.py`.
  - Depends on: T2.1, T2.2.

- [ ] **T4.2 Destroy operators** — Spec 5.3
  - Random removal, worst removal (bias exponent `d_wr`), Shaw/related removal
    (`rel` with `α=β=1, γ=T/4, δ=1`). `q` drawn from `[q_min, q_max]`.
  - Produces: `src/solvers/alns.py`.
  - Depends on: T4.1.

- [ ] **T4.3 Repair operators** — Spec 5.4
  - Greedy insertion and regret-k insertion (`k∈{2,3,4}`).
  - Produces: `src/solvers/alns.py`.
  - Depends on: T4.1.

- [ ] **T4.4 Adaptive operator selection** — Spec 5.5
  - Roulette by weights `ω`; per-segment scoring with rewards `σ1/σ2/σ3`; weight update
    `ω = (1-λ)ω + λ·(score/uses)` every `segment_length` iters.
  - Produces: `src/solvers/alns.py`.
  - Depends on: T4.2, T4.3.

- [ ] **T4.5 SA acceptance + stopping + reproducibility** — Spec 5.6, 5.7, 5.8
  - SA accept with cooling; `Temp0 = -(start_temp_ctrl·obj(x0))/ln(0.5)`; stop on
    `max_iterations` or `time_limit_sec`; single seeded RNG; report `iterations`, `gap_pct`.
  - Produces: `src/solvers/alns.py`.
  - Depends on: T4.4.

- [ ] **T4.6 ALNS Solver wrapper & hyperparameter loading** — Spec 5.9, 3.1
  - Wire into `Solver` interface; accept param dict (defaults from 5.9); load from
    `config/alns_params.json` when present.
  - Produces: `src/solvers/alns.py`.
  - Depends on: T4.5, T1.6.

---

## Phase 5 — Instance generator

- [ ] **T5.1 `gen_instance`** — Spec 6
  - `gen_instance(seed, M, N, G, T=None, p_range, rel_frac, w1, w2)`; auto-size `T` with
    slack ensuring precheck passes; regenerate `T` if needed.
  - Produces: `src/instance_gen.py`.
  - Depends on: T1.1, T1.3.

- [ ] **T5.2 Suite generation (TUNE / TEST / PROFILE)** — Spec 6
  - Generate seeded, disjoint suites across `K ∈ {5..400}`, `G ∈ {1,2,3,5}`, T slack;
    save to `data/instances/`.
  - Produces: instance JSON files + a manifest.
  - Depends on: T5.1, T1.5.

---

## Phase 6 — Offline studies

- [ ] **T6.1 ALNS hyperparameter tuning** — Spec 6H
  - Minimize mean relative gap over TUNE (ref = CP-SAT optimum or best-known proxy);
    `>=3` seeds/config, fixed per-run budget. Random search (fallback) or irace (preferred).
    Write winner to `config/alns_params.json`; log to `data/results/alns_tuning.csv`.
  - Produces: `src/tuning/tune_alns.py`, `config/alns_params.json`.
  - Depends on: T4.6, T3.2, T5.2.

- [ ] **T6.2 Tier-switch profiling** — Spec 7
  - On PROFILE suite measure CP-SAT time-to-optimal (cap `C`) and ALNS obj/runtime; pick
    `τ` = largest `K` with P95 CP-SAT time <= budget `B` and optimality proven; apply safety
    margin. Write `config/switch_policy.json`; emit crossover plot to `data/results/`.
  - Produces: `src/tuning/profile_switch.py`, `config/switch_policy.json`, crossover plot.
  - Depends on: T3.2, T4.6, T5.2.

---

## Phase 7 — Dispatcher

- [ ] **T7.1 Runtime dispatcher** — Spec 8, 13
  - Precheck -> greedy warm start (always) -> O(1) tier choice from `switch_policy.json`
    -> CP-SAT or ALNS; optional fall-back guard; `validate`; return `best_of(sol, warm)`.
    Provide safe fallbacks (threshold_K=20, T_cap=null, 5s limit) when configs missing.
  - Produces: `src/dispatch.py`.
  - Depends on: T7 deps: T2.2, T3.2, T4.6, T1.4; configs optional (T6.x).

---

## Phase 8 — Experiments & reporting

- [ ] **T8.1 Benchmark runner** — Spec 9
  - On TEST suite: per-instance & aggregated comparison table (`greedy/alns/cpsat` ×
    `objective/runtime/gap_pct`); crossover plot; ALNS convergence plot; optional ablations.
    Output CSV + PNG/HTML to `data/results/`.
  - Produces: `src/experiments/run_benchmark.py`.
  - Depends on: T2.2, T3.2, T4.6, T5.2.

---

## Phase 9 — UI

- [ ] **T9.1 Streamlit app with Gantt** — Spec 10
  - Inputs panel (`M,N,T,G,w1,w2`,seed, generate/upload/solve); color-coded Gantt
    (gates × time, delivery/pickup colorblind-safe colors, release markers); results table;
    which-tier-fired display; plots tab; reproducible seed.
  - Produces: `src/ui/app.py`.
  - Depends on: T7.1, T8.1.
  - Fallback: Jupyter notebook rendering the same Gantt + table + plots.

---

## Phase 10 — Testing

- [ ] **T10.1 Validator tests** — Spec 11
  - Feasible/infeasible hand-built cases (overlap, release, horizon).
  - Produces: `tests/test_validate.py`. Depends on: T1.4.

- [ ] **T10.2 Greedy tests** — Spec 11
  - Known-schedule tiny instances; determinism; feasibility.
  - Produces: `tests/test_greedy.py`. Depends on: T2.2.

- [ ] **T10.3 CP-SAT tests** — Spec 11
  - `K <= 8` brute-force optimum match; `proven_optimal` flag.
  - Produces: `tests/test_cpsat_small.py`. Depends on: T3.2.

- [ ] **T10.4 ALNS tests** — Spec 11
  - ALNS obj <= greedy; reproducible with fixed seed; feasibility on many random instances.
  - Produces: `tests/test_alns.py`. Depends on: T4.6.

- [ ] **T10.5 Dispatcher tests** — Spec 11
  - Correct tier per policy; feasible; never worse than greedy.
  - Produces: `tests/test_dispatch.py`. Depends on: T7.1.

- [ ] **T10.6 Cross-checks** — Spec 11
  - On small instances: `alns.obj >= cpsat.obj` and `greedy.obj >= cpsat.obj`.
  - Produces: added assertions in relevant tests. Depends on: T10.2-T10.4.

---

## Phase 11 — Documentation

- [ ] **T11.1 README** — Spec 12
  - How to generate instances, tune, profile, run UI, reproduce results; deliverables
    checklist.
  - Produces: `README.md`. Depends on: all prior phases.

---

## Suggested build order (critical path)

1. Phase 0 -> Phase 1 (model, validate, io, interface).
2. Phase 2 (greedy) — unblocks warm starts everywhere.
3. Phase 3 (CP-SAT) and Phase 4 (ALNS) — parallelizable after greedy.
4. Phase 5 (generator) — parallelizable with 3/4.
5. Phase 7 (dispatcher) with fallback configs -> end-to-end runnable early.
6. Phase 6 (offline studies) -> replace fallbacks with tuned configs.
7. Phase 8 (experiments), Phase 9 (UI), Phase 10 (tests throughout), Phase 11 (docs).
