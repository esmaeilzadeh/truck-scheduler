# Implementation Specification — External Truck Gate Scheduling

**Status:** Implementable spec (v1) — updated to match the implemented codebase
**Source problem:** `project-5-trucks-scheduling.md` (the corrected/polished formulation)
**Goal of this document:** Give a complete, unambiguous, build-ready specification for a
**3-tier solver** for the gate-scheduling problem, including the exact algorithms,
the offline hyperparameter-tuning procedure, the offline tier-switching procedure,
and **compare metaheuristics** (Tabu, GA, GA+Tabu hybrid) used for reporting only.

---

## 0. Summary of the decision

The problem is parallel-machine scheduling with release dates and a weighted
sum-of-start-times objective: \(Pm \mid r_j \mid \sum w_j C_j\) (strongly NP-hard).
We implement three interchangeable solvers ("tiers") behind one uniform interface:

| Tier | Solver | Role | When used |
|------|--------|------|-----------|
| **Tier 1** | **OR-Tools CP-SAT** (exact) | Proven optimum | Small instances (`M+N ≤ τ`) |
| **Tier 2** | **ALNS** (metaheuristic) | Near-optimal, scalable | Large instances (`M+N > τ`) |
| **Tier 3** | **Greedy ERD/SPT** (list scheduler) | Baseline **and** warm start for ALNS | Always (internally) |

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

## 1. Problem model (authoritative definitions)

### 1.1 Entities
- `M` outbound deliveries (SSCGD), indexed `i = 1..M`.
- `N` inbound pickups (SSCPI), indexed `j = 1..N`.
- Unified set of **operations** `K = M + N`, indexed `k = 0..K-1`.
  Each operation has: `type ∈ {DELIVERY, PICKUP}`, processing time `p_k = TPG_k`,
  release time `r_k`, weight `w_k`.
  - Delivery: `r_k = RDT_i` (default `1`), `w_k = w1`.
  - Pickup:   `r_k = TSSCPI_j`, `w_k = w2`.

### 1.2 Parameters
- `T` — number of time periods; constraint `T > max(p_k)`.
- `G` — number of gates, `G ≥ 1`.
- `w1, w2 ≥ 0` — objective weights (default `w1 = w2 = 1`).

### 1.3 Decision (solution) representation (canonical, solver-independent)
A solution assigns to every operation `k`:
- `start_k ∈ {1, …, T}` — start period.
- `gate_k ∈ {1, …, G}` — assigned gate.

Occupancy interval is **half-open** `[start_k, start_k + p_k)`; last occupied period is
`start_k + p_k − 1`.

### 1.4 Constraints
1. **Release:** `start_k ≥ r_k` for all `k`.
2. **Horizon:** `start_k + p_k − 1 ≤ T` ⇔ `start_k ≤ T − p_k + 1`.
3. **Gate non-overlap:** for any two ops `a ≠ b` with `gate_a = gate_b`:
   `start_a + p_a ≤ start_b` OR `start_b + p_b ≤ start_a`.

### 1.5 Objective
Minimize
\[
\text{Cost} = w_1 \sum_{i} DT_i + w_2 \sum_{j} PT_j = \sum_{k} w_k \cdot start_k .
\]

### 1.6 Feasibility precheck (run before any solver)
- `T > max(p_k)`.
- Workload fits: `sum(p_k) ≤ G · T`. If violated → declare **infeasible**, do not solve.
- Per-op fits: `r_k + p_k − 1 ≤ T` for every `k`. If any op cannot fit after its
  release, declare infeasible.

---

## 2. Data formats & I/O contract

### 2.1 Instance JSON (input)
```json
{
  "id": "inst_0001",
  "T": 48,
  "G": 3,
  "w1": 1.0,
  "w2": 1.0,
  "deliveries": [ {"id": 0, "p": 3, "rdt": 1} ],
  "pickups":    [ {"id": 0, "p": 2, "release": 5} ]
}
```
- `rdt` optional (default 1). `release` = `TSSCPI_j` (required for pickups).

### 2.2 Solution JSON (output)
```json
{
  "instance_id": "inst_0001",
  "solver": "cpsat|alns|greedy|tabu|ga|ga_tabu",
  "objective": 123.0,
  "is_optimal": true,
  "proven_optimal": true,
  "runtime_sec": 0.42,
  "assignments": [
    {"op_type": "delivery", "op_id": 0, "start": 1, "gate": 1, "p": 3, "weight": 1.0}
  ],
  "meta": { "iterations": 0, "gap_pct": 0.0 }
}
```

### 2.3 Internal `Operation` / `Instance` / `Solution` structures
```python
@dataclass(frozen=True)
class Operation:
    uid: int            # 0..K-1 global id
    kind: str           # "delivery" | "pickup"
    local_id: int       # i or j
    p: int              # processing time
    r: int              # release time
    w: float            # weight

@dataclass
class Instance:
    id: str
    T: int
    G: int
    ops: list[Operation]        # length K
    w1: float; w2: float

@dataclass
class Solution:
    starts: dict[int, int]      # uid -> start
    gates:  dict[int, int]      # uid -> gate
    # objective computed on demand; must be feasible before scoring
```

### 2.4 Validator (shared, mandatory)
Implement `validate(instance, solution) -> None` that raises on any violation of
Section 1.4 constraints and horizon/release bounds. **Every** solver's output must
pass the validator before its objective is trusted. Objective:
`sum(op.w * solution.starts[op.uid] for op in ops)`.

---

## 3. Module / repo architecture

```
truck_scheduling/
  SPEC.md                      # this file
  project-5-trucks-scheduling.md
  README.md
  requirements.txt             # ortools, numpy, pandas, matplotlib/plotly, streamlit, (optional) numba, irace-py or smac
  src/
    model.py                   # dataclasses (Section 2.3), objective, feasibility precheck (1.6)
    io_utils.py                # instance/solution JSON read/write
    validate.py                # validator (2.4)
    instance_gen.py            # random instance generator (Section 6)
    solvers/
      base.py                  # Solver interface (3.1)
      greedy.py                # Tier 3 (Section 4)
      alns.py                  # Tier 2 (Section 5)
      cpsat.py                 # Tier 1 (Section 4A)
      tabu.py                  # Compare: Tabu Search (Section 4B)
      ga.py                    # Compare: Genetic Algorithm (Section 4B)
      ga_tabu.py               # Compare: GA+Tabu hybrid (Section 4B)
    dispatch.py                # tier selection at runtime (Section 8)
    tuning/
      tune_alns.py             # offline hyperparameter search (Section 6-tuning / 6H)
      profile_switch.py        # offline tier-threshold study (Section 7)
    experiments/
      run_benchmark.py         # compares solvers, produces tables/plots (Section 9)
    ui/
      app.py                   # Streamlit UI (Section 10)
  config/
    alns_params.json           # tuned hyperparameters (output of tuning)
    switch_policy.json         # tuned threshold τ + cpsat_time_limit_sec
    tabu_params.json           # Tabu defaults
    ga_params.json             # GA defaults
    ga_tabu_params.json        # hybrid defaults
  data/
    instances/                 # generated instances
    results/                   # solver outputs, benchmark CSVs, plots
  tests/
    test_validate.py test_greedy.py test_cpsat_small.py test_alns.py
    test_tabu.py test_ga.py test_ga_tabu.py test_dispatch.py
```

### 3.1 Uniform solver interface
```python
class Solver(Protocol):
    name: str
    def solve(self, inst: Instance, *, time_limit_sec: float | None = None,
              seed: int | None = None, warm_start: Solution | None = None) -> Solution: ...
```
All three tiers **and** the compare solvers (Tabu, GA, GA+Tabu) implement this.
The dispatcher and experiments depend only on this interface, so solvers are
interchangeable and directly comparable.

---

## 4. Tier 3 — Greedy ERD/SPT list scheduler (baseline + warm start)

Deterministic, `O(K log K)`. Used both as a reported baseline and as the ALNS/CP-SAT
warm start.

### 4.1 Algorithm
```
sort ops by key (r ascending, then p ascending, then w descending, then uid) -> order
gate_free[g] = list of occupied intervals per gate (start with empty)
for k in order:
    best = None
    for g in 1..G:
        t = earliest_start(g, r_k, p_k, T)      # earliest feasible start >= r_k on gate g
        if t is not None:
            cost = w_k * t
            if best is None or (t, g) < best.key:   # minimize start, tie -> lowest gate
                best = (t, g)
    if best is None: raise Infeasible
    assign start_k = best.t, gate_k = best.g
    insert interval [t, t+p_k) into gate g's interval list
```

### 4.2 `earliest_start(g, r, p, T)`
Given sorted non-overlapping intervals on gate `g`, return the smallest `t ≥ r` such
that `[t, t+p)` fits in a gap and `t ≤ T − p + 1`; else `None`.
- Scan gaps: before first interval, between consecutive intervals, after last.
- Candidate `t = max(r, gap_start)`; accept if `t + p ≤ gap_end+1` and `t ≤ T−p+1`.

### 4.3 Variants (for the report / ablation)
- **ERD-only** (sort by `r` then `uid`).
- **SPT-within-ready** (event-driven: when a gate frees, pick shortest ready job).
Provide both `greedy_erd_spt` (default) and `greedy_spt_ready` for comparison.

---

## 4A. Tier 1 — OR-Tools CP-SAT exact solver

### 4A.1 Variables
For each op `k`:
- `s_k = NewIntVar(r_k, T − p_k + 1, "s_k")` (start; release & horizon baked into domain).
- For each gate `g`: presence literal `x[k,g] = NewBoolVar` and an **optional interval**
  `iv[k,g] = NewOptionalIntervalVar(s_k, p_k, s_k + p_k, x[k,g], "iv")`.
  (Start var `s_k` is shared across gates so the start is gate-independent.)

### 4A.2 Constraints
- **Exactly one gate:** `AddExactlyOne(x[k,g] for g in 1..G)` for each `k`.
- **No overlap per gate:** for each gate `g`: `AddNoOverlap(iv[k,g] for all k)`.
  (Optional intervals with `presence=False` are ignored automatically.)

### 4A.3 Objective
Minimize the true weighted sum of starts. CP-SAT requires integer coefficients, so
map float weights `w_k` to integers that **preserve ratios**:
- If all `w_k` are integers → use them as-is (`SCALE = 1`).
- Else choose the smallest decimal scale `10^d` such that `round(w_k · 10^d)` recovers
  each weight exactly (within float tolerance); fall back to `SCALE = 1000` if needed.

Do **not** use naive `round(w_k)` with `SCALE = 1` on fractional weights (e.g. `1.3` and
`0.7` both become `1`) — that optimizes the wrong objective and can make a “CP-SAT”
result worse than greedy under the float cost, which previously caused silent greedy
substitution in the dispatcher.

### 4A.4 Solve settings
- `solver.parameters.max_time_in_seconds = time_limit_sec` (if given).
- Default / forced CP-SAT wall budget comes from `switch_policy.json` field
  `cpsat_time_limit_sec` (default **10** s) when the caller does not pass an override.
- `solver.parameters.num_search_workers = 8` (configurable).
- Warm start (optional): use `AddHint(s_k, warm.starts[k])` and `AddHint(x[k,gate], 1)`
  from the greedy solution.
- Cooperative cancel: UI / caller may set a `stop_event` or call `StopSearch()`; search
  stops on proven optimal, timeout, or stop request.
- Report `proven_optimal = (status == OPTIMAL)` and not user-stopped. If `FEASIBLE` only
  (hit time limit / stopped with incumbent), return best found with `proven_optimal=False`.
- If `INFEASIBLE` / no incumbent → raise (should have been caught by precheck 1.6).
- Forced `force_tier="cpsat"`: return the CP-SAT incumbent **as-is** (do not silently
  replace with greedy via `best_of` while still labeling the run as CP-SAT).

**Why CP-SAT (not manual Big-M MILP):** native `NoOverlap`/interval primitives map
directly onto the gate constraint, less code, fewer bugs. (A Big-M or time-indexed MILP
is documented in the brief and may be added as an optional appendix solver, but is not
required.)

---

## 4B. Compare metaheuristics (Tabu, GA, GA+Tabu) — reporting only

These solvers are **not** selected by Auto. Encoding matches ALNS Section 5.1:
permutation of uids + `decode` / objective via greedy earliest-start. Init order =
Tier-3 ERD/SPT. Never return worse than the greedy warm start. Infeasible permutations
(under a tight horizon) are rejected / scored as `+∞`.

### 4B.1 Tabu Search (`src/solvers/tabu.py`, `name="tabu"`)
- Neighborhood each iteration: random sample of **swap** and **relocate** moves
  (`swap_prob`, `neighborhood_size`).
- Tabu list of move keys with tenure `tabu_tenure`; **aspiration** if a move improves
  the global best.
- Shared kernel `improve_order(...)` used by standalone Tabu and by the hybrid.
- Stop on `max_iterations` or `time_limit_sec`. Config: `config/tabu_params.json`
  (defaults: tenure 7, neighborhood 40, max_iterations 20000, swap_prob 0.5).

### 4B.2 Genetic Algorithm (`src/solvers/ga.py`, `name="ga"`)
- Population of permutations; fitness = weighted sum of starts (minimize).
- Selection: tournament (`tournament_k`); crossover: **OX**; mutation: swap / insert;
  elitism: keep top `elite_count`.
- Reject infeasible offspring (keep parent copy). Config: `config/ga_params.json`
  (defaults: pop 40, elite 2, tournament 3, crossover 0.9, mutation 0.2,
  max_generations 5000).

### 4B.3 Hybrid GA+Tabu (`src/solvers/ga_tabu.py`, `name="ga_tabu"`)
Memetic algorithm: GA outer loop for global exploration; after each generation run a
**short Tabu** local search on all elites and on each remaining individual with
probability `local_search_rate` (via `improve_order`). Config: `config/ga_tabu_params.json`
(defaults: pop 30, elite 2, tournament 3, crossover 0.9, mutation 0.2,
max_generations 3000, local_search_rate 0.3, local_tabu_iters 80,
local_neighborhood_size 20, tabu_tenure 7, swap_prob 0.5).

---

## 5. Tier 2 — Adaptive Large Neighborhood Search (ALNS)

### 5.1 Solution encoding & decoder (order-based)
A solution is a **permutation** `π` of all `K` operation uids. The **decoder** turns `π`
into concrete `(start, gate)` via the greedy earliest-start rule of Section 4 applied in
the order given by `π`:
```
decode(π) -> Solution:
    reset gate interval lists
    for k in π:
        place k at earliest feasible start over all gates (as in 4.1/4.2)
    return Solution + objective
```
This guarantees feasibility for any permutation (given the feasibility precheck passed)
and makes the objective purely a function of the order. ALNS therefore searches over
orderings.

### 5.2 Initial solution
`π0 = order produced by Tier-3 greedy ERD/SPT` (Section 4.1). `x_best = x_cur = decode(π0)`.

### 5.3 Destroy operators (remove `q` ops from `π`)
`q` drawn each iteration uniformly from `[q_min, q_max]` where
`q_min = max(1, ceil(ρ_min · K))`, `q_max = max(q_min, ceil(ρ_max · K))`,
then capped by `q_cap` when `q_cap > 0`.
1. **Random removal** — remove `q` uniformly random ops.
2. **Worst removal** — remove the `q` ops with the largest `w_k · start_k` contribution
   (highest cost first; add small randomization via a `p`-biased selection, exponent `d_wr`).
3. **Related (Shaw) removal** — pick a seed op, remove the `q−1` most "related" ops, where
   relatedness `rel(a,b) = α·|r_a − r_b| + β·|p_a − p_b| + γ·[gate_a≠gate_b] + δ·|start_a − start_b|`
   (smaller = more related). Coefficients are fixed constants (`α=β=1, γ=T/4, δ=1`).
4. **Block removal** — decode gate assignments, pick a gate with ≥2 ops, remove a contiguous
   run of length `min(q, run)` in start-time order on that gate (falls back to random if no
   eligible gate).
5. **History removal** — track each op's best (lowest) start time seen in accepted solutions;
   remove the `q` ops with largest `w_k · (start_k − best_start_seen_k)`.

Removed ops go to a "removal bank"; remaining ops keep relative order in `π`.

### 5.4 Repair operators (reinsert removal bank into `π`)
1. **Greedy insertion** — for each removed op (processed in random order), try inserting at
   every position in `π`, decode, keep the position with min objective; commit; repeat.
2. **Regret-k insertion** (`k=2` or `3`) — for each removed op compute the best `k`
   insertion costs; insert the op with the **largest regret**
   `sum_{m=2..k}(cost_m − cost_1)` first; repeat until bank empty.
3. **Noised greedy insertion** — same as greedy, but each position cost is multiplied by
   `Uniform(1−η, 1+η)` (`noise_level` = `η`) before choosing the best slot.

Insertion costs use incremental prefix decode (exact suffix re-decode per candidate). When
`screen_top > 0`, only the best `screen_top` positions by cheap proxy (`w·t` of the inserted
op on the prefix state) are exact-decoded; others are treated as `+∞`.

### 5.5 Adaptive operator selection (Ropke–Pisinger)
- Maintain weights `ω_i` (destroy) and `ω'_j` (repair), all init `1.0`.
- Select operator with probability proportional to its weight (roulette).
- Track per-segment scores. After applying a (destroy, repair) pair, add reward `ψ`:
  - `ψ = σ1` if new **global best**;
  - else `ψ = σ2` if better than current;
  - else `ψ = σ3` if accepted but not better;
  - else `0`.
- After every `segment_length` iterations, update each used operator:
  `ω = (1 − λ)·ω + λ·(accumulated_score / times_used)`, then reset segment scores.
  `λ` = reaction factor.

### 5.6 Acceptance criterion (Simulated Annealing)
- Temperature `Temp = Temp0` initially; after each iteration `Temp *= cooling`.
- Accept candidate `x'` if `obj(x') ≤ obj(x_cur)` OR with prob
  `exp(−(obj(x') − obj(x_cur)) / Temp)`.
- `Temp0` set so a solution `start_temp_ctrl` (e.g. 5%) worse than initial is accepted with
  prob 0.5: `Temp0 = −(0.05 · obj(x0)) / ln(0.5)`.
- Always update `x_best` when improved.
- **Reheat on stagnation:** if no global-best improvement for `stagnation_iters` iterations,
  set `Temp = Temp0 · reheat_factor`, reset the current solution to `x_best`, and continue.
  Report `reheat_count` in solution meta.

### 5.7 Stopping criterion
Stop when **either** `max_iterations` reached **or** `time_limit_sec` elapsed (whichever
first). Both configurable; the dispatcher passes a time limit.

### 5.8 Reproducibility
All randomness from a single seeded `random.Random(seed)` / `numpy` Generator. `solve()`
accepts `seed`. Report `iterations`, final `gap_pct` (vs best known if available).

### 5.9 ALNS hyperparameters (the tunable set)
| Param | Symbol | Default (lit.) | Search range |
|-------|--------|----------------|--------------|
| min destroy fraction | `ρ_min` | 0.10 | [0.05, 0.20] |
| max destroy fraction | `ρ_max` | 0.30 | [0.20, 0.35] |
| reaction factor | `λ` | 0.15 | [0.10, 0.20] |
| segment length | `seg` | 150 | [100, 200] |
| score: new best | `σ1` | 33 | [10, 50] |
| score: better | `σ2` | 9 | [5, 25] |
| score: accepted | `σ3` | 13 | [1, 20] |
| SA cooling | `cooling` | 0.99975 | [0.995, 0.99999] |
| SA start-worse accept | `start_temp_ctrl` | 0.05 | [0.01, 0.20] |
| regret depth | `k` | 3 | {2, 3, 4} |
| worst-removal bias | `d_wr` | 3 | [1, 6] |
| max iterations | `max_iterations` | 25000 | fixed per time budget |
| destroy size cap | `q_cap` | 25 | [4, 40] (0 = uncapped) |
| insertion screen | `screen_top` | 5 | [0, 10] (0 = exact all) |
| noised-greedy η | `noise_level` | 0.02 | [0.0, 0.10] |
| SA stagnation reheat | `stagnation_iters` | 2000 | [500, 5000] |
| reheat temp factor | `reheat_factor` | 0.5 | [0.25, 1.0] |

Defaults come from Ropke & Pisinger (2006) and are the fallback if no tuning is run.

---

## 6. Instance generator (for tuning & profiling)

`instance_gen.py` produces reproducible random instances:
```
gen_instance(seed, M, N, G, T=None, p_range=(1,pmax), rel_frac=0.5, w1=1, w2=1):
    p_k ~ UniformInt(p_range)
    T = T or ceil( ( sum(p_k)/G ) * horizon_slack )   # slack e.g. 1.5, ensure T > max p
    deliveries: rdt = 1 (or Uniform(1, rel_frac*T) if release extension enabled)
    pickups: release ~ UniformInt(1, max(1, floor(rel_frac*T)))
    assert feasibility precheck (1.6); regenerate T if needed
```
Generate three named suites (all seeded, saved to `data/instances/`):
- **TUNE suite** — used only to tune ALNS hyperparameters.
- **TEST suite** — disjoint from TUNE; used for final reported results (never tuned on).
- **PROFILE suite** — size-sweep grid for the switch study (Section 7).

Sizes span `K = M+N ∈ {5, 10, 15, 20, 25, 30, 40, 60, 100, 200, 400}`, several seeds each,
varying `G ∈ {1,2,3,5}` and `T` slack. Keep TUNE and TEST disjoint by seed ranges.

---

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
  "notes": "Manual Auto policy: CP-SAT when K=M+N<=25, ALNS when K>25; cpsat_time_limit_sec=10s"
}
```
- `budget_sec` — Auto ALNS / shared UI default for metaheuristics.
- `cpsat_time_limit_sec` — default wall budget for forced / exact CP-SAT (UI default when
  Algorithm = CP-SAT).
Also emit the **crossover plot** (CP-SAT vs ALNS runtime & quality vs `K`) to
`data/results/` for the report.

---

## 8. Runtime dispatcher (production path)

```python
def solve(inst, policy=load("config/switch_policy.json"),
          params=load("config/alns_params.json"),
          exact_time_limit=None, alns_time_limit=None, seed=0,
          force_tier=None, stop_event=None):
    # force_tier in {None/"auto", "greedy", "cpsat", "alns", "tabu", "ga", "ga_tabu"}
    feasibility_precheck(inst)
    warm = GreedyERDSPT().solve(inst)
    if force_tier == "greedy":
        return warm, "greedy"
    # exact_time_limit defaults to policy.cpsat_time_limit_sec (else budget_sec)
    # alns_time_limit defaults to policy.budget_sec
    if force_tier == "cpsat" or (auto and K <= threshold_K and T ok):
        sol = CPSAT().solve(..., time_limit_sec=exact_time_limit,
                            warm_start=warm, stop_event=stop_event)
        if auto and not sol.proven_optimal:
            sol = ALNS(...).solve(..., warm_start=warm)
            return best_of(sol, warm), "alns_fallback"
        if force_tier == "cpsat":
            return sol, "cpsat"          # no best_of swap; no ALNS fallback
        return best_of(sol, warm), "cpsat"
    if force_tier == "tabu":
        return best_of(TabuSearch().solve(...), warm), "tabu"
    if force_tier == "ga":
        return best_of(GeneticAlgorithm().solve(...), warm), "ga"
    if force_tier == "ga_tabu":
        return best_of(HybridGATabu().solve(...), warm), "ga_tabu"
    # force_tier == "alns" or auto with K > threshold
    return best_of(ALNS(...).solve(...), warm), "alns"
```

- **Primary Auto decision is O(1)** (size comparison), no timeout paid.
- Auto never selects `tabu` / `ga` / `ga_tabu`.
- Forced CP-SAT: no ALNS fallback; return CP-SAT incumbent as-is.
- `best_of` keeps metaheuristic Auto/ALNS/compare paths never worse than greedy.

---

## 9. Experiments & reporting (`experiments/run_benchmark.py`)

Produce, on the TEST suite:
1. **Comparison table** per instance and aggregated: columns
   `{greedy, alns, tabu, ga, ga_tabu, cpsat}` ×
   `{objective, runtime_sec, gap_pct_vs_optimum}`. `gap_pct = (obj − opt)/opt · 100`
   where `opt` = CP-SAT optimum when available.
2. **Crossover plot** (from Section 7): runtime vs `K`, CP-SAT vs ALNS (log y-axis).
3. **ALNS convergence plot**: best objective vs iteration for a few representative instances.
4. **Ablations** (optional): greedy variants; ALNS with/without each operator; tuned vs
   default `θ`; Tabu vs GA vs hybrid.
All outputs to `data/results/` as CSV + PNG/HTML.

---

## 10. UI (`ui/app.py`, Streamlit) — presentation for the teacher

Centerpiece = **color-coded Gantt chart**. Requirements:
- **Inputs panel:** sliders/fields for `M, N, T, G, w1, w2`, seed; button "Generate";
  or upload an instance JSON. Button "Solve".
- **Algorithm select:** Auto (policy) | Greedy | ALNS | Tabu Search | Genetic Algorithm |
  GA+Tabu Hybrid | CP-SAT. When not Auto, run only the chosen solver.
- **Time limit:** defaults from `config/switch_policy.json`
  (`cpsat_time_limit_sec` when CP-SAT selected; `budget_sec` otherwise).
- **Compare all solvers** (Auto only): also run greedy, ALNS, Tabu, GA, hybrid, CP-SAT
  side by side.
- **CP-SAT / Auto async:** Stop button cooperatively cancels CP-SAT via `StopSearch`.
- **Gantt chart:** Y-axis = gates `1..G`; X-axis = time `1..T`; each op a horizontal bar
  `[start, start+p)`. **Deliveries and pickups in two distinct colorblind-safe colors**
  (e.g. blue / orange). Show release markers/shaded "unavailable" region per pickup so
  release compliance is visually evident. Labels/tooltips: op id, start, p, gate.
- **Results table:** objective, runtime, proven-optimal, tier/solver name.
- **Downloads:** instance JSON and solution JSON.
- **Which tier fired:** display the dispatcher's choice (Auto) or forced solver.
- **Live & reproducible:** re-solve on a fresh/random instance on demand; fixed seed field.
- **Accessibility:** colorblind-safe palette, large fonts for projection, clear legend/units.

Fallback if time is short: a **Jupyter notebook** rendering the same Gantt + table + plots
(top-to-bottom narrative). Streamlit preferred for live interaction.

---

## 11. Testing (`tests/`)

- `test_validate.py`: hand-built feasible & infeasible solutions → validator accepts/rejects
  (overlap, release violation, horizon violation).
- `test_greedy.py`: tiny instances with known schedules; determinism; feasibility.
- `test_cpsat_small.py`: instances small enough to enumerate/verify optimum; check
  `proven_optimal` and objective equals brute-force optimum for `K ≤ 8`; fractional
  weights; stop_event.
- `test_alns.py`: ALNS ≤ greedy objective (never worse than warm start); reproducible with
  fixed seed; feasibility on many random instances.
- `test_tabu.py` / `test_ga.py` / `test_ga_tabu.py`: never worse than greedy; seed
  reproducibility; feasibility; time-limit respect; small-instance obj ≥ CP-SAT.
- `test_dispatch.py`: dispatcher picks correct tier per policy; `force_tier` for
  greedy/cpsat/alns/tabu/ga/ga_tabu; forced CP-SAT never returns `alns_fallback`.
- **Cross-check:** on small instances, metaheuristic objs ≥ `cpsat.obj` (optimum is a
  lower bound) and `greedy.obj ≥ cpsat.obj`.

---

## 12. Deliverables checklist

- [x] `src/` with all modules and the uniform `Solver` interface.
- [x] Feasibility precheck + shared validator.
- [x] Tier 3 greedy (ERD/SPT) + variants.
- [x] Tier 1 CP-SAT (optional-interval model, weight scaling, warm start, proven-optimal,
      stop/timeout).
- [x] Tier 2 ALNS (destroy/repair operators, adaptive weights, SA acceptance, seeds).
- [x] Compare solvers: Tabu Search, Genetic Algorithm, GA+Tabu hybrid (Section 4B).
- [x] `config/alns_params.json` from tuning (Section 6H) + tuning log.
- [x] `config/switch_policy.json` from profiling (Section 7) + `cpsat_time_limit_sec`.
- [x] `config/tabu_params.json`, `ga_params.json`, `ga_tabu_params.json`.
- [x] Dispatcher with `force_tier` (Section 8).
- [x] Benchmark + plots (Section 9).
- [x] Streamlit UI with Gantt, algorithm select, compare-all, Stop, downloads (Section 10).
- [x] Tests (Section 11), all passing.
- [x] README: how to generate instances, tune, profile, run UI, reproduce results.

---

## 13. Defaults if you skip tuning/profiling (safe fallbacks)

- ALNS params: literature defaults in Section 5.9 (or tuned file if present).
- Switch threshold: `threshold_K = 25`, `T_cap = null`, `budget_sec` for ALNS,
  **`cpsat_time_limit_sec = 10`** for forced/exact CP-SAT, with Auto ALNS fall-back when
  CP-SAT does not prove optimality.
- Tabu / GA / hybrid: defaults in Section 4B config files (hand-chosen starting points;
  not offline-tuned).
- Weights: `w1 = w2 = 1`.

These make the system runnable end-to-end before the offline studies are done; the studies
then replace the fallbacks with tuned values and give you report material.
