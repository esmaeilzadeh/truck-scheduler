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
If a `warm_start` solution is provided, `π0` is its ops ordered by `(start, gate)`.
Otherwise `π0` is the Tier-3 greedy ERD/SPT order (Section 4.1).
`x_best = x_cur = decode(π0)`. Never return worse than the greedy / warm-start baseline.

### 5.3 Destroy operators (remove `q` ops from the decoded schedule)
Destroy/repair operate on a **schedule state** (per-gate interval lists + start/gate maps),
not by re-decoding a permutation at every candidate. `q` is drawn each iteration uniformly
from `[q_min, q_max]` where `q_min = max(1, ceil(ρ_min · K))`,
`q_max = max(q_min, ceil(ρ_max · K))`. Optional `q_cap > 0` clamps both bounds
(`q_cap = 0` disables the cap).

1. **Random removal** — remove `q` uniformly random ops.
2. **Worst removal** — remove the `q` ops with the largest `w_k · start_k` contribution
   (highest cost first; biased rank selection with exponent `d_wr`), reading starts from
   the current schedule state.
3. **Related (Shaw) removal** — pick a seed op, remove the `q−1` most "related" ops, where
   relatedness `rel(a,b) = α·|r_a − r_b| + β·|p_a − p_b| + γ·[gate_a≠gate_b] + δ·|start_a − start_b|`
   (smaller = more related). Coefficients are fixed constants (`α=β=1, γ=T/4, δ=1`).

Removed ops go to a removal bank. After destroy, remaining ops are **compacted** once via
`decode(order_by_(start,gate))` so survivors left-shift before repair.

### 5.4 Repair operators (reinsert removal bank into the schedule)
Insertion cost for an op is the cheapest earliest-start over all gates (one gap scan per
gate, ~O(K/G) per candidate), committed via incremental insert into the schedule state.

1. **Greedy insertion** — for each removed op (random order), insert at its cheapest
   `(start, gate)`.
2. **Regret-k insertion** — for each removed op compute the best `k` **per-gate** insertion
   costs; insert the op with the **largest regret**
   `sum_{m=2..k}(cost_m − cost_1)` first; repeat until bank empty.
   (With `G=2` this is effectively regret-2 over gates.)

After the bank is empty, **one compaction decode** per iteration rebuilds the schedule from
`order_by_(start, gate)` so the objective remains a function of the order (left-shift never
hurts). Deadline is checked once per iteration (and per bank op on large `K`).

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
- Accept candidate `x'` if `obj(x') ≤ obj(x_cur)` OR with prob
  `exp(−(obj(x') − obj(x_cur)) / Temp)`.
- `Temp0` set so a solution `start_temp_ctrl` (e.g. 5%) worse than initial is accepted with
  prob 0.5: `Temp0 = −(start_temp_ctrl · obj(x0)) / ln(0.5)`.
- **Time-based cooling** when `time_limit_sec` is set:
  `Temp = Temp0 · final_temp_ratio^(elapsed / budget)` so the search intensifies by the
  deadline regardless of iteration throughput.
- Without a time limit, classic geometric cooling: after each iteration `Temp *= cooling`.
- Always update `x_best` when improved.

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
| SA cooling (no time limit) | `cooling` | 0.99975 | [0.995, 0.99999] |
| SA start-worse accept | `start_temp_ctrl` | 0.05 | [0.01, 0.20] |
| SA final temp ratio | `final_temp_ratio` | 0.002 | [0.0005, 0.01] |
| regret depth | `k` | 3 | {2, 3, 4} |
| worst-removal bias | `d_wr` | 3 | [1, 6] |
| destroy size cap | `q_cap` | 0 (disabled) | [0, 30] |
| max iterations | `max_iterations` | 25000 | fixed per time budget |

Defaults come from Ropke & Pisinger (2006) and are the fallback if no tuning is run.
`q_cap = 0` means use the full `ρ_min`/`ρ_max` destroy-size range.

---
