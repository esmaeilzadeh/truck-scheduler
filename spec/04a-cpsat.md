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
