## 4B. Metaheuristics (Tabu, GA, GA+Tabu, ALNS+Tabu)

Encoding matches ALNS Section 5.1: permutation of uids + `decode` / objective via greedy
earliest-start. Init order = Tier-3 ERD/SPT (or warm-start order). Never return worse than
the greedy / warm-start baseline. Infeasible permutations (under a tight horizon) are
rejected / scored as `+∞`.

**Auto:** selects **ALNS+Tabu** (§4B.4) when `K > threshold_K`.
**Reporting-only (never Auto):** Tabu, GA, GA+Tabu (§4B.1–4B.3) — UI / `force_tier` only.
Plain ALNS (§5) remains forceable via `force_tier="alns"`.

### 4B.1 Tabu Search (`src/solvers/tabu.py`, `name="tabu"`)
- Neighborhood each iteration: random sample of **swap** and **relocate** moves
  (`swap_prob`, `neighborhood_size`).
- Tabu list of move keys with tenure `tabu_tenure`; **aspiration** if a move improves
  the global best.
- Shared kernel `improve_order(...)` used by standalone Tabu and by the hybrids.
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

### 4B.4 Hybrid ALNS+Tabu (`src/solvers/alns_tabu.py`, `name="alns_tabu"`)
Memetic algorithm used as the **Auto Tier-2** path: schedule-based ALNS outer loop
(Section 5) for large-neighborhood exploration; after an accepted ALNS move, run a
**short Tabu** polish via `improve_order` (a) always on a new global best, and
(b) with probability `local_search_rate` on other accepted moves. Polished orders rebuild
the schedule state.

**Polish wall-time gate:** when `time_limit_sec` is set, cumulative time spent in Tabu
polish is capped at `local_search_budget_frac × time_limit_sec` (default **0.15**). Each
polish call also gets a per-call deadline of
`min(global_deadline, now + remaining_polish_budget)`. Further polish is skipped once the
cumulative budget is exhausted so ALNS keeps most of the wall clock on medium/large `K`.
With no time limit, the frac gate is disabled.

Config: `config/alns_tabu_params.json` (ALNS knobs from §5.9 plus defaults:
`local_search_rate` 0.15, `local_tabu_iters` 40, `local_neighborhood_size` 20,
`tabu_tenure` 7, `swap_prob` 0.5, `local_search_budget_frac` 0.15, `q_cap` 0). Offline
light tune via `src/tuning/tune_alns_tabu.py` (searches polish knobs / `rho_max` /
`local_search_budget_frac`; keeps `q_cap` frozen at 0).

---
