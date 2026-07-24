# ALNS vs ALNS+Tabu vs GA+Tabu — size × budget report

Source CSV: [`alns_alns_tabu_ga_tabu_size_budget.csv`](alns_alns_tabu_ga_tabu_size_budget.csv)

## Setup

- Solvers: `alns`, `alns_tabu`, `ga_tabu`
- Sizes (M=N): 15, 30, 50, 100, 150, 200 — **10 instances each**
- Gates: G cycles through **2..10** across the 10 instances
- Budgets: **10s**, **60s (1m)**, **120s (2m)**
- Metric: mean **improvement vs greedy** (%) = `(greedy_obj - obj) / greedy_obj * 100` (higher better)
- Wins: unique best-of-3 objective on the same `(instance, budget)` (ties listed separately)
- Rows: **540** (complete grid)

## By size × budget — mean improvement vs greedy (%)

| Size | Budget | ALNS | ALNS+Tabu | GA+Tabu | Best | Wins (alns/at/gt) | Ties | n |
|---|---:|---:|---:|---:|---|---|---:|---:|
| 15x15 | 10s | 1.907 | 2.121 | 2.182 | GA+Tabu | 1/0/1 | 8 | 10 |
| 15x15 | 60s | 2.075 | 2.229 | 2.260 | GA+Tabu | 0/0/1 | 9 | 10 |
| 15x15 | 120s | 2.068 | 2.229 | 2.260 | GA+Tabu | 0/0/1 | 9 | 10 |
| 30x30 | 10s | 1.641 | 2.081 | 2.773 | GA+Tabu | 0/1/8 | 1 | 10 |
| 30x30 | 60s | 2.286 | 2.663 | 2.888 | GA+Tabu | 0/3/6 | 1 | 10 |
| 30x30 | 120s | 2.319 | 2.767 | 3.074 | GA+Tabu | 0/0/9 | 1 | 10 |
| 50x50 | 10s | 1.616 | 1.984 | 3.206 | GA+Tabu | 0/0/10 | 0 | 10 |
| 50x50 | 60s | 2.181 | 2.512 | 3.549 | GA+Tabu | 0/0/10 | 0 | 10 |
| 50x50 | 120s | 2.247 | 2.802 | 3.712 | GA+Tabu | 0/0/10 | 0 | 10 |
| 100x100 | 10s | 2.343 | 2.437 | 2.710 | GA+Tabu | 1/4/5 | 0 | 10 |
| 100x100 | 60s | 2.941 | 2.999 | 3.995 | GA+Tabu | 0/0/10 | 0 | 10 |
| 100x100 | 120s | 3.049 | 3.347 | 4.007 | GA+Tabu | 0/1/9 | 0 | 10 |
| 150x150 | 10s | 1.341 | 1.564 | 1.546 | ALNS+Tabu | 1/3/6 | 0 | 10 |
| 150x150 | 60s | 2.216 | 2.438 | 3.336 | GA+Tabu | 0/2/8 | 0 | 10 |
| 150x150 | 120s | 2.599 | 2.451 | 3.545 | GA+Tabu | 2/0/8 | 0 | 10 |
| 200x200 | 10s | 1.408 | 1.026 | 0.963 | ALNS | 5/2/3 | 0 | 10 |
| 200x200 | 60s | 2.099 | 2.303 | 3.328 | GA+Tabu | 0/2/8 | 0 | 10 |
| 200x200 | 120s | 2.479 | 2.568 | 3.978 | GA+Tabu | 1/0/9 | 0 | 10 |

## By size (budgets pooled)

| Size | ALNS | ALNS+Tabu | GA+Tabu | Wins (alns/at/gt) | Ties | n |
|---|---:|---:|---:|---|---:|---:|
| 15x15 | 2.017 | 2.193 | 2.234 | 1/0/3 | 26 | 30 |
| 30x30 | 2.082 | 2.503 | 2.912 | 0/4/23 | 3 | 30 |
| 50x50 | 2.015 | 2.433 | 3.489 | 0/0/30 | 0 | 30 |
| 100x100 | 2.777 | 2.928 | 3.571 | 1/5/24 | 0 | 30 |
| 150x150 | 2.052 | 2.151 | 2.809 | 3/5/22 | 0 | 30 |
| 200x200 | 1.995 | 1.966 | 2.756 | 6/4/20 | 0 | 30 |

## By budget (sizes pooled)

| Budget | ALNS | ALNS+Tabu | GA+Tabu | Wins (alns/at/gt) | Ties | n |
|---:|---:|---:|---:|---|---:|---:|
| 10s | 1.709 | 1.869 | 2.230 | 8/10/33 | 9 | 60 |
| 60s | 2.299 | 2.524 | 3.226 | 0/7/43 | 10 | 60 |
| 120s | 2.460 | 2.694 | 3.429 | 3/1/46 | 10 | 60 |

## By size × budget — mean % worse than best-of-3

Lower is better (0 = was the best solver on that job).

| Size | Budget | ALNS | ALNS+Tabu | GA+Tabu |
|---|---:|---:|---:|---:|
| 15x15 | 10s | 0.330 | 0.110 | 0.048 |
| 15x15 | 60s | 0.190 | 0.032 | 0.000 |
| 15x15 | 120s | 0.196 | 0.032 | 0.000 |
| 30x30 | 10s | 1.206 | 0.752 | 0.034 |
| 30x30 | 60s | 0.681 | 0.290 | 0.057 |
| 30x30 | 120s | 0.785 | 0.318 | 0.000 |
| 50x50 | 10s | 1.645 | 1.261 | 0.000 |
| 50x50 | 60s | 1.421 | 1.075 | 0.000 |
| 50x50 | 120s | 1.528 | 0.946 | 0.000 |
| 100x100 | 10s | 0.655 | 0.557 | 0.275 |
| 100x100 | 60s | 1.100 | 1.039 | 0.000 |
| 100x100 | 120s | 1.031 | 0.720 | 0.033 |
| 150x150 | 10s | 0.493 | 0.266 | 0.287 |
| 150x150 | 60s | 1.261 | 1.032 | 0.099 |
| 150x150 | 120s | 1.062 | 1.215 | 0.074 |
| 200x200 | 10s | 0.196 | 0.584 | 0.651 |
| 200x200 | 60s | 1.457 | 1.246 | 0.183 |
| 200x200 | 120s | 1.618 | 1.523 | 0.052 |

## Pairwise win rate by size × budget

Share of instances where A is **strictly better** than B.

| Size | Budget | ALNS+Tabu > ALNS | GA+Tabu > ALNS | GA+Tabu > ALNS+Tabu |
|---|---:|---:|---:|---:|
| 15x15 | 10s | 40% | 40% | 10% |
| 15x15 | 60s | 20% | 30% | 10% |
| 15x15 | 120s | 30% | 40% | 10% |
| 30x30 | 10s | 70% | 80% | 80% |
| 30x30 | 60s | 70% | 90% | 60% |
| 30x30 | 120s | 90% | 100% | 90% |
| 50x50 | 10s | 50% | 100% | 100% |
| 50x50 | 60s | 70% | 100% | 100% |
| 50x50 | 120s | 80% | 100% | 100% |
| 100x100 | 10s | 50% | 70% | 50% |
| 100x100 | 60s | 70% | 100% | 100% |
| 100x100 | 120s | 70% | 90% | 90% |
| 150x150 | 10s | 80% | 70% | 60% |
| 150x150 | 60s | 70% | 90% | 80% |
| 150x150 | 120s | 40% | 80% | 90% |
| 200x200 | 10s | 20% | 30% | 50% |
| 200x200 | 60s | 90% | 90% | 80% |
| 200x200 | 120s | 50% | 90% | 90% |

## Takeaways

- **15×15:** mostly ties; differences are tiny.
- **30–100:** GA+Tabu dominates; ALNS+Tabu usually beats pure ALNS.
- **150×150 @ 10s:** ALNS+Tabu slightly edges GA+Tabu on mean improvement; GA+Tabu still has more unique wins.
- **200×200 @ 10s:** pure ALNS is best on average (short budget / large K).
- **200×200 @ 60s/120s:** GA+Tabu takes over again.
- Overall: ALNS+Tabu improves on ALNS, but **GA+Tabu is strongest on average**, especially at 1–2 minute budgets.

