# Truck Gate Scheduler

Multi-solver toolkit for external truck gate scheduling
(\(Pm \mid r_j \mid \sum w_j C_j\)):

| Solver | Role |
|--------|------|
| OR-Tools CP-SAT | Exact optimum on small instances |
| ALNS | Near-optimal metaheuristic (Auto tier for large `K`) |
| Tabu Search | Local-search metaheuristic (compare) |
| Genetic Algorithm | Global population search (compare) |
| GA+Tabu Hybrid | Memetic: GA global + Tabu local (compare) |
| Greedy ERD/SPT | Baseline and warm start |

The **dispatcher** picks CP-SAT vs ALNS from `config/switch_policy.json` (O(1) at solve time).
Tabu, GA, and the hybrid are available for comparison via UI / `force_tier` (never selected by Auto).

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -r requirements.txt
```

Run commands from the repo root so `src` imports resolve.

## Quick start (UI)

```bash
.venv/bin/streamlit run src/ui/app.py
```

Generate or upload an instance, choose an **Algorithm**
(Auto / Greedy / ALNS / Tabu Search / Genetic Algorithm / GA+Tabu Hybrid / CP-SAT),
click **Solve**. Enable **Compare all solvers** to run all solvers side by side.

## Generate instance suites

```bash
.venv/bin/python -m src.instance_gen
```

Writes disjoint **TUNE / TEST / PROFILE** JSON suites under `data/instances/`
plus `data/instances/manifest.json`.

## Offline studies (short course budgets)

```bash
# ALNS hyperparameter search → config/alns_params.json
.venv/bin/python -m src.tuning.tune_alns

# Tier-switch profiling → config/switch_policy.json + crossover plot
.venv/bin/python -m src.tuning.profile_switch

# Benchmark on TEST suite → data/results/
.venv/bin/python -m src.experiments.run_benchmark
```

Without running these, SPEC §13 fallbacks in `config/` already make the dispatcher work.

## Tests

```bash
.venv/bin/pytest -q
```

## Python API

```python
from src.instance_gen import gen_instance
from src.dispatch import solve

inst = gen_instance(seed=42, M=5, N=5, G=2)
sol, tier = solve(inst, seed=0)                    # Auto: size policy
sol, tier = solve(inst, force_tier="alns")         # force ALNS
sol, tier = solve(inst, force_tier="tabu")         # Tabu Search
sol, tier = solve(inst, force_tier="ga")           # Genetic Algorithm
sol, tier = solve(inst, force_tier="ga_tabu")      # GA + Tabu hybrid
sol, tier = solve(inst, force_tier="cpsat")        # force CP-SAT (no ALNS fallback)
sol, tier = solve(inst, force_tier="greedy")       # warm start only
print(tier, sol.objective(inst), sol.starts, sol.gates)
```

## Deliverables checklist

- [x] Core model, validator, I/O, Solver protocol
- [x] Greedy ERD/SPT (+ variants)
- [x] CP-SAT exact solver
- [x] ALNS (destroy/repair, adaptive weights, SA)
- [x] Tabu Search + Genetic Algorithm + GA/Tabu hybrid (compare solvers)
- [x] Instance generator + TUNE/TEST/PROFILE suites
- [x] `config/alns_params.json` + `config/switch_policy.json`
- [x] `config/tabu_params.json` + `config/ga_params.json` + `config/ga_tabu_params.json`
- [x] Dispatcher
- [x] Benchmark runner + plots
- [x] Streamlit UI with Gantt
- [x] Tests
- [x] This README
