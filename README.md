# Truck Gate Scheduler

3-tier solver for external truck gate scheduling
(\(Pm \mid r_j \mid \sum w_j C_j\)):

| Tier | Solver | Role |
|------|--------|------|
| 1 | OR-Tools CP-SAT | Exact optimum on small instances |
| 2 | ALNS | Near-optimal metaheuristic on larger instances |
| 3 | Greedy ERD/SPT | Baseline and warm start |

The **dispatcher** picks Tier 1 vs 2 from `config/switch_policy.json` (O(1) at solve time).

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

Generate or upload an instance, choose an **Algorithm** (Auto / Greedy / ALNS / CP-SAT),
click **Solve**. Enable **Compare all solvers** to also run greedy / ALNS / CP-SAT side by side.

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
sol, tier = solve(inst, force_tier="cpsat")        # force CP-SAT (no ALNS fallback)
sol, tier = solve(inst, force_tier="greedy")       # warm start only
print(tier, sol.objective(inst), sol.starts, sol.gates)
```

## Deliverables checklist

- [x] Core model, validator, I/O, Solver protocol
- [x] Greedy ERD/SPT (+ variants)
- [x] CP-SAT exact solver
- [x] ALNS (destroy/repair, adaptive weights, SA)
- [x] Instance generator + TUNE/TEST/PROFILE suites
- [x] `config/alns_params.json` + `config/switch_policy.json`
- [x] Dispatcher
- [x] Benchmark runner + plots
- [x] Streamlit UI with Gantt
- [x] Tests
- [x] This README
