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
      alns.py                  # Tier 2 core ALNS (Section 5)
      alns_tabu.py             # Tier 2 Auto path: ALNS+Tabu (Section 4B.4)
      cpsat.py                 # Tier 1 (Section 4A)
      tabu.py                  # Compare: Tabu Search (Section 4B)
      ga.py                    # Compare: Genetic Algorithm (Section 4B)
      ga_tabu.py               # Compare: GA+Tabu hybrid (Section 4B)
    dispatch.py                # tier selection at runtime (Section 8)
    tuning/
      tune_alns.py             # offline ALNS hyperparameter search (Section 6H)
      tune_alns_tabu.py        # offline ALNS+Tabu random search
      tune_ga_tabu.py          # offline GA+Tabu random search
      random_search.py         # shared TunerSpec / random-search framework
      profile_switch.py        # offline tier-threshold study (Section 7)
    experiments/
      run_benchmark.py         # compares solvers, produces tables/plots (Section 9)
    ui/
      app.py                   # Streamlit UI (Section 10)
  config/
    alns_params.json           # tuned ALNS hyperparameters
    alns_tabu_params.json      # tuned ALNS+Tabu hyperparameters
    switch_policy.json         # tuned threshold τ + cpsat_time_limit_sec
    tabu_params.json           # Tabu defaults
    ga_params.json             # GA defaults
    ga_tabu_params.json        # GA+Tabu hybrid defaults
  data/
    instances/                 # generated instances
    results/                   # solver outputs, benchmark CSVs, plots
  tests/
    test_validate.py test_greedy.py test_cpsat_small.py test_alns.py
    test_alns_tabu.py test_tabu.py test_ga.py test_ga_tabu.py test_dispatch.py
```

### 3.1 Uniform solver interface
```python
class Solver(Protocol):
    name: str
    def solve(self, inst: Instance, *, time_limit_sec: float | None = None,
              seed: int | None = None, warm_start: Solution | None = None) -> Solution: ...
```
All three tiers **and** the metaheuristics (Tabu, GA, GA+Tabu, ALNS+Tabu, plain ALNS)
implement this. The dispatcher and experiments depend only on this interface, so solvers
are interchangeable and directly comparable.

---
