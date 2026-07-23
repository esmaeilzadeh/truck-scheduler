## 8. Runtime dispatcher (production path)

```python
def solve(inst, policy=load("config/switch_policy.json"),
          params=load("config/alns_params.json"),
          exact_time_limit=None, alns_time_limit=None, seed=0,
          force_tier=None, stop_event=None):
    # force_tier in {None/"auto", "greedy", "cpsat", "alns", "alns_tabu",
    #                "tabu", "ga", "ga_tabu"}
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
    if force_tier == "alns":
        return best_of(ALNS(...).solve(...), warm), "alns"
    # force_tier == "alns_tabu" or auto with K > threshold
    return best_of(HybridALNSTabu().solve(...), warm), "alns_tabu"
```

- **Primary Auto decision is O(1)** (size comparison), no timeout paid.
- Auto large-K selects `alns_tabu`. Auto never selects `tabu` / `ga` / `ga_tabu`.
- Forced CP-SAT: no ALNS fallback; return CP-SAT incumbent as-is.
- `best_of` keeps metaheuristic Auto/ALNS/compare paths never worse than greedy.

---
