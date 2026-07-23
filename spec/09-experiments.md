## 9. Experiments & reporting (`experiments/run_benchmark.py`)

Produce, on the TEST suite:
1. **Comparison table** per instance and aggregated: columns
   `{greedy, alns, alns_tabu, tabu, ga, ga_tabu, cpsat}` ×
   `{objective, runtime_sec, gap_pct_vs_optimum}`. `gap_pct = (obj − opt)/opt · 100`
   where `opt` = CP-SAT optimum when available.
2. **Crossover plot** (from Section 7): runtime vs `K`, CP-SAT vs ALNS/ALNS+Tabu (log y-axis).
3. **ALNS convergence plot**: best objective vs iteration for a few representative instances.
4. **Ablations** (optional): greedy variants; ALNS with/without each operator; tuned vs
   default `θ`; Tabu vs GA vs hybrids.
All outputs to `data/results/` as CSV + PNG/HTML.

---
