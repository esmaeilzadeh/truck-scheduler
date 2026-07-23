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
