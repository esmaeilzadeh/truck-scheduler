## 4. Tier 3 — Greedy ERD/SPT list scheduler (baseline + warm start)

Deterministic, `O(K log K)`. Used both as a reported baseline and as the ALNS/CP-SAT
warm start.

### 4.1 Algorithm
```
sort ops by key (r ascending, then p ascending, then w descending, then uid) -> order
gate_free[g] = list of occupied intervals per gate (start with empty)
for k in order:
    best = None
    for g in 1..G:
        t = earliest_start(g, r_k, p_k, T)      # earliest feasible start >= r_k on gate g
        if t is not None:
            cost = w_k * t
            if best is None or (t, g) < best.key:   # minimize start, tie -> lowest gate
                best = (t, g)
    if best is None: raise Infeasible
    assign start_k = best.t, gate_k = best.g
    insert interval [t, t+p_k) into gate g's interval list
```

### 4.2 `earliest_start(g, r, p, T)`
Given sorted non-overlapping intervals on gate `g`, return the smallest `t ≥ r` such
that `[t, t+p)` fits in a gap and `t ≤ T − p + 1`; else `None`.
- Scan gaps: before first interval, between consecutive intervals, after last.
- Candidate `t = max(r, gap_start)`; accept if `t + p ≤ gap_end+1` and `t ≤ T−p+1`.

### 4.3 Variants (for the report / ablation)
- **ERD-only** (sort by `r` then `uid`).
- **SPT-within-ready** (event-driven: when a gate frees, pick shortest ready job).
Provide both `greedy_erd_spt` (default) and `greedy_spt_ready` for comparison.

---
