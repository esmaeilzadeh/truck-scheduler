## 1. Problem model (authoritative definitions)

### 1.1 Entities
- `M` outbound deliveries (SSCGD), indexed `i = 1..M`.
- `N` inbound pickups (SSCPI), indexed `j = 1..N`.
- Unified set of **operations** `K = M + N`, indexed `k = 0..K-1`.
  Each operation has: `type ∈ {DELIVERY, PICKUP}`, processing time `p_k = TPG_k`,
  release time `r_k`, weight `w_k`.
  - Delivery: `r_k = RDT_i` (default `1`), `w_k = w1`.
  - Pickup:   `r_k = TSSCPI_j`, `w_k = w2`.

### 1.2 Parameters
- `T` — number of time periods; constraint `T > max(p_k)`.
- `G` — number of gates, `G ≥ 1`.
- `w1, w2 ≥ 0` — objective weights (default `w1 = w2 = 1`).

### 1.3 Decision (solution) representation (canonical, solver-independent)
A solution assigns to every operation `k`:
- `start_k ∈ {1, …, T}` — start period.
- `gate_k ∈ {1, …, G}` — assigned gate.

Occupancy interval is **half-open** `[start_k, start_k + p_k)`; last occupied period is
`start_k + p_k − 1`.

### 1.4 Constraints
1. **Release:** `start_k ≥ r_k` for all `k`.
2. **Horizon:** `start_k + p_k − 1 ≤ T` ⇔ `start_k ≤ T − p_k + 1`.
3. **Gate non-overlap:** for any two ops `a ≠ b` with `gate_a = gate_b`:
   `start_a + p_a ≤ start_b` OR `start_b + p_b ≤ start_a`.

### 1.5 Objective
Minimize
\[
\text{Cost} = w_1 \sum_{i} DT_i + w_2 \sum_{j} PT_j = \sum_{k} w_k \cdot start_k .
\]

### 1.6 Feasibility precheck (run before any solver)
- `T > max(p_k)`.
- Workload fits: `sum(p_k) ≤ G · T`. If violated → declare **infeasible**, do not solve.
- Per-op fits: `r_k + p_k − 1 ≤ T` for every `k`. If any op cannot fit after its
  release, declare infeasible.

---
