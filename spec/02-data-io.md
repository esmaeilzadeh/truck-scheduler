## 2. Data formats & I/O contract

### 2.1 Instance JSON (input)
```json
{
  "id": "inst_0001",
  "T": 48,
  "G": 3,
  "w1": 1.0,
  "w2": 1.0,
  "deliveries": [ {"id": 0, "p": 3, "rdt": 1} ],
  "pickups":    [ {"id": 0, "p": 2, "release": 5} ]
}
```
- `rdt` optional (default 1). `release` = `TSSCPI_j` (required for pickups).

### 2.2 Solution JSON (output)
```json
{
  "instance_id": "inst_0001",
  "solver": "cpsat|alns|alns_tabu|greedy|tabu|ga|ga_tabu",
  "objective": 123.0,
  "is_optimal": true,
  "proven_optimal": true,
  "runtime_sec": 0.42,
  "assignments": [
    {"op_type": "delivery", "op_id": 0, "start": 1, "gate": 1, "p": 3, "weight": 1.0}
  ],
  "meta": { "iterations": 0, "gap_pct": 0.0 }
}
```

### 2.3 Internal `Operation` / `Instance` / `Solution` structures
```python
@dataclass(frozen=True)
class Operation:
    uid: int            # 0..K-1 global id
    kind: str           # "delivery" | "pickup"
    local_id: int       # i or j
    p: int              # processing time
    r: int              # release time
    w: float            # weight

@dataclass
class Instance:
    id: str
    T: int
    G: int
    ops: list[Operation]        # length K
    w1: float; w2: float

@dataclass
class Solution:
    starts: dict[int, int]      # uid -> start
    gates:  dict[int, int]      # uid -> gate
    # objective computed on demand; must be feasible before scoring
```

### 2.4 Validator (shared, mandatory)
Implement `validate(instance, solution) -> None` that raises on any violation of
Section 1.4 constraints and horizon/release bounds. **Every** solver's output must
pass the validator before its objective is trusted. Objective:
`sum(op.w * solution.starts[op.uid] for op in ops)`.

---
