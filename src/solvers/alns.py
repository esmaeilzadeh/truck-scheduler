"""Tier 2 — Adaptive Large Neighborhood Search (ALNS) metaheuristic."""

from __future__ import annotations

import bisect
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.model import Infeasible, Instance, Operation, Solution, feasibility_precheck
from src.solvers.greedy import _place_op, earliest_start, greedy_erd_spt


# ---------------------------------------------------------------------------
# Permutation decoder (Section 5.1 / 5.2)
# ---------------------------------------------------------------------------

def decode(inst: Instance, order: list[int]) -> Solution:
    """Decode a permutation of uid's into a Solution via greedy earliest-start."""
    ops_by_uid = {op.uid: op for op in inst.ops}
    gate_intervals: dict[int, list[tuple[int, int]]] = {
        g: [] for g in range(1, inst.G + 1)
    }
    starts: dict[int, int] = {}
    gates: dict[int, int] = {}

    for uid in order:
        op = ops_by_uid[uid]
        t, g = _place_op(op, gate_intervals, inst.G, inst.T)
        starts[uid] = t
        gates[uid] = g
        bisect.insort(gate_intervals[g], (t, t + op.p))

    return Solution(starts=starts, gates=gates)


def _objective_from_order(inst: Instance, order: list[int]) -> float:
    ops_by_uid = {op.uid: op for op in inst.ops}
    gate_intervals: dict[int, list[tuple[int, int]]] = {
        g: [] for g in range(1, inst.G + 1)
    }
    total = 0.0

    for uid in order:
        op = ops_by_uid[uid]
        t, g = _place_op(op, gate_intervals, inst.G, inst.T)
        bisect.insort(gate_intervals[g], (t, t + op.p))
        total += op.w * t

    return total


# ---------------------------------------------------------------------------
# Schedule state (incremental destroy / repair evaluation)
# ---------------------------------------------------------------------------

@dataclass
class _ScheduleState:
    """Decoded schedule with incremental remove / insert for ALNS moves."""

    inst: Instance
    ops_by_uid: dict[int, Operation]
    gate_intervals: dict[int, list[tuple[int, int, int]]]
    starts: dict[int, int]
    gates: dict[int, int]
    objective: float

    @classmethod
    def from_solution(cls, inst: Instance, sol: Solution) -> _ScheduleState:
        ops_by_uid = {op.uid: op for op in inst.ops}
        gate_intervals: dict[int, list[tuple[int, int, int]]] = {
            g: [] for g in range(1, inst.G + 1)
        }
        starts = dict(sol.starts)
        gates = dict(sol.gates)
        objective = 0.0
        for uid, t in starts.items():
            op = ops_by_uid[uid]
            g = gates[uid]
            bisect.insort(gate_intervals[g], (t, t + op.p, uid))
            objective += op.w * t
        return cls(
            inst=inst,
            ops_by_uid=ops_by_uid,
            gate_intervals=gate_intervals,
            starts=starts,
            gates=gates,
            objective=objective,
        )

    @classmethod
    def from_order(cls, inst: Instance, order: list[int]) -> _ScheduleState:
        return cls.from_solution(inst, decode(inst, order))

    def copy(self) -> _ScheduleState:
        return _ScheduleState(
            inst=self.inst,
            ops_by_uid=self.ops_by_uid,
            gate_intervals={
                g: list(intervals) for g, intervals in self.gate_intervals.items()
            },
            starts=dict(self.starts),
            gates=dict(self.gates),
            objective=self.objective,
        )

    def remove(self, uid: int) -> None:
        t = self.starts.pop(uid)
        g = self.gates.pop(uid)
        op = self.ops_by_uid[uid]
        interval = (t, t + op.p, uid)
        intervals = self.gate_intervals[g]
        idx = bisect.bisect_left(intervals, interval)
        if idx >= len(intervals) or intervals[idx] != interval:
            raise KeyError(f"interval for uid {uid} not found on gate {g}")
        intervals.pop(idx)
        self.objective -= op.w * t

    def best_insertion(self, op: Operation) -> tuple[float, int, int]:
        """Cheapest insertion: (delta_obj = w*t, start t, gate g)."""
        best: tuple[float, int, int] | None = None
        for g in range(1, self.inst.G + 1):
            t = earliest_start(
                self.gate_intervals[g], op.r, op.p, self.inst.T,
            )
            if t is None:
                continue
            cost = op.w * t
            if best is None or (cost, t, g) < best:
                best = (cost, t, g)
        if best is None:
            raise Infeasible(f"Cannot place operation {op.uid}")
        return best

    def insert(self, uid: int, t: int, g: int) -> None:
        op = self.ops_by_uid[uid]
        bisect.insort(self.gate_intervals[g], (t, t + op.p, uid))
        self.starts[uid] = t
        self.gates[uid] = g
        self.objective += op.w * t

    def order(self) -> list[int]:
        """Canonical permutation: uids sorted by (start, gate)."""
        return sorted(self.starts.keys(), key=lambda u: (self.starts[u], self.gates[u]))


def _deadline_hit(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


# ---------------------------------------------------------------------------
# Destroy operators (Section 5.3) — operate on schedule state
# ---------------------------------------------------------------------------

def _random_remove(
    state: _ScheduleState, q: int, rng: random.Random, **_kwargs,
) -> list[int]:
    """Remove q random ops from the schedule; return the removal bank."""
    uids = list(state.starts.keys())
    removed = rng.sample(uids, min(q, len(uids)))
    for uid in removed:
        state.remove(uid)
    return removed


def _worst_remove(
    state: _ScheduleState, q: int, rng: random.Random, d_wr: float = 3.0, **_kwargs,
) -> list[int]:
    """Remove q ops with highest w·start contribution (biased rank selection)."""
    cost_list = [
        (state.ops_by_uid[uid].w * state.starts[uid], uid)
        for uid in state.starts
    ]
    cost_list.sort(key=lambda x: -x[0])

    removed: list[int] = []
    for _ in range(min(q, len(cost_list))):
        n = len(cost_list)
        if n == 0:
            break
        weights = [(n - i) ** d_wr for i in range(n)]
        idx = rng.choices(range(n), weights=weights, k=1)[0]
        _, uid = cost_list.pop(idx)
        removed.append(uid)
        state.remove(uid)
    return removed


def _related_removal(
    state: _ScheduleState, q: int, rng: random.Random, **_kwargs,
) -> list[int]:
    """Shaw/related removal: seed + most related ops."""
    uids = list(state.starts.keys())
    if not uids:
        return []

    ops_by_uid = state.ops_by_uid
    T = state.inst.T
    alpha, beta, gamma, delta = 1.0, 1.0, T / 4, 1.0

    seed_uid = rng.choice(uids)

    def rel(a: int, b: int) -> float:
        oa, ob = ops_by_uid[a], ops_by_uid[b]
        return (
            alpha * abs(oa.r - ob.r)
            + beta * abs(oa.p - ob.p)
            + gamma * (1 if state.gates[a] != state.gates[b] else 0)
            + delta * abs(state.starts[a] - state.starts[b])
        )

    candidates = [(rel(seed_uid, uid), uid) for uid in uids if uid != seed_uid]
    candidates.sort()

    removed = [seed_uid]
    for _, uid in candidates:
        if len(removed) >= q:
            break
        removed.append(uid)

    for uid in removed:
        state.remove(uid)
    return removed


# ---------------------------------------------------------------------------
# Repair operators (Section 5.4) — schedule-based insertion
# ---------------------------------------------------------------------------

def _gate_insertion_costs(
    state: _ScheduleState, op: Operation,
) -> list[tuple[float, int, int]]:
    """Per-gate insertion costs as (cost, t, g), sorted ascending by cost."""
    costs: list[tuple[float, int, int]] = []
    for g in range(1, state.inst.G + 1):
        t = earliest_start(state.gate_intervals[g], op.r, op.p, state.inst.T)
        if t is None:
            continue
        costs.append((op.w * t, t, g))
    costs.sort()
    return costs


def _greedy_repair(
    state: _ScheduleState,
    removed: list[int],
    rng: random.Random,
    deadline: float | None = None,
    **_kwargs,
) -> None:
    """Insert each removed op at its cheapest (t, g) via best_insertion."""
    bank = list(removed)
    rng.shuffle(bank)
    for i, uid in enumerate(bank):
        if _deadline_hit(deadline):
            for rest in bank[i:]:
                if rest not in state.starts:
                    _, t, g = state.best_insertion(state.ops_by_uid[rest])
                    state.insert(rest, t, g)
            return
        _, t, g = state.best_insertion(state.ops_by_uid[uid])
        state.insert(uid, t, g)


def _regret_k_repair(
    state: _ScheduleState,
    removed: list[int],
    rng: random.Random,
    k: int = 3,
    deadline: float | None = None,
    **_kwargs,
) -> None:
    """Regret-k over per-gate insertion costs; insert largest-regret op first."""
    del rng
    bank = list(removed)

    while bank:
        if _deadline_hit(deadline):
            for uid in bank:
                if uid not in state.starts:
                    _, t, g = state.best_insertion(state.ops_by_uid[uid])
                    state.insert(uid, t, g)
            return

        best_uid: int | None = None
        best_regret = -float("inf")
        best_tg: tuple[int, int] = (0, 1)

        for uid in bank:
            if _deadline_hit(deadline):
                for rest in bank:
                    if rest not in state.starts:
                        _, t, g = state.best_insertion(state.ops_by_uid[rest])
                        state.insert(rest, t, g)
                return

            costs = _gate_insertion_costs(state, state.ops_by_uid[uid])
            if not costs:
                raise Infeasible(f"Cannot place operation {uid}")
            c1 = costs[0][0]
            regret = sum(costs[m][0] - c1 for m in range(1, min(k, len(costs))))
            if regret > best_regret or (
                regret == best_regret and (best_uid is None or uid < best_uid)
            ):
                best_regret = regret
                best_uid = uid
                best_tg = (costs[0][1], costs[0][2])

        if best_uid is None:
            break
        state.insert(best_uid, best_tg[0], best_tg[1])
        bank.remove(best_uid)


# ---------------------------------------------------------------------------
# ALNS solver (Section 5.5 - 5.9)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "rho_min": 0.10,
    "rho_max": 0.30,
    "lambda": 0.15,
    "segment_length": 150,
    "sigma1": 33,
    "sigma2": 9,
    "sigma3": 13,
    "cooling": 0.99975,
    "start_temp_ctrl": 0.05,
    "final_temp_ratio": 0.002,
    "regret_k": 3,
    "d_wr": 3.0,
    "max_iterations": 25000,
    # 0 = disabled (use full rho_min/rho_max range)
    "q_cap": 0,
}

DESTROY_OPS = [
    _random_remove,
    _worst_remove,
    _related_removal,
]

REPAIR_OPS = [
    _greedy_repair,
    _regret_k_repair,
]

_DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[2] / "config" / "alns_params.json"


def _load_params_file(path: Path | None = None) -> dict:
    path = path or _DEFAULT_PARAMS_PATH
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


@dataclass
class ALNS:
    """ALNS metaheuristic solver implementing the Solver protocol."""

    name: str = "alns"
    params: dict | None = None
    params_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.params_path is None:
            self.params_path = _DEFAULT_PARAMS_PATH

    def solve(
        self,
        inst: Instance,
        *,
        time_limit_sec: float | None = None,
        seed: int | None = None,
        warm_start: Solution | None = None,
    ) -> Solution:
        feasibility_precheck(inst)
        t0 = time.perf_counter()
        deadline = (
            t0 + time_limit_sec if time_limit_sec is not None else None
        )

        file_params = _load_params_file(
            Path(self.params_path) if self.params_path else None
        )
        p = {**DEFAULT_PARAMS, **file_params, **(self.params or {})}
        rng = random.Random(seed if seed is not None else 0)
        K = len(inst.ops)

        # Initial solution: warm-start order when provided, else ERD/SPT (5.2)
        greedy_sol = greedy_erd_spt(inst)
        if warm_start is not None:
            init_order = sorted(
                warm_start.starts.keys(),
                key=lambda u: (warm_start.starts[u], warm_start.gates[u]),
            )
            # Guard: use greedy if warm start is incomplete
            if len(init_order) != K:
                init_order = [
                    op.uid
                    for op in sorted(
                        inst.ops, key=lambda op: (op.r, op.p, -op.w, op.uid)
                    )
                ]
            greedy_ref = warm_start
        else:
            init_order = [
                op.uid
                for op in sorted(inst.ops, key=lambda op: (op.r, op.p, -op.w, op.uid))
            ]
            greedy_ref = greedy_sol

        cur_state = _ScheduleState.from_order(inst, init_order)
        best_order = list(init_order)
        obj_cur = cur_state.objective
        obj_best = obj_cur
        obj_greedy = greedy_ref.objective(inst)

        temp0 = (
            -(p["start_temp_ctrl"] * obj_cur) / math.log(0.5) if obj_cur > 0 else 1.0
        )
        temp = temp0
        final_temp_ratio = float(p.get("final_temp_ratio", 0.002))

        n_destroy = len(DESTROY_OPS)
        n_repair = len(REPAIR_OPS)
        w_destroy = [1.0] * n_destroy
        w_repair = [1.0] * n_repair
        score_destroy = [0.0] * n_destroy
        score_repair = [0.0] * n_repair
        use_destroy = [0] * n_destroy
        use_repair = [0] * n_repair

        q_min = max(1, math.ceil(p["rho_min"] * K))
        q_max = max(q_min, math.ceil(p["rho_max"] * K))
        q_cap = int(p.get("q_cap", 0) or 0)
        if q_cap > 0:
            q_min = min(q_min, q_cap)
            q_max = min(q_max, q_cap)
            q_max = max(q_min, q_max)

        max_iter = p["max_iterations"]
        iterations = 0

        for it in range(max_iter):
            if _deadline_hit(deadline):
                break

            iterations = it + 1

            d_idx = rng.choices(range(n_destroy), weights=w_destroy, k=1)[0]
            r_idx = rng.choices(range(n_repair), weights=w_repair, k=1)[0]
            q = rng.randint(q_min, q_max)

            cand = cur_state.copy()
            destroy_kwargs = {"d_wr": p["d_wr"]} if d_idx == 1 else {}
            removed = DESTROY_OPS[d_idx](cand, q, rng, **destroy_kwargs)

            # Left-shift remaining ops so repair insertions see contiguous gaps
            if removed:
                cand = _ScheduleState.from_solution(inst, decode(inst, cand.order()))

            if _deadline_hit(deadline):
                break

            repair_kwargs: dict = {"deadline": deadline}
            if r_idx == 1:
                repair_kwargs["k"] = p["regret_k"]
            REPAIR_OPS[r_idx](cand, removed, rng, **repair_kwargs)

            # One compaction decode per iteration (left-shift / order invariant)
            compacted = decode(inst, cand.order())
            cand = _ScheduleState.from_solution(inst, compacted)
            obj_new = cand.objective

            delta_obj = obj_new - obj_cur
            accept = False
            if delta_obj <= 0:
                accept = True
            elif temp > 0:
                if rng.random() < math.exp(-delta_obj / temp):
                    accept = True

            # Scoring only for accepted moves (SPEC 5.5)
            psi = 0
            if accept:
                if obj_new < obj_best:
                    psi = p["sigma1"]
                elif obj_new < obj_cur:
                    psi = p["sigma2"]
                else:
                    psi = p["sigma3"]

                cur_state = cand
                obj_cur = obj_new
                if obj_new < obj_best:
                    obj_best = obj_new
                    best_order = cand.order()

                score_destroy[d_idx] += psi
                score_repair[r_idx] += psi

            use_destroy[d_idx] += 1
            use_repair[r_idx] += 1

            # Time-based cooling when a budget is set; else classic geometric
            if time_limit_sec is not None and time_limit_sec > 0:
                elapsed = time.perf_counter() - t0
                frac = min(1.0, max(0.0, elapsed / time_limit_sec))
                temp = temp0 * (final_temp_ratio ** frac)
            else:
                temp *= p["cooling"]

            if (it + 1) % p["segment_length"] == 0:
                lam = p["lambda"]
                for i in range(n_destroy):
                    if use_destroy[i] > 0:
                        w_destroy[i] = (1 - lam) * w_destroy[i] + lam * (
                            score_destroy[i] / use_destroy[i]
                        )
                    score_destroy[i] = 0.0
                    use_destroy[i] = 0
                for i in range(n_repair):
                    if use_repair[i] > 0:
                        w_repair[i] = (1 - lam) * w_repair[i] + lam * (
                            score_repair[i] / use_repair[i]
                        )
                    score_repair[i] = 0.0
                    use_repair[i] = 0

        runtime = time.perf_counter() - t0
        sol = decode(inst, best_order)

        # Never worse than greedy / warm-start baseline
        if obj_best > obj_greedy:
            sol = Solution(
                starts=dict(greedy_ref.starts),
                gates=dict(greedy_ref.gates),
            )
            obj_best = obj_greedy

        gap_pct = 0.0
        if obj_greedy > 0:
            gap_pct = max(0.0, (obj_best - obj_greedy) / obj_greedy * 100)

        sol.proven_optimal = False
        sol.runtime_sec = runtime
        sol.meta = {
            "iterations": iterations,
            "gap_pct": gap_pct,
        }
        return sol
