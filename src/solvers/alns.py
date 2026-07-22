"""Tier 2 — Adaptive Large Neighborhood Search (ALNS) metaheuristic."""

from __future__ import annotations

import bisect
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.model import Instance, Solution, feasibility_precheck
from src.solvers.greedy import _place_op, greedy_erd_spt


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


def _deadline_hit(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _append_remaining_random(
    order: list[int], remaining: list[int], rng: random.Random,
) -> list[int]:
    """Cheap fallback when the time budget expires mid-repair."""
    out = list(order)
    bag = list(remaining)
    rng.shuffle(bag)
    for uid in bag:
        out.insert(rng.randint(0, len(out)), uid)
    return out


# ---------------------------------------------------------------------------
# Destroy operators (Section 5.3)
# ---------------------------------------------------------------------------

def _random_remove(
    order: list[int], q: int, rng: random.Random, inst: Instance,
    **_kwargs,
) -> tuple[list[int], list[int]]:
    """Remove q random ops from order."""
    del inst
    to_remove = set(rng.sample(range(len(order)), min(q, len(order))))
    remaining = [uid for i, uid in enumerate(order) if i not in to_remove]
    removed = [uid for i, uid in enumerate(order) if i in to_remove]
    return remaining, removed


def _worst_remove(
    order: list[int], q: int, rng: random.Random, inst: Instance,
    d_wr: float = 3.0,
    **_kwargs,
) -> tuple[list[int], list[int]]:
    """Remove q ops with highest cost contribution, biased selection."""
    ops_by_uid = {op.uid: op for op in inst.ops}
    gate_intervals: dict[int, list[tuple[int, int]]] = {
        g: [] for g in range(1, inst.G + 1)
    }
    temp_starts: dict[int, int] = {}
    for uid in order:
        op = ops_by_uid[uid]
        t, g = _place_op(op, gate_intervals, inst.G, inst.T)
        temp_starts[uid] = t
        bisect.insort(gate_intervals[g], (t, t + op.p))

    # Highest cost first (rank 0 = worst)
    cost_list = [(ops_by_uid[uid].w * temp_starts[uid], uid) for uid in order]
    cost_list.sort(key=lambda x: -x[0])

    remaining_order = list(order)
    removed: list[int] = []
    for _ in range(min(q, len(cost_list))):
        n = len(cost_list)
        if n == 0:
            break
        # Prefer worse ranks: weight ∝ (n - i)^d_wr so index 0 gets highest weight
        weights = [(n - i) ** d_wr for i in range(n)]
        idx = rng.choices(range(n), weights=weights, k=1)[0]
        _, uid = cost_list.pop(idx)
        removed.append(uid)
        remaining_order.remove(uid)

    return remaining_order, removed


def _related_removal(
    order: list[int], q: int, rng: random.Random, inst: Instance,
    **_kwargs,
) -> tuple[list[int], list[int]]:
    """Shaw/related removal: seed + most related ops."""
    ops_by_uid = {op.uid: op for op in inst.ops}

    gate_intervals: dict[int, list[tuple[int, int]]] = {
        g: [] for g in range(1, inst.G + 1)
    }
    temp_starts: dict[int, int] = {}
    temp_gates: dict[int, int] = {}
    for uid in order:
        op = ops_by_uid[uid]
        t, g = _place_op(op, gate_intervals, inst.G, inst.T)
        temp_starts[uid] = t
        temp_gates[uid] = g
        bisect.insort(gate_intervals[g], (t, t + op.p))

    T = inst.T
    alpha, beta, gamma, delta = 1.0, 1.0, T / 4, 1.0

    seed_idx = rng.randint(0, len(order) - 1)
    seed_uid = order[seed_idx]

    def rel(a: int, b: int) -> float:
        oa, ob = ops_by_uid[a], ops_by_uid[b]
        return (
            alpha * abs(oa.r - ob.r)
            + beta * abs(oa.p - ob.p)
            + gamma * (1 if temp_gates.get(a, 0) != temp_gates.get(b, 0) else 0)
            + delta * abs(temp_starts.get(a, 0) - temp_starts.get(b, 0))
        )

    candidates = [(rel(seed_uid, uid), uid) for uid in order if uid != seed_uid]
    candidates.sort()

    removed = [seed_uid]
    for _, uid in candidates:
        if len(removed) >= q:
            break
        removed.append(uid)

    remaining = [uid for uid in order if uid not in set(removed)]
    return remaining, removed


# ---------------------------------------------------------------------------
# Repair operators (Section 5.4)
# ---------------------------------------------------------------------------

def _greedy_repair(
    remaining: list[int], removed: list[int], rng: random.Random, inst: Instance,
    deadline: float | None = None,
    **_kwargs,
) -> list[int]:
    """Insert each removed op at best position (min objective)."""
    rng.shuffle(removed)
    order = list(remaining)
    pending = list(removed)
    while pending:
        if _deadline_hit(deadline):
            return _append_remaining_random(order, pending, rng)
        uid = pending.pop(0)
        best_pos = 0
        best_cost = float("inf")
        for pos in range(len(order) + 1):
            if pos % 8 == 0 and _deadline_hit(deadline):
                return _append_remaining_random(order, [uid] + pending, rng)
            trial = order[:pos] + [uid] + order[pos:]
            cost = _objective_from_order(inst, trial)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        order.insert(best_pos, uid)
    return order


def _regret_k_repair(
    remaining: list[int], removed: list[int], rng: random.Random,
    inst: Instance, k: int = 3, deadline: float | None = None, **_kwargs,
) -> list[int]:
    """Regret-k insertion: insert op with largest regret first."""
    order = list(remaining)
    bank = list(removed)

    while bank:
        if _deadline_hit(deadline):
            return _append_remaining_random(order, bank, rng)

        best_uid = None
        best_regret = -float("inf")
        best_pos = 0

        for uid in bank:
            if _deadline_hit(deadline):
                return _append_remaining_random(order, bank, rng)
            costs: list[float] = []
            positions: list[int] = []
            for pos in range(len(order) + 1):
                if pos % 8 == 0 and _deadline_hit(deadline):
                    return _append_remaining_random(order, bank, rng)
                trial = order[:pos] + [uid] + order[pos:]
                cost = _objective_from_order(inst, trial)
                costs.append(cost)
                positions.append(pos)

            sorted_pairs = sorted(zip(costs, positions))
            c1 = sorted_pairs[0][0]
            regret = sum(
                sorted_pairs[m][0] - c1 for m in range(1, min(k, len(sorted_pairs)))
            )

            if regret > best_regret or (
                regret == best_regret and (best_uid is None or uid < best_uid)
            ):
                best_regret = regret
                best_uid = uid
                best_pos = sorted_pairs[0][1]

        if best_uid is None:
            break

        order.insert(best_pos, best_uid)
        bank.remove(best_uid)

    return order


# ---------------------------------------------------------------------------
# ALNS solver (Section 5.5 - 5.9)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "rho_min": 0.10,
    "rho_max": 0.40,
    "lambda": 0.10,
    "segment_length": 100,
    "sigma1": 33,
    "sigma2": 9,
    "sigma3": 13,
    "cooling": 0.99975,
    "start_temp_ctrl": 0.05,
    "regret_k": 3,
    "d_wr": 3.0,
    "max_iterations": 25000,
    # Cap destroy size so one repair stays cheap on large K
    "q_cap": 12,
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

        # Initial solution from greedy ERD/SPT order (Section 5.2)
        greedy_sol = warm_start if warm_start is not None else greedy_erd_spt(inst)
        init_order = [
            op.uid
            for op in sorted(inst.ops, key=lambda op: (op.r, op.p, -op.w, op.uid))
        ]

        x_cur = decode(inst, init_order)
        best_order = list(init_order)
        cur_order = list(init_order)

        obj_cur = x_cur.objective(inst)
        obj_best = obj_cur
        obj_greedy = greedy_sol.objective(inst)

        temp0 = (
            -(p["start_temp_ctrl"] * obj_cur) / math.log(0.5) if obj_cur > 0 else 1.0
        )
        temp = temp0

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
        q_cap = int(p.get("q_cap", 12))
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

            destroy_kwargs = {"d_wr": p["d_wr"]} if d_idx == 1 else {}
            remaining, removed = DESTROY_OPS[d_idx](
                cur_order, q, rng, inst, **destroy_kwargs
            )

            if _deadline_hit(deadline):
                break

            repair_kwargs = {"k": p["regret_k"]} if r_idx == 1 else {}
            repair_kwargs["deadline"] = deadline
            new_order = REPAIR_OPS[r_idx](
                remaining, removed, rng, inst, **repair_kwargs
            )

            obj_new = _objective_from_order(inst, new_order)

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

                cur_order = new_order
                obj_cur = obj_new
                if obj_new < obj_best:
                    obj_best = obj_new
                    best_order = list(new_order)

                score_destroy[d_idx] += psi
                score_repair[r_idx] += psi

            use_destroy[d_idx] += 1
            use_repair[r_idx] += 1
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

        # Never worse than greedy warm start
        if obj_best > obj_greedy:
            sol = Solution(
                starts=dict(greedy_sol.starts),
                gates=dict(greedy_sol.gates),
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
