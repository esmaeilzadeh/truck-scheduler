"""Tier 2 — Adaptive Large Neighborhood Search (ALNS) metaheuristic."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from src.model import Infeasible, Instance, Solution, feasibility_precheck
from src.solvers.greedy import _decode_order, _place_op, earliest_start, greedy_erd_spt


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
        gate_intervals[g].append((t, t + op.p))
        gate_intervals[g].sort()

    return Solution(starts=starts, gates=gates)


def _objective_from_order(inst: Instance, order: list[int]) -> float:
    ops_by_uid = {op.uid: op for op in inst.ops}
    gate_intervals: dict[int, list[tuple[int, int]]] = {
        g: [] for g in range(1, inst.G + 1)
    }
    starts: dict[int, int] = {}
    gates: dict[int, int] = {}
    total = 0.0

    for uid in order:
        op = ops_by_uid[uid]
        t, g = _place_op(op, gate_intervals, inst.G, inst.T)
        starts[uid] = t
        gates[uid] = g
        gate_intervals[g].append((t, t + op.p))
        gate_intervals[g].sort()
        total += op.w * t

    return total


# ---------------------------------------------------------------------------
# Destroy operators (Section 5.3)
# ---------------------------------------------------------------------------

def _random_remove(
    order: list[int], q: int, rng: random.Random, inst: Instance,
) -> tuple[list[int], list[int]]:
    """Remove q random ops from order."""
    to_remove = set(rng.sample(range(len(order)), min(q, len(order))))
    remaining = [uid for i, uid in enumerate(order) if i not in to_remove]
    removed = [uid for i, uid in enumerate(order) if i in to_remove]
    return remaining, removed


def _worst_remove(
    order: list[int], q: int, rng: random.Random, inst: Instance,
    d_wr: float = 3.0,
) -> tuple[list[int], list[int]]:
    """Remove q ops with highest cost contribution, biased selection."""
    ops_by_uid = {op.uid: op for op in inst.ops}
    # Compute current cost contribution for each op
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
        gate_intervals[g].append((t, t + op.p))
        gate_intervals[g].sort()

    cost_list = [(ops_by_uid[uid].w * temp_starts[uid], uid) for uid in order]
    cost_list.sort(key=lambda x: -x[0])  # highest cost first

    remaining_order = list(order)
    removed: list[int] = []
    for _ in range(min(q, len(cost_list))):
        # Biased selection: pick from remaining with probability ∝ rank^d_wr
        n = len(cost_list)
        if n == 0:
            break
        weights = [(i + 1) ** d_wr for i in range(n)]
        idx = rng.choices(range(n), weights=weights, k=1)[0]
        _, uid = cost_list.pop(idx)
        removed.append(uid)
        remaining_order.remove(uid)

    return remaining_order, removed


def _related_removal(
    order: list[int], q: int, rng: random.Random, inst: Instance,
) -> tuple[list[int], list[int]]:
    """Shaw/related removal: seed + most related ops."""
    ops_by_uid = {op.uid: op for op in inst.ops}

    # Compute current starts for relatedness
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
        gate_intervals[g].append((t, t + op.p))
        gate_intervals[g].sort()

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

    # Compute relatedness to seed for all other ops
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
) -> list[int]:
    """Insert each removed op at best position (min objective)."""
    rng.shuffle(removed)
    order = list(remaining)
    for uid in removed:
        best_pos = 0
        best_cost = float("inf")
        for pos in range(len(order) + 1):
            trial = order[:pos] + [uid] + order[pos:]
            cost = _objective_from_order(inst, trial)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        order.insert(best_pos, uid)
    return order


def _regret_k_repair(
    remaining: list[int], removed: list[int], rng: random.Random,
    inst: Instance, k: int = 3,
) -> list[int]:
    """Regret-k insertion: insert op with largest regret first."""
    order = list(remaining)
    bank = list(removed)

    while bank:
        best_uid = None
        best_regret = -float("inf")
        best_pos = 0

        for uid in bank:
            costs: list[float] = []
            positions: list[int] = []
            for pos in range(len(order) + 1):
                trial = order[:pos] + [uid] + order[pos:]
                cost = _objective_from_order(inst, trial)
                costs.append(cost)
                positions.append(pos)

            # Sort by cost
            sorted_pairs = sorted(zip(costs, positions))
            c1 = sorted_pairs[0][0]
            regret = sum(
                sorted_pairs[m][0] - c1 for m in range(1, min(k, len(sorted_pairs)))
            )

            if regret > best_regret or (regret == best_regret and (best_uid is None or uid < best_uid)):
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


@dataclass
class ALNS:
    """ALNS metaheuristic solver implementing the Solver protocol."""

    name: str = "alns"
    params: dict | None = None

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

        p = {**DEFAULT_PARAMS, **(self.params or {})}
        rng = random.Random(seed if seed is not None else 0)
        K = len(inst.ops)

        # Initial solution from greedy (Section 5.2)
        init_order = [op.uid for op in inst.ops]
        if warm_start is not None:
            # Use warm start order if available
            init_order = list(warm_start.starts.keys())
            # Ensure all ops are present
            all_uids = {op.uid for op in inst.ops}
            present = set(init_order)
            for uid in all_uids:
                if uid not in present:
                    init_order.append(uid)

        # Use greedy order as initial
        greedy_sol = greedy_erd_spt(inst)
        init_order = sorted(inst.ops, key=lambda op: (op.r, op.p, -op.w, op.uid))
        init_order = [op.uid for op in init_order]

        x_cur = decode(inst, init_order)
        x_best = x_cur
        best_order = list(init_order)
        cur_order = list(init_order)

        obj_cur = x_cur.objective(inst)
        obj_best = obj_cur

        # Temperature initialization (Section 5.6)
        temp0 = -(p["start_temp_ctrl"] * obj_cur) / math.log(0.5) if obj_cur > 0 else 1.0
        temp = temp0

        # Destroy/repair weight arrays
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

        max_iter = p["max_iterations"]
        iterations = 0

        for it in range(max_iter):
            if time_limit_sec is not None:
                elapsed = time.perf_counter() - t0
                if elapsed >= time_limit_sec:
                    break

            iterations = it + 1

            # Select destroy operator (roulette)
            d_idx = rng.choices(range(n_destroy), weights=w_destroy, k=1)[0]
            r_idx = rng.choices(range(n_repair), weights=w_repair, k=1)[0]

            q = rng.randint(q_min, q_max)

            # Destroy
            remaining, removed = DESTROY_OPS[d_idx](cur_order, q, rng, inst)

            # Repair
            new_order = REPAIR_OPS[r_idx](remaining, removed, rng, inst,
                                          k=p["regret_k"] if r_idx == 1 else 2)

            # Evaluate candidate
            obj_new = _objective_from_order(inst, new_order)

            # Scoring
            psi = 0
            if obj_new < obj_best:
                psi = p["sigma1"]
            elif obj_new < obj_cur:
                psi = p["sigma2"]
            else:
                psi = p["sigma3"]

            # SA acceptance
            delta_obj = obj_new - obj_cur
            accept = False
            if delta_obj <= 0:
                accept = True
            elif temp > 0:
                prob = math.exp(-delta_obj / temp)
                if rng.random() < prob:
                    accept = True

            if accept:
                cur_order = new_order
                obj_cur = obj_new

                if obj_new < obj_best:
                    obj_best = obj_new
                    best_order = list(new_order)

                score_destroy[d_idx] += psi
                score_repair[r_idx] += psi

            use_destroy[d_idx] += 1
            use_repair[r_idx] += 1

            # Update temperature
            temp *= p["cooling"]

            # Update weights every segment
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
        sol.proven_optimal = False
        sol.runtime_sec = runtime
        sol.meta = {
            "iterations": iterations,
            "gap_pct": 0.0,
        }
        return sol
