"""Tier 2 — Adaptive Large Neighborhood Search (ALNS) metaheuristic."""

from __future__ import annotations

import bisect
import json
import math
import random
import time
from collections.abc import Callable
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


def _empty_gate_intervals(G: int) -> dict[int, list[tuple[int, int]]]:
    return {g: [] for g in range(1, G + 1)}


def _copy_gate_intervals(
    gate_intervals: dict[int, list[tuple[int, int]]],
) -> dict[int, list[tuple[int, int]]]:
    return {g: list(ivs) for g, ivs in gate_intervals.items()}


def _decode_suffix_cost(
    inst: Instance,
    ops_by_uid: dict,
    suffix: list[int],
    gate_intervals: dict[int, list[tuple[int, int]]],
    prefix_cost: float,
) -> float:
    total = prefix_cost
    for uid in suffix:
        op = ops_by_uid[uid]
        t, g = _place_op(op, gate_intervals, inst.G, inst.T)
        bisect.insort(gate_intervals[g], (t, t + op.p))
        total += op.w * t
    return total


def _insertion_costs(
    inst: Instance,
    order: list[int],
    uid: int,
    deadline: float | None = None,
    screen_top: int = 0,
) -> list[float] | None:
    """Objective for inserting ``uid`` at every position in ``order``.

    Uses incremental prefix decode. When ``screen_top > 0``, only the best
    ``screen_top`` positions by cheap proxy (``w·t`` of the inserted op on the
    prefix state) get an exact suffix decode; others are ``+inf``.

    Returns ``None`` if the deadline is hit mid-evaluation.
    """
    n = len(order)
    n_pos = n + 1
    ops_by_uid = {op.uid: op for op in inst.ops}
    insert_op = ops_by_uid[uid]
    costs = [float("inf")] * n_pos

    prefix_gi = _empty_gate_intervals(inst.G)
    prefix_cost = 0.0

    if screen_top <= 0:
        for pos in range(n_pos):
            if pos % 8 == 0 and _deadline_hit(deadline):
                return None
            trial_gi = _copy_gate_intervals(prefix_gi)
            t, g = _place_op(insert_op, trial_gi, inst.G, inst.T)
            bisect.insort(trial_gi[g], (t, t + insert_op.p))
            costs[pos] = _decode_suffix_cost(
                inst, ops_by_uid, order[pos:], trial_gi, prefix_cost + insert_op.w * t,
            )
            if pos < n:
                op = ops_by_uid[order[pos]]
                t2, g2 = _place_op(op, prefix_gi, inst.G, inst.T)
                bisect.insort(prefix_gi[g2], (t2, t2 + op.p))
                prefix_cost += op.w * t2
        return costs

    # Screened mode: collect cheap proxies, exact-decode top screen_top.
    proxies: list[tuple[float, int]] = []
    prefix_snapshots: list[
        tuple[float, dict[int, list[tuple[int, int]]]]
    ] = []

    for pos in range(n_pos):
        if pos % 8 == 0 and _deadline_hit(deadline):
            return None
        prefix_snapshots.append((prefix_cost, _copy_gate_intervals(prefix_gi)))
        trial_gi = _copy_gate_intervals(prefix_gi)
        t, g = _place_op(insert_op, trial_gi, inst.G, inst.T)
        proxies.append((insert_op.w * t, pos))
        if pos < n:
            op = ops_by_uid[order[pos]]
            t2, g2 = _place_op(op, prefix_gi, inst.G, inst.T)
            bisect.insort(prefix_gi[g2], (t2, t2 + op.p))
            prefix_cost += op.w * t2

    proxies.sort()
    for _, pos in proxies[: min(screen_top, len(proxies))]:
        if _deadline_hit(deadline):
            return None
        pcost, pgi = prefix_snapshots[pos]
        trial_gi = _copy_gate_intervals(pgi)
        t, g = _place_op(insert_op, trial_gi, inst.G, inst.T)
        bisect.insort(trial_gi[g], (t, t + insert_op.p))
        costs[pos] = _decode_suffix_cost(
            inst, ops_by_uid, order[pos:], trial_gi, pcost + insert_op.w * t,
        )
    return costs


def _decode_starts_gates(
    inst: Instance, order: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    ops_by_uid = {op.uid: op for op in inst.ops}
    gate_intervals = _empty_gate_intervals(inst.G)
    starts: dict[int, int] = {}
    gates: dict[int, int] = {}
    for uid in order:
        op = ops_by_uid[uid]
        t, g = _place_op(op, gate_intervals, inst.G, inst.T)
        starts[uid] = t
        gates[uid] = g
        bisect.insort(gate_intervals[g], (t, t + op.p))
    return starts, gates


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
    temp_starts, _ = _decode_starts_gates(inst, order)

    cost_list = [(ops_by_uid[uid].w * temp_starts[uid], uid) for uid in order]
    cost_list.sort(key=lambda x: -x[0])

    remaining_order = list(order)
    removed: list[int] = []
    for _ in range(min(q, len(cost_list))):
        n = len(cost_list)
        if n == 0:
            break
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
    temp_starts, temp_gates = _decode_starts_gates(inst, order)

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


def _block_removal(
    order: list[int], q: int, rng: random.Random, inst: Instance,
    **_kwargs,
) -> tuple[list[int], list[int]]:
    """Remove a contiguous start-time block of ops on one gate."""
    if not order:
        return [], []

    temp_starts, temp_gates = _decode_starts_gates(inst, order)
    by_gate: dict[int, list[int]] = {}
    for uid in order:
        by_gate.setdefault(temp_gates[uid], []).append(uid)
    for g in by_gate:
        by_gate[g].sort(key=lambda u: (temp_starts[u], u))

    eligible = [g for g, uids in by_gate.items() if len(uids) >= 2]
    if not eligible:
        # Fall back to random when no gate has a block of size >= 2
        return _random_remove(order, q, rng, inst)

    gate = rng.choice(eligible)
    run = by_gate[gate]
    take = min(q, len(run))
    # Prefer a contiguous sub-run; if take == len(run) start must be 0
    start = rng.randint(0, len(run) - take) if take < len(run) else 0
    removed_set = set(run[start: start + take])
    remaining = [uid for uid in order if uid not in removed_set]
    removed = [uid for uid in order if uid in removed_set]
    return remaining, removed


def _history_removal(
    order: list[int], q: int, rng: random.Random, inst: Instance,
    best_start_seen: dict[int, int] | None = None,
    **_kwargs,
) -> tuple[list[int], list[int]]:
    """Remove ops farthest above their historical best start (weighted)."""
    del rng
    ops_by_uid = {op.uid: op for op in inst.ops}
    temp_starts, _ = _decode_starts_gates(inst, order)
    hist = best_start_seen or {}

    ranked = []
    for uid in order:
        best = hist.get(uid, temp_starts[uid])
        ranked.append((ops_by_uid[uid].w * (temp_starts[uid] - best), uid))
    ranked.sort(key=lambda x: -x[0])

    removed_set = {uid for _, uid in ranked[: min(q, len(ranked))]}
    remaining = [uid for uid in order if uid not in removed_set]
    removed = [uid for uid in order if uid in removed_set]
    return remaining, removed


# ---------------------------------------------------------------------------
# Repair operators (Section 5.4)
# ---------------------------------------------------------------------------

def _greedy_repair(
    remaining: list[int], removed: list[int], rng: random.Random, inst: Instance,
    deadline: float | None = None,
    screen_top: int = 0,
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
        costs = _insertion_costs(
            inst, order, uid, deadline=deadline, screen_top=screen_top,
        )
        if costs is None:
            return _append_remaining_random(order, [uid] + pending, rng)
        best_pos = min(range(len(costs)), key=lambda i: (costs[i], i))
        order.insert(best_pos, uid)
    return order


def _regret_k_repair(
    remaining: list[int], removed: list[int], rng: random.Random,
    inst: Instance, k: int = 3, deadline: float | None = None,
    screen_top: int = 0, **_kwargs,
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
            costs = _insertion_costs(
                inst, order, uid, deadline=deadline, screen_top=screen_top,
            )
            if costs is None:
                return _append_remaining_random(order, bank, rng)

            sorted_pairs = sorted(
                ((c, pos) for pos, c in enumerate(costs) if c < float("inf")),
            )
            if not sorted_pairs:
                # All screened out / empty — treat as no regret
                continue
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
            # Fallback: append remaining randomly
            return _append_remaining_random(order, bank, rng)

        order.insert(best_pos, best_uid)
        bank.remove(best_uid)

    return order


def _noised_greedy_repair(
    remaining: list[int], removed: list[int], rng: random.Random, inst: Instance,
    deadline: float | None = None,
    screen_top: int = 0,
    noise_level: float = 0.02,
    **_kwargs,
) -> list[int]:
    """Greedy insertion with multiplicative noise on insertion costs."""
    rng.shuffle(removed)
    order = list(remaining)
    pending = list(removed)
    eta = max(0.0, float(noise_level))
    while pending:
        if _deadline_hit(deadline):
            return _append_remaining_random(order, pending, rng)
        uid = pending.pop(0)
        costs = _insertion_costs(
            inst, order, uid, deadline=deadline, screen_top=screen_top,
        )
        if costs is None:
            return _append_remaining_random(order, [uid] + pending, rng)
        noised = [
            (c * rng.uniform(1.0 - eta, 1.0 + eta) if c < float("inf") else c)
            for c in costs
        ]
        best_pos = min(range(len(noised)), key=lambda i: (noised[i], i))
        order.insert(best_pos, uid)
    return order


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
    "regret_k": 3,
    "d_wr": 3.0,
    "max_iterations": 25000,
    # Cap destroy size so one repair stays cheap on large K
    "q_cap": 25,
    "screen_top": 5,
    "noise_level": 0.02,
    "stagnation_iters": 2000,
    "reheat_factor": 0.5,
}

DESTROY_OPS = [
    _random_remove,
    _worst_remove,
    _related_removal,
    _block_removal,
    _history_removal,
]

REPAIR_OPS = [
    _greedy_repair,
    _regret_k_repair,
    _noised_greedy_repair,
]

# Per-operator kwargs drawn from the param dict / solve-loop state.
DESTROY_KWARG_KEYS: dict[Callable, tuple[str, ...]] = {
    _worst_remove: ("d_wr",),
    _history_removal: ("best_start_seen",),
}

REPAIR_KWARG_KEYS: dict[Callable, tuple[str, ...]] = {
    _regret_k_repair: ("regret_k",),
    _noised_greedy_repair: ("noise_level",),
}

# Map param-dict key → repair kwarg name when they differ
_REPAIR_PARAM_ALIAS = {"regret_k": "k"}

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


def _build_destroy_kwargs(
    op: Callable, p: dict, best_start_seen: dict[int, int],
) -> dict:
    keys = DESTROY_KWARG_KEYS.get(op, ())
    out: dict = {}
    for key in keys:
        if key == "best_start_seen":
            out[key] = best_start_seen
        elif key in p:
            out[key] = p[key]
    return out


def _build_repair_kwargs(
    op: Callable, p: dict, deadline: float | None,
) -> dict:
    out: dict = {
        "deadline": deadline,
        "screen_top": int(p.get("screen_top", 0)),
    }
    for key in REPAIR_KWARG_KEYS.get(op, ()):
        if key in p:
            out[_REPAIR_PARAM_ALIAS.get(key, key)] = p[key]
    return out


def _update_best_starts(
    best_start_seen: dict[int, int], starts: dict[int, int],
) -> None:
    for uid, t in starts.items():
        prev = best_start_seen.get(uid)
        if prev is None or t < prev:
            best_start_seen[uid] = t


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

        best_start_seen: dict[int, int] = dict(x_cur.starts)
        _update_best_starts(best_start_seen, x_cur.starts)

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
        q_cap = int(p.get("q_cap", 25))
        if q_cap > 0:
            q_min = min(q_min, q_cap)
            q_max = min(q_max, q_cap)
            q_max = max(q_min, q_max)

        max_iter = p["max_iterations"]
        stagnation_iters = int(p.get("stagnation_iters", 2000))
        reheat_factor = float(p.get("reheat_factor", 0.5))
        iterations = 0
        iters_since_best = 0
        reheat_count = 0

        for it in range(max_iter):
            if _deadline_hit(deadline):
                break

            iterations = it + 1

            if stagnation_iters > 0 and iters_since_best >= stagnation_iters:
                temp = temp0 * reheat_factor
                cur_order = list(best_order)
                obj_cur = obj_best
                iters_since_best = 0
                reheat_count += 1

            d_idx = rng.choices(range(n_destroy), weights=w_destroy, k=1)[0]
            r_idx = rng.choices(range(n_repair), weights=w_repair, k=1)[0]
            q = rng.randint(q_min, q_max)

            destroy_op = DESTROY_OPS[d_idx]
            destroy_kwargs = _build_destroy_kwargs(
                destroy_op, p, best_start_seen,
            )
            remaining, removed = destroy_op(
                cur_order, q, rng, inst, **destroy_kwargs
            )

            if _deadline_hit(deadline):
                break

            repair_op = REPAIR_OPS[r_idx]
            repair_kwargs = _build_repair_kwargs(repair_op, p, deadline)
            new_order = repair_op(
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
                accepted_starts, _ = _decode_starts_gates(inst, new_order)
                _update_best_starts(best_start_seen, accepted_starts)

                if obj_new < obj_best:
                    obj_best = obj_new
                    best_order = list(new_order)
                    iters_since_best = 0
                else:
                    iters_since_best += 1

                score_destroy[d_idx] += psi
                score_repair[r_idx] += psi
            else:
                iters_since_best += 1

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
            "reheat_count": reheat_count,
        }
        return sol
