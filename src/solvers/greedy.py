"""Tier 3 — Greedy ERD/SPT list scheduler."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from src.model import Infeasible, Instance, Operation, Solution, feasibility_precheck


def earliest_start(
    intervals: list[tuple[int, int]],
    r: int,
    p: int,
    T: int,
) -> int | None:
    """Smallest t >= r fitting [t, t+p) in a gap on one gate; else None."""
    latest_start = T - p + 1
    if latest_start < r:
        return None

    if not intervals:
        return r if r <= latest_start else None

    sorted_iv = sorted(intervals)

    # Gap before first interval
    gap_end = sorted_iv[0][0]
    t = max(r, 1)
    if t + p <= gap_end and t <= latest_start:
        return t

    # Gaps between consecutive intervals
    for i in range(len(sorted_iv) - 1):
        gap_start = sorted_iv[i][1]
        gap_end = sorted_iv[i + 1][0]
        t = max(r, gap_start)
        if t + p <= gap_end and t <= latest_start:
            return t

    # Gap after last interval
    gap_start = sorted_iv[-1][1]
    t = max(r, gap_start)
    if t + p <= T + 1 and t <= latest_start:
        return t

    return None


def _place_op(
    op: Operation,
    gate_intervals: dict[int, list[tuple[int, int]]],
    G: int,
    T: int,
) -> tuple[int, int]:
    best: tuple[int, int] | None = None
    for g in range(1, G + 1):
        t = earliest_start(gate_intervals[g], op.r, op.p, T)
        if t is not None and (best is None or (t, g) < best):
            best = (t, g)
    if best is None:
        raise Infeasible(f"Cannot place operation {op.uid}")
    return best


def _decode_order(
    inst: Instance,
    order: list[Operation],
) -> Solution:
    gate_intervals: dict[int, list[tuple[int, int]]] = {
        g: [] for g in range(1, inst.G + 1)
    }
    starts: dict[int, int] = {}
    gates: dict[int, int] = {}

    for op in order:
        t, g = _place_op(op, gate_intervals, inst.G, inst.T)
        starts[op.uid] = t
        gates[op.uid] = g
        gate_intervals[g].append((t, t + op.p))
        gate_intervals[g].sort()

    return Solution(starts=starts, gates=gates)


def greedy_erd_spt(inst: Instance) -> Solution:
    """Sort by (r asc, p asc, w desc, uid); place at min-start gate."""
    feasibility_precheck(inst)
    order = sorted(
        inst.ops,
        key=lambda op: (op.r, op.p, -op.w, op.uid),
    )
    return _decode_order(inst, order)


def greedy_erd(inst: Instance) -> Solution:
    """Sort by release time then uid."""
    feasibility_precheck(inst)
    order = sorted(inst.ops, key=lambda op: (op.r, op.uid))
    return _decode_order(inst, order)


def greedy_spt_ready(inst: Instance) -> Solution:
    """Event-driven: when a gate frees, pick shortest ready job."""
    feasibility_precheck(inst)
    gate_intervals: dict[int, list[tuple[int, int]]] = {
        g: [] for g in range(1, inst.G + 1)
    }
    gate_free_at: dict[int, int] = {g: 1 for g in range(1, inst.G + 1)}
    unscheduled = set(op.uid for op in inst.ops)
    ops_by_uid = {op.uid: op for op in inst.ops}
    starts: dict[int, int] = {}
    gates: dict[int, int] = {}

    ready_heap: list[tuple[int, int, int]] = []
    current_time = 1

    while unscheduled or ready_heap:
        for uid in list(unscheduled):
            op = ops_by_uid[uid]
            if op.r <= current_time:
                heapq.heappush(ready_heap, (op.p, op.uid, op.r))
                unscheduled.remove(uid)

        if ready_heap:
            free_gates = [g for g in range(1, inst.G + 1) if gate_free_at[g] <= current_time]
            free_gates.sort()
            for g in free_gates:
                if not ready_heap:
                    break
                _, uid, _ = heapq.heappop(ready_heap)
                op = ops_by_uid[uid]
                t = max(current_time, op.r)
                latest = inst.T - op.p + 1
                if t > latest:
                    raise Infeasible(f"Cannot place operation {uid}")
                starts[uid] = t
                gates[uid] = g
                finish = t + op.p
                gate_intervals[g].append((t, finish))
                gate_intervals[g].sort()
                gate_free_at[g] = finish
        else:
            if unscheduled:
                next_release = min(ops_by_uid[uid].r for uid in unscheduled)
                current_time = max(current_time + 1, next_release)
            else:
                break

        if ready_heap or unscheduled:
            next_events = []
            if unscheduled:
                next_events.append(min(ops_by_uid[uid].r for uid in unscheduled))
            next_events.extend(gate_free_at[g] for g in range(1, inst.G + 1))
            current_time = max(current_time, min(next_events))

    if len(starts) != len(inst.ops):
        raise Infeasible("Could not schedule all operations")

    return Solution(starts=starts, gates=gates)


@dataclass
class GreedyERDSPT:
    """Solver wrapper for greedy ERD/SPT."""

    name: str = "greedy"

    def solve(
        self,
        inst: Instance,
        *,
        time_limit_sec: float | None = None,
        seed: int | None = None,
        warm_start: Solution | None = None,
    ) -> Solution:
        del time_limit_sec, seed, warm_start
        return greedy_erd_spt(inst)
