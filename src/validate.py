"""Feasibility validation for solutions."""

from __future__ import annotations

from src.model import Instance, Solution


class ValidationError(Exception):
    """Raised when a solution violates scheduling constraints."""


def validate(instance: Instance, solution: Solution) -> None:
    """Raise ValidationError on any release, horizon, or gate-overlap violation."""
    ops = {op.uid: op for op in instance.ops}

    for uid, op in ops.items():
        if uid not in solution.starts or uid not in solution.gates:
            raise ValidationError(f"Missing assignment for operation {uid}")

        start = solution.starts[uid]
        gate = solution.gates[uid]

        if start < op.r:
            raise ValidationError(
                f"Operation {uid} starts at {start} before release {op.r}"
            )

        if start + op.p - 1 > instance.T:
            raise ValidationError(
                f"Operation {uid} exceeds horizon T={instance.T}"
            )

        if gate < 1 or gate > instance.G:
            raise ValidationError(
                f"Operation {uid} assigned invalid gate {gate}"
            )

    by_gate: dict[int, list[tuple[int, int, int]]] = {
        g: [] for g in range(1, instance.G + 1)
    }
    for uid, op in ops.items():
        start = solution.starts[uid]
        gate = solution.gates[uid]
        by_gate[gate].append((start, start + op.p, uid))

    for gate, intervals in by_gate.items():
        intervals.sort()
        for i in range(1, len(intervals)):
            prev_start, prev_end, prev_uid = intervals[i - 1]
            cur_start, _, cur_uid = intervals[i]
            if cur_start < prev_end:
                raise ValidationError(
                    f"Gate {gate}: operations {prev_uid} and {cur_uid} overlap"
                )
