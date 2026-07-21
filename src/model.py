"""Core data structures for truck gate scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    uid: int  # 0..K-1 global id
    kind: str  # "delivery" | "pickup"
    local_id: int  # i or j within kind
    p: int  # processing time
    r: int  # release time
    w: float  # weight


@dataclass
class Instance:
    id: str
    T: int
    G: int
    ops: list[Operation]
    w1: float
    w2: float


@dataclass
class Solution:
    starts: dict[int, int]  # uid -> start
    gates: dict[int, int]  # uid -> gate
    proven_optimal: bool = False
    runtime_sec: float = 0.0
    meta: dict | None = None

    def objective(self, inst: Instance) -> float:
        """Weighted sum of start times."""
        return sum(op.w * self.starts[op.uid] for op in inst.ops)


class Infeasible(Exception):
    """Raised when an instance or placement cannot satisfy constraints."""


def objective(instance: Instance, solution: Solution) -> float:
    """Weighted sum of start times."""
    return sum(op.w * solution.starts[op.uid] for op in instance.ops)


def feasibility_precheck(inst: Instance) -> None:
    """Verify instance-level feasibility; raise Infeasible on violation."""
    if not inst.ops:
        return

    max_p = max(op.p for op in inst.ops)
    if inst.T <= max_p:
        raise Infeasible(f"T={inst.T} must be greater than max processing time {max_p}")

    total_work = sum(op.p for op in inst.ops)
    if total_work > inst.G * inst.T:
        raise Infeasible(
            f"Total work {total_work} exceeds capacity G*T={inst.G * inst.T}"
        )

    for op in inst.ops:
        if op.r + op.p - 1 > inst.T:
            raise Infeasible(
                f"Operation {op.uid} cannot fit: release {op.r}, p={op.p}, T={inst.T}"
            )
