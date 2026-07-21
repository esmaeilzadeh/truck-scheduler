"""JSON I/O for instances and solutions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.model import Instance, Operation, Solution


def _build_operations(data: dict[str, Any], w1: float, w2: float) -> list[Operation]:
    ops: list[Operation] = []
    uid = 0

    for d in data.get("deliveries", []):
        ops.append(
            Operation(
                uid=uid,
                kind="delivery",
                local_id=d["id"],
                p=d["p"],
                r=d.get("rdt", 1),
                w=w1,
            )
        )
        uid += 1

    for p in data.get("pickups", []):
        ops.append(
            Operation(
                uid=uid,
                kind="pickup",
                local_id=p["id"],
                p=p["p"],
                r=p["release"],
                w=w2,
            )
        )
        uid += 1

    return ops


def instance_to_dict(inst: Instance) -> dict[str, Any]:
    """Serialize an Instance to SPEC §2.1 JSON shape."""
    return {
        "id": inst.id,
        "T": inst.T,
        "G": inst.G,
        "w1": inst.w1,
        "w2": inst.w2,
        "deliveries": [
            {"id": op.local_id, "p": op.p, "rdt": op.r}
            for op in inst.ops
            if op.kind == "delivery"
        ],
        "pickups": [
            {"id": op.local_id, "p": op.p, "release": op.r}
            for op in inst.ops
            if op.kind == "pickup"
        ],
    }


def solution_to_dict(
    *,
    instance_id: str,
    solver: str,
    solution: Solution,
    inst: Instance,
    proven_optimal: bool = False,
    runtime_sec: float = 0.0,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a Solution to SPEC §2.2 JSON shape."""
    assignments = []
    for op in inst.ops:
        assignments.append(
            {
                "op_type": op.kind,
                "op_id": op.local_id,
                "start": solution.starts[op.uid],
                "gate": solution.gates[op.uid],
                "p": op.p,
                "weight": op.w,
            }
        )

    return {
        "instance_id": instance_id,
        "solver": solver,
        "objective": solution.objective(inst),
        "is_optimal": proven_optimal,
        "proven_optimal": proven_optimal,
        "runtime_sec": runtime_sec,
        "assignments": assignments,
        "meta": meta or {},
    }


def read_instance(path: str | Path) -> Instance:
    """Read an instance JSON file into an Instance."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    w1 = float(data.get("w1", 1.0))
    w2 = float(data.get("w2", 1.0))

    return Instance(
        id=data["id"],
        T=data["T"],
        G=data["G"],
        ops=_build_operations(data, w1, w2),
        w1=w1,
        w2=w2,
    )


def write_solution(
    path: str | Path,
    *,
    instance_id: str,
    solver: str,
    solution: Solution,
    inst: Instance,
    proven_optimal: bool = False,
    runtime_sec: float = 0.0,
    meta: dict[str, Any] | None = None,
) -> None:
    """Write a solution to JSON."""
    payload = solution_to_dict(
        instance_id=instance_id,
        solver=solver,
        solution=solution,
        inst=inst,
        proven_optimal=proven_optimal,
        runtime_sec=runtime_sec,
        meta=meta,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
