"""Tests for instance/solution JSON serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.io_utils import instance_to_dict, read_instance, solution_to_dict, write_solution
from src.model import Instance, Operation, Solution
from src.solvers.greedy import greedy_erd_spt
from src.validate import validate


def _hand_instance() -> Instance:
    ops = [
        Operation(uid=0, kind="delivery", local_id=0, p=2, r=1, w=1.0),
        Operation(uid=1, kind="pickup", local_id=0, p=3, r=5, w=2.0),
    ]
    return Instance(id="hand_io", T=20, G=2, ops=ops, w1=1.0, w2=2.0)


def test_instance_to_dict_round_trip(tmp_path: Path):
    inst = _hand_instance()
    payload = instance_to_dict(inst)

    assert set(payload) == {"id", "T", "G", "w1", "w2", "deliveries", "pickups"}
    assert payload["deliveries"] == [{"id": 0, "p": 2, "rdt": 1}]
    assert payload["pickups"] == [{"id": 0, "p": 3, "release": 5}]

    path = tmp_path / f"{inst.id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = read_instance(path)

    assert loaded.id == inst.id
    assert loaded.T == inst.T
    assert loaded.G == inst.G
    assert loaded.w1 == inst.w1
    assert loaded.w2 == inst.w2
    assert len(loaded.ops) == len(inst.ops)
    for a, b in zip(loaded.ops, inst.ops):
        assert (a.kind, a.local_id, a.p, a.r, a.w) == (b.kind, b.local_id, b.p, b.r, b.w)


def test_solution_to_dict_matches_write_solution(tmp_path: Path):
    inst = _hand_instance()
    sol = greedy_erd_spt(inst)
    validate(inst, sol)

    payload = solution_to_dict(
        instance_id=inst.id,
        solver="greedy",
        solution=sol,
        inst=inst,
        proven_optimal=sol.proven_optimal,
        runtime_sec=sol.runtime_sec,
        meta=sol.meta,
    )

    expected_keys = {
        "instance_id",
        "solver",
        "objective",
        "is_optimal",
        "proven_optimal",
        "runtime_sec",
        "assignments",
        "meta",
    }
    assert set(payload) == expected_keys
    assert payload["instance_id"] == inst.id
    assert payload["solver"] == "greedy"
    assert payload["objective"] == sol.objective(inst)
    assert len(payload["assignments"]) == len(inst.ops)

    path = tmp_path / f"{inst.id}_solution.json"
    write_solution(
        path,
        instance_id=inst.id,
        solver="greedy",
        solution=sol,
        inst=inst,
        proven_optimal=sol.proven_optimal,
        runtime_sec=sol.runtime_sec,
        meta=sol.meta,
    )
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == payload
