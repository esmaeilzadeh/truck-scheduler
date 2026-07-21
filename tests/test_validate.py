"""Validator and feasibility tests (T10.1)."""

from __future__ import annotations

import pytest

from src.model import Infeasible, Instance, Operation, Solution, feasibility_precheck
from src.validate import validate


def _tiny_inst() -> Instance:
    ops = [
        Operation(uid=0, kind="delivery", local_id=0, p=2, r=1, w=1.0),
        Operation(uid=1, kind="pickup", local_id=0, p=2, r=1, w=1.0),
    ]
    return Instance(id="tiny", T=10, G=1, ops=ops, w1=1.0, w2=1.0)


def test_validate_feasible():
    inst = _tiny_inst()
    sol = Solution(starts={0: 1, 1: 3}, gates={0: 1, 1: 1})
    validate(inst, sol)


def test_validate_overlap():
    inst = _tiny_inst()
    sol = Solution(starts={0: 1, 1: 2}, gates={0: 1, 1: 1})
    with pytest.raises(Exception):
        validate(inst, sol)


def test_validate_release():
    ops = [
        Operation(uid=0, kind="delivery", local_id=0, p=2, r=1, w=1.0),
        Operation(uid=1, kind="pickup", local_id=0, p=2, r=5, w=1.0),
    ]
    inst = Instance(id="release", T=10, G=2, ops=ops, w1=1.0, w2=1.0)
    sol = Solution(starts={0: 1, 1: 3}, gates={0: 1, 1: 2})  # start 3 < release 5
    with pytest.raises(Exception):
        validate(inst, sol)


def test_validate_horizon():
    inst = _tiny_inst()
    sol = Solution(starts={0: 1, 1: 10}, gates={0: 1, 1: 1})  # 10+2-1=11 > T=10
    with pytest.raises(Exception):
        validate(inst, sol)


def test_precheck_infeasible_workload():
    ops = [
        Operation(uid=0, kind="delivery", local_id=0, p=5, r=1, w=1.0),
        Operation(uid=1, kind="pickup", local_id=0, p=5, r=1, w=1.0),
    ]
    inst = Instance(id="bad", T=5, G=1, ops=ops, w1=1.0, w2=1.0)
    with pytest.raises(Infeasible):
        feasibility_precheck(inst)
