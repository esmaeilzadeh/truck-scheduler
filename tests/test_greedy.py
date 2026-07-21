"""Greedy solver tests (T10.2)."""

from __future__ import annotations

from src.model import Instance, Operation, Solution
from src.solvers.greedy import GreedyERDSPT, greedy_erd_spt
from src.validate import validate


def _hand_instance() -> Instance:
    """Two ops, one gate: ERD/SPT places both sequentially from t=1."""
    ops = [
        Operation(uid=0, kind="delivery", local_id=0, p=2, r=1, w=1.0),
        Operation(uid=1, kind="pickup", local_id=0, p=3, r=1, w=1.0),
    ]
    return Instance(id="hand", T=20, G=1, ops=ops, w1=1.0, w2=1.0)


def test_greedy_known_schedule():
    inst = _hand_instance()
    sol = greedy_erd_spt(inst)
    validate(inst, sol)
    # SPT among equal release: op0 (p=2) then op1 (p=3)
    assert sol.starts[0] == 1
    assert sol.starts[1] == 3
    assert sol.gates[0] == 1
    assert sol.gates[1] == 1
    assert sol.objective(inst) == 1 * 1 + 1 * 3


def test_greedy_determinism():
    inst = _hand_instance()
    a = greedy_erd_spt(inst)
    b = greedy_erd_spt(inst)
    assert a.starts == b.starts
    assert a.gates == b.gates


def test_greedy_solver_wrapper_feasible():
    inst = _hand_instance()
    sol = GreedyERDSPT().solve(inst)
    validate(inst, sol)
    assert isinstance(sol, Solution)
