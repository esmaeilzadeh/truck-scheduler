"""CP-SAT small-instance tests (T10.3)."""

from __future__ import annotations

import itertools

import pytest

from src.model import Instance, Operation
from src.solvers.cpsat import CPSAT
from src.solvers.greedy import greedy_erd_spt
from src.validate import validate


def _small_inst() -> Instance:
    ops = [
        Operation(uid=0, kind="delivery", local_id=0, p=2, r=1, w=1.0),
        Operation(uid=1, kind="pickup", local_id=0, p=2, r=1, w=1.0),
        Operation(uid=2, kind="delivery", local_id=1, p=1, r=2, w=1.0),
    ]
    return Instance(id="small3", T=12, G=1, ops=ops, w1=1.0, w2=1.0)


def _brute_force_optimum(inst: Instance) -> float:
    """Enumerate permutations of placement order via greedy decode."""
    from src.solvers.alns import decode

    best = float("inf")
    uids = [op.uid for op in inst.ops]
    for perm in itertools.permutations(uids):
        sol = decode(inst, list(perm))
        validate(inst, sol)
        best = min(best, sol.objective(inst))
    return best


def test_cpsat_proven_optimal_and_matches_brute():
    inst = _small_inst()
    opt = _brute_force_optimum(inst)
    sol = CPSAT(num_workers=2).solve(inst, time_limit_sec=10.0, warm_start=greedy_erd_spt(inst))
    validate(inst, sol)
    assert sol.proven_optimal is True
    assert sol.objective(inst) == pytest.approx(opt)


def test_crosscheck_greedy_ge_cpsat():
    inst = _small_inst()
    g = greedy_erd_spt(inst)
    c = CPSAT(num_workers=2).solve(inst, time_limit_sec=10.0, warm_start=g)
    assert g.objective(inst) >= c.objective(inst) - 1e-9


def test_cpsat_respects_fractional_weights():
    """CP-SAT must not collapse w=1.3/0.7 to identical integer weights."""
    from src.instance_gen import gen_instance

    inst = gen_instance(seed=1, M=5, N=5, G=2, w1=1.3, w2=0.7)
    g = greedy_erd_spt(inst)
    c = CPSAT(num_workers=2).solve(inst, time_limit_sec=5.0, warm_start=g)
    validate(inst, c)
    assert c.objective(inst) <= g.objective(inst) + 1e-6


def test_cpsat_stop_event():
    """stop_event should interrupt a long CP-SAT search."""
    import threading
    import time

    from src.instance_gen import gen_instance
    from src.model import Infeasible

    inst = gen_instance(seed=0, M=12, N=12, G=2)
    stop = threading.Event()
    solver = CPSAT(num_workers=2)

    def _stop_soon() -> None:
        time.sleep(0.3)
        stop.set()
        solver.request_stop()

    threading.Thread(target=_stop_soon, daemon=True).start()
    t0 = time.perf_counter()
    try:
        sol = solver.solve(inst, time_limit_sec=60.0, stop_event=stop)
        # If an incumbent existed, solve returns FEASIBLE with stopped meta
        assert sol.runtime_sec < 10.0
        assert sol.meta is not None and sol.meta.get("stopped") is True
        assert sol.proven_optimal is False
    except Infeasible:
        # Stopped before any incumbent — also acceptable
        assert time.perf_counter() - t0 < 10.0
