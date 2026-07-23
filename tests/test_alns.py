"""ALNS tests (T10.4)."""

from __future__ import annotations

from src.instance_gen import gen_instance
from src.solvers.alns import ALNS
from src.solvers.cpsat import CPSAT
from src.solvers.greedy import greedy_erd_spt
from src.validate import validate


def test_alns_never_worse_than_greedy():
    inst = gen_instance(seed=1, M=4, N=4, G=2)
    g = greedy_erd_spt(inst)
    sol = ALNS().solve(inst, time_limit_sec=1.0, seed=0, warm_start=g)
    validate(inst, sol)
    assert sol.objective(inst) <= g.objective(inst) + 1e-9


def test_alns_reproducible_seed():
    inst = gen_instance(seed=2, M=3, N=3, G=2)
    a = ALNS().solve(inst, time_limit_sec=0.5, seed=7)
    b = ALNS().solve(inst, time_limit_sec=0.5, seed=7)
    assert a.starts == b.starts
    assert a.gates == b.gates
    assert a.objective(inst) == b.objective(inst)


def test_alns_feasibility_many_instances():
    alns = ALNS()
    for seed in range(8):
        inst = gen_instance(seed=seed, M=3, N=2, G=2)
        sol = alns.solve(inst, time_limit_sec=0.3, seed=seed)
        validate(inst, sol)


def test_crosscheck_alns_ge_cpsat():
    inst = gen_instance(seed=0, M=2, N=2, G=1)
    g = greedy_erd_spt(inst)
    c = CPSAT(num_workers=2).solve(inst, time_limit_sec=10.0, warm_start=g)
    a = ALNS().solve(inst, time_limit_sec=1.0, seed=0, warm_start=g)
    if c.proven_optimal:
        assert a.objective(inst) >= c.objective(inst) - 1e-9
        assert g.objective(inst) >= c.objective(inst) - 1e-9


def test_alns_respects_time_limit_on_large_instance():
    """Repair used to overrun the budget; keep instance modest for fast CI."""
    inst = gen_instance(seed=42, M=8, N=7, G=2)
    limit = 1.0
    sol = ALNS().solve(inst, time_limit_sec=limit, seed=0)
    validate(inst, sol)
    assert sol.runtime_sec <= limit + 0.5
    assert (sol.meta or {}).get("iterations", 0) >= 1


def test_alns_schedule_objective_consistency():
    """Returned solution validates and objective matches recomputation."""
    inst = gen_instance(seed=11, M=5, N=5, G=2)
    sol = ALNS().solve(inst, time_limit_sec=1.0, seed=0)
    validate(inst, sol)
    recomputed = sum(
        op.w * sol.starts[op.uid] for op in inst.ops
    )
    assert abs(sol.objective(inst) - recomputed) < 1e-9


def test_alns_throughput_schedule_based():
    """Schedule-based repair should far exceed the old ~136 iters/s."""
    inst = gen_instance(seed=42, M=8, N=7, G=2)
    sol = ALNS().solve(inst, time_limit_sec=1.0, seed=0)
    validate(inst, sol)
    iters = (sol.meta or {}).get("iterations", 0)
    # Generous vs ~136 with position-scan repair; allow CI load variance
    assert iters >= 500, f"expected >= 500 iterations, got {iters}"
