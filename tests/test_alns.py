"""ALNS tests (T10.4)."""

from __future__ import annotations

import random

from src.instance_gen import gen_instance
from src.solvers.alns import (
    ALNS,
    _block_removal,
    _decode_starts_gates,
    _history_removal,
    _insertion_costs,
    _objective_from_order,
)
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


def test_insertion_costs_match_brute_force():
    inst = gen_instance(seed=3, M=3, N=3, G=2)
    order = [op.uid for op in inst.ops]
    insert_uid = order[-1]
    base = order[:-1]
    costs = _insertion_costs(inst, base, insert_uid, screen_top=0)
    assert costs is not None
    for pos, cost in enumerate(costs):
        trial = base[:pos] + [insert_uid] + base[pos:]
        assert abs(cost - _objective_from_order(inst, trial)) < 1e-9


def test_block_removal_partition_and_same_gate_run():
    inst = gen_instance(seed=5, M=4, N=4, G=2)
    order = [op.uid for op in sorted(inst.ops, key=lambda o: (o.r, o.p, o.uid))]
    rng = random.Random(0)
    remaining, removed = _block_removal(order, q=3, rng=rng, inst=inst)
    assert set(remaining) | set(removed) == set(order)
    assert set(remaining).isdisjoint(set(removed))
    assert len(removed) >= 1

    # If block removal succeeded (not random fallback with 1-op gates only),
    # removed ops share a gate and form a contiguous start-time run.
    starts, gates = _decode_starts_gates(inst, order)
    if len(removed) >= 2:
        g0 = gates[removed[0]]
        assert all(gates[u] == g0 for u in removed)
        run = sorted(
            (u for u in order if gates[u] == g0),
            key=lambda u: (starts[u], u),
        )
        idxs = [run.index(u) for u in removed]
        assert max(idxs) - min(idxs) + 1 == len(removed)


def test_history_removal_partition():
    inst = gen_instance(seed=6, M=3, N=3, G=2)
    order = [op.uid for op in inst.ops]
    starts, _ = _decode_starts_gates(inst, order)
    # Pretend history saw everyone starting earlier
    hist = {uid: max(1, t - 5) for uid, t in starts.items()}
    remaining, removed = _history_removal(
        order, q=2, rng=random.Random(1), inst=inst, best_start_seen=hist,
    )
    assert set(remaining) | set(removed) == set(order)
    assert set(remaining).isdisjoint(set(removed))
    assert len(removed) == 2


def test_alns_reheat_on_stagnation():
    inst = gen_instance(seed=7, M=3, N=3, G=2)
    sol = ALNS(
        params={
            "stagnation_iters": 5,
            "max_iterations": 40,
            "screen_top": 0,
        },
    ).solve(inst, time_limit_sec=2.0, seed=0)
    validate(inst, sol)
    assert (sol.meta or {}).get("reheat_count", 0) >= 1
