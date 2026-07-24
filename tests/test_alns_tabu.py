"""Hybrid ALNS+Tabu solver tests."""

from __future__ import annotations

from src.instance_gen import gen_instance
from src.solvers.alns_tabu import HybridALNSTabu
from src.solvers.cpsat import CPSAT
from src.solvers.greedy import greedy_erd_spt
from src.validate import validate


def test_alns_tabu_never_worse_than_greedy():
    inst = gen_instance(seed=1, M=4, N=4, G=2)
    g = greedy_erd_spt(inst)
    sol = HybridALNSTabu().solve(inst, time_limit_sec=1.0, seed=0, warm_start=g)
    validate(inst, sol)
    assert sol.objective(inst) <= g.objective(inst) + 1e-9


def test_alns_tabu_reproducible_seed():
    inst = gen_instance(seed=2, M=3, N=3, G=2)
    a = HybridALNSTabu().solve(inst, time_limit_sec=0.5, seed=7)
    b = HybridALNSTabu().solve(inst, time_limit_sec=0.5, seed=7)
    assert a.starts == b.starts
    assert a.gates == b.gates


def test_alns_tabu_feasible_many_seeds():
    for seed in range(5):
        inst = gen_instance(seed=seed, M=3, N=3, G=2)
        sol = HybridALNSTabu().solve(inst, time_limit_sec=0.3, seed=seed)
        validate(inst, sol)


def test_crosscheck_alns_tabu_ge_cpsat():
    inst = gen_instance(seed=0, M=2, N=2, G=1)
    g = greedy_erd_spt(inst)
    c = CPSAT(num_workers=2).solve(inst, time_limit_sec=10.0, warm_start=g)
    hy = HybridALNSTabu().solve(inst, time_limit_sec=1.0, seed=0, warm_start=g)
    if c.proven_optimal:
        assert hy.objective(inst) >= c.objective(inst) - 1e-9


def test_alns_tabu_respects_time_limit():
    inst = gen_instance(seed=3, M=10, N=10, G=2)
    limit = 0.4
    sol = HybridALNSTabu().solve(inst, time_limit_sec=limit, seed=0)
    validate(inst, sol)
    assert sol.runtime_sec < limit + 1.5
    assert sol.meta is not None
    assert "local_searches" in sol.meta
    assert "iterations" in sol.meta
    assert "local_search_time_sec" in sol.meta
    assert "local_search_budget_sec" in sol.meta


def test_alns_tabu_polish_budget_gate():
    """Tiny polish frac must bound Tabu time and leave ALNS iterations."""
    inst = gen_instance(seed=0, M=30, N=30, G=2)
    limit = 2.0
    frac = 0.01
    sol = HybridALNSTabu(
        params={
            "local_search_budget_frac": frac,
            "local_search_rate": 1.0,
            "local_tabu_iters": 80,
            "local_neighborhood_size": 40,
            "q_cap": 0,
        }
    ).solve(inst, time_limit_sec=limit, seed=0)
    validate(inst, sol)
    assert sol.meta is not None
    budget = sol.meta["local_search_budget_sec"]
    assert budget is not None
    assert abs(budget - frac * limit) < 1e-9
    # Allow a little overshoot from the in-flight polish call
    assert sol.meta["local_search_time_sec"] <= budget + 0.75
    assert sol.meta["iterations"] >= 20
