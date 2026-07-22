"""Tabu Search solver tests."""

from __future__ import annotations

from src.instance_gen import gen_instance
from src.solvers.cpsat import CPSAT
from src.solvers.greedy import greedy_erd_spt
from src.solvers.tabu import TabuSearch
from src.validate import validate


def test_tabu_never_worse_than_greedy():
    inst = gen_instance(seed=1, M=4, N=4, G=2)
    g = greedy_erd_spt(inst)
    sol = TabuSearch().solve(inst, time_limit_sec=1.0, seed=0, warm_start=g)
    validate(inst, sol)
    assert sol.objective(inst) <= g.objective(inst) + 1e-9


def test_tabu_reproducible_seed():
    inst = gen_instance(seed=2, M=3, N=3, G=2)
    a = TabuSearch().solve(inst, time_limit_sec=0.5, seed=7)
    b = TabuSearch().solve(inst, time_limit_sec=0.5, seed=7)
    assert a.starts == b.starts
    assert a.gates == b.gates


def test_tabu_feasible_many_seeds():
    for seed in range(5):
        inst = gen_instance(seed=seed, M=3, N=3, G=2)
        sol = TabuSearch().solve(inst, time_limit_sec=0.3, seed=seed)
        validate(inst, sol)


def test_crosscheck_tabu_ge_cpsat():
    inst = gen_instance(seed=0, M=2, N=2, G=1)
    g = greedy_erd_spt(inst)
    c = CPSAT(num_workers=2).solve(inst, time_limit_sec=10.0, warm_start=g)
    t = TabuSearch().solve(inst, time_limit_sec=1.0, seed=0, warm_start=g)
    assert t.objective(inst) >= c.objective(inst) - 1e-9


def test_tabu_respects_time_limit():
    inst = gen_instance(seed=3, M=10, N=10, G=2)
    limit = 0.4
    sol = TabuSearch().solve(inst, time_limit_sec=limit, seed=0)
    validate(inst, sol)
    assert sol.runtime_sec < limit + 1.5
