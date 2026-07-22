"""Dispatcher tests (T10.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dispatch import solve as dispatch_solve
from src.instance_gen import gen_instance
from src.solvers.greedy import greedy_erd_spt
from src.validate import validate


def test_dispatch_picks_cpsat_for_small_k(tmp_path: Path):
    policy = {
        "budget_sec": 5.0,
        "threshold_K": 20,
        "T_cap": None,
        "safety_margin_steps": 1,
    }
    policy_path = tmp_path / "switch_policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    inst = gen_instance(seed=0, M=3, N=3, G=2)  # K=6 <= 20
    sol, tier = dispatch_solve(
        inst,
        policy_path=policy_path,
        params_path=None,
        exact_time_limit=5.0,
        alns_time_limit=2.0,
        seed=0,
    )
    validate(inst, sol)
    assert tier in ("cpsat", "alns_fallback")
    g = greedy_erd_spt(inst)
    assert sol.objective(inst) <= g.objective(inst) + 1e-9


def test_dispatch_picks_alns_for_large_k(tmp_path: Path):
    policy = {
        "budget_sec": 2.0,
        "threshold_K": 5,
        "T_cap": None,
        "safety_margin_steps": 1,
    }
    policy_path = tmp_path / "switch_policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    inst = gen_instance(seed=1, M=5, N=5, G=2)  # K=10 > 5
    sol, tier = dispatch_solve(
        inst,
        policy_path=policy_path,
        params_path=None,
        exact_time_limit=2.0,
        alns_time_limit=1.0,
        seed=0,
    )
    validate(inst, sol)
    assert tier == "alns"
    g = greedy_erd_spt(inst)
    assert sol.objective(inst) <= g.objective(inst) + 1e-9


def test_dispatch_fallback_configs():
    """Missing config files → SPEC §13 defaults still work."""
    inst = gen_instance(seed=3, M=2, N=2, G=1)
    sol, tier = dispatch_solve(
        inst,
        policy_path="/nonexistent/switch_policy.json",
        params_path="/nonexistent/alns_params.json",
        exact_time_limit=3.0,
        alns_time_limit=1.0,
        seed=1,
    )
    validate(inst, sol)
    assert tier in ("cpsat", "alns", "alns_fallback")


def test_force_tier_alns_overrides_policy(tmp_path: Path):
    """Small-K instance would pick CP-SAT under policy; force_tier=alns wins."""
    policy = {
        "budget_sec": 5.0,
        "threshold_K": 20,
        "T_cap": None,
        "safety_margin_steps": 1,
    }
    policy_path = tmp_path / "switch_policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    inst = gen_instance(seed=0, M=3, N=3, G=2)  # K=6 <= 20
    sol, tier = dispatch_solve(
        inst,
        policy_path=policy_path,
        params_path=None,
        exact_time_limit=5.0,
        alns_time_limit=1.0,
        seed=0,
        force_tier="alns",
    )
    validate(inst, sol)
    assert tier == "alns"


def test_force_tier_cpsat_no_fallback(tmp_path: Path):
    """Forced CP-SAT never returns alns_fallback, even if not proven optimal."""
    policy = {
        "budget_sec": 0.01,
        "cpsat_time_limit_sec": 2.0,
        "threshold_K": 20,
        "T_cap": None,
        "safety_margin_steps": 1,
    }
    policy_path = tmp_path / "switch_policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    # Small instance so CP-SAT finds a feasible incumbent quickly under a short cap
    inst = gen_instance(seed=2, M=3, N=2, G=2)
    sol, tier = dispatch_solve(
        inst,
        policy_path=policy_path,
        params_path=None,
        exact_time_limit=2.0,
        alns_time_limit=1.0,
        seed=0,
        force_tier="cpsat",
    )
    validate(inst, sol)
    assert tier == "cpsat"
    assert sol.runtime_sec > 0.0


def test_force_tier_cpsat_not_silently_replaced_by_greedy(tmp_path: Path):
    """Forced CP-SAT must return the CP-SAT incumbent (not greedy via best_of)."""
    policy = {
        "budget_sec": 1.0,
        "cpsat_time_limit_sec": 5.0,
        "threshold_K": 20,
        "T_cap": None,
        "safety_margin_steps": 1,
    }
    policy_path = tmp_path / "switch_policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    # Fractional weights previously made CP-SAT optimize the wrong obj, then
    # best_of swapped in greedy with runtime≈0 while still labeling tier=cpsat.
    inst = gen_instance(seed=1, M=5, N=5, G=2, w1=1.3, w2=0.7)
    g = greedy_erd_spt(inst)
    sol, tier = dispatch_solve(
        inst,
        policy_path=policy_path,
        params_path=None,
        exact_time_limit=5.0,
        force_tier="cpsat",
    )
    validate(inst, sol)
    assert tier == "cpsat"
    assert sol.objective(inst) <= g.objective(inst) + 1e-6
    assert sol.runtime_sec >= 0.001
    # meta marks a real CP-SAT solve (not a bare greedy Solution)
    assert sol.meta is not None
    assert "cpsat_status" in sol.meta


def test_cpsat_time_limit_from_config(tmp_path: Path):
    from src.dispatch import cpsat_time_limit_from_policy, load_switch_policy

    policy_path = tmp_path / "switch_policy.json"
    policy_path.write_text(
        json.dumps({"budget_sec": 3.0, "cpsat_time_limit_sec": 120.0}),
        encoding="utf-8",
    )
    policy = load_switch_policy(policy_path)
    assert cpsat_time_limit_from_policy(policy) == 120.0


def test_force_tier_greedy():
    inst = gen_instance(seed=4, M=3, N=3, G=2)
    sol, tier = dispatch_solve(
        inst,
        policy_path=None,
        params_path=None,
        force_tier="greedy",
    )
    validate(inst, sol)
    assert tier == "greedy"
    g = greedy_erd_spt(inst)
    assert sol.objective(inst) == pytest.approx(g.objective(inst))


def test_force_tier_tabu():
    inst = gen_instance(seed=6, M=3, N=3, G=2)
    sol, tier = dispatch_solve(
        inst,
        policy_path=None,
        params_path=None,
        alns_time_limit=0.5,
        force_tier="tabu",
    )
    validate(inst, sol)
    assert tier == "tabu"


def test_force_tier_ga():
    inst = gen_instance(seed=7, M=3, N=3, G=2)
    sol, tier = dispatch_solve(
        inst,
        policy_path=None,
        params_path=None,
        alns_time_limit=0.5,
        force_tier="ga",
    )
    validate(inst, sol)
    assert tier == "ga"


def test_force_tier_ga_tabu():
    inst = gen_instance(seed=8, M=3, N=3, G=2)
    sol, tier = dispatch_solve(
        inst,
        policy_path=None,
        params_path=None,
        alns_time_limit=0.5,
        force_tier="ga_tabu",
    )
    validate(inst, sol)
    assert tier == "ga_tabu"


def test_force_tier_invalid():
    inst = gen_instance(seed=5, M=2, N=2, G=1)
    with pytest.raises(ValueError, match="force_tier"):
        dispatch_solve(inst, force_tier="bogus")
