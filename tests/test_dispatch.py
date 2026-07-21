"""Dispatcher tests (T10.5)."""

from __future__ import annotations

import json
from pathlib import Path

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
