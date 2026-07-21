"""Runtime dispatcher — tier selection at production time (Section 8)."""

from __future__ import annotations

import json
from pathlib import Path

from src.model import Instance, Solution, feasibility_precheck
from src.solvers.greedy import GreedyERDSPT
from src.validate import validate

_DEFAULT_SWITCH_POLICY = {
    "budget_sec": 5.0,
    "threshold_K": 20,
    "T_cap": None,
    "safety_margin_steps": 1,
}

_DEFAULT_ALNS_PARAMS = {
    "rho_min": 0.10,
    "rho_max": 0.40,
    "lambda": 0.10,
    "segment_length": 100,
    "sigma1": 33,
    "sigma2": 9,
    "sigma3": 13,
    "cooling": 0.99975,
    "start_temp_ctrl": 0.05,
    "regret_k": 3,
    "d_wr": 3.0,
    "max_iterations": 25000,
}


def _load_json(path: str | Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _best_of(sol: Solution, warm: Solution, inst: Instance) -> Solution:
    """Return the solution with lower objective."""
    obj_sol = sol.objective(inst)
    obj_warm = warm.objective(inst)
    return sol if obj_sol <= obj_warm else warm


def solve(
    inst: Instance,
    policy_path: str | Path | None = "config/switch_policy.json",
    params_path: str | Path | None = "config/alns_params.json",
    exact_time_limit: float | None = None,
    alns_time_limit: float | None = None,
    seed: int = 0,
) -> tuple[Solution, str]:
    """Run the dispatcher. Returns (solution, tier_used).

    Parameters
    ----------
    inst : Instance
        The instance to solve.
    policy_path : path or None
        Path to switch_policy.json. None or missing → fallback defaults.
    params_path : path or None
        Path to alns_params.json. None or missing → fallback defaults.
    exact_time_limit : float or None
        Time limit for CP-SAT. None → use policy budget_sec.
    alns_time_limit : float or None
        Time limit for ALNS. None → use policy budget_sec.
    seed : int
        Random seed for ALNS.
    """
    feasibility_precheck(inst)

    # Tier 3: greedy warm start (always)
    greedy = GreedyERDSPT()
    warm = greedy.solve(inst)

    # Load configs with safe fallbacks
    policy = _load_json(policy_path) or _DEFAULT_SWITCH_POLICY
    params = _load_json(params_path) or _DEFAULT_ALNS_PARAMS

    threshold_K = policy.get("threshold_K", 20)
    T_cap = policy.get("T_cap", None)
    budget = policy.get("budget_sec", 5.0)

    if exact_time_limit is None:
        exact_time_limit = budget
    if alns_time_limit is None:
        alns_time_limit = budget

    K = len(inst.ops)
    use_exact = (K <= threshold_K) and (T_cap is None or inst.T <= T_cap)

    tier = ""
    if use_exact:
        from src.solvers.cpsat import CPSAT

        cpsat = CPSAT()
        sol = cpsat.solve(
            inst,
            time_limit_sec=exact_time_limit,
            warm_start=warm,
        )
        # Guard: if not proven optimal, fall back to ALNS
        if exact_time_limit is not None and not sol.proven_optimal:
            from src.solvers.alns import ALNS

            alns = ALNS(params=params)
            sol = alns.solve(
                inst,
                time_limit_sec=alns_time_limit,
                seed=seed,
                warm_start=warm,
            )
            tier = "alns_fallback"
        else:
            tier = "cpsat"
    else:
        from src.solvers.alns import ALNS

        alns = ALNS(params=params)
        sol = alns.solve(
            inst,
            time_limit_sec=alns_time_limit,
            seed=seed,
            warm_start=warm,
        )
        tier = "alns"

    validate(inst, sol)
    return _best_of(sol, warm, inst), tier
