"""Runtime dispatcher — tier selection at production time (Section 8)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from src.model import Instance, Solution, feasibility_precheck
from src.solvers.greedy import GreedyERDSPT
from src.validate import validate

_DEFAULT_SWITCH_POLICY = {
    "budget_sec": 10.0,
    "cpsat_time_limit_sec": 10.0,
    "threshold_K": 25,
    "T_cap": None,
    "safety_margin_steps": 1,
}

_DEFAULT_ALNS_PARAMS = {
    "rho_min": 0.10,
    "rho_max": 0.30,
    "lambda": 0.15,
    "segment_length": 150,
    "sigma1": 33,
    "sigma2": 9,
    "sigma3": 13,
    "cooling": 0.99975,
    "start_temp_ctrl": 0.05,
    "final_temp_ratio": 0.002,
    "regret_k": 3,
    "d_wr": 3.0,
    "q_cap": 0,
    "max_iterations": 25000,
}


def _load_json(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_switch_policy(
    policy_path: str | Path | None = "config/switch_policy.json",
) -> dict:
    """Load switch policy with defaults (used by UI for timeout defaults)."""
    policy = dict(_DEFAULT_SWITCH_POLICY)
    loaded = _load_json(policy_path)
    if loaded:
        policy.update(loaded)
    return policy


def cpsat_time_limit_from_policy(policy: dict | None = None) -> float:
    """CP-SAT wall-clock budget from config (default 10s)."""
    if policy is None:
        policy = load_switch_policy()
    if "cpsat_time_limit_sec" in policy and policy["cpsat_time_limit_sec"] is not None:
        return float(policy["cpsat_time_limit_sec"])
    return float(policy.get("budget_sec", 10.0))


def _best_of(sol: Solution, warm: Solution, inst: Instance) -> Solution:
    """Return the solution with lower objective."""
    obj_sol = sol.objective(inst)
    obj_warm = warm.objective(inst)
    return sol if obj_sol <= obj_warm else warm


_VALID_FORCE_TIERS = frozenset(
    {None, "auto", "greedy", "cpsat", "alns", "tabu", "ga", "ga_tabu"}
)


def solve(
    inst: Instance,
    policy_path: str | Path | None = "config/switch_policy.json",
    params_path: str | Path | None = "config/alns_params.json",
    exact_time_limit: float | None = None,
    alns_time_limit: float | None = None,
    seed: int = 0,
    force_tier: str | None = None,
    stop_event: threading.Event | None = None,
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
        Time limit for CP-SAT. None → config ``cpsat_time_limit_sec`` (else budget_sec).
    alns_time_limit : float or None
        Time limit for ALNS / Tabu / GA / hybrid. None → use policy budget_sec.
    seed : int
        Random seed for metaheuristics.
    force_tier : str or None
        None/"auto" → size policy;
        "greedy" | "cpsat" | "alns" | "tabu" | "ga" | "ga_tabu" → force that solver.
        Auto never selects tabu/ga/ga_tabu.
    stop_event : threading.Event or None
        When set, cooperative cancel for CP-SAT (and ignored by greedy).
    """
    if force_tier not in _VALID_FORCE_TIERS:
        raise ValueError(
            f"Unknown force_tier={force_tier!r}; "
            f"expected one of None, 'auto', 'greedy', 'cpsat', 'alns', "
            f"'tabu', 'ga', 'ga_tabu'"
        )

    feasibility_precheck(inst)

    # Tier 3: greedy warm start (always)
    greedy = GreedyERDSPT()
    warm = greedy.solve(inst)

    if force_tier == "greedy":
        validate(inst, warm)
        return warm, "greedy"

    # Load configs with safe fallbacks
    if policy_path is None:
        policy = dict(_DEFAULT_SWITCH_POLICY)
    else:
        policy = load_switch_policy(policy_path)
    params = _load_json(params_path) or _DEFAULT_ALNS_PARAMS

    threshold_K = policy.get("threshold_K", 25)
    T_cap = policy.get("T_cap", None)
    budget = float(policy.get("budget_sec", 10.0))
    cpsat_budget = cpsat_time_limit_from_policy(policy)

    if exact_time_limit is None:
        exact_time_limit = cpsat_budget
    if alns_time_limit is None:
        alns_time_limit = budget

    auto = force_tier in (None, "auto")

    if force_tier == "cpsat" or (
        auto
        and (len(inst.ops) <= threshold_K)
        and (T_cap is None or inst.T <= T_cap)
    ):
        from src.solvers.cpsat import CPSAT

        cpsat = CPSAT()
        sol = cpsat.solve(
            inst,
            time_limit_sec=exact_time_limit,
            warm_start=warm,
            stop_event=stop_event,
        )
        # Auto only: if not proven optimal, fall back to ALNS
        if (
            auto
            and exact_time_limit is not None
            and not sol.proven_optimal
        ):
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

        # Forced CP-SAT: return the CP-SAT incumbent as-is (never swap to greedy
        # while still labeling the run as cpsat — that looked like an instant
        # greedy result). Auto/ALNS paths keep the never-worse-than-greedy guard.
        if force_tier == "cpsat":
            validate(inst, sol)
            return sol, "cpsat"
    elif force_tier == "tabu":
        from src.solvers.tabu import TabuSearch

        sol = TabuSearch().solve(
            inst,
            time_limit_sec=alns_time_limit,
            seed=seed,
            warm_start=warm,
        )
        tier = "tabu"
    elif force_tier == "ga":
        from src.solvers.ga import GeneticAlgorithm

        sol = GeneticAlgorithm().solve(
            inst,
            time_limit_sec=alns_time_limit,
            seed=seed,
            warm_start=warm,
        )
        tier = "ga"
    elif force_tier == "ga_tabu":
        from src.solvers.ga_tabu import HybridGATabu

        sol = HybridGATabu().solve(
            inst,
            time_limit_sec=alns_time_limit,
            seed=seed,
            warm_start=warm,
        )
        tier = "ga_tabu"
    else:
        # force_tier == "alns" or auto with K > threshold
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
