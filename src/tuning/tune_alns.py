"""ALNS hyperparameter tuning — offline, one-time (Section 6H)."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

from src.model import Instance
from src.solvers.alns import ALNS, DEFAULT_PARAMS
from src.solvers.cpsat import CPSAT
from src.validate import validate


PARAM_RANGES = {
    "rho_min": (0.05, 0.20),
    "rho_max": (0.20, 0.50),
    "lambda": (0.01, 0.50),
    "segment_length": (50, 500),
    "sigma1": (10, 50),
    "sigma2": (5, 25),
    "sigma3": (1, 20),
    "cooling": (0.995, 0.99999),
    "start_temp_ctrl": (0.01, 0.20),
    "regret_k": (2, 4),
    "d_wr": (1.0, 6.0),
}


def _sample_config(rng: random.Random) -> dict:
    cfg = {}
    for k, (lo, hi) in PARAM_RANGES.items():
        if k in ("segment_length", "regret_k", "sigma1", "sigma2", "sigma3"):
            cfg[k] = rng.randint(int(lo), int(hi))
        elif k == "d_wr":
            cfg[k] = rng.uniform(lo, hi)
        else:
            cfg[k] = rng.uniform(lo, hi)
    cfg["max_iterations"] = DEFAULT_PARAMS["max_iterations"]
    return cfg


def _get_ref(inst: Instance, cpsat: CPSAT, budget: float) -> float | None:
    """Try to get CP-SAT optimum as reference. Returns None if not solved in time."""
    try:
        sol = cpsat.solve(inst, time_limit_sec=budget)
        if sol.proven_optimal:
            return sol.objective(inst)
    except Exception:
        pass
    return None


def tune(
    instances: list[Instance],
    n_configs: int = 200,
    seeds_per_config: int = 3,
    run_budget_sec: float = 5.0,
    output_dir: str = "data/results",
    config_path: str = "config/alns_params.json",
) -> dict:
    """Run random-search tuning. Returns best config dict."""
    rng = random.Random(42)
    cpsat = CPSAT()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    # Compute references
    refs: dict[str, float] = {}
    for inst in instances:
        ref = _get_ref(inst, cpsat, run_budget_sec)
        if ref is not None:
            refs[inst.id] = ref

    # For instances without CP-SAT opt, use a proxy (run ALNS with defaults, best across seeds)
    proxy: dict[str, float] = {}
    for inst in instances:
        if inst.id not in refs:
            best_obj = float("inf")
            for s in range(seeds_per_config):
                try:
                    alns = ALNS()
                    sol = alns.solve(inst, time_limit_sec=run_budget_sec, seed=s)
                    validate(inst, sol)
                    best_obj = min(best_obj, sol.objective(inst))
                except Exception:
                    pass
            proxy[inst.id] = best_obj

    def ref(inst: Instance) -> float:
        return refs.get(inst.id, proxy.get(inst.id, 1.0))

    # Random search
    results: list[dict] = []
    best_mean_gap = float("inf")
    best_cfg: dict = {}

    for trial in range(n_configs):
        cfg = _sample_config(rng)
        gaps = []
        for inst in instances:
            for s in range(seeds_per_config):
                try:
                    alns = ALNS(params=cfg)
                    sol = alns.solve(inst, time_limit_sec=run_budget_sec, seed=s)
                    validate(inst, sol)
                    gap = (sol.objective(inst) - ref(inst)) / ref(inst)
                    gaps.append(gap)
                except Exception:
                    gaps.append(1.0)

        mean_gap = sum(gaps) / len(gaps) if gaps else 1.0
        results.append({"trial": trial, "mean_gap": mean_gap, **cfg})

        if mean_gap < best_mean_gap:
            best_mean_gap = mean_gap
            best_cfg = dict(cfg)

        if (trial + 1) % 10 == 0:
            print(f"  trial {trial + 1}/{n_configs}, best gap so far: {best_mean_gap:.4f}")

    # Write results CSV
    csv_path = Path(output_dir) / "alns_tuning.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Write best config
    with open(config_path, "w") as f:
        json.dump(best_cfg, f, indent=2)

    print(f"Best mean gap: {best_mean_gap:.4f}")
    print(f"Config written to {config_path}")
    return best_cfg


if __name__ == "__main__":
    from src.io_utils import read_instance
    from src.instance_gen import gen_instance

    # Prefer on-disk TUNE suite; fall back to a tiny in-memory set
    tune_dir = Path("data/instances/TUNE")
    instances: list[Instance] = []
    if tune_dir.is_dir():
        for path in sorted(tune_dir.glob("*.json"))[:12]:
            instances.append(read_instance(path))
    if not instances:
        for seed in range(4):
            for M, N in [(3, 3), (5, 5)]:
                instances.append(gen_instance(seed=seed, M=M, N=N, G=2))

    # Short course-project budget (not full SPEC 6H)
    best = tune(
        instances,
        n_configs=20,
        seeds_per_config=2,
        run_budget_sec=2.0,
    )
    print("Best config:", best)
