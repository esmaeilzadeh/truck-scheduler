"""ALNS hyperparameter tuning — offline, one-time (Section 6H).

Uses the shared random-search framework in ``src.tuning.random_search``.
"""

from __future__ import annotations

import argparse
import json

from src.solvers.alns import ALNS, DEFAULT_PARAMS
from src.tuning.random_search import (
    TunerSpec,
    load_tune_instances,
    run_random_search,
)

# Re-exported for tune_alns_vs_hybrid and tests
PARAM_RANGES = {
    "rho_min": (0.05, 0.20),
    "rho_max": (0.20, 0.35),
    "lambda": (0.10, 0.20),
    "segment_length": (100, 200),
    "sigma1": (10, 50),
    "sigma2": (5, 25),
    "sigma3": (1, 20),
    "cooling": (0.995, 0.99999),
    "start_temp_ctrl": (0.01, 0.20),
    "regret_k": (2, 4),
    "d_wr": (1.0, 6.0),
    # Cap destroy size so repairs stay cheap on large K (0 = uncapped)
    "q_cap": (4, 40),
}

# SPEC 6H.3 course-project light search (others frozen at DEFAULT_PARAMS)
LIGHT_PARAM_KEYS = ("rho_max", "cooling", "segment_length", "lambda")

_INT_PARAMS = {
    "segment_length",
    "regret_k",
    "sigma1",
    "sigma2",
    "sigma3",
    "q_cap",
}


def _normalize_alns(cfg: dict) -> dict:
    if cfg.get("rho_min", 0) > cfg.get("rho_max", 0):
        cfg["rho_min"], cfg["rho_max"] = cfg["rho_max"], cfg["rho_min"]
    return cfg


def alns_tuner_spec() -> TunerSpec:
    return TunerSpec(
        name="alns",
        default_params=dict(DEFAULT_PARAMS),
        param_ranges=PARAM_RANGES,
        int_params=_INT_PARAMS,
        make_solver=lambda cfg: ALNS(params=cfg),
        light_param_keys=LIGHT_PARAM_KEYS,
        frozen_keys=("max_iterations",),
        normalize=_normalize_alns,
        csv_filename="alns_tuning.csv",
        config_path="config/alns_params.json",
    )


def tune(
    instances,
    n_configs: int = 200,
    seeds_per_config: int = 3,
    run_budget_sec: float = 5.0,
    output_dir: str = "data/results",
    config_path: str = "config/alns_params.json",
    light: bool = False,
) -> dict:
    """Run random-search tuning. Returns best config dict."""
    return run_random_search(
        alns_tuner_spec(),
        instances,
        n_configs=n_configs,
        seeds_per_config=seeds_per_config,
        run_budget_sec=run_budget_sec,
        output_dir=output_dir,
        config_path=config_path,
        light=light,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALNS hyperparameter random search (SPEC 6H)"
    )
    parser.add_argument("--n-configs", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--max-instances", type=int, default=8)
    parser.add_argument(
        "--no-large", action="store_true", help="Only on-disk TUNE (K≤20)"
    )
    parser.add_argument(
        "--light",
        "--course",
        action="store_true",
        dest="light",
        help="SPEC 6H.3 light search: only {rho_max, cooling, segment_length, lambda}",
    )
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--config-path", default="config/alns_params.json")
    args = parser.parse_args()

    instances = load_tune_instances(
        max_total=args.max_instances,
        include_large=not args.no_large,
    )
    best = tune(
        instances,
        n_configs=args.n_configs,
        seeds_per_config=args.seeds,
        run_budget_sec=args.budget,
        output_dir=args.output_dir,
        config_path=args.config_path,
        light=args.light,
    )
    print("Best config:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
