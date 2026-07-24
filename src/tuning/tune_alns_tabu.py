"""ALNS+Tabu hybrid hyperparameter tuning — offline random search.

Reuses the shared SPEC 6H-style framework in ``src.tuning.random_search``.
Searches gated-Tabu knobs plus light ALNS destroy-size params.
"""

from __future__ import annotations

import argparse
import json

from src.instance_gen import gen_instance
from src.model import Instance
from src.solvers.alns_tabu import DEFAULT_PARAMS, HybridALNSTabu
from src.tuning.random_search import (
    TunerSpec,
    load_tune_instances,
    run_random_search,
    select_stratified,
)

PARAM_RANGES = {
    "local_search_rate": (0.02, 0.20),
    "local_tabu_iters": (5, 40),
    "local_neighborhood_size": (8, 25),
    "tabu_tenure": (3, 15),
    "rho_max": (0.20, 0.35),
    "local_search_budget_frac": (0.05, 0.25),
}

LIGHT_PARAM_KEYS = (
    "local_search_rate",
    "local_tabu_iters",
    "local_neighborhood_size",
    "rho_max",
    "local_search_budget_frac",
)

_INT_PARAMS = {
    "local_tabu_iters",
    "local_neighborhood_size",
    "tabu_tenure",
}

_FROZEN_KEYS = (
    "rho_min",
    "lambda",
    "segment_length",
    "sigma1",
    "sigma2",
    "sigma3",
    "cooling",
    "start_temp_ctrl",
    "final_temp_ratio",
    "regret_k",
    "d_wr",
    "max_iterations",
    "swap_prob",
    "q_cap",  # keep destroy uncapped (aligned with pure ALNS)
)


def _normalize_alns_tabu(cfg: dict) -> dict:
    rho_min = float(cfg.get("rho_min", DEFAULT_PARAMS["rho_min"]))
    rho_max = float(cfg.get("rho_max", DEFAULT_PARAMS["rho_max"]))
    if rho_min > rho_max:
        cfg["rho_min"], cfg["rho_max"] = rho_max, rho_min
    cfg["q_cap"] = 0
    frac = float(cfg.get("local_search_budget_frac", 0.15))
    cfg["local_search_budget_frac"] = max(0.0, min(1.0, frac))
    return cfg


def alns_tabu_tuner_spec() -> TunerSpec:
    return TunerSpec(
        name="alns_tabu",
        default_params=dict(DEFAULT_PARAMS),
        param_ranges=PARAM_RANGES,
        int_params=_INT_PARAMS,
        make_solver=lambda cfg: HybridALNSTabu(params=cfg),
        light_param_keys=LIGHT_PARAM_KEYS,
        frozen_keys=_FROZEN_KEYS,
        normalize=_normalize_alns_tabu,
        csv_filename="alns_tabu_tuning.csv",
        config_path="config/alns_tabu_params.json",
    )


def load_alns_tabu_tune_instances(
    max_total: int = 8,
    include_large: bool = True,
) -> list[Instance]:
    """TUNE set plus medium/large generated instances (50x50, 100x100)."""
    base = load_tune_instances(max_total=max_total * 2, include_large=include_large)
    extras: list[Instance] = []
    if include_large:
        for seed, (M, N, G) in ((0, (50, 50, 2)), (0, (100, 100, 2))):
            inst = gen_instance(seed=seed, M=M, N=N, G=G)
            inst.id = f"TUNE_extra_s{seed}_M{M}_N{N}_G{G}"
            extras.append(inst)
    return select_stratified(base + extras, per_bucket=1, max_total=max_total)


def tune(
    instances,
    n_configs: int = 40,
    seeds_per_config: int = 2,
    run_budget_sec: float = 3.0,
    output_dir: str = "data/results",
    config_path: str = "config/alns_tabu_params.json",
    light: bool = False,
) -> dict:
    """Run random-search tuning for HybridALNSTabu. Returns best config."""
    return run_random_search(
        alns_tabu_tuner_spec(),
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
        description="ALNS+Tabu hybrid hyperparameter random search"
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
        action="store_true",
        help=(
            "Vary local_search_rate, local_tabu_iters, "
            "local_neighborhood_size, rho_max, local_search_budget_frac"
        ),
    )
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--config-path", default="config/alns_tabu_params.json")
    args = parser.parse_args()

    instances = load_alns_tabu_tune_instances(
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
