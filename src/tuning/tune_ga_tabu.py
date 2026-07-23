"""GA+Tabu hybrid hyperparameter tuning — offline random search.

Reuses the shared SPEC 6H-style framework in ``src.tuning.random_search``.
There is no legacy GA-only or Tabu-only tuner; both component knobs are
searched jointly through this hybrid space.
"""

from __future__ import annotations

import argparse
import json

from src.solvers.ga_tabu import DEFAULT_PARAMS, HybridGATabu
from src.tuning.random_search import (
    TunerSpec,
    load_tune_instances,
    run_random_search,
)

PARAM_RANGES = {
    "population_size": (20, 60),
    "elite_count": (1, 5),
    "tournament_k": (2, 5),
    "crossover_rate": (0.7, 1.0),
    "mutation_rate": (0.05, 0.4),
    "local_search_rate": (0.1, 0.6),
    "local_tabu_iters": (20, 150),
    "local_neighborhood_size": (10, 40),
    "tabu_tenure": (3, 15),
    "swap_prob": (0.3, 0.7),
}

# Light search: intensify local-search / population knobs most affecting quality
LIGHT_PARAM_KEYS = (
    "population_size",
    "local_search_rate",
    "local_tabu_iters",
    "mutation_rate",
)

_INT_PARAMS = {
    "population_size",
    "elite_count",
    "tournament_k",
    "local_tabu_iters",
    "local_neighborhood_size",
    "tabu_tenure",
}


def _normalize_ga_tabu(cfg: dict) -> dict:
    # elite_count must be < population_size
    pop = int(cfg.get("population_size", DEFAULT_PARAMS["population_size"]))
    elite = int(cfg.get("elite_count", DEFAULT_PARAMS["elite_count"]))
    if elite >= pop:
        cfg["elite_count"] = max(1, pop // 5)
    # tournament_k must be <= population
    tk = int(cfg.get("tournament_k", DEFAULT_PARAMS["tournament_k"]))
    if tk > pop:
        cfg["tournament_k"] = max(2, min(tk, pop))
    return cfg


def ga_tabu_tuner_spec() -> TunerSpec:
    return TunerSpec(
        name="ga_tabu",
        default_params=dict(DEFAULT_PARAMS),
        param_ranges=PARAM_RANGES,
        int_params=_INT_PARAMS,
        make_solver=lambda cfg: HybridGATabu(params=cfg),
        light_param_keys=LIGHT_PARAM_KEYS,
        frozen_keys=("max_generations",),
        normalize=_normalize_ga_tabu,
        csv_filename="ga_tabu_tuning.csv",
        config_path="config/ga_tabu_params.json",
    )


def tune(
    instances,
    n_configs: int = 40,
    seeds_per_config: int = 2,
    run_budget_sec: float = 3.0,
    output_dir: str = "data/results",
    config_path: str = "config/ga_tabu_params.json",
    light: bool = False,
) -> dict:
    """Run random-search tuning for HybridGATabu. Returns best config."""
    return run_random_search(
        ga_tabu_tuner_spec(),
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
        description="GA+Tabu hybrid hyperparameter random search"
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
        help="Vary only population_size, local_search_rate, local_tabu_iters, mutation_rate",
    )
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--config-path", default="config/ga_tabu_params.json")
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
