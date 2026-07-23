"""Smoke tests for shared random-search tuner and GA-Tabu wrapper."""

from __future__ import annotations

import random

from src.instance_gen import gen_instance
from src.tuning.random_search import (
    load_tune_instances,
    sample_config,
)
from src.tuning.tune_alns import alns_tuner_spec
from src.tuning.tune_ga_tabu import ga_tabu_tuner_spec, tune as tune_ga_tabu


def test_sample_config_alns_and_ga_tabu():
    rng = random.Random(0)
    alns_cfg = sample_config(rng, alns_tuner_spec(), light=True)
    assert "rho_max" in alns_cfg
    assert "cooling" in alns_cfg

    gt_cfg = sample_config(rng, ga_tabu_tuner_spec(), light=True)
    assert "population_size" in gt_cfg
    assert "local_search_rate" in gt_cfg
    assert gt_cfg["elite_count"] < gt_cfg["population_size"]


def test_ga_tabu_tune_smoke(tmp_path):
    inst = gen_instance(seed=0, M=3, N=3, G=2)
    inst.id = "smoke_ga_tabu"
    out = tmp_path / "results"
    cfg_path = tmp_path / "ga_tabu_params.json"
    best = tune_ga_tabu(
        [inst],
        n_configs=2,
        seeds_per_config=1,
        run_budget_sec=0.2,
        output_dir=str(out),
        config_path=str(cfg_path),
        light=True,
    )
    assert cfg_path.is_file()
    assert (out / "ga_tabu_tuning.csv").is_file()
    assert "population_size" in best


def test_load_tune_instances_nonempty():
    insts = load_tune_instances(max_total=3, include_large=False)
    assert len(insts) >= 1
