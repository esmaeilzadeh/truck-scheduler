"""Smoke tests for shared random-search tuner and hybrid wrappers."""

from __future__ import annotations

import random

from src.instance_gen import gen_instance
from src.tuning.random_search import (
    load_tune_instances,
    sample_config,
)
from src.tuning.tune_alns import alns_tuner_spec
from src.tuning.tune_alns_tabu import alns_tabu_tuner_spec, tune as tune_alns_tabu
from src.tuning.tune_ga_tabu import ga_tabu_tuner_spec, tune as tune_ga_tabu


def test_sample_config_alns_and_hybrids():
    rng = random.Random(0)
    alns_cfg = sample_config(rng, alns_tuner_spec(), light=True)
    assert "rho_max" in alns_cfg
    assert "cooling" in alns_cfg

    gt_cfg = sample_config(rng, ga_tabu_tuner_spec(), light=True)
    assert "population_size" in gt_cfg
    assert "local_search_rate" in gt_cfg
    assert gt_cfg["elite_count"] < gt_cfg["population_size"]

    at_cfg = sample_config(rng, alns_tabu_tuner_spec(), light=True)
    assert "local_search_rate" in at_cfg
    assert "local_tabu_iters" in at_cfg
    assert "rho_max" in at_cfg
    assert "local_search_budget_frac" in at_cfg
    assert at_cfg["q_cap"] == 0
    assert 0.05 <= at_cfg["local_search_budget_frac"] <= 0.25


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


def test_alns_tabu_tune_smoke(tmp_path):
    inst = gen_instance(seed=0, M=3, N=3, G=2)
    inst.id = "smoke_alns_tabu"
    out = tmp_path / "results"
    cfg_path = tmp_path / "alns_tabu_params.json"
    best = tune_alns_tabu(
        [inst],
        n_configs=2,
        seeds_per_config=1,
        run_budget_sec=0.2,
        output_dir=str(out),
        config_path=str(cfg_path),
        light=True,
    )
    assert cfg_path.is_file()
    assert (out / "alns_tabu_tuning.csv").is_file()
    assert "local_search_rate" in best
    assert "q_cap" in best
    assert best["q_cap"] == 0
    assert "local_search_budget_frac" in best


def test_load_tune_instances_nonempty():
    insts = load_tune_instances(max_total=3, include_large=False)
    assert len(insts) >= 1
