"""Smoke tests for ALNS vs Hybrid constrained tuner."""

from __future__ import annotations

from src.tuning.tune_alns_vs_hybrid import (
    budget_for_k,
    evaluate_config,
    meets_criteria,
    parse_buckets,
    tune_loop,
)
from src.solvers.alns import DEFAULT_PARAMS


def test_parse_buckets():
    assert parse_buckets("50x50,100x100") == [(50, 50), (100, 100)]


def test_budget_for_k_anchors():
    assert budget_for_k(100) == 3.0
    assert budget_for_k(800) == 24.0
    mid = budget_for_k(300)
    assert 6.0 < mid < 12.0


def test_tune_loop_smoke(tmp_path):
    out = tmp_path / "results"
    cfg_path = tmp_path / "alns_params.json"
    validation = tune_loop(
        buckets=[(4, 4), (5, 5)],
        train_seeds=[0],
        holdout_seeds=[1],
        eval_seeds=[0],
        n_configs_per_round=1,
        max_rounds=1,
        target_win_rate=0.0,  # always pass quality bar for plumbing
        max_time_ratio=50.0,  # very loose for tiny budgets
        budget_scale=0.05,
        output_dir=out,
        config_path=cfg_path,
        max_wall_sec=120.0,
        rng_seed=0,
    )
    assert cfg_path.is_file()
    assert (out / "alns_vs_hybrid_tuning.csv").is_file()
    assert (out / "alns_vs_hybrid_validation.json").is_file()
    assert "overall" in validation
    assert validation["best_config"]
    assert "rho_min" in validation["best_config"]


def test_evaluate_config_runs():
    from src.instance_gen import gen_instance

    inst = gen_instance(seed=0, M=3, N=3, G=2)
    inst.id = "TRAIN_s0_M3_N3_G2"
    cfg = {**DEFAULT_PARAMS}
    summary = evaluate_config(
        cfg,
        [inst],
        [0],
        budget_scale=0.05,
        max_time_ratio=1.5,
    )
    assert summary.n_runs == 1
    assert summary.per_bucket
    # Loose criteria should hold on a single short run for plumbing
    assert meets_criteria(
        summary,
        target_win_rate=0.0,
        max_time_ratio=100.0,
    )
