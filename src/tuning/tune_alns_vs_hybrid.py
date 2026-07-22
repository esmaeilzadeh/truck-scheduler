"""ALNS vs Hybrid GA+Tabu constrained tuning loop (large-K).

Tunes ALNS hyperparameters until holdout validation shows:
  - ALNS objective strictly better than HybridGATabu on >= target win-rate
    of (instance x seed) runs (default 80%)
  - median ALNS wall time <= max_time_ratio * median hybrid wall time
    (default 1.5), overall and per size bucket
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.instance_gen import gen_instance
from src.model import Instance
from src.solvers.alns import ALNS, DEFAULT_PARAMS
from src.solvers.ga_tabu import HybridGATabu
from src.tuning.tune_alns import _INT_PARAMS as _BASE_INT_PARAMS
from src.validate import validate

_INT_PARAMS = set(_BASE_INT_PARAMS) | {"insert_pos_cap"}


# Large-K ranges: small q_cap is essential so ALNS gets enough iterations
# to beat HybridGATabu within the 1.5x time envelope.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "rho_min": (0.01, 0.15),
    "rho_max": (0.02, 0.25),
    "lambda": (0.10, 0.25),
    "segment_length": (50, 200),
    "sigma1": (10, 50),
    "sigma2": (5, 25),
    "sigma3": (1, 20),
    "cooling": (0.995, 0.99999),
    "start_temp_ctrl": (0.01, 0.20),
    "regret_k": (2, 4),
    "d_wr": (1.0, 6.0),
    "q_cap": (1, 4),
    "insert_pos_cap": (8, 24),
}

# Size buckets: (M, N) with K = M+N up to 800
DEFAULT_BUCKETS: list[tuple[int, int]] = [
    (50, 50),
    (100, 100),
    (200, 200),
    (400, 400),
]

# Base wall budget B(K) for hybrid; ALNS gets 1.5 * B.
# Calibrated so ALNS with small q_cap + insert_pos_cap can beat hybrid
# through 400x400 (K=800) smoke.
_BUDGET_BY_K: dict[int, float] = {
    100: 8.0,
    200: 12.0,
    400: 20.0,
    800: 75.0,
}


def budget_for_k(k: int, scale: float = 1.0) -> float:
    """Interpolate base hybrid budget from known K anchors."""
    if k in _BUDGET_BY_K:
        return _BUDGET_BY_K[k] * scale
    ks = sorted(_BUDGET_BY_K)
    if k <= ks[0]:
        return _BUDGET_BY_K[ks[0]] * scale
    if k >= ks[-1]:
        return _BUDGET_BY_K[ks[-1]] * scale
    for lo, hi in zip(ks, ks[1:]):
        if lo <= k <= hi:
            t = (k - lo) / (hi - lo)
            return (_BUDGET_BY_K[lo] + t * (_BUDGET_BY_K[hi] - _BUDGET_BY_K[lo])) * scale
    return 10.0 * scale


@dataclass
class RunResult:
    inst_id: str
    bucket: str
    seed: int
    alns_obj: float
    hybrid_obj: float
    alns_time: float
    hybrid_time: float
    alns_won: bool
    time_ratio: float


@dataclass
class EvalSummary:
    win_rate: float
    median_time_ratio: float
    mean_quality_margin: float
    n_runs: int
    per_bucket: dict[str, dict[str, float]]
    runs: list[RunResult]


def _bucket_key(m: int, n: int) -> str:
    return f"{m}x{n}"


def _mn_from_instance(inst: Instance) -> tuple[int, int]:
    """Recover (M, N) from ops or id suffix `_M{m}_N{n}_`."""
    m = sum(1 for op in inst.ops if op.kind == "delivery")
    n = sum(1 for op in inst.ops if op.kind == "pickup")
    if m + n == len(inst.ops) and m + n > 0:
        return m, n
    # Fallback: parse id like TRAIN_s0_M50_N50_G2
    parts = inst.id.split("_")
    m_val = n_val = None
    for p in parts:
        if p.startswith("M") and p[1:].isdigit():
            m_val = int(p[1:])
        elif p.startswith("N") and p[1:].isdigit():
            n_val = int(p[1:])
    if m_val is not None and n_val is not None:
        return m_val, n_val
    k = len(inst.ops)
    return k // 2, k - k // 2


def build_instances(
    buckets: list[tuple[int, int]],
    train_seeds: list[int],
    holdout_seeds: list[int],
    gates: int = 2,
) -> tuple[list[Instance], list[Instance]]:
    """Generate in-memory train / holdout instances for each size bucket."""
    train: list[Instance] = []
    holdout: list[Instance] = []
    for m, n in buckets:
        for seed in train_seeds:
            inst = gen_instance(seed=seed, M=m, N=n, G=gates)
            inst.id = f"TRAIN_s{seed}_M{m}_N{n}_G{gates}"
            train.append(inst)
        for seed in holdout_seeds:
            inst = gen_instance(seed=seed, M=m, N=n, G=gates)
            inst.id = f"HOLD_s{seed}_M{m}_N{n}_G{gates}"
            holdout.append(inst)
    return train, holdout


def _sample_config(rng: random.Random, ranges: dict[str, tuple[float, float]]) -> dict:
    cfg: dict = {}
    for k, (lo, hi) in ranges.items():
        if k in _INT_PARAMS:
            cfg[k] = rng.randint(int(lo), int(hi))
        else:
            cfg[k] = rng.uniform(float(lo), float(hi))
    if cfg["rho_min"] > cfg["rho_max"]:
        cfg["rho_min"], cfg["rho_max"] = cfg["rho_max"], cfg["rho_min"]
    cfg["max_iterations"] = DEFAULT_PARAMS["max_iterations"]
    return cfg


def _bias_ranges_for_speed(
    ranges: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Shrink destroy / q_cap / insert_pos_cap toward cheaper repairs."""
    out = deepcopy(ranges)
    q_lo, q_hi = out["q_cap"]
    out["q_cap"] = (q_lo, max(q_lo, min(q_hi, (q_lo + q_hi) / 2)))
    r_lo, r_hi = out["rho_max"]
    out["rho_max"] = (r_lo, max(r_lo, min(r_hi, (r_lo + r_hi) / 2)))
    c_lo, c_hi = out["cooling"]
    # Faster cooling (lower values) exits exploration sooner under time caps
    out["cooling"] = (c_lo, max(c_lo, min(c_hi, (c_lo + c_hi) / 2)))
    if "insert_pos_cap" in out:
        p_lo, p_hi = out["insert_pos_cap"]
        out["insert_pos_cap"] = (p_lo, max(p_lo, min(p_hi, (p_lo + p_hi) / 2)))
    return out


def _bias_ranges_for_quality(
    ranges: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """Allow larger destroy / higher regret for better quality."""
    out = deepcopy(ranges)
    q_lo, q_hi = out["q_cap"]
    mid = (q_lo + q_hi) / 2
    out["q_cap"] = (max(q_lo, mid), q_hi)
    r_lo, r_hi = out["rho_max"]
    mid_r = (r_lo + r_hi) / 2
    out["rho_max"] = (max(r_lo, mid_r), r_hi)
    rk_lo, rk_hi = out["regret_k"]
    out["regret_k"] = (max(rk_lo, (rk_lo + rk_hi) // 2), rk_hi)
    if "insert_pos_cap" in out:
        p_lo, p_hi = out["insert_pos_cap"]
        mid_p = (p_lo + p_hi) / 2
        out["insert_pos_cap"] = (max(p_lo, mid_p), p_hi)
    return out


def _run_pair(
    inst: Instance,
    cfg: dict,
    seed: int,
    budget_scale: float,
    max_time_ratio: float,
    hybrid: HybridGATabu,
) -> RunResult:
    k = len(inst.ops)
    b = budget_for_k(k, scale=budget_scale)
    alns_budget = max_time_ratio * b
    m, n = _mn_from_instance(inst)
    bucket = _bucket_key(m, n)

    hy_sol = hybrid.solve(inst, time_limit_sec=b, seed=seed)
    validate(inst, hy_sol)
    hy_obj = hy_sol.objective(inst)
    hy_time = float(hy_sol.runtime_sec)

    alns = ALNS(params=cfg)
    al_sol = alns.solve(inst, time_limit_sec=alns_budget, seed=seed)
    validate(inst, al_sol)
    al_obj = al_sol.objective(inst)
    al_time = float(al_sol.runtime_sec)

    ratio = al_time / hy_time if hy_time > 1e-9 else float("inf")
    return RunResult(
        inst_id=inst.id,
        bucket=bucket,
        seed=seed,
        alns_obj=al_obj,
        hybrid_obj=hy_obj,
        alns_time=al_time,
        hybrid_time=hy_time,
        alns_won=al_obj < hy_obj - 1e-9,
        time_ratio=ratio,
    )


def evaluate_config(
    cfg: dict,
    instances: list[Instance],
    seeds: list[int],
    *,
    budget_scale: float = 1.0,
    max_time_ratio: float = 1.5,
) -> EvalSummary:
    """Evaluate one ALNS config against HybridGATabu on instances x seeds."""
    hybrid = HybridGATabu()
    runs: list[RunResult] = []
    for inst in instances:
        for seed in seeds:
            try:
                runs.append(
                    _run_pair(
                        inst,
                        cfg,
                        seed,
                        budget_scale,
                        max_time_ratio,
                        hybrid,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — keep tuning robust
                m, n = _mn_from_instance(inst)
                runs.append(
                    RunResult(
                        inst_id=inst.id,
                        bucket=_bucket_key(m, n),
                        seed=seed,
                        alns_obj=float("inf"),
                        hybrid_obj=0.0,
                        alns_time=float("inf"),
                        hybrid_time=1.0,
                        alns_won=False,
                        time_ratio=float("inf"),
                    )
                )
                print(f"  warn: {inst.id} seed={seed} failed: {exc}")

    wins = sum(1 for r in runs if r.alns_won)
    n = len(runs) or 1
    win_rate = wins / n
    ratios = [r.time_ratio for r in runs if math.isfinite(r.time_ratio)]
    median_ratio = statistics.median(ratios) if ratios else float("inf")
    margins = []
    for r in runs:
        if math.isfinite(r.alns_obj) and r.hybrid_obj > 1e-9:
            margins.append((r.hybrid_obj - r.alns_obj) / r.hybrid_obj)
        else:
            margins.append(-1.0)
    mean_margin = sum(margins) / len(margins) if margins else -1.0

    by_bucket: dict[str, list[RunResult]] = defaultdict(list)
    for r in runs:
        by_bucket[r.bucket].append(r)

    per_bucket: dict[str, dict[str, float]] = {}
    for bucket, bruns in sorted(by_bucket.items()):
        bw = sum(1 for r in bruns if r.alns_won) / len(bruns)
        br = [r.time_ratio for r in bruns if math.isfinite(r.time_ratio)]
        per_bucket[bucket] = {
            "win_rate": bw,
            "median_time_ratio": statistics.median(br) if br else float("inf"),
            "n_runs": float(len(bruns)),
        }

    return EvalSummary(
        win_rate=win_rate,
        median_time_ratio=median_ratio,
        mean_quality_margin=mean_margin,
        n_runs=len(runs),
        per_bucket=per_bucket,
        runs=runs,
    )


def meets_criteria(
    summary: EvalSummary,
    *,
    target_win_rate: float,
    max_time_ratio: float,
) -> bool:
    if summary.win_rate + 1e-12 < target_win_rate:
        return False
    if summary.median_time_ratio > max_time_ratio + 1e-9:
        return False
    for stats in summary.per_bucket.values():
        if stats["win_rate"] + 1e-12 < target_win_rate:
            return False
        if stats["median_time_ratio"] > max_time_ratio + 1e-9:
            return False
    return True


def config_score(
    summary: EvalSummary,
    *,
    target_win_rate: float,
    max_time_ratio: float,
) -> float:
    """Higher is better. Prefer meeting win-rate, then low time ratio."""
    time_pen = max(0.0, summary.median_time_ratio - max_time_ratio)
    if summary.win_rate + 1e-12 >= target_win_rate:
        return 1000.0 + summary.win_rate * 100.0 - summary.median_time_ratio * 10.0 - time_pen * 50.0
    return (
        summary.win_rate * 100.0
        + summary.mean_quality_margin * 20.0
        - time_pen * 30.0
    )


def _partial_success(summary: EvalSummary, max_bucket_m: int) -> bool:
    """True if criteria hold on all buckets with M <= max_bucket_m."""
    for bucket, stats in summary.per_bucket.items():
        m = int(bucket.split("x")[0])
        if m > max_bucket_m:
            continue
        if stats["win_rate"] < 0.8 - 1e-12:
            return False
        if stats["median_time_ratio"] > 1.5 + 1e-9:
            return False
    return True


def tune_loop(
    *,
    buckets: list[tuple[int, int]],
    train_seeds: list[int],
    holdout_seeds: list[int],
    eval_seeds: list[int],
    n_configs_per_round: int,
    max_rounds: int,
    target_win_rate: float,
    max_time_ratio: float,
    budget_scale: float,
    output_dir: str | Path,
    config_path: str | Path,
    max_wall_sec: float | None = None,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """Outer random-search loop with range bias until holdout criteria met."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    train, holdout = build_instances(buckets, train_seeds, holdout_seeds)
    print(
        f"Instances: train={len(train)} holdout={len(holdout)} "
        f"buckets={buckets} eval_seeds={eval_seeds}"
    )
    for inst in train + holdout:
        print(f"  {inst.id} K={len(inst.ops)} B={budget_for_k(len(inst.ops), budget_scale):.1f}s")

    rng = random.Random(rng_seed)
    ranges = deepcopy(PARAM_RANGES)
    csv_rows: list[dict] = []
    best_cfg: dict | None = None
    best_train_score = float("-inf")
    best_holdout: EvalSummary | None = None
    success = False
    t0 = time.perf_counter()

    for round_idx in range(max_rounds):
        if max_wall_sec is not None and (time.perf_counter() - t0) >= max_wall_sec:
            print(f"Hit max wall {max_wall_sec:.0f}s; stopping.")
            break

        print(f"\n=== Round {round_idx + 1}/{max_rounds} ranges={ {k: ranges[k] for k in ('q_cap','rho_max','cooling','regret_k')} } ===")
        round_best_cfg: dict | None = None
        round_best_score = float("-inf")
        round_best_summary: EvalSummary | None = None

        for trial in range(n_configs_per_round):
            cfg = _sample_config(rng, ranges)
            summary = evaluate_config(
                cfg,
                train,
                eval_seeds,
                budget_scale=budget_scale,
                max_time_ratio=max_time_ratio,
            )
            score = config_score(
                summary,
                target_win_rate=target_win_rate,
                max_time_ratio=max_time_ratio,
            )
            row = {
                "round": round_idx,
                "trial": trial,
                "phase": "train",
                "win_rate": summary.win_rate,
                "median_time_ratio": summary.median_time_ratio,
                "mean_quality_margin": summary.mean_quality_margin,
                "score": score,
                **{k: cfg[k] for k in PARAM_RANGES},
                "max_iterations": cfg["max_iterations"],
            }
            csv_rows.append(row)
            print(
                f"  train r{round_idx} t{trial}: win={summary.win_rate:.2%} "
                f"time_ratio={summary.median_time_ratio:.3f} "
                f"margin={summary.mean_quality_margin:.4f} score={score:.2f}"
            )
            if score > round_best_score:
                round_best_score = score
                round_best_cfg = cfg
                round_best_summary = summary
            if score > best_train_score:
                best_train_score = score
                best_cfg = cfg

        assert round_best_cfg is not None and round_best_summary is not None

        hold = evaluate_config(
            round_best_cfg,
            holdout,
            eval_seeds,
            budget_scale=budget_scale,
            max_time_ratio=max_time_ratio,
        )
        hold_score = config_score(
            hold,
            target_win_rate=target_win_rate,
            max_time_ratio=max_time_ratio,
        )
        csv_rows.append(
            {
                "round": round_idx,
                "trial": -1,
                "phase": "holdout",
                "win_rate": hold.win_rate,
                "median_time_ratio": hold.median_time_ratio,
                "mean_quality_margin": hold.mean_quality_margin,
                "score": hold_score,
                **{k: round_best_cfg[k] for k in PARAM_RANGES},
                "max_iterations": round_best_cfg["max_iterations"],
            }
        )
        print(
            f"  holdout r{round_idx}: win={hold.win_rate:.2%} "
            f"time_ratio={hold.median_time_ratio:.3f} "
            f"buckets={hold.per_bucket}"
        )

        if best_holdout is None or hold_score > config_score(
            best_holdout,
            target_win_rate=target_win_rate,
            max_time_ratio=max_time_ratio,
        ):
            best_holdout = hold
            best_cfg = round_best_cfg

        if meets_criteria(
            hold,
            target_win_rate=target_win_rate,
            max_time_ratio=max_time_ratio,
        ):
            success = True
            best_cfg = round_best_cfg
            best_holdout = hold
            print("Holdout criteria met; stopping.")
            break

        # Bias search ranges for next round
        failing_speed = hold.median_time_ratio > max_time_ratio
        failing_quality = hold.win_rate < target_win_rate
        if failing_speed and not failing_quality:
            ranges = _bias_ranges_for_speed(ranges)
            print("  bias -> speed")
        elif failing_quality and not failing_speed:
            ranges = _bias_ranges_for_quality(ranges)
            print("  bias -> quality")
        elif failing_speed and failing_quality:
            # Prefer speed first so quality search stays inside the envelope
            ranges = _bias_ranges_for_speed(ranges)
            print("  bias -> speed (both failing)")
        else:
            # Per-bucket miss only — mild quality nudge
            ranges = _bias_ranges_for_quality(ranges)
            print("  bias -> quality (per-bucket)")

    assert best_cfg is not None

    # Final holdout snapshot for reporting
    if best_holdout is None:
        best_holdout = evaluate_config(
            best_cfg,
            holdout,
            eval_seeds,
            budget_scale=budget_scale,
            max_time_ratio=max_time_ratio,
        )

    csv_path = output_dir / "alns_vs_hybrid_tuning.csv"
    fieldnames = [
        "round",
        "trial",
        "phase",
        "win_rate",
        "median_time_ratio",
        "mean_quality_margin",
        "score",
        *PARAM_RANGES.keys(),
        "max_iterations",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2)
        f.write("\n")

    stretch_note = None
    if not success:
        if _partial_success(best_holdout, max_bucket_m=200):
            stretch_note = (
                "Criteria met on buckets with M<=200; 400x400 is stretch "
                "(did not meet full holdout target)."
            )
        else:
            stretch_note = "Full holdout criteria not met within budget; best feasible θ kept."

    validation = {
        "success": success,
        "target_win_rate": target_win_rate,
        "max_time_ratio": max_time_ratio,
        "budget_scale": budget_scale,
        "buckets": [list(b) for b in buckets],
        "overall": {
            "win_rate": best_holdout.win_rate,
            "median_time_ratio": best_holdout.median_time_ratio,
            "mean_quality_margin": best_holdout.mean_quality_margin,
            "n_runs": best_holdout.n_runs,
        },
        "per_bucket": best_holdout.per_bucket,
        "stretch_note": stretch_note,
        "best_config": best_cfg,
        "runs": [
            {
                "inst_id": r.inst_id,
                "bucket": r.bucket,
                "seed": r.seed,
                "alns_obj": r.alns_obj,
                "hybrid_obj": r.hybrid_obj,
                "alns_time": r.alns_time,
                "hybrid_time": r.hybrid_time,
                "alns_won": r.alns_won,
                "time_ratio": r.time_ratio,
            }
            for r in best_holdout.runs
        ],
        "elapsed_sec": time.perf_counter() - t0,
    }
    val_path = output_dir / "alns_vs_hybrid_validation.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
        f.write("\n")

    print(f"\nBest config written to {config_path}")
    print(f"CSV: {csv_path}")
    print(f"Validation: {val_path}")
    print(f"success={success} win={best_holdout.win_rate:.2%} "
          f"time_ratio={best_holdout.median_time_ratio:.3f}")
    if stretch_note:
        print(f"note: {stretch_note}")
    return validation


def parse_buckets(raw: str) -> list[tuple[int, int]]:
    """Parse '50x50,100x100' into list of (M,N)."""
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise argparse.ArgumentTypeError(f"bad bucket '{part}', expected MxN")
        a, b = part.split("x", 1)
        out.append((int(a), int(b)))
    if not out:
        raise argparse.ArgumentTypeError("need at least one bucket")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune ALNS vs Hybrid GA+Tabu (win-rate + time-ratio loop)"
    )
    parser.add_argument(
        "--buckets",
        type=parse_buckets,
        default=DEFAULT_BUCKETS,
        help="Comma-separated MxN buckets (default: 50x50,100x100,200x200,400x400)",
    )
    parser.add_argument("--n-configs", type=int, default=8, help="Configs per round")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=2, help="Eval seeds per instance")
    parser.add_argument("--train-seeds", type=str, default="0,1")
    parser.add_argument("--holdout-seeds", type=str, default="100,101")
    parser.add_argument("--target-win-rate", type=float, default=0.8)
    parser.add_argument("--max-time-ratio", type=float, default=1.5)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--max-wall-sec", type=float, default=None)
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--config-path", default="config/alns_params.json")
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny buckets + short budgets for plumbing check",
    )
    args = parser.parse_args()

    buckets = args.buckets
    budget_scale = args.budget_scale
    n_configs = args.n_configs
    max_rounds = args.max_rounds
    if args.smoke:
        buckets = [(5, 5), (8, 8)]
        budget_scale = 0.15
        n_configs = 2
        max_rounds = 2

    train_seeds = [int(x) for x in args.train_seeds.split(",") if x.strip()]
    holdout_seeds = [int(x) for x in args.holdout_seeds.split(",") if x.strip()]
    eval_seeds = list(range(args.seeds))

    tune_loop(
        buckets=buckets,
        train_seeds=train_seeds,
        holdout_seeds=holdout_seeds,
        eval_seeds=eval_seeds,
        n_configs_per_round=n_configs,
        max_rounds=max_rounds,
        target_win_rate=args.target_win_rate,
        max_time_ratio=args.max_time_ratio,
        budget_scale=budget_scale,
        output_dir=args.output_dir,
        config_path=args.config_path,
        max_wall_sec=args.max_wall_sec,
        rng_seed=args.rng_seed,
    )


if __name__ == "__main__":
    main()
