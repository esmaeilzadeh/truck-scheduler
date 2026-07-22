"""ALNS hyperparameter tuning — offline, one-time (Section 6H)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
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
    # Cap destroy size so repairs stay cheap on large K (0 = uncapped)
    "q_cap": (4, 40),
}

_INT_PARAMS = {
    "segment_length",
    "regret_k",
    "sigma1",
    "sigma2",
    "sigma3",
    "q_cap",
}


def _sample_config(rng: random.Random) -> dict:
    cfg: dict = {}
    for k, (lo, hi) in PARAM_RANGES.items():
        if k in _INT_PARAMS:
            cfg[k] = rng.randint(int(lo), int(hi))
        else:
            cfg[k] = rng.uniform(lo, hi)
    if cfg["rho_min"] > cfg["rho_max"]:
        cfg["rho_min"], cfg["rho_max"] = cfg["rho_max"], cfg["rho_min"]
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


def _select_stratified(
    candidates: list[Instance],
    per_bucket: int = 1,
    max_total: int = 10,
) -> list[Instance]:
    """Pick across K buckets so small and large instances both appear."""
    by_k: dict[int, list[Instance]] = defaultdict(list)
    for inst in candidates:
        by_k[len(inst.ops)].append(inst)

    for k in by_k:
        by_k[k] = sorted(by_k[k], key=lambda i: (0 if i.G == 2 else 1, i.id))

    ks = sorted(by_k)
    if not ks:
        return []

    # Spread slots across the K range (include extremes)
    if len(ks) <= max_total:
        target_ks = ks
    else:
        # Always keep smallest + largest; sample middles evenly
        n_mid = max_total - 2
        mids = ks[1:-1]
        if n_mid <= 0:
            target_ks = [ks[0], ks[-1]][:max_total]
        else:
            step = max(1, len(mids) / n_mid)
            chosen_mid = [mids[int(i * step)] for i in range(n_mid)]
            # de-dupe while preserving order
            seen: set[int] = set()
            target_ks = []
            for k in [ks[0], *chosen_mid, ks[-1]]:
                if k not in seen:
                    seen.add(k)
                    target_ks.append(k)

    picked: list[Instance] = []
    for k in target_ks:
        for inst in by_k[k][:per_bucket]:
            picked.append(inst)
            if len(picked) >= max_total:
                return picked
    return picked


def load_tune_instances(
    tune_dir: str | Path = "data/instances/TUNE",
    max_total: int = 10,
    include_large: bool = True,
) -> list[Instance]:
    """Load stratified TUNE set; optionally add larger in-memory instances."""
    from src.instance_gen import gen_instance
    from src.io_utils import read_instance

    tune_dir = Path(tune_dir)
    candidates: list[Instance] = []
    if tune_dir.is_dir():
        for path in sorted(tune_dir.glob("*.json")):
            candidates.append(read_instance(path))

    # Supplement with mid/large K so θ transfers beyond K≤20
    if include_large:
        for seed in (0, 1):
            for M, N, G in ((20, 20, 2), (30, 30, 2)):
                inst = gen_instance(seed=seed, M=M, N=N, G=G)
                inst.id = f"TUNE_extra_s{seed}_M{M}_N{N}_G{G}"
                candidates.append(inst)

    if not candidates:
        for seed in range(4):
            for M, N in ((3, 3), (5, 5)):
                candidates.append(gen_instance(seed=seed, M=M, N=N, G=2))

    return _select_stratified(candidates, per_bucket=1, max_total=max_total)


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

    print(
        f"Tuning on {len(instances)} instances "
        f"(K={[len(i.ops) for i in instances]}), "
        f"{n_configs} configs × {seeds_per_config} seeds × {run_budget_sec}s"
    )

    # Exact refs where CP-SAT proves optimality within the same budget
    exact_refs: dict[str, float] = {}
    for inst in instances:
        ref = _get_ref(inst, cpsat, run_budget_sec)
        if ref is not None:
            exact_refs[inst.id] = ref
            print(f"  ref[exact] {inst.id} K={len(inst.ops)} obj={ref:.1f}")
        else:
            print(f"  ref[proxy] {inst.id} K={len(inst.ops)} (no proven opt)")

    # Random search — record every (trial, inst, seed) objective
    results: list[dict] = []
    # trial -> list of (inst_id, obj)
    trial_objs: dict[int, list[tuple[str, float]]] = defaultdict(list)
    best_known: dict[str, float] = dict(exact_refs)

    t0 = time.perf_counter()
    for trial in range(n_configs):
        cfg = _sample_config(rng)
        for inst in instances:
            for s in range(seeds_per_config):
                try:
                    alns = ALNS(params=cfg)
                    sol = alns.solve(inst, time_limit_sec=run_budget_sec, seed=s)
                    validate(inst, sol)
                    obj = sol.objective(inst)
                except Exception:
                    obj = float("inf")
                trial_objs[trial].append((inst.id, obj))
                if obj < float("inf"):
                    prev = best_known.get(inst.id, float("inf"))
                    if obj < prev:
                        best_known[inst.id] = obj

        results.append({"trial": trial, **cfg})

        if (trial + 1) % 5 == 0 or trial == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  trial {trial + 1}/{n_configs} "
                f"({elapsed:.0f}s elapsed)"
            )

    # Rescore with final best-known / exact refs (SPEC 6H.1)
    def ref_of(inst_id: str) -> float:
        return best_known.get(inst_id, 1.0)

    best_mean_gap = float("inf")
    best_cfg: dict = {}
    for row in results:
        trial = row["trial"]
        gaps = []
        for inst_id, obj in trial_objs[trial]:
            r = ref_of(inst_id)
            if r <= 0 or obj == float("inf"):
                gaps.append(1.0)
            else:
                gaps.append((obj - r) / r)
        mean_gap = sum(gaps) / len(gaps) if gaps else 1.0
        row["mean_gap"] = mean_gap
        if mean_gap < best_mean_gap:
            best_mean_gap = mean_gap
            best_cfg = {k: v for k, v in row.items() if k not in ("trial", "mean_gap")}

    # Stable column order for CSV
    csv_path = Path(output_dir) / "alns_tuning.csv"
    if results:
        fieldnames = ["trial", "mean_gap", *PARAM_RANGES.keys(), "max_iterations"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    with open(config_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
        f.write("\n")

    print(f"Best mean gap: {best_mean_gap:.4f}")
    print(f"Config written to {config_path}")
    print(f"Log written to {csv_path}")
    return best_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="ALNS hyperparameter random search (SPEC 6H)")
    parser.add_argument("--n-configs", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--max-instances", type=int, default=8)
    parser.add_argument("--no-large", action="store_true", help="Only on-disk TUNE (K≤20)")
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
    )
    print("Best config:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
