"""Shared offline random-search hyperparameter tuning (SPEC 6H-style).

Algorithm-specific wrappers (ALNS, GA-Tabu, …) supply a ``TunerSpec`` and
delegate sampling, instance loading, scoring, and IO here.
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.model import Instance
from src.solvers.cpsat import CPSAT
from src.validate import validate


class SolverLike(Protocol):
    def solve(
        self,
        inst: Instance,
        *,
        time_limit_sec: float | None = None,
        seed: int | None = None,
        warm_start: Any = None,
    ) -> Any: ...


@dataclass
class TunerSpec:
    """Describes one algorithm's searchable hyperparameter space."""

    name: str
    default_params: dict
    param_ranges: dict[str, tuple[float, float]]
    int_params: set[str]
    make_solver: Callable[[dict], SolverLike]
    light_param_keys: tuple[str, ...] = ()
    # Extra keys always copied from defaults into a sampled config
    frozen_keys: tuple[str, ...] = ()
    normalize: Callable[[dict], dict] | None = None
    csv_filename: str = "tuning.csv"
    config_path: str = "config/params.json"


def sample_one(
    rng: random.Random,
    key: str,
    ranges: dict[str, tuple[float, float]],
    int_params: set[str],
) -> float | int:
    lo, hi = ranges[key]
    if key in int_params:
        return rng.randint(int(lo), int(hi))
    return rng.uniform(float(lo), float(hi))


def sample_config(
    rng: random.Random,
    spec: TunerSpec,
    *,
    light: bool = False,
) -> dict:
    """Sample a config from ``spec.param_ranges`` (or light subset)."""
    if light and spec.light_param_keys:
        cfg = dict(spec.default_params)
        for key in spec.light_param_keys:
            cfg[key] = sample_one(rng, key, spec.param_ranges, spec.int_params)
    else:
        cfg = {
            k: sample_one(rng, k, spec.param_ranges, spec.int_params)
            for k in spec.param_ranges
        }
        for key in spec.frozen_keys:
            if key in spec.default_params:
                cfg[key] = spec.default_params[key]

    if spec.normalize is not None:
        cfg = spec.normalize(cfg)
    return cfg


def select_stratified(
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

    if len(ks) <= max_total:
        target_ks = ks
    else:
        n_mid = max_total - 2
        mids = ks[1:-1]
        if n_mid <= 0:
            target_ks = [ks[0], ks[-1]][:max_total]
        else:
            step = max(1, len(mids) / n_mid)
            chosen_mid = [mids[int(i * step)] for i in range(n_mid)]
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

    return select_stratified(candidates, per_bucket=1, max_total=max_total)


def get_cpsat_ref(inst: Instance, cpsat: CPSAT, budget: float) -> float | None:
    """Try to get CP-SAT optimum as reference. Returns None if not proven."""
    try:
        sol = cpsat.solve(inst, time_limit_sec=budget)
        if sol.proven_optimal:
            return sol.objective(inst)
    except Exception:
        pass
    return None


def run_random_search(
    spec: TunerSpec,
    instances: list[Instance],
    *,
    n_configs: int = 200,
    seeds_per_config: int = 3,
    run_budget_sec: float = 5.0,
    output_dir: str | Path = "data/results",
    config_path: str | Path | None = None,
    light: bool = False,
    rng_seed: int = 42,
) -> dict:
    """Random-search tuning. Writes CSV + best JSON; returns best config."""
    rng = random.Random(rng_seed)
    cpsat = CPSAT()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_config = Path(config_path) if config_path is not None else Path(spec.config_path)
    out_config.parent.mkdir(parents=True, exist_ok=True)

    mode = (
        f"light ({', '.join(spec.light_param_keys)})"
        if light and spec.light_param_keys
        else f"full {spec.name} ranges"
    )
    print(
        f"Tuning {spec.name} ({mode}) on {len(instances)} instances "
        f"(K={[len(i.ops) for i in instances]}), "
        f"{n_configs} configs × {seeds_per_config} seeds × {run_budget_sec}s"
    )

    exact_refs: dict[str, float] = {}
    for inst in instances:
        ref = get_cpsat_ref(inst, cpsat, run_budget_sec)
        if ref is not None:
            exact_refs[inst.id] = ref
            print(f"  ref[exact] {inst.id} K={len(inst.ops)} obj={ref:.1f}")
        else:
            print(f"  ref[proxy] {inst.id} K={len(inst.ops)} (no proven opt)")

    results: list[dict] = []
    trial_objs: dict[int, list[tuple[str, float]]] = defaultdict(list)
    best_known: dict[str, float] = dict(exact_refs)

    t0 = time.perf_counter()
    for trial in range(n_configs):
        cfg = sample_config(rng, spec, light=light)
        solver = spec.make_solver(cfg)
        for inst in instances:
            for s in range(seeds_per_config):
                try:
                    sol = solver.solve(inst, time_limit_sec=run_budget_sec, seed=s)
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
            best_cfg = {
                k: v for k, v in row.items() if k not in ("trial", "mean_gap")
            }

    csv_path = output_dir / spec.csv_filename
    if results:
        if light and spec.light_param_keys:
            param_keys = list(spec.default_params.keys())
        else:
            param_keys = [
                *spec.param_ranges.keys(),
                *[k for k in spec.frozen_keys if k not in spec.param_ranges],
            ]
        fieldnames = ["trial", "mean_gap", *param_keys]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    with open(out_config, "w") as f:
        json.dump(best_cfg, f, indent=2)
        f.write("\n")

    print(f"Best mean gap: {best_mean_gap:.4f}")
    print(f"Config written to {out_config}")
    print(f"Log written to {csv_path}")
    return best_cfg
