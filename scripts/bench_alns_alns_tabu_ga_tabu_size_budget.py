#!/usr/bin/env python3
"""Compare ALNS / ALNS+Tabu / GA+Tabu across size × wall-clock budget.

Generates 10 instances per square size (M=N), gates G cycling through 2..10,
runs the three solvers under multiple equal budgets, and appends long-form CSV
rows (resume-safe).

Default sizes: 15, 30, 50, 100, 150, 200.
Default budgets: 10s, 60s (1m), 120s (2m).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instance_gen import gen_instance
from src.solvers.alns import ALNS
from src.solvers.alns_tabu import HybridALNSTabu
from src.solvers.ga_tabu import HybridGATabu
from src.solvers.greedy import greedy_erd_spt
from src.validate import validate

DEFAULT_SIZES = (15, 30, 50, 100, 150, 200)
DEFAULT_BUDGETS = (10.0, 60.0, 120.0)  # includes 1m and 2m
INSTANCES_PER_SIZE = 10
G_MIN, G_MAX = 2, 10
SOLVER_SEED = 0
OUT_DEFAULT = ROOT / "data" / "results" / "alns_alns_tabu_ga_tabu_size_budget.csv"

FIELDNAMES = [
    "size_label",
    "M",
    "N",
    "K",
    "G",
    "T",
    "instance_id",
    "instance_index",
    "instance_seed",
    "budget_sec",
    "solver",
    "solver_seed",
    "objective",
    "runtime_sec",
    "greedy_obj",
    "gap_vs_greedy_pct",
    "iterations",
    "generations",
    "local_searches",
    "local_search_time_sec",
    "error",
]


def _gate_for_index(i: int) -> int:
    """Map instance index 0..9 onto G in {2,...,10} (cycles)."""
    span = G_MAX - G_MIN + 1  # 9
    return G_MIN + (i % span)


def build_instances(sizes: tuple[int, ...] | list[int]):
    """Yield (size_label, instance_index, inst) for the study grid."""
    for side in sizes:
        for i in range(INSTANCES_PER_SIZE):
            G = _gate_for_index(i)
            # Distinct seeds per (size, index); avoid colliding across sizes.
            seed = 10_000 * side + i
            inst = gen_instance(
                seed=seed,
                M=side,
                N=side,
                G=G,
                id_prefix="CMP",
                horizon_slack=2.0,
            )
            yield f"{side}x{side}", i, inst


def _base_row(payload: dict, inst, greedy_obj: float) -> dict:
    side = payload["side"]
    return {
        "size_label": payload["size_label"],
        "M": side,
        "N": side,
        "K": len(inst.ops),
        "G": inst.G,
        "T": inst.T,
        "instance_id": inst.id,
        "instance_index": payload["instance_index"],
        "instance_seed": payload["instance_seed"],
        "budget_sec": payload["budget_sec"],
        "solver_seed": payload["solver_seed"],
        "greedy_obj": greedy_obj,
        "error": "",
    }


def _solve_one(payload: dict) -> list[dict]:
    """Worker: run all three solvers on one (instance, budget)."""
    side = payload["side"]
    seed = payload["instance_seed"]
    G = payload["G"]
    budget = payload["budget_sec"]
    solver_seed = payload["solver_seed"]

    try:
        # Slightly looser horizon so high-G draws stay ALNS-repairable.
        inst = gen_instance(
            seed=seed,
            M=side,
            N=side,
            G=G,
            id_prefix="CMP",
            horizon_slack=2.0,
        )
        warm = greedy_erd_spt(inst)
        greedy_obj = warm.objective(inst)
    except Exception as exc:
        # Fatal instance build — emit three error rows with synthetic ids.
        err = f"{type(exc).__name__}: {exc}"
        rows = []
        for name in ("alns", "alns_tabu", "ga_tabu"):
            rows.append(
                {
                    "size_label": payload["size_label"],
                    "M": side,
                    "N": side,
                    "K": 2 * side,
                    "G": G,
                    "T": None,
                    "instance_id": payload.get("instance_id")
                    or f"CMP_s{seed}_M{side}_N{side}_G{G}",
                    "instance_index": payload["instance_index"],
                    "instance_seed": seed,
                    "budget_sec": budget,
                    "solver": name,
                    "solver_seed": solver_seed,
                    "objective": float("inf"),
                    "runtime_sec": 0.0,
                    "greedy_obj": float("inf"),
                    "gap_vs_greedy_pct": float("inf"),
                    "iterations": None,
                    "generations": None,
                    "local_searches": None,
                    "local_search_time_sec": None,
                    "error": err,
                }
            )
        return rows

    base = _base_row(payload, inst, greedy_obj)

    solvers = [
        ("alns", ALNS()),
        ("alns_tabu", HybridALNSTabu()),
        ("ga_tabu", HybridGATabu()),
    ]
    rows: list[dict] = []
    for name, solver in solvers:
        row = {
            **base,
            "solver": name,
            "objective": float("inf"),
            "runtime_sec": 0.0,
            "gap_vs_greedy_pct": float("inf"),
            "iterations": None,
            "generations": None,
            "local_searches": None,
            "local_search_time_sec": None,
            "error": "",
        }
        try:
            sol = solver.solve(
                inst,
                time_limit_sec=budget,
                seed=solver_seed,
                warm_start=warm,
            )
            validate(inst, sol)
            obj = sol.objective(inst)
            meta = sol.meta or {}
            gap = 0.0
            if greedy_obj > 0:
                gap = max(0.0, (obj - greedy_obj) / greedy_obj * 100.0)
            row.update(
                {
                    "objective": obj,
                    "runtime_sec": float(sol.runtime_sec),
                    "gap_vs_greedy_pct": gap,
                    "iterations": meta.get("iterations"),
                    "generations": meta.get("generations"),
                    "local_searches": meta.get("local_searches"),
                    "local_search_time_sec": meta.get("local_search_time_sec"),
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _load_done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    done: set[tuple] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(
                (
                    row["instance_id"],
                    str(float(row["budget_sec"])),
                    row["solver"],
                )
            )
    return done


def _append_rows(path: Path, rows: list[dict], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALNS vs ALNS+Tabu vs GA+Tabu size×budget comparison CSV"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Square sizes M=N (default: 15 30 50 100 150 200)",
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Wall budgets in seconds (default: 10 60 120)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Process pool size (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DEFAULT,
        help=f"CSV path (default: {OUT_DEFAULT})",
    )
    parser.add_argument(
        "--solver-seed",
        type=int,
        default=SOLVER_SEED,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs and exit",
    )
    args = parser.parse_args()

    sizes = tuple(args.sizes)
    budgets = tuple(float(b) for b in args.budgets)
    out_path: Path = args.output

    jobs: list[dict] = []
    for size_label, idx, inst in build_instances(sizes):
        side = int(size_label.split("x")[0])
        seed = 10_000 * side + idx
        G = _gate_for_index(idx)
        for budget in budgets:
            jobs.append(
                {
                    "side": side,
                    "size_label": size_label,
                    "instance_index": idx,
                    "instance_seed": seed,
                    "G": G,
                    "budget_sec": budget,
                    "solver_seed": args.solver_seed,
                    "instance_id": inst.id,
                }
            )

    done = _load_done_keys(out_path)
    pending = []
    for job in jobs:
        # A job is done if all three solvers already present
        keys = [
            (job["instance_id"], str(float(job["budget_sec"])), s)
            for s in ("alns", "alns_tabu", "ga_tabu")
        ]
        if all(k in done for k in keys):
            continue
        pending.append(job)

    n_inst = len(sizes) * INSTANCES_PER_SIZE
    print(
        f"sizes={list(sizes)} budgets={list(budgets)} "
        f"instances={n_inst} jobs_total={len(jobs)} "
        f"jobs_pending={len(pending)} workers={args.workers}"
    )
    print(f"output={out_path}")
    if args.dry_run:
        for j in pending[:12]:
            print(
                f"  {j['size_label']} idx={j['instance_index']} "
                f"G={j['G']} budget={j['budget_sec']}s id={j['instance_id']}"
            )
        if len(pending) > 12:
            print(f"  ... +{len(pending) - 12} more")
        return

    if not pending:
        print("Nothing to do; CSV already complete.")
        return

    write_header = not out_path.is_file() or out_path.stat().st_size == 0
    t0 = time.perf_counter()
    finished = 0

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_solve_one, job): job for job in pending}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:
                print(
                    f"FAIL {job['instance_id']} budget={job['budget_sec']}: {exc}",
                    flush=True,
                )
                continue
            _append_rows(out_path, rows, write_header=write_header)
            write_header = False
            finished += 1
            objs = {r["solver"]: r["objective"] for r in rows}
            print(
                f"[{finished}/{len(pending)}] {job['size_label']} "
                f"G={job['G']} budget={job['budget_sec']}s "
                f"alns={objs.get('alns')} "
                f"alns_tabu={objs.get('alns_tabu')} "
                f"ga_tabu={objs.get('ga_tabu')} "
                f"({time.perf_counter() - t0:.0f}s elapsed)",
                flush=True,
            )

    print(f"Done. Wrote {out_path} ({time.perf_counter() - t0:.0f}s wall)")


if __name__ == "__main__":
    main()
