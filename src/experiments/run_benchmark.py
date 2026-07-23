"""Benchmark runner — compares tiers on TEST suite (Section 9)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from src.model import Instance, Solution
from src.solvers.alns import ALNS
from src.solvers.alns_tabu import HybridALNSTabu
from src.solvers.base import Solver
from src.solvers.cpsat import CPSAT
from src.solvers.ga import GeneticAlgorithm
from src.solvers.ga_tabu import HybridGATabu
from src.solvers.greedy import GreedyERDSPT
from src.solvers.tabu import TabuSearch
from src.validate import validate


def run_benchmark(
    instances: list[Instance],
    solvers: list[Solver] | None = None,
    time_limit_sec: float = 5.0,
    output_dir: str = "data/results",
    seed: int = 0,
) -> dict:
    """Run all solvers on all instances. Returns results dict."""
    if solvers is None:
        solvers = [
            GreedyERDSPT(),
            ALNS(),
            HybridALNSTabu(),
            TabuSearch(),
            GeneticAlgorithm(),
            HybridGATabu(),
            CPSAT(),
        ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for inst in instances:
        row = {"inst_id": inst.id, "K": len(inst.ops), "G": inst.G, "T": inst.T}

        for solver in solvers:
            try:
                t0 = time.perf_counter()
                sol = solver.solve(inst, time_limit_sec=time_limit_sec, seed=seed)
                elapsed = time.perf_counter() - t0
                validate(inst, sol)
                obj = sol.objective(inst)
                row[f"{solver.name}_obj"] = obj
                row[f"{solver.name}_time"] = elapsed
                row[f"{solver.name}_optimal"] = getattr(sol, "proven_optimal", False)
            except Exception as e:
                row[f"{solver.name}_obj"] = float("inf")
                row[f"{solver.name}_time"] = time_limit_sec
                row[f"{solver.name}_optimal"] = False
                row[f"{solver.name}_error"] = str(e)

        # Compute gaps vs CP-SAT optimum
        opt = row.get("cpsat_obj", float("inf"))
        if opt < float("inf") and opt > 0:
            for s in solvers:
                obj = row.get(f"{s.name}_obj", float("inf"))
                if obj < float("inf"):
                    row[f"{s.name}_gap_pct"] = round((obj - opt) / opt * 100, 2)
                else:
                    row[f"{s.name}_gap_pct"] = float("inf")

        rows.append(row)

    # Write CSV
    csv_path = Path(output_dir) / "benchmark_results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Crossover plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for s in solvers:
            times = [r.get(f"{s.name}_time", 0) for r in rows]
            ks = [r["K"] for r in rows]
            # Group by K
            from collections import defaultdict

            by_k = defaultdict(list)
            for k, t in zip(ks, times):
                by_k[k].append(t)
            avg_k = sorted(by_k.keys())
            avg_t = [sum(by_k[k]) / len(by_k[k]) for k in avg_k]
            ax.plot(avg_k, avg_t, "o-", label=s.name)

        ax.set_xlabel("K (M + N)")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Crossover: Runtime vs Instance Size")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "crossover_benchmark.png", dpi=150)
        plt.close()

    except Exception as e:
        print(f"Could not generate plot: {e}")

    # Summary table
    summary = {}
    for s in solvers:
        objs = [r.get(f"{s.name}_obj", float("inf")) for r in rows if r.get(f"{s.name}_obj", float("inf")) < float("inf")]
        times = [r.get(f"{s.name}_time", 0) for r in rows]
        summary[s.name] = {
            "mean_obj": sum(objs) / len(objs) if objs else float("inf"),
            "mean_time": sum(times) / len(times) if times else 0,
            "solved": len(objs),
        }

    summary_path = Path(output_dir) / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Benchmark complete. Results:", summary)
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    from src.io_utils import read_instance
    from src.instance_gen import gen_instance

    test_dir = Path("data/instances/TEST")
    instances: list[Instance] = []
    if test_dir.is_dir():
        for path in sorted(test_dir.glob("*.json"))[:10]:
            instances.append(read_instance(path))
    if not instances:
        for seed in range(3):
            for M, N in [(3, 3), (5, 5), (8, 7)]:
                instances.append(gen_instance(seed=seed, M=M, N=N, G=2))

    run_benchmark(instances, time_limit_sec=3.0)
