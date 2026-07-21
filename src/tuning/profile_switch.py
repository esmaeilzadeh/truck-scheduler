"""Tier-switch profiling — offline, one-time (Section 7)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.model import Instance
from src.solvers.alns import ALNS
from src.solvers.cpsat import CPSAT
from src.solvers.greedy import GreedyERDSPT
from src.validate import validate


def profile_switch(
    instances: list[Instance],
    budget_sec: float = 5.0,
    cpsat_cap_sec: float = 300.0,
    output_dir: str = "data/results",
    policy_path: str = "config/switch_policy.json",
) -> dict:
    """Profile CP-SAT and ALNS across instance sizes.

    Returns the switch policy dict and saves it + crossover plot.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(policy_path).parent.mkdir(parents=True, exist_ok=True)

    cpsat = CPSAT()
    greedy_solver = GreedyERDSPT()
    alns = ALNS()

    # Group by K
    from collections import defaultdict

    by_k: dict[int, list[dict]] = defaultdict(list)

    for inst in instances:
        K = len(inst.ops)
        row = {"inst_id": inst.id, "K": K, "T": inst.T}

        # CP-SAT
        try:
            t0 = time.perf_counter()
            sol = cpsat.solve(inst, time_limit_sec=cpsat_cap_sec)
            elapsed = time.perf_counter() - t0
            row["cpsat_time"] = elapsed
            row["cpsat_optimal"] = sol.proven_optimal
            row["cpsat_obj"] = sol.objective(inst)
        except Exception as e:
            row["cpsat_time"] = cpsat_cap_sec
            row["cpsat_optimal"] = False
            row["cpsat_obj"] = float("inf")

        # ALNS
        try:
            t0 = time.perf_counter()
            sol_a = alns.solve(inst, time_limit_sec=budget_sec, seed=0)
            elapsed = time.perf_counter() - t0
            validate(inst, sol_a)
            row["alns_time"] = elapsed
            row["alns_obj"] = sol_a.objective(inst)
        except Exception as e:
            row["alns_time"] = budget_sec
            row["alns_obj"] = float("inf")

        # Greedy
        try:
            sol_g = greedy_solver.solve(inst)
            validate(inst, sol_g)
            row["greedy_obj"] = sol_g.objective(inst)
        except Exception:
            row["greedy_obj"] = float("inf")

        by_k[K].append(row)

    # Determine threshold tau
    import numpy as np

    ks = sorted(by_k.keys())
    tau = None
    for K in ks:
        rows = by_k[K]
        times = [r["cpsat_time"] for r in rows if r["cpsat_time"] < cpsat_cap_sec]
        optimal = [r.get("cpsat_optimal", False) for r in rows]
        if not times:
            continue
        p95 = float(np.percentile(times, 95))
        all_opt = all(optimal)
        if p95 <= budget_sec and all_opt:
            tau = K

    if tau is None:
        tau = 20  # safe fallback

    safety_margin = 1
    tau_deploy = max(5, tau - safety_margin)

    policy = {
        "budget_sec": budget_sec,
        "threshold_K": tau_deploy,
        "T_cap": None,
        "safety_margin_steps": safety_margin,
        "notes": f"P95 CP-SAT time <= {budget_sec}s up to K={tau}; deployed {tau_deploy} with {safety_margin}-step margin",
    }

    with open(policy_path, "w") as f:
        json.dump(policy, f, indent=2)

    # Crossover plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        cpsat_times = []
        alns_times = []
        k_vals = []

        for K in ks:
            rows = by_k[K]
            ct = [r["cpsat_time"] for r in rows]
            at = [r["alns_time"] for r in rows]
            if ct and at:
                k_vals.append(K)
                cpsat_times.append(sum(ct) / len(ct))
                alns_times.append(sum(at) / len(at))

        ax1.plot(k_vals, cpsat_times, "o-", label="CP-SAT (avg time)")
        ax1.plot(k_vals, alns_times, "s-", label="ALNS (avg time)")
        ax1.axhline(y=budget_sec, color="r", linestyle="--", label=f"Budget = {budget_sec}s")
        ax1.set_xlabel("K (M + N)")
        ax1.set_ylabel("Runtime (s)")
        ax1.set_title("Runtime vs Instance Size")
        ax1.legend()
        ax1.set_yscale("log")

        # Quality plot
        cpsat_objs = []
        alns_objs = []
        for K in ks:
            rows = by_k[K]
            co = [r["cpsat_obj"] for r in rows if r["cpsat_obj"] < float("inf")]
            ao = [r["alns_obj"] for r in rows if r["alns_obj"] < float("inf")]
            if co and ao:
                cpsat_objs.append(sum(co) / len(co))
                alns_objs.append(sum(ao) / len(ao))

        ax2.plot(k_vals, cpsat_objs, "o-", label="CP-SAT (avg obj)")
        ax2.plot(k_vals, alns_objs, "s-", label="ALNS (avg obj)")
        ax2.set_xlabel("K (M + N)")
        ax2.set_ylabel("Objective")
        ax2.set_title("Objective vs Instance Size")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "crossover_plot.png", dpi=150)
        plt.close()

    except Exception as e:
        print(f"Could not generate plot: {e}")

    print(f"Switch policy: threshold_K={tau_deploy}, budget={budget_sec}s")
    print(f"Policy written to {policy_path}")
    return policy


if __name__ == "__main__":
    from src.instance_gen import gen_instance

    instances = []
    for seed in range(5):
        for M, N in [(3, 3), (5, 5), (10, 10), (15, 15), (20, 20)]:
            inst = gen_instance(seed=seed, M=M, N=N, G=2)
            instances.append(inst)

    profile_switch(instances, budget_sec=3.0, cpsat_cap_sec=30.0)
