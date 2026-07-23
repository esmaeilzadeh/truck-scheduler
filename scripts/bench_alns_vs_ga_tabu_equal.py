#!/usr/bin/env python3
"""Equal-budget ALNS vs GA-Tabu bake-off (schedule-based ALNS check).

Same wall budget for both solvers. PROFILE M15/M30 + generated 50x50/100x100,
solver seeds 0–1. Writes JSON under data/results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instance_gen import gen_instance
from src.io_utils import read_instance
from src.solvers.alns import ALNS
from src.solvers.ga_tabu import HybridGATabu
from src.solvers.greedy import greedy_erd_spt
from src.validate import validate

BUDGET_SEC = 10.0
SEEDS = [0, 1]
OUT_PATH = ROOT / "data" / "results" / "alns_vs_ga_tabu_equal_budget.json"


def _load_cases():
    profile = ROOT / "data" / "instances" / "PROFILE"
    cases = [
        ("PROFILE_M15", read_instance(profile / "PROFILE_s200_M15_N15_G2.json")),
        ("PROFILE_M30", read_instance(profile / "PROFILE_s200_M30_N30_G2.json")),
    ]
    for m, n in ((50, 50), (100, 100)):
        inst = gen_instance(seed=0, M=m, N=n, G=2)
        inst.id = f"GEN_M{m}_N{n}_G2"
        cases.append((f"GEN_{m}x{n}", inst))
    return cases


def main() -> None:
    cases = _load_cases()
    alns = ALNS()
    hybrid = HybridGATabu()
    rows: list[dict] = []

    for label, inst in cases:
        warm = greedy_erd_spt(inst)
        for seed in SEEDS:
            print(f"=== {label} K={len(inst.ops)} seed={seed} budget={BUDGET_SEC}s ===")
            a = alns.solve(
                inst, time_limit_sec=BUDGET_SEC, seed=seed, warm_start=warm,
            )
            validate(inst, a)
            h = hybrid.solve(
                inst, time_limit_sec=BUDGET_SEC, seed=seed, warm_start=warm,
            )
            validate(inst, h)
            a_obj = a.objective(inst)
            h_obj = h.objective(inst)
            row = {
                "case": label,
                "instance_id": inst.id,
                "K": len(inst.ops),
                "seed": seed,
                "budget_sec": BUDGET_SEC,
                "alns_obj": a_obj,
                "hybrid_obj": h_obj,
                "alns_time": float(a.runtime_sec),
                "hybrid_time": float(h.runtime_sec),
                "alns_iters": (a.meta or {}).get("iterations"),
                "alns_won": a_obj < h_obj - 1e-9,
                "alns_le_hybrid": a_obj <= h_obj + 1e-9,
                "obj_gap_pct": (
                    (a_obj - h_obj) / h_obj * 100.0 if h_obj > 0 else 0.0
                ),
            }
            rows.append(row)
            print(
                f"  ALNS obj={a_obj:.4f} iters={row['alns_iters']} "
                f"t={row['alns_time']:.2f}s | "
                f"Hybrid obj={h_obj:.4f} t={row['hybrid_time']:.2f}s | "
                f"ALNS<=Hybrid={row['alns_le_hybrid']}"
            )

    n = len(rows)
    n_le = sum(1 for r in rows if r["alns_le_hybrid"])
    n_win = sum(1 for r in rows if r["alns_won"])
    summary = {
        "budget_sec": BUDGET_SEC,
        "n_runs": n,
        "alns_le_hybrid_count": n_le,
        "alns_le_hybrid_rate": n_le / n if n else 0.0,
        "alns_strict_win_count": n_win,
        "alns_strict_win_rate": n_win / n if n else 0.0,
        "median_alns_iters": sorted(r["alns_iters"] or 0 for r in rows)[n // 2],
        "runs": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")
    print(
        f"ALNS <= Hybrid on {n_le}/{n} "
        f"({summary['alns_le_hybrid_rate']:.0%}); "
        f"median ALNS iters={summary['median_alns_iters']}"
    )


if __name__ == "__main__":
    main()
