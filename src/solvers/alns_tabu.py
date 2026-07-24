"""Hybrid ALNS + Tabu — ALNS exploration with gated Tabu intensification."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.model import Instance, Solution, feasibility_precheck
from src.solvers.alns import (
    DEFAULT_PARAMS as ALNS_DEFAULT_PARAMS,
    DESTROY_OPS,
    REPAIR_OPS,
    _ScheduleState,
    _deadline_hit,
    decode,
)
from src.solvers.greedy import greedy_erd_spt
from src.solvers.tabu import improve_order

_DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[2] / "config" / "alns_tabu_params.json"

# ALNS literature defaults + light Tabu local-search knobs (SPEC 4B.4)
DEFAULT_PARAMS: dict = {
    **ALNS_DEFAULT_PARAMS,
    "local_search_rate": 0.15,
    "local_tabu_iters": 40,
    "local_neighborhood_size": 20,
    "tabu_tenure": 7,
    "swap_prob": 0.5,
    # Cap cumulative Tabu wall-time as a fraction of time_limit_sec
    "local_search_budget_frac": 0.15,
}


def _load_params_file(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@dataclass
class HybridALNSTabu:
    """Memetic hybrid: schedule-based ALNS outer loop + gated Tabu polish."""

    name: str = "alns_tabu"
    params: dict | None = None
    params_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.params_path is None:
            self.params_path = _DEFAULT_PARAMS_PATH

    def solve(
        self,
        inst: Instance,
        *,
        time_limit_sec: float | None = None,
        seed: int | None = None,
        warm_start: Solution | None = None,
    ) -> Solution:
        feasibility_precheck(inst)
        t0 = time.perf_counter()
        deadline = (
            t0 + time_limit_sec if time_limit_sec is not None else None
        )

        file_params = _load_params_file(
            Path(self.params_path) if self.params_path else None
        )
        p = {**DEFAULT_PARAMS, **file_params, **(self.params or {})}
        rng = random.Random(seed if seed is not None else 0)
        K = len(inst.ops)

        greedy_sol = greedy_erd_spt(inst)
        if warm_start is not None:
            init_order = sorted(
                warm_start.starts.keys(),
                key=lambda u: (warm_start.starts[u], warm_start.gates[u]),
            )
            if len(init_order) != K:
                init_order = [
                    op.uid
                    for op in sorted(
                        inst.ops, key=lambda op: (op.r, op.p, -op.w, op.uid)
                    )
                ]
            greedy_ref = warm_start
        else:
            init_order = [
                op.uid
                for op in sorted(inst.ops, key=lambda op: (op.r, op.p, -op.w, op.uid))
            ]
            greedy_ref = greedy_sol

        cur_state = _ScheduleState.from_order(inst, init_order)
        best_order = list(init_order)
        obj_cur = cur_state.objective
        obj_best = obj_cur
        obj_greedy = greedy_ref.objective(inst)

        temp0 = (
            -(p["start_temp_ctrl"] * obj_cur) / math.log(0.5) if obj_cur > 0 else 1.0
        )
        temp = temp0
        final_temp_ratio = float(p.get("final_temp_ratio", 0.002))

        local_rate = float(p["local_search_rate"])
        local_iters = max(1, int(p["local_tabu_iters"]))
        local_nhood = max(1, int(p["local_neighborhood_size"]))
        tenure = int(p["tabu_tenure"])
        swap_prob = float(p["swap_prob"])
        polish_frac = max(
            0.0, min(1.0, float(p.get("local_search_budget_frac", 0.15)))
        )
        if time_limit_sec is not None and time_limit_sec > 0:
            polish_budget_total = polish_frac * float(time_limit_sec)
            polish_remaining = polish_budget_total
        else:
            polish_budget_total = None
            polish_remaining = float("inf")

        n_destroy = len(DESTROY_OPS)
        n_repair = len(REPAIR_OPS)
        w_destroy = [1.0] * n_destroy
        w_repair = [1.0] * n_repair
        score_destroy = [0.0] * n_destroy
        score_repair = [0.0] * n_repair
        use_destroy = [0] * n_destroy
        use_repair = [0] * n_repair

        q_min = max(1, math.ceil(p["rho_min"] * K))
        q_max = max(q_min, math.ceil(p["rho_max"] * K))
        q_cap = int(p.get("q_cap", 0) or 0)
        if q_cap > 0:
            q_min = min(q_min, q_cap)
            q_max = min(q_max, q_cap)
            q_max = max(q_min, q_max)

        max_iter = int(p["max_iterations"])
        iterations = 0
        local_searches = 0
        local_search_time = 0.0

        for it in range(max_iter):
            if _deadline_hit(deadline):
                break

            iterations = it + 1

            d_idx = rng.choices(range(n_destroy), weights=w_destroy, k=1)[0]
            r_idx = rng.choices(range(n_repair), weights=w_repair, k=1)[0]
            q = rng.randint(q_min, q_max)

            cand = cur_state.copy()
            destroy_kwargs = {"d_wr": p["d_wr"]} if d_idx == 1 else {}
            removed = DESTROY_OPS[d_idx](cand, q, rng, **destroy_kwargs)

            if removed:
                cand = _ScheduleState.from_solution(inst, decode(inst, cand.order()))

            if _deadline_hit(deadline):
                break

            repair_kwargs: dict = {"deadline": deadline}
            if r_idx == 1:
                repair_kwargs["k"] = p["regret_k"]
            REPAIR_OPS[r_idx](cand, removed, rng, **repair_kwargs)

            compacted = decode(inst, cand.order())
            cand = _ScheduleState.from_solution(inst, compacted)
            obj_new = cand.objective

            delta_obj = obj_new - obj_cur
            accept = False
            if delta_obj <= 0:
                accept = True
            elif temp > 0:
                if rng.random() < math.exp(-delta_obj / temp):
                    accept = True

            psi = 0
            if accept:
                new_global_best = obj_new < obj_best
                if new_global_best:
                    psi = p["sigma1"]
                elif obj_new < obj_cur:
                    psi = p["sigma2"]
                else:
                    psi = p["sigma3"]

                cur_state = cand
                obj_cur = obj_new
                if new_global_best:
                    obj_best = obj_new
                    best_order = cand.order()

                # Gated Tabu polish: always on new best; else with local_search_rate.
                # Skip when cumulative polish wall-time budget is exhausted.
                do_tabu = new_global_best or (rng.random() < local_rate)
                if (
                    do_tabu
                    and not _deadline_hit(deadline)
                    and polish_remaining > 1e-12
                ):
                    polish_t0 = time.perf_counter()
                    if polish_remaining < float("inf"):
                        polish_deadline = polish_t0 + polish_remaining
                        if deadline is not None:
                            polish_deadline = min(deadline, polish_deadline)
                    else:
                        polish_deadline = deadline
                    polished, polished_obj, _ = improve_order(
                        inst,
                        cur_state.order(),
                        max_iters=local_iters,
                        neighborhood_size=local_nhood,
                        tabu_tenure=tenure,
                        swap_prob=swap_prob,
                        rng=rng,
                        deadline=polish_deadline,
                    )
                    spent = time.perf_counter() - polish_t0
                    local_search_time += spent
                    if polish_remaining < float("inf"):
                        polish_remaining = max(0.0, polish_remaining - spent)
                    local_searches += 1
                    if polished_obj < obj_cur - 1e-12:
                        cur_state = _ScheduleState.from_order(inst, polished)
                        obj_cur = cur_state.objective
                        if obj_cur < obj_best:
                            obj_best = obj_cur
                            best_order = list(polished)
                            psi = max(psi, p["sigma1"])

                score_destroy[d_idx] += psi
                score_repair[r_idx] += psi

            use_destroy[d_idx] += 1
            use_repair[r_idx] += 1

            if time_limit_sec is not None and time_limit_sec > 0:
                elapsed = time.perf_counter() - t0
                frac = min(1.0, max(0.0, elapsed / time_limit_sec))
                temp = temp0 * (final_temp_ratio ** frac)
            else:
                temp *= p["cooling"]

            if (it + 1) % int(p["segment_length"]) == 0:
                lam = p["lambda"]
                for i in range(n_destroy):
                    if use_destroy[i] > 0:
                        w_destroy[i] = (1 - lam) * w_destroy[i] + lam * (
                            score_destroy[i] / use_destroy[i]
                        )
                    score_destroy[i] = 0.0
                    use_destroy[i] = 0
                for i in range(n_repair):
                    if use_repair[i] > 0:
                        w_repair[i] = (1 - lam) * w_repair[i] + lam * (
                            score_repair[i] / use_repair[i]
                        )
                    score_repair[i] = 0.0
                    use_repair[i] = 0

        runtime = time.perf_counter() - t0
        sol = decode(inst, best_order)

        if obj_best > obj_greedy:
            sol = Solution(
                starts=dict(greedy_ref.starts),
                gates=dict(greedy_ref.gates),
            )
            obj_best = obj_greedy

        gap_pct = 0.0
        if obj_greedy > 0:
            gap_pct = max(0.0, (obj_best - obj_greedy) / obj_greedy * 100)

        sol.proven_optimal = False
        sol.runtime_sec = runtime
        sol.meta = {
            "iterations": iterations,
            "local_searches": local_searches,
            "local_search_time_sec": local_search_time,
            "local_search_budget_sec": polish_budget_total,
            "gap_pct": gap_pct,
        }
        return sol
