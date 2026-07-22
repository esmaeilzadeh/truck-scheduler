"""Tabu Search metaheuristic — permutation neighborhood over ALNS decode."""

from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from src.model import Infeasible, Instance, Solution, feasibility_precheck
from src.solvers.alns import _objective_from_order, decode
from src.solvers.greedy import greedy_erd_spt

_DEFAULT_PARAMS_PATH = Path("config/tabu_params.json")

DEFAULT_PARAMS: dict = {
    "tabu_tenure": 7,
    "neighborhood_size": 40,
    "max_iterations": 20000,
    "swap_prob": 0.5,
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


def _erd_spt_order(inst: Instance) -> list[int]:
    return [
        op.uid
        for op in sorted(inst.ops, key=lambda op: (op.r, op.p, -op.w, op.uid))
    ]


def _apply_swap(order: list[int], i: int, j: int) -> list[int]:
    out = list(order)
    out[i], out[j] = out[j], out[i]
    return out


def _apply_relocate(order: list[int], i: int, j: int) -> list[int]:
    """Move item at index i to index j."""
    out = list(order)
    uid = out.pop(i)
    if j > i:
        j -= 1
    out.insert(j, uid)
    return out


def _deadline_hit(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _safe_objective(inst: Instance, order: list[int]) -> float:
    try:
        return _objective_from_order(inst, order)
    except Infeasible:
        return float("inf")


def improve_order(
    inst: Instance,
    order: list[int],
    *,
    max_iters: int,
    neighborhood_size: int,
    tabu_tenure: int,
    swap_prob: float,
    rng: random.Random,
    deadline: float | None = None,
) -> tuple[list[int], float, int]:
    """Run Tabu local search from ``order``.

    Returns (best_order, best_obj, iterations_run).
    """
    K = len(order)
    cur_order = list(order)
    best_order = list(order)
    obj_cur = _safe_objective(inst, cur_order)
    obj_best = obj_cur

    tenure = max(1, int(tabu_tenure))
    n_size = max(1, int(neighborhood_size))
    max_iter = max(0, int(max_iters))
    swap_p = float(swap_prob)

    tabu: deque[tuple[tuple, int]] = deque()
    tabu_set: set[tuple] = set()

    iterations = 0
    for it in range(max_iter):
        if _deadline_hit(deadline):
            break
        iterations = it + 1

        while tabu and tabu[0][1] <= it:
            key, _ = tabu.popleft()
            tabu_set.discard(key)

        best_cand_order: list[int] | None = None
        best_cand_obj = float("inf")
        best_cand_key: tuple | None = None

        for _ in range(n_size):
            if K < 2:
                break
            if rng.random() < swap_p:
                i, j = rng.sample(range(K), 2)
                if i > j:
                    i, j = j, i
                cand = _apply_swap(cur_order, i, j)
                move_key: tuple = ("swap", cur_order[i], cur_order[j])
            else:
                i = rng.randrange(K)
                j = rng.randrange(K)
                if i == j:
                    continue
                cand = _apply_relocate(cur_order, i, j)
                move_key = ("relocate", cur_order[i], i, j)

            obj = _safe_objective(inst, cand)
            if obj == float("inf"):
                continue
            is_tabu = move_key in tabu_set
            aspirates = obj < obj_best
            if is_tabu and not aspirates:
                continue
            if obj < best_cand_obj:
                best_cand_obj = obj
                best_cand_order = cand
                best_cand_key = move_key

        if best_cand_order is None:
            diversified = False
            if K >= 2:
                for _ in range(min(20, n_size)):
                    i, j = rng.sample(range(K), 2)
                    cand = _apply_swap(cur_order, i, j)
                    obj = _safe_objective(inst, cand)
                    if obj < float("inf"):
                        best_cand_order = cand
                        best_cand_obj = obj
                        best_cand_key = ("swap", cur_order[i], cur_order[j])
                        diversified = True
                        break
            if not diversified:
                break

        cur_order = best_cand_order
        obj_cur = best_cand_obj

        if best_cand_key is not None and best_cand_key not in tabu_set:
            tabu.append((best_cand_key, it + tenure))
            tabu_set.add(best_cand_key)

        if obj_cur < obj_best:
            obj_best = obj_cur
            best_order = list(cur_order)

    return best_order, obj_best, iterations


@dataclass
class TabuSearch:
    """Tabu Search solver implementing the Solver protocol."""

    name: str = "tabu"
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
        deadline = t0 + time_limit_sec if time_limit_sec is not None else None

        file_params = _load_params_file(
            Path(self.params_path) if self.params_path else None
        )
        p = {**DEFAULT_PARAMS, **file_params, **(self.params or {})}
        rng = random.Random(seed if seed is not None else 0)

        greedy_sol = warm_start if warm_start is not None else greedy_erd_spt(inst)
        obj_greedy = greedy_sol.objective(inst)

        init_order = _erd_spt_order(inst)
        best_order, obj_best, iterations = improve_order(
            inst,
            init_order,
            max_iters=int(p["max_iterations"]),
            neighborhood_size=int(p["neighborhood_size"]),
            tabu_tenure=int(p["tabu_tenure"]),
            swap_prob=float(p["swap_prob"]),
            rng=rng,
            deadline=deadline,
        )

        runtime = time.perf_counter() - t0
        if obj_best == float("inf"):
            sol = Solution(
                starts=dict(greedy_sol.starts),
                gates=dict(greedy_sol.gates),
            )
            obj_best = obj_greedy
        else:
            sol = decode(inst, best_order)

        if obj_best > obj_greedy:
            sol = Solution(
                starts=dict(greedy_sol.starts),
                gates=dict(greedy_sol.gates),
            )
            obj_best = obj_greedy

        gap_pct = 0.0
        if obj_greedy > 0:
            gap_pct = max(0.0, (obj_best - obj_greedy) / obj_greedy * 100)

        sol.proven_optimal = False
        sol.runtime_sec = runtime
        sol.meta = {
            "iterations": iterations,
            "gap_pct": gap_pct,
        }
        return sol
