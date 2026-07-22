"""Hybrid GA + Tabu — global GA exploration with Tabu local intensification."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.model import Instance, Solution, feasibility_precheck
from src.solvers.alns import decode
from src.solvers.ga import (
    _deadline_hit,
    _erd_spt_order,
    _mutate_insert,
    _mutate_swap,
    _safe_objective,
    _tournament,
    order_crossover,
)
from src.solvers.greedy import greedy_erd_spt
from src.solvers.tabu import improve_order

_DEFAULT_PARAMS_PATH = Path("config/ga_tabu_params.json")

DEFAULT_PARAMS: dict = {
    "population_size": 30,
    "elite_count": 2,
    "tournament_k": 3,
    "crossover_rate": 0.9,
    "mutation_rate": 0.2,
    "max_generations": 3000,
    "local_search_rate": 0.3,
    "local_tabu_iters": 80,
    "local_neighborhood_size": 20,
    "tabu_tenure": 7,
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


@dataclass
class HybridGATabu:
    """Memetic hybrid: GA for global search, short Tabu for local search."""

    name: str = "ga_tabu"
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
        K = len(inst.ops)

        greedy_sol = warm_start if warm_start is not None else greedy_erd_spt(inst)
        obj_greedy = greedy_sol.objective(inst)

        pop_size = max(2, int(p["population_size"]))
        elite_count = max(0, min(int(p["elite_count"]), pop_size - 1))
        tournament_k = max(2, int(p["tournament_k"]))
        crossover_rate = float(p["crossover_rate"])
        mutation_rate = float(p["mutation_rate"])
        max_gen = int(p["max_generations"])
        local_rate = float(p["local_search_rate"])
        local_iters = max(1, int(p["local_tabu_iters"]))
        local_nhood = max(1, int(p["local_neighborhood_size"]))
        tenure = int(p["tabu_tenure"])
        swap_prob = float(p["swap_prob"])

        base = _erd_spt_order(inst)
        population: list[list[int]] = [list(base)]
        attempts = 0
        while len(population) < pop_size and attempts < pop_size * 20:
            attempts += 1
            perm = list(base)
            rng.shuffle(perm)
            if _safe_objective(inst, perm) < float("inf"):
                population.append(perm)
        while len(population) < pop_size:
            population.append(list(base))

        fitness = [_safe_objective(inst, ind) for ind in population]
        best_idx = min(range(len(population)), key=lambda i: fitness[i])
        best_order = list(population[best_idx])
        obj_best = fitness[best_idx]

        generations = 0
        local_searches = 0

        for gen in range(max_gen):
            if _deadline_hit(deadline):
                break
            generations = gen + 1

            ranked = sorted(range(len(population)), key=lambda i: fitness[i])
            new_pop: list[list[int]] = [
                list(population[i]) for i in ranked[:elite_count]
            ]

            while len(new_pop) < pop_size:
                if _deadline_hit(deadline):
                    break
                parent_a = _tournament(population, fitness, tournament_k, rng)
                parent_b = _tournament(population, fitness, tournament_k, rng)
                if rng.random() < crossover_rate and K >= 2:
                    child = order_crossover(parent_a, parent_b, rng)
                else:
                    child = list(parent_a)
                if rng.random() < mutation_rate:
                    if rng.random() < 0.5:
                        child = _mutate_swap(child, rng)
                    else:
                        child = _mutate_insert(child, rng)
                if _safe_objective(inst, child) == float("inf"):
                    child = list(parent_a)
                new_pop.append(child)

            if len(new_pop) < pop_size:
                break

            # Tabu local search on elites + random fraction of the rest
            for i, ind in enumerate(new_pop):
                if _deadline_hit(deadline):
                    break
                apply_ls = i < elite_count or rng.random() < local_rate
                if not apply_ls:
                    continue
                improved, obj_ls, _ = improve_order(
                    inst,
                    ind,
                    max_iters=local_iters,
                    neighborhood_size=local_nhood,
                    tabu_tenure=tenure,
                    swap_prob=swap_prob,
                    rng=rng,
                    deadline=deadline,
                )
                local_searches += 1
                if obj_ls < float("inf"):
                    new_pop[i] = improved

            population = new_pop
            fitness = [_safe_objective(inst, ind) for ind in population]
            gen_best = min(range(len(population)), key=lambda i: fitness[i])
            if fitness[gen_best] < obj_best:
                obj_best = fitness[gen_best]
                best_order = list(population[gen_best])

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
            "generations": generations,
            "local_searches": local_searches,
            "gap_pct": gap_pct,
        }
        return sol
