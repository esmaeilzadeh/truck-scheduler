"""Genetic Algorithm metaheuristic — permutation chromosomes + ALNS decode."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from src.model import Infeasible, Instance, Solution, feasibility_precheck
from src.solvers.alns import _objective_from_order, decode
from src.solvers.greedy import greedy_erd_spt

_DEFAULT_PARAMS_PATH = Path("config/ga_params.json")

DEFAULT_PARAMS: dict = {
    "population_size": 40,
    "elite_count": 2,
    "tournament_k": 3,
    "crossover_rate": 0.9,
    "mutation_rate": 0.2,
    "max_generations": 5000,
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


def _deadline_hit(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _safe_objective(inst: Instance, order: list[int]) -> float:
    """Objective or +inf if the permutation cannot be decoded into the horizon."""
    try:
        return _objective_from_order(inst, order)
    except Infeasible:
        return float("inf")


def order_crossover(
    parent_a: list[int], parent_b: list[int], rng: random.Random,
) -> list[int]:
    """OX (order crossover) for permutations."""
    n = len(parent_a)
    if n <= 1:
        return list(parent_a)
    i, j = sorted(rng.sample(range(n), 2))
    child: list[int | None] = [None] * n
    child[i : j + 1] = parent_a[i : j + 1]
    filled = set(parent_a[i : j + 1])
    pos = (j + 1) % n
    for uid in parent_b[j + 1 :] + parent_b[: j + 1]:
        if uid in filled:
            continue
        child[pos] = uid
        filled.add(uid)
        pos = (pos + 1) % n
    assert all(x is not None for x in child)
    return [int(x) for x in child]  # type: ignore[arg-type]


def _mutate_swap(order: list[int], rng: random.Random) -> list[int]:
    out = list(order)
    if len(out) < 2:
        return out
    i, j = rng.sample(range(len(out)), 2)
    out[i], out[j] = out[j], out[i]
    return out


def _mutate_insert(order: list[int], rng: random.Random) -> list[int]:
    out = list(order)
    if len(out) < 2:
        return out
    i = rng.randrange(len(out))
    j = rng.randrange(len(out))
    if i == j:
        return out
    uid = out.pop(i)
    if j > i:
        j -= 1
    out.insert(j, uid)
    return out


def _tournament(
    pop: list[list[int]],
    fitness: list[float],
    k: int,
    rng: random.Random,
) -> list[int]:
    idxs = rng.sample(range(len(pop)), min(k, len(pop)))
    best = min(idxs, key=lambda i: fitness[i])
    return list(pop[best])


@dataclass
class GeneticAlgorithm:
    """Genetic Algorithm solver implementing the Solver protocol."""

    name: str = "ga"
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

        # Initial population: greedy + random permutations (retry infeasible)
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
                # Reject infeasible offspring — keep a parent copy instead
                if _safe_objective(inst, child) == float("inf"):
                    child = list(parent_a)
                new_pop.append(child)

            if len(new_pop) < pop_size:
                break

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
            "gap_pct": gap_pct,
        }
        return sol
