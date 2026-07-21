"""Random instance generator (Section 6)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from src.io_utils import instance_to_dict
from src.model import Infeasible, Instance, Operation, feasibility_precheck


def gen_instance(
    seed: int,
    M: int,
    N: int,
    G: int,
    T: int | None = None,
    p_range: tuple[int, int] = (1, 5),
    rel_frac: float = 0.5,
    w1: float = 1.0,
    w2: float = 1.0,
    horizon_slack: float = 1.5,
    id_prefix: str = "inst",
) -> Instance:
    """Generate a random instance that passes feasibility_precheck."""
    rng = random.Random(seed)
    K = M + N

    p_vals = [rng.randint(p_range[0], p_range[1]) for _ in range(K)]
    max_p = max(p_vals)
    total_work = sum(p_vals)

    auto_t = T is None
    if T is None:
        T = max(max_p + 1, math.ceil((total_work / G) * horizon_slack))
    if T <= max_p:
        T = max_p + 1

    # May bump T a few times if precheck fails (tight horizon / workload)
    for _attempt in range(20):
        ops: list[Operation] = []
        uid = 0
        for i in range(M):
            p = p_vals[uid]
            r = 1
            if rel_frac > 0 and rng.random() < rel_frac:
                r = rng.randint(1, max(1, int(rel_frac * T)))
            r = min(r, T - p + 1)
            ops.append(Operation(uid=uid, kind="delivery", local_id=i, p=p, r=r, w=w1))
            uid += 1

        for j in range(N):
            p = p_vals[uid]
            r = rng.randint(1, max(1, int(rel_frac * T)))
            r = min(r, T - p + 1)
            ops.append(Operation(uid=uid, kind="pickup", local_id=j, p=p, r=r, w=w2))
            uid += 1

        instance = Instance(
            id=f"{id_prefix}_s{seed}_M{M}_N{N}_G{G}",
            T=T,
            G=G,
            ops=ops,
            w1=w1,
            w2=w2,
        )
        try:
            feasibility_precheck(instance)
            return instance
        except Infeasible:
            if not auto_t and _attempt == 0:
                # User-supplied T may be tight; bump and retry
                auto_t = True
            T = max(T + 1, math.ceil((total_work / G) * (horizon_slack + 0.25 * (_attempt + 1))))
            if T <= max_p:
                T = max_p + 1

    raise Infeasible(f"Could not generate feasible instance for seed={seed} M={M} N={N} G={G}")


def _instance_to_json_dict(inst: Instance) -> dict:
    return instance_to_dict(inst)


def generate_suite(
    seeds: list[int],
    sizes: list[tuple[int, int]],
    gates: list[int],
    output_dir: str | None = None,
    suite_name: str = "suite",
    p_range: tuple[int, int] = (1, 5),
    rel_frac: float = 0.5,
) -> list[Instance]:
    """Generate a suite of instances, optionally saving JSON files."""
    instances: list[Instance] = []
    for seed in seeds:
        for M, N in sizes:
            for G in gates:
                inst = gen_instance(
                    seed=seed,
                    M=M,
                    N=N,
                    G=G,
                    p_range=p_range,
                    rel_frac=rel_frac,
                    id_prefix=suite_name,
                )
                instances.append(inst)

                if output_dir is not None:
                    path = Path(output_dir)
                    path.mkdir(parents=True, exist_ok=True)
                    fname = path / f"{inst.id}.json"
                    with open(fname, "w", encoding="utf-8") as f:
                        json.dump(_instance_to_json_dict(inst), f, indent=2)

    return instances


# Reduced K grid for course practicality (SPEC lists up to 400; we cap at 60)
_SIZE_BY_K: dict[int, tuple[int, int]] = {
    5: (3, 2),
    10: (5, 5),
    15: (8, 7),
    20: (10, 10),
    25: (13, 12),
    30: (15, 15),
    40: (20, 20),
    60: (30, 30),
}


def generate_suites(
    output_dir: str | Path = "data/instances",
    write_manifest: bool = True,
) -> dict[str, list[Instance]]:
    """Generate disjoint TUNE / TEST / PROFILE suites under output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Disjoint seed ranges
    tune_seeds = list(range(0, 5))
    test_seeds = list(range(100, 105))
    profile_seeds = list(range(200, 203))

    tune_sizes = [_SIZE_BY_K[k] for k in (5, 10, 15, 20)]
    test_sizes = [_SIZE_BY_K[k] for k in (5, 10, 15, 20, 30)]
    profile_sizes = [_SIZE_BY_K[k] for k in (5, 10, 15, 20, 25, 30, 40, 60)]

    suites = {
        "TUNE": generate_suite(
            seeds=tune_seeds,
            sizes=tune_sizes,
            gates=[1, 2],
            output_dir=str(output_dir / "TUNE"),
            suite_name="TUNE",
        ),
        "TEST": generate_suite(
            seeds=test_seeds,
            sizes=test_sizes,
            gates=[2],
            output_dir=str(output_dir / "TEST"),
            suite_name="TEST",
        ),
        "PROFILE": generate_suite(
            seeds=profile_seeds,
            sizes=profile_sizes,
            gates=[2],
            output_dir=str(output_dir / "PROFILE"),
            suite_name="PROFILE",
        ),
    }

    if write_manifest:
        manifest = {
            name: [
                {"id": inst.id, "K": len(inst.ops), "G": inst.G, "T": inst.T}
                for inst in insts
            ]
            for name, insts in suites.items()
        }
        with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return suites


if __name__ == "__main__":
    suites = generate_suites()
    for name, insts in suites.items():
        print(f"{name}: {len(insts)} instances")
