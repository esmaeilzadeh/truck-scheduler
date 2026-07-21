"""Random instance generator (Section 6)."""

from __future__ import annotations

import math
import random

from src.model import Instance, Operation, feasibility_precheck, Infeasible


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
    """Generate a random instance.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    M : int
        Number of outbound deliveries.
    N : int
        Number of inbound pickups.
    G : int
        Number of gates.
    T : int or None
        Horizon length. If None, auto-sized with horizon_slack.
    p_range : tuple[int, int]
        (min, max) processing time range (uniform integer).
    rel_frac : float
        Fraction of T for release time generation.
    w1, w2 : float
        Weights for deliveries and pickups.
    horizon_slack : float
        Multiplier for auto-sizing T.
    """
    rng = random.Random(seed)

    K = M + N

    # Generate processing times
    p_vals = [rng.randint(p_range[0], p_range[1]) for _ in range(K)]
    max_p = max(p_vals)
    total_work = sum(p_vals)

    # Auto-size T if not provided
    if T is None:
        T = max(max_p + 1, math.ceil((total_work / G) * horizon_slack))

    # Ensure T > max(p_k)
    if T <= max_p:
        T = max_p + 1

    # Generate release times
    ops: list[Operation] = []
    uid = 0

    for i in range(M):
        p = p_vals[uid]
        r = 1  # default rdt = 1
        # Optionally randomize release
        if rel_frac > 0 and rng.random() < rel_frac:
            r = rng.randint(1, max(1, int(rel_frac * T)))
        # Ensure feasibility: r + p - 1 <= T
        r = min(r, T - p + 1)
        ops.append(Operation(uid=uid, kind="delivery", local_id=i, p=p, r=r, w=w1))
        uid += 1

    for j in range(N):
        p = p_vals[uid]
        r = rng.randint(1, max(1, int(rel_frac * T)))
        # Ensure feasibility
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

    return instance


def generate_suite(
    seeds: list[int],
    sizes: list[tuple[int, int]],
    gates: list[int],
    output_dir: str | None = None,
    suite_name: str = "suite",
    p_range: tuple[int, int] = (1, 5),
    rel_frac: float = 0.5,
) -> list[Instance]:
    """Generate a suite of instances.

    Parameters
    ----------
    seeds : list[int]
        Seeds to use.
    sizes : list[tuple[int, int]]
        List of (M, N) pairs.
    gates : list[int]
        Gate counts to try.
    output_dir : str or None
        Directory to save JSON files. None = don't save.
    suite_name : str
        Prefix for instance IDs.
    """
    instances = []
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
                    import json
                    from pathlib import Path

                    path = Path(output_dir)
                    path.mkdir(parents=True, exist_ok=True)
                    fname = path / f"{inst.id}.json"
                    data = {
                        "id": inst.id,
                        "T": inst.T,
                        "G": inst.G,
                        "w1": inst.w1,
                        "w2": inst.w2,
                        "deliveries": [
                            {"id": op.local_id, "p": op.p, "rdt": op.r}
                            for op in inst.ops if op.kind == "delivery"
                        ],
                        "pickups": [
                            {"id": op.local_id, "p": op.p, "release": op.r}
                            for op in inst.ops if op.kind == "pickup"
                        ],
                    }
                    with open(fname, "w") as f:
                        json.dump(data, f, indent=2)

    return instances
