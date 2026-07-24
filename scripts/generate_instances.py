#!/usr/bin/env python3
"""Generate and save instance JSON files (deterministic via gen_instance).

Examples:
  .venv/bin/python scripts/generate_instances.py \\
      --ids CMP_s2000009_M200_N200_G2 CMP_s2000008_M200_N200_G10 \\
      --output-dir data/instances/BENCHMARK --horizon-slack 2.0

  .venv/bin/python scripts/generate_instances.py \\
      --seed 42 --M 10 --N 10 --G 2 --id-prefix BENCH \\
      --output-dir data/instances/BENCHMARK
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instance_gen import gen_instance
from src.io_utils import instance_to_dict

_ID_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+)_s(?P<seed>\d+)_M(?P<M>\d+)_N(?P<N>\d+)_G(?P<G>\d+)$"
)


def parse_instance_id(instance_id: str) -> dict:
    m = _ID_RE.match(instance_id.strip())
    if not m:
        raise ValueError(
            f"Unrecognized instance id {instance_id!r}; "
            "expected PREFIX_s{{seed}}_M{{M}}_N{{N}}_G{{G}}"
        )
    return {
        "id_prefix": m.group("prefix"),
        "seed": int(m.group("seed")),
        "M": int(m.group("M")),
        "N": int(m.group("N")),
        "G": int(m.group("G")),
    }


def write_instance(
    *,
    seed: int,
    M: int,
    N: int,
    G: int,
    output_dir: Path,
    id_prefix: str = "inst",
    horizon_slack: float = 1.5,
) -> Path:
    inst = gen_instance(
        seed=seed,
        M=M,
        N=N,
        G=G,
        id_prefix=id_prefix,
        horizon_slack=horizon_slack,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{inst.id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(instance_to_dict(inst), f, indent=2)
        f.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate instance JSON files")
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Instance ids like CMP_s2000009_M200_N200_G2",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--M", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--G", type=int)
    parser.add_argument("--id-prefix", default="inst")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "instances" / "BENCHMARK",
    )
    parser.add_argument(
        "--horizon-slack",
        type=float,
        default=2.0,
        help="Passed to gen_instance (benchmark runner used 2.0)",
    )
    args = parser.parse_args()

    specs: list[dict] = []
    if args.ids:
        for instance_id in args.ids:
            specs.append(parse_instance_id(instance_id))
    elif None not in (args.seed, args.M, args.N, args.G):
        specs.append(
            {
                "id_prefix": args.id_prefix,
                "seed": args.seed,
                "M": args.M,
                "N": args.N,
                "G": args.G,
            }
        )
    else:
        parser.error("Provide --ids ... or all of --seed --M --N --G")

    for spec in specs:
        path = write_instance(
            seed=spec["seed"],
            M=spec["M"],
            N=spec["N"],
            G=spec["G"],
            output_dir=args.output_dir,
            id_prefix=spec["id_prefix"],
            horizon_slack=args.horizon_slack,
        )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
