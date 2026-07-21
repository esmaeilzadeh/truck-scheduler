"""Tier 1 — OR-Tools CP-SAT exact solver."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from src.model import Infeasible, Instance, Solution, feasibility_precheck
from src.solvers.greedy import greedy_erd_spt


@dataclass
class CPSAT:
    """CP-SAT exact solver implementing the Solver protocol."""

    name: str = "cpsat"
    num_workers: int = 8

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

        model = cp_model.CpModel()
        ops = inst.ops
        K = len(ops)
        G = inst.G
        T = inst.T

        # --- Variables ---
        s: dict[int, cp_model.IntVar] = {}
        x: dict[tuple[int, int], cp_model.IntVar] = {}
        iv: dict[tuple[int, int], cp_model.IntervalVar] = {}

        for op in ops:
            s[op.uid] = model.NewIntVar(op.r, T - op.p + 1, f"s_{op.uid}")
            for g in range(1, G + 1):
                x[op.uid, g] = model.NewBoolVar(f"x_{op.uid}_{g}")
                iv[op.uid, g] = model.NewOptionalIntervalVar(
                    s[op.uid], op.p, s[op.uid] + op.p,
                    x[op.uid, g], f"iv_{op.uid}_{g}",
                )

        # --- Constraints ---
        for op in ops:
            model.AddExactlyOne(x[op.uid, g] for g in range(1, G + 1))

        for g in range(1, G + 1):
            model.AddNoOverlap(iv[op.uid, g] for op in ops)

        # --- Objective ---
        SCALE = 1
        model.Minimize(
            sum(round(op.w * SCALE) * s[op.uid] for op in ops)
        )

        # --- Warm start ---
        if warm_start is not None:
            for op in ops:
                if op.uid in warm_start.starts:
                    model.AddHint(s[op.uid], warm_start.starts[op.uid])
                    g = warm_start.gates[op.uid]
                    model.AddHint(x[op.uid, g], 1)

        # --- Solve ---
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = self.num_workers
        if time_limit_sec is not None:
            solver.parameters.max_time_in_seconds = time_limit_sec

        status = solver.Solve(model)
        runtime = time.perf_counter() - t0

        if status == cp_model.INFEASIBLE:
            raise Infeasible("CP-SAT determined the model is infeasible")

        if status == cp_model.MODEL_INVALID:
            raise Infeasible("CP-SAT model is invalid")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise Infeasible(f"CP-SAT unexpected status: {status}")

        proven_optimal = status == cp_model.OPTIMAL

        starts = {op.uid: solver.Value(s[op.uid]) for op in ops}
        gates = {}
        for op in ops:
            for g in range(1, G + 1):
                if solver.Value(x[op.uid, g]):
                    gates[op.uid] = g
                    break

        sol = Solution(
            starts=starts,
            gates=gates,
            proven_optimal=proven_optimal,
            runtime_sec=runtime,
        )
        return sol
