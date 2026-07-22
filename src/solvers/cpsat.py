"""Tier 1 — OR-Tools CP-SAT exact solver."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from src.model import Infeasible, Instance, Operation, Solution, feasibility_precheck


def _integer_weights(ops: list[Operation]) -> list[int]:
    """Map float weights to integers that preserve the true objective ratios.

    SCALE=1 + round() collapses e.g. w=1.3 and w=0.7 both to 1, so CP-SAT
    optimizes the wrong objective and can lose to greedy on the real float obj.
    """
    weights = [float(op.w) for op in ops]
    if all(w == math.floor(w) for w in weights):
        return [int(w) for w in weights]

    for d in range(0, 8):
        scale = 10**d
        scaled = [round(w * scale) for w in weights]
        if all(abs(w * scale - s) < 1e-9 for w, s in zip(weights, scaled)):
            return scaled

    scale = 1000
    return [max(1, round(w * scale)) for w in weights]


class _StopWatcher(cp_model.CpSolverSolutionCallback):
    """Stop search when an external event is set (UI cancel)."""

    def __init__(self, stop_event: threading.Event | None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._stop_event = stop_event

    def on_solution_callback(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            self.StopSearch()


@dataclass
class CPSAT:
    """CP-SAT exact solver implementing the Solver protocol."""

    name: str = "cpsat"
    num_workers: int = 8
    _active_solver: cp_model.CpSolver | None = field(default=None, init=False, repr=False)

    def request_stop(self) -> None:
        """Ask a running Solve() to stop (safe from another thread)."""
        solver = self._active_solver
        if solver is not None:
            solver.StopSearch()

    def solve(
        self,
        inst: Instance,
        *,
        time_limit_sec: float | None = None,
        seed: int | None = None,
        warm_start: Solution | None = None,
        stop_event: threading.Event | None = None,
    ) -> Solution:
        del seed  # CP-SAT uses its own RNG; seed reserved for Solver protocol.
        feasibility_precheck(inst)
        t0 = time.perf_counter()

        if stop_event is not None and stop_event.is_set():
            raise Infeasible("CP-SAT stopped before search started")

        model = cp_model.CpModel()
        ops = inst.ops
        G = inst.G
        T = inst.T
        int_w = _integer_weights(ops)

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

        # --- Objective (true weighted sum of starts, integerized) ---
        model.Minimize(sum(int_w[i] * s[op.uid] for i, op in enumerate(ops)))

        # --- Warm start ---
        if warm_start is not None:
            for op in ops:
                if op.uid in warm_start.starts:
                    model.AddHint(s[op.uid], warm_start.starts[op.uid])
                    g = warm_start.gates[op.uid]
                    model.AddHint(x[op.uid, g], 1)

        # --- Solve until OPTIMAL, timeout, or stop_event ---
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = self.num_workers
        if time_limit_sec is not None:
            solver.parameters.max_time_in_seconds = float(time_limit_sec)

        self._active_solver = solver
        callback = _StopWatcher(stop_event) if stop_event is not None else None
        try:
            # Poll stop_event from a side thread: solution callback alone only
            # fires when a new incumbent is found.
            poller: threading.Thread | None = None
            poll_stop = threading.Event()

            def _poll_stop() -> None:
                while not poll_stop.wait(0.05):
                    if stop_event is not None and stop_event.is_set():
                        solver.StopSearch()
                        return

            if stop_event is not None:
                poller = threading.Thread(target=_poll_stop, daemon=True)
                poller.start()

            try:
                if callback is not None:
                    status = solver.Solve(model, callback)
                else:
                    status = solver.Solve(model)
            finally:
                poll_stop.set()
                if poller is not None:
                    poller.join(timeout=1.0)
        finally:
            self._active_solver = None

        runtime = time.perf_counter() - t0
        stopped = stop_event is not None and stop_event.is_set()

        if status == cp_model.INFEASIBLE:
            raise Infeasible("CP-SAT determined the model is infeasible")

        if status == cp_model.MODEL_INVALID:
            raise Infeasible("CP-SAT model is invalid")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # UNKNOWN / ABORTED: no incumbent (or search interrupted before one).
            reason = "stopped by user" if stopped else f"status={status}"
            raise Infeasible(f"CP-SAT found no feasible solution ({reason})")

        proven_optimal = status == cp_model.OPTIMAL and not stopped

        starts = {op.uid: solver.Value(s[op.uid]) for op in ops}
        gates = {}
        for op in ops:
            for g in range(1, G + 1):
                if solver.Value(x[op.uid, g]):
                    gates[op.uid] = g
                    break

        return Solution(
            starts=starts,
            gates=gates,
            proven_optimal=proven_optimal,
            runtime_sec=runtime,
            meta={
                "cpsat_status": int(status),
                "stopped": stopped,
            },
        )
