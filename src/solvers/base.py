"""Uniform solver interface."""

from __future__ import annotations

from typing import Protocol

from src.model import Instance, Solution


class Solver(Protocol):
    name: str

    def solve(
        self,
        inst: Instance,
        *,
        time_limit_sec: float | None = None,
        seed: int | None = None,
        warm_start: Solution | None = None,
    ) -> Solution: ...
