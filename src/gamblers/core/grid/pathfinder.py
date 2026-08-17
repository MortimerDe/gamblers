from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

from gamblers.core.grid.geo import octile_dist
from gamblers.core.grid.nav_grid import NavView
from gamblers.core.types import Cell

DEFAULT_MAX_EXPANSIONS: int = 20_000


@dataclass(slots=True)
class Path:
    """a planner route (including the cell the agent already occupies)"""

    cells: list[Cell]
    index: int = 0

    @property
    def is_finished(self) -> bool:
        return self.index >= len(self.cells) - 1

    @property
    def current(self) -> Cell:
        return self.cells[self.index]

    @property
    def goal(self) -> Cell:
        return self.cells[-1]

    def peek_next(self) -> Cell | None:
        if self.is_finished:
            return None
        return self.cells[self.index + 1]

    def advance(self) -> Cell:
        if self.is_finished:
            raise RuntimeError("advance() called on a finished path")
        self.index += 1
        return self.cells[self.index]

    def remaining(self) -> list[Cell]:
        return self.cells[self.index + 1 :]

    def to_payload(self) -> dict[str, Any]:
        return {"cells": [list(c) for c in self.cells], "index": self.index}

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> Path:
        return Path(
            cells=[(int(x), int(y)) for x, y in payload["cells"]],
            index=int(payload["index"]),
        )


@dataclass(order=True, slots=True)
class _OpenNode:
    f_score: int
    tie_break: int
    cell: Cell


class Pathfinder:
    def __init__(self, max_expansions: int = DEFAULT_MAX_EXPANSIONS) -> None:
        self.max_expansions = max_expansions
        self.last_expansions: int = 0

    def find(self, start: Cell, goal: Cell, grid: NavView) -> Path | None:
        if start == goal:
            return Path(cells=[start])
        if not grid.is_passable(goal):
            return None

        came_from: dict[Cell, Cell] = {}
        g_score: dict[Cell, int] = {start: 0}
        closed: set[Cell] = set()
        counter = 0
        open_heap: list[_OpenNode] = [
            _OpenNode(octile_dist(start, goal), counter, start)
        ]
        self.last_expansions = 0

        while open_heap:
            node = heapq.heappop(open_heap)
            current = node.cell
            if current in closed:
                continue
            if current == goal:
                return Path(cells=self._reconstruct(came_from, current))

            closed.add(current)
            self.last_expansions += 1
            if self.last_expansions > self.max_expansions:
                return None

            for neighbour, direction in grid.neighbours(current):
                if neighbour in closed:
                    continue
                tentative = (
                    g_score[current] + direction.step_cost + grid.enter_cost(neighbour)
                )
                if tentative < g_score.get(neighbour, 1 << 30):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        _OpenNode(
                            tentative + octile_dist(neighbour, goal),
                            counter,
                            neighbour,
                        ),
                    )

        return None

    @staticmethod
    def _reconstruct(came_from: dict[Cell, Cell], goal: Cell) -> list[Cell]:
        cells = [goal]
        cursor = goal
        while cursor in came_from:
            cursor = came_from[cursor]
            cells.append(cursor)
        cells.reverse()
        return cells


@dataclass(frozen=True, slots=True)
class ReplanPolicy:
    """when a walking agent should throw its path away and plan a new one"""

    period_ticks: int = 30
    jitter_ticks: int = 10

    def should_replan(
        self,
        *,
        path: Path,
        desired_goal: Cell,
        next_cell_blocked: bool,
        ticks_since_replan: int,
        jitter: int,
    ) -> bool:
        if path.goal != desired_goal:
            return True
        if next_cell_blocked:
            return True
        return ticks_since_replan >= self.period_ticks + jitter % max(
            self.jitter_ticks, 1
        )
