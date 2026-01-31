from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def place_sand(
    hue_grid: np.ndarray,
    grid_x: int,
    grid_y: int,
    hue: float,
    placement_radius: int,
) -> None:
    """Place sand around a grid cell, filling only empty cells."""
    if placement_radius == 0:
        if hue_grid[grid_y, grid_x] == 0:
            hue_grid[grid_y, grid_x] = hue
        return

    max_rows, max_cols = hue_grid.shape
    y_start = max(0, grid_y - placement_radius)
    y_end = min(max_rows, grid_y + placement_radius)
    x_start = max(0, grid_x - placement_radius)
    x_end = min(max_cols, grid_x + placement_radius)

    placement_area = hue_grid[y_start:y_end, x_start:x_end]
    placement_area[placement_area == 0] = hue


def erase_sand(hue_grid: np.ndarray, grid_x: int, grid_y: int) -> None:
    """Erase sand at a grid cell."""
    hue_grid[grid_y, grid_x] = 0


def apply_rainbow(hue_grid: np.ndarray) -> None:
    """Shift hues for a rainbow effect, in place."""
    mask = hue_grid > 0
    hue_grid[mask] = (hue_grid[mask] % 359) + 1


@dataclass
class SimulationBuffers:
    next_grid: np.ndarray
    filled: np.ndarray
    below_empty: np.ndarray
    can_fall: np.ndarray
    remaining: np.ndarray
    down_right_empty: np.ndarray
    down_left_empty: np.ndarray
    right_sources: np.ndarray
    left_sources: np.ndarray
    moved_sources: np.ndarray
    stationary: np.ndarray

    @classmethod
    def create(cls, shape: tuple[int, int], dtype: np.dtype) -> "SimulationBuffers":
        return cls(
            next_grid=np.zeros(shape, dtype=dtype),
            filled=np.zeros(shape, dtype=bool),
            below_empty=np.zeros(shape, dtype=bool),
            can_fall=np.zeros(shape, dtype=bool),
            remaining=np.zeros(shape, dtype=bool),
            down_right_empty=np.zeros(shape, dtype=bool),
            down_left_empty=np.zeros(shape, dtype=bool),
            right_sources=np.zeros(shape, dtype=bool),
            left_sources=np.zeros(shape, dtype=bool),
            moved_sources=np.zeros(shape, dtype=bool),
            stationary=np.zeros(shape, dtype=bool),
        )


def step_simulation(
    hue_grid: np.ndarray,
    buffers: SimulationBuffers | None = None,
    rng=np.random,
) -> np.ndarray:
    """Advance the simulation by one tick and return the next grid."""
    if buffers is None:
        filled = hue_grid > 0
        below_empty = np.zeros_like(filled, dtype=bool)
        below_empty[:-1] = hue_grid[1:] == 0
        can_fall = filled & below_empty

        next_grid = np.zeros_like(hue_grid)

        fall_rows, fall_cols = np.where(can_fall[:-1])
        next_grid[fall_rows + 1, fall_cols] = hue_grid[fall_rows, fall_cols]

        remaining = filled.copy()
        remaining[:-1] &= ~can_fall[:-1]

        down_right_empty = np.zeros_like(filled, dtype=bool)
        down_left_empty = np.zeros_like(filled, dtype=bool)
        down_right_empty[:-1, :-1] = hue_grid[1:, 1:] == 0
        down_left_empty[:-1, 1:] = hue_grid[1:, :-1] == 0
        direction_choice = rng.randint(0, 2, size=hue_grid.shape, dtype=np.int8)

        right_sources = remaining.copy()
        right_sources[:-1, :-1] &= down_right_empty[:-1, :-1] & (
            direction_choice[:-1, :-1] == 0
        )
        right_rows, right_cols = np.where(right_sources[:-1, :-1])
        right_targets_empty = next_grid[right_rows + 1, right_cols + 1] == 0
        right_rows = right_rows[right_targets_empty]
        right_cols = right_cols[right_targets_empty]
        next_grid[right_rows + 1, right_cols + 1] = hue_grid[right_rows, right_cols]

        left_sources = remaining.copy()
        left_sources[:-1, 1:] &= down_left_empty[:-1, 1:] & (
            direction_choice[:-1, 1:] == 1
        )
        left_rows, left_cols = np.where(left_sources[:-1, 1:])
        left_targets_empty = next_grid[left_rows + 1, left_cols] == 0
        left_rows = left_rows[left_targets_empty]
        left_cols = left_cols[left_targets_empty]
        next_grid[left_rows + 1, left_cols] = hue_grid[left_rows, left_cols + 1]

        moved_sources = np.zeros_like(filled, dtype=bool)
        moved_sources[fall_rows, fall_cols] = True
        moved_sources[right_rows, right_cols] = True
        moved_sources[left_rows, left_cols + 1] = True
        stationary = filled & ~moved_sources
        next_grid[stationary] = hue_grid[stationary]

        return next_grid

    filled = buffers.filled
    np.greater(hue_grid, 0, out=filled)

    below_empty = buffers.below_empty
    below_empty.fill(False)
    below_empty[:-1] = hue_grid[1:] == 0

    can_fall = buffers.can_fall
    np.logical_and(filled, below_empty, out=can_fall)

    next_grid = buffers.next_grid
    next_grid.fill(0)

    fall_rows, fall_cols = np.where(can_fall[:-1])
    next_grid[fall_rows + 1, fall_cols] = hue_grid[fall_rows, fall_cols]

    remaining = buffers.remaining
    np.copyto(remaining, filled)
    remaining[:-1] &= ~can_fall[:-1]

    down_right_empty = buffers.down_right_empty
    down_right_empty.fill(False)
    down_right_empty[:-1, :-1] = hue_grid[1:, 1:] == 0

    down_left_empty = buffers.down_left_empty
    down_left_empty.fill(False)
    down_left_empty[:-1, 1:] = hue_grid[1:, :-1] == 0

    direction_choice = rng.randint(0, 2, size=hue_grid.shape, dtype=np.int8)

    right_sources = buffers.right_sources
    np.copyto(right_sources, remaining)
    right_sources[:-1, :-1] &= down_right_empty[:-1, :-1] & (
        direction_choice[:-1, :-1] == 0
    )
    right_rows, right_cols = np.where(right_sources[:-1, :-1])
    right_targets_empty = next_grid[right_rows + 1, right_cols + 1] == 0
    right_rows = right_rows[right_targets_empty]
    right_cols = right_cols[right_targets_empty]
    next_grid[right_rows + 1, right_cols + 1] = hue_grid[right_rows, right_cols]

    left_sources = buffers.left_sources
    np.copyto(left_sources, remaining)
    left_sources[:-1, 1:] &= down_left_empty[:-1, 1:] & (
        direction_choice[:-1, 1:] == 1
    )
    left_rows, left_cols = np.where(left_sources[:-1, 1:])
    left_targets_empty = next_grid[left_rows + 1, left_cols] == 0
    left_rows = left_rows[left_targets_empty]
    left_cols = left_cols[left_targets_empty]
    next_grid[left_rows + 1, left_cols] = hue_grid[left_rows, left_cols + 1]

    moved_sources = buffers.moved_sources
    moved_sources.fill(False)
    moved_sources[fall_rows, fall_cols] = True
    moved_sources[right_rows, right_cols] = True
    moved_sources[left_rows, left_cols + 1] = True

    stationary = buffers.stationary
    np.logical_not(moved_sources, out=stationary)
    np.logical_and(filled, stationary, out=stationary)
    next_grid[stationary] = hue_grid[stationary]

    return next_grid
