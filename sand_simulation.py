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


def apply_rainbow(
    hue_grid: np.ndarray,
    active_slices: tuple[slice, slice] | None = None,
) -> None:
    """Shift hues for a rainbow effect, in place."""
    if active_slices is None:
        grid_view = hue_grid
    else:
        row_slice, col_slice = active_slices
        grid_view = hue_grid[row_slice, col_slice]

    occupied = grid_view > 0
    np.add(grid_view, 1.0, out=grid_view, where=occupied)
    # Reuse the occupied mask to avoid allocating another array.
    np.greater(grid_view, 359.0, out=occupied)
    grid_view[occupied] -= 359.0


def compute_active_bounds(
    hue_grid: np.ndarray,
    padding: int = 1,
) -> tuple[slice, slice] | None:
    """Return the bounding slices of non-empty cells, expanded by padding."""
    rows_any = np.any(hue_grid, axis=1)
    if not rows_any.any():
        return None

    cols_any = np.any(hue_grid, axis=0)

    row_first = int(rows_any.argmax())
    row_last_exclusive = int(rows_any.size - rows_any[::-1].argmax())
    col_first = int(cols_any.argmax())
    col_last_exclusive = int(cols_any.size - cols_any[::-1].argmax())

    row_start = max(row_first - padding, 0)
    row_end = min(row_last_exclusive + padding, hue_grid.shape[0])
    col_start = max(col_first - padding, 0)
    col_end = min(col_last_exclusive + padding, hue_grid.shape[1])
    return slice(row_start, row_end), slice(col_start, col_end)


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
    changed_mask: np.ndarray

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
            changed_mask=np.zeros(shape, dtype=bool),
        )


def step_simulation(
    hue_grid: np.ndarray,
    buffers: SimulationBuffers | None = None,
    rng=np.random,
    active_slices: tuple[slice, slice] | None = None,
) -> np.ndarray:
    """Advance the simulation by one tick and return the next grid."""
    if buffers is None:
        return _step_simulation_simple(hue_grid, rng)

    return _step_simulation_buffered(hue_grid, buffers, rng, active_slices)


def _step_simulation_simple(hue_grid: np.ndarray, rng) -> np.ndarray:
    """Advance the simulation without reusing preallocated buffers."""
    filled = hue_grid > 0
    empty = hue_grid == 0
    empty_any_from_bottom = np.logical_or.accumulate(empty[::-1], axis=0)[::-1]
    empty_below = np.zeros_like(empty, dtype=bool)
    empty_below[:-1] = empty_any_from_bottom[1:]
    can_fall = filled & empty_below

    next_grid = np.zeros_like(hue_grid)

    # Straight-down movement.
    fall_rows, fall_cols = np.where(can_fall[:-1])
    next_grid[fall_rows + 1, fall_cols] = hue_grid[fall_rows, fall_cols]

    remaining = filled.copy()
    remaining[:-1] &= ~can_fall[:-1]

    # Diagonal movement.
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


def _step_simulation_buffered(
    hue_grid: np.ndarray,
    buffers: SimulationBuffers,
    rng,
    active_slices: tuple[slice, slice] | None,
) -> np.ndarray:
    """Advance the simulation using preallocated buffers and optional slices."""
    row_slice, col_slice = active_slices or (slice(None), slice(None))
    grid_view = hue_grid[row_slice, col_slice]

    filled = buffers.filled[row_slice, col_slice]
    np.greater(grid_view, 0, out=filled)

    can_fall = buffers.can_fall[row_slice, col_slice]
    empty = buffers.below_empty[row_slice, col_slice]
    np.equal(grid_view, 0, out=empty)
    np.logical_or.accumulate(empty[::-1], axis=0, out=can_fall[::-1])
    can_fall[:-1] = can_fall[1:]
    can_fall[-1] = False
    np.logical_and(filled, can_fall, out=can_fall)

    next_grid = buffers.next_grid
    next_region = next_grid[row_slice, col_slice]
    next_region.fill(0)

    # Straight-down movement.
    fall_rows, fall_cols = np.where(can_fall[:-1])
    next_region[fall_rows + 1, fall_cols] = grid_view[fall_rows, fall_cols]

    remaining = buffers.remaining[row_slice, col_slice]
    np.copyto(remaining, filled)
    remaining[:-1] &= ~can_fall[:-1]

    # Diagonal movement.
    down_right_empty = buffers.down_right_empty[row_slice, col_slice]
    down_right_empty.fill(False)
    down_right_empty[:-1, :-1] = grid_view[1:, 1:] == 0

    down_left_empty = buffers.down_left_empty[row_slice, col_slice]
    down_left_empty.fill(False)
    down_left_empty[:-1, 1:] = grid_view[1:, :-1] == 0

    direction_choice = rng.randint(0, 2, size=grid_view.shape, dtype=np.int8)

    right_sources = buffers.right_sources[row_slice, col_slice]
    np.copyto(right_sources, remaining)
    right_sources[:-1, :-1] &= down_right_empty[:-1, :-1] & (
        direction_choice[:-1, :-1] == 0
    )
    right_rows, right_cols = np.where(right_sources[:-1, :-1])
    right_targets_empty = next_region[right_rows + 1, right_cols + 1] == 0
    right_rows = right_rows[right_targets_empty]
    right_cols = right_cols[right_targets_empty]
    next_region[right_rows + 1, right_cols + 1] = grid_view[right_rows, right_cols]

    left_sources = buffers.left_sources[row_slice, col_slice]
    np.copyto(left_sources, remaining)
    left_sources[:-1, 1:] &= down_left_empty[:-1, 1:] & (
        direction_choice[:-1, 1:] == 1
    )
    left_rows, left_cols = np.where(left_sources[:-1, 1:])
    left_targets_empty = next_region[left_rows + 1, left_cols] == 0
    left_rows = left_rows[left_targets_empty]
    left_cols = left_cols[left_targets_empty]
    next_region[left_rows + 1, left_cols] = grid_view[left_rows, left_cols + 1]

    moved_sources = buffers.moved_sources[row_slice, col_slice]
    moved_sources.fill(False)
    moved_sources[fall_rows, fall_cols] = True
    moved_sources[right_rows, right_cols] = True
    moved_sources[left_rows, left_cols + 1] = True

    stationary = buffers.stationary[row_slice, col_slice]
    np.logical_not(moved_sources, out=stationary)
    np.logical_and(filled, stationary, out=stationary)
    next_region[stationary] = grid_view[stationary]

    return next_grid
