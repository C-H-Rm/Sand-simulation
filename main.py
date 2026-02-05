from __future__ import annotations

import numpy as np
import pygame

from sand_simulation import (
    SimulationBuffers,
    apply_rainbow,
    compute_active_bounds,
    erase_sand,
    place_sand,
    step_simulation,
)

FPS = 120
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
FULLSCREEN_ENABLED = True

GRAIN_SIZE = 10
PLACEMENT_RADIUS = 2

HUE_SPEED = 0.2
SATURATION = 20
VALUE_START = 75

BREATHING_ENABLED = False
BREATHING_MIN = 40
BREATHING_MAX = 100

RAINBOW_ENABLED = False
BACKGROUND_COLOR = (0, 0, 0)
FPS_COLOR = (240, 240, 240)
FPS_FONT_SIZE = 18
FPS_UPDATE_MS = 200
FPS_PADDING = 8
DIRTY_RENDERING = True
DIRTY_FULL_REDRAW_THRESHOLD = 0.4


def merge_bounds(
    bounds: tuple[slice, slice] | None,
    new_bounds: tuple[slice, slice] | None,
) -> tuple[slice, slice] | None:
    if bounds is None:
        return new_bounds
    if new_bounds is None:
        return bounds
    row_bounds, col_bounds = bounds
    new_rows, new_cols = new_bounds
    return (
        slice(min(row_bounds.start, new_rows.start), max(row_bounds.stop, new_rows.stop)),
        slice(min(col_bounds.start, new_cols.start), max(col_bounds.stop, new_cols.stop)),
    )


def update_breathing(
    value: int,
    rising: bool,
    min_value: int,
    max_value: int,
) -> tuple[int, bool]:
    if rising:
        value += 1
        if value >= max_value:
            rising = False
    else:
        value -= 1
        if value < min_value:
            rising = True
    return value, rising


def handle_events(rainbow_enabled: bool) -> tuple[bool, bool]:
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                rainbow_enabled = not rainbow_enabled
    return running, rainbow_enabled


def handle_mouse_input(
    hue_grid: np.ndarray,
    hue: float,
    hue_speed: float,
    grain_size: int,
    placement_radius: int,
    screen_width: int,
    screen_height: int,
) -> tuple[float, tuple[slice, slice] | None]:
    mouse_buttons = pygame.mouse.get_pressed()
    if not (mouse_buttons[0] or mouse_buttons[2]):
        return hue, None

    x, y = pygame.mouse.get_pos()
    if not (0 <= x < screen_width and 0 <= y < screen_height):
        return hue, None

    grid_x = x // grain_size
    grid_y = y // grain_size
    max_rows, max_cols = hue_grid.shape
    dirty_bounds: tuple[slice, slice] | None = None

    if mouse_buttons[0]:
        hue = (hue + hue_speed) % 360
        place_sand(hue_grid, grid_x, grid_y, hue, placement_radius)
        if placement_radius == 0:
            dirty_bounds = merge_bounds(
                dirty_bounds,
                (slice(grid_y, grid_y + 1), slice(grid_x, grid_x + 1)),
            )
        else:
            y_start = max(0, grid_y - placement_radius)
            y_end = min(max_rows, grid_y + placement_radius)
            x_start = max(0, grid_x - placement_radius)
            x_end = min(max_cols, grid_x + placement_radius)
            dirty_bounds = merge_bounds(
                dirty_bounds,
                (slice(y_start, y_end), slice(x_start, x_end)),
            )

    if mouse_buttons[2]:
        erase_sand(hue_grid, grid_x, grid_y)
        dirty_bounds = merge_bounds(
            dirty_bounds,
            (slice(grid_y, grid_y + 1), slice(grid_x, grid_x + 1)),
        )

    return hue, dirty_bounds


def draw_grid(
    screen: pygame.Surface,
    hue_grid: np.ndarray,
    grain_size: int,
    color: pygame.Color,
    saturation: int,
    value: int,
) -> None:
    grid_height, grid_width = hue_grid.shape
    for row in range(grid_height):
        row_values = hue_grid[row]
        y_pos = row * grain_size
        for col in range(grid_width):
            hue_value = row_values[col]
            if hue_value > 0:
                color.hsva = (hue_value, saturation, value, 100)
                pygame.draw.rect(
                    screen,
                    color,
                    (col * grain_size, y_pos, grain_size, grain_size),
                )


def draw_changed_cells(
    screen: pygame.Surface,
    hue_grid: np.ndarray,
    grain_size: int,
    color: pygame.Color,
    saturation: int,
    value: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> list[pygame.Rect]:
    rects: list[pygame.Rect] = []
    for row, col in zip(rows, cols):
        hue_value = hue_grid[row, col]
        rect = pygame.Rect(
            col * grain_size,
            row * grain_size,
            grain_size,
            grain_size,
        )
        if hue_value > 0:
            color.hsva = (hue_value, saturation, value, 100)
            pygame.draw.rect(screen, color, rect)
        else:
            pygame.draw.rect(screen, BACKGROUND_COLOR, rect)
        rects.append(rect)
    return rects


def update_fps_surface(
    font: pygame.font.Font,
    fps_value: float,
) -> pygame.Surface:
    fps_text = f"{fps_value:5.1f} FPS"
    return font.render(fps_text, True, FPS_COLOR)


def rect_to_grid_bounds(
    rect: pygame.Rect,
    grain_size: int,
    grid_width: int,
    grid_height: int,
) -> tuple[slice, slice]:
    col_start = max(0, rect.left // grain_size)
    col_end = min(grid_width, (rect.right + grain_size - 1) // grain_size)
    row_start = max(0, rect.top // grain_size)
    row_end = min(grid_height, (rect.bottom + grain_size - 1) // grain_size)
    return slice(row_start, row_end), slice(col_start, col_end)


def setup_screen(
    fullscreen_enabled: bool,
    screen_size: tuple[int, int],
) -> tuple[pygame.Surface, int, int]:
    if fullscreen_enabled:
        display_info = pygame.display.Info()
        screen_width = display_info.current_w
        screen_height = display_info.current_h
        screen = pygame.display.set_mode(
            (screen_width, screen_height),
            pygame.FULLSCREEN,
        )
    else:
        screen_width, screen_height = screen_size
        screen = pygame.display.set_mode((screen_width, screen_height))
    return screen, screen_width, screen_height


def update_fps_display(
    font: pygame.font.Font,
    fps_value: float,
    screen_width: int,
    padding: int,
    grain_size: int,
    grid_width: int,
    grid_height: int,
) -> tuple[pygame.Surface, pygame.Rect, tuple[slice, slice]]:
    fps_surface = update_fps_surface(font, fps_value)
    fps_rect = fps_surface.get_rect()
    fps_rect.top = padding
    fps_rect.right = screen_width - padding
    fps_grid_slice = rect_to_grid_bounds(
        fps_rect,
        grain_size,
        grid_width,
        grid_height,
    )
    return fps_surface, fps_rect, fps_grid_slice


def render_full_frame(
    screen: pygame.Surface,
    hue_grid: np.ndarray,
    grain_size: int,
    color: pygame.Color,
    saturation: int,
    value: int,
    fps_surface: pygame.Surface,
    fps_rect: pygame.Rect,
) -> None:
    screen.fill(BACKGROUND_COLOR)
    draw_grid(screen, hue_grid, grain_size, color, saturation, value)
    screen.blit(fps_surface, fps_rect)
    pygame.display.flip()


def main() -> None:
    pygame.init()
    clock = pygame.time.Clock()
    screen, screen_width, screen_height = setup_screen(
        FULLSCREEN_ENABLED,
        SCREEN_SIZE,
    )
    pygame.display.set_caption("Sand simulation")
    fps_font = pygame.font.Font(None, FPS_FONT_SIZE)

    grid_width = screen_width // GRAIN_SIZE
    grid_height = screen_height // GRAIN_SIZE
    hue_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
    buffers = SimulationBuffers.create(hue_grid.shape, hue_grid.dtype)

    hue = 1.0
    brightness = VALUE_START
    breathing_rising = True
    sand_color = pygame.Color(0, 0, 0, 0)
    fps_surface, fps_rect, fps_grid_slice = update_fps_display(
        fps_font,
        0.0,
        screen_width,
        FPS_PADDING,
        GRAIN_SIZE,
        grid_width,
        grid_height,
    )
    last_fps_update = 0
    first_frame = True
    last_active_bounds: tuple[slice, slice] | None = None

    running = True
    rainbow_enabled = RAINBOW_ENABLED
    while running:
        running, rainbow_enabled = handle_events(rainbow_enabled)

        if BREATHING_ENABLED:
            brightness, breathing_rising = update_breathing(
                brightness,
                breathing_rising,
                BREATHING_MIN,
                BREATHING_MAX,
            )

        hue, input_dirty_bounds = handle_mouse_input(
            hue_grid,
            hue,
            HUE_SPEED,
            GRAIN_SIZE,
            PLACEMENT_RADIUS,
            screen_width,
            screen_height,
        )

        active_slices = compute_active_bounds(hue_grid, padding=1)
        clear_slices = merge_bounds(last_active_bounds, active_slices)
        if clear_slices is not None:
            buffers.next_grid[clear_slices].fill(0)
        if active_slices is None:
            next_grid = buffers.next_grid
        else:
            next_grid = step_simulation(
                hue_grid,
                buffers=buffers,
                active_slices=active_slices,
            )
        hue_grid, buffers.next_grid = next_grid, hue_grid
        last_active_bounds = active_slices

        if rainbow_enabled:
            apply_rainbow(hue_grid, active_slices)

        now_ms = pygame.time.get_ticks()
        if now_ms - last_fps_update >= FPS_UPDATE_MS:
            fps_surface, fps_rect, fps_grid_slice = update_fps_display(
                fps_font,
                clock.get_fps(),
                screen_width,
                FPS_PADDING,
                GRAIN_SIZE,
                grid_width,
                grid_height,
            )
            last_fps_update = now_ms

        if not DIRTY_RENDERING or first_frame:
            render_full_frame(
                screen,
                hue_grid,
                GRAIN_SIZE,
                sand_color,
                SATURATION,
                brightness,
                fps_surface,
                fps_rect,
            )
            first_frame = False
            clock.tick(FPS)
            continue

        previous_grid = buffers.next_grid
        changed_mask = buffers.changed_mask
        np.not_equal(hue_grid, previous_grid, out=changed_mask)
        row_slice, col_slice = fps_grid_slice
        changed_mask[row_slice, col_slice] = True
        if input_dirty_bounds is not None:
            row_slice, col_slice = input_dirty_bounds
            changed_mask[row_slice, col_slice] = True

        changed_count = int(np.count_nonzero(changed_mask))
        if changed_count == 0:
            screen.blit(fps_surface, fps_rect)
            pygame.display.update(fps_rect)
            clock.tick(FPS)
            continue

        if changed_count > hue_grid.size * DIRTY_FULL_REDRAW_THRESHOLD:
            render_full_frame(
                screen,
                hue_grid,
                GRAIN_SIZE,
                sand_color,
                SATURATION,
                brightness,
                fps_surface,
                fps_rect,
            )
            clock.tick(FPS)
            continue

        changed_rows, changed_cols = np.where(changed_mask)
        dirty_rects = draw_changed_cells(
            screen,
            hue_grid,
            GRAIN_SIZE,
            sand_color,
            SATURATION,
            brightness,
            changed_rows,
            changed_cols,
        )
        screen.blit(fps_surface, fps_rect)
        dirty_rects.append(fps_rect)
        pygame.display.update(dirty_rects)

        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
