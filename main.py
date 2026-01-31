from __future__ import annotations

import numpy as np
import pygame

from sand_simulation import apply_rainbow, erase_sand, place_sand, step_simulation

FPS = 120
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)

GRAIN_SIZE = 8
PLACEMENT_RADIUS = 2

HUE_SPEED = 0.2
SATURATION = 20
VALUE_START = 75

BREATHING_ENABLED = False
BREATHING_MIN = 40
BREATHING_MAX = 100

RAINBOW_ENABLED = True
BACKGROUND_COLOR = (0, 0, 0)


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


def handle_quit_events() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True


def handle_mouse_input(
    hue_grid: np.ndarray,
    hue: float,
    hue_speed: float,
    grain_size: int,
    placement_radius: int,
    screen_width: int,
    screen_height: int,
) -> float:
    mouse_buttons = pygame.mouse.get_pressed()
    if not (mouse_buttons[0] or mouse_buttons[2]):
        return hue

    x, y = pygame.mouse.get_pos()
    if not (0 <= x < screen_width and 0 <= y < screen_height):
        return hue

    grid_x = x // grain_size
    grid_y = y // grain_size

    if mouse_buttons[0]:
        hue = (hue + hue_speed) % 360
        place_sand(hue_grid, grid_x, grid_y, hue, placement_radius)

    if mouse_buttons[2]:
        erase_sand(hue_grid, grid_x, grid_y)

    return hue


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
            if hue_value >= 1:
                color.hsva = (hue_value, saturation, value, 100)
                pygame.draw.rect(
                    screen,
                    color,
                    (col * grain_size, y_pos, grain_size, grain_size),
                )


def main() -> None:
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Sand simulation")

    grid_width = SCREEN_WIDTH // GRAIN_SIZE
    grid_height = SCREEN_HEIGHT // GRAIN_SIZE
    hue_grid = np.zeros((grid_height, grid_width), dtype=np.float32)

    hue = 1.0
    brightness = VALUE_START
    breathing_rising = True
    sand_color = pygame.Color(0, 0, 0, 0)

    running = True
    while running:
        running = handle_quit_events()

        if BREATHING_ENABLED:
            brightness, breathing_rising = update_breathing(
                brightness,
                breathing_rising,
                BREATHING_MIN,
                BREATHING_MAX,
            )

        hue = handle_mouse_input(
            hue_grid,
            hue,
            HUE_SPEED,
            GRAIN_SIZE,
            PLACEMENT_RADIUS,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
        )

        hue_grid = step_simulation(hue_grid)

        if RAINBOW_ENABLED:
            apply_rainbow(hue_grid)

        screen.fill(BACKGROUND_COLOR)
        draw_grid(screen, hue_grid, GRAIN_SIZE, sand_color, SATURATION, brightness)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
