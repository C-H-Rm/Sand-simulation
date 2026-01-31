import pygame
import numpy as np

clock = pygame.time.Clock()



def main():
    fps = 120

    # changes color in hsv format
    saturation = 20
    value = 75

    # speed at which the hue moves
    hue_speed = 0.2 

    # makes the colour breathe
    breathing = False

    rainbow = True


    grain_size = 8
    screen_width = 1000
    screen_height = 1000

    placement_size = 2
    


    running = True
    hue = 1
    breathing_rising = True
    grain_amount_width = screen_width // grain_size
    grain_amount_height = screen_height // grain_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Sand simulation")
    grid = np.zeros((grain_amount_height, grain_amount_width), dtype=np.float32)
    while running:
        if breathing:
            if breathing_rising:
                value += 1

                if not value < 100:
                    breathing_rising = not breathing_rising

            else:
                value -= 1

                if value < 40:
                    breathing_rising = not breathing_rising

        color = pygame.Color(0, 0, 0, 0)
        screen.fill(pygame.Color(0,0,0))
        for event in pygame.event.get(pygame.QUIT):
            if event.type == pygame.QUIT:
                running = False
            
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            hue = (hue + hue_speed) % 360

            if placement_size == 0:
                if grid[y // grain_size, x // grain_size] == 0:
                    grid[y // grain_size, x // grain_size] = hue

            elif -1 < x < screen_width and -1 < y < screen_height:
                y_index = y // grain_size
                x_index = x // grain_size
                y_start = max(0, y_index - placement_size)
                y_end = min(grain_amount_height, y_index + placement_size)
                x_start = max(0, x_index - placement_size)
                x_end = min(grain_amount_width, x_index + placement_size)
                placement_area = grid[y_start:y_end, x_start:x_end]
                placement_area[placement_area == 0] = hue
        
        if pygame.mouse.get_pressed()[2]:
            x, y = pygame.mouse.get_pos()

            if -1 < x < screen_width and -1 < y < screen_height:
                grid[y // grain_size, x // grain_size] = 0

        occupied = grid > 0
        below_empty = np.zeros_like(occupied, dtype=bool)
        below_empty[:-1] = grid[1:] == 0
        can_fall = occupied & below_empty
        newgrid = np.zeros_like(grid)

        fall_rows, fall_cols = np.where(can_fall[:-1])
        newgrid[fall_rows + 1, fall_cols] = grid[fall_rows, fall_cols]

        remaining = occupied.copy()
        remaining[:-1] &= ~can_fall[:-1]

        down_right_empty = np.zeros_like(occupied, dtype=bool)
        down_left_empty = np.zeros_like(occupied, dtype=bool)
        down_right_empty[:-1, :-1] = grid[1:, 1:] == 0
        down_left_empty[:-1, 1:] = grid[1:, :-1] == 0
        direction_choice = np.random.randint(0, 2, size=grid.shape, dtype=np.int8)

        right_sources = remaining.copy()
        right_sources[:-1, :-1] &= down_right_empty[:-1, :-1] & (direction_choice[:-1, :-1] == 0)
        right_rows, right_cols = np.where(right_sources[:-1, :-1])
        right_targets_empty = newgrid[right_rows + 1, right_cols + 1] == 0
        right_rows = right_rows[right_targets_empty]
        right_cols = right_cols[right_targets_empty]
        newgrid[right_rows + 1, right_cols + 1] = grid[right_rows, right_cols]

        left_sources = remaining.copy()
        left_sources[:-1, 1:] &= down_left_empty[:-1, 1:] & (direction_choice[:-1, 1:] == 1)
        left_rows, left_cols = np.where(left_sources[:-1, 1:])
        left_targets_empty = newgrid[left_rows + 1, left_cols] == 0
        left_rows = left_rows[left_targets_empty]
        left_cols = left_cols[left_targets_empty]
        newgrid[left_rows + 1, left_cols] = grid[left_rows, left_cols + 1]

        moved_sources = np.zeros_like(occupied, dtype=bool)
        moved_sources[fall_rows, fall_cols] = True
        moved_sources[right_rows, right_cols] = True
        moved_sources[left_rows, left_cols + 1] = True
        stationary = occupied & ~moved_sources
        newgrid[stationary] = grid[stationary]

        grid = newgrid
        if rainbow:
            grid[grid > 0] = (grid[grid > 0] % 359) + 1

        for row in range(0, grain_amount_height):
            for column in range(0, grain_amount_width):
                if grid[row][column] >= 1:
                    color.hsva = (grid[row][column], saturation, value, 100)
                    pygame.draw.rect(screen, color, (column*grain_size, row*grain_size, grain_size, grain_size))
        
        pygame.display.flip()
        clock.tick(fps)

main()
