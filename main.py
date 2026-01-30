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
    rng = np.random.default_rng()
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
                grid_y = y // grain_size
                grid_x = x // grain_size
                y_min = max(grid_y - placement_size, 0)
                y_max = min(grid_y + placement_size, grain_amount_height - 1)
                x_min = max(grid_x - placement_size, 0)
                x_max = min(grid_x + placement_size, grain_amount_width - 1)
                region = grid[y_min : y_max + 1, x_min : x_max + 1]
                region[region == 0] = hue
        
        if pygame.mouse.get_pressed()[2]:
            x, y = pygame.mouse.get_pos()

            if -1 < x < screen_width and -1 < y < screen_height:
                grid[y // grain_size, x // grain_size] = 0

        newgrid = np.zeros_like(grid)
        occupied = grid >= 1
        if grain_amount_height > 1:
            below_empty = grid[1:, :] == 0
            can_fall = occupied[:-1, :] & below_empty
            newgrid[1:, :][can_fall] = grid[:-1, :][can_fall]

            blocked = occupied[:-1, :] & ~below_empty
            moved_from = np.zeros_like(grid, dtype=bool)
            moved_from[:-1, :][can_fall] = True

            if grain_amount_width > 1:
                blocked_left = blocked[:, 0]
                down_right_empty_left = grid[1:, 1] == 0
                rand_left = rng.integers(0, 2, size=blocked_left.shape)
                move_right_left = blocked_left & (rand_left == 0) & down_right_empty_left
                newgrid[1:, 1][move_right_left] = grid[:-1, 0][move_right_left]
                moved_from[:-1, 0][move_right_left] = True

                blocked_right = blocked[:, -1]
                down_left_empty_right = grid[1:, -2] == 0
                rand_right = rng.integers(0, 2, size=blocked_right.shape)
                move_left_right = blocked_right & (rand_right == 1) & down_left_empty_right
                newgrid[1:, -2][move_left_right] = grid[:-1, -1][move_left_right]
                moved_from[:-1, -1][move_left_right] = True

            if grain_amount_width > 2:
                blocked_mid = blocked[:, 1:-1]
                down_right_empty_mid = grid[1:, 2:] == 0
                down_left_empty_mid = grid[1:, :-2] == 0
                rand_mid = rng.integers(0, 2, size=blocked_mid.shape)
                move_right = blocked_mid & (rand_mid == 0) & down_right_empty_mid
                move_left = blocked_mid & (rand_mid == 1) & down_left_empty_mid
                newgrid[1:, 2:][move_right] = grid[:-1, 1:-1][move_right]
                newgrid[1:, :-2][move_left] = grid[:-1, 1:-1][move_left]
                moved_from[:-1, 1:-1][move_right | move_left] = True

            stay = occupied & ~moved_from
            newgrid[stay] = grid[stay]
        else:
            newgrid[occupied] = grid[occupied]

        grid = newgrid
        occupied = grid >= 1
        if rainbow:
            grid[occupied] = (grid[occupied] + 1) % 360
            grid[grid < 1] = 1
        positions = np.argwhere(occupied)
        for row, column in positions:
            color.hsva = (grid[row, column], saturation, value, 100)
            pygame.draw.rect(screen, color, (column * grain_size, row * grain_size, grain_size, grain_size))
        
        pygame.display.flip()
        clock.tick(fps)

main()
