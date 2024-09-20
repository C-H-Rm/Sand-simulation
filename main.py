import pygame
import random

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
    grid = [[0] * grain_amount_width for _ in range(grain_amount_height)]
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
                if grid[y // grain_size][x // grain_size] == 0:
                    grid[y // grain_size][x // grain_size] = hue

            elif -1 < x < screen_width and -1 < y < screen_height:
                for yoffset in range(-placement_size, placement_size):
                    for xoffset in range(-placement_size, placement_size):
                        if -1 < y // grain_size + yoffset < grain_amount_height and -1 < x // grain_size + xoffset < grain_amount_width:
                            if grid[y // grain_size + yoffset][x // grain_size + xoffset] == 0:
                                grid[y // grain_size + yoffset][x // grain_size + xoffset] = hue
        
        if pygame.mouse.get_pressed()[2]:
            x, y = pygame.mouse.get_pos()

            if -1 < x < screen_width and -1 < y < screen_height:
                grid[y // grain_size][x // grain_size] = 0

        newgrid = [[0] * grain_amount_width for _ in range(grain_amount_height)]

        for row in range(0, grain_amount_height):
            for column in range(0, grain_amount_width):
                if grid[row][column] >= 1:
                    if row < len(newgrid)-1:
                        if grid[row+1][column] == 0 and grid[row][column] >= 1:
                            newgrid[row+1][column] = grid[row][column]

                        elif column < len(grid[0])-1:
                            num = random.choice([0, 1])
                            
                            if grid[row+1][column+1] == 0 and grid[row][column] >= 1 and num == 0:
                                newgrid[row+1][column+1] = grid[row][column]

                            elif grid[row+1][column-1] == 0 and grid[row][column] >= 1 and num == 1 and not column == 0:
                                newgrid[row+1][column-1] = grid[row][column]

                            elif grid[row][column] >= 1:
                                newgrid[row][column] = grid[row][column]

                        elif grid[row+1][column-1] == 0 and grid[row][column] >= 1 and num == 1 and not column == 0:
                                newgrid[row+1][column-1] = grid[row][column]

                        elif grid[row][column] >= 1:
                                newgrid[row][column] = grid[row][column]

                    elif grid[row][column] >= 1:
                        newgrid[row][column] = grid[row][column]

        grid = newgrid
        for row in range(0, grain_amount_height):
            for column in range(0, grain_amount_width):
                if grid[row][column] >= 1:
                    grid[row][column] = (grid[row][column] + (rainbow*1)) % 360
                    if grid[row][column] < 1:
                        grid[row][column] = 1
                    color.hsva = (grid[row][column], saturation, value, 100)
                    pygame.draw.rect(screen, color, (column*grain_size, row*grain_size, grain_size, grain_size))
        
        pygame.display.flip()
        clock.tick(fps)

main()