import pygame

from config.config import World_config
from src.world.world import World
from src.world.blocs import GROUND_TABLE, CONSTR_TABLE


def main():
    world = World.generate(World_config)
    world = world.add_beacon()
    state = world.get_state()    
 
    pygame.init()
    screen = pygame.display.set_mode((state.shape[0] * 16, state.shape[1] * 16))
    pygame.display.set_caption("Test")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                rect = (row * 16, col * 16, 16, 16)
                pygame.draw.rect(screen, GROUND_TABLE.colors[world.Ground_grid[row, col]], rect)
                if world.Constr_grid[row, col] != 0:
                    pygame.draw.rect(screen, CONSTR_TABLE.colors[world.Constr_grid[row, col]], rect)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

