import jax.numpy as jnp

from config.config import World_config
from src.world.world import World

def main():
    world = World.generate(World_config)
    state = world.get_state()    
    print("state shape:", state.shape)


if __name__ == "__main__":
    main()

