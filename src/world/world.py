from flax import struct
import jax
import jax.numpy as jnp

from src.world.blocs import GROUND_TABLE

@struct.dataclass
class World:
    time: jax.Array

    Ground_grid: jax.Array
    Constr_grid: jax.Array
    
    @classmethod
    def generate(cls, config):
        row, col = config.row, config.col
        return cls(
                time = jnp.zeros((), dtype=jnp.uint32),
                Ground_grid = jnp.zeros((row, col), dtype=jnp.uint16),
                Constr_grid = jnp.zeros((row, col), dtype=jnp.uint16)
                )

    def get_state(self):
        f = lambda x: x.astype(jnp.float32)
        return jnp.concatenate([
            f(self.Ground_grid)[..., None],
            f(self.Constr_grid)[..., None]
            ], axis=-1)


