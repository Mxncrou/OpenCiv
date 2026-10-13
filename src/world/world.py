from flax import struct
import jax
import jax.numpy as jnp

from src.world.blocs import GROUND_TABLE

@struct.dataclass
class World:
    time: jax.Array
    key: jax.Array

    Ground_grid: jax.Array
    Constr_grid: jax.Array
    
    @classmethod
    def generate(cls, config):
        row, col, seed = config.row, config.col, config.seed
        return cls(
                time = jnp.zeros((), dtype=jnp.uint32),
                key = jax.random.PRNGKey(seed),
                Ground_grid = jnp.zeros((row, col), dtype=jnp.uint16),
                Constr_grid = jnp.zeros((row, col), dtype=jnp.uint16)
                )

    def add_beacon(self):
        key, subkey = jax.random.split(self.key)
        pos = jax.random.randint(subkey, (2,), 0, jnp.array(self.Constr_grid.shape))
        print(pos)
        return self.replace(
                Constr_grid = self.Constr_grid.at[tuple(pos)].set(1)
                )

    def get_state(self):
        f = lambda x: x.astype(jnp.float32)
        return jnp.concatenate([
            f(self.Ground_grid)[..., None],
            f(self.Constr_grid)[..., None]
            ], axis=-1)




