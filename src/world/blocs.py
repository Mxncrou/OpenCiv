from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class Bloc:
    colors: jax.Array

    @classmethod
    def create_table(cls, colors):
        return cls(
                colors = jnp.array(colors, dtype=jnp.uint8)
                )

GROUND_TABLE = Bloc.create_table([[0, 255, 0], [0, 0, 255]])
CONSTR_TABLE = Bloc.create_table([[0, 0, 0], [255, 0, 0]])
