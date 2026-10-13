from dataclasses import dataclass
import random

@dataclass(frozen=True)
class World_config:
    row: int = 32
    col: int = 32

    seed: int = random.randint(0, 256)
