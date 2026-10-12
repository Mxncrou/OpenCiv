from dataclasses import dataclass


@dataclass(frozen=True)
class World_config:
    row: int = 256
    col: int = 256
