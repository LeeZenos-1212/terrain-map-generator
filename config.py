"""Configuration models for terrain generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TerrainConfig:
    """Parameters shared by terrain generation and metadata output."""

    seed: int = 42
    width: int = 512
    height: int = 512
    base_frequency: int = 2
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers")
        if self.base_frequency <= 0:
            raise ValueError("base_frequency must be a positive integer")
        if self.octaves <= 0:
            raise ValueError("octaves must be a positive integer")
        if not 0.0 < self.persistence <= 1.0:
            raise ValueError("persistence must be greater than 0 and at most 1")
        if self.lacunarity <= 1.0:
            raise ValueError("lacunarity must be greater than 1")

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation of this configuration."""

        return asdict(self)
