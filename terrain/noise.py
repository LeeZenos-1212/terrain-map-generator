"""Deterministic noise primitives used to build terrain heightmaps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Float32Array = NDArray[np.float32]


def _fade(values: Float32Array) -> Float32Array:
    """Smooth interpolation weights with zero slope at both endpoints."""

    return values * values * values * (values * (values * 6.0 - 15.0) + 10.0)


def _validate_grid(
    shape: tuple[int, int],
    resolution: tuple[int, int],
) -> None:
    if len(shape) != 2 or any(size <= 0 for size in shape):
        raise ValueError("shape must contain two positive integers")
    if len(resolution) != 2 or any(periods <= 0 for periods in resolution):
        raise ValueError("resolution must contain two positive integers")


def _generate_gradient_noise_2d(
    shape: tuple[int, int],
    resolution: tuple[int, int],
    random: np.random.Generator,
) -> Float32Array:
    height, width = shape
    periods_y, periods_x = resolution

    angles = random.uniform(
        0.0,
        2.0 * np.pi,
        size=(periods_y + 1, periods_x + 1),
    )
    gradient_x = np.cos(angles).astype(np.float32)
    gradient_y = np.sin(angles).astype(np.float32)

    sample_y = np.arange(height, dtype=np.float32) * (periods_y / height)
    sample_x = np.arange(width, dtype=np.float32) * (periods_x / width)
    cell_y = np.floor(sample_y).astype(np.intp)
    cell_x = np.floor(sample_x).astype(np.intp)
    offset_y = sample_y - cell_y
    offset_x = sample_x - cell_x

    x0 = offset_x[None, :]
    y0 = offset_y[:, None]
    x1 = x0 - 1.0
    y1 = y0 - 1.0

    rows = cell_y[:, None]
    columns = cell_x[None, :]
    dot_top_left = gradient_x[rows, columns] * x0 + gradient_y[rows, columns] * y0
    dot_top_right = (
        gradient_x[rows, columns + 1] * x1
        + gradient_y[rows, columns + 1] * y0
    )
    dot_bottom_left = (
        gradient_x[rows + 1, columns] * x0
        + gradient_y[rows + 1, columns] * y1
    )
    dot_bottom_right = (
        gradient_x[rows + 1, columns + 1] * x1
        + gradient_y[rows + 1, columns + 1] * y1
    )

    weight_x = _fade(offset_x)[None, :]
    weight_y = _fade(offset_y)[:, None]
    top = dot_top_left + weight_x * (dot_top_right - dot_top_left)
    bottom = dot_bottom_left + weight_x * (dot_bottom_right - dot_bottom_left)
    noise = np.sqrt(np.float32(2.0)) * (top + weight_y * (bottom - top))

    return np.asarray(noise, dtype=np.float32)


def generate_gradient_noise_2d(
    shape: tuple[int, int],
    resolution: tuple[int, int],
    seed: int,
) -> Float32Array:
    """Generate one layer of deterministic two-dimensional gradient noise.

    ``shape`` is ``(height, width)``. ``resolution`` controls how many noise
    periods appear along each axis. Unlike many compact Perlin-noise
    implementations, the output dimensions do not need to be divisible by the
    resolution.
    """

    _validate_grid(shape, resolution)
    return _generate_gradient_noise_2d(shape, resolution, np.random.default_rng(seed))


def generate_fractal_noise_2d(
    shape: tuple[int, int],
    base_resolution: tuple[int, int],
    *,
    seed: int,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> Float32Array:
    """Combine gradient-noise octaves into a normalized terrain heightmap.

    Each octave increases spatial frequency by ``lacunarity`` and decreases
    amplitude by ``persistence``. The returned ``float32`` array is normalized
    to the closed interval ``0.0`` through ``1.0``.
    """

    _validate_grid(shape, base_resolution)
    if octaves <= 0:
        raise ValueError("octaves must be a positive integer")
    if not 0.0 < persistence <= 1.0:
        raise ValueError("persistence must be greater than 0 and at most 1")
    if lacunarity <= 1.0:
        raise ValueError("lacunarity must be greater than 1")

    random = np.random.default_rng(seed)
    combined = np.zeros(shape, dtype=np.float32)
    amplitude = np.float32(1.0)

    for octave in range(octaves):
        frequency_scale = lacunarity**octave
        resolution = tuple(
            max(1, round(periods * frequency_scale))
            for periods in base_resolution
        )
        combined += amplitude * _generate_gradient_noise_2d(
            shape,
            resolution,
            random,
        )
        amplitude *= np.float32(persistence)

    minimum = np.min(combined)
    value_range = np.max(combined) - minimum
    if value_range <= np.finfo(np.float32).eps:
        return np.zeros(shape, dtype=np.float32)

    return np.asarray((combined - minimum) / value_range, dtype=np.float32)
