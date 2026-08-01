"""Image outputs derived from terrain heightmaps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import image as matplotlib_image
from numpy.typing import NDArray


def save_heightmap_preview(
    heightmap: NDArray[np.float32],
    output_path: Path,
) -> Path:
    """Save a normalized heightmap as a borderless grayscale PNG."""

    if heightmap.ndim != 2:
        raise ValueError("heightmap must be a two-dimensional array")
    if not np.isfinite(heightmap).all():
        raise ValueError("heightmap must contain only finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_image.imsave(
        output_path,
        heightmap,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    return output_path
