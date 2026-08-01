"""Command-line entry point for the terrain generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from config import TerrainConfig
from render.map_renderer import save_heightmap_preview
from terrain.noise import generate_fractal_noise_2d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reproducible terrain-generation workspace."
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--width", type=int, default=512, help="map width in pixels")
    parser.add_argument("--height", type=int, default=512, help="map height in pixels")
    parser.add_argument(
        "--base-frequency",
        type=int,
        default=2,
        help="number of large-scale noise periods per axis (default: 2)",
    )
    parser.add_argument("--octaves", type=int, default=6, help="number of noise layers")
    parser.add_argument(
        "--persistence", type=float, default=0.5, help="amplitude multiplier per octave"
    )
    parser.add_argument(
        "--lacunarity", type=float, default=2.0, help="frequency multiplier per octave"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="root directory for generated files (default: output)",
    )
    return parser


def prepare_run(config: TerrainConfig, output_root: Path) -> Path:
    """Create one seed-specific output directory and save its configuration."""

    run_dir = output_root / str(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def generate_heightmap(config: TerrainConfig) -> NDArray[np.float32]:
    """Generate a normalized heightmap from one validated configuration."""

    return generate_fractal_noise_2d(
        (config.height, config.width),
        (config.base_frequency, config.base_frequency),
        seed=config.seed,
        octaves=config.octaves,
        persistence=config.persistence,
        lacunarity=config.lacunarity,
    )


def save_heightmap(heightmap: NDArray[np.float32], run_dir: Path) -> Path:
    """Save a heightmap without converting or embedding Python objects."""

    heightmap_path = run_dir / "heightmap.npy"
    np.save(heightmap_path, heightmap, allow_pickle=False)
    return heightmap_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = TerrainConfig(
            seed=args.seed,
            width=args.width,
            height=args.height,
            base_frequency=args.base_frequency,
            octaves=args.octaves,
            persistence=args.persistence,
            lacunarity=args.lacunarity,
        )
    except ValueError as error:
        parser.error(str(error))

    metadata_path = prepare_run(config, args.output_dir)
    heightmap = generate_heightmap(config)
    heightmap_path = save_heightmap(heightmap, metadata_path.parent)
    preview_path = save_heightmap_preview(
        heightmap,
        metadata_path.parent / "heightmap_preview.png",
    )
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved heightmap: {heightmap_path}")
    print(f"Saved preview: {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
