"""Command-line entry point for the terrain generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from config import TerrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reproducible terrain-generation workspace."
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--width", type=int, default=512, help="map width in pixels")
    parser.add_argument("--height", type=int, default=512, help="map height in pixels")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = TerrainConfig(
            seed=args.seed,
            width=args.width,
            height=args.height,
            octaves=args.octaves,
            persistence=args.persistence,
            lacunarity=args.lacunarity,
        )
    except ValueError as error:
        parser.error(str(error))

    metadata_path = prepare_run(config, args.output_dir)
    print(f"Prepared terrain run: {metadata_path}")
    print("Stage 0 complete. Heightmap generation will be added in stage 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
