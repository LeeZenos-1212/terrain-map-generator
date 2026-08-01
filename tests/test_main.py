from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from matplotlib import image as matplotlib_image

from config import TerrainConfig
from main import main, prepare_run


class PrepareRunTests(unittest.TestCase):
    def test_writes_seed_specific_metadata(self) -> None:
        config = TerrainConfig(seed=20260801, width=64, height=32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_path = prepare_run(config, Path(temporary_directory))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata_path.parent.name, "20260801")
        self.assertEqual(metadata, config.to_dict())

    def test_main_saves_reproducible_heightmap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            arguments = [
                "--seed",
                "123",
                "--width",
                "53",
                "--height",
                "37",
                "--base-frequency",
                "3",
                "--octaves",
                "4",
                "--output-dir",
                temporary_directory,
            ]

            with redirect_stdout(StringIO()):
                exit_code = main(arguments)
                first = np.load(
                    Path(temporary_directory) / "123" / "heightmap.npy",
                    allow_pickle=False,
                )
                main(arguments)
                second = np.load(
                    Path(temporary_directory) / "123" / "heightmap.npy",
                    allow_pickle=False,
                )
                preview = matplotlib_image.imread(
                    Path(temporary_directory) / "123" / "heightmap_preview.png"
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(first.shape, (37, 53))
        self.assertEqual(first.dtype, np.float32)
        self.assertEqual(preview.shape[:2], (37, 53))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
