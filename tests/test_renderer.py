from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from matplotlib import image as matplotlib_image

from render.map_renderer import save_heightmap_preview


class HeightmapPreviewTests(unittest.TestCase):
    def test_saves_exact_size_grayscale_png(self) -> None:
        heightmap = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape(5, 7)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "preview.png"
            returned_path = save_heightmap_preview(heightmap, output_path)
            image = matplotlib_image.imread(output_path)

        self.assertEqual(returned_path, output_path)
        self.assertEqual(image.shape[:2], (5, 7))
        np.testing.assert_array_equal(image[..., 0], image[..., 1])
        np.testing.assert_array_equal(image[..., 1], image[..., 2])

    def test_rejects_invalid_heightmaps(self) -> None:
        invalid_heightmaps = (
            np.zeros((2, 2, 2), dtype=np.float32),
            np.array([[0.0, np.nan]], dtype=np.float32),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "preview.png"
            for heightmap in invalid_heightmaps:
                with self.subTest(shape=heightmap.shape):
                    with self.assertRaises(ValueError):
                        save_heightmap_preview(heightmap, output_path)


if __name__ == "__main__":
    unittest.main()
