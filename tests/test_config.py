from __future__ import annotations

import unittest

from config import TerrainConfig


class TerrainConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        config = TerrainConfig()

        self.assertEqual(config.seed, 42)
        self.assertEqual((config.height, config.width), (512, 512))
        self.assertEqual(config.base_frequency, 2)

    def test_non_positive_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "width and height"):
            TerrainConfig(width=0)

    def test_invalid_noise_parameters_are_rejected(self) -> None:
        invalid_values = (
            {"base_frequency": 0},
            {"octaves": 0},
            {"persistence": 0.0},
            {"persistence": 1.1},
            {"lacunarity": 1.0},
        )

        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TerrainConfig(**values)


if __name__ == "__main__":
    unittest.main()
