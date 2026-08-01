from __future__ import annotations

import unittest

import numpy as np

from terrain.noise import generate_fractal_noise_2d, generate_gradient_noise_2d


class GradientNoiseTests(unittest.TestCase):
    def test_same_seed_produces_identical_noise(self) -> None:
        first = generate_gradient_noise_2d((37, 53), (3, 5), seed=42)
        second = generate_gradient_noise_2d((37, 53), (3, 5), seed=42)

        np.testing.assert_array_equal(first, second)

    def test_different_seed_changes_noise(self) -> None:
        first = generate_gradient_noise_2d((32, 32), (4, 4), seed=1)
        second = generate_gradient_noise_2d((32, 32), (4, 4), seed=2)

        self.assertFalse(np.array_equal(first, second))

    def test_supports_non_divisible_dimensions(self) -> None:
        noise = generate_gradient_noise_2d((37, 53), (3, 5), seed=7)

        self.assertEqual(noise.shape, (37, 53))
        self.assertEqual(noise.dtype, np.float32)
        self.assertTrue(np.isfinite(noise).all())

    def test_invalid_dimensions_are_rejected(self) -> None:
        invalid_arguments = (
            ((0, 32), (4, 4)),
            ((32, -1), (4, 4)),
            ((32, 32), (0, 4)),
            ((32, 32), (4, -1)),
        )

        for shape, resolution in invalid_arguments:
            with self.subTest(shape=shape, resolution=resolution):
                with self.assertRaises(ValueError):
                    generate_gradient_noise_2d(shape, resolution, seed=42)


class FractalNoiseTests(unittest.TestCase):
    def test_same_seed_and_parameters_are_reproducible(self) -> None:
        parameters = {
            "shape": (47, 61),
            "base_resolution": (2, 3),
            "seed": 42,
            "octaves": 4,
            "persistence": 0.55,
            "lacunarity": 1.8,
        }

        first = generate_fractal_noise_2d(**parameters)
        second = generate_fractal_noise_2d(**parameters)

        np.testing.assert_array_equal(first, second)

    def test_output_is_normalized_float32(self) -> None:
        heightmap = generate_fractal_noise_2d(
            (64, 96),
            (2, 3),
            seed=7,
            octaves=5,
        )

        self.assertEqual(heightmap.shape, (64, 96))
        self.assertEqual(heightmap.dtype, np.float32)
        self.assertTrue(np.isfinite(heightmap).all())
        self.assertEqual(float(heightmap.min()), 0.0)
        self.assertEqual(float(heightmap.max()), 1.0)

    def test_different_seed_changes_heightmap(self) -> None:
        first = generate_fractal_noise_2d((32, 32), (2, 2), seed=1)
        second = generate_fractal_noise_2d((32, 32), (2, 2), seed=2)

        self.assertFalse(np.array_equal(first, second))

    def test_constant_heightmap_is_normalized_safely(self) -> None:
        heightmap = generate_fractal_noise_2d((1, 1), (1, 1), seed=42)

        np.testing.assert_array_equal(heightmap, np.zeros((1, 1), dtype=np.float32))

    def test_invalid_fractal_parameters_are_rejected(self) -> None:
        invalid_parameters = (
            {"octaves": 0},
            {"persistence": 0.0},
            {"persistence": 1.1},
            {"lacunarity": 1.0},
        )

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    generate_fractal_noise_2d(
                        (32, 32),
                        (2, 2),
                        seed=42,
                        **parameters,
                    )


if __name__ == "__main__":
    unittest.main()
