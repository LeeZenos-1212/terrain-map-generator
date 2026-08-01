from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config import TerrainConfig
from main import prepare_run


class PrepareRunTests(unittest.TestCase):
    def test_writes_seed_specific_metadata(self) -> None:
        config = TerrainConfig(seed=20260801, width=64, height=32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_path = prepare_run(config, Path(temporary_directory))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata_path.parent.name, "20260801")
        self.assertEqual(metadata, config.to_dict())


if __name__ == "__main__":
    unittest.main()
