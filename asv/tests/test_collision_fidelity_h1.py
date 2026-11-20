# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate H1 initialization inputs without importing a legacy runtime."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))


class TestH1Inputs(unittest.TestCase):
    """Check complete public state and exact authored hand trajectories."""

    def test_four_complete_input_fixtures(self):
        """Each input has free cloth, two commands, and no solver-history data."""
        from collision_fidelity.h1_inputs import load_input  # noqa: PLC0415

        counts = {"j3_h1_jacket": 33703, "r7_h1_towel": 3438, "r7_h1_sheet": 5806, "r7_h1_gusset_bag": 5578}
        for name, count in counts.items():
            with self.subTest(case=name):
                metadata, arrays = load_input(name)
                self.assertEqual(arrays["state0.particle_q"].shape, (count, 3))
                self.assertEqual(arrays["rest.vertices"].shape, (count, 3))
                self.assertTrue(
                    all(not key.startswith(("solver", "prepare", "driver0._", "driver1._")) for key in arrays)
                )
                self.assertEqual(metadata["case_id"], name)
                for index in range(2):
                    times = arrays[f"driver{index}.times"]
                    self.assertTrue(np.all(np.diff(times) > 0))
                    self.assertEqual(arrays[f"driver{index}.fingers"].shape, (len(times), 12))

    def test_changed_fixture_metadata_is_rejected(self):
        """Reject altered commands or metadata even when the array file is intact."""
        from collision_fidelity import h1_inputs  # noqa: PLC0415

        source = Path(h1_inputs.__file__).with_name("assets")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "assets" / "h1"
            folder.mkdir(parents=True)
            shutil.copyfile(source / "manifest.json", root / "assets" / "manifest.json")
            case_id = "r7_h1_towel"
            shutil.copyfile(source / "h1" / f"{case_id}.npz", folder / f"{case_id}.npz")
            metadata = json.loads((source / "h1" / f"{case_id}.json").read_text())
            metadata["case_id"] = "altered"
            (folder / f"{case_id}.json").write_text(json.dumps(metadata))
            with patch.object(h1_inputs, "__file__", str(root / "h1_inputs.py")):
                with self.assertRaisesRegex(ValueError, "metadata checksum"):
                    h1_inputs.load_input(case_id)

    def test_towel_closed_thumb_correction_is_explicit(self):
        """Only closed thumb/mimic columns differ from the preserved input."""
        from collision_fidelity.h1_inputs import load_input  # noqa: PLC0415

        metadata, arrays = load_input("r7_h1_towel")
        self.assertEqual(metadata["command_adjustment"]["rows"], list(range(1, 8)))
        self.assertEqual(metadata["command_adjustment"]["columns"], [1, 2, 3])
        for index, pitch in enumerate((0.16051281243702709, 0.1629728749436688)):
            np.testing.assert_array_equal(arrays[f"driver{index}.fingers"][1:8, 1], np.full(7, pitch, np.float32))


if __name__ == "__main__":
    unittest.main()
