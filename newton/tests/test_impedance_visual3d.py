# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the impedance project's Visual3D export boundary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from projects.impedance_instron.cartesian.visual3d import (
    _ascii,
    _clock_rate,
    _main,
    export_manifest_template,
    inspect_visual3d_root,
    load_visual3d_export,
)


def _write_signal(path: Path, labels: list[str], values: np.ndarray) -> None:
    signal_name = path.name
    if signal_name.startswith("motion_FP1_"):
        labels = ["FP1"]
        axes = (signal_name[-5],)
        kind = path.parent.name
    elif "time" in signal_name:
        axes = ("X",)
        kind = "FRAME_NUMBERS"
    else:
        axes = (
            ("X", "Y", "Z") * (len(labels) // 3)
            if len(labels) % 3 == 0 and len(set(labels)) < len(labels)
            else ("X",) * len(labels)
        )
        kind = "TARGET"
    header = [
        "source",
        "\t".join(labels),
        "\t".join([kind] * len(labels)),
        "\t".join(["ORIGINAL"] * len(labels)),
        "ITEM\t" + "\t".join(axes),
    ]
    rows = [f"{i + 1}\t" + "\t".join(f"{value:.8f}" for value in row) for i, row in enumerate(values)]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


class Visual3DExportTest(unittest.TestCase):
    """Validate the Visual3D ASCII contract."""

    def test_accept_float32_rounded_clock_and_reject_jitter(self) -> None:
        """Accept timestamp rounding while rejecting a true timing deviation."""
        times = (np.arange(60000, dtype=np.float64) / 1000.0).astype(np.float32).astype(np.float64)
        self.assertEqual(_clock_rate(times, 1000.0, "analog clock"), 1000.0)
        times[30000] += 0.0001
        with self.assertRaisesRegex(ValueError, "uniform native sample clock"):
            _clock_rate(times, 1000.0, "analog clock")

    def test_load_export_converts_units_and_clocks(self) -> None:
        """Convert declared units and construct separate point and analog clocks."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "FORCE").mkdir()
            (root / "FREEMOMENT").mkdir()
            (root / "COFP").mkdir()
            marker = np.arange(12, dtype=float).reshape(2, 6)
            marker[1, 1] = np.nan
            _write_signal(root / "motion_all_targets.txt", ["LHIP"] * 3 + ["LTOE"] * 3, marker)
            for name, values in (("FORCE", 6), ("FREEMOMENT", 6), ("COFP", 6)):
                for axis in "XYZ":
                    _write_signal(
                        root / name / f"motion_FP1_{axis}.txt", [axis], np.arange(values, dtype=float)[:, None]
                    )
            _write_signal(root / "motion_time.txt", ["TIME"], np.arange(2, dtype=float)[:, None] / 100.0)
            _write_signal(root / "motion_analog_time.txt", ["ANALOGTIME"], np.arange(6, dtype=float)[:, None] / 500.0)
            manifest = {
                "schema": "visual3d_export_manifest_1",
                "point_rate_hz": 100.0,
                "analog_rate_hz": 500.0,
                "marker_units": "cm",
                "force_units": "N",
                "moment_units": "N*m",
                "cop_units": "m",
                "up_axis": "+Z",
                "forward_axis": "-Y",
            }
            (root / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            trial = load_visual3d_export(root)
            np.testing.assert_allclose(
                trial.marker_positions_m[0, 0], (marker[0, :3] * 0.01) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            )
            self.assertEqual(trial.marker_positions_m.shape, (2, 2, 3))
            self.assertEqual(trial.force_n.shape, (6, 3))
            self.assertFalse(trial.marker_valid[1, 0])
            self.assertAlmostEqual(trial.marker_time_s[-1], 0.01)
            self.assertAlmostEqual(trial.analog_time_s[-1], 0.01)

    def test_reject_missing_manifest(self) -> None:
        """Reject exports without explicit rate and unit metadata."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing .*visual3d_manifest"):
                load_visual3d_export(directory)

    def test_reject_bad_item_indices_and_axes(self) -> None:
        """Reject malformed Visual3D row indices and component headers."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signal.txt"
            _write_signal(path, ["LHIP"] * 3, np.ones((2, 3)))
            text = path.read_text(encoding="utf-8").replace("2\t1.00000000", "4\t1.00000000")
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "noncontiguous ITEM"):
                _ascii(path)
            path.write_text(
                path.read_text(encoding="utf-8").replace("ITEM\tX\tY\tZ", "ITEM\tX\tQ\tZ"), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "matching label/type/folder"):
                _ascii(path)

    def test_reject_missing_axes_and_nonmonotonic_clock(self) -> None:
        """Reject manifests without axes and clocks with repeated timestamps."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "FORCE").mkdir()
            (root / "FREEMOMENT").mkdir()
            (root / "COFP").mkdir()
            _write_signal(root / "motion_all_targets.txt", ["LHIP"] * 3, np.ones((2, 3)))
            for name in ("FORCE", "FREEMOMENT", "COFP"):
                for axis in "XYZ":
                    _write_signal(root / name / f"motion_FP1_{axis}.txt", [axis], np.ones((2, 1)))
            _write_signal(root / "motion_time.txt", ["TIME"], np.array([[0.0], [0.0]]))
            _write_signal(root / "motion_analog_time.txt", ["ANALOGTIME"], np.array([[0.0], [0.1]]))
            manifest = {
                "schema": "visual3d_export_manifest_1",
                "point_rate_hz": 100.0,
                "analog_rate_hz": 10.0,
                "marker_units": "m",
                "force_units": "N",
                "moment_units": "N*m",
                "cop_units": "m",
            }
            (root / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required fields.*up_axis"):
                load_visual3d_export(root)
            manifest.update(up_axis="+Z", forward_axis="-Y")
            (root / "visual3d_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                load_visual3d_export(root)

    def test_discover_generic_trial_inventory(self) -> None:
        """Discover a non-FR3 trial directory and report missing channels."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "subject"
            root.mkdir()
            trial = root / "walk_a"
            trial.mkdir()
            (trial / "motion_all_targets.txt").write_text("source\n", encoding="utf-8")
            report = inspect_visual3d_root(root)
            self.assertEqual(report["trials"][0]["name"], "walk_a")
            self.assertIn("FORCE/motion_FP1_X.txt", report["trials"][0]["missing_files"])

    def test_template_is_conservative(self) -> None:
        """Write a manifest template that requires physical metadata to be filled in."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            export_manifest_template(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(data["point_rate_hz"])
            self.assertIsNone(data["up_axis"])

    def test_parse_actual_f01_headers_when_present(self) -> None:
        """Parse the supplied Visual3D header shape without decoding assumptions."""
        path = Path("data/F01/FR3_1/motion_all_targets.txt")
        if not path.exists():
            self.skipTest("F01 export is not present")
        labels, values = _ascii(path)
        self.assertEqual(len(labels), 32)
        self.assertEqual(values.shape, (12000, 96))

    def test_normalize_records_sources_and_refuses_overwrite(self) -> None:
        """Record consumed source hashes and refuse replacing a normalized artifact."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "FORCE").mkdir()
            (root / "FREEMOMENT").mkdir()
            (root / "COFP").mkdir()
            _write_signal(root / "motion_all_targets.txt", ["LHIP"] * 3, np.ones((2, 3)))
            for name in ("FORCE", "FREEMOMENT", "COFP"):
                for axis in "XYZ":
                    _write_signal(root / name / f"motion_FP1_{axis}.txt", [axis], np.ones((2, 1)))
            _write_signal(root / "motion_time.txt", ["TIME"], np.array([[5.0], [5.01]]))
            _write_signal(root / "motion_analog_time.txt", ["ANALOGTIME"], np.array([[7.0], [7.01]]))
            manifest = {
                "schema": "visual3d_manifest_1",
                "point_rate_hz": 100.0,
                "analog_rate_hz": 100.0,
                "marker_units": "m",
                "force_units": "N",
                "moment_units": "N*m",
                "cop_units": "m",
                "up_axis": "+Z",
                "forward_axis": "-Y",
            }
            manifest_path = root / "visual3d_manifest.json"
            manifest_path.write_text(json.dumps({**manifest, "schema": "visual3d_export_manifest_1"}), encoding="utf-8")
            output = root / "normalized.npz"
            _main(["normalize", str(root), "--output", str(output)])
            with np.load(output, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"]))
                self.assertAlmostEqual(float(archive["marker_time_s"][0]), 5.0)
                self.assertAlmostEqual(float(archive["force_time_s"][0]), 7.0)
            self.assertIn("motion_all_targets.txt", " ".join(metadata["source_files"]))
            self.assertTrue(any(path.endswith("visual3d_manifest.json") for path in metadata["source_files"]))
            with self.assertRaises(FileExistsError):
                _main(["normalize", str(root), "--output", str(output)])

    def test_reject_inconsistent_signal_length(self) -> None:
        """Reject force channels that cannot be aligned to one analog clock."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "FORCE").mkdir()
            (root / "FREEMOMENT").mkdir()
            (root / "COFP").mkdir()
            _write_signal(root / "motion_all_targets.txt", ["LHIP"] * 3, np.ones((2, 3)))
            for name in ("FORCE", "FREEMOMENT", "COFP"):
                for axis in "XYZ":
                    _write_signal(
                        root / name / f"motion_FP1_{axis}.txt", [axis], np.ones((3 if name == "FORCE" else 2, 1))
                    )
            _write_signal(root / "motion_time.txt", ["TIME"], np.arange(2, dtype=float)[:, None] / 100.0)
            _write_signal(root / "motion_analog_time.txt", ["ANALOGTIME"], np.arange(3, dtype=float)[:, None] / 500.0)
            (root / "visual3d_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "visual3d_export_manifest_1",
                        "point_rate_hz": 100.0,
                        "analog_rate_hz": 500.0,
                        "marker_units": "m",
                        "force_units": "N",
                        "moment_units": "N*m",
                        "cop_units": "m",
                        "up_axis": "+Z",
                        "forward_axis": "-Y",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must contain one column and 3 samples"):
                load_visual3d_export(root)


if __name__ == "__main__":
    unittest.main()
