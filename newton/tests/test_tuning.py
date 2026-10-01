# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tuning contracts."""

from __future__ import annotations

import dataclasses
import json
import math
import tempfile
import unittest

import numpy as np

from newton._src.tuning.data_source import CableDataSource
from newton._src.tuning.evidence import CableEvidenceBundle, CableRecording
from newton._src.tuning.result import STATUS_OK, CableCalibrationResult
from newton._src.tuning.schema import SCHEMA_VERSION
from newton._src.tuning.trajectory import MaterializedTrajectorySource

# Quaternions are (qx, qy, qz, qw). A turn by angle a about z is (0, 0, sin(a/2), cos(a/2)).
IDENTITY = (0.0, 0.0, 0.0, 1.0)
QUARTER_TURN_Z = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
EIGHTH_TURN_Z = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))


def make_recording(**overrides):
    """A valid driven recording with three reference frames."""
    values = {
        "label": "cam0",
        "recording_key": "rec0",
        "applied_settings": {"crop": [0, 0, 32, 24]},
        "rgb_frames": ["frames/cam0/0.png", "frames/cam0/1.png", "frames/cam0/2.png"],
        "masks": ["masks/cam0/0.png", "masks/cam0/1.png", "masks/cam0/2.png"],
        "frame_timestamps": [0.0, 0.1, 0.2],
        "sensor_pos": [1.0, 0.0, 0.5],
        "sensor_quat": [0.0, 0.0, 0.0, 1.0],
        "camera_intrinsics": [32, 24, 40.0, 40.0, 16.0, 12.0],
        "cable_start": [0.0, 0.0, 0.8],
        "driven": True,
        "trajectory_path": "trajectories/rec0.json",
        "start_ns": 1_000_000_000,
    }
    values.update(overrides)
    return CableRecording(**values)


def make_trajectory(**overrides):
    """A valid three-sample trajectory: 1 m along x per 10 ns, a quarter turn about z at the end."""
    values = {
        "source_id": "rec0",
        "timestamps_ns": [0, 10, 20],
        "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        "quaternions": [IDENTITY, IDENTITY, QUARTER_TURN_Z],
    }
    values.update(overrides)
    return MaterializedTrajectorySource(**values)


def make_result(**overrides):
    values = {
        "status": STATUS_OK,
        "fit": {"bend_stiffness": 0.35, "bend_angles": [[0.1, 0.0], [0.0, 0.2]]},
        "metrics": {"loss": 0.01, "iterations": 12},
        "provenance": {"run_id": "abc", "seed": 42},
    }
    values.update(overrides)
    return CableCalibrationResult(**values)


def json_round_trip(d):
    return json.loads(json.dumps(d))


class TestTuningContracts(unittest.TestCase):
    """Serialization and validation of the tuning contracts."""

    def test_contracts_round_trip_through_json(self):
        """Verify every contract survives to_dict/from_dict via a JSON round trip."""
        bundle = CableEvidenceBundle(recordings=[make_recording()])
        loaded = CableEvidenceBundle.from_dict(json_round_trip(bundle.to_dict()))
        self.assertEqual(loaded, bundle)
        loaded.validate()

        source = make_trajectory()
        loaded = MaterializedTrajectorySource.from_dict(json_round_trip(source.to_dict()))
        # JSON turns tuples into lists, so compare the stored form.
        self.assertEqual(json_round_trip(loaded.to_dict()), json_round_trip(source.to_dict()))
        loaded.validate()

        result = make_result()
        loaded = CableCalibrationResult.from_dict(json_round_trip(result.to_dict()))
        self.assertEqual(loaded, result)
        loaded.validate()

    def test_bundle_save_and_load(self):
        """Verify a bundle written to a directory loads back unchanged."""
        bundle = CableEvidenceBundle(recordings=[make_recording()])
        with tempfile.TemporaryDirectory() as tmp:
            bundle.save(tmp)
            self.assertEqual(CableEvidenceBundle.load(tmp), bundle)

    def test_unsupported_schema_version_is_refused(self):
        """Verify loading refuses a schema_version this build does not understand.

        Including an absent version, which predates versioning and cannot be read
        safely.
        """
        contracts = [
            (CableEvidenceBundle, CableEvidenceBundle(recordings=[make_recording()]).to_dict()),
            (MaterializedTrajectorySource, make_trajectory().to_dict()),
            (CableCalibrationResult, make_result().to_dict()),
        ]
        for cls, d in contracts:
            with self.subTest(contract=cls.__name__, case="future version"):
                with self.assertRaisesRegex(ValueError, "not supported"):
                    cls.from_dict({**d, "schema_version": SCHEMA_VERSION + 1})
            with self.subTest(contract=cls.__name__, case="absent version"):
                absent = {k: v for k, v in d.items() if k != "schema_version"}
                with self.assertRaisesRegex(ValueError, "no schema_version"):
                    cls.from_dict(absent)

    def test_unknown_fields_are_refused(self):
        """Verify loading names a field the format does not define, instead of failing in the constructor."""
        bundle = CableEvidenceBundle(recordings=[make_recording()]).to_dict()
        nested = json_round_trip(bundle)
        nested["recordings"][0]["crop"] = [0, 0, 32, 24]
        cases = [
            ("bundle", CableEvidenceBundle, {**bundle, "units": "meters"}, "units"),
            ("recording", CableEvidenceBundle, nested, "crop"),
            (
                "trajectory",
                MaterializedTrajectorySource,
                {**make_trajectory().to_dict(), "frame_names": None},
                "frame_names",
            ),
            ("result", CableCalibrationResult, {**make_result().to_dict(), "config": {}}, "config"),
        ]
        for name, cls, d, field_name in cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, f"unknown field.*{field_name}"):
                    cls.from_dict(d)

    def test_bundle_validation_names_the_offending_recording(self):
        """Verify bundle validation rejects malformed evidence, naming the recording."""
        cases = {
            "no frames": ({"rgb_frames": [], "masks": [], "frame_timestamps": []}, "no rgb frames"),
            "mask count": ({"masks": ["masks/cam0/0.png"]}, "mask"),
            "frame_timestamps length": ({"frame_timestamps": [0.0]}, "frame_timestamps"),
            "frame before start_ns": ({"frame_timestamps": [-0.1, 0.1, 0.2]}, "precede start_ns"),
            "repeated frame time": ({"frame_timestamps": [0.0, 0.1, 0.1]}, "strictly increasing"),
            "decreasing frame time": ({"frame_timestamps": [0.0, 0.2, 0.1]}, "strictly increasing"),
            "no camera position": ({"sensor_pos": None}, "sensor_pos"),
            "no camera orientation": ({"sensor_quat": None}, "sensor_quat"),
            "malformed camera orientation": ({"sensor_quat": [0.0, 0.0, 1.0]}, "sensor_quat"),
            "malformed intrinsics": ({"camera_intrinsics": [40.0, 40.0, 16.0, 12.0]}, "camera_intrinsics"),
            "no camera": ({"camera_intrinsics": None}, "camera_intrinsics or render_size"),
            "malformed render size": ({"camera_intrinsics": None, "render_size": [], "fov_deg": 60.0}, "render_size"),
            "no field of view": ({"camera_intrinsics": None, "render_size": [32, 24]}, "fov_deg"),
            "no cable start": ({"cable_start": None}, "cable_start"),
            "malformed cable start": ({"cable_start": [0.0, 0.8]}, "cable_start"),
            "driven without trajectory": ({"trajectory_path": None}, "trajectory_path"),
            "driven without start_ns": ({"start_ns": None}, "start_ns"),
        }
        for name, (overrides, message) in cases.items():
            with self.subTest(case=name):
                bad = make_recording(label="bad_view", **overrides)
                bundle = CableEvidenceBundle(recordings=[make_recording(), bad])
                with self.assertRaisesRegex(ValueError, f"recording 'bad_view'.*{message}"):
                    bundle.validate()

        with self.subTest(case="uncalibrated camera with render size and field of view"):
            uncalibrated = make_recording(camera_intrinsics=None, render_size=[32, 24], fov_deg=60.0)
            CableEvidenceBundle(recordings=[uncalibrated]).validate()

        with self.subTest(case="static recording without start_ns"):
            static = make_recording(driven=False, trajectory_path=None, start_ns=None)
            CableEvidenceBundle(recordings=[static]).validate()

        with self.subTest(case="duplicate label"):
            bundle = CableEvidenceBundle(recordings=[make_recording(), make_recording()])
            with self.assertRaisesRegex(ValueError, "recording 'cam0': duplicate label"):
                bundle.validate()

        with self.subTest(case="no recordings"):
            with self.assertRaisesRegex(ValueError, "no recordings"):
                CableEvidenceBundle().validate()

    def test_kept_frames_may_start_after_start_ns(self):
        """Verify frames trimmed by a time window still validate.

        A time window can keep only later frames, while start_ns still marks
        the start of the recording.
        """
        CableEvidenceBundle(recordings=[make_recording(frame_timestamps=[1.5, 1.6, 1.7])]).validate()

    def test_frame_timestamps_are_required(self):
        """Verify a recording cannot be built or loaded without frame timestamps."""
        with self.assertRaises(TypeError):
            CableRecording(label="cam0", recording_key="rec0", applied_settings={})
        stored = CableEvidenceBundle(recordings=[make_recording()]).to_dict()
        del stored["recordings"][0]["frame_timestamps"]
        with self.assertRaises(TypeError):
            CableEvidenceBundle.from_dict(stored)

    def test_result_validation(self):
        """Verify a result needs a known status and at least one fitted value."""
        make_result().validate()
        with self.assertRaisesRegex(ValueError, "unknown status"):
            make_result(status="done").validate()
        with self.assertRaisesRegex(ValueError, "no fitted values"):
            make_result(fit={}).validate()


class TestTuningDataSource(unittest.TestCase):
    """The drive-trajectory interface and its stored form."""

    def test_interface_cannot_be_instantiated(self):
        """Verify CableDataSource is abstract and refuses direct construction."""
        with self.assertRaises(TypeError):
            CableDataSource()

    def test_drive_buffer_samples_the_requested_grid(self):
        """Verify drive_buffer returns num_frames * sim_substeps poses at the given rate.

        A source returning the wrong count must raise rather than produce a drive
        that silently misaligns with the timeline.
        """

        @dataclasses.dataclass
        class RecordingSource(CableDataSource):
            source_id: str = "rec"
            drop: int = 0
            stamps: list = dataclasses.field(default_factory=list)

            def tcp_transforms(self, stamps_ns):
                self.stamps = list(stamps_ns)
                pose = ((0.0, 0.0, 0.0), IDENTITY)
                return [pose] * (len(stamps_ns) - self.drop)

        source = RecordingSource()
        buffer = source.drive_buffer(start_ns=1_000, num_frames=2, substep_rate=600, sim_substeps=3)
        self.assertEqual(buffer.shape, (6,))
        self.assertEqual(source.stamps, [1_000 + (k * 1_000_000_000) // 600 for k in range(6)])

        with self.assertRaisesRegex(ValueError, "asked for 6 pose"):
            RecordingSource(drop=1).drive_buffer(start_ns=1_000, num_frames=2, substep_rate=600, sim_substeps=3)


class TestTuningTrajectory(unittest.TestCase):
    """Interpolation and validation of a materialized trajectory."""

    def test_interpolates_between_samples(self):
        """Verify positions interpolate linearly and orientations along the shortest arc."""
        (pos_a, quat_a), (pos_b, quat_b) = make_trajectory().tcp_transforms([5, 15])
        np.testing.assert_allclose(pos_a, (0.5, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(quat_a, IDENTITY, atol=1e-6)
        np.testing.assert_allclose(pos_b, (1.5, 0.0, 0.0), atol=1e-6)
        # Halfway through a quarter turn about z is an eighth turn.
        np.testing.assert_allclose(quat_b, EIGHTH_TURN_Z, atol=1e-6)

    def test_takes_the_shortest_arc(self):
        """Verify a sign-flipped end quaternion does not send SLERP the long way round."""
        # -q is the same rotation as q, but slerping towards it naively goes the long way.
        negated = tuple(-c for c in QUARTER_TURN_Z)
        flipped = make_trajectory(quaternions=[IDENTITY, IDENTITY, negated])
        _, quat = flipped.tcp_transforms([15])[0]
        self.assertAlmostEqual(abs(float(np.dot(quat, EIGHTH_TURN_Z))), 1.0, places=6)

    def test_holds_the_nearest_end_pose_outside_the_samples(self):
        """Verify queries before and after the stored span return the end poses."""
        source = make_trajectory()
        (before, _), (after, _) = source.tcp_transforms([-100, 100])
        self.assertEqual(tuple(before), source.positions[0])
        self.assertEqual(tuple(after), source.positions[-1])
        self.assertEqual(source.coverage_ns(), (0, 20))

    def test_returns_float_tuples_everywhere(self):
        """Verify every pose is a pair of float tuples, also at the ends of a trajectory loaded from JSON."""
        source = MaterializedTrajectorySource.from_dict(json_round_trip(make_trajectory().to_dict()))
        for pos, quat in source.tcp_transforms([-100, 0, 5, 20, 100]):
            for values, length in ((pos, 3), (quat, 4)):
                self.assertIsInstance(values, tuple)
                self.assertEqual(len(values), length)
                self.assertTrue(all(type(v) is float for v in values))

    def test_validation_rejects_malformed_samples(self):
        """Verify validate() refuses too few samples, mismatched counts, wrongly sized poses and unordered timestamps."""
        cases = {
            "one sample": (
                {"timestamps_ns": [0], "positions": [(0.0, 0.0, 0.0)], "quaternions": [IDENTITY]},
                "at least 2",
            ),
            "position count": ({"positions": [(0.0, 0.0, 0.0)] * 2}, "2 positions"),
            "quaternion count": ({"quaternions": [IDENTITY] * 2}, "2 quaternions"),
            "position size": ({"positions": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]}, "position 0 has 2 values"),
            "quaternion size": ({"quaternions": [IDENTITY, IDENTITY, (0.0, 0.0, 1.0)]}, "quaternion 2 has 3 values"),
            "repeated timestamp": ({"timestamps_ns": [0, 10, 10]}, "strictly increasing"),
            "decreasing timestamp": ({"timestamps_ns": [0, 20, 10]}, "strictly increasing"),
        }
        for name, (overrides, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, message):
                    make_trajectory(**overrides).validate()


if __name__ == "__main__":
    unittest.main()
