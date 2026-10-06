# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the calibration contracts, goals, bundle rehydration and cable evaluation."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.calibration import evaluate as evaluate_module
from newton._src.calibration.data_source import CableDataSource
from newton._src.calibration.evaluate import CableCandidate, CableEvaluator, _goal_index_for_sim_frame
from newton._src.calibration.evidence import CableEvidenceBundle, CableRecording
from newton._src.calibration.goal import CableGoal, group_goals
from newton._src.calibration.model import ANGLE_PARAM_EXP, CableWorld
from newton._src.calibration.rehydrate import bundle_to_goals
from newton._src.calibration.result import STATUS_OK, CableCalibrationResult
from newton._src.calibration.schema import SCHEMA_VERSION
from newton._src.calibration.trajectory import MaterializedTrajectorySource

# Quaternions are (qx, qy, qz, qw). A turn by angle a about the unit axis u is (u sin(a/2), cos(a/2)).
IDENTITY = (0.0, 0.0, 0.0, 1.0)
QUARTER_TURN_Z = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
EIGHTH_TURN_Z = (0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
QUARTER_TURN_X = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))


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


class TestCalibrationContracts(unittest.TestCase):
    """Serialization and validation of the calibration contracts."""

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


class TestCalibrationDataSource(unittest.TestCase):
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


class TestCalibrationTrajectory(unittest.TestCase):
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


def make_goal(**overrides):
    """A valid undriven goal with one 20x20 reference mask, recorded at 1 s."""
    mask = np.zeros((20, 20), np.uint8)
    mask[8:12, 4:16] = 255
    values = {
        "label": "cam0",
        "recording_key": "rec0",
        "masks": [mask],
        "frame_times": [0.0],
        "cable_start": (0.0, 0.0, 0.5),
        "sensor_pos": (1.0, 0.0, 0.5),
        "crop": [0, 0, 20, 20],
        "attachment_transform": ((0.0, 0.0, 0.0), IDENTITY),
        "clamp_position": 0.0,
        "sensor_quat": [0.0, 0.0, 0.0, 1.0],
        "camera_intrinsics": (20, 20, 10.0, 10.0, 10.0, 10.0),
        "start_ns": 1_000_000_000,
    }
    values.update(overrides)
    return CableGoal(**values)


class DriveRecorder:
    """A data source that records the drive it is asked for and returns ``drive`` instead of sampling one."""

    def __init__(self, drive="drive"):
        self.calls = []
        self.drive = drive

    def drive_buffer(self, start_ns, num_frames, substep_rate, sim_substeps):
        self.calls.append((start_ns, num_frames, substep_rate, sim_substeps))
        return self.drive


class TestCalibrationGoals(unittest.TestCase):
    """Grouping co-recorded views onto one simulation."""

    def test_views_of_one_recording_share_a_group(self):
        """Verify views sharing a recording key are grouped, and others are not.

        A goal without start_ns cannot be put on a common clock, so it gets a
        group of its own even when it shares the recording key.
        """
        goals = [
            make_goal(label="a", recording_key="rec0"),
            make_goal(label="b", recording_key="rec1"),
            make_goal(label="c", recording_key="rec0"),
            make_goal(label="d", recording_key="rec0", start_ns=None),
        ]
        groups = group_goals(goals, 60, 10)
        self.assertEqual([[v.label for v in g.views] for g in groups], [["a", "c"], ["b"], ["d"]])
        self.assertEqual([g.label for g in groups], ["rec0", "b", "d"])
        # "d" is still a camera of rec0, so rec0's one unit of weight is split over three views.
        np.testing.assert_allclose([w for g in groups for w in g.weights], [1 / 3, 1 / 3, 1.0, 1 / 3])

    def test_multi_view_group_adopts_the_earliest_origin(self):
        """Verify a multi-view group re-anchors every view to the earliest origin.

        Frame times shift by each view's offset and the timeline extends to cover
        the latest frame of any view; otherwise the shared simulation misaligns
        the views against each other. The drive is read from the earliest origin
        and the group takes the earliest view's cable start.
        """
        source = DriveRecorder()
        later = make_goal(
            label="later",
            start_ns=1_100_000_000,
            frame_times=[0.0, 0.5],
            masks=make_goal().masks * 2,
            cable_start=(9.0, 9.0, 9.0),
            driven=True,
            data_source=source,
        )
        earlier = make_goal(
            label="earlier",
            frame_times=[0.0, 0.75],
            masks=make_goal().masks * 2,
            driven=True,
            data_source=source,
        )
        (group,) = group_goals([later, earlier], 60, 10)

        self.assertEqual([v.label for v in group.views], ["later", "earlier"])
        np.testing.assert_allclose(group.views[0].frame_times, [0.1, 0.6])
        np.testing.assert_allclose(group.views[1].frame_times, [0.0, 0.75])
        # The latest frame is the second view's, at 0.75 s: frame 45 at 60 fps,
        # so the timeline has 46 frames.
        self.assertEqual(group.num_frames, 46)
        self.assertEqual(group.cable_start, (0.0, 0.0, 0.5))
        self.assertEqual(source.calls, [(1_000_000_000, 46, 600, 10)])
        self.assertEqual(group.transform_buffer, "drive")

    def test_timeline_rounds_to_the_nearest_frame(self):
        """Verify the last frame time rounds to the nearest simulated frame, not down.

        Camera frames rarely fall on the simulation's frame grid. A last frame at
        0.03 s is 1.8 frames at 60 fps; truncating would end the timeline before it.
        A single view reads the drive from its own start time over that timeline.
        """
        masks = make_goal().masks * 2
        source = DriveRecorder()
        goal = make_goal(frame_times=[0.0, 0.03], masks=masks, driven=True, data_source=source)
        (single,) = group_goals([goal], 60, 10)
        self.assertEqual(single.num_frames, 3)
        self.assertEqual(source.calls, [(1_000_000_000, 3, 600, 10)])
        (multi,) = group_goals(
            [make_goal(label="a", frame_times=[0.0, 0.03], masks=masks), make_goal(label="b")], 60, 10
        )
        self.assertEqual(multi.num_frames, 3)

    def test_grouping_does_not_mutate_the_goals(self):
        """Verify grouping leaves the caller's goals unchanged and is idempotent.

        Re-anchoring shifts each view's frame times onto the group's clock. Doing
        that in place would compound: grouping one list twice, as two evaluators
        over the same goals would, shifts the offset twice and silently scores
        against a misaligned clock.
        """
        goals = [make_goal(label="a"), make_goal(label="b", start_ns=1_100_000_000)]
        mask = goals[0].masks[0]
        first = group_goals(goals, 60, 10)
        self.assertEqual([g.frame_times for g in goals], [[0.0], [0.0]])

        second = group_goals(goals, 60, 10)
        self.assertEqual(
            [v.frame_times for v in first[0].views],
            [v.frame_times for v in second[0].views],
        )
        # The later view is re-anchored onto the earlier one's clock.
        self.assertAlmostEqual(first[0].views[1].frame_times[0], 0.1)
        # Shallow copy: the mask arrays are shared, not duplicated per group.
        self.assertIs(first[0].views[0].masks[0], mask)

    def test_group_rejects_views_disagreeing_on_the_grasp(self):
        """Verify grouping raises, naming the recording, when views disagree on the grasp.

        The attachment, clamp position and cable axis belong to the recording,
        not the camera. Taking the first view's value would silently ignore the
        other views.
        """
        identity = ((0.0, 0.0, 0.0), IDENTITY)
        cases = {
            "attachment_transform": (identity, ((0.1, 0.0, 0.0), IDENTITY)),
            "clamp_position": (0.0, 0.1),
            "cable_axis": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        }
        for name, (value, other) in cases.items():
            with self.subTest(name):
                goals = [make_goal(label="a", **{name: value}), make_goal(label="b", **{name: other})]
                with self.assertRaisesRegex(ValueError, f"rec0.*{name}"):
                    group_goals(goals, 60, 10)

                # Agreeing views group without complaint, and the group carries the value.
                goals = [make_goal(label="a", **{name: value}), make_goal(label="b", **{name: value})]
                (group,) = group_goals(goals, 60, 10)
                self.assertEqual(getattr(group, name), value)

    def test_goal_validation_requires_a_usable_camera(self):
        """Verify a goal without intrinsics needs both a render size and a field of view.

        There is no default field of view, so omitting it must raise rather than
        silently render through a guessed camera.
        """
        with self.assertRaisesRegex(ValueError, "render_size"):
            make_goal(camera_intrinsics=None).validate()
        with self.assertRaisesRegex(ValueError, "fov_deg"):
            make_goal(camera_intrinsics=None, render_size=(20, 20)).validate()
        make_goal(camera_intrinsics=None, render_size=(20, 20), fov_deg=60.0).validate()

    def test_goal_validation_rejects_unscorable_goals(self):
        """Verify a goal without masks, with a frame time count that differs, or without a usable drive is refused."""
        cases = {
            "no masks": {"masks": [], "frame_times": []},
            "frame time": {"frame_times": [0.0, 0.1]},
            "data_source": {"driven": True},
            "start_ns": {"driven": True, "data_source": DriveRecorder(), "start_ns": None},
        }
        for match, overrides in cases.items():
            with self.subTest(match):
                with self.assertRaisesRegex(ValueError, match):
                    make_goal(**overrides).validate()


def write_mask(path, value, shape=(24, 32)):
    """Write a ``(height, width)`` mask whose pixels all equal ``value``."""
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.full(shape, value, np.uint8)).save(path)


@unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "Requires Pillow")
class TestCalibrationRehydrate(unittest.TestCase):
    """Reading a stored bundle back into goals."""

    ATTACHMENT = ((0.0, 0.0, 0.1), IDENTITY)

    def write_bundle(self, directory, recordings):
        """Save ``recordings`` with their masks and the shared trajectory, but no RGB frames."""
        for rec in recordings:
            for k, rel in enumerate(rec.masks):
                write_mask(os.path.join(directory, rel), 10 * (k + 1))
        os.makedirs(os.path.join(directory, "trajectories"), exist_ok=True)
        with open(os.path.join(directory, "trajectories", "rec0.json"), "w") as fh:
            json.dump(make_trajectory().to_dict(), fh)
        CableEvidenceBundle(recordings=recordings).save(directory)
        return CableEvidenceBundle.load(directory)

    def test_bundle_rehydrates_into_scorable_goals(self):
        """Verify a bundle written to disk reads back into valid goals.

        Only the masks are decoded: the bundle has no RGB frames on disk, and
        reading succeeds. The scored region is the whole mask, each recording's
        fields reach its goal, the grasp arguments are applied to every goal,
        and recordings that share a trajectory file share one data source.
        """
        cam1 = make_recording(
            label="cam1",
            masks=["masks/cam1/0.png", "masks/cam1/1.png", "masks/cam1/2.png"],
            camera_intrinsics=None,
            render_size=[32, 24],
            fov_deg=60.0,
        )
        attachment = [[0.0, 0.0, 0.1], list(IDENTITY)]
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording(), cam1])
            goals = bundle_to_goals(
                bundle, tmp, attachment_transform=attachment, clamp_position=0.2, cable_axis=[0.0, 0.0, 1.0]
            )

        self.assertEqual([g.label for g in goals], ["cam0", "cam1"])
        goal = goals[0]
        self.assertEqual([int(m[0, 0]) for m in goal.masks], [10, 20, 30])
        self.assertEqual(goal.masks[0].shape, (24, 32))
        self.assertEqual(goal.crop, [0, 0, 32, 24])
        self.assertEqual(goal.frame_times, [0.0, 0.1, 0.2])
        self.assertEqual(goal.recording_key, "rec0")
        self.assertEqual(goal.start_ns, 1_000_000_000)
        self.assertTrue(goal.driven)
        self.assertEqual(goal.cable_start, (0.0, 0.0, 0.8))
        self.assertEqual(goal.sensor_pos, (1.0, 0.0, 0.5))
        self.assertEqual(goal.sensor_quat, [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(goal.camera_intrinsics, (32, 24, 40.0, 40.0, 16.0, 12.0))
        self.assertEqual((goals[1].camera_intrinsics, goals[1].render_size, goals[1].fov_deg), (None, (32, 24), 60.0))
        for g in goals:
            self.assertEqual(g.attachment_transform, self.ATTACHMENT)
            self.assertEqual(g.clamp_position, 0.2)
            self.assertEqual(g.cable_axis, (0.0, 0.0, 1.0))
        self.assertIsInstance(goal.data_source, MaterializedTrajectorySource)
        self.assertIs(goals[1].data_source, goal.data_source)

    def test_bundle_and_trajectory_are_validated(self):
        """Verify a malformed bundle or stored trajectory is refused before any goal is built."""
        duplicate = CableEvidenceBundle(recordings=[make_recording(), make_recording()])
        with self.assertRaisesRegex(ValueError, "duplicate label"):
            bundle_to_goals(duplicate, ".", attachment_transform=self.ATTACHMENT, clamp_position=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording()])
            with open(os.path.join(tmp, "trajectories", "rec0.json"), "w") as fh:
                json.dump(make_trajectory(timestamps_ns=[0, 10, 10]).to_dict(), fh)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT, clamp_position=0.0)

    def test_masks_must_match_each_other_and_the_camera(self):
        """Verify masks of different sizes, or of another size than the camera describes, are refused.

        The camera data describes the stored masks. A mask of another size would
        be scored against a projection at the wrong scale.
        """
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording()])
            write_mask(os.path.join(tmp, "masks", "cam0", "2.png"), 30, shape=(12, 16))
            with self.assertRaisesRegex(ValueError, r"cam0.*masks/cam0/2\.png.*16x12"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT, clamp_position=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording(camera_intrinsics=[64, 48, 80.0, 80.0, 32.0, 24.0])])
            with self.assertRaisesRegex(ValueError, "cam0.*32x24.*64x48"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT, clamp_position=0.0)

    def test_missing_files_are_reported_with_the_recording(self):
        """Verify a missing mask or trajectory raises, naming the recording and the file."""
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording()])
            os.remove(os.path.join(tmp, "masks", "cam0", "1.png"))
            with self.assertRaisesRegex(ValueError, r"cam0.*masks/cam0/1\.png"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT, clamp_position=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording()])
            os.remove(os.path.join(tmp, "trajectories", "rec0.json"))
            with self.assertRaisesRegex(ValueError, r"cam0.*trajectories/rec0\.json"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT, clamp_position=0.0)


ORIGIN = (0.0, 0.0, 0.0)
TCP_POS = (0.3, 0.1, 0.4)
SIM_SUBSTEPS = 20


def fixed_drive(tcp_quat=IDENTITY, num_frames=2):
    """A drive that holds the TCP at ``TCP_POS`` with orientation ``tcp_quat``."""
    pose = wp.transform(TCP_POS, tcp_quat)
    return wp.array([pose] * (num_frames * SIM_SUBSTEPS), dtype=wp.transform)


def make_camera(**overrides):
    """A calibrated camera descriptor for CableWorld, for tests that do not look at the image."""
    values = {
        "sensor_pos": (1.0, 0.0, 0.5),
        "sensor_quat": IDENTITY,
        "camera_intrinsics": (32, 24, 40.0, 40.0, 16.0, 12.0),
        "render_size": None,
        "fov_deg": None,
    }
    values.update(overrides)
    return values


def make_world(population=1, num_elements=3, **overrides):
    """A CableWorld of ``population`` straight, undriven cables of ``num_elements`` capsules each."""
    values = {
        "cable_start": ORIGIN,
        "angles_list": [[(0.0, 0.0)] * num_elements] * population,
        "bend_stiffness_list": [1.0] * population,
        "transform_buffer": None,
        "cameras": [make_camera()],
        "sim_iterations": 10,
        "settle_mode": "dynamic",
        "clamp_position": 0.0,
        "sim_substeps": SIM_SUBSTEPS,
        "bend_damping_list": [0.1] * population,
        "twist_stiffness_list": [1.0] * population,
        "twist_damping_list": [0.1] * population,
        "stretch_stiffness": 1000.0,
        "num_elements": num_elements,
        "segment_length": 0.05,
        "cable_radius": 0.005,
        "cable_mass": 0.02,
        "angle_parametrization": ANGLE_PARAM_EXP,
        "attachment_transform": (ORIGIN, IDENTITY),
    }
    values.update(overrides)
    return CableWorld(**values)


def body_pose(world, body):
    """``(x, y, z, qx, qy, qz, qw)`` of ``body`` in ``state_0``."""
    return world.state_0.body_q.numpy()[body].copy()


def capsule_axis(world, capsule):
    """World direction of the local +Z axis of capsule ``capsule`` in world 0."""
    q = body_pose(world, world.cable_bodies_list[0][capsule])
    return np.array(wp.quat_rotate(wp.quat(*(float(v) for v in q[3:])), wp.vec3(0.0, 0.0, 1.0)))


class TestCalibrationAttachmentTransform(unittest.TestCase):
    """CableWorld composing attachment_transform into the build and the drive."""

    def anchor_pose(self, world):
        """Pose of the clamp capsule of world 0."""
        return body_pose(world, world.cable_bodies_list[0][world.clamp_capsule])

    def test_frame_zero_is_continuous_for_non_identity_attachment(self):
        """Verify the built anchor pose equals the pose the drive imposes at frame 0.

        Otherwise the anchor jumps on the first substep. The TCP is rotated, so
        the attachment translation must be rotated with it. The TCP turns about z
        and the attachment about x, so the order of the two rotations matters.
        """
        world = make_world(
            transform_buffer=fixed_drive(EIGHTH_TURN_Z), attachment_transform=((0.02, -0.01, 0.03), QUARTER_TURN_X)
        )
        built = self.anchor_pose(world)
        world._apply_drive(0)
        np.testing.assert_allclose(self.anchor_pose(world), built, atol=1e-6)

    def test_nonzero_translation_offsets_the_anchor_from_the_tcp(self):
        """Verify the attachment translation moves the anchor by that offset.

        With a drive, the offset is from the TCP. Without a drive, it is from
        cable_start.
        """
        offset = (0.05, 0.0, 0.0)
        for name, drive in (("driven", fixed_drive()), ("undriven", None)):
            with self.subTest(name):
                plain = make_world(transform_buffer=drive)
                shifted = make_world(transform_buffer=drive, attachment_transform=(offset, IDENTITY))
                np.testing.assert_allclose(
                    self.anchor_pose(shifted)[:3] - self.anchor_pose(plain)[:3], offset, atol=1e-6
                )


class TestCalibrationClampPosition(unittest.TestCase):
    """CableWorld clamping the cable at a point along its length."""

    def build(self, clamp_position):
        """A six-capsule, 0.3 m cable clamped at ``clamp_position`` [m] and held at ``TCP_POS``."""
        return make_world(num_elements=6, transform_buffer=fixed_drive(), clamp_position=clamp_position)

    def grasp_point(self, world):
        """World position of the grasp: ``clamp_offset`` along the clamp capsule from its start."""
        start = body_pose(world, world.cable_bodies_list[0][world.clamp_capsule])[:3]
        return start + world.clamp_offset * capsule_axis(world, world.clamp_capsule)

    def test_interior_clamp_drives_an_interior_capsule(self):
        """Verify an interior clamp_position drives the capsule that starts there, not the end one.

        0.15 m is exactly three capsule lengths, which the floating-point
        division puts just below 3.
        """
        world = self.build(0.15)
        self.assertEqual(world.clamp_capsule, 3)
        self.assertEqual(int(world.kinematic_bodies.numpy()[0]), world.cable_bodies_list[0][3])

    def test_interior_clamp_is_the_only_massless_capsule(self):
        """Verify only the clamp capsule is massless, and every other capsule carries an equal share of the cable mass.

        The share is 0.02 kg over six capsules.
        """
        world = self.build(0.15)
        bodies = world.cable_bodies_list[0]
        mass = world.model.body_mass.numpy()
        expected = [0.02 / 6] * 6
        expected[world.clamp_capsule] = 0.0
        np.testing.assert_allclose(mass[bodies], expected, atol=1e-7)

    def test_grasp_point_lands_on_the_tcp_for_a_sub_capsule_offset(self):
        """Verify a clamp inside a capsule puts the grasp point, not the capsule start, on the TCP."""
        world = self.build(0.17)
        self.assertEqual(world.clamp_capsule, 3)
        self.assertAlmostEqual(world.clamp_offset, 0.02, places=6)
        np.testing.assert_allclose(self.grasp_point(world), TCP_POS, atol=1e-6)

    def test_interior_clamp_holds_the_bent_rest_shape_at_the_clamp(self):
        """Verify an interior clamp capsule's rest pose equals its built pose for a bent cable.

        The rest shape is built from the bend angles and then moved so that the
        clamp capsule lands on the clamp pose; the straight initial state must
        place the clamp capsule at the same pose. The TCP is rotated, so the clamp
        rotation does not commute with the bent capsule's rotation.
        """
        world = make_world(
            num_elements=6,
            transform_buffer=fixed_drive(QUARTER_TURN_Z),
            clamp_position=0.15,
            angles_list=[[(0.2, 0.1)] * 6],
        )
        body = world.cable_bodies_list[0][world.clamp_capsule]
        rest = world.model.body_q.numpy()[body]
        built = body_pose(world, body)
        np.testing.assert_allclose(rest[:3], built[:3], atol=1e-6)
        # q and -q are the same rotation.
        sign = 1.0 if np.dot(rest[3:], built[3:]) >= 0.0 else -1.0
        np.testing.assert_allclose(rest[3:], sign * built[3:], atol=1e-5)

    def test_clamp_follows_a_moving_drive(self):
        """Verify the clamp capsule holds the frame 0 drive while settling, then follows the TCP frame by frame.

        The drive index advances with the frame, also when the substeps replay as
        one CUDA graph. After frame f the clamp is at the last substep sample of
        that frame. After the last recorded frame, the clamp holds the last sample.
        """
        substeps = SIM_SUBSTEPS
        recorded_frames = 3
        samples = [wp.transform((0.3 + 0.001 * k, 0.1, 0.4), IDENTITY) for k in range(recorded_frames * substeps)]
        world = make_world(transform_buffer=wp.array(samples, dtype=wp.transform))
        clamp = int(world.kinematic_bodies.numpy()[0])
        world.settle(3)
        np.testing.assert_allclose(body_pose(world, clamp)[:3], (0.3 + 0.001 * (substeps - 1), 0.1, 0.4), atol=1e-6)
        for frame in range(recorded_frames + 2):
            world._step(frame)
            sample = min(frame * substeps + substeps - 1, len(samples) - 1)
            np.testing.assert_allclose(body_pose(world, clamp)[:3], (0.3 + 0.001 * sample, 0.1, 0.4), atol=1e-6)

    def test_clamp_position_beyond_the_cable_is_refused(self):
        """Verify a clamp_position beyond the 0.3 m cable is refused, naming the field."""
        with self.assertRaisesRegex(ValueError, "clamp_position"):
            self.build(1.0)


class TestCalibrationCableAxis(unittest.TestCase):
    """CableWorld orienting the clamp from a direction in the TCP frame."""

    def clamp_axis(self, cable_axis, tcp_quat=IDENTITY):
        world = make_world(transform_buffer=fixed_drive(tcp_quat), cable_axis=cable_axis)
        return capsule_axis(world, 0)

    def test_cable_axis_rotates_with_the_tcp(self):
        """Verify cable_axis is relative to the TCP: a quarter turn of the TCP about z turns x into y."""
        np.testing.assert_allclose(self.clamp_axis((1.0, 0.0, 0.0), QUARTER_TURN_Z), (0.0, 1.0, 0.0), atol=1e-6)

    def test_zero_or_malformed_direction_is_refused(self):
        """Verify a zero-length cable_axis, or one without three components, is refused."""
        for name, cable_axis in (("zero", ORIGIN), ("two components", (1.0, 0.0))):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "cable_axis"):
                    self.clamp_axis(cable_axis)


# The scene for rendering and scoring: three 0.05 m capsules hang straight down
# from HANG_START. The scene camera looks along +y at the middle of the cable
# from 0.5 m, so the cable nodes project to column 16, rows 3, 9, 15 and 21 of
# a 32x24 image.
HANG_START = (0.0, 0.0, 0.5)
HANG_DOWN = (ORIGIN, (1.0, 0.0, 0.0, 0.0))  # A half turn about x points local +Z along -z.
CABLE_MIDDLE = (0.0, 0.0, 0.425)
SCENE_INTRINSICS = (32, 24, 60.0, 60.0, 16.0, 12.0)
SCENE_CROP = [0, 0, 32, 24]
NODE_ROWS = [3.0, 9.0, 15.0, 21.0]


def viewing_dir_to_camera_quat(viewing_dir):
    """Orientation ``(qx, qy, qz, qw)`` of an upright camera that looks along ``viewing_dir``.

    The camera looks along its local -z axis, with x right and y up, and world Z
    maps to image up. For a near-vertical direction, world Y is the
    reference, because the cross product with Z is ill-conditioned there.
    """
    fwd = np.asarray(viewing_dir, dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    ref = np.array([0.0, 1.0, 0.0]) if abs(fwd[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, ref)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    rot = np.column_stack([right, up, -fwd])
    return tuple(float(v) for v in wp.quat_from_matrix(wp.mat33(*rot.flatten().tolist())))


def scene_camera(viewing_dir=(0.0, 1.0, 0.0), distance=0.5):
    """A camera descriptor for CableWorld, ``distance`` [m] from the cable middle and looking along ``viewing_dir``."""
    pos = tuple(m - distance * d for m, d in zip(CABLE_MIDDLE, viewing_dir, strict=True))
    return {
        "sensor_pos": pos,
        "sensor_quat": viewing_dir_to_camera_quat(viewing_dir),
        "camera_intrinsics": SCENE_INTRINSICS,
    }


def make_scene(population=1, cameras=None, **overrides):
    """A CableWorld of the hanging scene, seen by the scene camera unless ``cameras`` is given."""
    return make_world(
        population,
        cable_start=HANG_START,
        attachment_transform=HANG_DOWN,
        cable_radius=0.01,
        cameras=cameras or [scene_camera()],
        **overrides,
    )


def scene_mask():
    """A goal mask covering the hanging cable in the scene camera: columns 15 to 17, rows 3 to 21."""
    mask = np.zeros((24, 32), np.uint8)
    mask[3:22, 15:18] = 255
    return mask


def make_scene_goal(viewing_dir=(0.0, 1.0, 0.0), **overrides):
    """A goal of the hanging scene with two reference frames, at 0 s and 0.05 s."""
    camera = scene_camera(viewing_dir)
    values = {
        "masks": [scene_mask()] * 2,
        "frame_times": [0.0, 0.05],
        "cable_start": HANG_START,
        "sensor_pos": camera["sensor_pos"],
        "sensor_quat": list(camera["sensor_quat"]),
        "camera_intrinsics": SCENE_INTRINSICS,
        "crop": list(SCENE_CROP),
        "start_ns": 0,
        "attachment_transform": HANG_DOWN,
        "clamp_position": 0.0,
    }
    values.update(overrides)
    return make_goal(**values)


class LossPixelCount:
    """A pixel loss: the number of cable pixels inside the crop of the rendered mask. Keeps every mask it receives."""

    wants_geometry = False
    wants_render = True
    supports_accum = False

    def __init__(self):
        self.sim_masks = []

    def prepare(self, goal_mask):
        return goal_mask

    def score(self, goal, sim_mask, crop, *, geom=None):
        self.sim_masks.append(sim_mask)
        x0, y0, x1, y1 = crop
        return float(np.count_nonzero(sim_mask[y0:y1, x0:x1]))


class PerWorldTotals:
    """The accumulator of LossPixelCountOnDevice: one running total per world."""

    def __init__(self, n):
        self.values = np.zeros(n)

    def totals(self):
        return self.values.tolist()


class LossPixelCountOnDevice(LossPixelCount):
    """LossPixelCount with on-device accumulation: each call scores the device masks of all worlds."""

    supports_accum = True

    def make_accum(self, n, device):
        return PerWorldTotals(n)

    def accum(self, goal, sim_mask, crop, totals, *, geom=None):
        x0, y0, x1, y1 = crop
        totals.values += np.count_nonzero(sim_mask.numpy()[:, y0:y1, x0:x1], axis=(1, 2))


class LossGoalValue:
    """A loss that scores each goal frame with its prepared value, so the loss shows which goal frames were scored."""

    wants_geometry = False
    wants_render = False
    supports_accum = False

    def prepare(self, value):
        return value

    def score(self, goal, sim_mask, crop, *, geom=None):
        return goal


class LossNodeDistance:
    """A geometry loss: the mean distance [px] of the projected cable nodes from the goal mask centroid.

    Keeps the ``(sim_mask, geom)`` pair of every call.
    """

    wants_geometry = True
    wants_render = False
    supports_accum = False

    def __init__(self):
        self.calls = []

    def prepare(self, goal_mask):
        rows, cols = np.nonzero(goal_mask)
        return np.array([cols.mean(), rows.mean()])

    def score(self, goal, sim_mask, crop, *, geom=None):
        self.calls.append((sim_mask, geom))
        return float(np.linalg.norm(geom - goal, axis=1).mean())


class TestCalibrationCableWorld(unittest.TestCase):
    """Building, stepping and rendering a CableWorld."""

    def test_mask_lies_where_the_cable_projects(self):
        """Verify the rendered cable mask lies where project_cable puts the cable.

        The camera is off-axis and off-centre, so a sign error in the rays or in
        the camera orientation would move the mask away from the projection.
        """
        camera = {**scene_camera((-1.0, 0.0, 0.0)), "sensor_pos": (0.5, 0.04, 0.45)}
        world = make_scene(cameras=[camera])
        uv = world.project_cable(0).numpy()[0]
        _, _, _, masks = world._render(with_mask=True)
        rows, cols = np.nonzero(masks[0])
        self.assertAlmostEqual(cols.mean(), uv[:, 0].mean(), delta=1.0)
        self.assertAlmostEqual(rows.min(), uv[:, 1].min(), delta=2.0)
        self.assertAlmostEqual(rows.max(), uv[:, 1].max(), delta=2.0)

    def test_image_axes_follow_their_own_focal_length(self):
        """Verify project_cable and the mask scale image x by fx and image y by fy about the principal point.

        The scene camera moves 0.05 m to the left and gets fy = 2/3 fx.
        """
        fx, fy, cx, cy = 60.0, 40.0, 16.0, 12.0
        left, distance = 0.05, 0.5
        camera = scene_camera(distance=distance)
        x, y, z = camera["sensor_pos"]
        camera.update(sensor_pos=(x - left, y, z), camera_intrinsics=(32, 24, fx, fy, cx, cy))
        world = make_scene(cameras=[camera])
        column = cx + fx * left / distance  # 22
        rows = [cy + (r - cy) * fy / SCENE_INTRINSICS[3] for r in NODE_ROWS]  # 6, 10, 14, 18
        uv = world.project_cable(0).numpy()[0]
        np.testing.assert_allclose(uv[:, 0], column, atol=1e-3)
        np.testing.assert_allclose(uv[:, 1], rows, atol=1e-3)
        _, _, _, masks = world._render(with_mask=True)
        mask_rows, mask_cols = np.nonzero(masks[0])
        # The cable is mirror symmetric about its centre, so the mask centroid is there.
        np.testing.assert_allclose((mask_cols.mean(), mask_rows.mean()), (column, cy), atol=0.1)
        # The 0.01 m radius at 0.5 m reaches fy / 50 = 0.8 px beyond the end nodes, so the
        # mask spans the projected rows, and a render with fx in place of fy spans more.
        self.assertEqual((mask_rows.min(), mask_rows.max()), (rows[0], rows[-1]))

    def test_field_of_view_is_vertical_about_the_image_centre(self):
        """Verify a render_size and fov_deg camera renders like intrinsics with fy from the vertical field of view.

        The principal point of such a camera is the image centre.
        """
        width, height, focal = 32, 24, 60.0
        fov_deg = math.degrees(2.0 * math.atan(0.5 * height / focal))
        centred = (width, height, focal, focal, 0.5 * (width - 1), 0.5 * (height - 1))
        cameras = [
            {**scene_camera(), "camera_intrinsics": None, "render_size": (width, height), "fov_deg": fov_deg},
            {**scene_camera(), "camera_intrinsics": centred},
        ]
        world = make_scene(cameras=cameras)
        _, _, _, fov_masks = world._render(cam=0, with_mask=True)
        _, _, _, intrinsics_masks = world._render(cam=1, with_mask=True)
        self.assertGreater(np.count_nonzero(intrinsics_masks[0]), 0)
        np.testing.assert_array_equal(fov_masks[0], intrinsics_masks[0])

    def test_node_behind_the_camera_is_invalid(self):
        """Verify a cable behind the camera projects to NaN."""
        world = make_scene(cameras=[scene_camera((0.0, -1.0, 0.0), distance=-0.5)])
        self.assertTrue(np.all(np.isnan(world.project_cable(0).numpy())))

    def test_settle_moves_a_cable_with_a_bent_rest_shape(self):
        """Verify the cable starts straight and settle() moves a cable with a bent rest shape, under gravity along -z.

        The model has one gravity entry per world and one for the global world.
        """
        world = make_scene(angles_list=[[(0.3, 0.0)] * 3])
        np.testing.assert_allclose(world.model.gravity.numpy(), [(0.0, 0.0, -9.81)] * (world.n + 1), atol=1e-6)
        np.testing.assert_allclose(capsule_axis(world, 2), capsule_axis(world, 0), atol=1e-6)
        tip = world.cable_bodies_list[0][-1]
        before = body_pose(world, tip)[:3]
        world.settle(3)
        self.assertGreater(np.linalg.norm(body_pose(world, tip)[:3] - before), 1e-3)

    def settle_until_frozen(self, world, settle_frames, moving_frames):
        """Settle ``world``, simulating only its first ``moving_frames`` steps. Return the step count and the warnings."""
        step, frames = world._step, []

        def step_until_frozen(frame_idx):
            frames.append(frame_idx)
            if len(frames) <= moving_frames:
                step(frame_idx)

        with patch.object(world, "_step", step_until_frozen), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            world.settle(settle_frames)
        return len(frames), caught

    def test_settle_stops_at_the_first_check_within_the_tolerance(self):
        """Verify settle() stops at the first check within settle_move_tol, and warns only when it stops at its cap.

        The 0.15 m cable cannot move 1 m, so a 1 m tolerance is met at the first
        check. Each check measures the movement since the previous check, so once
        the steps stop simulating, the next check is within a small tolerance. A
        zero tolerance is never met. A settle shorter than one check interval does
        no check, so it does not warn. The warning points at the caller of settle().
        """
        check_every, cap = 2, 5
        cases = {
            # name: (settle_move_tol, settle_frames, simulated frames, expected steps, warns)
            "converges at the first check": (1.0, cap, cap, check_every, False),
            "stops once the cable stops moving": (1.0e-6, cap, check_every, 2 * check_every, False),
            "reaches the cap": (0.0, cap, cap, cap, True),
            "shorter than one check interval": (0.0, check_every - 1, cap, check_every - 1, False),
        }
        for name, (move_tol, settle_frames, moving_frames, expected_steps, warns) in cases.items():
            with self.subTest(name):
                world = make_world(
                    angles_list=[[(0.3, 0.0)] * 3], settle_check_every=check_every, settle_move_tol=move_tol
                )
                steps, caught = self.settle_until_frozen(world, settle_frames, moving_frames)
                self.assertEqual(steps, expected_steps)
                self.assertEqual(len(caught), int(warns))
                for w in caught:
                    self.assertIn(f"cap of {settle_frames} frame(s) without converging", str(w.message))
                    self.assertEqual(w.filename, __file__)

    def test_per_world_lists_must_match_the_population(self):
        """Verify a per-world list of another length than the population is refused, naming the list."""
        for name in ("bend_stiffness_list", "bend_damping_list", "twist_stiffness_list", "twist_damping_list"):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, name):
                    make_world(population=2, **{name: [1.0]})

    def test_unsupported_model_inputs_are_refused(self):
        """Verify an unsupported angle parametrization or settle mode, or a settle check interval below 1, is refused.

        The error names the input.
        """
        cases = {
            "angle_parametrization": "euler_xy",
            "settle_mode": "static",
            "settle_check_every": 0,
        }
        for name, value in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, name):
                    make_world(**{name: value})

    def test_incomplete_camera_is_refused(self):
        """Verify an incomplete camera, or no camera at all, is refused.

        A camera without a position or orientation, without a size and field of
        view, or with a fractional size is refused. No default replaces a missing
        value. Sizes stored as whole floats, as JSON can give them, are accepted.
        """
        with self.assertRaisesRegex(ValueError, "camera"):
            make_world(cameras=[])
        cases = {
            "no position": {"sensor_pos": None},
            "no orientation": {"sensor_quat": None},
            "no field of view": {"camera_intrinsics": None, "render_size": (32, 24)},
            "no size": {"camera_intrinsics": None, "fov_deg": 60.0},
            "fractional size": {"camera_intrinsics": (32.5, 24.0, 40.0, 40.0, 16.0, 12.0)},
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "camera"):
                    make_world(cameras=[make_camera(**overrides)])
        make_world(
            cameras=[
                make_camera(camera_intrinsics=(32.0, 24.0, 40.0, 40.0, 16.0, 12.0)),
                make_camera(camera_intrinsics=None, render_size=(32.0, 24.0), fov_deg=60.0),
            ]
        )


class TestCalibrationCableMasks(unittest.TestCase):
    """Binary cable masks rendered from shape indices."""

    def scene_with(self, *, occluder=False, cameras=None):
        """The hanging scene of two worlds, with an optional occluding box."""
        finalize = newton.ModelBuilder.finalize

        def finalize_scene(builder, *args, **kwargs):
            if occluder:
                # A static box between the scene camera and the cable that fills the view.
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(wp.vec3(0.0, -0.25, 0.425), wp.quat_identity()),
                    hx=0.2,
                    hy=0.02,
                    hz=0.2,
                )
            return finalize(builder, *args, **kwargs)

        with patch.object(newton.ModelBuilder, "finalize", finalize_scene):
            return make_scene(population=2, cameras=cameras)

    def test_occluder_is_excluded_and_hides_the_cable(self):
        """Verify a shape in front of the cable is not part of the mask and hides the cable."""
        _, _, _, masks = self.scene_with(occluder=True)._render(with_mask=True)
        self.assertEqual(np.count_nonzero(masks), 0)

    def test_views_keep_their_masks_when_buffers_are_reused(self):
        """Verify a view's returned masks stay unchanged after another view of equal size renders."""
        cameras = [scene_camera(), {**scene_camera(), "sensor_pos": (0.04, -0.5, 0.425)}]
        world = self.scene_with(cameras=cameras)
        _, _, _, first = world._render(cam=0, with_mask=True)
        expected = np.array(first, copy=True)
        _, _, _, second = world._render(cam=1, with_mask=True)
        self.assertEqual(len(world._mask_buffers), 1)
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first, expected)


class TestCalibrationRunSequence(unittest.TestCase):
    """Scoring a rollout with CableWorld.run_sequence."""

    def test_pixel_loss_receives_the_binary_mask(self):
        """Verify run_sequence gives a pixel loss each world's mask of the current state, with or without recording.

        The cable starts straight and has a bent rest shape, so a mask of the built shape is different.
        The expected masks come from a cable whose rest shape is straight, where the built shape is
        the current state.
        """
        bent = [[(0.3, 0.0)] * 3] * 2
        _, _, _, expected = make_scene(population=2)._render(with_mask=True)
        loss = LossPixelCount()
        recorded_world = make_scene(population=2, angles_list=bent)
        values, _, masks = recorded_world.run_sequence(1, [None], SCENE_CROP, loss, record_fps=recorded_world.fps)
        unrecorded, _, _ = make_scene(population=2, angles_list=bent).run_sequence(
            1, [None], SCENE_CROP, LossPixelCount()
        )
        self.assertGreater(values[0], 0)
        self.assertEqual(values, [np.count_nonzero(mask) for mask in expected])
        self.assertEqual(unrecorded, values)
        for recorded, mask in zip(masks, expected, strict=True):
            np.testing.assert_array_equal(recorded[0], mask)
        self.assertEqual(len(loss.sim_masks), 2)
        for mask in loss.sim_masks:
            self.assertEqual(mask.shape, (24, 32))
            self.assertEqual(set(np.unique(mask)), {0, 255})

    def test_goal_frames_are_scored_at_their_simulation_frames(self):
        """Verify each goal frame is scored at the simulation frame nearest its capture time.

        The loss is the mean over goal frames. Goal frame j has the value 10**j,
        so each frame loss shows which goal frames it contains. Without capture
        times, the middle goal frame of three over four frames lies at frame 1.5,
        which rounds to 2.
        """
        goals = [1.0, 10.0, 100.0]
        cases = {
            # name: (fps, num_frames, goal_times, [(simulation frame, frame loss)])
            "evenly spaced without capture times": (60, 4, None, [(0, 1.0), (2, 10.0), (3, 100.0)]),
            "two goal frames nearest one frame": (60, 5, [0.0, 0.001, 2 / 60], [(0, 11.0), (2, 100.0)]),
            "at another frame rate": (30, 5, [0.0, 1 / 30, 2 / 30], [(0, 1.0), (1, 10.0), (2, 100.0)]),
            "after the last frame": (60, 3, [0.0, 1 / 60, 1.0], [(0, 1.0), (1, 10.0), (2, 100.0)]),
        }
        # The loss does not read the cable, so one world per frame rate serves every case.
        worlds = {fps: make_world(fps=fps) for fps in (60, 30)}
        for name, (fps, num_frames, goal_times, frame_losses) in cases.items():
            with self.subTest(name):
                world = worlds[fps]
                values, _, _ = world.run_sequence(
                    num_frames, goals, SCENE_CROP, LossGoalValue(), record_fps=fps, goal_times=goal_times
                )
                self.assertEqual(world.last_frame_losses, [[(f / fps, v) for f, v in frame_losses]])
                self.assertEqual(values, [sum(goals) / len(goals)])

    def test_each_view_is_scored_with_its_own_goal_frames_and_crop(self):
        """Verify run_sequence scores each view against its own goal frames and crop.

        Two views see the cable through the same camera. The crop of the second view is left of the cable.
        """
        world = make_scene(cameras=[scene_camera(), scene_camera()])
        left_of_cable = [0, 0, 12, 24]
        pixels, _, _ = world.run_sequence(1, [[None], [None]], [SCENE_CROP, left_of_cable], LossPixelCount())
        self.assertGreater(pixels[0][0], 0)
        self.assertEqual(pixels[1], [0.0])

        goals = [[1.0, 10.0, 100.0], [1000.0]]
        values, _, _ = world.run_sequence(1, goals, [SCENE_CROP] * 2, LossGoalValue())
        self.assertEqual(values, [[sum(g) / len(g)] for g in goals])

    def test_on_device_scoring_agrees_with_host_scoring(self):
        """Verify a loss that accumulates on the device gives the same losses and frame losses as host scoring.

        The goal frames are at simulation frames 0 and 2, and only frame 0 is recorded.
        """
        if not wp.get_device().is_cuda:
            self.skipTest("On-device scoring needs CUDA")
        on_host, on_device = LossPixelCount(), LossPixelCountOnDevice()
        results = []
        for loss in (on_host, on_device):
            world = make_scene(population=2)
            values, _, _ = world.run_sequence(3, [None, None], SCENE_CROP, loss, record_fps=1, goal_times=[0.0, 2 / 60])
            results.append((values, world.last_frame_losses))
        self.assertEqual(results[1], results[0])
        self.assertGreater(results[0][0][0], 0)
        self.assertEqual([t for t, _ in results[0][1][0]], [0.0, 2 / 60])
        # On the device path, the host score() is not called.
        self.assertEqual(on_device.sim_masks, [])

    def test_geometry_loss_needs_no_mask_or_mask_allocation(self):
        """Verify a geometry loss gets the projected cable and no mask, and nothing is rendered.

        Recording the frames renders masks but leaves the loss values unchanged.
        """
        loss = LossNodeDistance()
        goals = [loss.prepare(scene_mask())]
        world = make_scene(population=2)
        with patch.object(world, "_render", side_effect=AssertionError("a geometry loss needs no render")):
            plain, _, _ = world.run_sequence(1, goals, SCENE_CROP, loss)
        self.assertFalse(world._mask_buffers)
        self.assertEqual(len(loss.calls), 2)
        for sim_mask, geom in loss.calls:
            self.assertIsNone(sim_mask)
            self.assertEqual(geom.shape, (4, 2))
        # The straight cable's nodes are 9, 3, 3 and 9 px from the goal centroid at (16, 12).
        np.testing.assert_allclose(plain, [6.0, 6.0], atol=1e-3)

        recorded_world = make_scene(population=2)
        recorded, frames, masks = recorded_world.run_sequence(1, goals, SCENE_CROP, loss, record_fps=recorded_world.fps)
        np.testing.assert_allclose(recorded, plain)
        self.assertEqual(len(frames), 2)
        self.assertEqual(len(masks[0]), 1)
        self.assertGreater(np.count_nonzero(masks[0][0]), 0)

    def test_recording_rate_spreads_frames_over_the_sequence(self):
        """Verify record_indices gives num_frames * record_fps / fps frames, evenly spaced from first to last.

        Nine frames at 30 fps recorded at 10 fps give three frames: 0, 4 and 8.
        Recording at the simulation rate or faster gives every frame. A rate that rounds to
        no frame still gives one. A rate that is not positive is refused.
        """
        num_frames, fps = 9, 30
        world = make_world(fps=fps)
        self.assertEqual(world.record_indices(num_frames, 10.0), [0, 4, 8])
        self.assertEqual(world.record_indices(num_frames, float(fps)), list(range(num_frames)))
        self.assertEqual(world.record_indices(num_frames, math.inf), list(range(num_frames)))
        self.assertEqual(world.record_indices(num_frames, 1.0), [0])
        with self.assertRaisesRegex(ValueError, "record_fps"):
            world.record_indices(num_frames, 0.0)


def make_candidate(angles=(0.0, 0.0)):
    """A candidate whose three joints all have rest angles ``angles`` [rad]."""
    return CableCandidate(
        angles=[angles] * 3, bend_stiffness=1.0, twist_stiffness=1.0, bend_damping=0.01, twist_damping=0.01
    )


def make_evaluator(goals, loss, **overrides):
    """A CableEvaluator over three-capsule cables, without settling."""
    values = {
        "settle_frames": 0,
        "settle_mode": "dynamic",
        "sim_iterations": 10,
        "stretch_stiffness": 1000.0,
        "num_elements": 3,
        "segment_length": 0.05,
        "cable_radius": 0.01,
        "cable_mass": 0.02,
        "angle_parametrization": ANGLE_PARAM_EXP,
    }
    values.update(overrides)
    return CableEvaluator(goals, loss, **values)


class TestTuningEvaluation(unittest.TestCase):
    """Scoring a population against goals with CableEvaluator."""

    def test_evaluator_passes_its_settings_to_the_world(self):
        """Verify evaluate() and record() build and settle the world with the evaluator's settings.

        The world also gets the goal's grasp and drive, and each candidate's
        values. The frame rate also sets the goal timeline: the last reference
        frame at 0.05 s is frame 1.5, which rounds to 2, so the timeline has 3
        frames at 30 fps.
        """
        built, settled = [], []

        class CableWorldRecorded(CableWorld):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                built.append((args, kwargs))

            def settle(self, settle_frames):
                settled.append(settle_frames)

        drive = fixed_drive()
        # A clamp one capsule from the end, 0.01 m along the TCP z axis, with the cable along the TCP x axis.
        grasp = {
            "attachment_transform": ((0.0, 0.0, 0.01), HANG_DOWN[1]),
            "clamp_position": 0.05,
            "cable_axis": (1.0, 0.0, 0.0),
        }
        goal = make_scene_goal(driven=True, data_source=DriveRecorder(drive), **grasp)
        settings = {
            "fps": 30,
            "sim_substeps": 7,
            "sim_iterations": 11,
            "settle_check_every": 5,
            "settle_move_tol": 2.0e-3,
            "stretch_stiffness": 500.0,
            "settle_mode": "dynamic",
            "num_elements": 3,
            "segment_length": 0.06,
            "cable_radius": 0.01,
            "cable_mass": 0.02,
            "angle_parametrization": ANGLE_PARAM_EXP,
        }
        evaluator = make_evaluator([goal], LossNodeDistance(), settle_frames=3, **settings)
        self.assertEqual(evaluator.groups[0].num_frames, 3)
        angles = [(0.1, 0.0), (0.0, 0.2), (0.3, 0.0)]
        candidate = CableCandidate(
            angles=angles, bend_stiffness=1.5, twist_stiffness=2.5, bend_damping=0.03, twist_damping=0.04
        )
        with patch.object(evaluate_module, "CableWorld", CableWorldRecorded):
            evaluator.evaluate([candidate])
            evaluator.record(candidate)

        self.assertEqual(len(built), 2)
        self.assertEqual(settled, [3, 3])
        for args, kwargs in built:
            self.assertEqual({name: kwargs[name] for name in settings}, settings)
            self.assertEqual({name: kwargs[name] for name in grasp}, grasp)
            # The angles, bend stiffness and drive are the second to fourth positional arguments.
            self.assertEqual(args[1], [angles])
            self.assertEqual(args[2], [1.5])
            self.assertIs(args[3], drive)
            self.assertEqual(kwargs["cameras"], evaluator.groups[0].cameras())
            self.assertEqual(kwargs["twist_stiffness_list"], [2.5])
            self.assertEqual(kwargs["bend_damping_list"], [0.03])
            self.assertEqual(kwargs["twist_damping_list"], [0.04])

    def test_population_is_scored_in_one_pass(self):
        """Verify a population is scored in one CableWorld and each candidate gets its own value.

        Candidates that differ only in rest angles get different values, equal
        candidates get equal values, and a candidate scores the same alone as in
        the population. An empty population scores nothing. With two recordings,
        each with its own goal mask, the value is the sum of the two recordings'
        values.
        """
        evaluator = make_evaluator([make_scene_goal()], LossNodeDistance())
        self.assertEqual(evaluator.evaluate([]), [])

        candidates = [make_candidate(), make_candidate((0.3, 0.0)), make_candidate((0.0, 0.3)), make_candidate()]
        built = []

        class CableWorldCounted(CableWorld):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                built.append(self.n)

        with patch.object(evaluate_module, "CableWorld", CableWorldCounted):
            values = evaluator.evaluate(candidates)
        self.assertEqual(built, [4])
        self.assertEqual(len(values), 4)
        self.assertEqual(len(set(values[:3])), 3)
        self.assertAlmostEqual(values[3], values[0], places=5)
        self.assertAlmostEqual(evaluator.evaluate([candidates[2]])[0], values[2], places=3)

        shifted = make_scene_goal(label="shifted", recording_key="rec1", masks=[np.roll(scene_mask(), 4, axis=1)] * 2)
        single = [
            make_evaluator([goal], LossNodeDistance()).evaluate([candidates[2]])[0]
            for goal in (make_scene_goal(), shifted)
        ]
        both = make_evaluator([make_scene_goal(), shifted], LossNodeDistance()).evaluate([candidates[2]])[0]
        # The recordings score differently, so a mix-up of their goals changes the sum.
        self.assertNotAlmostEqual(single[0], single[1], places=1)
        self.assertAlmostEqual(both, sum(single), places=3)

    def test_record_returns_one_trace_per_view(self):
        """Verify record() returns one trace per view, with every frame paired with its reference frame.

        Two cameras see one recording, so each view has half the weight. The
        reference frames at 0 s and 0.05 s span four simulated frames; frames 0
        and 1 are nearest the first reference, frames 2 and 3 the second. The
        per-frame losses add up to the view's loss, and the weighted view losses
        add up to the value evaluate() gives. Each view keeps its own frames,
        masks and loss.
        """
        goals = [make_scene_goal(label="front"), make_scene_goal(label="side", viewing_dir=(-1.0, 0.0, 0.0))]
        evaluator = make_evaluator(goals, LossNodeDistance())
        candidate = make_candidate((0.0, 0.3))
        views = evaluator.record(candidate)

        self.assertEqual([v.label for v in views], ["front", "side"])
        # At frame 0 both views see the same straight cable; the bent cable then looks different from each side.
        self.assertFalse(np.array_equal(views[0].masks[-1], views[1].masks[-1]))
        self.assertFalse(np.array_equal(views[0].frames[-1], views[1].frames[-1]))
        self.assertGreater(abs(views[0].loss - views[1].loss), 0.05)
        for view in views:
            with self.subTest(view.label):
                self.assertEqual(view.weight, 0.5)
                self.assertEqual(view.crop, SCENE_CROP)
                self.assertEqual([f.shape for f in view.frames], [(24, 32, 4)] * 4)
                self.assertEqual([m.shape for m in view.masks], [(24, 32)] * 4)
                self.assertEqual(view.goal_indices, [0, 0, 1, 1])
                np.testing.assert_allclose([t for t, _ in view.frame_losses], [0.0, 0.05])
                self.assertAlmostEqual(sum(v for _, v in view.frame_losses) / 2, view.loss, places=5)
        self.assertAlmostEqual(evaluator.evaluate([candidate])[0], sum(v.weight * v.loss for v in views), places=4)

    def test_record_keeps_frames_at_the_record_rate(self):
        """Verify record() keeps only the frames of record_fps and pairs each with its reference frame.

        The references at 0 s and 0.05 s span four frames at 60 fps. At half that
        rate, record() keeps two frames, the first and the last.
        """
        evaluator = make_evaluator([make_scene_goal()], LossNodeDistance())
        (view,) = evaluator.record(make_candidate(), record_fps=evaluator.fps / 2)
        self.assertEqual(len(view.frames), 2)
        self.assertEqual(len(view.masks), 2)
        self.assertEqual(view.goal_indices, [0, 1])

    def test_record_pairs_frames_by_capture_time(self):
        """Verify evaluate() and record() score each reference frame at the simulated frame of its capture time.

        The references at 0 s, 0.01 s and 0.05 s are not evenly spaced. At 60 fps
        they fall on frames 0, 1 and 3 (0.6 frames rounds to 1). Pairing by index
        would score frames 0, 2 and 3. record() also pairs each simulated frame
        with the nearest reference: frame 2 (0.033 s) is nearest 0.05 s.
        """
        goal = make_scene_goal(masks=[scene_mask()] * 3, frame_times=[0.0, 0.01, 0.05])
        evaluator = make_evaluator([goal], LossNodeDistance())
        candidate = make_candidate((0.0, 0.3))
        (view,) = evaluator.record(candidate)
        self.assertEqual(view.goal_indices, [0, 1, 2, 2])
        fps = evaluator.fps
        np.testing.assert_allclose([t for t, _ in view.frame_losses], [0.0, 1 / fps, 3 / fps])
        self.assertAlmostEqual(evaluator.evaluate([candidate])[0], view.loss, places=5)

    def test_recorded_frames_pair_with_the_reference_nearest_in_time(self):
        """Verify a simulated frame pairs with the reference frame nearest to it in capture time.

        This is the inverse of the reference-to-simulation matching in
        run_sequence.
        """
        fps = 60.0
        # References at 0.0, 0.5 and 1.0 s; simulated frame 30 is at 0.5 s.
        times = [0.0, 0.5, 1.0]
        self.assertEqual(_goal_index_for_sim_frame(0, times, fps), 0)
        self.assertEqual(_goal_index_for_sim_frame(30, times, fps), 1)
        self.assertEqual(_goal_index_for_sim_frame(60, times, fps), 2)
        # Nearest, not preceding: frame 25 (0.417 s) is closer to 0.5 s than to 0.0 s.
        self.assertEqual(_goal_index_for_sim_frame(25, times, fps), 1)
        # Every frame pairs with a single reference.
        self.assertEqual(_goal_index_for_sim_frame(42, times[:1], fps), 0)


if __name__ == "__main__":
    unittest.main()
