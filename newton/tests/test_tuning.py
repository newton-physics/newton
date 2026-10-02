# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tuning contracts, goals and bundle rehydration."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import tempfile
import unittest

import numpy as np

from newton._src.tuning.data_source import CableDataSource
from newton._src.tuning.evidence import CableEvidenceBundle, CableRecording
from newton._src.tuning.goal import CableGoal, group_goals
from newton._src.tuning.rehydrate import bundle_to_goals
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
        "sensor_quat": [0.0, 0.0, 0.0, 1.0],
        "camera_intrinsics": (20, 20, 10.0, 10.0, 10.0, 10.0),
        "start_ns": 1_000_000_000,
    }
    values.update(overrides)
    return CableGoal(**values)


class DriveRecorder:
    """A data source that records the drive it is asked for instead of sampling one."""

    def __init__(self):
        self.calls = []

    def drive_buffer(self, start_ns, num_frames, substep_rate, sim_substeps):
        self.calls.append((start_ns, num_frames, substep_rate, sim_substeps))
        return "drive"


class TestTuningGoals(unittest.TestCase):
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

    def test_group_weights_split_one_unit_per_recording(self):
        """Verify each recording contributes one unit of weight, split over its views.

        Two views of one recording weigh half each, so adding a camera does not
        multiply that recording's say in the parameter trade-offs.
        """
        goals = [
            make_goal(label="a", recording_key="rec0"),
            make_goal(label="b", recording_key="rec0"),
            make_goal(label="c", recording_key="rec1"),
        ]
        groups = group_goals(goals, 60, 10)
        self.assertEqual([g.weights for g in groups], [[0.5, 0.5], [1.0]])

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

    def test_single_view_group_reads_its_own_drive(self):
        """Verify a single view keeps its frame times and reads the drive from its own origin."""
        source = DriveRecorder()
        goal = make_goal(frame_times=[0.0, 0.5], masks=make_goal().masks * 2, driven=True, data_source=source)
        (group,) = group_goals([goal], 60, 10)
        self.assertIs(group.views[0], goal)
        self.assertEqual(group.num_frames, 31)
        self.assertEqual(source.calls, [(1_000_000_000, 31, 600, 10)])

    def test_timeline_rounds_to_the_nearest_frame(self):
        """Verify the last frame time rounds to the nearest simulated frame, not down.

        Camera frames rarely fall on the simulation's frame grid. A last frame at
        0.03 s is 1.8 frames at 60 fps; truncating would end the timeline before it.
        """
        masks = make_goal().masks * 2
        (single,) = group_goals([make_goal(frame_times=[0.0, 0.03], masks=masks)], 60, 10)
        self.assertEqual(single.num_frames, 3)
        (multi,) = group_goals(
            [make_goal(label="a", frame_times=[0.0, 0.03], masks=masks), make_goal(label="b")], 60, 10
        )
        self.assertEqual(multi.num_frames, 3)

    def test_group_adapts_views_for_one_or_several_cameras(self):
        """Verify the group's camera, argument and result helpers follow the camera count.

        A single-camera group passes bare values to the simulation and wraps its
        result in a list; a multi-camera group passes and returns per-view lists.
        """
        (single,) = group_goals([make_goal(render_size=(20, 20), fov_deg=60.0)], 60, 10)
        self.assertEqual(single.run_args(["r"]), ("r", [0, 0, 20, 20], [0.0]))
        self.assertEqual(single.per_view("x"), ["x"])
        (camera,) = single.cameras()
        self.assertEqual(camera["sensor_pos"], (1.0, 0.0, 0.5))
        self.assertEqual(camera["sensor_quat"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(camera["camera_intrinsics"], (20, 20, 10.0, 10.0, 10.0, 10.0))
        self.assertEqual(camera["render_size"], (20, 20))
        self.assertEqual(camera["fov_deg"], 60.0)

        (multi,) = group_goals([make_goal(label="a"), make_goal(label="b")], 60, 10)
        self.assertEqual(multi.run_args(["r0", "r1"]), (["r0", "r1"], [[0, 0, 20, 20]] * 2, [[0.0], [0.0]]))
        self.assertEqual(multi.per_view(["x", "y"]), ["x", "y"])
        self.assertEqual(len(multi.cameras()), 2)

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

    def test_grasp_given_as_lists_agrees_with_tuples(self):
        """Verify views that give the grasp as lists compare like views that give tuples.

        Goals built in code may hold lists; they must agree or disagree by value,
        not fail because a list cannot be hashed.
        """
        as_tuples = ((0.0, 0.0, 0.0), IDENTITY)
        as_lists = [[0.0, 0.0, 0.0], list(IDENTITY)]
        goals = [
            make_goal(label="a", attachment_transform=as_tuples),
            make_goal(label="b", attachment_transform=as_lists),
        ]
        (group,) = group_goals(goals, 60, 10)
        self.assertEqual(group.attachment_transform, as_tuples)

        goals = [make_goal(label="a", cable_axis=[1.0, 0.0, 0.0]), make_goal(label="b", cable_axis=[0.0, 1.0, 0.0])]
        with self.assertRaisesRegex(ValueError, "rec0.*cable_axis"):
            group_goals(goals, 60, 10)

    def test_group_rejects_views_disagreeing_on_the_drive(self):
        """Verify grouping raises, naming the recording, when views disagree on the drive.

        The drive belongs to the recording, so a group reads it once for all
        views. A view without the drive, or with another data source, would
        otherwise be replayed with the wrong motion.
        """
        undriven = make_goal(label="a")
        driven = make_goal(label="b", start_ns=1_100_000_000, driven=True, data_source=DriveRecorder())
        with self.assertRaisesRegex(ValueError, "rec0.*driven"):
            group_goals([undriven, driven], 60, 10)

        other = make_goal(label="c", driven=True, data_source=DriveRecorder())
        with self.assertRaisesRegex(ValueError, "rec0.*data_source"):
            group_goals([driven, other], 60, 10)

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
class TestTuningRehydrate(unittest.TestCase):
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
            bundle_to_goals(duplicate, ".", attachment_transform=self.ATTACHMENT)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording()])
            with open(os.path.join(tmp, "trajectories", "rec0.json"), "w") as fh:
                json.dump(make_trajectory(timestamps_ns=[0, 10, 10]).to_dict(), fh)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT)

    def test_masks_must_match_each_other_and_the_camera(self):
        """Verify masks of different sizes, or of another size than the camera describes, are refused.

        The camera data describes the stored masks. A mask of another size would
        be scored against a projection at the wrong scale.
        """
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording()])
            write_mask(os.path.join(tmp, "masks", "cam0", "2.png"), 30, shape=(12, 16))
            with self.assertRaisesRegex(ValueError, r"cam0.*masks/cam0/2\.png.*16x12"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording(camera_intrinsics=[64, 48, 80.0, 80.0, 32.0, 24.0])])
            with self.assertRaisesRegex(ValueError, "cam0.*32x24.*64x48"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT)

    def test_missing_files_are_reported_with_the_recording(self):
        """Verify a missing mask or trajectory raises, naming the recording and the file."""
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording()])
            os.remove(os.path.join(tmp, "masks", "cam0", "1.png"))
            with self.assertRaisesRegex(ValueError, r"cam0.*masks/cam0/1\.png"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.write_bundle(tmp, [make_recording()])
            os.remove(os.path.join(tmp, "trajectories", "rec0.json"))
            with self.assertRaisesRegex(ValueError, r"cam0.*trajectories/rec0\.json"):
                bundle_to_goals(bundle, tmp, attachment_transform=self.ATTACHMENT)


if __name__ == "__main__":
    unittest.main()
