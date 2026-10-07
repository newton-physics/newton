# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the calibration run spec, search space, optimizer, search orchestration and cable problem."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.calibration.calibrate import calibrate
from newton._src.calibration.evaluate import CableTraceView
from newton._src.calibration.evidence import CableEvidenceBundle, CableRecording
from newton._src.calibration.optimizer import OptimizerCMA, SearchOutcome, optimizer_from_spec
from newton._src.calibration.problem import CableCalibrationProblem
from newton._src.calibration.result import STATUS_INTERRUPTED, STATUS_OK, CableCalibrationResult
from newton._src.calibration.run_spec import CableRunSpec, CalibrationOutputSpec
from newton._src.calibration.search_space import CableSearchSpace
from newton._src.calibration.setup_spec import CableSetupSpec
from newton._src.calibration.trace import CableTraceWriter

SETUP_FIELDS = {
    "num_elements",
    "segment_length",
    "cable_radius",
    "cable_mass",
    "angle_parametrization",
    "attachment_transform",
    "clamp_position",
    "stretch_stiffness",
}


def make_run_spec(**overrides):
    """A small cable run spec that searches only the log10 bend stiffness."""
    values = {
        "num_elements": 3,
        "segment_length": 0.05,
        "cable_radius": 0.003,
        "cable_mass": 0.03,
        "angle_parametrization": "exp_map",
        "attachment_transform": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "clamp_position": 0.0,
        "stretch_stiffness": 1000.0,
        "optimizer": {"kind": "cma"},
        "loss": "chamfer",
        "init_bend_stiffness": 100.0,
        "init_bend_damping": 0.01,
        "settle_mode": "dynamic",
        "settle_frames": 0,
        "sim_iterations": 20,
        "opt_rest_config": False,
        "opt_bend_damping": False,
    }
    setup = CableSetupSpec(**{name: values.pop(name) for name in SETUP_FIELDS})
    return CableRunSpec(**({"setup": setup} | values | overrides))


def write_bundle(directory):
    """Write a one-frame static bundle with a mask and an RGB frame."""
    from PIL import Image

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 7:9] = 255
    Image.fromarray(mask).save(directory / "mask.png")
    Image.fromarray(np.repeat(mask[:, :, None], 3, axis=2)).save(directory / "rgb.png")
    bundle = CableEvidenceBundle(
        recordings=[
            CableRecording(
                label="camera",
                recording_key="static",
                applied_settings={},
                rgb_frames=["rgb.png"],
                masks=["mask.png"],
                frame_timestamps=[0.0],
                sensor_pos=[0.0, 0.0, -1.0],
                sensor_quat=[0.0, 0.0, 0.0, 1.0],
                camera_intrinsics=[16, 16, 20.0, 20.0, 8.0, 8.0],
                cable_start=[0.0, 0.0, 0.5],
            )
        ],
    )
    bundle.save(directory)
    return bundle


class TestCalibrationRunSpec(unittest.TestCase):
    """Serialization and validation of the run spec and the search space."""

    def test_round_trip_through_json(self):
        """Verify a run spec reads back equal from JSON and refuses unknown fields."""
        spec = make_run_spec(optimizer={"kind": "cma", "seed": 17, "maxiter": 2}, output=CalibrationOutputSpec("out"))
        data = json.loads(json.dumps(spec.to_dict()))
        self.assertEqual(CableRunSpec.from_dict(data), spec)
        for part in (data, data["setup"], data["output"]):
            with self.subTest(keys=sorted(part)), self.assertRaisesRegex(ValueError, "unknown field"):
                part["unknown_option"] = True
                CableRunSpec.from_dict(data)
            del part["unknown_option"]

    def test_unsupported_schema_version_is_refused(self):
        """Verify a missing or unknown schema version is refused."""
        for version in (None, 2):
            with self.subTest(version=version), self.assertRaises(ValueError):
                CableRunSpec.from_dict({"schema_version": version})

    def test_setup_is_checked(self):
        """Verify a missing or malformed setup value is refused, not replaced by a default."""
        for settings in (
            {"num_elements": True},
            {"segment_length": float("nan")},
            {"angle_parametrization": "euler_xy"},
            {"cable_axis": [0.0, 0.0, 0.0]},
        ):
            spec = make_run_spec()
            for name, value in settings.items():
                setattr(spec.setup, name, value)
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                spec.validate()
        for transform in (None, [], [[0, 0, 0], [0, 0, 0, 0]]):
            spec = make_run_spec()
            spec.setup.attachment_transform = transform
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                spec.validate()
        for clamp_position in (None, -0.01, 0.16):
            spec = make_run_spec()
            spec.setup.clamp_position = clamp_position
            with self.subTest(clamp_position=clamp_position), self.assertRaises(ValueError):
                spec.validate()
        for name in ("attachment_transform", "clamp_position", "stretch_stiffness"):
            data = make_run_spec().to_dict()
            del data["setup"][name]
            with self.subTest(name=name), self.assertRaises(TypeError):
                CableRunSpec.from_dict(data)

    def test_fit_settings_are_checked(self):
        """Verify the loss, start, simulation and output settings are checked before a fit."""
        for settings in (
            {"loss": "l2"},
            {"init_bend_stiffness": float("nan")},
            {"init_twist_damping": -1.0},
            {"twist_damping_mode": True},
            {"twist_stiffness_mode": -1.0},
            {"init_angles": [[0.0, 0.0, 0.0]] * 3},
            {"init_angles": [[0.0, 0.0]] * 2},
            {"settle_mode": "newton"},
            {"settle_frames": -1},
            {"sim_iterations": 0},
            {"output": CalibrationOutputSpec(None, 2)},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_run_spec(**settings).validate()
        with self.assertRaisesRegex(ValueError, "mapping"):
            CableRunSpec.from_dict({**make_run_spec().to_dict(), "setup": [3]})

        # One element has no searchable rest angle, so this spec searches nothing.
        spec = make_run_spec(opt_rest_config=True, opt_bend_stiffness=False)
        spec.setup.num_elements = 1
        with self.assertRaisesRegex(ValueError, "no parameter"):
            spec.validate()

    def test_decode_inverts_the_search_start(self):
        """Verify each searched group has its own slot and decoding the start gives the start values."""
        angles = [[0.1, -0.2], [0.3, 0.0], [0.5, 0.6]]
        spec = make_run_spec(
            opt_rest_config=True,
            opt_bend_damping=True,
            twist_stiffness_mode="fit",
            init_twist_stiffness=7.0,
            twist_damping_mode="fit",
            init_twist_damping=0.25,
            init_angles=angles,
        )
        space = CableSearchSpace(spec)
        scalars = ["bend_stiffness", "twist_stiffness", "bend_damping", "twist_damping"]
        self.assertEqual(space.dim, 2 * 2 + 4)
        self.assertEqual(space.searched_scalars(), scalars)
        self.assertEqual(spec.searched_groups(), ["rest_config", *scalars])
        candidate = space.decode(space.x0())
        np.testing.assert_allclose(candidate.angles, angles)
        self.assertAlmostEqual(candidate.bend_stiffness, 100.0)
        self.assertAlmostEqual(candidate.twist_stiffness, 7.0)
        self.assertAlmostEqual(candidate.bend_damping, 0.01)
        self.assertAlmostEqual(candidate.twist_damping, 0.25)
        self.assertEqual(
            space.describe(space.x0()),
            "bend_stiffness=100, twist_stiffness=7, bend_damping=0.01, twist_damping=0.25, 4 rest angles",
        )

        coupled = CableSearchSpace(make_run_spec(twist_stiffness_mode="coupled", twist_damping_mode=0.25))
        self.assertEqual(coupled.decode([2.5]).twist_stiffness, 10.0**2.5)
        self.assertEqual(coupled.decode([2.5]).twist_damping, 0.25)

    def test_non_positive_search_start_is_refused(self):
        """Verify a log-space search refuses a start value that is not positive."""
        for settings in (
            {"twist_damping_mode": "fit", "init_twist_damping": 0.0},
            {"twist_stiffness_mode": "fit", "init_twist_stiffness": -1.0},
        ):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "positive"):
                CableSearchSpace(make_run_spec(**settings))

    def test_optimizer_factory(self):
        """Verify the factory refuses unknown kinds and settings and does not change its input."""
        for settings in (
            {"kind": "missing"},
            {"kind": "cma", "unknown_option": 1},
            {"kind": "cma", "seed": True},
            {"kind": "cma", "seed": 2**32},
            {"kind": "cma", "sigma0": 0.0},
            {"kind": "cma", "maxiter": 2.5},
            {"kind": "cma", "popsize": 1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                optimizer_from_spec(settings)
        settings = {"kind": "cma", "seed": 5, "popsize": 8, "maxiter": 2}
        before = dict(settings)
        optimizer = optimizer_from_spec(settings)
        self.assertIsInstance(optimizer, OptimizerCMA)
        self.assertEqual(settings, before)
        self.assertEqual(optimizer.to_dict(), {"kind": "cma", "seed": 5, "sigma0": 0.1, "popsize": 8, "maxiter": 2})


@unittest.skipUnless(importlib.util.find_spec("cma"), "requires newton[calibration]")
class TestCalibrationOrchestration(unittest.TestCase):
    """:func:`calibrate` and the optimizer on a problem that is not a cable."""

    class Space:
        dim = 2

        def __init__(self, start=(3.0, -2.0)):
            self.start = list(start)

        def x0(self):
            return self.start

        def describe(self, vector):
            return str(vector)

    class Problem:
        """A quadratic problem, to show that :func:`calibrate` does not depend on the model type."""

        def __init__(self, start=(3.0, -2.0)):
            self.search_space = TestCalibrationOrchestration.Space(start)
            self.objective = self
            self.closed = False

        def evaluate(self, vectors):
            return [float(np.sum(np.square(x))) for x in vectors]

        def build_result(self, outcome, *, optimizer, artifacts):
            return outcome

        def close(self):
            self.closed = True

    def test_cma_solves_a_problem_reproducibly(self):
        """Verify CMA-ES with a seed finds the minimum, records the best so far and gives the same result twice."""
        outcomes = []
        for _ in range(2):
            problem = self.Problem()
            result = calibrate(problem, optimizer=OptimizerCMA(seed=5, popsize=12, maxiter=80, sigma0=0.5))
            self.assertTrue(problem.closed)
            self.assertLess(result.loss, 1e-8)
            self.assertEqual(result.history[0], (13.0, [3.0, -2.0]))
            self.assertEqual(len(result.history), result.iterations + 1)
            self.assertEqual(result.history[-1], (result.loss, result.vector))
            losses = [loss for loss, _ in result.history]
            self.assertEqual(losses, sorted(losses, reverse=True))
            outcomes.append(result)
        self.assertEqual(outcomes[0], outcomes[1])

    def test_start_is_kept_when_no_sample_is_better(self):
        """Verify the search returns the start when it is already the minimum."""
        result = calibrate(self.Problem(start=(0.0, 0.0)), optimizer=OptimizerCMA(seed=1, popsize=4, maxiter=2))
        self.assertEqual(result.vector, [0.0, 0.0])
        self.assertEqual(result.loss, 0.0)

    def test_problem_is_closed_when_evaluation_fails(self):
        """Verify a non-finite objective value stops the search and closes the problem."""
        problem = self.Problem()
        problem.evaluate = lambda vectors: [float("nan")] * len(vectors)
        with self.assertRaisesRegex(ValueError, "finite scalar"):
            calibrate(problem, optimizer=OptimizerCMA(seed=1, popsize=4, maxiter=2))
        self.assertTrue(problem.closed)

    def test_interrupt_keeps_the_best_candidate(self):
        """Verify an interrupt after the first iteration returns the best candidate so far."""
        problem = self.Problem()
        evaluate, calls = problem.evaluate, []

        def interrupt_third_batch(vectors):
            calls.append(len(vectors))
            if len(calls) == 3:
                raise KeyboardInterrupt
            return evaluate(vectors)

        problem.evaluate = interrupt_third_batch
        result = calibrate(problem, optimizer=OptimizerCMA(seed=1, popsize=4, maxiter=5))
        self.assertTrue(result.interrupted)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.history[-1], (result.loss, result.vector))
        self.assertLessEqual(result.loss, 13.0)
        self.assertTrue(problem.closed)


class TestCalibrationCableProblem(unittest.TestCase):
    """:class:`CableCalibrationProblem` from a bundle on disk, with the simulation replaced."""

    @unittest.skipUnless(importlib.util.find_spec("cma"), "requires newton[calibration]")
    def test_fit_and_output_round_trip(self):
        """Verify a fit from a bundle on disk gives a result that reads back from its output directory.

        A known function of the bend stiffness replaces the simulation.
        """

        class Evaluator:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate(self, candidates):
                return [(np.log10(c.bend_stiffness) - 1.5) ** 2 for c in candidates]

        with tempfile.TemporaryDirectory() as tmp, patch("newton._src.calibration.problem.CableEvaluator", Evaluator):
            directory = Path(tmp)
            spec = make_run_spec()
            problem = CableCalibrationProblem.from_bundle(write_bundle(directory), directory, spec)
            spec.init_bend_stiffness = 999.0
            out = directory / "result"
            optimizer = OptimizerCMA(seed=3, sigma0=0.5, popsize=6, maxiter=40)
            result = calibrate(problem, optimizer=optimizer, output_dir=out)

            self.assertEqual(result.status, STATUS_OK)
            self.assertLess(result.metrics["loss"], 1e-10)
            self.assertAlmostEqual(math.log10(result.fit["bend_stiffness"]), 1.5, places=4)
            self.assertEqual(result.fit["twist_stiffness"], result.fit["bend_stiffness"])
            self.assertEqual(result.fit["bend_damping"], 0.01)
            loaded = CableCalibrationResult.from_dict(json.loads((out / "result.json").read_text()))
            self.assertEqual(loaded, result)
            saved = CableRunSpec.from_dict(result.provenance["run_spec"])
            self.assertEqual(saved.init_bend_stiffness, 100.0)
            self.assertEqual(saved.setup, spec.setup)
            self.assertEqual(saved.optimizer, optimizer.to_dict())
            self.assertEqual(result.provenance["seed"], 3)
            self.assertEqual(result.artifacts, {"result": "result.json", "history": "history.json"})
            for relative in result.artifacts.values():
                self.assertTrue((out / relative).exists())
            history = json.loads((out / "history.json").read_text())
            self.assertEqual(len(history), result.metrics["iterations"] + 1)
            self.assertEqual(history[0]["loss"], 0.25)
            self.assertEqual(history[-1]["loss"], result.metrics["loss"])
            self.assertIsNone(problem._evaluator)

            stopped = SearchOutcome([2.0], 0.0, 0, True, [(0.0, [2.0])])
            self.assertEqual(
                problem.build_result(stopped, optimizer=optimizer, artifacts={}).status, STATUS_INTERRUPTED
            )

    @unittest.skipUnless(importlib.util.find_spec("cma"), "requires newton[calibration]")
    def test_failed_trace_keeps_the_result(self):
        """Verify the result and the history are on disk when rendering a trace fails."""

        class Evaluator:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate(self, candidates):
                return [(np.log10(c.bend_stiffness) - 1.5) ** 2 for c in candidates]

            def record(self, candidate):
                raise RuntimeError("render failed")

        with tempfile.TemporaryDirectory() as tmp, patch("newton._src.calibration.problem.CableEvaluator", Evaluator):
            directory = Path(tmp)
            problem = CableCalibrationProblem.from_bundle(write_bundle(directory), directory, make_run_spec())
            out = directory / "result"
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                calibrate(problem, optimizer=OptimizerCMA(seed=3, popsize=4, maxiter=2), output_dir=out, trace_every=1)
            for name in ("result.json", "history.json"):
                self.assertTrue((out / name).exists(), name)

    def test_evaluator_receives_the_setup_and_run_settings(self):
        """Verify the setup reaches the goals and the evaluator."""
        with tempfile.TemporaryDirectory() as tmp, patch("newton._src.calibration.problem.CableEvaluator") as evaluator:
            directory = Path(tmp)
            spec = make_run_spec(sim_iterations=37, settle_frames=12)
            spec.setup.attachment_transform = [[0.1, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            spec.setup.clamp_position = 0.05
            spec.setup.cable_axis = [1.0, 0.0, 0.0]
            problem = CableCalibrationProblem.from_bundle(write_bundle(directory), directory, spec)
            goal = problem.goals[0]
            self.assertEqual(goal.attachment_transform, ((0.1, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)))
            self.assertEqual(goal.clamp_position, 0.05)
            self.assertEqual(goal.cable_axis, (1.0, 0.0, 0.0))

            problem._prepare_evaluator()
            settings = evaluator.call_args.kwargs
            self.assertEqual(settings["sim_iterations"], 37)
            self.assertEqual(settings["settle_frames"], 12)
            self.assertEqual(settings["settle_mode"], "dynamic")
            self.assertEqual(settings["stretch_stiffness"], 1000.0)
            self.assertEqual(settings["num_elements"], 3)
            self.assertEqual(settings["segment_length"], 0.05)
            self.assertEqual(settings["cable_radius"], 0.003)
            self.assertEqual(settings["cable_mass"], 0.03)
            problem.close()

    def test_trace_writes_the_cropped_masks_and_rgb_frames(self):
        """Verify the trace writes the recorded mask cropped like the reference, and the RGB channels of the frame."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle = write_bundle(directory)
            rgba = np.zeros((16, 16, 4), dtype=np.uint8)
            rgba[..., 0] = 200
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[5:8, 6:9] = 255
            view = CableTraceView(
                label="camera",
                crop=[3, 4, 12, 13],
                weight=1.0,
                loss=0.0,
                frames=[rgba, rgba],
                masks=[mask, mask],
                goal_indices=[0, 0],
                frame_losses=[(0.0, 0.0), (0.1, 0.0)],
            )
            output = directory / "trace"
            CableTraceWriter(str(output), bundle, str(directory)).write(1, [view], 0.0)
            with Image.open(output / "iter_001_camera_simmask/frame_000.png") as image:
                np.testing.assert_array_equal(np.asarray(image), mask[4:13, 3:12])
            with Image.open(output / "iter_001_camera/frame_000.png") as image:
                np.testing.assert_array_equal(np.asarray(image), rgba[..., :3])
            self.assertTrue((output / "goal_seq_camera_mask/frame_001.png").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
