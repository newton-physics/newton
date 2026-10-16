# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the calibration setup, search space, optimizer, search orchestration and cable problem."""

from __future__ import annotations

import importlib.util
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newton._src.calibration.cable_setup import CableSetup
from newton._src.calibration.calibrate import calibrate
from newton._src.calibration.evaluate import CableParameters, CableTraceView
from newton._src.calibration.evidence import CableEvidenceBundle, CableRecording
from newton._src.calibration.optimizer import CMASettings, SearchOutcome
from newton._src.calibration.problem import CableCalibrationProblem
from newton._src.calibration.result import STATUS_FAILED, STATUS_INTERRUPTED, STATUS_OK, CableCalibrationResult
from newton._src.calibration.search_space import CableSearchSpace
from newton._src.calibration.trace import CableTraceWriter

SIMULATION = {"settle_mode": "dynamic", "settle_frames": 0, "sim_iterations": 20}


def make_setup(**overrides):
    """A small three-element cable setup."""
    values = {
        "num_elements": 3,
        "segment_length": 0.05,
        "cable_radius": 0.003,
        "cable_mass": 0.03,
        "angle_parametrization": "exp_map",
        "attachment_transform": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "clamp_position": 0.0,
        "stretch_stiffness": 1000.0,
    }
    return CableSetup(**(values | overrides))


def make_start(**overrides):
    """Start values of a straight three-element cable."""
    values = {
        "angles": [(0.0, 0.0)] * 3,
        "bend_stiffness": 100.0,
        "twist_stiffness": 100.0,
        "bend_damping": 0.01,
        "twist_damping": 0.01,
    }
    return CableParameters(**(values | overrides))


def make_space(start=None, **overrides):
    """A search space that searches only the bend stiffness, with the twist coupled to the bend."""
    values = {"search": ["bend_stiffness"], "bounds": {"bend_stiffness": (1.0, 1.0e4)}, "couple_twist": True}
    return CableSearchSpace(start or make_start(), **(values | overrides))


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


class TestCalibrationSetupAndSearchSpace(unittest.TestCase):
    """Validation of the cable setup, the search space and the optimizer settings."""

    def test_setup_is_checked(self):
        """Verify a malformed setup value is refused, not replaced by a default."""
        for settings in (
            {"num_elements": True},
            {"segment_length": float("nan")},
            {"angle_parametrization": "euler_xy"},
            {"cable_axis": [0.0, 0.0, 0.0]},
            {"attachment_transform": None},
            {"attachment_transform": []},
            {"attachment_transform": [[0, 0, 0], [0, 0, 0, 0]]},
            {"clamp_position": None},
            {"clamp_position": -0.01},
            {"clamp_position": 0.16},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_setup(**settings).validate()

    def test_decode_inverts_the_search_start(self):
        """Verify each searched field has its own slot, decoding the start gives the start, and bounds hold."""
        angles = [(0.1, -0.2), (0.3, 0.0), (0.5, 0.6)]
        start = make_start(angles=angles, twist_stiffness=7.0, twist_damping=0.25)
        scalars = ["bend_stiffness", "twist_stiffness", "bend_damping", "twist_damping"]
        bounds = {"bend_stiffness": (1.0, 1e4), "twist_stiffness": (1.0, 1e3), "bend_damping": (1e-4, 1.0)}
        space = CableSearchSpace(
            start,
            search=["twist_damping", "angles", *scalars[:3]],
            bounds=bounds | {"twist_damping": (1e-3, 1.0)},
            couple_twist=False,
        )
        self.assertEqual(space.dim, 2 * 2 + 4)
        self.assertEqual(space.search, ["angles", *scalars])
        self.assertEqual(space.searched_scalars(), scalars)
        decoded = space.decode(space.x0())
        np.testing.assert_allclose(decoded.angles, angles)
        for name in scalars:
            self.assertAlmostEqual(getattr(decoded, name), getattr(start, name))
        lower, upper = space.bounds()
        self.assertEqual(lower[:4], [-math.inf] * 4)
        np.testing.assert_allclose(upper[4:], [4.0, 3.0, 0.0, 0.0])
        self.assertEqual(
            space.describe(space.x0()),
            "bend_stiffness=100, twist_stiffness=7, bend_damping=0.01, twist_damping=0.25, 4 rest angles",
        )
        with self.assertRaisesRegex(ValueError, "within the bounds"):
            space.decode([0.0] * 4 + [400.0, 0.0, -2.0, -1.0])

        coupled = make_space()
        self.assertEqual(coupled.decode([2.5]).twist_stiffness, 10.0**2.5)
        self.assertEqual(coupled.decode([2.5]).twist_damping, 0.01)

    def test_search_space_is_checked(self):
        """Verify unknown fields, missing or invalid bounds, contradicting coupling and an empty search are refused."""
        for settings in (
            {"search": ["rest_config"]},
            {"search": "bend_stiffness"},
            {"bounds": {}},
            {"bounds": {"bend_stiffness": (1.0, 1e4), "bend_damping": (1e-4, 1.0)}},
            {"bounds": {"bend_stiffness": (0.0, 1e4)}},
            {"bounds": {"bend_stiffness": (1e4, 1.0)}},
            {"bounds": {"bend_stiffness": (200.0, 1e4)}},
            {"search": ["twist_stiffness"], "bounds": {"twist_stiffness": (1.0, 1e4)}},
            {"couple_twist": 1},
            {"start": make_start(twist_damping=0.5)},
            {"start": make_start(bend_stiffness=float("nan"))},
            {"start": make_start(angles=[(0.0, 0.0)]), "search": ["angles"], "bounds": {}},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_space(**settings)

    def test_optimizer_settings_are_checked(self):
        """Verify the CMA-ES settings are refused when out of range or of the wrong type."""
        for settings in ({"seed": True}, {"seed": 2**32}, {"sigma0": 0.0}, {"maxiter": 2.5}, {"popsize": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                CMASettings(**settings)
        optimizer = CMASettings(seed=5, popsize=8, maxiter=2)
        self.assertEqual(optimizer.to_dict(), {"kind": "cma", "seed": 5, "sigma0": 0.1, "popsize": 8, "maxiter": 2})


@unittest.skipUnless(importlib.util.find_spec("cma"), "requires newton[calibration]")
class TestCalibrationOrchestration(unittest.TestCase):
    """:func:`calibrate` and the optimizer on a problem that is not a cable."""

    class Space:
        dim = 2

        def __init__(self, start=(3.0, -2.0), lower=(-math.inf, -math.inf)):
            self.start = list(start)
            self.lower = list(lower)

        def x0(self):
            return self.start

        def bounds(self):
            return self.lower, [math.inf, math.inf]

        def describe(self, vector):
            return str(vector)

    class Problem:
        """A quadratic problem, to show that :func:`calibrate` does not depend on the model type."""

        def __init__(self, start=(3.0, -2.0), lower=(-math.inf, -math.inf)):
            self.search_space = TestCalibrationOrchestration.Space(start, lower)
            self.closed = False
            self.vectors = []

        def evaluate(self, vectors):
            self.vectors += [list(x) for x in vectors]
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
            result = calibrate(problem, optimizer=CMASettings(seed=5, popsize=12, maxiter=80, sigma0=0.5))
            self.assertTrue(problem.closed)
            self.assertLess(result.loss, 1e-8)
            self.assertEqual(result.history[0], (13.0, [3.0, -2.0]))
            self.assertEqual(len(result.history), result.iterations + 1)
            self.assertEqual(result.history[-1], (result.loss, result.vector))
            losses = [loss for loss, _ in result.history]
            self.assertEqual(losses, sorted(losses, reverse=True))
            outcomes.append(result)
        self.assertEqual(outcomes[0], outcomes[1])

    def test_caller_drives_the_search(self):
        """Verify the ask/tell loop scores the start first, can stop on the caller's criterion and refuses a tell without an ask."""
        problem = self.Problem()
        search = CMASettings(seed=1, popsize=4, maxiter=50, sigma0=1.0).start(problem.search_space)
        with self.assertRaisesRegex(RuntimeError, "ask"):
            search.tell([1.0])
        self.assertEqual(search.ask(), [[3.0, -2.0]])
        search.tell([13.0])
        while search.stop_reason is None and search.best_loss > 1.0:
            search.tell(problem.evaluate(search.ask()))
        outcome = search.outcome()
        self.assertEqual(outcome.stop_reason, "stopped by the caller")
        self.assertLessEqual(outcome.loss, 1.0)
        self.assertEqual(outcome.iterations, search.iteration)

    def test_pickled_search_continues_like_the_original(self):
        """Verify a search saved with pickle between iterations gives the same history as the original."""
        problem = self.Problem()
        search = CMASettings(seed=1, popsize=4, maxiter=30).start(problem.search_space)
        for _ in range(3):
            search.tell(problem.evaluate(search.ask()))
        resumed = pickle.loads(pickle.dumps(search))
        for s in (search, resumed):
            for _ in range(3):
                s.tell(problem.evaluate(s.ask()))
        self.assertEqual(resumed.history, search.history)

    def test_search_stays_within_the_bounds(self):
        """Verify every evaluated vector respects a lower bound that excludes the unbounded minimum."""
        problem = self.Problem(lower=(1.0, -math.inf))
        result = calibrate(problem, optimizer=CMASettings(seed=2, popsize=8, maxiter=30, sigma0=1.0))
        self.assertTrue(all(x[0] >= 1.0 for x in problem.vectors))
        self.assertAlmostEqual(result.vector[0], 1.0, places=3)

    def test_start_is_kept_when_no_sample_is_better(self):
        """Verify the search returns the start when it is already the minimum."""
        result = calibrate(self.Problem(start=(0.0, 0.0)), optimizer=CMASettings(seed=1, popsize=4, maxiter=2))
        self.assertEqual(result.vector, [0.0, 0.0])
        self.assertEqual(result.loss, 0.0)
        self.assertEqual(result.stop_reason, "maxiter")

    def test_invalid_start_is_refused_and_the_problem_closed(self):
        """Verify a start with a non-finite objective value stops the search and closes the problem."""
        problem = self.Problem()
        problem.evaluate = lambda vectors: [math.nan] * len(vectors)
        with self.assertRaisesRegex(ValueError, "start is invalid"):
            calibrate(problem, optimizer=CMASettings(seed=1, popsize=4, maxiter=2))
        self.assertTrue(problem.closed)

    def test_invalid_candidates_are_ranked_last(self):
        """Verify non-finite values never become the best, are counted, and stop the search when a whole iteration is invalid."""
        problem = self.Problem()
        evaluate = problem.evaluate
        # The region x[0] < 1 is invalid, so the minimum at the origin cannot be reached.
        problem.evaluate = lambda vectors: [
            v if x[0] >= 1.0 else math.nan for x, v in zip(vectors, evaluate(vectors), strict=True)
        ]
        result = calibrate(problem, optimizer=CMASettings(seed=2, popsize=8, maxiter=30, sigma0=1.0))
        self.assertGreater(result.invalid, 0)
        self.assertLess(result.loss, 1.03)  # the best valid point is (1, 0), with loss 1
        self.assertTrue(all(math.isfinite(loss) and x[0] >= 1.0 for loss, x in result.history))
        self.assertNotEqual(result.stop_reason, "all candidates invalid")

        problem = self.Problem()
        problem.evaluate = lambda vectors: [13.0 if list(x) == [3.0, -2.0] else math.nan for x in vectors]
        result = calibrate(problem, optimizer=CMASettings(seed=1, popsize=4, maxiter=5))
        self.assertEqual(result.stop_reason, "all candidates invalid")
        self.assertEqual((result.loss, result.vector, result.invalid), (13.0, [3.0, -2.0], 4))

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
        result = calibrate(problem, optimizer=CMASettings(seed=1, popsize=4, maxiter=5))
        self.assertTrue(result.interrupted)
        self.assertEqual(result.stop_reason, "interrupted")
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

            def evaluate(self, candidates, *, return_poses=False):
                return [(np.log10(c.bend_stiffness) - 1.5) ** 2 for c in candidates]

        with tempfile.TemporaryDirectory() as tmp, patch("newton._src.calibration.problem.CableEvaluator", Evaluator):
            directory = Path(tmp)
            setup, space = make_setup(), make_space()
            problem = CableCalibrationProblem.from_bundle(
                write_bundle(directory), directory, setup, space, **SIMULATION
            )
            space.start.bend_stiffness = 999.0
            out = directory / "result"
            optimizer = CMASettings(seed=3, sigma0=0.5, popsize=6, maxiter=40)
            result = calibrate(problem, optimizer=optimizer, output_dir=out)

            self.assertEqual(result.status, STATUS_OK)
            self.assertLess(result.metrics["loss"], 1e-10)
            self.assertAlmostEqual(math.log10(result.fit["bend_stiffness"]), 1.5, places=4)
            self.assertEqual(result.fit["twist_stiffness"], result.fit["bend_stiffness"])
            self.assertEqual(result.fit["bend_damping"], 0.01)
            loaded = CableCalibrationResult.from_dict(json.loads((out / "result.json").read_text()))
            self.assertEqual(loaded, result)
            provenance = result.provenance
            self.assertEqual(provenance["setup"], setup.to_dict())
            self.assertEqual(provenance["search_space"]["start"]["bend_stiffness"], 100.0)
            self.assertEqual(provenance["search_space"]["bounds"], {"bend_stiffness": [1.0, 1.0e4]})
            self.assertEqual(provenance["simulation"], SIMULATION)
            self.assertEqual(provenance["optimizer"], optimizer.to_dict())
            self.assertEqual(provenance["seed"], 3)
            self.assertEqual(result.artifacts, {"result": "result.json", "history": "history.json"})
            for relative in result.artifacts.values():
                self.assertTrue((out / relative).exists())
            history = json.loads((out / "history.json").read_text())
            self.assertEqual(len(history), result.metrics["iterations"] + 1)
            self.assertEqual(history[0]["loss"], 0.25)
            self.assertEqual(history[-1]["loss"], result.metrics["loss"])
            self.assertIsNone(problem._evaluator)

            stopped = SearchOutcome([2.0], 0.0, 0, True, [(0.0, [2.0])], "interrupted", 3, None)
            interrupted = problem.build_result(stopped, optimizer=optimizer, artifacts={})
            self.assertEqual(interrupted.status, STATUS_INTERRUPTED)
            self.assertEqual(interrupted.diagnostics["stop_reason"], "interrupted")
            self.assertEqual(interrupted.diagnostics["invalid_candidates"], 3)

    @unittest.skipUnless(importlib.util.find_spec("cma"), "requires newton[calibration]")
    def test_failed_trace_keeps_the_result(self):
        """Verify the result and the history are on disk when rendering a trace fails."""

        class Evaluator:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate(self, candidates, *, return_poses=False):
                return [(np.log10(c.bend_stiffness) - 1.5) ** 2 for c in candidates]

            def record(self, candidate):
                raise RuntimeError("render failed")

        with tempfile.TemporaryDirectory() as tmp, patch("newton._src.calibration.problem.CableEvaluator", Evaluator):
            directory = Path(tmp)
            problem = CableCalibrationProblem.from_bundle(
                write_bundle(directory), directory, make_setup(), make_space(), **SIMULATION
            )
            out = directory / "result"
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                calibrate(problem, optimizer=CMASettings(seed=3, popsize=4, maxiter=2), output_dir=out, trace_every=1)
            for name in ("result.json", "history.json"):
                self.assertTrue((out / name).exists(), name)

    @unittest.skipUnless(importlib.util.find_spec("cma"), "requires newton[calibration]")
    def test_failed_search_keeps_the_completed_iterations(self):
        """Verify an error during the search writes a failed result with the completed iterations, then raises.

        The evaluator fails on its third call: the start and iteration 1 complete. An error
        before the start is scored writes nothing.
        """

        class Evaluator:
            fail_on_call = 3

            def __init__(self, *args, **kwargs):
                self.calls = 0

            def evaluate(self, candidates, *, return_poses=False):
                self.calls += 1
                if self.calls == self.fail_on_call:
                    raise RuntimeError("simulation failed")
                return [(np.log10(c.bend_stiffness) - 1.5) ** 2 for c in candidates]

        with tempfile.TemporaryDirectory() as tmp, patch("newton._src.calibration.problem.CableEvaluator", Evaluator):
            directory = Path(tmp)
            bundle = write_bundle(directory)
            out = directory / "result"
            problem = CableCalibrationProblem.from_bundle(bundle, directory, make_setup(), make_space(), **SIMULATION)
            with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                calibrate(problem, optimizer=CMASettings(seed=3, popsize=4, maxiter=5), output_dir=out, trace_every=1)
            result = CableCalibrationResult.from_dict(json.loads((out / "result.json").read_text()))
            history = json.loads((out / "history.json").read_text())
            self.assertEqual(result.status, STATUS_FAILED)
            self.assertEqual(result.diagnostics["error"], "RuntimeError: simulation failed")
            self.assertEqual(result.metrics["iterations"], 1)
            self.assertEqual(len(history), 2)
            self.assertEqual(result.metrics["loss"], history[-1]["loss"])
            self.assertNotIn("trace", result.artifacts)
            self.assertFalse((out / "trace").exists())

            Evaluator.fail_on_call = 1
            problem = CableCalibrationProblem.from_bundle(bundle, directory, make_setup(), make_space(), **SIMULATION)
            with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                calibrate(problem, optimizer=CMASettings(seed=3, popsize=4, maxiter=5), output_dir=directory / "early")
            self.assertFalse((directory / "early").exists())

    def test_evaluator_receives_the_setup_and_simulation_settings(self):
        """Verify the setup reaches the goals and the evaluator, and invalid settings are refused."""
        with tempfile.TemporaryDirectory() as tmp, patch("newton._src.calibration.problem.CableEvaluator") as evaluator:
            directory = Path(tmp)
            bundle = write_bundle(directory)
            setup = make_setup(
                attachment_transform=[[0.1, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                clamp_position=0.05,
                cable_axis=[1.0, 0, 0],
            )
            for settings in (
                {"settle_mode": "newton"},
                {"settle_frames": -1},
                {"sim_iterations": 0},
                {"search_space": make_space(make_start(angles=[(0.0, 0.0)] * 2))},
            ):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    values = {"search_space": make_space(), **SIMULATION} | settings
                    CableCalibrationProblem.from_bundle(bundle, directory, setup, **values)

            problem = CableCalibrationProblem.from_bundle(
                bundle, directory, setup, make_space(), settle_mode="dynamic", settle_frames=12, sim_iterations=37
            )
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
