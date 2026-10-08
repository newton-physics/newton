# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise integrated runner identification with optional CUDA execution."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.tests.test_impedance_hogan import _chain, _reference, _tiny_shoe
from projects.impedance_instron.hogan import generate, gpu_objective, gpu_runner, identify, quick_fit
from projects.impedance_instron.hogan.runner import RolloutConfig, Runner, State, Task, simulate


def _initial():
    return State(
        np.array([0.01, 0.8 * math.cos(0.12) + 0.101, math.pi / 2, 0.12, -0.24, 0.12]),
        np.array([0.12, -0.5, 0.08, 0.3, -0.25, 0.1]),
        phase_rad=2 * math.pi - 0.03,
        normal_load_bw=0.25,
        torque_nm=np.array([0.5, -0.75, 0.25]),
    )


class _WorkflowFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.shoe = _tiny_shoe(directory.name)
        self.chain = _chain()
        self.config = RolloutConfig(dt_s=1e-4, contact_threshold_n=0.01)

    def trial(self, name="train", *, split="train", duration=0.008):
        initial = _initial()
        time = duration * np.array([0.0, 0.127, 0.43, 0.79, 1.0])
        force_time = duration * np.array([-0.1, 0.11, 0.39, 0.64, 0.96, 1.2])
        return identify.Trial(
            name,
            split,
            self.chain,
            self.shoe,
            Task(0.3),
            initial,
            time,
            initial.q + time[:, None] * initial.v,
            force_time,
            np.array([[2, 0], [-4, 0], [8, 40], [-1, 80], [4, 10], [600, 5000]], dtype=float),
            {
                "compatibility": {"passed": True},
                "shoe_artifact": str(self.shoe.artifact_path),
                "shoe_sha256": self.shoe.metadata["sha256"],
                "mount_m": self.shoe.mount_m.tolist(),
                "pitch_rad": self.shoe.static_pitch_rad,
                "friction_model": self.shoe.metadata["friction_model"],
            },
        )

    def synthetic(self):
        baseline = Runner.seed(reference_speed_m_s=0.3)
        search = identify.FitConfig(population=4, generations=2, seed=8, regularization=0.0)
        parameters = identify.Parameterization(baseline, [0.3])
        # A seeded candidate supplies a reproducible nontrivial fitting target.
        offsets = np.random.default_rng(search.seed).normal(0, search.sigma, parameters.size)
        target = parameters.model(offsets)
        trial = self.trial()
        trace, summary = simulate(
            target, self.chain, self.shoe, trial.initial, trial.task, duration_s=trial.duration_s, config=self.config
        )
        self.assertEqual(summary["status"], "completed")
        self.assertGreater(summary["contact_duration_s"], 0)
        trial = identify.Trial(
            trial.id,
            trial.split,
            trial.chain,
            trial.shoe,
            trial.task,
            trial.initial,
            trace["time_s"],
            trace["state"],
            trace["time_s"],
            np.vstack((trace["grf_n"], trace["grf_n"][-1])),
            trial.provenance,
        )
        return baseline, trial, search

    def contact_trial(self, conflict, *, split="train"):
        reference = _reference(hip_z=0.905 if conflict == "model" else 0.901, frames=4, grf_z=100.0, cop_x=0.0)
        reference["pelvis_target_rad"] = np.full(4, math.pi / 2)
        q = identify.coordinates(reference)
        reference["ankle_target_m"] = np.array([self.chain.point(row, 3, np.zeros(2))[0] for row in q])
        if conflict == "measured":
            reference["ankle_target_m"][:, 1] += 0.004
        elif conflict == "model":
            # Measured contact fits, and FK error stays below the separate 10 mm gate.
            reference["ankle_target_m"][:, 1] -= 0.004
        elif conflict == "fk":
            reference["ankle_target_m"][:, 0] += 0.02
            reference["cop_target_m"][:] = 0.02
        elif conflict == "cop":
            reference["cop_target_m"][:] = 0.05
        qc = identify.compatibility(reference, q, self.chain, self.shoe)
        trial = self.trial(f"{split}_{conflict}", split=split)
        trial.provenance["compatibility"] = qc
        return trial

    def assertPredictionParity(self, actual, expected):
        trace, summary = actual
        reference, info = expected
        self.assertEqual(trace.keys(), reference.keys())
        self.assertEqual(summary.keys(), info.keys())
        for name, values in reference.items():
            with self.subTest(field=name):
                if name == "torque_saturated":
                    np.testing.assert_array_equal(trace[name], values)
                else:
                    # The unchanged shoe law uses float32 on both backends.
                    tolerance = 1e-4 if name == "grf_n" else 2e-5
                    np.testing.assert_allclose(trace[name], values, rtol=2e-5, atol=tolerance)
        for name, value in info.items():
            with self.subTest(summary=name):
                if value is None or isinstance(value, (str, bool)):
                    self.assertEqual(summary[name], value)
                else:
                    tolerance = 1e-4 if name == "peak_grf_n" else 2e-5
                    np.testing.assert_allclose(summary[name], value, rtol=2e-5, atol=tolerance)

    def assertTargetIsolation(self, device):
        trial, model = self.trial(), Runner.seed(reference_speed_m_s=0.3)
        before = identify.predict_many([model], [trial], self.config, device=device)[0][0]
        first_loss = identify.score(*before, trial, model)["loss"]
        scenario = identify.scenario(trial, self.config)
        trial.q[:] += 5
        trial.grf_n[:] += 1000
        trial.time_s[1:-1] *= 0.5
        trial.force_time_s[1:-1] *= 0.5
        after = identify.predict(model, trial, self.config, device=device)
        for name in before[0]:
            np.testing.assert_array_equal(after[0][name], before[0][name])
        self.assertEqual(after[1], before[1])
        self.assertEqual(identify.scenario(trial, self.config), scenario)
        self.assertFalse(after[1]["reference_inputs_used"])
        np.testing.assert_array_equal(after[0]["load"][:, :3], 0)
        result = identify.evaluate(model, [trial], self.config, device=device)
        self.assertGreater(result["mean_loss"], first_loss)
        self.assertAlmostEqual(result["mean_loss"], identify.score(*after, trial, model)["loss"], delta=1e-10)


class TestGpuWorkflowCpu(_WorkflowFixture):
    """Keep CLI routing and compatibility checks executable without CUDA hardware."""

    def test_identify_cli_default_cuda_and_explicit_cpu(self):
        """Default every identification command to CUDA and accept an explicit CPU override."""
        for command in ("inspect", "fit", "evaluate"):
            with self.subTest(command=command):
                argv = [command, "--dataset", str(self.root), "--output", str(self.root / "result")]
                self.assertEqual(identify._parser().parse_args(argv).device, "cuda:0")
                self.assertEqual(identify._parser().parse_args([*argv, "--device", "cpu"]).device, "cpu")

    def test_generate_cli_default_cuda_and_explicit_cpu(self):
        """Route default generation to CUDA while running the explicit CPU solver independently."""
        trial, model = self.trial(duration=0.001), Runner.seed()
        scenario_path, model_path = self.root / "scenario.json", self.root / "runner.json"
        scenario_path.write_text(json.dumps(identify.scenario(trial, self.config)), encoding="utf-8")
        model.save(model_path)
        argv = ["--model", str(model_path), "--scenario", str(scenario_path)]
        expected = identify.predict(model, trial, self.config, device="cpu")
        with (
            patch.object(gpu_runner, "GpuBatch") as batch,
            patch.object(generate, "simulate", side_effect=AssertionError("Unexpected CPU fallback")),
        ):
            batch.return_value.evaluate.return_value = [[expected]]
            generate.main([*argv, "--output", str(self.root / "cuda")])
            self.assertEqual(batch.call_args.kwargs["device"], "cuda:0")
            self.assertEqual(batch.call_args.kwargs["candidates"], 1)
            self.assertEqual(batch.call_args.args[4], [trial.duration_s])
            batch.return_value.evaluate.assert_called_once()
        with patch.object(gpu_runner, "GpuBatch", side_effect=AssertionError("CPU requested CUDA")):
            generate.main([*argv, "--output", str(self.root / "cpu"), "--device", "cpu"])
        for device in ("cuda", "cpu"):
            with np.load(self.root / device / "trace.npz") as trace:
                for name, values in expected[0].items():
                    np.testing.assert_array_equal(trace[name], values)
            summary = json.loads((self.root / device / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["device"], "cuda:0" if device == "cuda" else "cpu")

    def test_loaded_clearance_is_reported_not_gated(self):
        """Report measured and model loaded clearance without failing the screen."""
        compatible = self.contact_trial("none").provenance["compatibility"]
        self.assertTrue(compatible["passed"])
        qc = self.contact_trial("model").provenance["compatibility"]
        self.assertTrue(qc["passed"])
        self.assertFalse(qc["clearance_gating"])
        self.assertEqual(qc["loaded_clearance_conflict_frames"], 0)
        self.assertEqual(qc["model_loaded_clearance_conflict_frames"], 4)
        self.assertAlmostEqual(qc["model_loaded_clearance_peak_m"], 0.005, places=12)
        qc = self.contact_trial("measured").provenance["compatibility"]
        self.assertTrue(qc["passed"])
        self.assertGreater(qc["loaded_clearance_conflict_frames"], 0)

    def test_full_compatibility_gate_precedes_gpu_evaluator_creation(self):
        """Block an ankle FK incompatibility in either split before allocating or scoring."""
        for split in ("train", "eval"):
            with self.subTest(split=split):
                incompatible = self.contact_trial("fk", split=split)
                qc = incompatible.provenance["compatibility"]
                self.assertFalse(qc["passed"])
                self.assertGreater(qc["ankle_fk_error_peak_m"], qc["ankle_fk_tolerance_m"])
                with (
                    patch.object(gpu_objective, "GpuEvaluator", side_effect=AssertionError("Allocated GPU")) as gpu,
                    patch.object(identify, "evaluate", side_effect=AssertionError("Scored blocked input")),
                    self.assertRaisesRegex(ValueError, "Input compatibility failed for 1 trials"),
                ):
                    identify.fit(Runner.seed(), [self.contact_trial("none"), incompatible], device="cuda:0")
                gpu.assert_not_called()

    def test_cop_outside_footprint_is_reported_not_gated(self):
        """Report loaded COP outside the sole without failing the screen, since treadmill COP is unreliable."""
        qc = self.contact_trial("cop").provenance["compatibility"]
        self.assertTrue(qc["passed"])
        self.assertFalse(qc["cop_gating"])
        self.assertGreater(qc["loaded_cop_outside_frames"], 0)

    def test_quick_fit_cli_devices_and_full_held_out_gate(self):
        """Preserve device selection and block incompatible held-out data before either search arm."""
        members = [
            {"id": name, "split": "train", "reference": "unused.npz"} for name in ("FR3_1_train_000", "FR3_2_train_000")
        ]
        members.extend({"id": f"eval_{i}", "split": "eval", "reference": "unused.npz"} for i in range(10))
        (self.root / "manifest.json").write_text(
            json.dumps({"members": members, "shared_assets": {}}), encoding="utf-8"
        )
        trials = [self.contact_trial("none", split=member["split"]) for member in members]
        for trial, member in zip(trials, members, strict=True):
            trial.id = member["id"]
        trials[-1].provenance["compatibility"] = self.contact_trial("fk").provenance["compatibility"]
        for device, extra in (("cuda:0", []), ("cpu", ["--device", "cpu"])):
            with self.subTest(device=device):
                output = self.root / device.replace(":", "_")
                with (
                    patch.object(quick_fit, "load_trials", return_value=trials),
                    patch.object(quick_fit, "search", side_effect=AssertionError("Launched search")) as search,
                    patch.object(gpu_objective, "GpuEvaluator", side_effect=AssertionError("Allocated GPU")) as gpu,
                    self.assertRaisesRegex(ValueError, "Contact compatibility failed; no fit launched"),
                ):
                    quick_fit.main(["--dataset", str(self.root), "--output", str(output), *extra])
                search.assert_not_called()
                gpu.assert_not_called()
                protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
                self.assertEqual(protocol["device"], device)
                blocked = json.loads((output / "blocked.json").read_text(encoding="utf-8"))
                self.assertEqual(blocked["trials"], [trials[-1].id])
                self.assertFalse((output / "seed.json").exists())

    def test_cpu_fit_remains_independent_of_cuda(self):
        """Fit a real tiny shoe on CPU without constructing any CUDA execution object."""
        baseline, trial, search = self.synthetic()
        with (
            patch.object(gpu_objective, "GpuEvaluator", side_effect=AssertionError("CPU requested CUDA scoring")),
            patch.object(gpu_runner, "GpuBatch", side_effect=AssertionError("CPU requested CUDA dynamics")),
        ):
            learned, report = identify.fit(baseline, [trial], config=self.config, search=search, device="cpu")
        self.assertNotEqual(learned.to_dict(), baseline.to_dict())
        rows = report["splits"]["train"]
        self.assertLess(rows["learned"]["mean_loss"], rows["baseline"]["mean_loss"])
        self.assertEqual(rows["learned"]["failed"], 0)
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["objective_backend"], "numpy")
        self.assertFalse(report["reference_inputs_used"])
        json.dumps(report, allow_nan=False)

    def test_explicit_compatibility_override_is_not_validation(self):
        """Allow deliberate diagnostic fitting while retaining the incompatibility disclosure."""
        trial = self.contact_trial("fk")
        _, report = identify.fit(
            Runner.seed(),
            [trial],
            config=self.config,
            search=identify.FitConfig(population=4, generations=1),
            allow_incompatible=True,
            device="cpu",
        )
        self.assertEqual(report["incompatible_trials"], [trial.id])
        self.assertTrue(report["allow_incompatible"])
        self.assertFalse(report["validated"])

    def test_cpu_predictions_never_receive_measurement_feedback(self):
        """Change offline targets and clocks without changing any CPU prediction input or trace."""
        self.assertTargetIsolation("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for integrated GPU workflow tests")
class TestGpuWorkflowCuda(_WorkflowFixture):
    """Qualify real CUDA fits, exported generation, and mixed-clock evaluation."""

    def test_seeded_fit_matches_cpu_and_excludes_held_out_selection(self):
        """Match seeded CPU selection using resident CUDA scoring and frozen held-out evaluation."""
        baseline, training, search = self.synthetic()
        held_out = self.trial("held_out", split="eval", duration=0.00305)
        trials = [training, held_out]
        cpu_model, cpu_report = identify.fit(baseline, trials, config=self.config, search=search, device="cpu")
        constructor, evaluate = gpu_objective.GpuEvaluator, gpu_objective.GpuEvaluator.evaluate
        with (
            patch.object(gpu_objective, "GpuEvaluator", wraps=constructor) as created,
            patch.object(constructor, "evaluate", autospec=True, side_effect=evaluate) as scored,
            patch.object(identify, "score", wraps=identify.score) as diagnostic_score,
            patch.object(identify, "simulate", side_effect=AssertionError("CUDA fit fell back to CPU")),
        ):
            gpu_model, report = identify.fit(baseline, trials, config=self.config, search=search, device="cuda:0")
        created.assert_called_once()
        self.assertEqual(len(created.call_args.args[0]), 1)
        self.assertIs(created.call_args.args[0][0], training)
        self.assertEqual(created.call_args.kwargs["device"], "cuda:0")
        self.assertEqual(created.call_args.kwargs["candidates"], search.population)
        self.assertEqual(scored.call_count, search.generations + 2)
        for call in scored.call_args_list:
            self.assertEqual(len(call.args[1]), search.population)
        for call in (scored.call_args_list[0], scored.call_args_list[-1]):
            self.assertTrue(all(model is call.args[1][0] for model in call.args[1]))
        self.assertEqual(
            diagnostic_score.call_count, 4, "Only final baseline/learned split diagnostics use host scores"
        )
        self.assertEqual(gpu_model.to_dict(), cpu_model.to_dict())
        self.assertNotEqual(gpu_model.to_dict(), baseline.to_dict())
        self.assertEqual(report["objective_backend"], "cuda")
        self.assertEqual(report["device"], "cuda:0")
        self.assertEqual(report["selection_split"], "train")
        self.assertFalse(report["validated"])
        self.assertFalse(report["reference_inputs_used"])
        for actual, expected in zip(report["history"], cpu_report["history"], strict=True):
            self.assertEqual(actual["best_failed"], expected["best_failed"])
            self.assertAlmostEqual(actual["best_score"], expected["best_score"], delta=1e-8)
            self.assertAlmostEqual(actual["sigma_mean"], expected["sigma_mean"], delta=1e-12)
        for split in ("train", "eval"):
            for label in ("baseline", "learned"):
                actual, expected = report["splits"][split][label], cpu_report["splits"][split][label]
                self.assertEqual(actual["failed"], expected["failed"])
                # Use the same float32-contact score tolerance as the lower-level suites.
                self.assertAlmostEqual(actual["mean_loss"], expected["mean_loss"], delta=1e-4)
        self.assertLess(
            report["splits"]["train"]["learned"]["mean_loss"], report["splits"]["train"]["baseline"]["mean_loss"]
        )
        held_out.q[:] += 10
        held_out.grf_n[:] += 1000
        repeated, changed = identify.fit(baseline, trials, config=self.config, search=search, device="cuda:0")
        self.assertEqual(repeated.to_dict(), gpu_model.to_dict())
        self.assertEqual(changed["history"], report["history"])
        self.assertGreater(
            changed["splits"]["eval"]["learned"]["mean_loss"], report["splits"]["eval"]["learned"]["mean_loss"]
        )
        json.dumps(report, allow_nan=False)

    def test_mixed_dt_predict_many_evaluate_and_resident_score_agree(self):
        """Preserve trial order and CPU scoring across the four-trial CUDA batch boundary."""
        durations = [0.00305, 0.0031, 0.0062, 0.00305, 0.00413]
        trials = [self.trial(str(i), duration=duration) for i, duration in enumerate(durations)]
        for i, trial in enumerate(trials):
            trial.initial.phase_rad += 0.2 * i
            trial.task = Task(0.3 + 0.1 * i)
        trials[-1].initial.q[1] = 0.1
        model = Runner.seed(reference_speed_m_s=0.3)
        parameters = identify.Parameterization(model, [0.3])
        models = [model, parameters.model(np.full(parameters.size, 0.1))]
        expected = identify.predict_many(models, trials, self.config, device="cpu")
        with patch.object(identify, "simulate", side_effect=AssertionError("CUDA prediction fell back to CPU")):
            actual = identify.predict_many(models, trials, self.config, device="cuda:0")
            evaluations = [identify.evaluate(model, trials, self.config, device="cuda:0") for model in models]
        resident = gpu_objective.GpuEvaluator(trials, candidates=2, config=self.config).evaluate(models)
        self.assertEqual([len(row) for row in actual], [5, 5])
        for candidate, model in enumerate(models):
            expected_scores = []
            for i, trial in enumerate(trials):
                with self.subTest(candidate=candidate, trial=trial.id):
                    prediction = actual[candidate][i]
                    self.assertPredictionParity(prediction, expected[candidate][i])
                    dt = trial.duration_s / math.ceil(trial.duration_s / self.config.dt_s)
                    self.assertEqual(prediction[1]["dt_s"], dt)
                    np.testing.assert_array_equal(prediction[0]["time_s"], np.arange(len(prediction[0]["state"])) * dt)
                    cpu_score = identify.score(*expected[candidate][i], trial, model)
                    gpu_score = identify.score(*prediction, trial, model)
                    expected_scores.append(cpu_score["loss"])
                    self.assertAlmostEqual(gpu_score["loss"], cpu_score["loss"], delta=1e-4)
                    self.assertEqual(evaluations[candidate]["trials"][i], gpu_score)
            self.assertNotEqual(actual[candidate][0][1]["dt_s"], actual[candidate][1][1]["dt_s"])
            self.assertEqual(evaluations[candidate]["failed"], 1)
            self.assertEqual(resident[candidate]["failed"], 1)
            self.assertAlmostEqual(evaluations[candidate]["mean_loss"], np.mean(expected_scores), delta=1e-4)
            self.assertAlmostEqual(resident[candidate]["mean_loss"], evaluations[candidate]["mean_loss"], delta=1e-9)

    def test_cuda_predictions_never_receive_measurement_feedback(self):
        """Change future measurements and clocks without changing CUDA dynamics or exported inputs."""
        with patch.object(identify, "simulate", side_effect=AssertionError("CUDA prediction fell back to CPU")):
            self.assertTargetIsolation("cuda:0")

    def test_generate_default_cuda_replays_target_free_scenario(self):
        """Replay an exported scenario with real CUDA after removing the measurement dataset."""
        trial, model = self.trial(duration=0.00305), Runner.seed()
        expected = identify.predict(model, trial, self.config, device="cuda:0")
        model_path, scenario_path = self.root / "runner.json", self.root / "scenario.json"
        model.save(model_path)
        scenario_path.write_text(json.dumps(identify.scenario(trial, self.config)), encoding="utf-8")
        measurements = self.root / "reference.npz"
        np.savez(measurements, q=trial.q, grf_n=trial.grf_n)
        measurements.unlink()
        output = self.root / "generated"
        with (
            patch.object(identify, "load_trials", side_effect=AssertionError("Read measurements")),
            patch.object(identify, "score", side_effect=AssertionError("Read fitting targets")),
            patch.object(generate, "simulate", side_effect=AssertionError("CUDA generation fell back to CPU")),
        ):
            generate.main(["--model", str(model_path), "--scenario", str(scenario_path), "--output", str(output)])
        with np.load(output / "trace.npz") as trace:
            for name, values in expected[0].items():
                np.testing.assert_array_equal(trace[name], values)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["device"], "cuda:0")
        self.assertFalse(summary["reference_inputs_used"])
        self.assertFalse(summary["validated"])
        self.assertEqual(summary["scenario"], identify.scenario(trial, self.config))

    def test_quick_search_cuda_matches_cpu_for_both_arms(self):
        """Match both real search arms with common seeded draws and final-mean GPU padding."""
        baseline, trial, settings = self.synthetic()
        data = baseline.to_dict()
        weights = np.asarray(data["weights"])
        weights[1:, :, 1:] = 0
        data["weights"] = weights.tolist()
        baseline = Runner.from_dict(data)
        protocol = {
            "search_dt_s": self.config.dt_s,
            "population": settings.population,
            "generations": settings.generations,
            "seed": settings.seed,
            "sigma": settings.sigma,
            "bound": settings.bound,
            "regularization": settings.regularization,
        }
        for variable in (False, True):
            results = []
            for device in ("cpu", "cuda:0"):
                with self.subTest(variable=variable, device=device):
                    destination = self.root / f"{variable}_{device.replace(':', '_')}"
                    destination.mkdir()
                    result = quick_fit.search(
                        baseline,
                        [trial],
                        variable=variable,
                        output=destination,
                        protocol={**protocol, "device": device},
                    )
                    results.append(result.to_dict())
                    report = json.loads((destination / "search.json").read_text(encoding="utf-8"))
                    self.assertEqual(report["candidate_evaluations"], settings.population * settings.generations + 1)
                    self.assertEqual(report["stance_rollouts"], report["candidate_evaluations"])
                    if not variable:
                        np.testing.assert_array_equal(result.weights[1:, :, 1:], 0)
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
