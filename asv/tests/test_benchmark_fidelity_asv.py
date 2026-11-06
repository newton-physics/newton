# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check ASV discovery, cache isolation, and scalar measurement coverage."""

import importlib.util
import inspect
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from asv.config import Config
from asv.environment import ExistingEnvironment
from asv.runner import Spawner
from asv_runner.benchmarks._base import get_setup_cache_key
from asv_runner.discovery import get_benchmark_from_name

import asv

ASV = Path(__file__).parents[1]
sys.path.insert(0, str(ASV))
from collision_fidelity.catalog import load_cases  # noqa: E402
from collision_fidelity.version import _case_version, semantic_version  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "bench_collision_fidelity", ASV / "benchmarks/simulation/bench_collision_fidelity.py"
)
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


class TestFidelityAsv(unittest.TestCase):
    """Exercise the actual ASV benchmark objects without running simulations."""

    def test_each_case_has_distinct_cache_key(self):
        """Give all 45 cases distinct actual ASV source-line cache keys."""
        classes = [cls for name, cls in vars(BENCH).items() if name.startswith("CollisionFidelity")]
        self.assertEqual(len(classes), 45)
        self.assertEqual({cls.case_id for cls in classes}, set(load_cases()))
        keys = [get_setup_cache_key(cls.setup_cache) for cls in classes]
        self.assertEqual(len(set(keys)), len(classes))
        for cls in classes:
            source = inspect.getsource(cls.setup_cache)
            self.assertIn(f'"{cls.case_id}"', source)
            self.assertNotIn("for ", source)

    def test_six_tracks_per_case_use_actual_asv_discovery(self):
        """Keep all 45 isolated caches and expose 270 lower-is-better tracks."""
        expected = {
            "track_jitter",
            "track_rms_speed",
            "track_rigid_soft_penetration_fixed_supports",
            "track_missing_stationary_windows",
            "track_incomplete_run",
            "track_overall_failure",
        }
        names = set()
        for name, cls in vars(BENCH).items():
            if not name.startswith("CollisionFidelity"):
                continue
            tracks = {key for key, value in inspect.getmembers(cls) if key.startswith("track_") and callable(value)}
            self.assertEqual(tracks, expected)
            for track in tracks:
                benchmark = get_benchmark_from_name(
                    str(ASV / "benchmarks"), f"simulation.bench_collision_fidelity.{name}.{track}"
                )
                names.add(benchmark.name)
        self.assertEqual(len(names), 270)

    def test_case_version_tracks_inputs_not_solver_revision(self):
        """Change case identity when inputs change without reading the Git revision."""
        case = next(iter(load_cases().values()))
        modified = dict(case, iterations=case["iterations"] + 1)
        self.assertNotEqual(_case_version(case, "source"), _case_version(modified, "source"))
        self.assertEqual(len(semantic_version(case["case_id"])), 64)
        self.assertNotIn("git", inspect.getsource(semantic_version))

    def test_discovery_does_not_import_physics(self):
        """Import the real wrapper without loading Newton, Warp, or scene builders."""
        code = (
            "import runpy,sys; runpy.run_path(sys.argv[1]); "
            "assert 'newton' not in sys.modules; assert 'warp' not in sys.modules; "
            "assert 'collision_fidelity.runner' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code, str(Path(BENCH.__file__))], check=True)

    def test_default_cache_skips_before_backend_import(self):
        """Require the exact opt-in value before importing Warp or the runner."""
        code = (
            "import runpy,sys; module=runpy.run_path(sys.argv[1]); "
            "result=module['CollisionFidelityS3Cloth10substeps']().setup_cache(); "
            "assert result['execution']['status']=='disabled', result; "
            "assert 'NEWTON_COLLISION_FIDELITY_FULL=1' in result['execution']['reason']; "
            "assert 'warp' not in sys.modules; assert 'newton' not in sys.modules; "
            "assert 'collision_fidelity.runner' not in sys.modules"
        )
        for value in (None, "0", "true"):
            with self.subTest(value=value):
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
                env.pop("NEWTON_COLLISION_FIDELITY_FULL", None)
                if value is not None:
                    env["NEWTON_COLLISION_FIDELITY_FULL"] = value
                completed = subprocess.run(
                    [sys.executable, "-c", code, str(Path(BENCH.__file__))],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_explicit_opt_in_runs_exactly_one_full_case(self):
        """Forward the selected full workload after the gate and CUDA check."""
        result = {"execution": {"status": "completed"}}
        run = Mock(return_value=result)
        modules = {
            "warp": SimpleNamespace(is_cuda_available=lambda: True),
            "collision_fidelity.runner": SimpleNamespace(run_case=run),
        }
        with patch.dict(os.environ, {"NEWTON_COLLISION_FIDELITY_FULL": "1"}), patch.dict(sys.modules, modules):
            self.assertIs(BENCH.CollisionFidelityS3Cloth10substeps().setup_cache(), result)
        run.assert_called_once_with("s3_cloth_10substeps", profile="full", device="cuda:0")

    def test_asv_spawner_preserves_opt_in_environment(self):
        """Pass shell opt-in through the installed ASV setup-cache subprocess."""
        with tempfile.TemporaryDirectory() as temporary:
            config = Config()
            config.env_dir = temporary
            environment = ExistingEnvironment(config, sys.executable, {}, {})
            spawner = Spawner(environment, ASV / "benchmarks")
            name = "simulation.bench_collision_fidelity.CollisionFidelityR2FrankaDuck.track_overall_failure"
            with patch.dict(os.environ, {"NEWTON_COLLISION_FIDELITY_FULL": "1"}):
                cache, error = spawner.create_setup_cache(name, 30, "{}")
            self.assertIsNone(error, error)
            try:
                result = pickle.loads((Path(cache) / "cache.pickle").read_bytes())
                self.assertEqual(result["execution"]["status"], "unsupported")
                self.assertEqual(result["execution"]["reason"], load_cases()["r2_franka_duck"]["support"]["reason"])
            finally:
                shutil.rmtree(cache)

    def test_actual_asv_cache_and_skip_from_temporary_cwd(self):
        """Skip an unrequested supported full workload through the real ASV entry."""
        entry = str(Path(asv.__file__).with_name("benchmark.py"))
        benchmark = "simulation.bench_collision_fidelity.CollisionFidelityS3Cloth10substeps.track_overall_failure"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        env.pop("NEWTON_COLLISION_FIDELITY_FULL", None)
        with tempfile.TemporaryDirectory() as temporary:
            cache = subprocess.run(
                [sys.executable, entry, "setup_cache", str(ASV / "benchmarks"), benchmark, "{}"],
                cwd=temporary,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(cache.returncode, 0, cache.stdout + cache.stderr)
            self.assertTrue((Path(temporary) / "cache.pickle").is_file())
            cached = pickle.loads((Path(temporary) / "cache.pickle").read_bytes())
            self.assertEqual(cached["execution"]["status"], "disabled")
            output = Path(temporary) / "result.json"
            run = subprocess.run(
                [sys.executable, entry, "run", str(ASV / "benchmarks"), benchmark, "{}", "None", str(output)],
                cwd=temporary,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertTrue(math.isnan(json.loads(output.read_text())))
            self.assertNotIn("Warp", cache.stdout + run.stdout)
            self.assertIn("NEWTON_COLLISION_FIDELITY_FULL=1", run.stdout + run.stderr)

    def test_scalar_units_and_coverage(self):
        """Convert meters once and reject partial-window or partial-trajectory values."""
        tracks = BENCH._FidelityTracks()
        result = {
            "execution": {"full_timeline_completed": True},
            "measurements": {
                "finite": {"all_samples_finite": True},
                "stability": {
                    "jitter_m": 0.002,
                    "rms_speed_m_s": 0.1,
                    "complete_window_count": 120,
                    "declared_complete_window_count": 120,
                },
                "penetration": {"rigid_soft_penetration_m": 0.0003},
            },
        }
        self.assertEqual(tracks.track_jitter(result), 2)
        self.assertEqual(tracks.track_rms_speed(result), 0.1)
        self.assertAlmostEqual(tracks.track_rigid_soft_penetration_fixed_supports(result), 0.3)
        result["measurements"]["stability"]["complete_window_count"] = 119
        self.assertTrue(math.isnan(tracks.track_jitter(result)))
        result["execution"]["full_timeline_completed"] = False
        self.assertTrue(math.isnan(tracks.track_rigid_soft_penetration_fixed_supports(result)))

    def test_missing_and_nonfinite_do_not_become_zero(self):
        """Keep absent values and invalid full trajectories out of scalar results."""
        tracks = BENCH._FidelityTracks()
        self.assertTrue(math.isnan(tracks.track_jitter({})))
        self.assertTrue(math.isnan(tracks.track_overall_failure({})))
        result = {
            "criteria": {"overall": {"status": "fail"}},
            "execution": {"full_timeline_completed": True},
            "measurements": {"finite": {"all_samples_finite": False}, "penetration": {"rigid_soft_penetration_m": 0.1}},
        }
        self.assertEqual(tracks.track_overall_failure(result), 1)
        self.assertEqual(tracks.track_incomplete_run(result), 1)
        self.assertTrue(math.isnan(tracks.track_rigid_soft_penetration_fixed_supports(result)))

    def test_missing_windows_count_and_invalid_evidence(self):
        """Measure missing windows, including prefixes, without inventing counts."""
        tracks = BENCH._FidelityTracks()
        for declared, complete, expected in ((120, 120, 0), (120, 119, 1), (120, 0, 120), (0, 0, 0)):
            result = {
                "measurements": {
                    "stability": {"declared_complete_window_count": declared, "complete_window_count": complete}
                }
            }
            self.assertEqual(tracks.track_missing_stationary_windows(result), expected)
        for declared, complete in ((None, 0), (120, None), (-1, 0), (120, -1), (120, 121), (120.5, 120), (True, 0)):
            with self.subTest(declared=declared, complete=complete):
                result = {
                    "measurements": {
                        "stability": {"declared_complete_window_count": declared, "complete_window_count": complete}
                    }
                }
                self.assertTrue(math.isnan(tracks.track_missing_stationary_windows(result)))

    def test_incomplete_run_requires_explicit_evidence(self):
        """Encode full finite execution as zero and known incompletion as one."""
        tracks = BENCH._FidelityTracks()
        for full, finite, expected in ((True, True, 0), (False, True, 1), (True, False, 1), (False, False, 1)):
            result = {
                "execution": {"full_timeline_completed": full},
                "measurements": {"finite": {"all_samples_finite": finite}},
            }
            self.assertEqual(tracks.track_incomplete_run(result), expected)
        self.assertEqual(tracks.track_incomplete_run({"execution": {"status": "nonfinite"}}), 1)
        for result in (
            {},
            {"execution": {"full_timeline_completed": True}},
            {"measurements": {"finite": {"all_samples_finite": True}}},
        ):
            self.assertTrue(math.isnan(tracks.track_incomplete_run(result)))

    def test_overall_failure_does_not_encode_incomplete_as_success(self):
        """Use zero for explicit Pass, one for Fail, and NaN for all other states."""
        tracks = BENCH._FidelityTracks()
        for status, expected in (("pass", 0), ("fail", 1)):
            self.assertEqual(tracks.track_overall_failure({"criteria": {"overall": {"status": status}}}), expected)
        for status in (None, "incomplete", "unsupported", "not_applicable"):
            self.assertTrue(math.isnan(tracks.track_overall_failure({"criteria": {"overall": {"status": status}}})))
        self.assertEqual(tracks.track_overall_failure.unit, "failure=1")

    def test_actual_asv_reports_execution_errors_and_retains_nonfinite_tracks(self):
        """Expose execution errors through ASV without skipping numerical outcomes."""
        entry = str(Path(asv.__file__).with_name("benchmark.py"))
        benchmark = "simulation.bench_collision_fidelity.CollisionFidelityM3FrankaToteEmpty.track_overall_failure"
        for status in ("error", "invalid", "nonfinite"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                result = {
                    "execution": {"status": status, "reason": "recorded failure", "traceback": "recorded traceback"},
                    "criteria": {"overall": {"status": "fail"}},
                }
                directory = Path(temporary)
                (directory / "cache.pickle").write_bytes(pickle.dumps(result))
                output = directory / "result.json"
                run = subprocess.run(
                    [sys.executable, entry, "run", str(ASV / "benchmarks"), benchmark, "{}", "None", str(output)],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if status == "nonfinite":
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertEqual(json.loads(output.read_text()), 1.0)
                else:
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn(status, run.stderr)
                    self.assertIn("recorded failure", run.stderr)
                    self.assertIn("recorded traceback", run.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
