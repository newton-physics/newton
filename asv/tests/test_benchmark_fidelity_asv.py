# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check ASV discovery, cache isolation, and scalar measurement coverage."""

import importlib.util
import inspect
import json
import math
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from asv_runner.benchmarks._base import get_setup_cache_key

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

    def test_actual_asv_cache_and_skip_from_temporary_cwd(self):
        """Resolve the sibling package and skip unsupported execution through real ASV."""
        entry = str(Path(asv.__file__).with_name("benchmark.py"))
        benchmark = "simulation.bench_collision_fidelity.CollisionFidelityR2FrankaDuck.track_overall"
        with tempfile.TemporaryDirectory() as temporary:
            cache = subprocess.run(
                [sys.executable, entry, "setup_cache", str(ASV / "benchmarks"), benchmark, "{}"],
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(cache.returncode, 0, cache.stdout + cache.stderr)
            self.assertTrue((Path(temporary) / "cache.pickle").is_file())
            output = Path(temporary) / "result.json"
            run = subprocess.run(
                [sys.executable, entry, "run", str(ASV / "benchmarks"), benchmark, "{}", "None", str(output)],
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertTrue(math.isnan(json.loads(output.read_text())))
            self.assertNotIn("Warp", cache.stdout + run.stdout)

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
        self.assertTrue(math.isnan(tracks.track_overall({})))
        result = {
            "criteria": {"overall": {"status": "fail"}},
            "execution": {"full_timeline_completed": True},
            "measurements": {"finite": {"all_samples_finite": False}, "penetration": {"rigid_soft_penetration_m": 0.1}},
        }
        self.assertEqual(tracks.track_overall(result), 0)
        self.assertEqual(tracks.track_finite_completion(result), 0)
        self.assertTrue(math.isnan(tracks.track_rigid_soft_penetration_fixed_supports(result)))

    def test_actual_asv_reports_execution_errors_and_retains_nonfinite_tracks(self):
        """Expose execution errors through ASV without skipping numerical outcomes."""
        entry = str(Path(asv.__file__).with_name("benchmark.py"))
        benchmark = "simulation.bench_collision_fidelity.CollisionFidelityM3FrankaToteEmpty.track_overall"
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
                    self.assertEqual(json.loads(output.read_text()), 0.0)
                else:
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn(status, run.stderr)
                    self.assertIn("recorded failure", run.stderr)
                    self.assertIn("recorded traceback", run.stderr)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
