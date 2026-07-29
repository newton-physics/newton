"""Test Isaac Sim development source validation."""

import tempfile
import tomllib
import unittest
from pathlib import Path

from verify_source import validate_source


class SourceValidationTest(unittest.TestCase):
    """Test source path acceptance and rejection."""

    def test_demo_contains_runtime_guards_and_lighting(self):
        """Require runtime source checks, Newton activation checks, and lighting."""
        source = (Path(__file__).parent / "demo_rigid_bodies.py").read_text()
        required_fragments = (
            "validate_source(newton.__file__",
            "SimulationManager.get_active_physics_engine()",
            "wp.get_device()",
            "UsdLux.DomeLight.Define",
            "UsdLux.DistantLight.Define",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)
        self.assertNotIn("from verify_source import", source)

    def test_launcher_enables_source_extension(self):
        """Require the launcher to enable the repository source extension."""
        source = (Path(__file__).parent / "launch.sh").read_text()

        self.assertIn('--ext-folder "${script_dir}/exts"', source)
        self.assertIn("--enable newton.dev.source", source)

    def test_source_extension_precedes_newton_prebundle(self):
        """Load the repository source extension before Isaac Sim's prebundle."""
        config_path = Path(__file__).parent / "exts/newton.dev.source/config/extension.toml"
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)

        self.assertLess(config["core"]["order"], -1000)
        self.assertEqual(config["python"]["module"][0]["path"], "../../../..")

    def test_accept_source_inside_expected_repository(self):
        """Accept a Newton module contained by the expected repository."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "newton-repo"
            module = repo / "newton" / "__init__.py"
            module.parent.mkdir(parents=True)
            module.touch()

            self.assertEqual(validate_source(str(module), str(repo)), repo.resolve())

    def test_reject_bundled_source_outside_expected_repository(self):
        """Reject a Newton module outside the expected repository."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "newton-repo"
            bundled = root / "isaac-sim" / "pip_prebundle" / "newton" / "__init__.py"
            repo.mkdir()
            bundled.parent.mkdir(parents=True)
            bundled.touch()

            with self.assertRaisesRegex(RuntimeError, "outside expected repository"):
                validate_source(str(bundled), str(repo))


if __name__ == "__main__":
    unittest.main()
