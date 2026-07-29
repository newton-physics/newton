"""Test Isaac Sim development source validation."""

import tempfile
import unittest
from pathlib import Path

from verify_source import validate_source


class SourceValidationTest(unittest.TestCase):
    """Test source path acceptance and rejection."""

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
