# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify cache defaults and lightweight configuration imports."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from warp._src.thirdparty.appdirs import user_cache_dir


class TestConfig(unittest.TestCase):
    def test_cache_defaults_share_newton_directory(self):
        """Initialize both caches from the existing environment or platform default."""
        with tempfile.TemporaryDirectory() as temporary:
            custom = Path(temporary) / "cache with spaces"
            for root in (None, str(custom)):
                with self.subTest(root=root):
                    env = os.environ.copy()
                    env.pop("NEWTON_CACHE_PATH", None)
                    if root is not None:
                        env["NEWTON_CACHE_PATH"] = root
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            "import json, sys; import newton; "
                            "assert not {'coacd', 'trimesh', 'git'} & sys.modules.keys(); "
                            "print(json.dumps([newton.config.cache_dir_assets, newton.config.cache_dir_meshes]))",
                        ],
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    expected = root if root is not None else user_cache_dir("newton", "newton-physics")
                    self.assertEqual(json.loads(result.stdout), [expected, os.path.join(expected, "meshes")])
            self.assertFalse(custom.exists())


if __name__ == "__main__":
    unittest.main()
