# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--warp-config KEY=VALUE`` CLI option."""

import contextlib
import io
import sys
import unittest
from unittest import mock

import warp as wp

from newton.examples import _apply_warp_config, create_parser, init


class TestWarpConfigCLI(unittest.TestCase):
    """Tests for :func:`_apply_warp_config`."""

    def setUp(self):
        deprecated_log_attrs = {"quiet", "verbose"}
        self._saved_config = {
            attr: getattr(wp.config, attr)
            for attr in dir(wp.config)
            if not attr.startswith("__") and attr not in deprecated_log_attrs
        }

    def tearDown(self):
        for attr, value in self._saved_config.items():
            setattr(wp.config, attr, value)

    def _parse(self, *cli_args):
        """Parse *cli_args* through :func:`create_parser` and return (parser, args)."""
        parser = create_parser()
        args = parser.parse_known_args(list(cli_args))[0]
        return parser, args

    def test_no_overrides(self):
        """No --warp-config flags should be a no-op."""
        parser, args = self._parse()
        _apply_warp_config(parser, args)
        self.assertEqual(wp.config.log_level, self._saved_config["log_level"])

    def test_int_override(self):
        """Integer values should be parsed via literal_eval."""
        parser, args = self._parse("--warp-config", "max_unroll=8")
        _apply_warp_config(parser, args)
        self.assertEqual(wp.config.max_unroll, 8)

    def test_string_fallback(self):
        """Bare words that aren't Python literals should be kept as strings."""
        parser, args = self._parse("--warp-config", "mode=release")
        _apply_warp_config(parser, args)
        self.assertEqual(wp.config.mode, "release")

    def test_bool_override(self):
        """Boolean values should be parsed correctly."""
        parser, args = self._parse("--warp-config", "verify_fp=True")
        _apply_warp_config(parser, args)
        self.assertIs(wp.config.verify_fp, True)

    def test_deprecated_log_config_keys_error(self):
        """Deprecated log config keys should point users to log_level."""
        for key in ("quiet", "verbose"):
            with self.subTest(key=key):
                parser, args = self._parse("--warp-config", f"{key}=True")
                stderr = io.StringIO()
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
                    _apply_warp_config(parser, args)
                self.assertIn(f"invalid --warp-config key '{key}': use 'log_level' instead", stderr.getvalue())

    def test_none_override(self):
        """None values should be accepted."""
        parser, args = self._parse("--warp-config", "cache_kernels=None")
        _apply_warp_config(parser, args)
        self.assertIsNone(wp.config.cache_kernels)

    def test_empty_string_override(self):
        """Empty value (KEY=) should produce an empty string."""
        parser, args = self._parse("--warp-config", "mode=")
        _apply_warp_config(parser, args)
        self.assertEqual(wp.config.mode, "")

    def test_repeated_overrides(self):
        """Later overrides should win."""
        parser, args = self._parse(
            "--warp-config",
            "max_unroll=4",
            "--warp-config",
            "max_unroll=16",
        )
        _apply_warp_config(parser, args)
        self.assertEqual(wp.config.max_unroll, 16)

    def test_unknown_key_errors(self):
        """An unknown key should produce a clear error naming the bad key."""
        parser, args = self._parse("--warp-config", "bogus_key_xyz=1")
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
            _apply_warp_config(parser, args)
        self.assertIn(
            "invalid --warp-config key 'bogus_key_xyz': not a recognized warp.config setting", stderr.getvalue()
        )

    def test_missing_equals_errors(self):
        """A missing '=' should produce a clear error showing the bad entry."""
        parser, args = self._parse("--warp-config", "no_equals")
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
            _apply_warp_config(parser, args)
        self.assertIn("invalid --warp-config format 'no_equals': expected KEY=VALUE", stderr.getvalue())

    def test_parser_has_warp_config_arg(self):
        """The base parser should include --warp-config."""
        parser = create_parser()
        args = parser.parse_known_args(["--warp-config", "mode=release"])[0]
        self.assertEqual(args.warp_config, ["mode=release"])

    def test_default_warp_config_empty(self):
        """Default value of --warp-config should be an empty list."""
        parser = create_parser()
        args = parser.parse_known_args([])[0]
        self.assertEqual(args.warp_config, [])

    def test_quiet_preserves_stricter_log_level(self):
        """--quiet should not lower an explicit stricter log_level override."""
        parser = create_parser()
        argv = ["example", "--viewer", "null", "--warp-config", "log_level=40", "--quiet"]
        with mock.patch.object(sys, "argv", argv):
            viewer, _args = init(parser)

        self.assertIsNotNone(viewer)
        self.assertEqual(wp.config.log_level, wp.LOG_ERROR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
