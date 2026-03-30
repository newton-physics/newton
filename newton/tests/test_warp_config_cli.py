# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--warp-config KEY=VALUE`` CLI option."""

import contextlib
import io
import unittest

import warp as wp

from newton.examples import _apply_warp_config, create_parser


class TestWarpConfigCLI(unittest.TestCase):
    """Tests for :func:`_apply_warp_config`."""

    def setUp(self):
        self._saved_config = {attr: getattr(wp.config, attr) for attr in dir(wp.config) if not attr.startswith("__")}

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
        self.assertEqual(wp.config.verbose, self._saved_config["verbose"])

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
        parser, args = self._parse("--warp-config", "verbose=True")
        _apply_warp_config(parser, args)
        self.assertIs(wp.config.verbose, True)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
