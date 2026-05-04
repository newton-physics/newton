# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton._src.sensors.warp_raytrace.types import RenderConfig


class TestRenderConfigLaunchShape(unittest.TestCase):
    """Cover the ``block_dim`` and ``launch_bounds`` knobs on
    :class:`~newton._src.sensors.warp_raytrace.types.RenderConfig`."""

    def test_defaults_preserve_existing_behavior(self):
        cfg = RenderConfig()
        self.assertEqual(cfg.block_dim, 0)
        self.assertIsNone(cfg.launch_bounds)

    def test_block_dim_field_round_trip(self):
        cfg = RenderConfig(block_dim=128)
        self.assertEqual(cfg.block_dim, 128)

    def test_launch_bounds_field_round_trip(self):
        cfg = RenderConfig(launch_bounds=(128, 4))
        self.assertEqual(cfg.launch_bounds, (128, 4))

    def test_block_dim_participates_in_cache_key(self):
        # RenderContext keys its kernel cache by ``hash(config)``, so a
        # change in ``block_dim`` must invalidate the cached kernel.
        a = RenderConfig()
        b = RenderConfig(block_dim=128)
        self.assertNotEqual(hash(a), hash(b))

    def test_launch_bounds_participates_in_cache_key(self):
        a = RenderConfig()
        b = RenderConfig(launch_bounds=(128, 4))
        self.assertNotEqual(hash(a), hash(b))

    def test_equal_configs_hash_equal(self):
        a = RenderConfig(block_dim=128, launch_bounds=(128, 4))
        b = RenderConfig(block_dim=128, launch_bounds=(128, 4))
        self.assertEqual(hash(a), hash(b))


if __name__ == "__main__":
    unittest.main()
