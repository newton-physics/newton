# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import ClassVar

from newton.utils import run_benchmark


class TestRunBenchmark(unittest.TestCase):
    def test_run_benchmark_with_setup_cache(self):
        """Pass one setup cache through the full benchmark lifecycle."""
        cache_events = []

        class CachedBenchmark:
            params: ClassVar = [[2, 3]]
            setup_cache_calls = 0
            cache_value: ClassVar = {"base": 10}

            def setup_cache(self):
                type(self).setup_cache_calls += 1
                return self.cache_value

            def setup(self, cache, value):
                cache_events.append(("setup", cache, value))

            def time_value(self, cache, value):
                cache_events.append(("time", cache, value))

            def track_value(self, cache, value):
                cache_events.append(("track", cache, value))
                return cache["base"] + value

            def teardown(self, cache, value):
                cache_events.append(("teardown", cache, value))

        results = run_benchmark(CachedBenchmark, print_results=False)

        self.assertEqual(CachedBenchmark.setup_cache_calls, 1)
        self.assertTrue(all(cache is CachedBenchmark.cache_value for _, cache, _ in cache_events))
        self.assertEqual([event for event, _, _ in cache_events].count("setup"), 2)
        self.assertEqual([event for event, _, _ in cache_events].count("time"), 4)
        self.assertEqual([event for event, _, _ in cache_events].count("track"), 2)
        self.assertEqual([event for event, _, _ in cache_events].count("teardown"), 2)
        self.assertEqual({value for _, _, value in cache_events}, {2, 3})
        self.assertEqual(results[("track_value", (2,))], 12)
        self.assertEqual(results[("track_value", (3,))], 13)


if __name__ == "__main__":
    unittest.main()
