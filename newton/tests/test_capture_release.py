# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scheduling tests for capture-deferred releases, using fake capture state and fake releases."""

import gc
import sys
import threading
import time
import unittest
from unittest import mock

from newton._src.core import capture_release

_WORKER_TIMEOUT_S = 60.0


class TestCaptureRelease(unittest.TestCase):
    def setUp(self):
        self._saved_releases = list(capture_release._pending_releases)
        capture_release._pending_releases.clear()
        self.capturing = False
        patcher = mock.patch.object(capture_release, "_any_cuda_device_capturing", lambda: self.capturing)
        patcher.start()
        self.addCleanup(patcher.stop)
        # The collection hook drains on whichever thread collects; each test opts in only where it checks the hook.
        gc.callbacks.remove(capture_release._release_deferred_after_collection)
        self.addCleanup(gc.callbacks.append, capture_release._release_deferred_after_collection)

    def tearDown(self):
        capture_release._pending_releases.clear()
        capture_release._pending_releases.extend(self._saved_releases)

    def test_release_outside_capture_runs_now_and_drains_the_queue(self):
        """A release outside a capture runs at once and also runs earlier deferred releases."""
        ran = []
        capture_release._pending_releases.append(lambda: ran.append("deferred"))
        capture_release._release_or_defer(lambda: ran.append("new"))
        self.assertEqual(ran, ["new", "deferred"])
        self.assertEqual(len(capture_release._pending_releases), 0)

    def test_release_during_capture_waits_for_the_next_collection(self):
        """A release during a capture waits, and the first collection after the capture runs it."""
        gc.callbacks.append(capture_release._release_deferred_after_collection)
        self.addCleanup(gc.callbacks.remove, capture_release._release_deferred_after_collection)
        ran = []
        self.capturing = True
        capture_release._release_or_defer(lambda: ran.append("texture"))
        gc.collect()
        self.assertEqual(ran, [])
        self.capturing = False
        gc.collect()
        self.assertEqual(ran, ["texture"])

    def test_concurrent_drains_run_each_release_once(self):
        """Drains running on several threads at once run every release exactly once."""
        count = 2000
        ran = []
        for i in range(count):
            capture_release._pending_releases.append(lambda i=i: (ran.append(i), time.sleep(0)))
        thread_count = 8
        barrier = threading.Barrier(thread_count, timeout=_WORKER_TIMEOUT_S)
        entered = threading.local()
        workers = set()

        def capture_check_that_lines_threads_up():
            # Every worker passes its first check together, so the drains overlap; other threads never wait.
            if threading.get_ident() in workers and not getattr(entered, "done", False):
                entered.done = True
                barrier.wait()
            return False

        def worker():
            workers.add(threading.get_ident())
            capture_release.release_deferred()

        switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, switch_interval)
        with mock.patch.object(capture_release, "_any_cuda_device_capturing", capture_check_that_lines_threads_up):
            threads = [threading.Thread(target=worker, daemon=True) for _ in range(thread_count)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(_WORKER_TIMEOUT_S)
        self.assertFalse(any(thread.is_alive() for thread in threads))

        self.assertEqual(sorted(ran), list(range(count)))
        self.assertEqual(len(capture_release._pending_releases), 0)

    def test_reentrant_drain_runs_each_release_once(self):
        """A drain started from inside a release runs each release, old and new, exactly once."""
        ran = []

        def release_a():
            ran.append("a")
            # A finalizer inside a release queues more work and drains again.
            capture_release._pending_releases.append(lambda: ran.append("c"))
            capture_release.release_deferred()

        capture_release._pending_releases.extend([release_a, lambda: ran.append("b")])
        capture_release.release_deferred()
        self.assertEqual(sorted(ran), ["a", "b", "c"])
        self.assertEqual(len(capture_release._pending_releases), 0)

    def test_capture_beginning_mid_drain_keeps_the_rest_queued(self):
        """A capture that begins during a release stops the drain until the capture ends."""
        ran = []

        def release_then_capture():
            ran.append("a")
            self.capturing = True

        capture_release._pending_releases.extend([release_then_capture, lambda: ran.append("b")])
        capture_release.release_deferred()
        self.assertEqual(ran, ["a"])
        self.assertEqual(len(capture_release._pending_releases), 1)
        self.capturing = False
        capture_release.release_deferred()
        self.assertEqual(ran, ["a", "b"])

    def test_raising_release_keeps_later_releases_queued(self):
        """A release that raises leaves the releases after it queued."""
        ran = []

        def failing_release():
            raise RuntimeError("release failed")

        capture_release._pending_releases.extend([failing_release, lambda: ran.append("b")])
        with self.assertRaises(RuntimeError):
            capture_release.release_deferred()
        self.assertEqual(len(capture_release._pending_releases), 1)
        capture_release.release_deferred()
        self.assertEqual(ran, ["b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
