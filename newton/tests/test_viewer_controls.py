# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton._src.viewer.viewer_null import ViewerNull


class _StubViewerGL:
    """Replicates only the pause/step state machine from ViewerGL.

    ViewerGL requires a display to instantiate, so this stub lets us test the
    should_step() logic in headless CI without a window.
    """

    def __init__(self):
        self._paused = False
        self._step_requested = False

    def should_step(self) -> bool:
        if not self._paused:
            self._step_requested = False
            return True
        if self._step_requested:
            self._step_requested = False
            return True
        return False


class TestViewerBaseShouldStep(unittest.TestCase):
    """ViewerBase.should_step() defaults to not self.is_paused()."""

    def test_returns_true_when_not_paused(self):
        viewer = ViewerNull()
        self.assertTrue(viewer.should_step())

    def test_returns_true_on_repeated_calls(self):
        viewer = ViewerNull()
        for _ in range(3):
            self.assertTrue(viewer.should_step())


class TestViewerGLShouldStep(unittest.TestCase):
    """ViewerGL.should_step() state machine: running, paused, and single-step."""

    def test_returns_true_when_running(self):
        v = _StubViewerGL()
        self.assertTrue(v.should_step())

    def test_returns_false_when_paused(self):
        v = _StubViewerGL()
        v._paused = True
        self.assertFalse(v.should_step())

    def test_returns_true_once_after_step_request(self):
        v = _StubViewerGL()
        v._paused = True
        v._step_requested = True
        self.assertTrue(v.should_step())
        self.assertFalse(v.should_step())

    def test_stale_request_cleared_when_running(self):
        # Reproduces the bug: . pressed while running, then SPACE to pause.
        # The flag must not survive into the paused state and fire a spurious step.
        v = _StubViewerGL()
        v._step_requested = True  # set while not paused
        v.should_step()  # running frame — must clear the flag
        v._paused = True
        self.assertFalse(v.should_step())

    def test_multiple_step_requests_fire_once_each(self):
        v = _StubViewerGL()
        v._paused = True
        v._step_requested = True
        self.assertTrue(v.should_step())
        v._step_requested = True
        self.assertTrue(v.should_step())
        self.assertFalse(v.should_step())


if __name__ == "__main__":
    unittest.main(verbosity=2)
