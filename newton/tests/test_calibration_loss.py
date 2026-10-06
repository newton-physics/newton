# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the calibration loss interface and the Chamfer loss."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.calibration.loss import CalibrationLoss
from newton._src.calibration.loss_chamfer import CHAMFER, CHAMFER_MISS, _goal_edt
from newton.tests.unittest_utils import add_function_test, get_test_devices

# The test goal is a horizontal line one pixel thick: row GOAL_ROW, columns
# GOAL_X0 to GOAL_X1 inclusive, in full-image pixels. It has 41 pixels and is
# 40 px long, which is the goal extent that normalizes the loss. The crop
# [x0, y0, x1, y1] has a nonzero origin.
CROP = [10, 5, 110, 45]
GOAL_ROW = 25
GOAL_X0, GOAL_X1 = 30, 70
GOAL_PIXELS = GOAL_X1 - GOAL_X0 + 1
GOAL_LENGTH = float(GOAL_X1 - GOAL_X0)


def make_goal_mask():
    mask = np.zeros((CROP[3] - CROP[1], CROP[2] - CROP[0]), dtype=np.uint8)
    mask[GOAL_ROW - CROP[1], GOAL_X0 - CROP[0] : GOAL_X1 - CROP[0] + 1] = 255
    return mask


def make_axis(x_start=GOAL_X0, x_end=GOAL_X1, offset_px=0.0, n_nodes=11):
    """A horizontal projected cable axis from ``x_start`` to ``x_end``, ``offset_px`` below the goal."""
    xs = np.linspace(x_start, x_end, n_nodes)
    return np.stack([xs, np.full(n_nodes, GOAL_ROW + offset_px)], axis=1)


def make_hidden(axis, nodes):
    """``axis`` with the given nodes moved behind the lens."""
    hidden = axis.copy()
    hidden[nodes] = -2.0e8
    return hidden


def chamfer_on(device, *axes):
    """Score each axis as one world of a single accum() call on ``device``."""
    goal = CHAMFER.prepare(make_goal_mask())
    geom = wp.array(np.stack(axes).astype(np.float32), dtype=wp.float32, device=device)
    accumulator = CHAMFER.make_accum(len(axes), device)
    CHAMFER.accum(goal, None, CROP, accumulator, geom=geom)
    return accumulator.totals()


class TestCalibrationLoss(unittest.TestCase):
    def test_incomplete_loss_fails_at_construction(self):
        """Reject a loss without score when it is constructed, not when it is first called."""

        class LossPrepareOnly(CalibrationLoss):
            def prepare(self, goal_mask):
                return goal_mask

        with self.assertRaises(TypeError):
            LossPrepareOnly()

    def test_accum_flag_requires_both_methods(self):
        """Reject supports_accum at class definition when accum is not implemented."""
        with self.assertRaisesRegex(TypeError, "supports_accum"):

            class LossAccumFlagOnly(CalibrationLoss):
                supports_accum = True

                def prepare(self, goal_mask):
                    return goal_mask

                def score(self, goal, sim_mask, crop, *, geom=None):
                    return 0.0

                def make_accum(self, n_worlds, device):
                    return None


class TestCalibrationChamfer(unittest.TestCase):
    """The chamfer objective."""

    def test_goal_edt_matches_an_exact_euclidean_transform(self):
        """Verify the goal distance field equals the exact distance to the nearest goal pixel."""
        mask = np.zeros((12, 15), dtype=np.uint8)
        mask[3, 4] = mask[8, 11] = mask[6, 2] = 255
        goal = np.argwhere(mask > 0)
        rows, cols = np.indices(mask.shape)
        exact = np.min(
            np.hypot(rows[..., None] - goal[:, 0], cols[..., None] - goal[:, 1]),
            axis=-1,
        )
        np.testing.assert_allclose(_goal_edt(mask), exact, atol=1e-9)

    def test_score_scores_one_world(self):
        """Verify score() gives one world's loss: an axis 4 px off the goal scores 4/40."""
        goal = CHAMFER.prepare(make_goal_mask())
        score = CHAMFER.score(goal, None, CROP, geom=make_axis(offset_px=4.0))
        self.assertAlmostEqual(score, 4.0 / GOAL_LENGTH, places=6)

    def test_empty_goal_mask_is_refused(self):
        """Refuse a goal frame without cable pixels instead of measuring to a point outside it."""
        empty = np.zeros_like(make_goal_mask())
        with self.assertRaisesRegex(ValueError, "no pixels"):
            CHAMFER.prepare(empty)

    def test_score_requires_the_projected_cable(self):
        """Refuse to score without geometry, which this loss needs instead of a render."""
        goal = CHAMFER.prepare(make_goal_mask())
        with self.assertRaisesRegex(ValueError, "geom"):
            CHAMFER.score(goal, None, CROP)


def test_loss_is_the_offset_in_goal_lengths(test, device):
    """Verify an axis d px off the goal scores d / 40, without saturating.

    Every axis sample is d px from the goal and every goal pixel is d px from the
    axis, so both directions are d px.
    """
    offsets = [0.0, 2.0, 5.0, 10.0, 18.0]
    losses = chamfer_on(device, *(make_axis(offset_px=d) for d in offsets))
    np.testing.assert_allclose(losses, np.array(offsets) / GOAL_LENGTH, rtol=1e-6)


def test_goal_to_sim_penalizes_an_axis_covering_half_the_goal(test, device):
    """Verify goal->sim: the axis covers x 30 to 50, so goal pixels 51 to 70 lie 1 to 20 px from it.

    sim->goal is 0 and goal->sim is (1 + ... + 20) / 41 = 210 / 41 px.
    """
    (loss,) = chamfer_on(device, make_axis(x_end=50.0))
    test.assertAlmostEqual(loss, 0.5 * (210.0 / GOAL_PIXELS) / GOAL_LENGTH, places=6)


def test_sim_to_goal_penalizes_an_axis_longer_than_the_goal(test, device):
    """Verify sim->goal: the 64 px axis extends 12 px past each end of the goal.

    goal->sim is 0. The samples are spread uniformly along the axis, so their
    mean distance to the goal is 2 * (12 * 12 / 2) / 64 = 2.25 px. The sample
    spacing is 1/8 px, so rounding each sample to its pixel keeps this exact.
    """
    (loss,) = chamfer_on(device, make_axis(x_start=18.0, x_end=82.0))
    test.assertAlmostEqual(loss, 0.5 * 2.25 / GOAL_LENGTH, places=6)


def test_axis_outside_the_crop_takes_the_clamp(test, device):
    """Verify samples outside the crop take the 1024 px clamp instead of being dropped.

    Otherwise a candidate could lower its score by moving the cable out of view.
    The axis is 30 px below the goal and below the crop, so sim->goal is 1024 px
    and goal->sim is 30 px.
    """
    (loss,) = chamfer_on(device, make_axis(offset_px=30.0))
    test.assertAlmostEqual(loss, 0.5 * (1024.0 + 30.0) / GOAL_LENGTH, places=5)


def test_nodes_behind_the_lens_are_skipped(test, device):
    """Verify segments touching a node behind the lens are left out.

    Hiding nodes 0 to 4 leaves the axis from x 50 to 70, which scores as that
    half axis alone. With every node hidden, the frame scores CHAMFER_MISS.
    """
    axis = make_axis()
    half, partly_hidden, all_hidden = chamfer_on(
        device, make_axis(x_start=50.0), make_hidden(axis, [0, 1, 2, 3, 4]), make_hidden(axis, list(range(11)))
    )
    test.assertAlmostEqual(partly_hidden, half, places=6)
    test.assertAlmostEqual(all_hidden, CHAMFER_MISS, places=6)


def test_accumulator_sums_over_frames(test, device):
    """Verify one accumulator sums the loss of several frames, as over a sequence."""
    goal = CHAMFER.prepare(make_goal_mask())
    accumulator = CHAMFER.make_accum(1, device)
    for offset in (2.0, 12.0):
        geom = wp.array(make_axis(offset_px=offset)[None].astype(np.float32), dtype=wp.float32, device=device)
        CHAMFER.accum(goal, None, CROP, accumulator, geom=geom)
    test.assertAlmostEqual(accumulator.totals()[0], (2.0 + 12.0) / GOAL_LENGTH, places=6)


devices = get_test_devices()
for test_func in (
    test_loss_is_the_offset_in_goal_lengths,
    test_goal_to_sim_penalizes_an_axis_covering_half_the_goal,
    test_sim_to_goal_penalizes_an_axis_longer_than_the_goal,
    test_axis_outside_the_crop_takes_the_clamp,
    test_nodes_behind_the_lens_are_skipped,
    test_accumulator_sums_over_frames,
):
    add_function_test(TestCalibrationChamfer, test_func.__name__, test_func, devices=devices)


if __name__ == "__main__":
    unittest.main()
