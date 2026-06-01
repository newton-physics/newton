# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for SolverKamino's handling of non-aligned joint frames.

Newton allows :attr:`Model.joint_X_p` and :attr:`Model.joint_X_c` to carry
different rotations on a single joint (i.e. ``q_pj != q_cj``). Kamino's joint
constraint formulation now consumes both frames natively via
:attr:`JointsModel.X_Bj` and :attr:`JointsModel.X_Fj`. The key invariants
the fix establishes are:

1. ``ModelKamino.from_newton`` must not mutate Newton's body-local arrays
   (``body_com``, ``body_inertia``, ``shape_transform``, ``joint_X_p``,
   ``joint_X_c``) — they are needed unchanged by ``eval_fk`` and the
   visualizer downstream.
2. For aligned joints the conversion still produces ``X_Bj == X_Fj``,
   keeping behaviour bit-identical for all existing models.
3. The Kamino solver runs cleanly to completion on a model with non-aligned
   joint frames (no constraint projection failure, no NaN propagation).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _build_revolute_with_offset(angle_pj: float, angle_cj: float) -> newton.Model:
    """Build a tiny model: world to single body via revolute joint about Y.

    The joint is placed with non-aligned parent/child rotations
    (``angle_pj`` and ``angle_cj`` about the world Z axis on the parent and
    follower side respectively), producing a non-identity ``q_pj * inv(q_cj)``
    offset that previously required body-frame absorption.
    """
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    builder.begin_world()
    bid = builder.add_link(
        label="link",
        mass=1.0,
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
        lock_inertia=True,
    )
    builder.add_shape_box(label="box", body=bid, hx=0.1, hy=0.1, hz=0.1)

    parent_rot = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), float(angle_pj))
    child_rot = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), float(angle_cj))

    jid = builder.add_joint_revolute(
        label="world_to_link",
        parent=-1,
        child=bid,
        axis=newton.Axis.Y,
        parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), parent_rot),
        child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), child_rot),
    )
    builder.add_articulation([jid])
    builder.end_world()

    return builder.finalize()


class TestKaminoNonAlignedJointFrames(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_conversion_does_not_mutate_newton_arrays(self):
        """``ModelKamino.from_newton`` must leave Newton's body-local arrays untouched.

        Pre-fix, the converter rotated ``model.body_com``, ``model.body_inertia``,
        ``model.shape_transform``, ``model.joint_X_p`` and ``model.joint_X_c`` in-place,
        which corrupted ``eval_fk`` and the renderer downstream. With the two-frame
        formulation the rotation is no longer absorbed into the body frame, so all
        of Newton's per-body and per-joint arrays survive the conversion unchanged.
        """
        model = _build_revolute_with_offset(angle_pj=0.4, angle_cj=-0.3)

        snapshot = {
            "body_com": model.body_com.numpy().copy(),
            "body_inertia": model.body_inertia.numpy().copy(),
            "shape_transform": model.shape_transform.numpy().copy(),
            "joint_X_p": model.joint_X_p.numpy().copy(),
            "joint_X_c": model.joint_X_c.numpy().copy(),
        }

        # Build Kamino model — this used to mutate Newton's arrays in-place.
        ModelKamino.from_newton(model)

        for name, before in snapshot.items():
            after = getattr(model, name).numpy()
            np.testing.assert_array_equal(after, before, err_msg=f"ModelKamino.from_newton mutated model.{name}")

    def test_aligned_joint_frames_are_bit_identical(self):
        """For aligned joint frames, ``X_Bj`` and ``X_Fj`` must be equal.

        This pins the invariant that aligned-joint behaviour is unchanged after
        the refactor. Every joint built with ``parent_xform.rotation == child_xform.rotation``
        produces a :class:`JointsModel` with ``X_Bj == X_Fj``, so existing
        models and solver behaviour remain bit-identical.
        """
        rot = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), 0.7)
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        bid = builder.add_link(
            label="link",
            mass=1.0,
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            lock_inertia=True,
        )
        builder.add_shape_box(label="box", body=bid, hx=0.1, hy=0.1, hz=0.1)
        jid = builder.add_joint_revolute(
            label="world_to_link",
            parent=-1,
            child=bid,
            axis=newton.Axis.Y,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), rot),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.0), rot),
        )
        builder.add_articulation([jid])
        builder.end_world()
        model = builder.finalize()

        kamino = ModelKamino.from_newton(model)
        np.testing.assert_array_equal(
            kamino.joints.X_Bj.numpy(),
            kamino.joints.X_Fj.numpy(),
            err_msg="Aligned joint frames must produce equal X_Bj and X_Fj",
        )

    def test_non_aligned_joint_frames_X_Bj_differs_from_X_Fj(self):
        """For non-aligned frames, ``X_Bj`` and ``X_Fj`` must differ.

        Sanity check that the converter actually writes both frames separately:
        a non-zero offset between parent and child rotations must show up as
        a non-zero difference between the two output matrices.
        """
        model = _build_revolute_with_offset(angle_pj=0.4, angle_cj=-0.3)
        kamino = ModelKamino.from_newton(model)

        X_Bj = kamino.joints.X_Bj.numpy()
        X_Fj = kamino.joints.X_Fj.numpy()

        diff = np.linalg.norm(X_Bj - X_Fj)
        self.assertGreater(
            diff,
            1e-3,
            f"Non-aligned joint frames should produce different X_Bj and X_Fj (diff={diff})",
        )

    def test_kamino_joint_X_Bj_matches_newton_joint_frame(self):
        """``X_Bj`` must equal ``R(q_pj) @ R_axis`` from Newton's ``joint_X_p``.

        With the two-frame formulation, the parent-side joint frame in Kamino
        is no longer mutated by absorption of the child-side rotation. It must
        equal the natural product of the Newton-side parent rotation and the
        DoF axis basis matrix — for a revolute joint about Y this is
        ``R(q_pj) @ identity == R(q_pj)``.
        """
        # Use a parent rotation about Z and a Y-axis revolute joint so the axis
        # basis is a known permutation matrix.
        angle_pj = 0.4
        model = _build_revolute_with_offset(angle_pj=angle_pj, angle_cj=-0.3)
        kamino = ModelKamino.from_newton(model)

        # The revolute Y joint stores its axis as the FIRST column of the joint
        # frame (Kamino canonicalises DoF axes to (X, Y, Z)). For a revolute about
        # the world Y axis with the parent-side rotation R(q_pj) applied, the
        # first column of X_Bj is therefore R(q_pj) * (0, 1, 0).
        X_Bj = kamino.joints.X_Bj.numpy()
        ax_col = X_Bj[0, :, 0]

        q_pj = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), angle_pj)
        R_q_pj = np.array(wp.quat_to_matrix(q_pj)).reshape(3, 3)
        expected = R_q_pj @ np.array([0.0, 1.0, 0.0], dtype=np.float32)

        np.testing.assert_allclose(
            ax_col,
            expected,
            atol=1e-6,
            err_msg="X_Bj first column must equal R(q_pj) * (0, 1, 0) for a Y-revolute",
        )

    def test_step_runs_without_nan_on_non_aligned_joint(self):
        """A full ``solver.step`` on a non-aligned-joint model must produce finite poses.

        Pre-fix, the body-frame absorption was applied as a runtime push/pop around
        each step, mutating ``state_in.body_q`` in place and risking divergence when
        the input/output state aliasing wasn't perfectly symmetric. This test runs a
        few solver iterations and verifies the resulting body poses are finite, the
        body has actually moved (joint not stuck), and the unit-quaternion invariant
        is preserved.
        """
        model = _build_revolute_with_offset(angle_pj=0.5, angle_cj=-0.2)

        solver = SolverKamino(model)

        state_in = model.state()
        state_out = model.state()

        body_q_initial = state_in.body_q.numpy().copy()

        # Run a handful of steps with no contacts and no actuation; body should
        # swing under gravity through the revolute joint without blowing up.
        for _ in range(5):
            solver.step(state_in, state_out, control=None, contacts=None, dt=1e-3)
            state_in, state_out = state_out, state_in

        body_q_final = state_in.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(body_q_final)), "body_q contains non-finite values")

        # Unit-quaternion invariant
        quat = body_q_final[:, 3:]
        norms = np.linalg.norm(quat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3, err_msg="body_q rotation is not unit-quaternion")

        # Sanity: under gravity, the body must have moved (rotation or translation).
        delta = np.linalg.norm(body_q_final - body_q_initial)
        self.assertGreater(delta, 1e-4, "body did not move under gravity (joint may be stuck)")


if __name__ == "__main__":
    unittest.main()
