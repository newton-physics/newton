# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The matrix-free solve on CPU (portable row kernels) tracks the CUDA solve."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel

_STEPS = 40
_DT = 1.0 / 240.0


def _add_arm(builder, x, top, *, links=3, free_root=False):
    """A driven, limited capsule chain hanging from ``top`` (optionally on a free root body)."""
    joints, parent = [], -1
    if free_root:
        root = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, top + 0.2), wp.quat_identity()))
        builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.05)
        joints.append(builder.add_joint_free(parent=-1, child=root))
        parent = root
    for i in range(links):
        link = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, top - 0.3 * i), wp.quat_identity()))
        builder.add_shape_capsule(link, radius=0.04, half_height=0.12)
        if parent < 0:
            parent_xform = wp.transform(wp.vec3(x, 0.0, top + 0.15), wp.quat_identity())
        elif free_root and i == 0:
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity())
        else:
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, -0.15), wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
                limit_lower=-0.6,
                limit_upper=0.6,
                target_ke=200.0,
                target_kd=5.0,
                target_q=0.5,
            )
        )
        parent = link
    builder.add_articulation(joints)


def _add_box(builder, x, z, half=0.08):
    body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z), wp.quat_rpy(0.05, 0.02, 0.1)))
    builder.add_shape_box(body, hx=half, hy=half, hz=half)


def _arm_on_boxes(world):
    """Two driven limited arms, one pressing on a free box, plus a lone box."""
    _add_arm(world, 0.0, 0.92)
    _add_arm(world, 1.5, 1.05)
    _add_box(world, 0.0, 0.09)
    _add_box(world, 3.0, 0.2)


def _box_stack(world):
    for k in range(3):
        _add_box(world, 0.01 * k, 0.1 + 0.17 * k)


def _arm_velocity_limits(world):
    _add_arm(world, 0.0, 1.05)
    _add_box(world, 0.6, 0.09)


def _mixed(world):
    """A floating-base chain falling onto free boxes next to a fixed arm."""
    _add_arm(world, 0.0, 0.6, links=2, free_root=True)
    _add_box(world, 0.0, 0.09)
    _add_box(world, 0.6, 0.09)
    _add_arm(world, 1.5, 0.92)


_SCENES = {
    "arm_on_boxes": (_arm_on_boxes, {}),
    "box_stack": (_box_stack, {}),
    "arm_velocity_limits": (
        _arm_velocity_limits,
        {"enable_joint_velocity_limits": True, "drive_mode": "rows", "velocity_limit_activation_fraction": 0.0},
    ),
    "mixed": (_mixed, {"row_schedule": "phased"}),
}


def _build(device, scene):
    world = newton.ModelBuilder()
    _SCENES[scene][0](world)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(world, 2, spacing=(0.0, 6.0, 0.0))
    model = builder.finalize(device=device)
    if scene == "arm_velocity_limits":
        model.joint_velocity_limit.fill_(0.5)
    return model


def _rollout(device, scene, mode):
    model = _build(device, scene)
    solver = SolverSteel(
        model,
        solve_mode=mode[0],
        articulated_contact_response=mode[1],
        iterations=12,
        enable_joint_limits=True,
        articulation_constraint_max=64,
        free_body_constraint_max=64,
        # Fixed contact and row order, so the two devices solve the same rows in the same sequence.
        deterministic=wp.DeterministicMode.RUN_TO_RUN,
        **_SCENES[scene][1],
    )
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_a, state_b, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_a)
    start = state_a.body_q.numpy().copy()
    q, qd = [], []
    for _ in range(_STEPS):
        state_a.clear_forces()
        pipeline.collide(state_a, contacts)
        solver.step(state_a, state_b, control, contacts, _DT)
        state_a, state_b = state_b, state_a
        q.append(state_a.body_q.numpy().copy())
        qd.append(state_a.body_qd.numpy().copy())
    return solver, start, np.asarray(q), np.asarray(qd)


# (solve_mode, articulated_contact_response); "colored" runs the serial propagation sweep on CPU.
_MODES = (("serial", "immediate"), ("serial", "propagation"), ("colored", "propagation"))


class TestSteelCpuFallback(unittest.TestCase):
    def test_cpu_rollouts_move_and_stay_finite(self):
        for scene in _SCENES:
            for mode in _MODES:
                with self.subTest(scene=scene, mode=mode):
                    solver, start, q, qd = _rollout("cpu", scene, mode)
                    self.assertTrue(np.isfinite(q).all() and np.isfinite(qd).all())
                    # Not a silent no-op: contacts hold the bodies above the ground and the drives swing the arms.
                    self.assertGreater(np.abs(q[-1, :, :3] - start[:, :3]).max(), 1.0e-3)
                    self.assertGreater(q[-1, :, 2].min(), 0.0)
                    self.assertFalse(solver._propagation_colored)

    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA")
    def test_cpu_tracks_cuda(self):
        for scene in _SCENES:
            for mode in _MODES:
                with self.subTest(scene=scene, mode=mode):
                    _, _, q_cpu, qd_cpu = _rollout("cpu", scene, mode)
                    _, _, q_gpu, qd_gpu = _rollout("cuda:0", scene, mode)
                    # Same rows in the same order; only the dot-product summation order differs. Velocities
                    # see isolated stick/slip flips at impacts, positions stay within float round-off growth.
                    np.testing.assert_allclose(q_cpu[..., :3], q_gpu[..., :3], atol=1.0e-4)
                    np.testing.assert_allclose(qd_cpu, qd_gpu, atol=0.15)

    def test_cuda_only_features_raise_on_cpu(self):
        model = _build("cpu", "box_stack")
        for options in (
            {"friction_mode": "bisection", "friction_anchor_beta": 0.0},
            {"enable_joint_friction": True},
            {"contact_compliance": True, "friction_anchor_beta": 0.0},
        ):
            with self.subTest(options=options), self.assertRaisesRegex(NotImplementedError, "require CUDA"):
                SolverSteel(model, **options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
