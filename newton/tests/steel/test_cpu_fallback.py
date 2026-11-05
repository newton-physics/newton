# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The matrix-free solve on CPU (portable row kernels) tracks the CUDA solve."""

import functools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.exceptions import NewtonWarning
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

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
    # Regularized updates and the unbiased velocity-only passes.
    "box_stack": (_box_stack, {"contact_regularization": 0.02, "velocity_iterations": 2}),
    # Velocity-only passes freeze the drive rows (velocity_drive_mode="freeze").
    "arm_velocity_limits": (
        _arm_velocity_limits,
        {
            "enable_joint_velocity_limits": True,
            "drive_mode": "rows",
            "velocity_limit_activation_fraction": 0.0,
            "velocity_iterations": 2,
        },
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
    with warnings.catch_warnings():
        # solve_mode="colored" on CPU warns that it runs the serial sweep; that is what this module covers.
        warnings.filterwarnings("ignore", message="SolverSteel: solve_mode='colored' runs", category=NewtonWarning)
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
    return solver._propagation_colored, start, np.asarray(q), np.asarray(qd)


@functools.cache
def _cpu_rollout(scene, mode):
    """CPU rollouts are the expensive half of this module; both tests share them."""
    return _rollout("cpu", scene, mode)


# (solve_mode, articulated_contact_response); "colored" runs the serial propagation sweep on CPU.
_MODES = (("serial", "immediate"), ("serial", "propagation"), ("colored", "propagation"))


class TestSteelCpuFallback(unittest.TestCase):
    def test_cpu_rollouts_move_and_stay_finite(self):
        for scene in _SCENES:
            for mode in _MODES:
                with self.subTest(scene=scene, mode=mode):
                    colored, start, q, qd = _cpu_rollout(scene, mode)
                    self.assertTrue(np.isfinite(q).all() and np.isfinite(qd).all())
                    # Not a silent no-op: contacts hold the bodies above the ground and the drives swing the arms.
                    self.assertGreater(np.abs(q[-1, :, :3] - start[:, :3]).max(), 1.0e-3)
                    self.assertGreater(q[-1, :, 2].min(), 0.0)
                    self.assertFalse(colored)

    def test_colored_warns_on_cpu(self):
        model = _build("cpu", "box_stack")
        with self.assertWarnsRegex(NewtonWarning, "serial propagation sweep on CPU"):
            SolverSteel(model, solve_mode="colored", articulated_contact_response="propagation")

    def test_cuda_only_features_raise_on_cpu(self):
        model = _build("cpu", "box_stack")
        for options in (
            {"friction_mode": "bisection", "friction_anchor_beta": 0.0},
            {"enable_joint_friction": True},
            {"contact_compliance": True, "friction_anchor_beta": 0.0},
        ):
            with self.subTest(options=options), self.assertRaisesRegex(NotImplementedError, "require CUDA"):
                SolverSteel(model, **options)


def test_cpu_tracks_cuda(test, device):
    for scene in _SCENES:
        for mode in _MODES:
            with test.subTest(scene=scene, mode=mode):
                _, _, q_cpu, qd_cpu = _cpu_rollout(scene, mode)
                _, _, q_gpu, qd_gpu = _rollout(device, scene, mode)
                # Serial modes visit the same rows in the same order and differ only in summation order. "colored"
                # runs the serial index-order sweep on CPU and the color-bucket sweep on CUDA, so conflicting rows
                # can run in a different order. Measured worst cases: position 1e-4, 1 - |dot| 3e-7.
                np.testing.assert_allclose(q_cpu[..., :3], q_gpu[..., :3], atol=2.0e-4)
                # q and -q are the same rotation; 1 - |dot| grows with the square of the angle error.
                dot = np.abs(np.sum(q_cpu[..., 3:] * q_gpu[..., 3:], axis=-1))
                np.testing.assert_array_less(1.0 - dot, 1.0e-5)
                # Velocities resolve impacts differently for a step or two (up to ~1 m/s in the regularized box
                # stack, measured), then agree again: bound those steps and the final velocities.
                velocity_gap = np.abs(qd_cpu - qd_gpu).max(axis=(1, 2))
                test.assertLessEqual(int((velocity_gap > 0.15).sum()), 2)
                test.assertLess(velocity_gap[-1], 0.1)


add_function_test(TestSteelCpuFallback, "test_cpu_tracks_cuda", test_cpu_tracks_cuda, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
