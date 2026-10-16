# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

# Interleave articulations of different sizes so each size group is non-contiguous in the model.
_ARTICULATION_JOINTS = (
    ("revolute", "revolute"),
    ("free", "ball", "revolute"),
    ("revolute", "revolute"),
    ("free", "ball", "revolute"),
)


def _build_model(device, requires_grad=False):
    builder = newton.ModelBuilder()
    for i, joint_types in enumerate(_ARTICULATION_JOINTS):
        parent = -1
        joints = []
        for j, joint_type in enumerate(joint_types):
            child = builder.add_link()
            builder.add_shape_box(child, hx=0.1, hy=0.05, hz=0.05)
            parent_xform = wp.transform((2.0 * i, 0.0, 1.0) if parent == -1 else (0.2, 0.0, 0.0), wp.quat_identity())
            if joint_type == "free":
                joint = builder.add_joint_free(child, parent=parent, parent_xform=parent_xform)
            elif joint_type == "ball":
                joint = builder.add_joint_ball(parent, child, parent_xform=parent_xform)
            else:
                joint = builder.add_joint_revolute(
                    parent, child, parent_xform=parent_xform, axis=newton.Axis.Y if j % 2 else newton.Axis.Z
                )
            joints.append(joint)
            parent = child
        builder.add_articulation(joints)
    builder.joint_armature = [0.01] * builder.joint_dof_count
    model = builder.finalize(device=device, requires_grad=requires_grad)
    rng = np.random.default_rng(42)
    model.joint_qd.assign(rng.uniform(-1.0, 1.0, model.joint_dof_count).astype(np.float32))
    return model


def _simulate(model, steps, tape=None, **solver_kwargs):
    solver = newton.solvers.SolverFeatherstone(model, **solver_kwargs)
    states = [model.state() for _ in range(steps + 1)]
    control = model.control()
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    with tape if tape is not None else contextlib.nullcontext():
        for i in range(steps):
            solver.step(states[i], states[i + 1], control, None, 1.0e-3)
    return states


@wp.kernel
def _joint_qd_loss(joint_qd: wp.array[float], loss: wp.array[float]):
    tid = wp.tid()
    wp.atomic_add(loss, 0, joint_qd[tid] * joint_qd[tid])


def test_tile_gemm_mixed_articulation_sizes(test, device):
    model = _build_model(device)
    expected = _simulate(model, 5)[-1]
    for fuse_cholesky in (False, True):
        with test.subTest(fuse_cholesky=fuse_cholesky):
            actual = _simulate(model, 5, use_tile_gemm=True, fuse_cholesky=fuse_cholesky)[-1]
            np.testing.assert_allclose(actual.body_q.numpy(), expected.body_q.numpy(), rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(actual.body_qd.numpy(), expected.body_qd.numpy(), rtol=1e-4, atol=1e-4)


def test_tile_gemm_mixed_articulation_sizes_grad(test, device):
    def compute_grad(**solver_kwargs):
        model = _build_model(device, requires_grad=True)
        tape = wp.Tape()
        states = _simulate(model, 2, tape, **solver_kwargs)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        with tape:
            wp.launch(
                _joint_qd_loss, dim=model.joint_dof_count, inputs=[states[-1].joint_qd], outputs=[loss], device=device
            )
        tape.backward(loss)
        return states[0].joint_qd.grad.numpy()

    expected = compute_grad()
    test.assertGreater(np.abs(expected).max(), 0.0)
    actual = compute_grad(use_tile_gemm=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


class TestSolverFeatherstone(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestSolverFeatherstone,
    "test_tile_gemm_mixed_articulation_sizes",
    test_tile_gemm_mixed_articulation_sizes,
    devices=devices,
)
add_function_test(
    TestSolverFeatherstone,
    "test_tile_gemm_mixed_articulation_sizes_grad",
    test_tile_gemm_mixed_articulation_sizes_grad,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
