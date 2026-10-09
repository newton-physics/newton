# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Propagation contact rows on several links of one articulation converge instead of oscillating."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 1.0 / 240.0
PROPAGATION_RESPONSES = ("propagation", "propagation-fused", "propagation-colored")
REGULARIZED = {"pgs_beta": 0.0, "pgs_cfm": 0.0, "pgs_contact_regularization": 3.0}
# (articulated_contact_response, propagation_cached_response)
PROPAGATION_ROUTES = (
    ("propagation", False),
    ("propagation-fused", False),
    ("propagation", True),
    ("propagation-colored", False),
    ("propagation-colored", True),
)


def _quadruped_model(device):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.6
    root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_box(root, hx=0.2, hy=0.12, hz=0.04)
    joints = [builder.add_joint_free(root)]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = builder.add_link(xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, 0.08), wp.quat_identity()))
            builder.add_shape_capsule(leg, radius=0.03, half_height=0.1)
            joints.append(
                builder.add_joint_revolute(
                    root,
                    leg,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, -0.03), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.13), wp.quat_identity()),
                )
            )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _three_foot_slider_model(device):
    """One prismatic-Z DOF carrying three spheres on fixed-linked bodies above a plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.mu = 0.0
    slider = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(
        slider, hx=0.3, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    )
    joints = [builder.add_joint_prismatic(-1, slider, axis=wp.vec3(0.0, 0.0, 1.0))]
    for x in (-0.2, 0.0, 0.2):
        foot = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_sphere(foot, radius=0.05)
        joints.append(
            builder.add_joint_fixed(
                slider,
                foot,
                parent_xform=wp.transform(wp.vec3(x, 0.0, -0.05), wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
        )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _solver(model, response, iterations, cached=False, **kwargs):
    options = {
        "articulated_contact_response": response,
        "propagation_cached_response": cached,
        "friction_anchor_beta": 0.0,
        "dense_max_constraints": 96,
        "mf_max_constraints": 96,
        "pgs_iterations": iterations,
    }
    options.update(kwargs)
    return SolverFeatherPGS(model, **options)


def _rollout(model, solver, joint_qd, steps):
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        yield state_0.joint_qd.numpy()


def test_shared_dof_feet_reach_rest(test, device):
    """Stop three feet on one prismatic DOF at the plane for every sweep count."""
    model = _three_foot_slider_model(device)
    for response in PROPAGATION_RESPONSES:
        for iterations in (1, 2, 12, 13):
            with test.subTest(response=response, iterations=iterations):
                solver = _solver(model, response, iterations, pgs_beta=0.0, pgs_cfm=0.0)
                (qd,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
                test.assertLess(abs(float(qd[0])), 1.0e-3)


def test_regularized_feet_reach_immediate_equilibrium(test, device):
    """Keep the regularized contact fixed point of three coupled feet under the split."""
    model = _three_foot_slider_model(device)
    for iterations in (32, 128):
        solver = _solver(model, "immediate", iterations, **REGULARIZED)
        (expected,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
        for response, cached in PROPAGATION_ROUTES:
            with test.subTest(response=response, cached=cached, iterations=iterations):
                solver = _solver(model, response, iterations, cached=cached, **REGULARIZED)
                (qd,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
                test.assertAlmostEqual(float(qd[0]), float(expected[0]), delta=1.0e-4)


def test_graph_replay_matches_eager_steps(test, device):
    """Recount the coupled bodies on every replay of a captured step pair."""
    model = _three_foot_slider_model(device)
    for response, cached in PROPAGATION_ROUTES:
        with test.subTest(response=response, cached=cached):
            eager = _solver(model, response, 12, cached=cached, **REGULARIZED)
            expected = list(_rollout(model, eager, np.array([-1.0], dtype=np.float32), 7))[-1]
            solver = _solver(model, response, 12, cached=cached, **REGULARIZED)
            state, output = model.state(), model.state()
            control = model.control()
            state.joint_qd.assign(np.array([-1.0], dtype=np.float32))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            solver.step(state, output, control, contacts, DT)
            with wp.ScopedCapture(device=model.device) as capture:
                solver.seed_double_buffer_events()
                for source, target in ((output, state), (state, output)):
                    pipeline.collide(source, contacts)
                    solver.step(source, target, control, contacts, DT)
            for _ in range(3):
                wp.capture_launch(capture.graph)
            np.testing.assert_allclose(output.joint_qd.numpy(), expected, atol=1.0e-6)
            test.assertEqual(int(solver.propagation_coupling_group_body_count.numpy().max()), 3)


def test_floating_quadruped_with_joint_velocities_stays_bounded(test, device):
    """Keep the roll rate of four feet coupled through a light free root physical."""
    model = _quadruped_model(device)
    joint_qd = np.linspace(-1.5, 1.5, model.joint_dof_count).astype(np.float32)
    for response in PROPAGATION_RESPONSES:
        with test.subTest(response=response):
            solver = _solver(model, response, 12)
            trajectory = list(_rollout(model, solver, joint_qd, 120))
            test.assertLess(float(np.abs(trajectory[0][3:6]).max()), 10.0)
            test.assertTrue(np.isfinite(trajectory[-1]).all())
            test.assertLess(float(np.abs(np.stack(trajectory)).max()), 50.0)


class TestFeatherPGSPropagationCoupledContacts(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (
    test_shared_dof_feet_reach_rest,
    test_regularized_feet_reach_immediate_equilibrium,
    test_graph_replay_matches_eager_steps,
    test_floating_quadruped_with_joint_velocities_stays_bounded,
):
    add_function_test(TestFeatherPGSPropagationCoupledContacts, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main()
