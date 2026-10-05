# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverFeatherPGS propagation responses combined with the matrix-free contact and drive options."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

RESPONSES = ("propagation", "propagation-fused", "propagation-colored")
PROPAGATION_PATH = 2


def _scene(device, restitution=0.0, mimic=False):
    """A driven, limited three-link chain over the ground with three boxes falling onto it, in two worlds."""
    template = newton.ModelBuilder()
    template.default_shape_cfg.mu = 0.6
    template.default_shape_cfg.restitution = restitution
    parent = -1
    joints = []
    for _ in range(3):
        link = template.add_link()
        template.add_shape_box(link, hx=0.15, hy=0.06, hz=0.045)
        joints.append(
            template.add_joint_revolute(
                parent,
                link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(
                    wp.vec3(0.0, 0.0, 0.3) if parent < 0 else wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()
                ),
                child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
                target_ke=40.0,
                target_kd=2.0,
                target_pos=0.2,
                limit_lower=-0.6,
                limit_upper=0.6,
            )
        )
        parent = link
    template.add_articulation(joints)
    if mimic:
        template.set_joint_mimic(joints[2], joints[1], coeffs=(0.0, 1.0))
    for x in (0.15, 0.45, 0.75):
        box = template.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.45), wp.quat_identity()))
        template.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    template.add_ground_plane()
    builder = newton.ModelBuilder()
    builder.replicate(template, 2)
    return builder.finalize(device=device)


def _solver(model, **kwargs):
    options = {"dense_max_constraints": 128, "mf_max_constraints": 128, "friction_anchor_beta": 0.0}
    options.update(kwargs)
    return SolverFeatherPGS(model, **options)


def _pipeline(model, warmstart=False):
    return newton.CollisionPipeline(model, deterministic=True, contact_matching="latest" if warmstart else "disabled")


def _settled_state(model, steps):
    """Return a mid-trajectory state of the immediate response (boxes landing on the chain)."""
    solver = _solver(model, pgs_iterations=32)
    pipeline = _pipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return state_0


def _one_step(model, state, **kwargs):
    solver = _solver(model, **kwargs)
    pipeline = _pipeline(model, kwargs.get("pgs_warmstart", False))
    contacts = pipeline.contacts()
    state_out = model.state()
    pipeline.collide(state, contacts)
    solver.step(state, state_out, model.control(), contacts, 1.0 / 240.0)
    solver.check_constraint_capacity()
    return solver, contacts, state_out.body_qd.numpy()


CASES = {
    "point friction": {},
    "friction patches": {"friction_anchor_beta": 0.2},
    "regularization": {"pgs_contact_regularization": 0.05},
    "velocity iterations": {"pgs_velocity_iterations": 300},
    "drive rows": {"drive_mode": "physx_pgs"},
    "drive rows and velocity limits": {
        "drive_mode": "physx_pgs",
        "enable_joint_velocity_limits": True,
        "velocity_limit_activation_fraction": 0.5,
    },
    "joint limits": {"enable_joint_limits": True},
}


def test_options_match_the_immediate_response_at_convergence(test, device):
    """Converge each option to the immediate response's step from the same state, in both responses."""
    for name, options in CASES.items():
        model = _scene(device)
        if "velocity_limit_activation_fraction" in options:
            model.joint_velocity_limit.fill_(0.5)
        state = _settled_state(model, 40)
        _, _, reference = _one_step(model, state, pgs_iterations=300, **options)
        for response in RESPONSES:
            with test.subTest(option=name, response=response):
                solver, contacts, body_qd = _one_step(
                    model, state, pgs_iterations=300, articulated_contact_response=response, **options
                )
                count = int(contacts.rigid_contact_count.numpy()[0])
                test.assertTrue((solver.contact_path.numpy()[:count] == PROPAGATION_PATH).any())
                np.testing.assert_allclose(body_qd, reference, rtol=0.0, atol=2.0e-3)


def test_restitution_rebounds_on_propagation_rows(test, device):
    """Rebound a box falling onto the chain with its restitution through the propagation rows."""
    for response in RESPONSES:
        with test.subTest(response=response):
            model = _scene(device, restitution=0.8)
            state = model.state()
            # The first box sits just above link 0 and moves down at 2 m/s.
            box_joint = 3
            q_start = int(model.joint_q_start.numpy()[box_joint])
            qd_start = int(model.joint_qd_start.numpy()[box_joint])
            joint_q = state.joint_q.numpy()
            joint_q[q_start + 2] = 0.3 + 0.045 + 0.05 + 0.002
            state.joint_q.assign(joint_q)
            joint_qd = state.joint_qd.numpy()
            joint_qd[qd_start + 2] = -2.0
            state.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver, contacts, after = _one_step(
                model, state, pgs_iterations=64, articulated_contact_response=response, enable_joint_limits=True
            )
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertTrue((solver.contact_path.numpy()[:count] == PROPAGATION_PATH).any())
            test.assertGreater(float(np.abs(solver.propagation_restitution_bias.numpy()).max()), 1.0)
            test.assertGreater(float(after[3, 2]), 0.5)
            _, _, reference = _one_step(model, state, pgs_iterations=64, enable_joint_limits=True)
            np.testing.assert_allclose(after[3], reference[3], rtol=0.0, atol=1.0e-2)


def test_regularization_weights_reach_the_propagation_rows(test, device):
    """Write the regularization weight of penetrating propagation contact rows.

    A row regularizes against its unsplit diagonal ``d``: ``w = 1 / (1 + g d / d_s)`` for the split diagonal ``d_s``.
    """
    model = _scene(device)
    state = _settled_state(model, 60)
    for response in RESPONSES:
        with test.subTest(response=response):
            solver, _, _ = _one_step(
                model, state, pgs_iterations=8, articulated_contact_response=response, pgs_contact_regularization=1.0
            )
            count = int(solver.propagation_constraint_count.numpy()[0])
            rows = solver.propagation_row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_CONTACT
            penetrating = solver.propagation_phi.numpy()[0, :count] <= 0.0
            weights = solver.propagation_row_w.numpy()[0, :count]
            # The unsplit diagonal from the bodies' own responses.
            body_response = solver.propagation_body_response.numpy()
            d = np.full(count, solver.pgs_cfm)
            for body, J in (
                (solver.propagation_body_a.numpy()[0, :count], solver.propagation_J_a.numpy()[0, :count]),
                (solver.propagation_body_b.numpy()[0, :count], solver.propagation_J_b.numpy()[0, :count]),
            ):
                has_body = body >= 0
                d[has_body] += np.einsum("ri,rij,rj->r", J[has_body], body_response[body[has_body]], J[has_body])
            d_split = 1.0 / solver.propagation_eff_mass_inv.numpy()[0, :count]
            checked = rows & penetrating
            test.assertTrue((checked & (d_split > 1.01 * d)).any(), "no mass-split row was checked")
            np.testing.assert_allclose(weights[checked], (1.0 / (1.0 + d / d_split))[checked], rtol=1.0e-4)


def test_warm_start_seeds_the_propagation_rows(test, device):
    """Carry the propagation impulses of resting contacts into the next step."""
    model = _scene(device)
    state = _settled_state(model, 150)
    for response in RESPONSES:
        with test.subTest(response=response):
            solver = _solver(model, pgs_iterations=64, pgs_warmstart=True, articulated_contact_response=response)
            pipeline = _pipeline(model, warmstart=True)
            contacts = pipeline.contacts()
            state_0, state_1 = model.state(), model.state()
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_0, name), getattr(state, name))
            for _ in range(3):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
                state_0, state_1 = state_1, state_0
            solved = solver.propagation_impulses.numpy().copy()
            # Without iterations the step publishes the seeded impulses unchanged.
            solver.pgs_iterations = 0
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
            seeded = solver.propagation_impulses.numpy()
            test.assertGreater(float(np.abs(solved).max()), 0.0)
            np.testing.assert_allclose(seeded.sum(axis=1), solved.sum(axis=1), rtol=0.1)


def test_mimic_rows_hold_with_propagation(test, device):
    """Keep a mimic follower on its leader with the propagation responses."""
    for response in RESPONSES:
        with test.subTest(response=response):
            model = _scene(device, mimic=True)
            solver = _solver(model, pgs_iterations=32, articulated_contact_response=response)
            pipeline = _pipeline(model)
            contacts = pipeline.contacts()
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            for _ in range(60):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
                state_0, state_1 = state_1, state_0
            joint_q = state_0.joint_q.numpy()
            test.assertGreater(solver._mimic_count, 0)
            test.assertLess(abs(float(joint_q[2] - joint_q[1])), 5.0e-3)


class TestFeatherPGSPropagationCombinations(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (
    test_options_match_the_immediate_response_at_convergence,
    test_restitution_rebounds_on_propagation_rows,
    test_regularization_weights_reach_the_propagation_rows,
    test_warm_start_seeds_the_propagation_rows,
    test_mimic_rows_hold_with_propagation,
):
    add_function_test(TestFeatherPGSPropagationCombinations, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main()
