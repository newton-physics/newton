# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Predictor and integrator agreement for a spinning, translating free root.

The integrator adds the ``omega x v`` transport term to a free root's linear coordinate.
Every conversion between the root's velocity coordinate and its acceleration must use the
same convention:

* the velocity predictor ``v_hat = qd + qdd * dt`` feeds the contact, friction and
  velocity-limit rows, so without the term those rows are built against a velocity the
  integrator never realizes, off by ``dt * (omega x v)``;
* the post-solve conversion ``qdd = (v_out - qd) / dt`` decides what the integrator
  receives, so without the inverse term the realized velocity is not the solved one.

Contact-free runs cannot see a mismatch end to end because the two conversions cancel, so the
tests pin both conversions directly as well as through active contacts.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    integrate_generalized_joints,
    remove_free_root_transport_from_qdd,
    update_qdd_from_velocity,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 1.0 / 200.0
STEPS = 80
RADIUS = 0.1
# Spin transverse to the slide so omega x v points along +/-z, straight into the contact normal.
OMEGA_Y = 20.0
SLIDE_VX = 2.0


@wp.kernel
def _record_slider_step(
    body_q: wp.array[wp.transform],
    rigid_contact_count: wp.array[int],
    step: int,
    # outputs
    heights: wp.array[float],
    peak_contacts: wp.array[int],
):
    """Record one slider sample without a per-step device synchronization."""
    heights[step] = wp.transform_get_translation(body_q[0])[2]
    peak_contacts[0] = wp.max(peak_contacts[0], rigid_contact_count[0])


def _undamped(model):
    """Disable the per-body angular damping, so the free root keeps its spin."""
    model.rigid_body_angular_damping.zero_()
    return model


def _new_builder(**kwargs):
    builder = newton.ModelBuilder(**kwargs)
    SolverFeatherPGS.register_custom_attributes(builder)
    return builder


def _build_slider(device):
    """A frictionless sphere resting on a frictionless ground plane."""
    builder = _new_builder(gravity=(0.0, 0.0, -9.81))
    builder.default_shape_cfg.mu = 0.0
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=wp.mat33(0.004, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.004),
    )
    builder.add_shape_sphere(body, radius=RADIUS)
    builder.add_articulation([builder.add_joint_free(body)])
    builder.add_ground_plane()
    return _undamped(builder.finalize(device=device))


def _slide_heights(model, omega_y):
    """Slide the sphere at ``SLIDE_VX`` while spinning at ``omega_y``; return heights and peak contacts."""
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = (SLIDE_VX, 0.0, 0.0)
    joint_qd[3:6] = (0.0, omega_y, 0.0)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverFeatherPGS(model)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    control = model.control()
    heights = wp.empty(STEPS, dtype=wp.float32, device=model.device)
    peak_contacts = wp.zeros(1, dtype=wp.int32, device=model.device)
    for step in range(STEPS):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        wp.launch(
            _record_slider_step,
            dim=1,
            inputs=[state_0.body_q, contacts.rigid_contact_count, step],
            outputs=[heights, peak_contacts],
            device=model.device,
        )
    return heights.numpy(), int(peak_contacts.numpy()[0])


def _world_com(model, state):
    """World position of the root body's center of mass [m]."""
    com_local = model.body_com.numpy().astype(np.float64).reshape(-1, 3)[0]
    q = state.body_q.numpy().astype(np.float64).reshape(-1, 7)[0]
    rot = wp.quat(*(float(c) for c in q[3:7]))
    return np.asarray(q[0:3]) + np.asarray(wp.quat_rotate(rot, wp.vec3(*com_local)))


def test_frictionless_slide_is_spin_invariant(test, device):
    """Keep a frictionless sliding sphere's height trajectory independent of its spin.

    Without friction the spin exerts no force. A predictor missing the ``omega x v`` term feeds
    the normal rows a phantom vertical velocity of ``dt * omega_y * vx`` (0.2 m/s here) and the
    heights diverge by millimetres within the run.
    """
    still, contacts_still = _slide_heights(_build_slider(device), 0.0)
    spinning, contacts_spin = _slide_heights(_build_slider(device), OMEGA_Y)
    test.assertGreater(min(contacts_still, contacts_spin), 0, "no contacts were generated")
    for label, heights in (("still", still), ("spinning", spinning)):
        test.assertLess(float(np.abs(heights - RADIUS).max()), 2e-3, f"{label} run left the ground support band")
    divergence = float(np.abs(spinning - still).max())
    test.assertLess(divergence, 1e-4, f"spin changed a frictionless slide by {divergence} m")


def test_anchored_spinner_com_stays_put(test, device):
    """Keep the center of mass of a force-free spinner with an offset joint anchor in place.

    The free joint's coordinate tracks the child anchor frame, offset from the center of mass by
    ``child_xform``; the anchor must orbit the stationary center of mass.
    """
    configs = (
        ("plain-offset", wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), wp.vec3(0.0, 0.0, 0.0)),
        (
            "rotated-anchor-offset-com",
            wp.transform(wp.vec3(0.1, 0.02, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.7)),
            wp.vec3(0.03, -0.05, 0.02),
        ),
    )
    for label, child_xform, com in configs:
        with test.subTest(config=label):
            builder = _new_builder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(
                mass=1.0, com=com, inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.014)
            )
            builder.add_articulation([builder.add_joint_free(body, child_xform=child_xform)])
            model = _undamped(builder.finalize(device=device))
            state_0, state_1 = model.state(), model.state()
            joint_qd = state_0.joint_qd.numpy()
            joint_qd[3:6] = (0.0, 0.0, 10.0)
            state_0.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            start = _world_com(model, state_0)
            solver = SolverFeatherPGS(model)
            drift = 0.0
            for _ in range(STEPS):
                solver.step(state_0, state_1, model.control(), None, DT)
                state_0, state_1 = state_1, state_0
                drift = max(drift, float(np.linalg.norm(_world_com(model, state_0) - start)))
            test.assertLess(drift, 1e-4, f"COM of a force-free spinner drifted {drift} m")


def test_integrator_transport_identities(test, device):
    """Add ``omega x v * dt`` to a root free joint's linear coordinate and pass a descendant's through."""
    builder = _new_builder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    parent = builder.add_link(mass=1.0, inertia=inertia)
    child = builder.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
    root = builder.add_joint_free(parent)
    nested = builder.add_joint_free(child, parent=parent)
    builder.add_articulation([root, nested])
    model = builder.finalize(device=device)

    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[0:3] = (0.4, 0.0, 0.0)
    joint_qd[3:6] = (0.0, 0.0, 3.0)
    joint_qd[6:9] = (0.4, 0.0, 0.0)
    joint_qd[9:12] = (0.0, 0.0, 3.0)
    state.joint_qd.assign(joint_qd)
    q_new = wp.zeros_like(state.joint_q)
    qd_new = wp.zeros_like(state.joint_qd)
    wp.launch(
        integrate_generalized_joints,
        dim=model.joint_count,
        inputs=[
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            wp.zeros(model.joint_count, dtype=wp.int32, device=device),
            model.joint_dof_dim,
            model.body_com,
            model.joint_X_c,
            state.joint_q,
            state.joint_qd,
            wp.zeros_like(state.joint_qd),
            DT,
            wp.zeros(model.body_count, dtype=wp.float32, device=device),
        ],
        outputs=[q_new, qd_new],
        device=device,
    )
    out = qd_new.numpy()
    transport = np.cross((0.0, 0.0, 3.0), (0.4, 0.0, 0.0)) * DT
    np.testing.assert_allclose(out[0:3], np.array((0.4, 0.0, 0.0)) + transport, rtol=1e-6)
    np.testing.assert_allclose(out[3:6], (0.0, 0.0, 3.0), atol=1e-7)
    np.testing.assert_allclose(out[6:9], (0.4, 0.0, 0.0), atol=1e-7, err_msg="descendant gained transport")
    np.testing.assert_allclose(out[9:12], (0.0, 0.0, 3.0), atol=1e-7)


def test_predictor_and_qdd_transport_identities(test, device):
    """Apply the transport term in the predictor and remove it in the solved-velocity conversion."""
    model = _build_slider(device)
    solver = SolverFeatherPGS(model)
    test.assertEqual(solver._free_root_joint_count, 1)
    state_in, state_out = model.state(), model.state()
    state_aug = solver._prepare_augmented_state(state_in)

    qd_np = np.array((2.0, 0.0, 0.0, 0.0, 20.0, 0.0), dtype=np.float32)
    qdd_np = np.array((0.3, -0.4, 0.5, 0.1, -0.2, 0.25), dtype=np.float32)
    state_in.joint_qd.assign(qd_np)
    solver.qd_work.assign(qd_np)
    state_aug.joint_qdd.assign(qdd_np)
    solver._stage3_compute_v_hat(state_in, state_aug, DT, solver.qd_work)
    transport = np.cross(qd_np[3:6], qd_np[0:3])
    expected_v_hat = qd_np + qdd_np * DT
    expected_v_hat[0:3] += transport * DT
    np.testing.assert_allclose(solver.v_hat.numpy(), expected_v_hat, rtol=1e-6, atol=1e-7)

    v_out_np = np.array((1.8, 0.1, -0.15, 0.05, 19.5, 0.25), dtype=np.float32)
    solver.v_out.assign(v_out_np)
    wp.launch(
        update_qdd_from_velocity,
        dim=model.joint_dof_count,
        inputs=[state_in.joint_qd, solver._kinematic_dof_mask, 1.0 / DT],
        outputs=[solver.v_out, state_aug.joint_qdd],
        device=device,
    )
    wp.launch(
        remove_free_root_transport_from_qdd,
        dim=solver._free_root_joint_count,
        inputs=[solver._free_root_joint_indices, model.joint_qd_start, solver._kinematic_joint_mask, state_in.joint_qd],
        outputs=[state_aug.joint_qdd],
        device=device,
    )
    expected_qdd = (v_out_np - qd_np) / DT
    expected_qdd[0:3] -= transport
    np.testing.assert_allclose(state_aug.joint_qdd.numpy(), expected_qdd, rtol=1e-6, atol=1e-5)
    wp.launch(
        integrate_generalized_joints,
        dim=model.joint_count,
        inputs=[
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            solver._kinematic_joint_mask,
            model.joint_dof_dim,
            model.body_com,
            model.joint_X_c,
            state_in.joint_q,
            state_in.joint_qd,
            state_aug.joint_qdd,
            DT,
            solver.rigid_body_angular_damping,
        ],
        outputs=[state_out.joint_q, state_out.joint_qd],
        device=device,
    )
    np.testing.assert_allclose(state_out.joint_qd.numpy(), v_out_np, rtol=1e-6, atol=1e-6)


def test_compact_root_metadata_includes_standalone_joint(test, device):
    """Keep every world-rooted free joint, but no descendant, in the compact root launch."""
    builder = _new_builder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    root_a = builder.add_link(mass=1.0, inertia=inertia)
    child = builder.add_link(mass=1.0, inertia=inertia)
    root_b = builder.add_link(mass=1.0, inertia=inertia)
    joint_a = builder.add_joint_free(root_a)
    nested = builder.add_joint_free(child, parent=root_a)
    joint_b = builder.add_joint_free(root_b)
    builder.add_articulation([joint_a, nested])
    solver = SolverFeatherPGS(builder.finalize(device=device))
    np.testing.assert_array_equal(solver._free_root_joint_indices.numpy(), (joint_a, joint_b))


def test_captured_step_matches_uncaptured(test, device):
    """Reproduce the eager trajectory by replaying a captured collide-and-step graph."""
    heights_ref, _ = _slide_heights(_build_slider(device), OMEGA_Y)

    model = _build_slider(device)
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = (SLIDE_VX, 0.0, 0.0)
    joint_qd[3:6] = (0.0, OMEGA_Y, 0.0)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    solver = SolverFeatherPGS(model)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    control = model.control()

    def one_step():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        # Copy back instead of swapping so the captured pointers stay valid.
        wp.copy(state_0.body_q, state_1.body_q)
        wp.copy(state_0.body_qd, state_1.body_qd)
        wp.copy(state_0.joint_q, state_1.joint_q)
        wp.copy(state_0.joint_qd, state_1.joint_qd)

    one_step()  # warm-up, matches the reference run's first step
    with wp.ScopedCapture(device) as capture:
        one_step()  # capture records this step without executing it
    heights = []
    for _ in range(STEPS - 1):
        wp.capture_launch(capture.graph)
        heights.append(float(state_0.body_q.numpy().reshape(-1, 7)[0, 2]))
    np.testing.assert_allclose(heights, heights_ref[1:], atol=1e-6)


class TestFeatherPgsFreeRootPredictor(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (
    ("test_frictionless_slide_is_spin_invariant", test_frictionless_slide_is_spin_invariant),
    ("test_anchored_spinner_com_stays_put", test_anchored_spinner_com_stays_put),
    ("test_integrator_transport_identities", test_integrator_transport_identities),
    ("test_predictor_and_qdd_transport_identities", test_predictor_and_qdd_transport_identities),
    ("test_compact_root_metadata_includes_standalone_joint", test_compact_root_metadata_includes_standalone_joint),
    ("test_captured_step_matches_uncaptured", test_captured_step_matches_uncaptured),
):
    add_function_test(TestFeatherPgsFreeRootPredictor, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main()
