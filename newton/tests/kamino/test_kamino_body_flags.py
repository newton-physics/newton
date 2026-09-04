# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Kamino's rigid body flag support (KINEMATIC / PROXY).

Kamino treats a body as immovable when either its inverse mass and inverse
inertia are both zero, or when its flags include KINEMATIC or PROXY. This
module exercises the resulting behavior end-to-end:

- masking of Kamino-owned ``inv_m_i`` / ``inv_i_I_i`` arrays,
- joint and contact constraint culling between two immovable endpoints,
- integrator/PADMM stability with kinematic bodies (isolated and connected),
- ``notify_model_changed`` propagation and rejection of runtime flag flips,
- non-mutation of the underlying Newton model arrays.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim import BodyFlags
from newton._src.solvers.coupled.model_view import ModelView
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context

###
# Test helpers
###


def _make_builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverKamino.register_custom_attributes(builder)
    return builder


def _build_two_body_articulation(
    *,
    is_kinematic_parent: bool = False,
    density_parent: float | None = None,
    density_child: float | None = None,
    joint_kwargs: dict | None = None,
    add_fixed_root: bool = True,
) -> tuple[newton.ModelBuilder, int, int, int]:
    """Build a two-body articulation used across the body-flag test cases.

    Newton's builder requires that only *root* bodies (parent == -1) may carry
    the KINEMATIC flag, so only the parent side accepts ``is_kinematic=True``.
    Two-endpoint immovability at the joint level is exercised elsewhere via
    :class:`ModelView` and ``mark_proxy_bodies``.

    Returns ``(builder, parent, child, revolute_joint)``. If ``add_fixed_root``
    is True, ``parent`` is anchored to the world via a fixed joint before the
    revolute is added.
    """
    builder = _make_builder()
    parent_cfg = None
    child_cfg = None
    if density_parent is not None:
        parent_cfg = newton.ModelBuilder.ShapeConfig(density=density_parent)
    if density_child is not None:
        child_cfg = newton.ModelBuilder.ShapeConfig(density=density_child)

    parent = builder.add_link(
        label="parent",
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 2.0), wp.quat_identity(dtype=wp.float32)),
        is_kinematic=is_kinematic_parent,
    )
    child = builder.add_link(
        label="child",
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quat_identity(dtype=wp.float32)),
    )
    if parent_cfg is not None:
        builder.add_shape_box(body=parent, hx=0.1, hy=0.1, hz=0.2, cfg=parent_cfg)
    else:
        builder.add_shape_box(body=parent, hx=0.1, hy=0.1, hz=0.2)
    if child_cfg is not None:
        builder.add_shape_box(body=child, hx=0.1, hy=0.1, hz=0.75, cfg=child_cfg)
    else:
        builder.add_shape_box(body=child, hx=0.1, hy=0.1, hz=0.75)

    joints = []
    if add_fixed_root:
        joints.append(
            builder.add_joint_fixed(
                parent=-1,
                child=parent,
                parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 2.0), wp.quat_identity(dtype=wp.float32)),
            )
        )
    else:
        joints.append(
            builder.add_joint_free(
                parent=-1,
                child=parent,
                parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 2.0), wp.quat_identity(dtype=wp.float32)),
            )
        )
    revolute = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=newton.Axis.X,
        parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.2), wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.75), wp.quat_identity(dtype=wp.float32)),
        **(joint_kwargs or {}),
    )
    joints.append(revolute)
    builder.add_articulation(joints)
    return builder, parent, child, revolute


def _mark_proxy(model, indices: list[int], name: str = "test") -> ModelView:
    """Return a :class:`ModelView` with the given body indices marked PROXY."""
    view = ModelView(model, name)
    view.mark_proxy_bodies(wp.array(indices, dtype=int, device=model.device))
    return view


###
# Tests
###


class TestKaminoBodyFlags(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def test_padmm_preserves_immovable_joint_anchor(self):
        """Keep zero-inverse-mass joint anchors fixed under PADMM."""
        for integrator in ("euler", "moreau"):
            for sparse_jacobian in (False, True):
                with self.subTest(integrator=integrator, sparse_jacobian=sparse_jacobian):
                    builder, anchor, link, _ = _build_two_body_articulation(
                        density_parent=0.0,
                    )
                    builder.joint_q[-1] = 0.5 * np.pi
                    model = builder.finalize(device=self.default_device)
                    newton.eval_fk(model, model.joint_q, model.joint_qd, model)

                    state_in = model.state()
                    state_out = model.state()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
                    anchor_q_initial = state_in.body_q.numpy()[anchor].copy()
                    link_q_initial = state_in.body_q.numpy()[link].copy()

                    config = SolverKamino.Config(
                        dynamics_solver="padmm",
                        integrator=integrator,
                        sparse_jacobian=sparse_jacobian,
                        use_collision_detector=False,
                    )
                    solver = SolverKamino(model, config=config)
                    np.testing.assert_array_equal(solver._model_kamino.joints.num_kinematic_cts.numpy(), [0, 5])

                    for _ in range(16):
                        state_in.clear_forces()
                        solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)
                        state_in, state_out = state_out, state_in

                    body_q = state_in.body_q.numpy()
                    body_qd = state_in.body_qd.numpy()
                    np.testing.assert_array_equal(body_q[anchor], anchor_q_initial)
                    np.testing.assert_array_equal(body_qd[anchor], np.zeros(6, dtype=np.float32))
                    self.assertGreater(float(np.linalg.norm(body_q[link, :3] - link_q_initial[:3])), 1.0e-4)
                    self.assertGreater(float(abs(body_qd[link, 3])), 1.0e-2)
                    self.assertTrue(np.all(np.isfinite(body_q)))
                    self.assertTrue(np.all(np.isfinite(body_qd)))

    def test_sparse_dvi_masks_kinematic_inverse_mass(self):
        """Zero the Kamino-side inverse mass of a kinematic body."""
        builder, anchor, _, _ = _build_two_body_articulation(is_kinematic_parent=True, add_fixed_root=False)
        builder.joint_q[-1] = 0.5 * np.pi
        model = builder.finalize(device=self.default_device)

        config = SolverKamino.Config.from_model(
            model,
            dynamics_solver="dvi",
            sparse_dynamics=True,
            sparse_jacobian=True,
            use_collision_detector=False,
        )
        solver = SolverKamino(model, config=config)
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        # Kamino owns its own copy: pointers should differ but values must match
        # for dynamic bodies and be zero for the kinematic anchor.
        self.assertNotEqual(solver._model_kamino.bodies.inv_m_i.ptr, model.body_inv_mass.ptr)
        self.assertGreater(model.body_inv_mass.numpy()[anchor], 0.0)
        np.testing.assert_array_equal(solver._model_kamino.bodies.inv_m_i.numpy()[anchor], np.float32(0.0))
        np.testing.assert_array_equal(
            solver._model_kamino.bodies.inv_i_I_i.numpy()[anchor], np.zeros((3, 3), dtype=np.float32)
        )

        state_in.clear_forces()
        solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)
        self.assertTrue(np.all(np.isfinite(state_out.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))

    def test_culls_joint_without_dynamic_endpoint(self):
        """Cull all constraints between static and kinematic bodies."""
        builder = _make_builder()
        static_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        static_body = builder.add_link()
        kinematic_body = builder.add_link(is_kinematic=True)
        builder.add_shape_box(body=static_body, hx=0.1, hy=0.1, hz=0.1, cfg=static_cfg)
        builder.add_shape_box(body=kinematic_body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(
            parent=kinematic_body,
            child=static_body,
            axis=newton.Axis.X,
            damping=1.0,
            effort_limit=1.0,
            friction=1.0,
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=self.default_device)

        joints = ModelKamino.from_newton(model).joints
        np.testing.assert_array_equal(joints.num_kinematic_cts.numpy(), [0])
        np.testing.assert_array_equal(joints.num_dynamic_cts.numpy(), [0])
        np.testing.assert_array_equal(joints.num_friction_cts.numpy(), [0])
        np.testing.assert_array_equal(joints.num_bounded_cts.numpy(), [0])

    def test_two_kinematic_bodies_joint_culled(self):
        """Joint between two flag-immovable bodies (both PROXY) is fully culled.

        Newton's builder only allows KINEMATIC on root bodies, so exercising the
        two-endpoint case requires marking both bodies as PROXY on a
        :class:`ModelView` after finalization.
        """
        builder, parent, child, _ = _build_two_body_articulation(add_fixed_root=False)
        model = builder.finalize(device=self.default_device)
        view = _mark_proxy(model, [parent, child])
        solver = SolverKamino(view, config=SolverKamino.Config(use_collision_detector=False))
        kamino_model = solver._model_kamino

        self.assertGreater(float(model.body_inv_mass.numpy().max()), 0.0)
        np.testing.assert_array_equal(kamino_model.bodies.inv_m_i.numpy(), np.zeros(model.body_count, dtype=np.float32))
        np.testing.assert_array_equal(kamino_model.joints.num_kinematic_cts.numpy(), [0, 0])
        np.testing.assert_array_equal(kamino_model.joints.num_dynamic_cts.numpy(), [0, 0])
        np.testing.assert_array_equal(kamino_model.joints.num_bounded_cts.numpy(), [0, 0])
        np.testing.assert_array_equal(kamino_model.joints.num_bilateral_cts.numpy(), [0, 0])

        state_in = view.state()
        state_out = view.state()
        newton.eval_fk(view, view.joint_q, view.joint_qd, state_in)
        state_in.clear_forces()
        solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)
        self.assertTrue(np.all(np.isfinite(state_out.body_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.body_qd.numpy())))

    def test_kinematic_middle_of_chain(self):
        """dyn - immovable - dyn chain preserves both incident joints.

        The middle body is marked PROXY via :class:`ModelView` because Newton's
        builder disallows KINEMATIC on non-root bodies.
        """
        builder = _make_builder()
        a = builder.add_link(
            label="a",
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 2.0), wp.quat_identity(dtype=wp.float32)),
        )
        b = builder.add_link(
            label="b",
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.5), wp.quat_identity(dtype=wp.float32)),
        )
        c = builder.add_link(
            label="c",
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
        )
        for body in (a, b, c):
            builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.2)
        j_ab = builder.add_joint_revolute(
            parent=a,
            child=b,
            axis=newton.Axis.X,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.2), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.2), wp.quat_identity(dtype=wp.float32)),
        )
        j_bc = builder.add_joint_revolute(
            parent=b,
            child=c,
            axis=newton.Axis.X,
            parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.2), wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.2), wp.quat_identity(dtype=wp.float32)),
        )
        builder.add_articulation([j_ab, j_bc])
        model = builder.finalize(device=self.default_device)

        view = _mark_proxy(model, [b])
        solver = SolverKamino(view, config=SolverKamino.Config(use_collision_detector=False))
        kamino_model = solver._model_kamino
        num_kin = kamino_model.joints.num_kinematic_cts.numpy()
        self.assertGreater(int(num_kin[0]), 0)
        self.assertGreater(int(num_kin[1]), 0)

        state_in = view.state()
        state_out = view.state()
        newton.eval_fk(view, view.joint_q, view.joint_qd, state_in)
        b_initial = state_in.body_q.numpy()[b].copy()
        for _ in range(8):
            state_in.clear_forces()
            solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)
            state_in, state_out = state_out, state_in
        np.testing.assert_allclose(state_in.body_q.numpy()[b], b_initial, atol=1e-5)

    def test_armature_between_immovable_bodies_culled(self):
        """Body-level culling ignores joint-DoF armature."""
        builder, parent, child, _ = _build_two_body_articulation(add_fixed_root=False, joint_kwargs={"armature": 0.1})
        model = builder.finalize(device=self.default_device)
        view = _mark_proxy(model, [parent, child])
        joints = ModelKamino.from_newton(view).joints
        np.testing.assert_array_equal(joints.num_kinematic_cts.numpy(), [0, 0])
        np.testing.assert_array_equal(joints.num_dynamic_cts.numpy(), [0, 0])

    def test_damping_between_immovable_bodies_culled(self):
        """Body-level culling ignores joint-DoF damping."""
        builder, parent, child, _ = _build_two_body_articulation(add_fixed_root=False, joint_kwargs={"damping": 1.0})
        model = builder.finalize(device=self.default_device)
        view = _mark_proxy(model, [parent, child])
        joints = ModelKamino.from_newton(view).joints
        np.testing.assert_array_equal(joints.num_kinematic_cts.numpy(), [0, 0])
        np.testing.assert_array_equal(joints.num_dynamic_cts.numpy(), [0, 0])

    def test_position_limits_between_immovable_bodies_culled(self):
        """Position-limit slots are not reserved for two-immovable-endpoint joints.

        A limit fired on such a joint would produce a structurally-singular
        constraint row (both Jacobians contribute zero), so ``LimitsKamino``
        drops it at capacity time (nothing to allocate) and at detection time
        (kernel skips the joint), matching the joint-bilateral / contact
        culling policy.
        """
        # Sanity check: same joint on a movable chain reserves one limit slot.
        builder_ctrl, _, _, _ = _build_two_body_articulation(
            add_fixed_root=True, joint_kwargs={"limit_lower": -1.0, "limit_upper": 1.0}
        )
        model_ctrl = builder_ctrl.finalize(device=self.default_device)
        solver_ctrl = SolverKamino(model_ctrl, config=SolverKamino.Config(use_collision_detector=False))
        self.assertEqual(solver_ctrl._solver_kamino._limits.model_max_limits_host, 1)

        # Both endpoints immovable via PROXY: no slot should be reserved.
        builder, parent, child, _ = _build_two_body_articulation(
            add_fixed_root=False, joint_kwargs={"limit_lower": -1.0, "limit_upper": 1.0}
        )
        model = builder.finalize(device=self.default_device)
        view = _mark_proxy(model, [parent, child])
        solver = SolverKamino(view, config=SolverKamino.Config(use_collision_detector=False))
        self.assertEqual(solver._solver_kamino._limits.model_max_limits_host, 0)

    def test_contact_between_immovable_bodies_culled(self):
        """Contacts between two immovable bodies are culled by both entry points."""
        for use_collision_detector in (False, True):
            with self.subTest(use_collision_detector=use_collision_detector):
                builder = _make_builder()
                a = builder.add_link(
                    xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
                    is_kinematic=True,
                )
                b = builder.add_link(
                    xform=wp.transformf(wp.vec3f(0.05, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
                    is_kinematic=True,
                )
                builder.add_shape_box(body=a, hx=0.2, hy=0.2, hz=0.2)
                builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2)
                # Kamino requires at least one joint (state.joint_q must be non-None).
                # Anchor each kinematic body to the world with a fixed joint;
                # both endpoints of the fixed joint are still immovable, so the
                # joint is culled and this is a valid test setup.
                j_a = builder.add_joint_fixed(parent=-1, child=a)
                j_b = builder.add_joint_fixed(parent=-1, child=b)
                builder.add_articulation([j_a])
                builder.add_articulation([j_b])
                model = builder.finalize(device=self.default_device)

                solver = SolverKamino(
                    model,
                    config=SolverKamino.Config(use_collision_detector=use_collision_detector),
                )
                state_in = model.state()
                state_out = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
                state_in.clear_forces()
                if use_collision_detector:
                    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)
                else:
                    from newton._src.sim.collide import CollisionPipeline  # noqa: PLC0415

                    pipeline = CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state_in, contacts)
                    solver.step(state_in, state_out, control=None, contacts=contacts, dt=1.0e-3)

                active = int(solver._contacts_kamino.model_active_contacts.numpy()[0])
                self.assertEqual(active, 0)

    def test_kinematic_animation_via_body_q(self):
        """Prescribed body_q on a kinematic root flows through to state_out."""
        builder, parent, _, _ = _build_two_body_articulation(is_kinematic_parent=True)
        model = builder.finalize(device=self.default_device)
        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=False))
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        prescribed_q = state_in.body_q.numpy().copy()
        prescribed_q[parent, 0] += 0.5
        state_in.body_q.assign(prescribed_q)
        prescribed_qd = state_in.body_qd.numpy().copy()
        prescribed_qd[parent, 3] = 0.75
        state_in.body_qd.assign(prescribed_qd)
        state_in.clear_forces()

        solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)
        np.testing.assert_allclose(state_out.body_q.numpy()[parent], prescribed_q[parent], atol=1e-6)
        np.testing.assert_allclose(state_out.body_qd.numpy()[parent], prescribed_qd[parent], atol=1e-6)

    def test_kinematic_velocity_does_not_advance_pose(self):
        """Non-zero body_qd on a kinematic root does not advance its pose across a step."""
        builder, parent, _, _ = _build_two_body_articulation(is_kinematic_parent=True)
        model = builder.finalize(device=self.default_device)
        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=False))
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        body_q_initial = state_in.body_q.numpy().copy()
        qd = state_in.body_qd.numpy().copy()
        qd[parent, 3] = 1.0
        state_in.body_qd.assign(qd)
        state_in.clear_forces()
        solver.step(state_in, state_out, control=None, contacts=None, dt=1.0 / 60.0)
        np.testing.assert_allclose(state_out.body_q.numpy()[parent], body_q_initial[parent], atol=1e-6)

    def test_isolated_kinematic_body(self):
        """Kinematic root anchored to world: state passes through untouched."""
        builder = _make_builder()
        body = builder.add_link(is_kinematic=True)
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_fixed(parent=-1, child=body)
        builder.add_articulation([joint])
        model = builder.finalize(device=self.default_device)
        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=False))
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        state_in.clear_forces()
        solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)
        np.testing.assert_array_equal(state_out.body_q.numpy(), state_in.body_q.numpy())
        np.testing.assert_array_equal(state_out.body_qd.numpy(), state_in.body_qd.numpy())

    def test_all_kinematic_world(self):
        """World with only kinematic bodies steps without NaNs."""
        builder = _make_builder()
        bodies = []
        joints = []
        for i in range(3):
            b = builder.add_link(
                xform=wp.transformf(wp.vec3f(float(i), 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
                is_kinematic=True,
            )
            builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1)
            j = builder.add_joint_fixed(parent=-1, child=b)
            joints.append(j)
            bodies.append(b)
        for j in joints:
            builder.add_articulation([j])
        model = builder.finalize(device=self.default_device)
        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=False))
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        state_in.clear_forces()
        solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)
        np.testing.assert_array_equal(state_out.body_q.numpy(), state_in.body_q.numpy())
        np.testing.assert_array_equal(state_out.body_qd.numpy(), state_in.body_qd.numpy())

    def test_proxy_body_treated_as_kinematic(self):
        """A PROXY body (via ModelView) masks inv-mass and culls incident joints."""
        builder, parent, child, _ = _build_two_body_articulation(add_fixed_root=False)
        model = builder.finalize(device=self.default_device)
        view = _mark_proxy(model, [parent, child])
        solver = SolverKamino(view, config=SolverKamino.Config(use_collision_detector=False))
        kamino_model = solver._model_kamino
        np.testing.assert_array_equal(kamino_model.bodies.inv_m_i.numpy(), np.zeros(model.body_count, dtype=np.float32))
        np.testing.assert_array_equal(kamino_model.joints.num_kinematic_cts.numpy(), [0, 0])
        # Base Newton model is unmodified.
        self.assertGreater(float(model.body_inv_mass.numpy().max()), 0.0)
        self.assertEqual(int((model.body_flags.numpy() & int(BodyFlags.PROXY)).sum()), 0)

    def test_newton_arrays_not_mutated(self):
        """SolverKamino must not mutate Newton's body_inv_mass / body_inv_inertia."""
        builder, _, _, _ = _build_two_body_articulation(is_kinematic_parent=True)
        model = builder.finalize(device=self.default_device)
        inv_mass_before = model.body_inv_mass.numpy().copy()
        inv_inertia_before = model.body_inv_inertia.numpy().copy()
        flags_before = model.body_flags.numpy().copy()

        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=False))
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        for _ in range(4):
            state_in.clear_forces()
            solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-3)
            state_in, state_out = state_out, state_in

        np.testing.assert_array_equal(model.body_inv_mass.numpy(), inv_mass_before)
        np.testing.assert_array_equal(model.body_inv_inertia.numpy(), inv_inertia_before)
        np.testing.assert_array_equal(model.body_flags.numpy(), flags_before)

    def test_culls_joint_without_dynamic_endpoint_all_joint_types(self):
        """Body-level culling holds across every Kamino-supported joint type."""
        joint_specs = [
            ("revolute", "add_joint_revolute", {"axis": newton.Axis.X}),
            ("prismatic", "add_joint_prismatic", {"axis": newton.Axis.X}),
            ("fixed", "add_joint_fixed", {}),
            ("ball", "add_joint_ball", {}),
            ("d6", "add_joint_d6", {}),
        ]
        for name, method, kwargs in joint_specs:
            with self.subTest(joint=name):
                builder = _make_builder()
                static_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
                static_body = builder.add_link()
                kinematic_body = builder.add_link(is_kinematic=True)
                builder.add_shape_box(body=static_body, hx=0.1, hy=0.1, hz=0.1, cfg=static_cfg)
                builder.add_shape_box(body=kinematic_body, hx=0.1, hy=0.1, hz=0.1)
                joint = getattr(builder, method)(parent=kinematic_body, child=static_body, **kwargs)
                builder.add_articulation([joint])
                model = builder.finalize(device=self.default_device)
                joints = ModelKamino.from_newton(model).joints
                np.testing.assert_array_equal(joints.num_kinematic_cts.numpy(), [0])
                np.testing.assert_array_equal(joints.num_dynamic_cts.numpy(), [0])
                np.testing.assert_array_equal(joints.num_friction_cts.numpy(), [0])
                np.testing.assert_array_equal(joints.num_bounded_cts.numpy(), [0])


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
