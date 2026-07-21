# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the coupled solver prototype."""

import unittest
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.geometry.flags import ParticleFlags, ShapeFlags
from newton._src.solvers.coupled.interface import CouplingEndpointKind, CouplingInterface
from newton._src.solvers.coupled.solver_coupled import _filter_soft_contacts_global_shape_ids_kernel
from newton._src.solvers.mujoco.equality import _add_equality_constraint
from newton.solvers import (
    SolverBase,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverVBD,
    SolverXPBD,
)
from newton.solvers.experimental.coupled import (
    ModelView,
    SolverCoupled,
    SolverCoupledProxy,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def _write_proxy_body_wrench_kernel(
    body_local_to_proxy_global: wp.array[int],
    out_body_f: wp.array[wp.spatial_vector],
):
    local_body = wp.tid()
    global_body = body_local_to_proxy_global[local_body]
    if global_body >= 0:
        out_body_f[global_body] = wp.spatial_vector(wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0))


@wp.kernel(enable_backward=False)
def _kick_proxy_particle_kernel(particle_qd: wp.array[wp.vec3]):
    particle_qd[0] = particle_qd[0] + wp.vec3(0.0, 2.0, 0.0)


@wp.kernel(enable_backward=False)
def _write_proxy_particle_force_kernel(
    particle_local_to_proxy_global: wp.array[int],
    out_particle_f: wp.array[wp.vec3],
):
    local_particle = wp.tid()
    global_particle = particle_local_to_proxy_global[local_particle]
    if global_particle >= 0:
        out_particle_f[global_particle] = wp.vec3(0.0, 7.0, 0.0)


class _BodyForceRecordingSolver(SolverBase, CouplingInterface):
    """Test solver that records body forces and otherwise copies state."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.input_body_f = []
        self.instances.append(self)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        self.input_body_f.append(state_in.body_f.numpy().copy())
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)


class _ParticleForceRecordingSolver(SolverBase, CouplingInterface):
    """Test solver that records particle forces and otherwise copies state."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.input_particle_f = []
        self.instances.append(self)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        self.input_particle_f.append(state_in.particle_f.numpy().copy())
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)


class _ControlRecordingSolver(SolverBase, CouplingInterface):
    """Test solver that records entry-local control arrays."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.joint_f = []
        self.joint_target_q = []
        self.joint_target_qd = []
        self.custom_gain = []
        self.instances.append(self)

    def step(self, state_in, state_out, control, contacts, dt):
        del contacts, dt
        self.joint_f.append(None if control is None or control.joint_f is None else control.joint_f.numpy().copy())
        self.joint_target_q.append(
            None if control is None or control.joint_target_q is None else control.joint_target_q.numpy().copy()
        )
        self.joint_target_qd.append(
            None if control is None or control.joint_target_qd is None else control.joint_target_qd.numpy().copy()
        )
        gain = None if control is None else getattr(control, "gain", None)
        self.custom_gain.append(None if gain is None else gain.numpy().copy())
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
        if state_in.joint_q is not None and state_out.joint_q is not None:
            wp.copy(state_out.joint_q, state_in.joint_q)
            wp.copy(state_out.joint_qd, state_in.joint_qd)


class _InPlaceRecordingParticleSolver(SolverBase, CouplingInterface):
    """Test solver that records whether it was stepped in-place."""

    instances: ClassVar[dict[str, "_InPlaceRecordingParticleSolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.in_place_calls = []
        self.dt_values = []
        self.instances[model.name] = self

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts
        self.in_place_calls.append(state_in is state_out)
        self.dt_values.append(dt)
        if state_in is not state_out:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
        wp.launch(_kick_proxy_particle_kernel, dim=1, inputs=[state_out.particle_qd], device=self.model.device)


class _ProxyParticleKickSolver(SolverBase, CouplingInterface):
    """Destination test solver that applies a fixed impulse to proxy particle 0."""

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)
        wp.launch(_kick_proxy_particle_kernel, dim=1, inputs=[state_out.particle_qd], device=self.model.device)


class _ProxyParticleHookSolver(SolverBase, CouplingInterface):
    """Destination test solver that exposes particle proxy rewind/harvest hooks."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.rewind_calls = 0
        self.harvest_calls = 0
        self.instances.append(self)

    def coupling_rewind_proxy_particle(
        self,
        particle_local_to_proxy_global,
        state,
        coupling_forces,
        particle_gravity_acceleration,
        dt,
    ):
        del particle_local_to_proxy_global, state, coupling_forces, particle_gravity_acceleration, dt
        self.rewind_calls += 1

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global,
        out_particle_f,
        *,
        particle_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        del particle_qd_before, state, state_out, contacts, dt
        self.harvest_calls += 1
        wp.launch(
            _write_proxy_particle_force_kernel,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[particle_local_to_proxy_global, out_particle_f],
            device=self.model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)


class _ZeroingProxyParticleHookSolver(_ProxyParticleHookSolver):
    """Destination test solver that clears proxy particle feedback before writing."""

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global,
        out_particle_f,
        *,
        particle_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        out_particle_f.zero_()
        super().coupling_harvest_proxy_particle_forces(
            particle_local_to_proxy_global,
            out_particle_f,
            particle_qd_before=particle_qd_before,
            state=state,
            state_out=state_out,
            contacts=contacts,
            dt=dt,
        )


class _ProxyBodyHookSolver(SolverBase, CouplingInterface):
    """Destination test solver that writes proxy-indexed body feedback."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.harvest_calls = 0
        self.instances.append(self)

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global,
        out_body_f,
        *,
        body_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        del body_qd_before, state, state_out, contacts, dt
        self.harvest_calls += 1
        wp.launch(
            _write_proxy_body_wrench_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[body_local_to_proxy_global, out_body_f],
            device=self.model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)


class _AffineBodyForceSourceSolver(SolverBase, CouplingInterface):
    """Map the input body-force x component to output linear velocity."""

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        body_qd = state_in.body_qd.numpy().copy()
        body_qd[:, 0] = state_in.body_f.numpy()[:, 0]
        state_out.body_qd.assign(body_qd)


class _AffineProxyBodyFeedbackSolver(SolverBase, CouplingInterface):
    """Return the scalar affine feedback map H(x) = -2x + 1."""

    def coupling_rewind_proxy_body(
        self,
        body_local_to_proxy_global,
        state,
        coupling_forces,
        body_gravity_acceleration,
        dt,
    ):
        del body_local_to_proxy_global, state, coupling_forces, body_gravity_acceleration, dt

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global,
        out_body_f,
        *,
        body_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        del state, state_out, contacts, dt
        proxy_ids = body_local_to_proxy_global.numpy()
        velocity = body_qd_before.numpy()
        force = np.zeros_like(out_body_f.numpy())
        for local_body, proxy_id in enumerate(proxy_ids):
            if proxy_id >= 0:
                force[proxy_id, 0] = -2.0 * velocity[local_body, 0] + 1.0
        out_body_f.assign(force)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)


class _StepCountingCopySolver(SolverBase, CouplingInterface):
    """Test solver that records how many times it is stepped."""

    instances: ClassVar[dict[str, "_StepCountingCopySolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.step_count = 0
        self.dt_values = []
        self.model_notify_flags = []
        self.instances[model.name] = self

    def notify_model_changed(self, flags: int) -> None:
        self.model_notify_flags.append(int(flags))

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts
        self.step_count += 1
        self.dt_values.append(dt)
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
        if state_in.particle_q is not None and state_out.particle_q is not None:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)


class _ContactRecordingCopySolver(_StepCountingCopySolver):
    """Copy solver that records rigid contact shape ids seen by step()."""

    instances: ClassVar[dict[str, "_ContactRecordingCopySolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.rigid_shape0_steps = []
        self.rigid_shape1_steps = []
        self.step_contacts = []

    def step(self, state_in, state_out, control, contacts, dt):
        self.step_contacts.append(contacts)
        if contacts is not None and contacts.rigid_contact_count is not None:
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
            self.rigid_shape0_steps.append(contacts.rigid_contact_shape0.numpy()[:contact_count].copy())
            self.rigid_shape1_steps.append(contacts.rigid_contact_shape1.numpy()[:contact_count].copy())
        super().step(state_in, state_out, control, contacts, dt)


class _ContactRecordingBodyHarvestSolver(_ContactRecordingCopySolver):
    """Contact-recording solver with custom body proxy contact hooks."""

    instances: ClassVar[dict[str, "_ContactRecordingBodyHarvestSolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.harvest_contacts = []

    def coupling_prepare_proxy_contacts(self, state, contacts, *, contacts_freshly_detected=False):
        del state, contacts_freshly_detected
        return contacts

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global,
        out_body_f,
        *,
        body_qd_before,
        state,
        state_out,
        contacts,
        dt,
    ):
        del body_local_to_proxy_global, out_body_f, body_qd_before, state, state_out, dt
        self.harvest_contacts.append(contacts)


class _FakeProxyCollisionPipeline:
    """Minimal collision pipeline used to test proxy-coupler scheduling."""

    def __init__(self, device, contacts=None):
        self.contacts_obj = contacts if contacts is not None else newton.Contacts(0, 0, device=device)
        self.contacts_calls = 0
        self.collide_calls = 0

    def contacts(self):
        self.contacts_calls += 1
        return self.contacts_obj

    def collide(self, state, contacts):
        del state
        self.collide_calls += 1
        self.last_contacts = contacts


class TestModelView(unittest.TestCase):
    """Test ModelView attribute delegation and overrides."""

    def setUp(self):
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=2.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=0, radius=0.1)
        builder.add_shape_sphere(body=1, radius=0.2)
        self.model = builder.finalize(device="cpu")

    def test_fallback_to_parent(self):
        """Unoverridden attributes should come from the parent model."""
        view = ModelView(self.model, "test")
        self.assertEqual(view.body_count, 2)
        self.assertIs(view.body_q, self.model.body_q)
        self.assertEqual(view.device, self.model.device)

    def test_override(self):
        """Overridden attributes should take precedence."""
        view = ModelView(self.model, "test")
        new_mass = wp.zeros(2, dtype=float, device="cpu")
        view.body_inv_mass = new_mass

        self.assertIs(view.body_inv_mass, new_mass)
        # Parent unchanged
        self.assertIsNot(self.model.body_inv_mass, new_mass)

    def test_override_accepts_set_subclass_parent(self):
        """Set overrides should accept a native set when the parent uses a set subclass."""
        view = ModelView(self.model, "test")
        filters = set(self.model.shape_collision_filter_pairs)

        view.shape_collision_filter_pairs = filters

        self.assertIs(view.shape_collision_filter_pairs, filters)

    def test_count_override_slices_frequency_arrays(self):
        """Frequency-matched arrays should follow view-local counts."""
        view = ModelView(self.model, "test")
        view.body_count = 1
        view.shape_count = 1

        self.assertEqual(view.body_mass.shape[0], 1)
        self.assertEqual(view.body_inv_mass.shape[0], 1)
        self.assertEqual(view.shape_flags.shape[0], 1)
        self.assertEqual(self.model.body_mass.shape[0], 2)

    def test_zero_count_override_exposes_empty_frequency_arrays(self):
        """Zero-count views should expose empty arrays, not parent arrays."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")
        view = ModelView(model, "test")
        view.particle_count = 0

        self.assertEqual(view.particle_mass.shape[0], 0)
        self.assertEqual(view.particle_inv_mass.shape[0], 0)
        self.assertEqual(model.particle_mass.shape[0], 1)

    def test_disable_body_dynamics(self):
        """disable_body_dynamics should zero inverse inertia without changing flags."""
        view = ModelView(self.model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        view.disable_body_dynamics(indices)

        mass = view.body_mass.numpy()
        inertia = view.body_inertia.numpy()
        inv_mass = view.body_inv_mass.numpy()
        inv_inertia = view.body_inv_inertia.numpy()
        flags = view.body_flags.numpy()
        parent_flags = self.model.body_flags.numpy()
        dynamic = int(newton.BodyFlags.DYNAMIC)
        kinematic = int(newton.BodyFlags.KINEMATIC)
        # Body 0 should be unchanged (non-zero)
        self.assertGreater(mass[0], 0.0)
        self.assertGreater(inv_mass[0], 0.0)
        self.assertNotEqual(flags[0] & dynamic, 0)
        self.assertEqual(flags[0] & kinematic, 0)
        # Body 1 should keep forward inertial metadata but become immovable.
        self.assertEqual(mass[1], self.model.body_mass.numpy()[1])
        self.assertEqual(inv_mass[1], 0.0)
        np.testing.assert_allclose(inertia[1], self.model.body_inertia.numpy()[1])
        np.testing.assert_allclose(inv_inertia[1], np.zeros((3, 3)))
        self.assertNotEqual(flags[1] & dynamic, 0)
        self.assertEqual(flags[1] & kinematic, 0)
        self.assertNotEqual(parent_flags[1] & dynamic, 0)
        self.assertEqual(parent_flags[1] & kinematic, 0)

    def test_disable_joints_rewrites_cable_type_in_view(self):
        """disable_joints should expose disabled cable joints as D6 in the view."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        child = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_cable(
            parent=parent,
            child=child,
            parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
        )
        model = builder.finalize(device="cpu")
        view = ModelView(model, "test")

        view.disable_joints(wp.array([joint], dtype=int, device="cpu"))

        self.assertFalse(bool(view.joint_enabled.numpy()[joint]))
        self.assertEqual(int(view.joint_type.numpy()[joint]), int(newton.JointType.D6))
        self.assertEqual(int(model.joint_type.numpy()[joint]), int(newton.JointType.CABLE))
        np.testing.assert_array_equal(view.joint_dof_dim.numpy()[joint], model.joint_dof_dim.numpy()[joint])

    def test_zero_particle_mass(self):
        """zero_particle_mass should zero forward and inverse mass arrays."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=(0.1, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0)
        model = builder.finalize(device="cpu")
        view = ModelView(model, "test")

        view.zero_particle_mass(wp.array([1], dtype=int, device="cpu"))

        np.testing.assert_allclose(view.particle_mass.numpy(), [1.0, 0.0])
        np.testing.assert_allclose(view.particle_inv_mass.numpy(), [1.0, 0.0])
        np.testing.assert_allclose(model.particle_mass.numpy(), [1.0, 2.0])

    def test_set_body_inertial_properties(self):
        """set_body_inertial_properties should replace mass and full inertia."""
        view = ModelView(self.model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        target_mass = wp.array([4.0], dtype=float, device="cpu")
        target_inertia_np = np.array([[[2.0, 0.25, 0.0], [0.25, 3.0, 0.5], [0.0, 0.5, 5.0]]])
        target_inertia = wp.array(target_inertia_np, dtype=wp.mat33, device="cpu")

        view.set_body_inertial_properties(indices, target_mass, target_inertia)

        np.testing.assert_allclose(view.body_mass.numpy()[1], 4.0)
        np.testing.assert_allclose(view.body_inv_mass.numpy()[1], 0.25)
        np.testing.assert_allclose(view.body_inertia.numpy()[1], target_inertia_np[0])
        np.testing.assert_allclose(view.body_inv_inertia.numpy()[1], np.linalg.inv(target_inertia_np[0]), rtol=1.0e-6)

    def test_mark_proxy_bodies(self):
        """mark_proxy_bodies should mark only the view-local body flags."""
        view = ModelView(self.model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        view.mark_proxy_bodies(indices)

        view_flags = view.body_flags.numpy()
        parent_flags = self.model.body_flags.numpy()
        self.assertEqual(view_flags[0] & int(newton.BodyFlags.PROXY), 0)
        self.assertNotEqual(view_flags[1] & int(newton.BodyFlags.PROXY), 0)
        self.assertEqual(parent_flags[1] & int(newton.BodyFlags.PROXY), 0)

    def test_disable_particles(self):
        """disable_particles should clear only view-local active flags."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=(0.1, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        view = ModelView(model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        view.disable_particles(indices)

        active = int(newton.ParticleFlags.ACTIVE)
        view_flags = view.particle_flags.numpy()
        parent_flags = model.particle_flags.numpy()
        self.assertNotEqual(view_flags[0] & active, 0)
        self.assertEqual(view_flags[1] & active, 0)
        self.assertNotEqual(parent_flags[1] & active, 0)

    def test_state_creation(self):
        """view.state() should create a valid State."""
        view = ModelView(self.model, "test")
        state = view.state()
        self.assertEqual(state.body_count, 2)

    def test_state_creation_uses_view_overrides(self):
        """view.state() should clone state-relevant view-local arrays."""
        view = ModelView(self.model, "test")
        body_qd = self.model.body_qd.numpy()
        body_qd[1, 0] = 3.0
        view.body_qd = wp.array(body_qd, dtype=wp.spatial_vector, device="cpu")

        state = view.state()

        np.testing.assert_allclose(state.body_qd.numpy()[1, 0], 3.0)
        self.assertIsNot(state.body_qd, view.body_qd)
        np.testing.assert_allclose(state.body_f.numpy(), np.zeros_like(body_qd))

    def test_state_creation_respects_view_count_overrides(self):
        """view.state() should size state arrays from view-local counts."""
        self.model.request_state_attributes("body_qdd", "body_parent_f")
        view = ModelView(self.model, "test")
        view.body_count = 1

        state = view.state()

        self.assertEqual(state.body_count, 1)
        self.assertEqual(state.body_qd.shape[0], 1)
        self.assertEqual(state.body_f.shape[0], 1)
        self.assertEqual(state.body_qdd.shape[0], 1)
        self.assertEqual(state.body_parent_f.shape[0], 1)

    def test_state_creation_respects_view_zero_count(self):
        """view.state() should clear state fields hidden by view-local counts."""
        view = ModelView(self.model, "test")
        view.body_count = 0

        state = view.state()

        self.assertIsNone(state.body_q)
        self.assertIsNone(state.body_qd)
        self.assertIsNone(state.body_f)

    def test_set_body_mass_rejects_static_to_dynamic_without_inertia(self):
        """set_body_mass should not create finite mass with zero inertia."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=0.0, inertia=wp.mat33(0.0))
        model = builder.finalize(device="cpu")
        view = ModelView(model, "test")

        with self.assertRaisesRegex(ValueError, "set_body_inertial_properties"):
            view.set_body_mass(wp.array([0], dtype=int, device="cpu"), wp.array([1.0], dtype=float, device="cpu"))

    def test_setattr_rejects_unknown_name(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(AttributeError, "no such attribute"):
            view.not_a_model_field = 0

    def test_setattr_rejects_dtype_mismatch(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(TypeError, "body_inv_mass"):
            view.body_inv_mass = wp.zeros(2, dtype=int, device="cpu")

    def test_setattr_rejects_ndim_mismatch(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(TypeError, "body_inv_mass"):
            view.body_inv_mass = wp.zeros((2, 2), dtype=float, device="cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_setattr_rejects_device_mismatch(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(TypeError, "body_inv_mass"):
            view.body_inv_mass = wp.zeros(2, dtype=float, device="cuda")

    def test_setattr_rejects_wrong_python_type(self):
        view = ModelView(self.model, "test")
        with self.assertRaisesRegex(TypeError, "body_count"):
            view.body_count = "two"

    def test_setattr_allows_none_when_parent_is_array(self):
        view = ModelView(self.model, "test")
        view.body_inv_mass = None
        self.assertIsNone(view.body_inv_mass)


class TestSolverCoupledBasic(unittest.TestCase):
    """Test SolverCoupled with two SemiImplicit solvers (simplest case)."""

    def setUp(self):
        builder = newton.ModelBuilder()

        # Two bodies: body 0 owned by solver A, body 1 owned by solver B
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=2.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=0, radius=0.1)
        builder.add_shape_sphere(body=1, radius=0.2)

        self.model = builder.finalize(device="cpu")

    def test_rejects_solver_without_coupling_interface_during_construction(self):
        with self.assertRaisesRegex(TypeError, "cannot participate in a coupled simulation"):
            SolverCoupled(
                model=self.model,
                entries=[SolverCoupled.Entry(name="unsupported", solver=SolverBase, bodies=[0])],
            )

    def test_configure_view_applies_after_compaction(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        cloth_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        soft_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        positions = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.0, 2.0),
        )
        for position in positions:
            builder.add_particle(pos=wp.vec3(*position), vel=wp.vec3(0.0), mass=1.0)

        builder.add_triangle(i=0, j=1, k=2)
        builder.add_triangle(i=3, j=2, k=1)
        builder.add_edge(i=0, j=3, k=1, l=2)
        builder.add_tetrahedron(i=4, j=5, k=6, l=7)
        model = builder.finalize(device="cpu")
        configured_body_counts = {}
        internal_body_counts = {}

        class _RecordingCoupledSolver(SolverCoupled):
            def _customize_compact_view(self, view: ModelView) -> None:
                internal_body_counts[view.name] = view.body_count

        def configure_soft_view(view: ModelView) -> None:
            configured_body_counts[view.name] = view.body_count
            view.tri_count = 0
            view.edge_count = 0

        def configure_cloth_view(view: ModelView) -> None:
            configured_body_counts[view.name] = view.body_count
            view.tet_count = 0

        coupled = _RecordingCoupledSolver(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="soft",
                    solver=_StepCountingCopySolver,
                    bodies=[soft_body],
                    particles=[4, 5, 6, 7],
                    configure_view=configure_soft_view,
                ),
                SolverCoupled.Entry(
                    name="cloth",
                    solver=_StepCountingCopySolver,
                    bodies=[cloth_body],
                    particles=[0, 1, 2, 3],
                    configure_view=configure_cloth_view,
                ),
            ],
        )

        self.assertEqual(internal_body_counts, {"soft": 1, "cloth": 1})
        self.assertEqual(configured_body_counts, {"soft": 1, "cloth": 1})

        soft_view = coupled.view("soft")
        self.assertEqual(soft_view.tri_count, 0)
        self.assertEqual(soft_view.edge_count, 0)
        self.assertEqual(soft_view.tet_count, 1)
        self.assertEqual(soft_view.tri_indices.shape[0], 0)
        self.assertEqual(soft_view.edge_indices.shape[0], 0)
        self.assertEqual(soft_view.tet_indices.shape[0], 1)

        cloth_view = coupled.view("cloth")
        self.assertEqual(cloth_view.tri_count, 2)
        self.assertEqual(cloth_view.edge_count, 1)
        self.assertEqual(cloth_view.tet_count, 0)
        self.assertEqual(cloth_view.tri_indices.shape[0], 2)
        self.assertEqual(cloth_view.edge_indices.shape[0], 1)
        self.assertEqual(cloth_view.tet_indices.shape[0], 0)

    def test_compaction_fallback_reports_reason(self):
        builder = newton.ModelBuilder()
        parent = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        child = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_revolute(parent=parent, child=child, axis=(0.0, 0.0, 1.0))
        model = builder.finalize(device="cpu")

        with self.assertLogs("newton._src.solvers.coupled.solver_coupled", level="INFO") as logs:
            coupled = SolverCoupled(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="child",
                        solver=SolverSemiImplicit,
                        bodies=[child],
                        joints=[joint],
                    )
                ],
            )

        self.assertRegex("\n".join(logs.output), r"entry 'child'.*joint.*outside.*full model layout")
        self.assertEqual(coupled.view("child").body_count, model.body_count)

    def test_entry_control_arrays_are_mapped_to_local_dofs(self):
        """Entry solvers should receive control arrays in their local DOF namespace."""
        _ControlRecordingSolver.instances.clear()
        builder = newton.ModelBuilder()
        body_a = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        body_b = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint_a = builder.add_joint_revolute(parent=-1, child=body_a, axis=(0.0, 0.0, 1.0))
        joint_b = builder.add_joint_revolute(parent=-1, child=body_b, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([joint_a])
        builder.add_articulation([joint_b])
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="A", solver=_ControlRecordingSolver, bodies=[body_a], joints=[joint_a]),
                SolverCoupled.Entry(name="B", solver=_ControlRecordingSolver, bodies=[body_b], joints=[joint_b]),
            ],
        )
        control = model.control()
        control.joint_f.assign(np.array([3.0, 7.0], dtype=np.float32))
        control.joint_target_q.assign(np.array([11.0, 13.0], dtype=np.float32))

        coupled.step(model.state(), model.state(), control, contacts=None, dt=1.0 / 60.0)

        solver_a, solver_b = _ControlRecordingSolver.instances
        np.testing.assert_array_equal(solver_a.joint_f[0], np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(solver_b.joint_f[0], np.array([7.0], dtype=np.float32))
        np.testing.assert_array_equal(solver_a.joint_target_q[0], np.array([11.0], dtype=np.float32))
        np.testing.assert_array_equal(solver_b.joint_target_q[0], np.array([13.0], dtype=np.float32))

    def test_compacted_joint_targets_use_local_layout(self):
        """Joint targets and their derived starts should use compact layout."""
        builder = newton.ModelBuilder()
        free_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        free_joint = builder.add_joint_free(child=free_body)
        builder.add_articulation([free_joint])
        first_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        first_joint = builder.add_joint_revolute(parent=-1, child=first_body, axis=(0.0, 0.0, 1.0))
        second_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        second_joint = builder.add_joint_revolute(
            parent=first_body,
            child=second_body,
            axis=(0.0, 1.0, 0.0),
        )
        builder.add_articulation([first_joint, second_joint])
        model = builder.finalize(device="cpu")
        model.joint_target_q.assign(np.arange(model.joint_dof_count, dtype=np.float32))

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="revolute",
                    solver=SolverSemiImplicit,
                    bodies=[first_body, second_body],
                    joints=[first_joint, second_joint],
                )
            ],
        )
        view = coupled.view("revolute")

        np.testing.assert_array_equal(view.joint_ancestor.numpy(), [-1, 0])
        np.testing.assert_array_equal(view.joint_target_q_start.numpy(), [0, 1, 2])
        np.testing.assert_array_equal(view.joint_target_q.numpy(), [6.0, 7.0])

        model.joint_target_q.assign(10.0 + np.arange(model.joint_dof_count, dtype=np.float32))
        model.joint_target_ke.assign(100.0 + np.arange(model.joint_dof_count, dtype=np.float32))
        coupled.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_array_equal(view.joint_target_q.numpy(), [16.0, 17.0])
        np.testing.assert_array_equal(view.joint_target_ke.numpy(), [106.0, 107.0])

        target_pos_spec = model._attribute_spec("joint_target_pos")
        self.assertTrue(target_pos_spec.deprecated)
        self.assertEqual(target_pos_spec.alias_of, "joint_target_q")
        with self.assertWarnsRegex(DeprecationWarning, "Model.joint_target_pos"):
            legacy_target_pos = view.joint_target_pos
        np.testing.assert_array_equal(legacy_target_pos.numpy(), [16.0, 17.0])

        legacy_override = wp.array([21.0, 22.0], dtype=float, device=model.device)
        with self.assertWarnsRegex(DeprecationWarning, "Model.joint_target_pos"):
            view.joint_target_pos = legacy_override
        np.testing.assert_array_equal(view.joint_target_q.numpy(), [21.0, 22.0])

    def test_custom_control_arrays_are_mapped_to_entries(self):
        """Custom CONTROL attributes should follow their compact frequency map."""
        _ControlRecordingSolver.instances.clear()
        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="gain",
                frequency=newton.Model.AttributeFrequency.BODY,
                assignment=newton.Model.AttributeAssignment.CONTROL,
                dtype=wp.float32,
            )
        )
        body_a = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"gain": 1.0},
        )
        body_b = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"gain": 2.0},
        )
        model = builder.finalize(device="cpu")
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="A", solver=_ControlRecordingSolver, bodies=[body_a]),
                SolverCoupled.Entry(name="B", solver=_ControlRecordingSolver, bodies=[body_b]),
            ],
        )
        control = model.control()
        control.gain.assign(np.array([3.0, 7.0], dtype=np.float32))

        coupled.step(model.state(), model.state(), control, contacts=None, dt=1.0 / 60.0)

        solver_a, solver_b = _ControlRecordingSolver.instances
        np.testing.assert_array_equal(solver_a.custom_gain[0], [3.0])
        np.testing.assert_array_equal(solver_b.custom_gain[0], [7.0])

    def test_notify_model_changed_refreshes_view_inertial_masks(self):
        """Runtime parent inertial edits should refresh derived view masks."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=_StepCountingCopySolver, bodies=[0]),
                SolverCoupled.Entry(name="B", solver=_StepCountingCopySolver, bodies=[1]),
            ],
        )

        self.model.body_inv_mass.assign(np.array([0.25, 0.125], dtype=np.float32))
        coupled.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

        view_a_inv_mass = coupled.view("A").body_inv_mass.numpy()
        view_b_inv_mass = coupled.view("B").body_inv_mass.numpy()
        np.testing.assert_allclose(view_a_inv_mass, [0.25])
        np.testing.assert_allclose(view_b_inv_mass, [0.125])

    def test_notify_model_changed_refreshes_compacted_properties(self):
        """Runtime parent property changes should reach compact model arrays."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=_StepCountingCopySolver, bodies=[0]),
                SolverCoupled.Entry(name="B", solver=_StepCountingCopySolver, bodies=[1]),
            ],
        )
        self.model.body_flags.assign(np.array([5, 9], dtype=np.int32))
        self.model.shape_material_mu.assign(np.array([0.25, 0.75], dtype=np.float32))

        coupled.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES | newton.ModelFlags.SHAPE_PROPERTIES)

        np.testing.assert_array_equal(coupled.view("A").body_flags.numpy(), [5])
        np.testing.assert_array_equal(coupled.view("B").body_flags.numpy(), [9])
        np.testing.assert_allclose(coupled.view("A").shape_material_mu.numpy(), [0.25, 0.75])
        np.testing.assert_allclose(coupled.view("B").shape_material_mu.numpy(), [0.25, 0.75])

    def test_compact_shape_namespace_option_is_not_exposed(self):
        """Coupled entries should use the shared global shape namespace."""
        with self.assertRaises(TypeError):
            SolverCoupled.Entry(
                name="A",
                solver=SolverSemiImplicit,
                bodies=[0],
                shapes=[0],
                preserve_shape_ids=False,
            )

    def test_entry_shapes_filter_shape_contact_pairs(self):
        """Entry shape masks should prune explicit contact pairs in each view."""
        self.assertEqual(self.model.shape_contact_pair_count, 1)

        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0], shapes=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[1]),
            ],
        )

        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        view_a = coupled.view("A")
        view_b = coupled.view("B")
        flags_a = view_a.shape_flags.numpy()
        flags_b = view_b.shape_flags.numpy()

        self.assertEqual(view_a.shape_flags.shape[0], self.model.shape_count)
        self.assertNotEqual(int(flags_a[0]) & collide, 0)
        self.assertEqual(int(flags_a[1]) & collide, 0)
        np.testing.assert_array_equal(view_a.shape_body.numpy(), np.array([0, -1], dtype=np.int32))
        self.assertEqual(view_a.shape_contact_pair_count, 0)

        self.assertEqual(view_b.shape_flags.shape[0], self.model.shape_count)
        self.assertEqual(int(flags_b[0]) & collide, 0)
        self.assertNotEqual(int(flags_b[1]) & collide, 0)
        np.testing.assert_array_equal(view_b.shape_body.numpy(), np.array([-1, 0], dtype=np.int32))
        self.assertEqual(view_b.shape_contact_pair_count, 0)

        self.assertEqual(self.model.shape_contact_pair_count, 1)

    def test_entries_preserve_global_shape_ids_by_default(self):
        """Entry shape views should keep global shape arrays with hidden dummies."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="A",
                    solver=SolverSemiImplicit,
                    bodies=[0],
                    shapes=[0],
                ),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[1]),
            ],
        )

        view_a = coupled.view("A")
        flags = view_a.shape_flags.numpy()
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)

        self.assertEqual(view_a.body_count, 1)
        self.assertEqual(view_a.shape_count, self.model.shape_count)
        self.assertEqual(view_a.shape_flags.shape[0], self.model.shape_count)
        np.testing.assert_array_equal(view_a.shape_body.numpy(), np.array([0, -1], dtype=np.int32))
        self.assertEqual(view_a.body_shapes, {-1: [], 0: [0]})
        self.assertNotEqual(int(flags[0]) & collide, 0)
        self.assertEqual(int(flags[1]) & collide, 0)
        self.assertEqual(view_a.shape_contact_pair_count, 0)

    def test_particle_entry_without_shapes_keeps_global_static_shapes(self):
        """Particle-only entries should inherit global static shapes by default."""
        builder = newton.ModelBuilder()
        ground_shape = builder.add_ground_plane()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        dynamic_shape = builder.add_shape_sphere(body=body, radius=0.1)
        particle = builder.add_particle(pos=(0.0, 0.0, 0.5), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="particles", solver=SolverSemiImplicit, particles=[particle]),
            ],
        )

        view = coupled.view("particles")
        flags = view.shape_flags.numpy()
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)

        self.assertEqual(view.shape_count, model.shape_count)
        self.assertEqual(view.body_shapes[-1], [ground_shape])
        self.assertNotIn(dynamic_shape, view.body_shapes[-1])
        self.assertNotEqual(int(flags[ground_shape]) & collide_particles, 0)
        self.assertEqual(int(flags[dynamic_shape]) & collide_particles, 0)
        body_shape_ids = np.array(view.body_shapes[-1], dtype=int)
        particle_collider_shapes = body_shape_ids[(flags[body_shape_ids] & collide_particles) > 0]
        np.testing.assert_array_equal(particle_collider_shapes, np.array([ground_shape], dtype=int))

    def test_particles_keep_global_connectivity_while_rigid_domains_compact(self):
        """Particle identity mappings must not prevent independent rigid compaction."""
        builder = newton.ModelBuilder()
        for mass in (1.0, 2.0, 3.0):
            body = builder.add_body(mass=mass, inertia=wp.mat33(np.eye(3)))
            builder.add_shape_sphere(body=body, radius=0.1)
        for x in (0.0, 1.0):
            builder.add_particle(pos=(x, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_spring(0, 1, ke=1.0, kd=0.1, control=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="mixed",
                    solver=SolverSemiImplicit,
                    bodies=[2],
                    particles=[0],
                    shapes=[2],
                )
            ],
        )
        view = coupled.view("mixed")

        self.assertEqual(view.body_count, 1)
        self.assertEqual(view.shape_count, model.shape_count)
        np.testing.assert_allclose(view.body_mass.numpy(), model.body_mass.numpy()[2:3])
        np.testing.assert_array_equal(view.shape_body.numpy(), [-1, -1, 0])
        self.assertEqual(view.body_shapes, {-1: [], 0: [2]})

        self.assertEqual(view.particle_count, model.particle_count)
        self.assertEqual(view.spring_count, model.spring_count)
        np.testing.assert_array_equal(view.spring_indices.numpy(), [0, 1])

    def test_preserved_global_shape_ids_remap_hidden_shapes_in_mixed_views(self):
        """Preserved shape ids should not leave hidden shapes attached to omitted bodies."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=0, radius=0.1)
        builder.add_shape_sphere(body=1, radius=0.1)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="A",
                    solver=SolverSemiImplicit,
                    bodies=[0],
                    particles=[0],
                    shapes=[0],
                ),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[1]),
            ],
        )

        view_a = coupled.view("A")

        self.assertEqual(view_a.body_count, 1)
        self.assertEqual(view_a.particle_count, 1)
        self.assertEqual(view_a.shape_count, model.shape_count)
        np.testing.assert_array_equal(view_a.shape_body.numpy(), np.array([0, -1], dtype=np.int32))
        self.assertEqual(view_a.body_shapes, {-1: [], 0: [0]})

    def test_proxy_shape_visibility_keeps_proxy_contact_pairs(self):
        """Proxy destination views should keep shape pairs touching proxy bodies."""
        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0], shapes=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[0]),
                ],
            ),
        )

        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        view_a = coupled.view("A")
        view_b = coupled.view("B")

        self.assertEqual(view_a.shape_contact_pair_count, 0)
        self.assertNotEqual(int(view_b.shape_flags.numpy()[0]) & collide, 0)
        self.assertNotEqual(int(view_b.shape_flags.numpy()[1]) & collide, 0)
        self.assertEqual(view_b.shape_contact_pair_count, 1)
        np.testing.assert_array_equal(view_b.shape_contact_pairs.numpy(), np.array([[0, 1]], dtype=np.int32))

    def test_proxy_harvest_uses_filtered_preserved_shape_contacts(self):
        """Custom proxy harvest should receive the contacts used by the step."""
        _StepCountingCopySolver.instances.clear()
        _ContactRecordingBodyHarvestSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        ground_shape = builder.add_ground_plane()
        src_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        src_shape = builder.add_shape_sphere(body=src_body, radius=0.1)
        dst_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        dst_shape = builder.add_shape_sphere(body=dst_body, radius=0.1)
        hidden_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        hidden_shape = builder.add_shape_sphere(body=hidden_body, radius=0.1)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[src_body], shapes=[src_shape]),
                SolverCoupled.Entry(
                    name="dst",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=[dst_body],
                    shapes=[ground_shape, dst_shape],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(source="src", destination="dst", bodies=[src_body]),
                ],
            ),
        )

        contacts = newton.Contacts(2, 0, device=model.device)
        contacts.rigid_contact_count.assign(np.array([2], dtype=np.int32))
        contacts.rigid_contact_shape0.assign(np.array([ground_shape, ground_shape], dtype=np.int32))
        contacts.rigid_contact_shape1.assign(np.array([dst_shape, hidden_shape], dtype=np.int32))

        coupled.step(model.state(), model.state(), control=None, contacts=contacts, dt=1.0 / 60.0)

        dst_solver = _ContactRecordingBodyHarvestSolver.instances["dst"]
        self.assertEqual(len(dst_solver.step_contacts), 1)
        self.assertEqual(len(dst_solver.harvest_contacts), 1)
        self.assertIs(dst_solver.harvest_contacts[0], dst_solver.step_contacts[0])
        self.assertIsNot(dst_solver.step_contacts[0], contacts)
        self.assertEqual(int(dst_solver.step_contacts[0].rigid_contact_count.numpy()[0]), 1)
        np.testing.assert_array_equal(dst_solver.rigid_shape1_steps[0], np.array([dst_shape], dtype=np.int32))

    def test_proxy_collision_contacts_bypass_preserved_shape_filter(self):
        """Proxy-local contacts are already generated in the destination view."""
        _StepCountingCopySolver.instances.clear()
        _ContactRecordingBodyHarvestSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        ground_shape = builder.add_ground_plane()
        src_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        src_shape = builder.add_shape_sphere(body=src_body, radius=0.1)
        dst_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        dst_shape = builder.add_shape_sphere(body=dst_body, radius=0.1)
        model = builder.finalize(device="cpu")

        proxy_contacts = newton.Contacts(1, 0, device=model.device)
        proxy_contacts.rigid_contact_count.assign(np.array([1], dtype=np.int32))
        proxy_contacts.rigid_contact_shape0.assign(np.array([ground_shape], dtype=np.int32))
        proxy_contacts.rigid_contact_shape1.assign(np.array([dst_shape], dtype=np.int32))

        def make_pipeline(view):
            del view
            return _FakeProxyCollisionPipeline(model.device, contacts=proxy_contacts)

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[src_body], shapes=[src_shape]),
                SolverCoupled.Entry(
                    name="dst",
                    solver=_ContactRecordingBodyHarvestSolver,
                    bodies=[dst_body],
                    shapes=[ground_shape, dst_shape],
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[src_body],
                        collision_pipeline=make_pipeline,
                    ),
                ],
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=1.0 / 60.0)

        dst_solver = _ContactRecordingBodyHarvestSolver.instances["dst"]
        self.assertEqual(len(dst_solver.step_contacts), 1)
        self.assertEqual(len(dst_solver.harvest_contacts), 1)
        self.assertIs(dst_solver.step_contacts[0], proxy_contacts)
        self.assertIs(dst_solver.harvest_contacts[0], proxy_contacts)

    def test_duplicate_shape_ownership_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "owned by more than one"):
            SolverCoupled(
                model=self.model,
                entries=[
                    SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0], shapes=[0]),
                    SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[0]),
                ],
            )

    def test_step(self):
        """SolverCoupled.step() should advance both bodies."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
            ],
        )

        state_0 = self.model.state()
        state_1 = self.model.state()
        collision_pipeline = newton.CollisionPipeline(self.model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_0, contacts)

        # Step and check bodies moved (due to gravity)
        coupled.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        q0_before = state_0.body_q.numpy()
        q1_after = state_1.body_q.numpy()

        # Bodies should have fallen under gravity
        for i in range(2):
            self.assertFalse(
                np.allclose(q0_before[i], q1_after[i]),
                f"Body {i} did not move after step",
            )

    def test_entry_in_place_steps_same_state(self):
        """Entries can opt into same-object state input/output stepping."""
        _InPlaceRecordingParticleSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="particles",
                    solver=lambda v: _InPlaceRecordingParticleSolver(model=v),
                    particles=[0],
                    in_place=True,
                ),
            ],
        )

        state = model.state()

        coupled.step(state, state, control=None, contacts=None, dt=1.0 / 60.0)

        solver = _InPlaceRecordingParticleSolver.instances["particles"]
        self.assertEqual(solver.in_place_calls, [True])
        np.testing.assert_allclose(state.particle_qd.numpy()[0], np.array([0.0, 2.0, 0.0]))

    def test_entry_in_place_substeps_same_state(self):
        """In-place entries can substep without allocating scratch states."""
        _InPlaceRecordingParticleSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="particles",
                    solver=lambda v: _InPlaceRecordingParticleSolver(model=v),
                    particles=[0],
                    substeps=3,
                    in_place=True,
                ),
            ],
        )

        state = model.state()
        coupled.step(state, state, control=None, contacts=None, dt=0.3)

        solver = _InPlaceRecordingParticleSolver.instances["particles"]
        self.assertEqual(solver.in_place_calls, [True, True, True])
        np.testing.assert_allclose(solver.dt_values, [0.1, 0.1, 0.1])
        np.testing.assert_allclose(state.particle_qd.numpy()[0], np.array([0.0, 6.0, 0.0]))

    def test_particle_views_deactivate_non_owned_particles(self):
        """Each particle owner view should expose only its owned particles as active."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=(0.1, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, particles=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, particles=[1]),
            ],
        )

        active = int(newton.ParticleFlags.ACTIVE)
        view_a_flags = coupled.view("A").particle_flags.numpy()
        view_b_flags = coupled.view("B").particle_flags.numpy()
        parent_flags = model.particle_flags.numpy()

        self.assertEqual(view_a_flags.shape[0], 2)
        self.assertNotEqual(view_a_flags[0] & active, 0)
        self.assertEqual(view_a_flags[1] & active, 0)
        self.assertEqual(view_b_flags[0] & active, 0)
        self.assertNotEqual(view_b_flags[1] & active, 0)
        self.assertNotEqual(parent_flags[0] & active, 0)
        self.assertNotEqual(parent_flags[1] & active, 0)

    def test_proxy_destination_view_marks_proxy_flags(self):
        """Proxy destination views should expose proxy bodies through body_flags."""
        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[0]),
                ],
            ),
        )

        view_a = coupled.view("A")
        view_b = coupled.view("B")
        proxy_flag = int(newton.BodyFlags.PROXY)

        self.assertEqual(view_a.body_flags.numpy()[0] & proxy_flag, 0)
        self.assertNotEqual(view_b.body_flags.numpy()[0] & proxy_flag, 0)
        self.assertEqual(self.model.body_flags.numpy()[0] & proxy_flag, 0)
        self.assertGreater(view_b.body_inv_mass.numpy()[0], 0.0)

    def test_proxy_coupling_rejects_more_than_two_entries(self):
        """Generic proxy coupling is currently limited to one solver pair."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "at most two solver entries"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="a", solver=SolverSemiImplicit, bodies=[0]),
                    SolverCoupled.Entry(name="b", solver=SolverSemiImplicit, bodies=[1]),
                    SolverCoupled.Entry(name="c", solver=SolverSemiImplicit, bodies=[2]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(source="a", destination="b", bodies=[0]),
                    ],
                ),
            )

    def test_proxy_coupling_rejects_invalid_numerical_config(self):
        entries = [
            SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
            SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
        ]

        for mass_scale in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(mass_scale=mass_scale), self.assertRaisesRegex(ValueError, "mass_scale"):
                SolverCoupledProxy(
                    model=self.model,
                    entries=entries,
                    coupling=SolverCoupledProxy.Config(
                        proxies=[
                            SolverCoupledProxy.Proxy(
                                source="A",
                                destination="B",
                                bodies=[0],
                                mass_scale=mass_scale,
                            )
                        ]
                    ),
                )

        for iterations in (0, -1, 1.5, float("nan")):
            with self.subTest(iterations=iterations), self.assertRaisesRegex(ValueError, "iterations"):
                SolverCoupledProxy(
                    model=self.model,
                    entries=entries,
                    coupling=SolverCoupledProxy.Config(
                        proxies=[SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[0])],
                        iterations=iterations,
                    ),
                )

        for collide_interval in (0, -1, 1.5, float("nan")):
            with (
                self.subTest(collide_interval=collide_interval),
                self.assertRaisesRegex(ValueError, "collide_interval"),
            ):
                SolverCoupledProxy(
                    model=self.model,
                    entries=entries,
                    coupling=SolverCoupledProxy.Config(
                        proxies=[
                            SolverCoupledProxy.Proxy(
                                source="A",
                                destination="B",
                                bodies=[0],
                                collision_pipeline=lambda model: None,
                                collide_interval=collide_interval,
                            )
                        ]
                    ),
                )

    def test_proxy_coupling_rejects_unowned_source_body(self):
        with self.assertRaisesRegex(ValueError, "owned by source entry"):
            SolverCoupledProxy(
                model=self.model,
                entries=[
                    SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
                    SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[SolverCoupledProxy.Proxy(source="A", destination="B", bodies=[1])]
                ),
            )

    def test_proxy_coupling_rejects_destination_owned_proxy_body(self):
        """Proxy body ids must not alias bodies owned by the destination."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body0 = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        body1 = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "owned by destination entry"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[body0]),
                    SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[body1]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[body0],
                            proxy_bodies=[body1],
                        ),
                    ],
                ),
            )

    def test_proxy_coupling_rejects_destination_owned_proxy_particle(self):
        """Proxy particle ids must not alias particles owned by the destination."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        particle0 = builder.add_particle(
            pos=(0.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
        )
        particle1 = builder.add_particle(
            pos=(1.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
        )
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "owned by destination entry"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, particles=[particle0]),
                    SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, particles=[particle1]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            particles=[particle0],
                            proxy_particles=[particle1],
                        ),
                    ],
                ),
            )


class TestSolverMuJoCoCouplingHooks(unittest.TestCase):
    """MuJoCo-specific coupling hook behavior."""

    def test_effective_inertia_preserves_anisotropic_free_body_inertia(self):
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as exc:
            self.skipTest(str(exc))

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(
            mass=2.0,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.5),
        )
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model=model, iterations=1, disable_contacts=True)

        endpoint_kind = wp.array([int(CouplingEndpointKind.BODY)], dtype=int, device=model.device)
        endpoint_index = wp.array([body], dtype=int, device=model.device)
        endpoint_local_pos = wp.zeros(1, dtype=wp.vec3, device=model.device)
        effective_mass = wp.empty(1, dtype=float, device=model.device)
        effective_inertia = wp.empty(1, dtype=wp.mat33, device=model.device)
        solver.coupling_eval_effective_mass_block(
            endpoint_kind,
            endpoint_index,
            endpoint_local_pos,
            effective_mass,
            effective_inertia,
        )

        np.testing.assert_allclose(effective_mass.numpy(), model.body_mass.numpy(), rtol=1.0e-5)
        np.testing.assert_allclose(effective_inertia.numpy(), model.body_inertia.numpy(), rtol=1.0e-5)

    def test_gravity_acceleration_hook_uses_body_gravcomp(self):
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as exc:
            self.skipTest(str(exc))

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0), up_axis=newton.Axis.Z)
        SolverMuJoCo.register_custom_attributes(builder)

        body0 = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"mujoco:gravcomp": 0.0},
        )
        body1 = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"mujoco:gravcomp": 0.5},
        )
        body2 = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"mujoco:gravcomp": 1.0},
        )
        builder.add_shape_box(body=body0, hx=0.05, hy=0.05, hz=0.05)
        builder.add_shape_box(body=body1, hx=0.05, hy=0.05, hz=0.05)
        builder.add_shape_box(body=body2, hx=0.05, hy=0.05, hz=0.05)
        joint0 = builder.add_joint_revolute(parent=-1, child=body0, axis=(0.0, 0.0, 1.0))
        joint1 = builder.add_joint_revolute(parent=body0, child=body1, axis=(0.0, 1.0, 0.0))
        joint2 = builder.add_joint_revolute(parent=body1, child=body2, axis=(1.0, 0.0, 0.0))
        builder.add_articulation([joint0, joint1, joint2])
        model = builder.finalize(device="cpu")

        solver = SolverMuJoCo(model=model, iterations=1, disable_contacts=True)
        body_acceleration = wp.empty(model.body_count, dtype=wp.vec3, device=model.device)
        solver.coupling_eval_gravity_acceleration(body_acceleration, None)

        np.testing.assert_allclose(
            body_acceleration.numpy(),
            np.array([[0.0, 0.0, -10.0], [0.0, 0.0, -5.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            atol=1.0e-6,
        )

        model.mujoco.gravcomp.assign(np.array([0.25, 0.5, 0.75], dtype=np.float32))
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        solver.coupling_eval_gravity_acceleration(body_acceleration, None)

        np.testing.assert_allclose(
            body_acceleration.numpy(),
            np.array([[0.0, 0.0, -7.5], [0.0, 0.0, -5.0], [0.0, 0.0, -2.5]], dtype=np.float32),
            atol=1.0e-6,
        )


def _coupled_vbd_reset_preserves_pose_history(test, device):
    """Preserve VBD pose history across coupled masked/full resets and restarts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))

    def add_free_body(*, is_kinematic=False):
        body = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            is_kinematic=is_kinematic,
        )
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])
        return body, joint

    builder.begin_world()
    dynamic_body, dynamic_joint = add_free_body()
    proxy_body, _ = add_free_body()
    builder.end_world()
    builder.begin_world()
    kinematic_body, kinematic_joint = add_free_body(is_kinematic=True)
    builder.end_world()
    builder.color()
    model = builder.finalize(device=device)

    coupled = SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda view: SolverVBD(view, iterations=0),
                bodies=[dynamic_body, kinematic_body],
                joints=[dynamic_joint, kinematic_joint],
            ),
            SolverCoupled.Entry(name="copy", solver=_StepCountingCopySolver),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="vbd",
                    destination="copy",
                    bodies=[dynamic_body],
                    proxy_bodies=[proxy_body],
                )
            ],
            iterations=2,
        ),
    )

    source_bodies = np.array([dynamic_body, kinematic_body])
    state_in = model.state()
    state_out = model.state()
    dt = 1.0e-2
    model_q = model.body_q.numpy().copy()
    model_qd = model.body_qd.numpy().copy()

    # Establish VBD's first-step pose baseline away from the model defaults.
    initial_q = model_q.copy()
    initial_q[source_bodies, 0] = [3.0, 4.0]
    state_in.body_q.assign(initial_q)
    state_in.body_qd.zero_()
    coupled.step(state_in, state_out, None, None, dt)
    np.testing.assert_allclose(state_out.body_q.numpy()[source_bodies], initial_q[source_bodies], atol=1.0e-6)
    np.testing.assert_allclose(state_out.body_qd.numpy()[source_bodies], 0.0, atol=1.0e-5)
    state_in, state_out = state_out, state_in

    # Reset world 1 while retaining world 0's authored displacement as motion.
    moved_q = state_in.body_q.numpy().copy()
    moved_q[source_bodies, 0] += 1.0
    state_in.body_q.assign(moved_q)
    state_in.body_qd.zero_()
    coupled.reset(
        state_in,
        world_mask=wp.array([False, True], dtype=wp.bool, device=device),
        flags=0,
    )
    steps_before = coupled.solver("copy").step_count
    coupled.step(state_in, state_out, None, None, dt)
    test.assertEqual(coupled.solver("copy").step_count, steps_before + 2)

    np.testing.assert_allclose(state_out.body_q.numpy()[source_bodies], moved_q[source_bodies], atol=1.0e-6)
    qd = state_out.body_qd.numpy()
    np.testing.assert_allclose(qd[dynamic_body, 0], 1.0 / dt, rtol=1.0e-5, atol=1.0e-3)
    np.testing.assert_allclose(qd[kinematic_body], 0.0, atol=1.0e-5)

    # Default reset restores model state and rebaselines both source bodies.
    state_in, state_out = state_out, state_in
    coupled.reset(state_in)
    coupled.step(state_in, state_out, None, None, dt)
    np.testing.assert_allclose(state_out.body_q.numpy()[source_bodies], model_q[source_bodies], atol=1.0e-6)
    np.testing.assert_allclose(state_out.body_qd.numpy()[source_bodies], model_qd[source_bodies], atol=1.0e-5)


class TestSolverVBDCouplingHooks(unittest.TestCase):
    """VBD-specific coupling hook behavior."""

    def test_external_rigid_solver_harvests_particle_soft_contacts(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=body, radius=0.1)
        builder.add_particle(pos=(0.15, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        builder.color()
        model = builder.finalize(device="cpu")
        solver = SolverVBD(model=model, iterations=1, integrate_with_external_rigid_solver=True)

        state_in = model.state()
        state_out = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
        solver.step(state_in, state_out, control=None, contacts=contacts, dt=1.0 / 60.0)

        out_particle_f = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        solver.coupling_harvest_proxy_particle_forces(
            wp.array([0], dtype=int, device=model.device),
            out_particle_f,
            particle_qd_before=state_in.particle_qd,
            state=state_in,
            state_out=state_out,
            contacts=contacts,
            dt=1.0 / 60.0,
        )
        self.assertTrue(np.all(np.isfinite(out_particle_f.numpy())))


class TestSolverCoupledProxyJoints(unittest.TestCase):
    """Proxy joints preserve source drive commands in destination solves."""

    def test_aliased_proxy_joint_copies_control_target_each_iteration(self):
        _ControlRecordingSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        source_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        proxy_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        source_joint = builder.add_joint_prismatic(parent=-1, child=source_body, axis=(1.0, 0.0, 0.0))
        proxy_joint = builder.add_joint_prismatic(parent=-1, child=proxy_body, axis=(1.0, 0.0, 0.0))
        builder.add_articulation([source_joint])
        builder.add_articulation([proxy_joint])
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=_ControlRecordingSolver,
                    bodies=[source_body],
                    joints=[source_joint],
                ),
                SolverCoupled.Entry(name="dst", solver=_ControlRecordingSolver, bodies=[proxy_body]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        joints=[source_joint],
                        proxy_joints=[proxy_joint],
                    )
                ],
                iterations=3,
            ),
        )
        control = model.control()
        control.joint_target_q.assign(np.array([0.25, 0.75], dtype=np.float32))
        control.joint_target_qd.assign(np.array([0.5, 1.5], dtype=np.float32))

        coupled.step(model.state(), model.state(), control, contacts=None, dt=1.0 / 60.0)

        source_solver, destination_solver = _ControlRecordingSolver.instances
        self.assertEqual(len(source_solver.joint_target_q), 3)
        self.assertEqual(len(destination_solver.joint_target_q), 3)
        for target_q, target_qd in zip(
            destination_solver.joint_target_q,
            destination_solver.joint_target_qd,
            strict=True,
        ):
            np.testing.assert_array_equal(target_q, np.array([0.25], dtype=np.float32))
            np.testing.assert_array_equal(target_qd, np.array([0.5], dtype=np.float32))


class TestSolverCoupledMuJoCoVBDMultiEnv(unittest.TestCase):
    """Regression tests for multi-world MuJoCo/VBD solver partitions."""

    def test_compacted_articulation_end_excludes_loop_joint(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        root_joint = builder.add_joint_fixed(parent=-1, child=base)
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        tree_joint = builder.add_joint_revolute(parent=base, child=link, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([root_joint, tree_joint])
        loop_joint = builder.add_joint_fixed(parent=base, child=link)
        builder.joint_articulation[loop_joint] = -1
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="loop",
                    solver=SolverSemiImplicit,
                    bodies=[base, link],
                    joints=[root_joint, tree_joint, loop_joint],
                )
            ],
        )

        view = coupled.view("loop")
        np.testing.assert_array_equal(view.articulation_start.numpy(), [0, 3])
        np.testing.assert_array_equal(view.articulation_end.numpy(), [2])

    def test_compacted_multi_world_articulation_end_is_rebased(self):
        """articulation_end must be rebased to local joint ids, matching articulation_start.

        Regression: compaction rebased articulation_start but left articulation_end as
        global joint indices, so a non-first-world articulation got an out-of-bounds
        end (e.g. end=9 in an 8-joint view), corrupting solver FK (fixed base displaced).
        """
        world_count = 2
        template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))

        # Articulation A: fixed base + one revolute link (the "rigid" entry).
        base = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="base")
        jf = template.add_joint_fixed(parent=-1, child=base)
        link = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="link")
        jr = template.add_joint_revolute(parent=base, child=link, axis=(0.0, 0.0, 1.0))
        template.add_articulation([jf, jr])
        # Articulation B: a free body owned by the other entry.
        free_body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="free")
        jfree = template.add_joint_free(child=free_body)
        template.add_articulation([jfree])

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.replicate(template, world_count=world_count)
        builder.color()
        model = builder.finalize(device="cpu")

        bpw, jpw = template.body_count, template.joint_count

        def expand(ids, stride):
            return [w * stride + i for w in range(world_count) for i in ids]

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="rigid",
                    solver=SolverSemiImplicit,
                    bodies=expand([base, link], bpw),
                    joints=expand([jf, jr], jpw),
                ),
                SolverCoupled.Entry(
                    name="free",
                    solver=SolverSemiImplicit,
                    bodies=expand([free_body], bpw),
                    joints=expand([jfree], jpw),
                ),
            ],
        )

        view = coupled.view("rigid")
        starts = view.articulation_start.numpy()
        ends = view.articulation_end.numpy()
        # Two articulations (one per world), each spanning 2 joints in the 4-joint view.
        self.assertEqual(starts.tolist(), [0, 2, 4])
        self.assertEqual(ends.tolist(), [2, 4])
        # End indices must stay within the compacted joint range (no OOB).
        self.assertTrue(all(e <= view.joint_count for e in ends))


class TestSolverCoupledBodyProxyInertia(unittest.TestCase):
    """Body proxy mappings install full proxy inertia tensors."""

    @staticmethod
    def _entry_body_local(coupled: SolverCoupledProxy, entry_name: str, body_id: int) -> int:
        return int(coupled._entries[entry_name].body_global_to_local.numpy()[body_id])

    def test_body_proxy_aitken_relaxation_converges_affine_fixed_point(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_AffineBodyForceSourceSolver, bodies=[body]),
                SolverCoupled.Entry(name="dst", solver=_AffineProxyBodyFeedbackSolver),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[body],
                        proxy_relaxation_mode="aitken",
                        proxy_relaxation=1.0,
                        proxy_relaxation_min=0.1,
                        proxy_relaxation_max=1.0,
                    )
                ],
                iterations=3,
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=1.0)

        mapping = coupled._proxy_mappings[0]
        np.testing.assert_allclose(mapping.coupling_forces.numpy()[body, 0], 1.0 / 3.0, atol=1.0e-6)
        np.testing.assert_allclose(mapping.aitken_relaxation.numpy()[0], 1.0 / 3.0, atol=1.0e-6)

    def test_duplicate_body_proxy_mapping_ids_are_rejected(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(3):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "Duplicate source body"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0, 1]),
                    SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[0, 0],
                            proxy_bodies=[1, 2],
                        ),
                    ],
                ),
            )

        with self.assertRaisesRegex(ValueError, "Duplicate proxy body"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0, 1]),
                    SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[0, 1],
                            proxy_bodies=[2, 2],
                        ),
                    ],
                ),
            )

    def test_cross_world_body_proxy_mapping_is_rejected(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.begin_world()
        source_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.end_world()
        builder.begin_world()
        proxy_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.end_world()
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "same world"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[source_body]),
                    SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver, bodies=[proxy_body]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[source_body],
                            proxy_bodies=[proxy_body],
                        ),
                    ],
                ),
            )

    def test_body_proxy_maps_proxy_indexed_feedback_to_source(self):
        _BodyForceRecordingSolver.instances.clear()
        _ProxyBodyHookSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_BodyForceRecordingSolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyBodyHookSolver, bodies=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                        proxy_bodies=[2],
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        src_solver = _BodyForceRecordingSolver.instances[-1]
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(src_solver.input_body_f[1].shape[0], 1)
        np.testing.assert_allclose(src_solver.input_body_f[1][0], expected, atol=1.0e-6)

    def test_body_proxy_feedback_relaxation_blends_next_step_force_input(self):
        _BodyForceRecordingSolver.instances.clear()
        _ProxyBodyHookSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_BodyForceRecordingSolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyBodyHookSolver, bodies=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                        proxy_bodies=[2],
                        proxy_relaxation=0.25,
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        src_solver = _BodyForceRecordingSolver.instances[-1]
        expected = 0.25 * np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(src_solver.input_body_f[1][0], expected, atol=1.0e-6)


class TestSolverCoupledParticleProxy(unittest.TestCase):
    """Particle proxy mappings keep proxy particles dynamic in the destination view."""

    def setUp(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        self.model = builder.finalize(device="cpu")

    def _make_coupled(self, dst_solver=_ProxyParticleKickSolver):
        return SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=dst_solver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                    ),
                ],
            ),
        )

    def test_duplicate_particle_proxy_mapping_ids_are_rejected(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for i in range(3):
            builder.add_particle(pos=(float(i), 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "Duplicate source particle"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0, 1]),
                    SolverCoupled.Entry(name="dst", solver=_ProxyParticleKickSolver, particles=[2]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            particles=[0, 0],
                            proxy_particles=[1, 2],
                        ),
                    ],
                ),
            )

        with self.assertRaisesRegex(ValueError, "Duplicate proxy particle"):
            SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0, 1]),
                    SolverCoupled.Entry(name="dst", solver=_ProxyParticleKickSolver, particles=[2]),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="src",
                            destination="dst",
                            particles=[0, 1],
                            proxy_particles=[2, 2],
                        ),
                    ],
                ),
            )

    def test_proxy_destination_view_keeps_and_scales_particle_mass(self):
        _ParticleForceRecordingSolver.instances.clear()
        coupled = self._make_coupled()

        src_view = coupled.view("src")
        dst_view = coupled.view("dst")

        self.assertEqual(src_view.particle_inv_mass.shape[0], 2)
        self.assertEqual(src_view.particle_inv_mass.numpy()[1], 0.0)
        np.testing.assert_allclose(dst_view.particle_mass.numpy(), [1.0, 2.0])
        np.testing.assert_allclose(dst_view.particle_inv_mass.numpy(), [1.0, 0.5])
        np.testing.assert_allclose(self.model.particle_mass.numpy(), [2.0, 2.0])

    def test_particle_proxy_feedback_is_applied_on_next_step(self):
        _ParticleForceRecordingSolver.instances.clear()
        coupled = self._make_coupled()

        state_0 = self.model.state()
        state_1 = self.model.state()
        control = self.model.control()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=control, contacts=None, dt=dt)

        solver = _ParticleForceRecordingSolver.instances[-1]
        self.assertEqual(len(solver.input_particle_f), 2)
        np.testing.assert_allclose(solver.input_particle_f[0][0], np.zeros(3), atol=1.0e-6)
        np.testing.assert_allclose(solver.input_particle_f[1][0], np.array([0.0, 4.0, 0.0]), atol=1.0e-6)

    def test_particle_proxy_feedback_relaxation_handles_zeroing_custom_harvest(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleHookSolver.instances.clear()

        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ZeroingProxyParticleHookSolver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                        proxy_relaxation=0.25,
                    ),
                ],
            ),
        )

        state_0 = self.model.state()
        state_1 = self.model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        solver = _ParticleForceRecordingSolver.instances[-1]
        self.assertEqual(len(solver.input_particle_f), 2)
        np.testing.assert_allclose(solver.input_particle_f[0][0], np.zeros(3), atol=1.0e-6)
        np.testing.assert_allclose(solver.input_particle_f[1][0], np.array([0.0, 1.75, 0.0]), atol=1.0e-6)

    def test_particle_proxy_feedback_overrelaxation_is_applied_on_next_step(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleHookSolver.instances.clear()

        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ZeroingProxyParticleHookSolver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                        proxy_relaxation=1.5,
                    ),
                ],
            ),
        )

        state_0 = self.model.state()
        state_1 = self.model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        solver = _ParticleForceRecordingSolver.instances[-1]
        self.assertEqual(len(solver.input_particle_f), 2)
        np.testing.assert_allclose(solver.input_particle_f[0][0], np.zeros(3), atol=1.0e-6)
        np.testing.assert_allclose(solver.input_particle_f[1][0], np.array([0.0, 10.5, 0.0]), atol=1.0e-6)

    def test_particle_proxy_aitken_relaxation_kernels(self):
        coupled = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyParticleHookSolver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        proxy_relaxation_mode="aitken",
                    )
                ],
                iterations=2,
            ),
        )

        coupled.step(self.model.state(), self.model.state(), control=None, contacts=None, dt=0.5)

        mapping = coupled._proxy_particle_mappings[0]
        np.testing.assert_allclose(mapping.coupling_forces.numpy()[0], np.array([0.0, 7.0, 0.0]), atol=1.0e-6)
        self.assertTrue(np.isfinite(mapping.aitken_relaxation.numpy()[0]))

    def test_particle_proxy_maps_proxy_indexed_feedback_to_source(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleHookSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        builder.add_particle(pos=(2.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyParticleHookSolver, particles=[1]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        proxy_particles=[2],
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        src_solver = _ParticleForceRecordingSolver.instances[-1]
        self.assertEqual(src_solver.input_particle_f[1].shape[0], 3)
        np.testing.assert_allclose(src_solver.input_particle_f[1][0], np.array([0.0, 7.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(src_solver.input_particle_f[1][2], np.zeros(3), atol=1.0e-6)

    def test_proxy_destination_view_marks_proxy_particle_flags(self):
        coupled = self._make_coupled()

        src_view = coupled.view("src")
        dst_view = coupled.view("dst")
        proxy_flag = int(newton.ParticleFlags.PROXY)

        self.assertEqual(src_view.particle_flags.numpy()[0] & proxy_flag, 0)
        self.assertNotEqual(dst_view.particle_flags.numpy()[0] & proxy_flag, 0)
        self.assertEqual(self.model.particle_flags.numpy()[0] & proxy_flag, 0)

    def test_xpbd_ignores_proxy_proxy_particle_contacts(self):
        flags = int(newton.ParticleFlags.ACTIVE) | int(newton.ParticleFlags.PROXY)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_particle(pos=(-0.02, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05, flags=flags)
        builder.add_particle(pos=(0.02, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05, flags=flags)
        model = builder.finalize(device="cpu")
        solver = SolverXPBD(model=model, iterations=4, soft_contact_relaxation=1.0)

        state_0 = model.state()
        state_1 = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        q_before = state_0.particle_q.numpy().copy()

        solver.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_allclose(state_1.particle_q.numpy(), q_before, atol=1.0e-6)

    def test_xpbd_ignores_proxy_static_particle_contacts(self):
        proxy_flags = int(newton.ParticleFlags.ACTIVE) | int(newton.ParticleFlags.PROXY)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_particle(pos=(-0.02, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05, flags=proxy_flags)
        builder.add_particle(
            pos=(0.02, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,
            radius=0.05,
            flags=int(newton.ParticleFlags.ACTIVE),
        )
        model = builder.finalize(device="cpu")
        solver = SolverXPBD(model=model, iterations=4, soft_contact_relaxation=1.0)

        state_0 = model.state()
        state_1 = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        q_before = state_0.particle_q.numpy().copy()

        solver.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_allclose(state_1.particle_q.numpy(), q_before, atol=1.0e-6)

    def test_xpbd_ignores_proxy_particle_proxy_body_contacts(self):
        proxy_particle_flags = int(newton.ParticleFlags.ACTIVE) | int(newton.ParticleFlags.PROXY)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=body, radius=0.05)
        builder.add_particle(
            pos=(0.08, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            mass=1.0,
            radius=0.05,
            flags=proxy_particle_flags,
        )
        model = builder.finalize(device="cpu")
        view = ModelView(model, "xpbd")
        view.mark_proxy_bodies(wp.array([body], dtype=int, device=model.device))
        solver = SolverXPBD(model=view, iterations=4, soft_contact_relaxation=1.0)

        state_0 = model.state()
        state_1 = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_0, contacts)
        self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
        q_before = state_0.particle_q.numpy().copy()

        solver.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_allclose(state_1.particle_q.numpy(), q_before, atol=1.0e-6)


class TestSolverCoupledVBDColoring(unittest.TestCase):
    """Compaction must remap ``body_color_groups`` for VBD entries.

    A VBD entry whose global body ids are not a 0-prefix gets compacted to dense
    local indices; the color groups must be remapped global->local, or two bodies
    joined by a joint can share a color, race in VBD's parallel solve, and the
    constraint diverges.
    """

    def test_compacted_vbd_entry_color_groups_are_valid(self):
        builder = newton.ModelBuilder()
        for _ in range(5):
            builder.add_body(mass=1.0)  # each auto-adds a free joint + articulation
        fixed_joint = builder.add_joint_fixed(parent=3, child=4)
        builder.color()
        model = builder.finalize(device="cpu")

        # "dst" owns {2,3,4} (not a 0-prefix) -> compaction maps it to local 0,1,2.
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=SolverSemiImplicit,
                    bodies=[0, 1],
                    joints=[0, 1],
                ),
                SolverCoupled.Entry(
                    name="dst",
                    solver=lambda view: SolverVBD(view, iterations=1),
                    bodies=[2, 3, 4],
                    joints=[2, 3, 4, fixed_joint],
                ),
            ],
        )

        view = coupled.view("dst")
        body_count = int(view.body_count)
        groups = [[int(x) for x in g.numpy()] for g in view.body_color_groups]
        parents = [int(x) for x in view.joint_parent.numpy()]
        children = [int(x) for x in view.joint_child.numpy()]

        # Color groups must partition the local body set.
        union = sorted(body for group in groups for body in group)
        self.assertEqual(union, list(range(body_count)), f"groups must partition local bodies; got {groups}")

        # No joint-connected pair may share a color.
        color_of = {body: color for color, group in enumerate(groups) for body in group}
        for parent, child in zip(parents, children, strict=True):
            if 0 <= parent < body_count and 0 <= child < body_count:
                self.assertNotEqual(
                    color_of.get(parent),
                    color_of.get(child),
                    f"joint-connected local bodies {parent},{child} share a color: {groups}",
                )

    def test_compacted_custom_namespace_does_not_mutate_parent(self):
        """Compacted entry namespaces must be view-local, not parent aliases."""
        builder = newton.ModelBuilder()
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
        for _ in range(5):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        soft_joint = builder.add_joint_fixed(parent=3, child=4, custom_attributes={"vbd:joint_is_hard": 0})
        builder.color()
        model = builder.finalize(device="cpu")
        model.vbd.namespace_marker = "parent metadata"

        parent_joint_is_hard = model.vbd.joint_is_hard.numpy().copy()
        vbd_joint_order = [2, 3, 4, soft_joint]

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=SolverSemiImplicit,
                    bodies=[0, 1],
                    joints=[0, 1],
                ),
                SolverCoupled.Entry(
                    name="dst",
                    solver=lambda view: SolverVBD(view, iterations=1),
                    bodies=[2, 3, 4],
                    joints=vbd_joint_order,
                ),
            ],
        )

        np.testing.assert_array_equal(model.vbd.joint_is_hard.numpy(), parent_joint_is_hard)

        view = coupled.view("dst")
        self.assertIsNot(view.vbd, model.vbd)
        self.assertEqual(view.vbd.namespace_marker, model.vbd.namespace_marker)
        self.assertEqual(view.vbd.joint_is_hard.shape[0], view.joint_count)
        np.testing.assert_array_equal(view.vbd.joint_is_hard.numpy(), parent_joint_is_hard[vbd_joint_order])

    def test_compacted_custom_frequency_namespace_metadata_is_generic(self):
        builder = newton.ModelBuilder()
        for _ in range(4):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
            body1=0,
            body2=1,
        )
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
            body1=2,
            body2=3,
        )
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0, 1]),
                SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[2, 3]),
            ],
        )

        view = coupled.view("dst")
        self.assertEqual(view.custom_frequency_counts["mujoco:equality_constraint"], 1)
        self.assertEqual(view.mujoco.equality_constraint_count, 1)
        self.assertEqual(view.mujoco.equality_constraint_type.shape[0], 1)
        np.testing.assert_array_equal(view.mujoco.equality_constraint_body1.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_equal(view.mujoco.equality_constraint_body2.numpy(), np.array([1], dtype=np.int32))
        self.assertEqual(int(view.mujoco.equality_constraint_world_start.numpy()[-1]), 1)
        self.assertNotIn("equality_constraint_count", view.overrides)
        self.assertNotIn("equality_constraint_body1", view.overrides)

    def test_metadata_projects_nonprefix_custom_references(self):
        builder = newton.ModelBuilder()
        for frequency in ("linkage", "entity", "link"):
            builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=frequency, namespace="test"))

        def add_attribute(name, frequency, dtype, *, references=None, assignment=None):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=frequency,
                    dtype=dtype,
                    namespace="test",
                    references=references,
                    assignment=assignment or newton.Model.AttributeAssignment.MODEL,
                )
            )

        add_attribute("linkage_body0", "test:linkage", wp.int32, references="body")
        add_attribute("linkage_body1", "test:linkage", wp.int32, references="body")
        add_attribute("linkage_bodies", "test:linkage", wp.vec2i, references="body")
        add_attribute("linkage_weight", "test:linkage", wp.float32)
        add_attribute("entity_body", "test:entity", wp.int32, references="body")
        add_attribute("link_entity", "test:link", wp.int32, references="test:entity")
        add_attribute(
            "state_seed",
            newton.Model.AttributeFrequency.BODY,
            wp.float32,
            assignment=newton.Model.AttributeAssignment.STATE,
        )

        for body in range(4):
            builder.add_body(
                mass=1.0,
                inertia=wp.mat33(np.eye(3)),
                custom_attributes={"test:state_seed": float(10 + body)},
            )
            builder.add_custom_values(**{"test:entity_body": body})
            builder.add_custom_values(**{"test:link_entity": body})
        builder.add_custom_values(
            **{
                "test:linkage_body0": 0,
                "test:linkage_body1": 2,
                "test:linkage_bodies": wp.vec2i(0, 2),
                "test:linkage_weight": 2.0,
            }
        )
        builder.add_custom_values(
            **{
                "test:linkage_body0": 1,
                "test:linkage_body1": 3,
                "test:linkage_bodies": wp.vec2i(1, 3),
                "test:linkage_weight": 4.0,
            }
        )
        builder.add_custom_values(
            **{
                "test:linkage_body0": -1,
                "test:linkage_body1": 1,
                "test:linkage_bodies": wp.vec2i(-1, 1),
                "test:linkage_weight": 6.0,
            }
        )
        model = builder.finalize(device="cpu")
        model.test.namespace_marker = "parent"

        self.assertEqual(
            model._attribute_reference_frequency("test:linkage_body0"),
            newton.Model.AttributeFrequency.BODY,
        )
        linkage_spec = model._attribute_spec("test:linkage_bodies")
        self.assertEqual(linkage_spec.frequency, "test:linkage")
        self.assertEqual(linkage_spec.references, newton.Model.AttributeFrequency.BODY)
        self.assertEqual(model._attribute_reference_frequency("test:link_entity"), "test:entity")
        self.assertEqual(
            model.attribute_assignment.get("test:linkage_body0", newton.Model.AttributeAssignment.MODEL),
            newton.Model.AttributeAssignment.MODEL,
        )
        self.assertIsInstance(model.attribute_specs["body_label"], newton.Model.AttributeSpec)
        for name, frequency in (
            ("body_label", newton.Model.AttributeFrequency.BODY),
            ("shape_color", newton.Model.AttributeFrequency.SHAPE),
            ("_shape_sdf_index", newton.Model.AttributeFrequency.SHAPE),
            ("tri_materials", newton.Model.AttributeFrequency.TRIANGLE),
        ):
            with self.subTest(core_attribute=name):
                self.assertEqual(model._resolve_attribute_frequency(name), frequency)
        self.assertEqual(
            model._resolve_attribute_frequency("joint_q"),
            newton.Model.AttributeFrequency.JOINT_COORD,
        )

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0, 2]),
                SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1, 3]),
            ],
        )
        view = coupled.view("dst")

        self.assertEqual(view.test.linkage_count, 2)
        self.assertEqual(view.test.entity_count, 2)
        self.assertEqual(view.test.link_count, 2)
        np.testing.assert_array_equal(view.test.linkage_body0.numpy(), [0, -1])
        np.testing.assert_array_equal(view.test.linkage_body1.numpy(), [1, 0])
        np.testing.assert_array_equal(view.test.linkage_bodies.numpy(), [[0, 1], [-1, 0]])
        np.testing.assert_array_equal(view.test.entity_body.numpy(), [0, 1])
        np.testing.assert_array_equal(view.test.link_entity.numpy(), [0, 1])
        np.testing.assert_allclose(view.test.linkage_weight.numpy(), [4.0, 6.0])

        state = view.state()
        np.testing.assert_allclose(state.test.state_seed.numpy(), [11.0, 13.0])

        view.test.namespace_marker = "view"
        view.test.linkage_weight.fill_(7.0)
        self.assertEqual(model.test.namespace_marker, "parent")
        np.testing.assert_allclose(model.test.linkage_weight.numpy(), [2.0, 4.0, 6.0])

    def test_compaction_projects_late_registered_attribute(self):
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")
        model.extra_values = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        model.attribute_frequency["extra_values"] = newton.Model.AttributeFrequency.BODY

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1]),
            ],
        )

        np.testing.assert_allclose(coupled.view("dst").extra_values.numpy(), [2.0])

    def test_compaction_projects_attribute_spec_registration(self):
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")
        model.spec_values = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
        model.attribute_specs["spec_values"] = newton.Model.AttributeSpec(newton.Model.AttributeFrequency.BODY)

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1]),
            ],
        )

        self.assertEqual(model.get_attribute_frequency("spec_values"), newton.Model.AttributeFrequency.BODY)
        np.testing.assert_allclose(coupled.view("dst").spec_values.numpy(), [2.0])

    def test_compaction_rejects_misaligned_late_registered_attribute(self):
        builder = newton.ModelBuilder()
        for _ in range(2):
            builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")
        model.body_misaligned = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        model.attribute_frequency["body_misaligned"] = newton.Model.AttributeFrequency.BODY

        with self.assertRaisesRegex(ValueError, "body_misaligned.*expected 2 values"):
            SolverCoupled(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0]),
                    SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1]),
                ],
            )

    def test_compaction_validates_custom_reference_storage(self):
        def build_model():
            builder = newton.ModelBuilder()
            builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="row", namespace="test"))
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name="row_body",
                    frequency="test:row",
                    dtype=wp.int32,
                    namespace="test",
                    references="body",
                )
            )
            for body in range(2):
                builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
                builder.add_custom_values(**{"test:row_body": body})
            return builder.finalize(device="cpu")

        for value, message in (
            (None, "registered value is missing"),
            (wp.array([0], dtype=wp.int32, device="cpu"), "expected 2 rows"),
        ):
            with self.subTest(message=message):
                model = build_model()
                model.test.row_body = value
                with self.assertRaisesRegex(ValueError, message):
                    SolverCoupled(
                        model=model,
                        entries=[
                            SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0]),
                            SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1]),
                        ],
                    )

    def test_metadata_projects_custom_rows_across_worlds(self):
        sub_builder = newton.ModelBuilder()
        sub_builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name="node", namespace="test"))
        for name, references in (("node_body", "body"), ("environment", "world")):
            sub_builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency="test:node",
                    dtype=wp.int32,
                    namespace="test",
                    references=references,
                )
            )
        sub_builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="node_value",
                frequency="test:node",
                dtype=wp.float32,
                namespace="test",
            )
        )
        for body in range(2):
            sub_builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
            sub_builder.add_custom_values(
                **{"test:node_body": body, "test:environment": -1, "test:node_value": float(body)}
            )

        builder = newton.ModelBuilder()
        builder.add_world(sub_builder)
        builder.add_world(sub_builder)
        model = builder.finalize(device="cpu")
        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=SolverSemiImplicit, bodies=[0, 2]),
                SolverCoupled.Entry(name="dst", solver=SolverSemiImplicit, bodies=[1, 3]),
            ],
        )
        view = coupled.view("dst")

        self.assertEqual(view.test.node_count, 2)
        np.testing.assert_array_equal(view.test.node_body.numpy(), [0, 1])
        np.testing.assert_array_equal(view.test.environment.numpy(), [0, 1])
        np.testing.assert_allclose(view.test.node_value.numpy(), [1.0, 1.0])
        np.testing.assert_array_equal(view.test.environment_start.numpy(), [0, 1, 2, 2])


add_function_test(
    TestSolverVBDCouplingHooks,
    "test_reset_preserves_pose_history",
    _coupled_vbd_reset_preserves_pose_history,
    devices=get_test_devices(mode="basic"),
)


def _coupled_soft_contact_filter_preserves_unified_fields(test, device):
    """The coupled per-solver soft-contact filter must copy soft_contact_indices/barycentric, not just
    soft_contact_particle. VBD consumes the unified fields, so dropping them delivers a particle contact
    to VBD as the (-1, -1, -1) sentinel and regresses coupled VBD even with full-surface contact off
    (E7)."""
    # One particle soft contact: particle 0 on shape 0, unified record (0, -1, -1) + (1, 0, 0).
    dst_indices = wp.full(1, wp.vec3i(-1, -1, -1), dtype=wp.vec3i, device=device)
    dst_barycentric = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(
        _filter_soft_contacts_global_shape_ids_kernel,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),  # update_filter (dirty)
            wp.array([1], dtype=wp.int32, device=device),  # src_count
            wp.array([0], dtype=wp.int32, device=device),  # src_particle
            wp.array([0], dtype=wp.int32, device=device),  # src_shape
            wp.array([wp.vec3(0.1, 0.2, 0.3)], dtype=wp.vec3, device=device),  # src_body_pos
            wp.zeros(1, dtype=wp.vec3, device=device),  # src_body_vel
            wp.array([wp.vec3(0.0, 0.0, 1.0)], dtype=wp.vec3, device=device),  # src_normal
            wp.array([7], dtype=wp.int32, device=device),  # src_tids
            wp.array([wp.vec3i(0, -1, -1)], dtype=wp.vec3i, device=device),  # src_indices
            wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),  # src_barycentric
            wp.array([int(ShapeFlags.COLLIDE_PARTICLES)], dtype=wp.int32, device=device),  # shape_flags
            wp.array([int(ParticleFlags.ACTIVE)], dtype=wp.int32, device=device),  # particle_flags
            int(ShapeFlags.COLLIDE_PARTICLES),
            int(ParticleFlags.ACTIVE),
            wp.zeros(1, dtype=wp.int32, device=device),  # dst_count
            wp.full(1, -1, dtype=wp.int32, device=device),  # dst_particle
            wp.full(1, -1, dtype=wp.int32, device=device),  # dst_shape
            wp.zeros(1, dtype=wp.vec3, device=device),  # dst_body_pos
            wp.zeros(1, dtype=wp.vec3, device=device),  # dst_body_vel
            wp.zeros(1, dtype=wp.vec3, device=device),  # dst_normal
            wp.full(1, -1, dtype=wp.int32, device=device),  # dst_tids
            dst_indices,
            dst_barycentric,
            wp.full(1, -1, dtype=wp.int32, device=device),  # src_to_dst
        ],
        device=device,
    )
    idx = dst_indices.numpy()[0]
    test.assertEqual(
        (int(idx[0]), int(idx[1]), int(idx[2])),
        (0, -1, -1),
        "unified particle record must be copied through the filter, not left at the (-1,-1,-1) sentinel",
    )
    np.testing.assert_allclose(dst_barycentric.numpy()[0], [1.0, 0.0, 0.0])


add_function_test(
    TestSolverCoupledBasic,
    "test_soft_contact_filter_preserves_unified_fields",
    _coupled_soft_contact_filter_preserves_unified_fields,
    devices=get_test_devices(mode="basic"),
)


if __name__ == "__main__":
    unittest.main()
