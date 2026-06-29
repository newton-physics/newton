# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the coupled solver prototype."""

import unittest
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled.interface import CouplingInterface
from newton._src.solvers.coupled.proxy_utils import (
    smooth_proxy_teleportation_kernel,
    sync_proxy_states_kernel,
)
from newton._src.solvers.mujoco.equality import _add_equality_constraint
from newton._src.solvers.vbd.rigid_vbd_kernels import forward_step_rigid_bodies
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
        self.joint_target_pos = []
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
        self.joint_target_pos.append(
            None if control is None or control.joint_target_pos is None else control.joint_target_pos.numpy().copy()
        )
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
        builder = newton.ModelBuilder(gravity=0.0)
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
        control.joint_target_pos.assign(np.array([11.0, 13.0], dtype=np.float32))

        coupled.step(model.state(), model.state(), control, contacts=None, dt=1.0 / 60.0)

        solver_a, solver_b = _ControlRecordingSolver.instances
        np.testing.assert_array_equal(solver_a.joint_f[0], np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(solver_b.joint_f[0], np.array([7.0], dtype=np.float32))
        np.testing.assert_array_equal(solver_a.joint_target_pos[0], np.array([11.0], dtype=np.float32))
        np.testing.assert_array_equal(solver_b.joint_target_pos[0], np.array([13.0], dtype=np.float32))

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

    def test_entry_can_compact_shape_ids_when_requested(self):
        """Entry views should still support compact local shape ids by opt-out."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="A",
                    solver=SolverSemiImplicit,
                    bodies=[0],
                    shapes=[0],
                    preserve_shape_ids=False,
                ),
                SolverCoupled.Entry(
                    name="B",
                    solver=SolverSemiImplicit,
                    bodies=[1],
                    shapes=[1],
                    preserve_shape_ids=False,
                ),
            ],
        )

        view_a = coupled.view("A")
        view_b = coupled.view("B")

        self.assertEqual(view_a.shape_count, 1)
        self.assertEqual(view_a.shape_flags.shape[0], 1)
        np.testing.assert_array_equal(view_a.shape_body.numpy(), np.array([0], dtype=np.int32))
        self.assertEqual(view_b.shape_count, 1)
        self.assertEqual(view_b.shape_flags.shape[0], 1)
        np.testing.assert_array_equal(view_b.shape_body.numpy(), np.array([0], dtype=np.int32))

    def test_preserved_global_shape_ids_remap_hidden_shapes_in_prefix_views(self):
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

        builder = newton.ModelBuilder(gravity=0.0)
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

        builder = newton.ModelBuilder(gravity=0.0)
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
        contacts = self.model.collide(state_0)

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
        builder = newton.ModelBuilder(gravity=0.0)
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
        builder = newton.ModelBuilder(gravity=0.0)
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
        builder = newton.ModelBuilder(gravity=0.0)
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

    def test_proxy_coupling_rejects_destination_owned_proxy_body(self):
        """Proxy body ids must not alias bodies owned by the destination."""
        builder = newton.ModelBuilder(gravity=0.0)
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
        builder = newton.ModelBuilder(gravity=0.0)
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

    def test_gravity_acceleration_hook_uses_body_gravcomp(self):
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as exc:
            self.skipTest(str(exc))

        builder = newton.ModelBuilder(gravity=-10.0, up_axis=newton.Axis.Z)
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


class TestSolverCoupledProxyJoints(unittest.TestCase):
    """Proxy joints preserve source drive commands in destination solves."""

    def test_aliased_proxy_joint_copies_control_target_each_iteration(self):
        _ControlRecordingSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
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

    def test_compacted_multi_world_articulation_end_is_rebased(self):
        """articulation_end must be rebased to local joint ids, matching articulation_start.

        Regression: compaction rebased articulation_start but left articulation_end as
        global joint indices, so a non-first-world articulation got an out-of-bounds
        end (e.g. end=9 in an 8-joint view), corrupting solver FK (fixed base displaced).
        """
        world_count = 2
        template = newton.ModelBuilder(gravity=0.0)

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

        builder = newton.ModelBuilder(gravity=0.0)
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

    def test_duplicate_body_proxy_mapping_ids_are_rejected(self):
        builder = newton.ModelBuilder(gravity=0.0)
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
        builder = newton.ModelBuilder(gravity=0.0)
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

        builder = newton.ModelBuilder(gravity=0.0)
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

        builder = newton.ModelBuilder(gravity=0.0)
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
        builder = newton.ModelBuilder(gravity=0.0)
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
        builder = newton.ModelBuilder(gravity=0.0)
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

    def test_particle_proxy_maps_proxy_indexed_feedback_to_source(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleHookSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
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
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(-0.02, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05, flags=flags)
        builder.add_particle(pos=(0.02, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05, flags=flags)
        model = builder.finalize(device="cpu")
        solver = SolverXPBD(model=model, iterations=4, soft_contact_relaxation=1.0)

        state_0 = model.state()
        state_1 = model.state()
        contacts = model.contacts()
        q_before = state_0.particle_q.numpy().copy()

        solver.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_allclose(state_1.particle_q.numpy(), q_before, atol=1.0e-6)

    def test_xpbd_ignores_proxy_static_particle_contacts(self):
        proxy_flags = int(newton.ParticleFlags.ACTIVE) | int(newton.ParticleFlags.PROXY)
        builder = newton.ModelBuilder(gravity=0.0)
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
        contacts = model.contacts()
        q_before = state_0.particle_q.numpy().copy()

        solver.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_allclose(state_1.particle_q.numpy(), q_before, atol=1.0e-6)

    def test_xpbd_ignores_proxy_particle_proxy_body_contacts(self):
        proxy_particle_flags = int(newton.ParticleFlags.ACTIVE) | int(newton.ParticleFlags.PROXY)
        builder = newton.ModelBuilder(gravity=0.0)
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
        contacts = model.collide(state_0)
        self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
        q_before = state_0.particle_q.numpy().copy()

        solver.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_allclose(state_1.particle_q.numpy(), q_before, atol=1.0e-6)


class TestSmoothTeleportRecovery(unittest.TestCase):
    """Validate that sync + smooth teleportation + VBD forward integration
    recovers the driving solver's end-of-step positions when there are no
    external forces or collisions.

    The coupling chain for proxy bodies each substep is:

        1. Sync: body_q <- mjc.state_0.body_q, body_qd <- mjc.state_1.body_qd
        2. Smooth teleport: fold (body_q - body_q_prev) into body_qd,
           reset body_q <- body_q_prev
        3. Forward integrate (semi-implicit Euler, zero forces/gravity)

    With com = 0 the integrated position is:

        p_new = p_prev + (v_mjc_end + (p_mjc_begin - p_prev) / dt) * dt
              = p_mjc_begin + v_mjc_end * dt
              = p_mjc_end

    For the angular part with spherical inertia (no coriolis), the same
    identity holds exactly when there is no angular jump and to first
    order when there is one.
    """

    device = "cpu"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_angle_to_quat(axis, angle):
        """Convert axis-angle to quaternion [qx, qy, qz, qw]."""
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        s = np.sin(angle / 2.0)
        c = np.cos(angle / 2.0)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])

    @staticmethod
    def _quat_angle_error(q1, q2):
        """Return the angular distance in radians between two quaternions."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
        return 2.0 * np.arccos(dot)

    def _assert_quat_close(self, q1, q2, angle_tol_rad, msg=""):
        """Assert two quaternions are within *angle_tol_rad* of each other."""
        err = self._quat_angle_error(q1, q2)
        self.assertLessEqual(
            err,
            angle_tol_rad,
            f"{msg}angular error {np.degrees(err):.4f} deg exceeds "
            f"tolerance {np.degrees(angle_tol_rad):.4f} deg "
            f"(q1={q1}, q2={q2})",
        )

    def _run_sync_teleport_forward(
        self,
        p_mjc_begin,
        v_mjc_end,
        p_prev,
        dt,
        w_mjc_end=None,
        r_prev=None,
        r_mjc_begin=None,
    ):
        """Run sync -> teleport -> forward-step for one proxy body.

        Returns:
            body_q: Integrated body transform (numpy, shape [7]).
            body_q_prev_out: body_q_prev after forward step (numpy, shape [7]).
        """
        dev = self.device
        if w_mjc_end is None:
            w_mjc_end = np.zeros(3)
        if r_prev is None:
            r_prev = np.array([0.0, 0.0, 0.0, 1.0])
        if r_mjc_begin is None:
            r_mjc_begin = r_prev.copy()

        # "MuJoCo" output arrays (1 body, index 0 is the proxy)
        mjc_body_q = wp.array(
            [wp.transform(p_mjc_begin, wp.quat(*r_mjc_begin))],
            dtype=wp.transform,
            device=dev,
        )
        mjc_body_qd = wp.array(
            [
                wp.spatial_vector(
                    v_mjc_end[0],
                    v_mjc_end[1],
                    v_mjc_end[2],
                    w_mjc_end[0],
                    w_mjc_end[1],
                    w_mjc_end[2],
                )
            ],
            dtype=wp.spatial_vector,
            device=dev,
        )

        # "VBD" arrays -- will be overwritten by sync
        vbd_body_q = wp.array(
            [wp.transform([0.0, 0.0, 0.0], wp.quat_identity())],
            dtype=wp.transform,
            device=dev,
        )
        vbd_body_qd = wp.zeros(1, dtype=wp.spatial_vector, device=dev)

        # Proxy mapping: identity (body 0 <-> body 0)
        src_to_dst = wp.array([0], dtype=int, device=dev)
        proxy_ids = wp.array([0], dtype=int, device=dev)

        # VBD's previous end-of-step transform
        body_q_prev = wp.array(
            [wp.transform(p_prev, wp.quat(*r_prev))],
            dtype=wp.transform,
            device=dev,
        )

        # --- Step 1: Sync ---
        wp.launch(
            sync_proxy_states_kernel,
            dim=1,
            inputs=[
                mjc_body_q,
                mjc_body_qd,
                src_to_dst,
                vbd_body_q,
                vbd_body_qd,
            ],
            device=dev,
        )

        # --- Step 2: Smooth teleportation ---
        wp.launch(
            smooth_proxy_teleportation_kernel,
            dim=1,
            inputs=[dt, proxy_ids, vbd_body_q, vbd_body_qd, body_q_prev],
            device=dev,
        )

        # --- Step 3: Forward integration (zero gravity, zero forces) ---
        gravity = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=dev)
        body_world = wp.array([0], dtype=wp.int32, device=dev)
        body_f = wp.zeros(1, dtype=wp.spatial_vector, device=dev)
        body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=dev)
        body_inertia = wp.array([wp.mat33(np.eye(3))], dtype=wp.mat33, device=dev)
        body_inv_mass = wp.array([1.0], dtype=float, device=dev)
        body_inv_inertia = wp.array([wp.mat33(np.eye(3))], dtype=wp.mat33, device=dev)
        body_inertia_q = wp.zeros(1, dtype=wp.transform, device=dev)

        wp.launch(
            forward_step_rigid_bodies,
            dim=1,
            inputs=[
                dt,
                gravity,
                body_world,
                body_f,
                body_com,
                body_inertia,
                body_inv_mass,
                body_inv_inertia,
                vbd_body_q,
                vbd_body_qd,
                body_inertia_q,
            ],
            device=dev,
        )

        return vbd_body_q.numpy()[0], body_q_prev.numpy()[0]

    def _reference_forward(self, p, v, r, w, dt):
        """Run forward_step_rigid_bodies directly (no sync/teleport) to get
        the reference end-of-step transform.

        Returns:
            body_q: Integrated body transform (numpy, shape [7]).
        """
        dev = self.device
        body_q = wp.array(
            [wp.transform(p, wp.quat(*r))],
            dtype=wp.transform,
            device=dev,
        )
        body_qd = wp.array(
            [wp.spatial_vector(v[0], v[1], v[2], w[0], w[1], w[2])],
            dtype=wp.spatial_vector,
            device=dev,
        )
        gravity = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=dev)
        body_world = wp.array([0], dtype=wp.int32, device=dev)
        body_f = wp.zeros(1, dtype=wp.spatial_vector, device=dev)
        body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=dev)
        body_inertia = wp.array([wp.mat33(np.eye(3))], dtype=wp.mat33, device=dev)
        body_inv_mass = wp.array([1.0], dtype=float, device=dev)
        body_inv_inertia = wp.array([wp.mat33(np.eye(3))], dtype=wp.mat33, device=dev)
        body_inertia_q = wp.zeros(1, dtype=wp.transform, device=dev)

        wp.launch(
            forward_step_rigid_bodies,
            dim=1,
            inputs=[
                dt,
                gravity,
                body_world,
                body_f,
                body_com,
                body_inertia,
                body_inv_mass,
                body_inv_inertia,
                body_q,
                body_qd,
                body_inertia_q,
            ],
            device=dev,
        )
        return body_q.numpy()[0]

    # ------------------------------------------------------------------
    # Translational tests
    # ------------------------------------------------------------------

    def test_teleport_jump(self):
        """body_q_prev != p_mjc_begin: teleport jump absorbed, VBD recovers p_mjc_end."""
        dt = 1.0 / 60.0
        p_begin = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, -0.3, 0.1])
        p_prev = np.array([0.8, 2.2, 2.7])

        p_expected = p_begin + v * dt
        result, _ = self._run_sync_teleport_forward(p_begin, v, p_prev, dt)
        np.testing.assert_allclose(result[:3], p_expected, atol=1e-6)

    def test_multi_step_chain(self):
        """Run several steps in sequence; each step should recover the
        analytic free-flight trajectory p(t) = p0 + v * t."""
        dt = 1.0 / 60.0
        p0 = np.array([0.0, 1.0, 0.0])
        v = np.array([2.0, 0.0, -1.0])
        n_steps = 10

        # Introduce an initial jump on the very first step
        p_prev = p0 + np.array([0.05, -0.02, 0.01])

        for i in range(n_steps):
            p_begin = p0 + v * dt * i
            p_end_expected = p0 + v * dt * (i + 1)

            result, _ = self._run_sync_teleport_forward(p_begin, v, p_prev, dt)
            np.testing.assert_allclose(
                result[:3],
                p_end_expected,
                atol=1e-6,
                err_msg=f"Step {i}: VBD position does not match MuJoCo end-of-step",
            )

            # Simulate update_body_velocity advancing body_q_prev to
            # the final pose.
            p_prev = result[:3].copy()

    # ------------------------------------------------------------------
    # Angular tests
    # ------------------------------------------------------------------

    def test_angular_jump_off_axis(self):
        """Angular jump around a different axis from the spinning axis.

        Cross-axis coupling introduces a second-order error
        O(dt * jump_angle * omega), which for a 5 deg jump at 3 rad/s
        is approximately 0.07 deg.
        """
        dt = 1.0 / 60.0
        p = np.array([1.0, 2.0, 0.0])
        v = np.array([0.5, 0.0, 0.0])

        w = np.array([0.0, 3.0, 0.0])
        r_begin = self._axis_angle_to_quat([1, 0, 0], np.radians(10))
        r_prev = self._axis_angle_to_quat([1, 0, 0], np.radians(5))

        ref = self._reference_forward(p, v, r_begin, w, dt)
        result, _ = self._run_sync_teleport_forward(p, v, p, dt, w_mjc_end=w, r_prev=r_prev, r_mjc_begin=r_begin)

        np.testing.assert_allclose(result[:3], ref[:3], atol=1e-6)
        self._assert_quat_close(result[3:], ref[3:], angle_tol_rad=np.radians(0.15))


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
        _add_equality_constraint(builder, constraint_type=newton.EqType.CONNECT, body1=0, body2=1)
        _add_equality_constraint(builder, constraint_type=newton.EqType.CONNECT, body1=2, body2=3)
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


if __name__ == "__main__":
    unittest.main()
