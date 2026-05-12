# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the coupled solver prototype."""

import unittest
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled.proxy_utils import (
    smooth_proxy_teleportation_kernel,
    subtract_proxy_forces_kernel,
    subtract_proxy_particle_forces_kernel,
    sync_proxy_states_kernel,
)
from newton._src.solvers.vbd.rigid_vbd_kernels import forward_step_rigid_bodies
from newton.solvers import (
    ModelView,
    SolverAdmmCoupled,
    SolverBase,
    SolverCoupled,
    SolverNotifyFlags,
    SolverProxyCoupled,
    SolverSemiImplicit,
    SolverXPBD,
)


@wp.kernel(enable_backward=False)
def _set_external_forces_kernel(body_f: wp.array[wp.spatial_vector], particle_f: wp.array[wp.vec3]):
    body_f[0] = wp.spatial_vector(wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0))
    particle_f[0] = wp.vec3(7.0, 8.0, 9.0)


@wp.kernel(enable_backward=False)
def _mutate_body_qd_kernel(body_qd: wp.array[wp.spatial_vector]):
    body_qd[0] = wp.spatial_vector(wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0))


@wp.kernel(enable_backward=False)
def _mutate_body_qd_at_kernel(body_qd: wp.array[wp.spatial_vector], body_id: int):
    body_qd[body_id] = wp.spatial_vector(wp.vec3(7.0, 8.0, 9.0), wp.vec3(10.0, 11.0, 12.0))


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


@wp.kernel(enable_backward=False)
def _set_first_particle_force_kernel(particle_f: wp.array[wp.vec3]):
    particle_f[0] = wp.vec3(2.0, 3.0, 4.0)


@wp.kernel(enable_backward=False)
def _fill_effective_body_inertia_kernel(out_mass: wp.array[float], out_inertia: wp.array[wp.mat33]):
    i = wp.tid()
    out_mass[i] = 6.0
    out_inertia[i] = wp.mat33(2.0, 0.5, 0.0, 0.5, 3.0, 0.25, 0.0, 0.25, 5.0)


class _InputMutatingSolver(SolverBase):
    """Test solver that mutates ``state_in`` to model solvers with input caches."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.input_qd = []
        self.instances.append(self)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        self.input_qd.append(state_in.body_qd.numpy()[0].copy())
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)
        wp.launch(_mutate_body_qd_kernel, dim=1, inputs=[state_in.body_qd], device=self.model.device)


class _BodyInputMutatingCopySolver(SolverBase):
    """Test solver that records and mutates a selected input body velocity."""

    instances: ClassVar[list] = []

    def __init__(self, model, mutate_body=1):
        super().__init__(model)
        self.mutate_body = mutate_body
        self.input_body_qd = []
        self.instances.append(self)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        self.input_body_qd.append(state_in.body_qd.numpy().copy())
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)
        wp.launch(
            _mutate_body_qd_at_kernel,
            dim=1,
            inputs=[state_in.body_qd, self.mutate_body],
            device=self.model.device,
        )


class _BodyForceRecordingSolver(SolverBase):
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


class _ParticleForceRecordingSolver(SolverBase):
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


class _InPlaceRecordingParticleSolver(SolverBase):
    """Test solver that records whether it was stepped in-place."""

    instances: ClassVar[dict[str, "_InPlaceRecordingParticleSolver"]] = {}

    def __init__(self, model):
        super().__init__(model)
        self.in_place_calls = []
        self.instances[model.name] = self

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        self.in_place_calls.append(state_in is state_out)
        if state_in is not state_out:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
        wp.launch(_kick_proxy_particle_kernel, dim=1, inputs=[state_out.particle_qd], device=self.model.device)


class _ParticleForceNotifySolver(_ParticleForceRecordingSolver):
    """Test solver that observes public particle-force input notifications."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.notified_flags = []
        self.notified_particle_f = []

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.NOTIFY_INPUT_STATE_UPDATE: SolverBase.CouplingCapability.CUSTOM}

    def coupling_notify_input_state_update(self, state, flags, dt=0.0):
        del dt
        flags = SolverBase.CouplingInputStateFlags(flags)
        self.notified_flags.append(flags)
        if flags & SolverBase.CouplingInputStateFlags.PARTICLE_F:
            self.notified_particle_f.append(state.particle_f.numpy().copy())


class _CustomEffectiveMassParticleSolver(_ParticleForceRecordingSolver):
    """Particle solver that reports a custom effective mass."""

    instances: ClassVar[list] = []

    def __init__(self, model, effective_mass=1.0):
        super().__init__(model)
        self.effective_mass = float(effective_mass)
        self.queried_endpoints = []

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.EFFECTIVE_MASS_DIAGONAL: SolverBase.CouplingCapability.CUSTOM}

    def coupling_eval_effective_mass(self, endpoint_kind, endpoint_index, endpoint_local_pos, out):
        del endpoint_kind, endpoint_local_pos
        self.queried_endpoints.extend(int(i) for i in endpoint_index.numpy())
        out.fill_(self.effective_mass)


class _CustomEffectiveMassBodySolver(SolverBase):
    """Body solver that reports a custom scalar effective mass."""

    instances: ClassVar[list] = []

    def __init__(self, model, effective_mass=1.0):
        super().__init__(model)
        self.effective_mass = float(effective_mass)
        self.queried_endpoints = []
        self.instances.append(self)

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.EFFECTIVE_MASS_DIAGONAL: SolverBase.CouplingCapability.CUSTOM}

    def coupling_eval_effective_mass(self, endpoint_kind, endpoint_index, endpoint_local_pos, out):
        del endpoint_kind, endpoint_local_pos
        self.queried_endpoints.extend(int(i) for i in endpoint_index.numpy())
        out.fill_(self.effective_mass)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)


class _CustomEffectiveBodyInertiaSolver(SolverBase):
    """Body solver that reports a custom full effective inertia."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.queried_endpoints = []
        self.received_inertia_buffer = False
        self.instances.append(self)

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.EFFECTIVE_MASS_BLOCK: SolverBase.CouplingCapability.CUSTOM}

    def coupling_eval_effective_mass_block(
        self,
        endpoint_kind,
        endpoint_index,
        endpoint_local_pos,
        out_mass,
        out_inertia=None,
    ):
        del endpoint_kind, endpoint_local_pos
        self.queried_endpoints.extend(int(i) for i in endpoint_index.numpy())
        self.received_inertia_buffer = out_inertia is not None
        if out_inertia is None:
            raise ValueError("Body effective mass block requires an inertia output")
        wp.launch(
            _fill_effective_body_inertia_kernel,
            dim=out_mass.shape[0],
            inputs=[out_mass, out_inertia],
            device=self.model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)


class _ProxyParticleKickSolver(SolverBase):
    """Destination test solver that applies a fixed impulse to proxy particle 0."""

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)
        wp.launch(_kick_proxy_particle_kernel, dim=1, inputs=[state_out.particle_qd], device=self.model.device)


class _ProxyParticleNotifyInputStateSolver(_ProxyParticleKickSolver):
    """Destination solver that observes proxy input-state updates."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.notified_flags = []
        self.notified_particle_qd = []
        self.instances.append(self)

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.NOTIFY_INPUT_STATE_UPDATE: SolverBase.CouplingCapability.CUSTOM}

    def coupling_notify_input_state_update(self, state, flags, dt=0.0):
        del dt
        self.notified_flags.append(SolverBase.CouplingInputStateFlags(flags))
        self.notified_particle_qd.append(state.particle_qd.numpy().copy())


class _ProxyParticleHookSolver(SolverBase):
    """Destination test solver that exposes particle proxy rewind/harvest hooks."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.rewind_calls = 0
        self.harvest_calls = 0
        self.instances.append(self)

    def coupling_capabilities(self):
        custom = SolverBase.CouplingCapability.CUSTOM
        return {
            SolverBase.CouplingHooks.PARTICLE_PROXY_REWIND_VELOCITY: custom,
            SolverBase.CouplingHooks.PARTICLE_PROXY_HARVEST: custom,
        }

    def coupling_rewind_proxy_particle_velocity(
        self,
        particle_local_to_proxy_global,
        state,
        coupling_forces,
        dt,
    ):
        del particle_local_to_proxy_global, state, coupling_forces, dt
        self.rewind_calls += 1

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global,
        out_particle_f,
        *,
        state=None,
        state_out=None,
        contacts=None,
        dt=0.0,
    ):
        del state, state_out, contacts, dt
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


class _ParticleHarvestStateRecordingSolver(SolverBase):
    """Destination solver that records which states are passed to custom harvest."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.harvest_particle_qd = None
        self.harvest_particle_qd_out = None
        self.harvest_contacts = None
        self.instances.append(self)

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.PARTICLE_PROXY_HARVEST: SolverBase.CouplingCapability.CUSTOM}

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global,
        out_particle_f,
        *,
        state=None,
        state_out=None,
        contacts=None,
        dt=0.0,
    ):
        del particle_local_to_proxy_global, out_particle_f, dt
        self.harvest_particle_qd = state.particle_qd.numpy().copy()
        self.harvest_particle_qd_out = state_out.particle_qd.numpy().copy()
        self.harvest_contacts = contacts

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)
        wp.launch(_kick_proxy_particle_kernel, dim=1, inputs=[state_out.particle_qd], device=self.model.device)


class _BodyHarvestStateRecordingSolver(SolverBase):
    """Destination solver that records which state is passed to custom harvest."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.harvest_body_qd = None
        self.instances.append(self)

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.BODY_PROXY_HARVEST: SolverBase.CouplingCapability.CUSTOM}

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global,
        out_body_f,
        *,
        state=None,
        state_out=None,
        contacts=None,
        dt=0.0,
    ):
        del body_local_to_proxy_global, out_body_f, state_out, contacts, dt
        self.harvest_body_qd = state.body_qd.numpy().copy()

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)
        wp.launch(
            _mutate_body_qd_at_kernel,
            dim=1,
            inputs=[state_out.body_qd, 0],
            device=self.model.device,
        )


class _ProxyBodyHookSolver(SolverBase):
    """Destination test solver that writes proxy-indexed body feedback."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.harvest_calls = 0
        self.instances.append(self)

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.BODY_PROXY_HARVEST: SolverBase.CouplingCapability.CUSTOM}

    def coupling_harvest_proxy_wrenches(
        self, body_local_to_proxy_global, out_body_f, *, state=None, state_out=None, contacts=None, dt=0.0
    ):
        del state, state_out, contacts, dt
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


class _StepCountingCopySolver(SolverBase):
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


class _ProxyContactRecordingSolver(_StepCountingCopySolver):
    """Test solver that records proxy-contact preparation calls."""

    def __init__(self, model):
        super().__init__(model)
        self.prepare_contact_calls = 0
        self.prepare_contact_fresh_flags = []
        self.prepare_contacts = []

    def coupling_prepare_proxy_contacts(self, state, contacts, *, contacts_freshly_detected=False):
        del state
        self.prepare_contact_calls += 1
        self.prepare_contacts.append(contacts)
        self.prepare_contact_fresh_flags.append(bool(contacts_freshly_detected))
        return contacts


class _CustomProxyContactRecordingSolver(_ProxyContactRecordingSolver):
    """Test solver that advertises custom proxy-contact preparation."""

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.PROXY_CONTACT_PREPARE: SolverBase.CouplingCapability.CUSTOM}


class _UnsupportedProxyContactRecordingSolver(_ProxyContactRecordingSolver):
    """Test solver that rejects proxy-contact preparation."""

    def coupling_capabilities(self):
        return {SolverBase.CouplingHooks.PROXY_CONTACT_PREPARE: SolverBase.CouplingCapability.UNSUPPORTED}


class _FakeProxyCollisionPipeline:
    """Minimal collision pipeline used to test proxy-coupler scheduling."""

    def __init__(self, device):
        self.contacts_obj = newton.Contacts(0, 0, device=device)
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

    def test_zero_body_mass(self):
        """zero_body_mass should zero inv_mass for specified indices."""
        view = ModelView(self.model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        view.zero_body_mass(indices)

        inv_mass = view.body_inv_mass.numpy()
        # Body 0 should be unchanged (non-zero)
        self.assertGreater(inv_mass[0], 0.0)
        # Body 1 should be zeroed
        self.assertEqual(inv_mass[1], 0.0)

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

    def test_deactivate_particles(self):
        """deactivate_particles should clear only view-local active flags."""
        builder = newton.ModelBuilder()
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=(0.1, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        view = ModelView(model, "test")
        indices = wp.array([1], dtype=int, device="cpu")
        view.deactivate_particles(indices)

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

    def test_repr(self):
        view = ModelView(self.model, "vbd")
        view.body_inv_mass = wp.zeros(2, dtype=float, device="cpu")
        r = repr(view)
        self.assertIn("vbd", r)
        self.assertIn("body_inv_mass", r)


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

    def test_construction(self):
        """SolverCoupled should construct sub-solvers and ModelViews."""
        coupled = SolverCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
            ],
        )

        solver_a = coupled.get_solver("A")
        solver_b = coupled.get_solver("B")
        self.assertIsInstance(solver_a, SolverSemiImplicit)
        self.assertIsInstance(solver_b, SolverSemiImplicit)

        view_a = coupled.get_view("A")
        view_b = coupled.get_view("B")
        # Solver A's view should have body 1 zeroed
        self.assertEqual(view_a.body_inv_mass.numpy()[1], 0.0)
        self.assertGreater(view_a.body_inv_mass.numpy()[0], 0.0)
        # Solver B's view should have body 0 zeroed
        self.assertEqual(view_b.body_inv_mass.numpy()[0], 0.0)
        self.assertGreater(view_b.body_inv_mass.numpy()[1], 0.0)

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
        view_a = coupled.get_view("A")
        view_b = coupled.get_view("B")

        self.assertNotEqual(int(view_a.shape_flags.numpy()[0]) & collide, 0)
        self.assertEqual(int(view_a.shape_flags.numpy()[1]) & collide, 0)
        self.assertEqual(view_a.shape_contact_pair_count, 0)

        self.assertEqual(int(view_b.shape_flags.numpy()[0]) & collide, 0)
        self.assertNotEqual(int(view_b.shape_flags.numpy()[1]) & collide, 0)
        self.assertEqual(view_b.shape_contact_pair_count, 0)

        self.assertEqual(self.model.shape_contact_pair_count, 1)

    def test_proxy_shape_visibility_keeps_proxy_contact_pairs(self):
        """Proxy destination views should keep shape pairs touching proxy bodies."""
        coupled = SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0], shapes=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1], shapes=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(source="A", destination="B", bodies=[0]),
                ],
            ),
        )

        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        view_a = coupled.get_view("A")
        view_b = coupled.get_view("B")

        self.assertEqual(view_a.shape_contact_pair_count, 0)
        self.assertNotEqual(int(view_b.shape_flags.numpy()[0]) & collide, 0)
        self.assertNotEqual(int(view_b.shape_flags.numpy()[1]) & collide, 0)
        self.assertEqual(view_b.shape_contact_pair_count, 1)
        np.testing.assert_array_equal(view_b.shape_contact_pairs.numpy(), np.array([[0, 1]], dtype=np.int32))

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
                    solver=_InPlaceRecordingParticleSolver,
                    particles=[0],
                    in_place=True,
                ),
            ],
        )

        self.assertIs(coupled._entries["particles"].state_0, coupled._entries["particles"].state_1)
        state = model.state()

        coupled.step(state, state, control=None, contacts=None, dt=1.0 / 60.0)

        solver = _InPlaceRecordingParticleSolver.instances["particles"]
        self.assertEqual(solver.in_place_calls, [True])
        np.testing.assert_allclose(state.particle_qd.numpy()[0], np.array([0.0, 2.0, 0.0]))

    def test_entry_in_place_rejects_substeps(self):
        """In-place entries are intentionally limited to single substeps."""
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        model = builder.finalize(device="cpu")

        with self.assertRaises(ValueError):
            SolverCoupled(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="particles",
                        solver=_InPlaceRecordingParticleSolver,
                        particles=[0],
                        substeps=2,
                        in_place=True,
                    ),
                ],
            )

    def test_distribute_state_preserves_external_forces(self):
        """External body and particle forces should reach sub-solver states."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0], particles=[0]),
            ],
        )

        state = model.state()
        wp.launch(_set_external_forces_kernel, dim=1, inputs=[state.body_f, state.particle_f], device=model.device)

        coupled._distribute_state(state)

        entry = coupled._entries["A"]
        np.testing.assert_allclose(entry.state_0.body_f.numpy()[0], np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        np.testing.assert_allclose(entry.state_0.particle_f.numpy()[0], np.array([7.0, 8.0, 9.0]))

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
        view_a_flags = coupled.get_view("A").particle_flags.numpy()
        view_b_flags = coupled.get_view("B").particle_flags.numpy()
        parent_flags = model.particle_flags.numpy()

        self.assertNotEqual(view_a_flags[0] & active, 0)
        self.assertEqual(view_a_flags[1] & active, 0)
        self.assertEqual(view_b_flags[0] & active, 0)
        self.assertNotEqual(view_b_flags[1] & active, 0)
        self.assertNotEqual(parent_flags[0] & active, 0)
        self.assertNotEqual(parent_flags[1] & active, 0)

    def test_proxy_destination_view_marks_proxy_flags(self):
        """Proxy destination views should expose proxy bodies through body_flags."""
        coupled = SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="A", solver=SolverSemiImplicit, bodies=[0]),
                SolverCoupled.Entry(name="B", solver=SolverSemiImplicit, bodies=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(source="A", destination="B", bodies=[0]),
                ],
            ),
        )

        view_a = coupled.get_view("A")
        view_b = coupled.get_view("B")
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
            SolverProxyCoupled(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="a", solver=SolverSemiImplicit, bodies=[0]),
                    SolverCoupled.Entry(name="b", solver=SolverSemiImplicit, bodies=[1]),
                    SolverCoupled.Entry(name="c", solver=SolverSemiImplicit, bodies=[2]),
                ],
                coupling=SolverProxyCoupled.CouplingProxy(
                    proxies=[
                        SolverProxyCoupled.Proxy(source="a", destination="b", bodies=[0]),
                    ],
                ),
            )

    def test_admm_gamma_zero_restores_mutated_input_velocities(self):
        """ADMM iterations should restore input velocities even without a proximal term."""
        _InputMutatingSolver.instances.clear()
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="A", solver=_InputMutatingSolver, bodies=[0]),
            ],
            coupling=SolverAdmmCoupled.CouplingAdmm(iterations=2, gamma=0.0),
        )

        state_0 = model.state()
        state_1 = model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 60.0)

        solver = _InputMutatingSolver.instances[-1]
        self.assertEqual(len(solver.input_qd), 2)
        np.testing.assert_allclose(solver.input_qd[0], np.zeros(6))
        np.testing.assert_allclose(solver.input_qd[1], np.zeros(6))


class TestSolverCoupledBodyProxyInertia(unittest.TestCase):
    """Body proxy mappings install full proxy inertia tensors."""

    def test_proxy_body_uses_source_inertia_shape(self):
        _StepCountingCopySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=2.0, inertia=wp.mat33(5.0, 0.1, 0.0, 0.1, 6.0, 0.2, 0.0, 0.2, 7.0))
        builder.add_body(mass=10.0, inertia=wp.mat33(np.eye(3) * 100.0))
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                        proxy_bodies=[1],
                        mass_scale=0.5,
                    ),
                ],
            ),
        )

        dst_view = coupled.get_view("dst")
        expected_inertia = np.array([[2.5, 0.05, 0.0], [0.05, 3.0, 0.1], [0.0, 0.1, 3.5]])
        np.testing.assert_allclose(dst_view.body_mass.numpy()[1], 1.0)
        np.testing.assert_allclose(dst_view.body_inertia.numpy()[1], expected_inertia)
        np.testing.assert_allclose(dst_view.body_inv_inertia.numpy()[1], np.linalg.inv(expected_inertia), rtol=1.0e-6)
        dst_solver = _StepCountingCopySolver.instances["dst"]
        self.assertTrue(
            any(flags & int(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES) for flags in dst_solver.model_notify_flags)
        )

    def test_body_proxy_maps_proxy_indexed_feedback_to_source(self):
        _BodyForceRecordingSolver.instances.clear()
        _ProxyBodyHookSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_BodyForceRecordingSolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyBodyHookSolver, bodies=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
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
        np.testing.assert_allclose(src_solver.input_body_f[1][0], expected, atol=1.0e-6)
        np.testing.assert_allclose(src_solver.input_body_f[1][2], np.zeros(6), atol=1.0e-6)

    def test_proxy_body_diagonal_effective_mass_scales_source_inertia(self):
        _CustomEffectiveMassBodySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=2.0, inertia=wp.mat33(4.0, 0.1, 0.0, 0.1, 6.0, 0.2, 0.0, 0.2, 8.0))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=_CustomEffectiveMassBodySolver,
                    bodies=[0],
                    solver_kwargs={"effective_mass": 8.0},
                ),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                        proxy_bodies=[1],
                        mass_scale=0.5,
                    ),
                ],
            ),
        )

        solver = _CustomEffectiveMassBodySolver.instances[-1]
        dst_view = coupled.get_view("dst")
        expected_inertia = np.array([[8.0, 0.2, 0.0], [0.2, 12.0, 0.4], [0.0, 0.4, 16.0]])

        self.assertEqual(len(solver.queried_endpoints), 1)
        np.testing.assert_allclose(dst_view.body_mass.numpy()[1], 4.0)
        np.testing.assert_allclose(dst_view.body_inertia.numpy()[1], expected_inertia)
        np.testing.assert_allclose(dst_view.body_inv_inertia.numpy()[1], np.linalg.inv(expected_inertia), rtol=1.0e-6)

    def test_proxy_body_can_use_custom_effective_mass_block(self):
        _CustomEffectiveBodyInertiaSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=2.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_CustomEffectiveBodyInertiaSolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                        proxy_bodies=[1],
                        mass_scale=0.5,
                    ),
                ],
            ),
        )

        solver = _CustomEffectiveBodyInertiaSolver.instances[-1]
        dst_view = coupled.get_view("dst")
        expected_inertia = np.array([[1.0, 0.25, 0.0], [0.25, 1.5, 0.125], [0.0, 0.125, 2.5]])

        self.assertTrue(solver.received_inertia_buffer)
        self.assertEqual(len(solver.queried_endpoints), 1)
        self.assertEqual(solver.queried_endpoints[0], 0)
        np.testing.assert_allclose(dst_view.body_mass.numpy()[1], 3.0)
        np.testing.assert_allclose(dst_view.body_inertia.numpy()[1], expected_inertia)
        np.testing.assert_allclose(dst_view.body_inv_inertia.numpy()[1], np.linalg.inv(expected_inertia), rtol=1.0e-6)


class TestSolverCoupledProxyContactHooks(unittest.TestCase):
    """Proxy contact preparation follows the coupling capability map."""

    def _make_coupled(self, dst_solver):
        _StepCountingCopySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")
        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=dst_solver, bodies=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                    ),
                ],
            ),
        )
        return model, coupled

    def test_default_proxy_contact_prepare_uses_generic_path(self):
        model, coupled = self._make_coupled(_ProxyContactRecordingSolver)

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=1.0 / 60.0)

        dst_solver = _StepCountingCopySolver.instances["dst"]
        self.assertEqual(dst_solver.prepare_contact_calls, 0)

    def test_custom_proxy_contact_prepare_is_called(self):
        model, coupled = self._make_coupled(_CustomProxyContactRecordingSolver)

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=1.0 / 60.0)

        dst_solver = _StepCountingCopySolver.instances["dst"]
        self.assertEqual(dst_solver.prepare_contact_calls, 1)
        self.assertEqual(dst_solver.prepare_contact_fresh_flags, [False])

    def test_proxy_collision_pipeline_respects_interval(self):
        _StepCountingCopySolver.instances.clear()
        pipeline_holder = []
        factory_models = []

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        def make_pipeline(view):
            factory_models.append(view)
            pipeline = _FakeProxyCollisionPipeline(model.device)
            pipeline_holder.append(pipeline)
            return pipeline

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_CustomProxyContactRecordingSolver, bodies=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                        collision_pipeline=make_pipeline,
                        collide_interval=2,
                    ),
                ],
                iterations=3,
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=1.0 / 60.0)

        self.assertEqual(len(factory_models), 1)
        self.assertEqual(factory_models[0].name, "dst")
        self.assertEqual(len(pipeline_holder), 1)
        self.assertEqual(pipeline_holder[0].contacts_calls, 1)
        self.assertEqual(pipeline_holder[0].collide_calls, 2)
        dst_solver = _StepCountingCopySolver.instances["dst"]
        self.assertEqual(dst_solver.prepare_contact_calls, 3)
        self.assertEqual(dst_solver.prepare_contact_fresh_flags, [True, False, True])

    def test_proxy_collision_pipeline_none_return_uses_outer_contacts(self):
        _StepCountingCopySolver.instances.clear()
        factory_models = []

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")
        outer_contacts = newton.Contacts(0, 0, device=model.device)

        def make_pipeline(view):
            factory_models.append(view)
            return None

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_CustomProxyContactRecordingSolver, bodies=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        bodies=[0],
                        collision_pipeline=make_pipeline,
                    ),
                ],
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=outer_contacts, dt=1.0 / 60.0)

        self.assertEqual(len(factory_models), 1)
        self.assertEqual(factory_models[0].name, "dst")
        self.assertIsNone(coupled.get_proxy_contacts("src", "dst"))
        dst_solver = _StepCountingCopySolver.instances["dst"]
        self.assertEqual(dst_solver.prepare_contact_calls, 1)
        self.assertEqual(dst_solver.prepare_contact_fresh_flags, [False])
        self.assertEqual(dst_solver.prepare_contacts, [outer_contacts])

    def test_proxy_collide_interval_requires_pipeline(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        with self.assertRaises(ValueError):
            SolverProxyCoupled(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0]),
                    SolverCoupled.Entry(name="dst", solver=_CustomProxyContactRecordingSolver, bodies=[1]),
                ],
                coupling=SolverProxyCoupled.CouplingProxy(
                    proxies=[
                        SolverProxyCoupled.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[0],
                            collide_interval=2,
                        ),
                    ],
                ),
            )

    def test_lagged_proxy_rejects_in_place_source_entry(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "in_place=True"):
            SolverProxyCoupled(
                model=model,
                entries=[
                    SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0], in_place=True),
                    SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver, bodies=[1]),
                ],
                coupling=SolverProxyCoupled.CouplingProxy(
                    proxies=[
                        SolverProxyCoupled.Proxy(
                            source="src",
                            destination="dst",
                            bodies=[0],
                        ),
                    ],
                ),
            )

    def test_unsupported_proxy_contact_prepare_rejects_body_proxy(self):
        model, coupled = self._make_coupled(_UnsupportedProxyContactRecordingSolver)

        with self.assertRaises(NotImplementedError):
            coupled.step(model.state(), model.state(), control=None, contacts=None, dt=1.0 / 60.0)


class TestSolverCoupledParticleProxy(unittest.TestCase):
    """Particle proxy mappings keep proxy particles dynamic in the destination view."""

    def setUp(self):
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        self.model = builder.finalize(device="cpu")

    def _make_coupled(self, dst_solver=_ProxyParticleKickSolver):
        return SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=dst_solver, particles=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                    ),
                ],
            ),
        )

    def test_proxy_destination_view_keeps_and_scales_particle_mass(self):
        _ParticleForceRecordingSolver.instances.clear()
        coupled = self._make_coupled()

        src_view = coupled.get_view("src")
        dst_view = coupled.get_view("dst")

        self.assertEqual(src_view.particle_inv_mass.numpy()[1], 0.0)
        np.testing.assert_allclose(dst_view.particle_mass.numpy(), [1.0, 2.0])
        np.testing.assert_allclose(dst_view.particle_inv_mass.numpy(), [1.0, 0.5])
        np.testing.assert_allclose(self.model.particle_mass.numpy(), [2.0, 2.0])

    def test_proxy_particle_mass_override_notifies_model_changed(self):
        _StepCountingCopySolver.instances.clear()
        coupled = SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver, particles=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                    ),
                ],
            ),
        )

        dst_view = coupled.get_view("dst")
        np.testing.assert_allclose(dst_view.particle_mass.numpy()[0], 1.0)
        dst_solver = _StepCountingCopySolver.instances["dst"]
        self.assertTrue(any(flags & int(SolverNotifyFlags.MODEL_PROPERTIES) for flags in dst_solver.model_notify_flags))

    def test_proxy_mass_uses_source_effective_mass_hook(self):
        _CustomEffectiveMassParticleSolver.instances.clear()
        coupled = SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="src",
                    solver=_CustomEffectiveMassParticleSolver,
                    particles=[0],
                    solver_kwargs={"effective_mass": 8.0},
                ),
                SolverCoupled.Entry(name="dst", solver=_ProxyParticleKickSolver, particles=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                    ),
                ],
            ),
        )

        dst_view = coupled.get_view("dst")
        np.testing.assert_allclose(dst_view.particle_mass.numpy()[0], 4.0)
        np.testing.assert_allclose(dst_view.particle_inv_mass.numpy()[0], 0.25)
        self.assertEqual(len(_CustomEffectiveMassParticleSolver.instances[-1].queried_endpoints), 1)

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

    def test_particle_proxy_uses_default_force_input_and_notifies(self):
        _ParticleForceNotifySolver.instances.clear()
        coupled = SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceNotifySolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyParticleKickSolver, particles=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                    ),
                ],
            ),
        )

        state_0 = self.model.state()
        state_1 = self.model.state()
        wp.launch(_set_first_particle_force_kernel, dim=1, inputs=[state_0.particle_f], device=self.model.device)

        coupled.step(state_0, state_1, control=None, contacts=None, dt=0.5)

        solver = _ParticleForceNotifySolver.instances[-1]
        self.assertTrue(any(flags & SolverBase.CouplingInputStateFlags.PARTICLE_F for flags in solver.notified_flags))
        np.testing.assert_allclose(solver.input_particle_f[0][0], np.array([2.0, 3.0, 4.0]), atol=1.0e-6)
        np.testing.assert_allclose(solver.notified_particle_f[-1][0], np.array([2.0, 3.0, 4.0]), atol=1.0e-6)
        np.testing.assert_allclose(
            coupled._entries["src"].state_0.particle_f.numpy()[0],
            np.array([2.0, 3.0, 4.0]),
            atol=1.0e-6,
        )

    def test_particle_proxy_rewind_notifies_input_state_update(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleNotifyInputStateSolver.instances.clear()
        coupled = self._make_coupled(dst_solver=_ProxyParticleNotifyInputStateSolver)

        state_0 = self.model.state()
        state_1 = self.model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=0.5)

        solver = _ProxyParticleNotifyInputStateSolver.instances[-1]
        self.assertGreaterEqual(len(solver.notified_flags), 3)
        self.assertTrue(
            any(
                (flags & SolverBase.CouplingInputStateFlags.PARTICLE) == SolverBase.CouplingInputStateFlags.PARTICLE
                for flags in solver.notified_flags
            )
        )
        self.assertEqual(solver.notified_flags[-1], SolverBase.CouplingInputStateFlags.PARTICLE_QD)
        np.testing.assert_allclose(solver.notified_particle_qd[-1][0], np.zeros(3), atol=1.0e-6)

    def test_particle_proxy_iteration_restart_notifies_input_state_update(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleNotifyInputStateSolver.instances.clear()
        coupled = SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyParticleNotifyInputStateSolver, particles=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="src",
                        destination="dst",
                        particles=[0],
                        mass_scale=0.5,
                    ),
                ],
                iterations=2,
            ),
        )

        state_0 = self.model.state()
        state_1 = self.model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=0.5)

        solver = _ProxyParticleNotifyInputStateSolver.instances[-1]
        self.assertTrue(
            any(
                bool(flags & SolverBase.CouplingInputStateFlags.ITERATION_RESTART)
                and bool(flags & SolverBase.CouplingInputStateFlags.PARTICLE)
                for flags in solver.notified_flags
            )
        )

    def test_particle_proxy_uses_solver_rewind_and_harvest_hooks(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleHookSolver.instances.clear()
        coupled = self._make_coupled(dst_solver=_ProxyParticleHookSolver)

        state_0 = self.model.state()
        state_1 = self.model.state()
        dt = 0.5

        coupled.step(state_0, state_1, control=None, contacts=None, dt=dt)
        coupled.step(state_1, state_0, control=None, contacts=None, dt=dt)

        src_solver = _ParticleForceRecordingSolver.instances[-1]
        dst_solver = _ProxyParticleHookSolver.instances[-1]
        self.assertEqual(dst_solver.rewind_calls, 2)
        self.assertEqual(dst_solver.harvest_calls, 2)
        np.testing.assert_allclose(src_solver.input_particle_f[0][0], np.zeros(3), atol=1.0e-6)
        np.testing.assert_allclose(src_solver.input_particle_f[1][0], np.array([0.0, 7.0, 0.0]), atol=1.0e-6)

    def test_custom_particle_harvest_receives_solve_context(self):
        _StepCountingCopySolver.instances.clear()
        _ParticleHarvestStateRecordingSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ParticleHarvestStateRecordingSolver, particles=[]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(source="src", destination="dst", particles=[0]),
                ],
            ),
        )

        coupled.step(model.state(), model.state(), control=None, contacts=None, dt=1.0 / 60.0)

        dst_solver = _ParticleHarvestStateRecordingSolver.instances[-1]
        np.testing.assert_allclose(dst_solver.harvest_particle_qd[0], np.zeros(3), atol=1.0e-6)
        np.testing.assert_allclose(dst_solver.harvest_particle_qd_out[0], np.array([0.0, 2.0, 0.0]), atol=1.0e-6)
        self.assertIsNone(dst_solver.harvest_contacts)

    def test_particle_proxy_maps_proxy_indexed_feedback_to_source(self):
        _ParticleForceRecordingSolver.instances.clear()
        _ProxyParticleHookSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        builder.add_particle(pos=(2.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_ParticleForceRecordingSolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_ProxyParticleHookSolver, particles=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(
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
        np.testing.assert_allclose(src_solver.input_particle_f[1][0], np.array([0.0, 7.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(src_solver.input_particle_f[1][2], np.zeros(3), atol=1.0e-6)

    def test_proxy_destination_view_marks_proxy_particle_flags(self):
        coupled = self._make_coupled()

        src_view = coupled.get_view("src")
        dst_view = coupled.get_view("dst")
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

    def test_mixed_body_particle_proxy_steps_solver_pair_once(self):
        _StepCountingCopySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0], particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver, bodies=[1], particles=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(source="src", destination="dst", bodies=[0], particles=[0]),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 60.0)

        self.assertEqual(_StepCountingCopySolver.instances["src"].step_count, 1)
        self.assertEqual(_StepCountingCopySolver.instances["dst"].step_count, 1)

    def test_proxy_coupling_honors_iteration_count(self):
        _StepCountingCopySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, particles=[0]),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver, particles=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(source="src", destination="dst", particles=[0]),
                ],
                iterations=3,
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 60.0)

        self.assertEqual(_StepCountingCopySolver.instances["src"].step_count, 3)
        self.assertEqual(_StepCountingCopySolver.instances["dst"].step_count, 3)

    def test_entry_substeps_split_dt(self):
        _StepCountingCopySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0], substeps=4),
            ],
        )

        state_0 = model.state()
        state_1 = model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=0.2)

        solver = _StepCountingCopySolver.instances["src"]
        self.assertEqual(solver.step_count, 4)
        np.testing.assert_allclose(solver.dt_values, [0.05, 0.05, 0.05, 0.05])

    def test_entry_substeps_preserve_input_state(self):
        _InputMutatingSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_InputMutatingSolver, bodies=[0], substeps=2),
            ],
        )

        state_0 = model.state()
        state_1 = model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=0.2)

        np.testing.assert_allclose(coupled._entries["src"].state_0.body_qd.numpy()[0], np.zeros(6), atol=1.0e-6)

    def test_proxy_entry_substeps_compose_with_iterations(self):
        _StepCountingCopySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0], substeps=4),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver, bodies=[1], substeps=2),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(source="src", destination="dst", bodies=[0]),
                ],
                iterations=3,
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=0.24)

        src_solver = _StepCountingCopySolver.instances["src"]
        dst_solver = _StepCountingCopySolver.instances["dst"]
        self.assertEqual(src_solver.step_count, 12)
        self.assertEqual(dst_solver.step_count, 6)
        np.testing.assert_allclose(src_solver.dt_values, [0.06] * 12)
        np.testing.assert_allclose(dst_solver.dt_values, [0.12] * 6)

    def test_custom_body_harvest_receives_prepared_input_state(self):
        _StepCountingCopySolver.instances.clear()
        _BodyHarvestStateRecordingSolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_BodyHarvestStateRecordingSolver, bodies=[]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(source="src", destination="dst", bodies=[0]),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 60.0)

        dst_solver = _BodyHarvestStateRecordingSolver.instances[-1]
        np.testing.assert_allclose(dst_solver.harvest_body_qd[0], np.zeros(6), atol=1.0e-6)

    def test_proxy_iterations_restore_subsolver_inputs(self):
        """Each proxy relaxation pass should restart from the top-level input state."""
        _StepCountingCopySolver.instances.clear()
        _BodyInputMutatingCopySolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[0]),
                SolverCoupled.Entry(name="dst", solver=_BodyInputMutatingCopySolver, bodies=[1]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(source="src", destination="dst", bodies=[0]),
                ],
                iterations=2,
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 60.0)

        dst_solver = _BodyInputMutatingCopySolver.instances[-1]
        self.assertEqual(len(dst_solver.input_body_qd), 2)
        np.testing.assert_allclose(dst_solver.input_body_qd[0][1], np.zeros(6), atol=1.0e-6)
        np.testing.assert_allclose(dst_solver.input_body_qd[1][1], np.zeros(6), atol=1.0e-6)

    def test_proxy_contact_filter_restores_shared_contacts(self):
        """Destination proxy filtering should not leak into later source solves."""
        _StepCountingCopySolver.instances.clear()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_ground_plane()
        body = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.05), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(body=body, radius=0.1)
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="src", solver=_StepCountingCopySolver, bodies=[body]),
                SolverCoupled.Entry(name="dst", solver=_StepCountingCopySolver, bodies=[]),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[
                    SolverProxyCoupled.Proxy(source="src", destination="dst", bodies=[body]),
                ],
                iterations=2,
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        contacts = model.collide(state_0)
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0)
        shape0_before = contacts.rigid_contact_shape0.numpy()[:contact_count].copy()
        shape1_before = contacts.rigid_contact_shape1.numpy()[:contact_count].copy()

        coupled.step(state_0, state_1, control=None, contacts=contacts, dt=1.0 / 60.0)

        np.testing.assert_array_equal(contacts.rigid_contact_shape0.numpy()[:contact_count], shape0_before)
        np.testing.assert_array_equal(contacts.rigid_contact_shape1.numpy()[:contact_count], shape1_before)


class TestBodyFlagsProxy(unittest.TestCase):
    """Test that BodyFlags.PROXY exists and is distinct."""

    def test_proxy_flag(self):
        self.assertEqual(newton.BodyFlags.PROXY, 4)
        self.assertNotEqual(newton.BodyFlags.PROXY, newton.BodyFlags.DYNAMIC)
        self.assertNotEqual(newton.BodyFlags.PROXY, newton.BodyFlags.KINEMATIC)
        self.assertEqual(newton.BodyFlags.ALL, 7)


class TestParticleFlagsProxy(unittest.TestCase):
    """Test that ParticleFlags.PROXY exists and is distinct."""

    def test_proxy_flag(self):
        self.assertEqual(newton.ParticleFlags.PROXY, 2)
        self.assertNotEqual(newton.ParticleFlags.PROXY, newton.ParticleFlags.ACTIVE)


class TestProxyVelocityRewind(unittest.TestCase):
    """Default proxy rewinds remove lagged feedback and public force inputs."""

    def test_body_rewind_subtracts_coupling_external_force_and_gravity(self):
        dev = "cpu"
        dt = 0.25
        gravity = wp.array([wp.vec3(0.0, -9.8, 0.0)], dtype=wp.vec3, device=dev)
        body_world = wp.array([0], dtype=wp.int32, device=dev)
        body_q = wp.array([wp.transform([0.0, 0.0, 0.0], wp.quat_identity())], dtype=wp.transform, device=dev)
        body_f = wp.array(
            [wp.spatial_vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)],
            dtype=wp.spatial_vector,
            device=dev,
        )
        coupling_f = wp.array(
            [wp.spatial_vector(2.0, 4.0, 6.0, 1.0, 3.0, 5.0)],
            dtype=wp.spatial_vector,
            device=dev,
        )
        body_local_to_proxy_global = wp.array([0], dtype=int, device=dev)
        body_inv_mass = wp.array([0.5], dtype=float, device=dev)
        body_inv_inertia = wp.array(
            [wp.mat33(0.25, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)],
            dtype=wp.mat33,
            device=dev,
        )
        body_qd = wp.array(
            [wp.spatial_vector(10.0, 11.0, 12.0, 13.0, 14.0, 15.0)],
            dtype=wp.spatial_vector,
            device=dev,
        )

        wp.launch(
            subtract_proxy_forces_kernel,
            dim=1,
            inputs=[
                dt,
                gravity,
                body_world,
                body_q,
                body_f,
                coupling_f,
                body_local_to_proxy_global,
                body_inv_mass,
                body_inv_inertia,
                body_qd,
            ],
            device=dev,
        )

        expected = np.array([9.625, 12.7, 10.875, 12.6875, 13.0, 12.25])
        np.testing.assert_allclose(body_qd.numpy()[0], expected, rtol=1.0e-6, atol=1.0e-6)

    def test_particle_rewind_subtracts_coupling_external_force_and_gravity(self):
        dev = "cpu"
        dt = 0.5
        gravity = wp.array([wp.vec3(0.0, -10.0, 0.0)], dtype=wp.vec3, device=dev)
        particle_world = wp.array([0], dtype=wp.int32, device=dev)
        particle_f = wp.array([wp.vec3(1.0, 0.0, -1.0)], dtype=wp.vec3, device=dev)
        coupling_f = wp.array([wp.vec3(3.0, 4.0, 5.0)], dtype=wp.vec3, device=dev)
        particle_local_to_proxy_global = wp.array([0], dtype=int, device=dev)
        particle_inv_mass = wp.array([0.25], dtype=float, device=dev)
        particle_qd = wp.array([wp.vec3(5.0, 6.0, 7.0)], dtype=wp.vec3, device=dev)

        wp.launch(
            subtract_proxy_particle_forces_kernel,
            dim=1,
            inputs=[
                dt,
                gravity,
                particle_world,
                particle_f,
                coupling_f,
                particle_local_to_proxy_global,
                particle_inv_mass,
                particle_qd,
            ],
            device=dev,
        )

        np.testing.assert_allclose(particle_qd.numpy()[0], np.array([4.5, 10.5, 6.5]), rtol=1.0e-6, atol=1.0e-6)


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

    def test_steady_state_no_jump(self):
        """body_q_prev == p_mjc_begin: no teleport jump, VBD recovers p_mjc_end."""
        dt = 1.0 / 60.0
        p_begin = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, -0.3, 0.1])
        p_prev = p_begin.copy()

        p_expected = p_begin + v * dt
        result, _ = self._run_sync_teleport_forward(p_begin, v, p_prev, dt)
        np.testing.assert_allclose(result[:3], p_expected, atol=1e-6)

    def test_teleport_jump(self):
        """body_q_prev != p_mjc_begin: teleport jump absorbed, VBD recovers p_mjc_end."""
        dt = 1.0 / 60.0
        p_begin = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, -0.3, 0.1])
        p_prev = np.array([0.8, 2.2, 2.7])

        p_expected = p_begin + v * dt
        result, _ = self._run_sync_teleport_forward(p_begin, v, p_prev, dt)
        np.testing.assert_allclose(result[:3], p_expected, atol=1e-6)

    def test_large_jump(self):
        """Large teleportation jump (e.g. after a solver reset) is still absorbed."""
        dt = 1.0 / 60.0
        p_begin = np.array([10.0, 0.0, 0.0])
        v = np.array([1.0, 0.0, 0.0])
        p_prev = np.array([0.0, 0.0, 0.0])

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

    def test_zero_velocity(self):
        """A stationary body with a teleport jump should still land at p_mjc_begin."""
        dt = 1.0 / 60.0
        p_begin = np.array([5.0, 5.0, 5.0])
        v = np.array([0.0, 0.0, 0.0])
        p_prev = np.array([4.9, 5.1, 4.8])

        result, _ = self._run_sync_teleport_forward(p_begin, v, p_prev, dt)
        np.testing.assert_allclose(result[:3], p_begin, atol=1e-6)

    # ------------------------------------------------------------------
    # Angular tests
    # ------------------------------------------------------------------

    def test_angular_steady_state(self):
        """No angular jump: VBD rotation matches the reference exactly."""
        dt = 1.0 / 60.0
        p = np.array([1.0, 0.0, 0.0])
        v = np.zeros(3)

        # Body spinning around Z at 2 rad/s, starting from 30 deg about Z
        r_begin = self._axis_angle_to_quat([0, 0, 1], np.radians(30))
        w = np.array([0.0, 0.0, 2.0])

        ref = self._reference_forward(p, v, r_begin, w, dt)
        result, _ = self._run_sync_teleport_forward(p, v, p, dt, w_mjc_end=w, r_prev=r_begin, r_mjc_begin=r_begin)

        np.testing.assert_allclose(result[:3], ref[:3], atol=1e-6)
        self._assert_quat_close(result[3:], ref[3:], angle_tol_rad=1e-6)

    def test_angular_jump_same_axis(self):
        """Same-axis angular jump: rotation commutes, so VBD matches
        the reference very closely."""
        dt = 1.0 / 60.0
        p = np.zeros(3)
        v = np.zeros(3)
        w = np.array([0.0, 0.0, 2.0])

        r_begin = self._axis_angle_to_quat([0, 0, 1], np.radians(30))
        r_prev = self._axis_angle_to_quat([0, 0, 1], np.radians(25))

        ref = self._reference_forward(p, v, r_begin, w, dt)
        result, _ = self._run_sync_teleport_forward(p, v, p, dt, w_mjc_end=w, r_prev=r_prev, r_mjc_begin=r_begin)

        np.testing.assert_allclose(result[:3], ref[:3], atol=1e-6)
        self._assert_quat_close(result[3:], ref[3:], angle_tol_rad=np.radians(0.01))

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

    def test_angular_multi_step(self):
        """Multi-step angular chain: verify rotation tracks the reference
        trajectory when there is no jump after the first step."""
        dt = 1.0 / 60.0
        p0 = np.zeros(3)
        v = np.zeros(3)
        w = np.array([0.0, 0.0, 1.5])
        n_steps = 20

        r_begin_0 = self._axis_angle_to_quat([0, 0, 1], 0.0)
        r_prev = self._axis_angle_to_quat([0, 0, 1], np.radians(3))

        r_ref = r_begin_0.copy()

        for i in range(n_steps):
            ref = self._reference_forward(p0, v, r_ref, w, dt)
            r_ref_end = ref[3:]

            result, _ = self._run_sync_teleport_forward(p0, v, p0, dt, w_mjc_end=w, r_prev=r_prev, r_mjc_begin=r_ref)

            if i == 0:
                # First step: same-axis jump, very small error
                self._assert_quat_close(
                    result[3:],
                    r_ref_end,
                    angle_tol_rad=np.radians(0.01),
                    msg=f"Step {i}: ",
                )
            else:
                # Subsequent steps: no jump, exact
                self._assert_quat_close(
                    result[3:],
                    r_ref_end,
                    angle_tol_rad=1e-5,
                    msg=f"Step {i}: ",
                )

            r_ref = r_ref_end / np.linalg.norm(r_ref_end)
            r_prev = result[3:] / np.linalg.norm(result[3:])

    def test_pure_angular_jump_no_spin(self):
        """Angular jump with zero angular velocity: body should land at
        r_mjc_begin (the jump is fully absorbed)."""
        dt = 1.0 / 60.0
        p = np.zeros(3)
        v = np.zeros(3)
        w = np.zeros(3)

        r_begin = self._axis_angle_to_quat([0, 1, 0], np.radians(45))
        r_prev = self._axis_angle_to_quat([0, 1, 0], np.radians(30))

        ref = self._reference_forward(p, v, r_begin, w, dt)
        result, _ = self._run_sync_teleport_forward(p, v, p, dt, w_mjc_end=w, r_prev=r_prev, r_mjc_begin=r_begin)

        np.testing.assert_allclose(result[:3], ref[:3], atol=1e-6)
        self._assert_quat_close(result[3:], ref[3:], angle_tol_rad=np.radians(0.1))


if __name__ == "__main__":
    unittest.main()
