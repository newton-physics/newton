# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverMuJoCo


class TestMuJoCoSolverNewtonContacts(unittest.TestCase):
    @staticmethod
    def _build_grounded_spheres(sphere_count):
        """Build a single-world model with separated spheres touching the ground."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for sphere_index in range(sphere_count):
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(float(sphere_index % 10) * 2.0, float(sphere_index // 10) * 2.0, 0.45),
                    wp.quat_identity(),
                )
            )
            builder.add_shape_sphere(body=body, radius=0.5)
        return builder.finalize()

    def _setup_sphere_on_plane(self):
        """Set up a simple model with a sphere and a plane."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()

        self.sphere_radius = 0.5
        sphere_body_idx = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(
            body=sphere_body_idx,
            radius=self.sphere_radius,
        )

        self.model = builder.finalize()
        self.sphere_body_idx = sphere_body_idx

    def test_sphere_on_plane_with_newton_contacts(self):
        """Test that a sphere correctly collides with a plane using Newton contacts."""
        self._setup_sphere_on_plane()
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        try:
            solver = SolverMuJoCo(self.model, use_mujoco_contacts=False)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        sim_dt = 1.0 / 240.0
        num_steps = 120  # Simulate for 0.5 seconds to ensure it settles

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        for _ in range(num_steps):
            self.collision_pipeline.collide(self.state_in, self.contacts)
            solver.step(self.state_in, self.state_out, self.control, self.contacts, sim_dt)
            self.state_in, self.state_out = self.state_out, self.state_in

        final_pos = self.state_in.body_q.numpy()[self.sphere_body_idx, :3]
        final_height = final_pos[2]  # Z-up in MuJoCo

        # The sphere should settle on the plane, with its center at its radius's height
        self.assertGreater(
            final_height,
            self.sphere_radius * 0.9,
            f"Sphere fell through the plane. Final height: {final_height}",
        )
        self.assertLess(
            final_height,
            self.sphere_radius * 1.2,
            f"Sphere is floating above the plane. Final height: {final_height}",
        )

    def test_initial_forward_skips_mujoco_contacts(self):
        """Verify Newton-contact initialization skips transient MuJoCo collision detection."""
        self._setup_sphere_on_plane()
        try:
            mujoco, _ = SolverMuJoCo.import_mujoco()
            original_forward = mujoco.mj_forward
            contact_disabled_during_forward: list[bool] = []

            def recording_forward(model, data):
                contact_disabled_during_forward.append(
                    bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
                )
                return original_forward(model, data)

            with patch.object(mujoco, "mj_forward", side_effect=recording_forward):
                solver = SolverMuJoCo(self.model, use_mujoco_contacts=False)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        self.assertEqual(contact_disabled_during_forward, [True])
        self.assertFalse(solver.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)

    def test_newton_contact_defaults_preserve_generated_contacts(self):
        """Preserve generated Newton contacts when sizing default MuJoCo buffers."""
        model = self._build_grounded_spheres(60)

        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state, contacts)
        generated_contact_count = int(contacts.rigid_contact_count.numpy()[0])

        self.assertGreater(generated_contact_count, 48)
        self.assertGreaterEqual(solver.mjw_data.naconmax, generated_contact_count)
        self.assertGreaterEqual(solver.mjw_data.njmax, generated_contact_count * 4)

        solver._convert_contacts_to_mjwarp(model, state, contacts)
        injected_contact_count = int(solver.mjw_data.nacon.numpy()[0])
        self.assertEqual(injected_contact_count, generated_contact_count)

    def test_newton_contact_explicit_capacities_are_preserved(self):
        """Preserve explicit MuJoCo capacities for Newton-generated contacts."""
        model = self._build_grounded_spheres(60)
        nconmax = 200
        njmax = 300

        try:
            solver = SolverMuJoCo(
                model,
                use_mujoco_contacts=False,
                nconmax=nconmax,
                njmax=njmax,
            )
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        self.assertEqual(solver.mjw_data.naconmax, nconmax)
        self.assertEqual(solver.mjw_data.njmax, njmax)

    def test_explicit_nconmax_below_initial_contacts_is_increased(self):
        """Increase an explicit contact capacity below initial MuJoCo contacts."""
        model = self._build_grounded_spheres(60)

        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        with self.assertWarnsRegex(UserWarning, r"Value for nconmax is changed from 16 to 60"):
            solver = SolverMuJoCo(model, use_mujoco_contacts=True, nconmax=16)

        self.assertEqual(solver.mjw_data.naconmax, 60)

    def test_explicit_njmax_below_initial_constraints_is_increased(self):
        """Increase an explicit constraint capacity below initial MuJoCo constraints."""
        builder = newton.ModelBuilder()
        inertia = wp.mat33(np.eye(3) * 0.1)
        for _ in range(5):
            body = builder.add_link(mass=1.0, com=wp.vec3(), inertia=inertia)
            joint = builder.add_joint_revolute(
                parent=-1,
                child=body,
                limit_lower=1.0,
                limit_upper=2.0,
            )
            builder.add_articulation([joint])
        model = builder.finalize()

        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        with self.assertWarnsRegex(UserWarning, r"Value for njmax is changed from 1 to 5"):
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, nconmax=1, njmax=1)

        self.assertEqual(solver.mjw_data.njmax, 5)

    def test_newton_contact_capacity_preserves_delayed_joint_limits(self):
        """Preserve default capacity for non-contact constraints activated after initialization."""
        joint_count = 10
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        inertia = wp.mat33(np.eye(3) * 0.1)
        for joint_index in range(joint_count):
            body = builder.add_link(
                mass=1.0,
                com=wp.vec3(0.0, 0.0, 0.0),
                inertia=inertia,
                label=f"limited_body_{joint_index}",
            )
            joint = builder.add_joint_revolute(
                parent=-1,
                child=body,
                limit_lower=-1.0,
                limit_upper=1.0,
                label=f"limited_joint_{joint_index}",
            )
            builder.add_articulation([joint])
        model = builder.finalize()

        collision_pipeline = newton.CollisionPipeline(model, rigid_contact_max=1)
        contacts = collision_pipeline.contacts()
        try:
            baseline_solver = SolverMuJoCo(model, use_mujoco_contacts=True)
            solver = SolverMuJoCo(model, use_mujoco_contacts=False)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        self.assertGreaterEqual(solver.mjw_data.njmax, baseline_solver.mjw_data.njmax)

        state_0 = model.state()
        state_1 = model.state()
        state_0.joint_q.assign(np.full(model.joint_coord_count, 2.0, dtype=np.float32))
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)

        constraints_per_world = solver.mjw_data.nefc.numpy()
        self.assertEqual(int(constraints_per_world[0]), joint_count)
        self.assertLessEqual(int(np.max(constraints_per_world)), solver.mjw_data.njmax)

    def test_newton_contact_constraints_cover_uneven_worlds(self):
        """Size per-world constraints for contacts concentrated in one world."""
        sphere_count = 30
        world = newton.ModelBuilder()
        for sphere_index in range(sphere_count):
            body = world.add_body(
                xform=wp.transform(
                    wp.vec3(float(sphere_index % 10) * 2.0, float(sphere_index // 10) * 2.0, 0.45),
                    wp.quat_identity(),
                )
            )
            world.add_shape_sphere(body=body, radius=0.5)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_world(world, xform=wp.transform_identity())
        builder.add_world(world, xform=wp.transform(wp.vec3(0.0, 0.0, 10.0), wp.quat_identity()))
        model = builder.finalize()

        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        collision_pipeline = newton.CollisionPipeline(model, rigid_contact_max=sphere_count)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_0, contacts)
        generated_contact_count = int(contacts.rigid_contact_count.numpy()[0])

        try:
            solver = SolverMuJoCo(model, separate_worlds=True, use_mujoco_contacts=False)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        self.assertEqual(generated_contact_count, sphere_count)
        solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)

        injected_contact_count = int(solver.mjw_data.nacon.numpy()[0])
        contact_worlds = solver.mjw_data.contact.worldid.numpy()[:injected_contact_count]
        contacts_per_world = np.bincount(contact_worlds, minlength=model.world_count)
        constraints_per_world = solver.mjw_data.nefc.numpy()

        self.assertEqual(injected_contact_count, generated_contact_count)
        np.testing.assert_array_equal(contacts_per_world, [sphere_count, 0])
        self.assertEqual(int(constraints_per_world[0]), sphere_count * 4)
        self.assertLessEqual(int(np.max(constraints_per_world)), solver.mjw_data.njmax)

    def test_newton_contact_constraints_scale_with_busiest_world(self):
        """Bound per-world constraint capacity by the busiest cloned world."""
        world_count = 64
        world = newton.ModelBuilder()
        body = world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.45), wp.quat_identity()))
        world.add_shape_sphere(body=body, radius=0.5)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.replicate(world, world_count)
        model = builder.finalize()
        newton.CollisionPipeline(model)

        try:
            baseline_solver = SolverMuJoCo(model, separate_worlds=True, use_mujoco_contacts=True)
            solver = SolverMuJoCo(model, separate_worlds=True, use_mujoco_contacts=False)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        self.assertGreaterEqual(solver.mjw_data.naconmax, model.rigid_contact_max)
        self.assertEqual(solver.mjw_data.njmax, baseline_solver.mjw_data.njmax)

    def test_sphere_rolls_without_slip_with_newton_contacts(self):
        radius = 0.1
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 2.0e3
        builder.default_shape_cfg.mu = 1.0
        builder.add_ground_plane()

        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e5, kd=2.0e3, mu=1.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity()))
        builder.add_shape_sphere(body=body, radius=radius, cfg=shape_cfg)
        model = builder.finalize()

        try:
            solver = SolverMuJoCo(
                model,
                use_mujoco_contacts=False,
                use_mujoco_cpu=False,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                iterations=50,
                ls_iterations=20,
                nconmax=64,
                njmax=256,
            )
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        joint_qd = state_0.joint_qd.numpy()
        joint_qd[:] = 0.0
        joint_qd[0] = 0.1
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        for _ in range(240):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        body_qd = state_0.body_qd.numpy()[body]
        self.assertAlmostEqual(float(body_qd[0]), float(body_qd[4] * radius), delta=5.0e-3)

    def test_efc_address_init(self):
        """efc_address is -1 for inactive contacts after Newton-to-mujoco_warp conversion.

        Regression: without initializing efc_address to -1 in write_contact, inactive
        contacts (dist >= includemargin) retain stale efc_address values from prior
        steps, corrupting contact force readback.  A tilted box on a ground plane with
        margin > 0 produces a mix of active and inactive contacts.
        """
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.default_shape_cfg.margin = 0.05
        builder.add_ground_plane()
        tilt = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.3)
        b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.18), tilt))
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize()

        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        for _ in range(5):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 0.002)
            state_in, state_out = state_out, state_in

        nacon = solver.mjw_data.nacon.numpy()[0]
        self.assertGreater(nacon, 0)
        dist = solver.mjw_data.contact.dist.numpy()[:nacon]
        includemargin = solver.mjw_data.contact.includemargin.numpy()[:nacon]
        efc_address = solver.mjw_data.contact.efc_address.numpy()[:nacon, 0]

        inactive = (dist - includemargin) >= 0
        self.assertGreater(inactive.sum(), 0, "No inactive contacts generated")
        n_stale = int((efc_address[inactive] >= 0).sum())
        self.assertEqual(n_stale, 0, f"{n_stale}/{inactive.sum()} inactive contacts have stale efc_address")

    def test_notify_body_inertial_invalidates_contact_fast_path(self):
        """notify_model_changed(BODY_INERTIAL_PROPERTIES) must invalidate the
        cached contact conversion fast path.

        Regression: ``set_const_fixed`` recomputes MuJoCo constants that feed
        into the cached MJWarp contact fields written by the fast path.  If the
        fast path is not invalidated, subsequent substeps reuse stale ``cid``
        and field values, which previously produced an illegal-memory-access
        when the underlying ``mjw_data`` contact arrays had been reallocated
        (observed in IsaacLab dexsuite training).
        """
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.18), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize()

        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        # Run one full substep so the fast path becomes "armed" (i.e. the next
        # substep would normally take the fast path).
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 0.002)
        state_in, state_out = state_out, state_in

        last_gen_before = int(solver._last_contact_generation.numpy()[0])
        self.assertNotEqual(
            last_gen_before,
            -1,
            "Fast path should be armed after one full substep (last_contact_generation != sentinel).",
        )

        # Notify body inertial properties — this must invalidate the fast path.
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        last_gen_after = int(solver._last_contact_generation.numpy()[0])
        self.assertEqual(
            last_gen_after,
            -1,
            "BODY_INERTIAL_PROPERTIES notify must invalidate the contact fast path.",
        )

        # Verify the next step still runs cleanly through the full path.
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 0.002)
        wp.synchronize()  # force surfacing any async device errors

    def test_contacts_instance_swap_invalidates_fast_path(self):
        """Swapping the bound :class:`Contacts` instance must invalidate the
        cached fast path so cached ``cid`` values aren't reused against a
        different output buffer.

        The cache is keyed on ``id(contacts.contact_generation)`` (the inner
        per-Contacts device array) rather than ``id(contacts)`` itself, so
        this test asserts the inner-array id is what gets tracked.
        """
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.18), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize()

        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        collision_pipeline = newton.CollisionPipeline(model)
        contacts_a = collision_pipeline.contacts()
        contacts_b = collision_pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts_a)
        solver.step(state_in, state_out, control, contacts_a, 0.002)
        state_in, state_out = state_out, state_in

        cached_id = solver._last_contacts_id
        self.assertEqual(cached_id, id(contacts_a.contact_generation))

        # Swap to a different Contacts instance — solver must notice and rebuild
        # the fast-path cache rather than reusing stale tid_to_cid mappings.
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts_b)
        solver.step(state_in, state_out, control, contacts_b, 0.002)
        wp.synchronize()

        self.assertEqual(solver._last_contacts_id, id(contacts_b.contact_generation))
        self.assertNotEqual(
            cached_id,
            id(contacts_b.contact_generation),
            "_last_contacts_id should track the bound Contacts object's contact_generation array",
        )

    def test_fast_path_buffers_eagerly_allocated(self):
        """Verify persistent contact buffers and mappings are initialized in ``__init__``.

        Regression (PR #2678 bisect, "Fix 2"): lazy ``wp.full(...)`` allocation
        on the first step often runs while a CUDA graph is being captured.  The
        resulting buffers can have a tangled lifetime, and
        :meth:`_invalidate_contact_fast_path` — which is called from outside
        the captured graph (e.g. from :meth:`notify_model_changed`) — would
        then touch stale captured memory and raise CUDA error 700 (illegal
        memory access).  Eager pre-allocation in ``__init__`` is the fix.
        """
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.18), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize()

        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        # Buffers must exist as Warp arrays on the model's device immediately
        # after construction — before any solver.step() call has run.
        self.assertIsNotNone(
            solver._last_contact_generation,
            "_last_contact_generation must be eagerly pre-allocated in __init__",
        )
        self.assertIsNotNone(
            solver._last_nacon_count,
            "_last_nacon_count must be eagerly pre-allocated in __init__",
        )
        self.assertIsNotNone(
            solver._contact_tid_to_cid,
            "_contact_tid_to_cid must be eagerly pre-allocated in __init__",
        )
        self.assertIsInstance(solver._last_contact_generation, wp.array)
        self.assertIsInstance(solver._last_nacon_count, wp.array)
        self.assertIsInstance(solver._contact_tid_to_cid, wp.array)
        self.assertEqual(solver._last_contact_generation.dtype, wp.int32)
        self.assertEqual(solver._last_nacon_count.dtype, wp.int32)
        self.assertEqual(solver._contact_tid_to_cid.dtype, wp.int32)
        self.assertEqual(solver._last_contact_generation.shape, (1,))
        self.assertEqual(solver._last_nacon_count.shape, (1,))
        self.assertEqual(solver._contact_tid_to_cid.shape, (solver.mjw_data.naconmax,))
        self.assertEqual(solver._last_contact_generation.device, model.device)
        self.assertEqual(solver._last_nacon_count.device, model.device)
        self.assertEqual(solver._contact_tid_to_cid.device, model.device)
        self.assertTrue(np.all(solver._contact_tid_to_cid.numpy() == -1))

        self.assertIsNotNone(solver.newton_shape_to_mjc_geom)
        fwd = solver.mjc_geom_to_newton_shape.numpy()
        inv = solver.newton_shape_to_mjc_geom.numpy()
        for geom, shape in enumerate(fwd[0]):
            if shape >= 0:
                self.assertEqual(inv[shape], geom)

        # Calling _invalidate_contact_fast_path() before any step must succeed
        # cleanly — this is the exact path that previously hit stale captured
        # memory when the buffers were still None / lazily allocated.
        solver._invalidate_contact_fast_path()
        self.assertEqual(int(solver._last_contact_generation.numpy()[0]), -1)
        self.assertEqual(int(solver._last_nacon_count.numpy()[0]), 0)

        # And again immediately after a notify before any step — the CUDA 700
        # repro path from the bug report.
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
        wp.synchronize()  # surface any async device errors

    def test_recapture_after_discarded_capture(self):
        """Replay a replacement graph after discarding the first capture without replay."""
        device = wp.get_device()
        if not device.is_cuda or not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires a CUDA device with the memory pool enabled.")

        model = self._build_grounded_spheres(1)
        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        state_in, state_out = model.state(), model.state()
        control = model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        def simulate():
            nonlocal state_in, state_out
            # Two steps keep input/output buffers identical between captures and replays.
            for _ in range(2):
                state_in.clear_forces()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
                state_in, state_out = state_out, state_in

        with wp.ScopedCapture(device=device) as capture:
            simulate()
        del capture

        with wp.ScopedCapture(device=device) as capture:
            simulate()
        for _ in range(3):
            wp.capture_launch(capture.graph)

        self.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_in.body_qd.numpy()).all())

    def test_ephemeral_contacts_wrapper_keeps_fast_path_armed(self):
        """A new ``Contacts`` wrapper that shares the same underlying
        ``contact_generation`` array must NOT invalidate the fast path.

        Regression (PR #2678 bisect, "Fix 1"): keying the cache on
        ``id(contacts)`` (the outer wrapper) over-invalidates in dexsuite /
        IsaacLab batched workflows where a fresh ``Contacts`` wrapper is
        constructed per substep around persistent device buffers.  Every step
        was forced to take the full path, and combined with the contact-count
        accumulation across forced full passes this drove the simulation to
        NaN within ~218 iterations.  The cache must instead be keyed on
        ``id(contacts.contact_generation)`` — the inner Warp array's identity,
        which survives wrapper recreation.
        """
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.18), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize()

        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        collision_pipeline = newton.CollisionPipeline(model)
        contacts_a = collision_pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        # Arm the fast path.
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts_a)
        solver.step(state_in, state_out, control, contacts_a, 0.002)
        state_in, state_out = state_out, state_in

        cached_gen_id = solver._last_contacts_id
        self.assertEqual(cached_gen_id, id(contacts_a.contact_generation))

        # Build an ephemeral wrapper that aliases the same device arrays as
        # ``contacts_a`` (in particular the same ``contact_generation``
        # Warp array).  This mimics the dexsuite/IsaacLab pattern where a
        # fresh Contacts wrapper is materialised each substep around a pool
        # of persistent device buffers.
        class _AliasedContacts:
            pass

        contacts_alias = _AliasedContacts()
        for attr in (
            "contact_counters",
            "rigid_contact_count",
            "contact_generation",
            "rigid_contact_max",
            "soft_contact_max",
            "rigid_contact_shape0",
            "rigid_contact_shape1",
            "rigid_contact_point0",
            "rigid_contact_point1",
            "rigid_contact_normal",
            "rigid_contact_offset0",
            "rigid_contact_offset1",
            "rigid_contact_margin0",
            "rigid_contact_margin1",
            "rigid_contact_stiffness",
            "rigid_contact_damping",
            "rigid_contact_friction",
        ):
            setattr(contacts_alias, attr, getattr(contacts_a, attr))

        # Sanity: the two wrappers are distinct objects but share the same
        # contact_generation array — the precise dexsuite scenario.
        self.assertNotEqual(id(contacts_a), id(contacts_alias))
        self.assertEqual(id(contacts_a.contact_generation), id(contacts_alias.contact_generation))

        # Step with the new wrapper.  The cache must NOT be invalidated, since
        # the underlying contact data is identical.
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts_a)  # populate via the original handle
        solver.step(state_in, state_out, control, contacts_alias, 0.002)
        wp.synchronize()

        self.assertEqual(
            solver._last_contacts_id,
            cached_gen_id,
            "Cache key must be id(contacts.contact_generation), not id(contacts) — "
            "an ephemeral wrapper around the same underlying buffers should not "
            "invalidate the fast path (dexsuite over-invalidation regression).",
        )

    def test_isaaclab_mass_randomization_loop(self):
        """End-to-end reproduction of the IsaacLab mass-randomization workflow.

        Mirrors the minimal repro from PR #2678: arm the fast path, then
        repeatedly interleave ``notify_model_changed(BODY_INERTIAL_PROPERTIES)``
        (which IsaacLab's ``randomize_rigid_body_mass`` event hook triggers)
        with substeps and force a D2H sync each time.  Without the fixes this
        loop crashes with ``RuntimeError: Warp copy error: ... an illegal
        memory access was encountered`` inside ``set_const_fixed`` ->
        ``body_gravcomp.numpy()``.
        """
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()
        body_idx = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.18), wp.quat_identity()))
        builder.add_shape_box(body_idx, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize()

        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        # Pull the body mass into NumPy so we can perturb and write it back.
        body_mass_np = model.body_mass.numpy().copy()
        body_inv_mass_np = model.body_inv_mass.numpy().copy()
        rng = np.random.default_rng(seed=2026)

        num_outer = 8  # randomization events
        substeps_per_outer = 5  # substeps between each randomization
        for outer in range(num_outer):
            for _ in range(substeps_per_outer):
                state_in.clear_forces()
                collision_pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, 0.002)
                state_in, state_out = state_out, state_in

            # IsaacLab-style mass randomization: jitter the body's mass and
            # notify the solver, which triggers set_const_fixed and reallocates
            # MJWarp contact buffers.  This is the exact crash trigger from the
            # bug report.
            scale = float(rng.uniform(0.7, 1.3))
            new_mass = body_mass_np[body_idx] * scale
            model.body_mass.assign(np.array([new_mass if i == body_idx else m for i, m in enumerate(body_mass_np)]))
            model.body_inv_mass.assign(
                np.array([1.0 / new_mass if i == body_idx else inv for i, inv in enumerate(body_inv_mass_np)])
            )
            solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

            # Force a D2H sync — this is where the CUDA 700 surfaced in the
            # bug report (set_const_fixed -> body_gravcomp.numpy()).  Reading
            # any field that requires a copy out is sufficient to trigger it.
            _ = solver.mjw_data.nacon.numpy()
            wp.synchronize()

            # The simulation must remain stable: no NaNs in body pose.
            body_q = state_in.body_q.numpy()
            self.assertTrue(
                np.all(np.isfinite(body_q)),
                f"NaN/Inf in body_q after randomization round {outer}: {body_q}",
            )


class TestMuJoCoSolverContactKf(unittest.TestCase):
    """Verify shape_material_kf maps to elliptic-contact solreffriction."""

    def _make_sphere_scene(self, kf_sphere, kf_plane, impratio=None, cone="elliptic"):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 100.0
        builder.default_shape_cfg.kf = kf_plane
        builder.add_ground_plane()
        # start slightly penetrating so the first collide() yields an active contact
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.45), wp.quat_identity()))
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e4, kd=100.0, kf=kf_sphere)
        builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg)
        model = builder.finalize()
        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, cone=cone, nconmax=32, njmax=128, impratio=impratio)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")
        return model, solver

    def _step_and_read_solreffriction(self, model, solver):
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        nacon = int(solver.mjw_data.nacon.numpy()[0])
        self.assertGreater(nacon, 0)
        return nacon, (collision_pipeline, contacts), (state_0, state_1, control)

    def _expected_solreffriction(self, solver, nacon, kf_pair, impratio=1.0):
        solimp = solver.mjw_data.contact.solimp.numpy()[:nacon]
        geom = solver.mjw_data.contact.geom.numpy()[:nacon]
        geom_bodyid = solver.mjw_model.geom_bodyid.numpy()
        body_invweight0 = solver.mjw_model.body_invweight0.numpy()[0]
        ir = 1.0 / math.sqrt(impratio)
        expected = []
        for i in range(nacon):
            invw = body_invweight0[geom_bodyid[geom[i][0]]][0] + body_invweight0[geom_bodyid[geom[i][1]]][0]
            dmax = solimp[i][1]
            # beta = kf*(1/D + A) with A ~= invw; timeconst = 2/(dmax*beta)
            expected.append(2.0 / (kf_pair * invw * ((1.0 - dmax) * ir * ir + dmax)))
        return expected

    def test_kf_sets_contact_solreffriction(self):
        """Verify positive kf sets the mixed contact solreffriction."""
        model, solver = self._make_sphere_scene(kf_sphere=800.0, kf_plane=200.0)
        nacon, _, _ = self._step_and_read_solreffriction(model, solver)
        solreffriction = solver.mjw_data.contact.solreffriction.numpy()[:nacon]
        # equal solmix/priority -> mix = 0.5
        expected = self._expected_solreffriction(solver, nacon, kf_pair=500.0)
        for i in range(nacon):
            self.assertAlmostEqual(float(solreffriction[i][0]) / expected[i], 1.0, places=5)
            self.assertEqual(float(solreffriction[i][1]), 1.0)

    def test_kf_zero_mixes_with_positive_value(self):
        """Verify zero kf participates in the usual contact-material mixing."""
        model, solver = self._make_sphere_scene(kf_sphere=800.0, kf_plane=0.0)
        nacon, _, _ = self._step_and_read_solreffriction(model, solver)
        solreffriction = solver.mjw_data.contact.solreffriction.numpy()[:nacon]
        expected = self._expected_solreffriction(solver, nacon, kf_pair=400.0)
        for i in range(nacon):
            self.assertAlmostEqual(float(solreffriction[i][0]) / expected[i], 1.0, places=5)
            self.assertEqual(float(solreffriction[i][1]), 1.0)

    def test_kf_zero_disables_friction(self):
        """Verify a resolved zero kf makes the contact frictionless."""
        model, solver = self._make_sphere_scene(kf_sphere=0.0, kf_plane=0.0)
        nacon, _, _ = self._step_and_read_solreffriction(model, solver)
        solreffriction = solver.mjw_data.contact.solreffriction.numpy()[:nacon]
        condim = solver.mjw_data.contact.dim.numpy()[:nacon]
        for i in range(nacon):
            self.assertEqual(int(condim[i]), 1)
            self.assertEqual(float(solreffriction[i][0]), 0.0)
            self.assertEqual(float(solreffriction[i][1]), 0.0)

    def test_kf_zero_does_not_change_pyramidal_contacts(self):
        """Verify zero kf leaves pyramidal friction contacts unchanged."""
        model, solver = self._make_sphere_scene(kf_sphere=0.0, kf_plane=0.0, cone="pyramidal")
        nacon, _, _ = self._step_and_read_solreffriction(model, solver)
        condim = solver.mjw_data.contact.dim.numpy()[:nacon]
        for i in range(nacon):
            self.assertEqual(int(condim[i]), 3)

    def test_kf_zero_inverse_weight_leaves_solreffriction_unset(self):
        """Verify a zero inverse weight cannot produce a non-finite reference."""
        model, solver = self._make_sphere_scene(kf_sphere=800.0, kf_plane=200.0)
        solver.mjw_model.body_invweight0.zero_()
        nacon, _, _ = self._step_and_read_solreffriction(model, solver)
        solreffriction = solver.mjw_data.contact.solreffriction.numpy()[:nacon]
        for i in range(nacon):
            self.assertEqual(float(solreffriction[i][0]), 0.0)
            self.assertEqual(float(solreffriction[i][1]), 0.0)

    def test_kf_runtime_update(self):
        """Verify runtime kf updates refresh the contact reference."""
        model, solver = self._make_sphere_scene(kf_sphere=800.0, kf_plane=200.0)
        nacon, (collision_pipeline, contacts), (state_0, state_1, control) = self._step_and_read_solreffriction(
            model, solver
        )
        model.shape_material_kf.fill_(400.0)
        solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        nacon = int(solver.mjw_data.nacon.numpy()[0])
        self.assertGreater(nacon, 0)
        solreffriction = solver.mjw_data.contact.solreffriction.numpy()[:nacon]
        expected = self._expected_solreffriction(solver, nacon, kf_pair=400.0)
        for i in range(nacon):
            self.assertAlmostEqual(float(solreffriction[i][0]) / expected[i], 1.0, places=5)

    def test_kf_impratio_scaling(self):
        """Verify impratio scaling preserves the requested force-space slope."""
        model, solver = self._make_sphere_scene(kf_sphere=800.0, kf_plane=200.0, impratio=4.0)
        nacon, _, _ = self._step_and_read_solreffriction(model, solver)
        solreffriction = solver.mjw_data.contact.solreffriction.numpy()[:nacon]
        expected = self._expected_solreffriction(solver, nacon, kf_pair=500.0, impratio=4.0)
        for i in range(nacon):
            self.assertAlmostEqual(float(solreffriction[i][0]) / expected[i], 1.0, places=5)

    def _slide_sphere_prismatic(self, kf, density, num_steps=60):
        """Single contact, rotation locked by a prismatic joint: pure force-space viscous friction."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 1.0e3
        builder.default_shape_cfg.kf = kf
        builder.default_shape_cfg.mu = 2.0
        builder.add_ground_plane()
        radius = 0.1
        # add_body() would auto-create a free joint; add_link()+add_articulation()
        # keeps the prismatic joint as the body's sole (tree) connection, and the
        # joint's parent_xform (not the link xform) is what places the body.
        body = builder.add_link()
        cfg = newton.ModelBuilder.ShapeConfig(density=density, ke=1.0e5, kd=1.0e3, kf=kf, mu=2.0)
        builder.add_shape_sphere(body=body, radius=radius, cfg=cfg)
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, radius - 0.02), wp.quat_identity()),
            axis=(1.0, 0.0, 0.0),
        )
        builder.add_articulation([joint])
        model = builder.finalize()
        try:
            solver = SolverMuJoCo(
                model,
                use_mujoco_contacts=False,
                cone="elliptic",
                solver="newton",
                integrator="implicitfast",
                iterations=50,
                ls_iterations=20,
                nconmax=32,
                njmax=256,
            )
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[0] = 0.05
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        for _ in range(num_steps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        # step() only auto-syncs body_qd for free-joint bodies; read the DOF velocity
        return float(state_0.joint_qd.numpy()[0])

    def test_kf_force_space_mass_dependence(self):
        """Verify the same kf produces mass-dependent velocity decay."""
        v_light = self._slide_sphere_prismatic(kf=120.0, density=1000.0)  # ~4.2 kg,  rate ~9.9/s, analytic ~0.0042
        v_heavy = self._slide_sphere_prismatic(kf=120.0, density=8000.0)  # ~33.5 kg, rate ~1.2/s, analytic ~0.037
        self.assertLess(v_light, 0.01)
        self.assertGreater(v_heavy, 0.02)

    def test_kf_scales_viscous_friction(self):
        """Verify larger kf values produce faster sliding decay."""
        v_soft = self._slide_sphere_prismatic(kf=30.0, density=1000.0)  # rate ~2.5/s, analytic ~0.027
        v_hard = self._slide_sphere_prismatic(kf=3000.0, density=1000.0)  # rate ~247/s, analytic ~0
        self.assertGreater(v_soft, 0.015)
        self.assertLess(v_soft, 0.04)
        self.assertGreater(v_soft, 3.0 * max(v_hard, 1.0e-6))

    def test_kf_zero_preserves_sliding_velocity(self):
        """Verify zero kf applies no sliding-friction force."""
        velocity = self._slide_sphere_prismatic(kf=0.0, density=1000.0)
        self.assertAlmostEqual(velocity, 0.05, places=5)


class TestFrictionPriority(unittest.TestCase):
    """Verify that contact friction respects geom priority.

    When two geoms have different priorities, the higher-priority geom's
    friction should be used directly (not the element-wise max).
    """

    def _build_model(self, priority_a, friction_a, priority_b, friction_b):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        cfg_a = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=1000.0, mu=friction_a)
        cfg_b = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=1000.0, mu=friction_b)

        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()))
        builder.add_shape_box(
            body=body_a,
            hx=0.2,
            hy=0.2,
            hz=0.1,
            cfg=cfg_a,
            custom_attributes={"mujoco:geom_priority": priority_a},
        )

        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(
            body=body_b,
            hx=0.5,
            hy=0.5,
            hz=0.1,
            cfg=cfg_b,
            custom_attributes={"mujoco:geom_priority": priority_b},
        )

        model = builder.finalize()
        return model

    def _get_contact_friction(self, model):
        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed: {e}")

        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 0.002)

        nacon = solver.mjw_data.nacon.numpy()[0]
        self.assertGreater(nacon, 0, "No contacts generated")
        return solver.mjw_data.contact.friction.numpy()[:nacon]

    def _assert_slide_friction(self, friction, expected, label):
        for i in range(len(friction)):
            self.assertAlmostEqual(
                float(friction[i, 0]),
                expected,
                places=5,
                msg=f"{label}: contact {i} slide friction should be {expected}, got {friction[i, 0]}",
            )

    def test_higher_priority_friction_used(self):
        """Contact friction should come from the higher-priority geom, regardless of operand order."""
        hi_mu = 0.3
        lo_mu = 0.9

        model_a = self._build_model(priority_a=5, friction_a=hi_mu, priority_b=0, friction_b=lo_mu)
        self._assert_slide_friction(self._get_contact_friction(model_a), hi_mu, "high-priority on body A")

        model_b = self._build_model(priority_a=0, friction_a=lo_mu, priority_b=5, friction_b=hi_mu)
        self._assert_slide_friction(self._get_contact_friction(model_b), hi_mu, "high-priority on body B")

    def test_equal_priority_uses_max(self):
        """When priorities are equal, contact friction should be the element-wise max, regardless of operand order."""
        mu_lo = 0.2
        mu_hi = 0.8
        expected = mu_hi

        model_a = self._build_model(priority_a=0, friction_a=mu_hi, priority_b=0, friction_b=mu_lo)
        self._assert_slide_friction(self._get_contact_friction(model_a), expected, "larger mu on body A")

        model_b = self._build_model(priority_a=0, friction_a=mu_lo, priority_b=0, friction_b=mu_hi)
        self._assert_slide_friction(self._get_contact_friction(model_b), expected, "larger mu on body B")


class TestImmovableContactFiltering(unittest.TestCase):
    """Verify that contacts between two immovable bodies are filtered out.

    The MuJoCo solver produces degenerate efc_D values when both sides of a
    contact have zero/near-zero invweight (both bodies are immovable).  The
    contact conversion kernel must skip such pairs.  Immovable bodies include
    static shapes (body < 0), kinematic bodies (BodyFlags.KINEMATIC), and
    bodies whose weld group has no degrees of freedom.
    """

    @staticmethod
    def _build_kinematic_on_ground():
        """Build a model with a kinematic free-joint body resting on a ground plane."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()

        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
            is_kinematic=True,
        )
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.05)
        return builder.finalize(), body

    @staticmethod
    def _build_two_kinematic_bodies():
        """Build a model with two kinematic free-joint bodies overlapping."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0

        b1 = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
            is_kinematic=True,
        )
        builder.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

        b2 = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
            is_kinematic=True,
        )
        builder.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        return builder.finalize(), b1, b2

    @staticmethod
    def _build_dynamic_on_ground():
        """Build a model with a dynamic body on a ground plane (should keep contacts)."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()

        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.05)
        return builder.finalize(), body

    def _get_nacon(self, model):
        """Run collision + one solver step and return the MuJoCo contact count."""
        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed: {e}")

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        return int(solver.mjw_data.nacon.numpy()[0])

    def test_kinematic_on_ground_contacts_filtered(self):
        """Contacts between a kinematic body and the static ground plane must be filtered."""
        model, _ = self._build_kinematic_on_ground()
        nacon = self._get_nacon(model)
        self.assertEqual(nacon, 0, f"Expected 0 MuJoCo contacts for kinematic-on-ground, got {nacon}")

    def test_two_kinematic_bodies_contacts_filtered(self):
        """Contacts between two kinematic bodies must be filtered."""
        model, _, _ = self._build_two_kinematic_bodies()
        nacon = self._get_nacon(model)
        self.assertEqual(nacon, 0, f"Expected 0 MuJoCo contacts for kinematic-kinematic, got {nacon}")

    def test_dynamic_on_ground_contacts_preserved(self):
        """Contacts between a dynamic body and the ground plane must be preserved."""
        model, _ = self._build_dynamic_on_ground()
        nacon = self._get_nacon(model)
        self.assertGreater(nacon, 0, "Expected contacts for dynamic-on-ground, got 0")

    def test_kinematic_vs_dynamic_contacts_preserved(self):
        """Contacts between a kinematic body and a dynamic body must be preserved."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0

        kinematic_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
            is_kinematic=True,
        )
        builder.add_shape_box(kinematic_body, hx=0.5, hy=0.5, hz=0.05)

        dynamic_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_shape_box(dynamic_body, hx=0.1, hy=0.1, hz=0.05)

        model = builder.finalize()
        nacon = self._get_nacon(model)
        self.assertGreater(nacon, 0, "Expected contacts for kinematic-vs-dynamic, got 0")

    def test_kinematic_vs_fixed_root_contacts_filtered(self):
        """Contacts between a kinematic body and a fixed-root body must be filtered.

        Both sides are immovable: the kinematic body via BodyFlags.KINEMATIC,
        the fixed-root body via its weld group having no degrees of freedom.
        """
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0

        kinematic_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
            is_kinematic=True,
        )
        builder.add_shape_box(kinematic_body, hx=0.2, hy=0.2, hz=0.05)

        fixed_root = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        j = builder.add_joint_fixed(parent=-1, child=fixed_root)
        builder.add_articulation([j])
        builder.add_shape_box(fixed_root, hx=0.2, hy=0.2, hz=0.05)

        model = builder.finalize()
        nacon = self._get_nacon(model)
        self.assertEqual(nacon, 0, f"Expected 0 MuJoCo contacts for kinematic-vs-fixed-root, got {nacon}")

    def test_fixed_root_vs_ground_contacts_filtered(self):
        """Contacts between a fixed-root body and the ground plane must be filtered."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()

        fixed_root = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        j = builder.add_joint_fixed(parent=-1, child=fixed_root)
        builder.add_articulation([j])
        builder.add_shape_box(fixed_root, hx=0.2, hy=0.2, hz=0.01)

        model = builder.finalize()
        nacon = self._get_nacon(model)
        self.assertEqual(nacon, 0, f"Expected 0 MuJoCo contacts for fixed-root-vs-ground, got {nacon}")

    def test_dynamic_vs_fixed_root_contacts_preserved(self):
        """Contacts between a dynamic body and a fixed-root body must be preserved."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0

        fixed_root = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        j = builder.add_joint_fixed(parent=-1, child=fixed_root)
        builder.add_articulation([j])
        builder.add_shape_box(fixed_root, hx=0.5, hy=0.5, hz=0.05)

        dynamic_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_shape_box(dynamic_body, hx=0.1, hy=0.1, hz=0.05)

        model = builder.finalize()
        nacon = self._get_nacon(model)
        self.assertGreater(nacon, 0, "Expected contacts for dynamic-vs-fixed-root, got 0")

    def test_two_fixed_root_bodies_contacts_filtered(self):
        """Contacts between two fixed-root bodies must be filtered.

        Neither body's weld group has any degrees of freedom, so both are
        immovable and the contact should be skipped.
        """
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0

        root_a = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        ja = builder.add_joint_fixed(parent=-1, child=root_a)
        builder.add_articulation([ja])
        builder.add_shape_box(root_a, hx=0.2, hy=0.2, hz=0.1)

        root_b = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        jb = builder.add_joint_fixed(parent=-1, child=root_b)
        builder.add_articulation([jb])
        builder.add_shape_box(root_b, hx=0.2, hy=0.2, hz=0.1)

        model = builder.finalize()
        nacon = self._get_nacon(model)
        self.assertEqual(nacon, 0, f"Expected 0 MuJoCo contacts for two-fixed-root-bodies, got {nacon}")

    def test_solver_does_not_freeze_with_kinematic_free_joint_on_ground(self):
        """A kinematic free-joint body on the ground must not freeze the solver.

        This is the key scenario from the MR comment: kinematic bodies with
        non-fixed joints (free joint + high armature) produce near-zero invweight
        that was not caught by the old weld-based filter.
        """
        # Build a scene with a kinematic body and a dynamic body
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()

        kb = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
            is_kinematic=True,
        )
        builder.add_shape_box(kb, hx=0.1, hy=0.1, hz=0.05)

        db = builder.add_body(
            xform=wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_shape_sphere(db, radius=0.1)
        model = builder.finalize()

        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed: {e}")

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        for _ in range(60):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in

        # The dynamic sphere should have fallen and settled on the ground
        sphere_z = state_in.body_q.numpy()[db, 2]
        self.assertGreater(sphere_z, 0.05, "Dynamic sphere fell through the ground")
        self.assertLess(sphere_z, 1.5, "Dynamic sphere is not settling (solver may be frozen)")


class TestMuJoCoContactForce(unittest.TestCase):
    """Verify that contact forces from Newton's MuJoCo solver are physically correct."""

    BOX_MASS = 8.0  # density=1000 kg/m³ * volume=(0.2*0.2*0.2) m³
    GRAVITY = 9.81

    def test_legacy_geometry_export_without_force_buffer(self):
        """Keep geometry-only contact export available without a deprecation warning."""
        model, _ = self._build_box_on_ground()
        solver = SolverMuJoCo(model)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=solver.get_max_contact_count())
        contacts = pipeline.contacts()
        self.assertIsNone(contacts.force)
        solver.step(model.state(), model.state(), None, None, 0.002)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            solver.update_contacts(contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        self.assertTrue(np.isfinite(contacts.rigid_contact_point0.numpy()[:count]).all())
        self.assertIsNone(contacts.force)

    def test_external_contact_observables_preserve_input_geometry(self):
        """Keep collision rows stable across substeps and map forces past culled rows."""
        model, _ = self._build_box_on_ground()
        solver = SolverMuJoCo(model, use_mujoco_contacts=False)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
        self.assertLess(model.rigid_contact_max, solver.get_max_contact_count())
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        joint_q = state_in.joint_q.numpy()
        joint_q[2] = 0.09  # Start in contact, rather than waiting for gravity to close the gap.
        state_in.joint_q.assign(joint_q)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline.collide(state_in, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 1)
        # An invalid first row is culled during Newton -> MuJoCo conversion,
        # making the compacted MuJoCo indices differ from the input row indices.
        shapes = contacts.rigid_contact_shape0.numpy()
        shapes[0] = -1
        contacts.rigid_contact_shape0.assign(shapes)
        fields = (
            "rigid_contact_count",
            "rigid_contact_shape0",
            "rigid_contact_shape1",
            "rigid_contact_point0",
            "rigid_contact_point1",
            "rigid_contact_normal",
            "rigid_contact_offset0",
            "rigid_contact_offset1",
            "rigid_contact_margin0",
            "rigid_contact_margin1",
            "contact_generation",
        )
        snapshots = {field: getattr(contacts, field).numpy().copy() for field in fields}
        observables = solver.observables({newton.solvers.SolverObservableFlags.CONTACT_F})
        for _ in range(3):
            observables.contact_f.fill_(float("nan"))
            solver.step(state_in, state_out, None, contacts, 0.002, observables=observables)
            for field, expected in snapshots.items():
                np.testing.assert_array_equal(getattr(contacts, field).numpy(), expected, err_msg=field)
            forces = observables.contact_f.numpy()
            np.testing.assert_array_equal(forces[0], 0.0)
            np.testing.assert_array_equal(forces[count:], 0.0)
            self.assertTrue(np.isfinite(forces).all())
            self.assertGreater(np.linalg.norm(forces[1:count, :3]), 0.0)
            state_in, state_out = state_out, state_in

    def _build_box_on_ground(self, *, friction: float = 1.0):
        """Create a box resting on a ground plane with the given friction."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.default_shape_cfg.mu = friction
        ground_shape = builder.add_ground_plane()
        # Place the box so its bottom face touches the ground (hz=0.1 → center at z=0.1)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize()
        return model, ground_shape

    def _run_and_collect_forces(self, model, cone: str = "pyramidal", settle: int = 10, avg: int = 10):
        """Run simulation, settle, then average total contact force over *avg* steps.

        Returns:
            force: Averaged total linear contact force, shape (3,).
            shape0: Shape index of shape0 in the first contact.
        """
        try:
            solver = SolverMuJoCo(model, cone=cone)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(
            model, rigid_contact_max=solver.get_max_contact_count(), soft_contact_max=0
        )
        contacts = collision_pipeline.contacts()
        observables = solver.observables({newton.solvers.SolverObservableFlags.CONTACT_F})
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

        dt = 0.002
        for _ in range(settle):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, contacts, dt, observables=observables)
            state_in, state_out = state_out, state_in

        force_acc = np.zeros(3)
        for _ in range(avg):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, contacts, dt, observables=observables)
            state_in, state_out = state_out, state_in

            nacon = int(solver.mjw_data.nacon.numpy()[0])
            if nacon > 0:
                f = observables.contact_f.numpy()[:nacon, :3]
                force_acc += np.sum(f, axis=0)

        total_force = force_acc / avg
        shape0 = int(contacts.rigid_contact_shape0.numpy()[0])
        return total_force, shape0

    def test_pyramidal_cone_weight(self):
        """Box weight via pyramidal cone contacts must match mg."""
        model, ground_shape = self._build_box_on_ground(friction=0.5)
        force, shape0 = self._run_and_collect_forces(model, cone="pyramidal")
        # Force on shape0: if ground is shape0 the box pushes it down (-Z), otherwise up (+Z).
        expected_fz = -self.BOX_MASS * self.GRAVITY if shape0 == ground_shape else self.BOX_MASS * self.GRAVITY
        np.testing.assert_allclose(force[2], expected_fz, rtol=0.05)
        # Horizontal forces should be near zero for a resting box.
        np.testing.assert_allclose(force[0], 0.0, atol=1.0)
        np.testing.assert_allclose(force[1], 0.0, atol=1.0)

    def test_elliptic_cone_weight(self):
        """Box weight via elliptic cone contacts must match mg."""
        model, ground_shape = self._build_box_on_ground(friction=0.5)
        force, shape0 = self._run_and_collect_forces(model, cone="elliptic")
        expected_fz = -self.BOX_MASS * self.GRAVITY if shape0 == ground_shape else self.BOX_MASS * self.GRAVITY
        np.testing.assert_allclose(force[2], expected_fz, rtol=0.05)
        # Horizontal forces should be near zero for a resting box.
        np.testing.assert_allclose(force[0], 0.0, atol=1.0)
        np.testing.assert_allclose(force[1], 0.0, atol=1.0)

    def _build_incline_model(self, incline_angle: float):
        """Create a box resting on an inclined ramp with mu=1.0 (static for angle < ~45°)."""
        hz = 0.1  # box half-height
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.default_shape_cfg.mu = 1.0
        # Static box as inclined ground (add_ground_plane doesn't support rotation)
        ramp_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), incline_angle)
        ramp_shape = builder.add_shape_box(
            body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), ramp_q), hx=5.0, hy=5.0, hz=0.5
        )
        # Place box center exactly hz above the ramp surface to avoid bounce.
        # The ramp top face is at ramp_center + 0.5 * normal (not at the origin).
        ramp_center = np.array([0.0, 0.0, -0.5])
        normal = np.array([0.0, -np.sin(incline_angle), np.cos(incline_angle)])
        box_center = ramp_center + (0.5 + hz) * normal
        body = builder.add_body(xform=wp.transform(wp.vec3(*box_center), ramp_q))
        builder.add_shape_box(body=body, hx=hz, hy=hz, hz=hz)
        model = builder.finalize()
        return model, ramp_shape

    def test_contact_forces_on_incline(self):
        """Contact force on an incline must balance gravity (tests rotated contact frame)."""
        incline_angle = 0.25  # rad (~14°); mu=1.0 > tan(0.25)≈0.26 → static
        model, ramp_shape = self._build_incline_model(incline_angle)
        force, shape0 = self._run_and_collect_forces(model, cone="pyramidal", settle=300, avg=80)
        expected_fz = -self.BOX_MASS * self.GRAVITY if shape0 == ramp_shape else self.BOX_MASS * self.GRAVITY
        np.testing.assert_allclose(force[2], expected_fz, rtol=0.05)


class TestContactObservablePointPositions(unittest.TestCase):
    """Test that contact observables populate rigid_contact_point0/point1."""

    def test_contact_points_populated(self):
        """Drop a box onto a ground plane with use_mujoco_contacts and verify contact points are nonzero."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        builder.add_ground_plane()
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
        )
        builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
        model = builder.finalize()

        solver = SolverMuJoCo(
            model,
            solver="newton",
            integrator="implicitfast",
            iterations=10,
            ls_iterations=20,
            use_mujoco_contacts=True,
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=solver.get_max_contact_count(), soft_contact_max=0)
        contacts = pipeline.contacts()
        observables = solver.observables({newton.solvers.SolverObservableFlags.CONTACT_F})
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        dt = 1.0 / 200.0
        found_contacts = False
        for _ in range(200):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, dt, observables=observables)
            state_0, state_1 = state_1, state_0

            n = contacts.rigid_contact_count.numpy()[0]
            if n > 0:
                found_contacts = True
                point0 = contacts.rigid_contact_point0.numpy()[:n]
                point1 = contacts.rigid_contact_point1.numpy()[:n]

                hx, hy, hz = 0.25, 0.25, 0.25
                for i in range(n):
                    # point0 is on the ground plane (body-local = world for static body).
                    # z should be near 0 (ground surface), x/y within box footprint.
                    self.assertAlmostEqual(point0[i][2], 0.0, delta=0.02)
                    self.assertLessEqual(abs(float(point0[i][0])), hx + 0.01)
                    self.assertLessEqual(abs(float(point0[i][1])), hy + 0.01)

                    # point1 is in box body frame.
                    # z should be near -hz (bottom face), x/y within box half-extents.
                    self.assertAlmostEqual(point1[i][2], -hz, delta=0.02)
                    self.assertLessEqual(abs(float(point1[i][0])), hx + 0.01)
                    self.assertLessEqual(abs(float(point1[i][1])), hy + 0.01)
                break

        self.assertTrue(found_contacts, "No contacts detected after 200 steps")
