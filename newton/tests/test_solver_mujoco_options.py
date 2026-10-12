# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco.equality import _add_equality_constraint
from newton.solvers import SolverMuJoCo
from newton.tests.utils.mujoco import _large_nv_mjcf


class TestMuJoCoOptions(unittest.TestCase):
    """Tests for MuJoCo solver options (impratio, etc.) with WORLD frequency."""

    def _create_multiworld_model(self, world_count=3):
        """Helper to create a multi-world model with MuJoCo custom attributes registered."""
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        pendulum = template_builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        template_builder.add_shape_box(body=pendulum, hx=0.05, hy=0.05, hz=0.05)
        joint = template_builder.add_joint_revolute(parent=-1, child=pendulum, axis=(0.0, 0.0, 1.0))
        template_builder.add_articulation([joint])

        builder = newton.ModelBuilder()
        builder.replicate(template_builder, world_count)
        return builder.finalize()

    def test_njmax_nnz_constructor_override(self):
        """Forward an explicit sparse Jacobian capacity to MuJoCo Warp."""
        model = self._create_multiworld_model(world_count=2)

        solver = SolverMuJoCo(
            model,
            iterations=1,
            disable_contacts=True,
            jacobian="sparse",
            njmax_nnz=123,
        )

        self.assertEqual(solver.mjw_data.njmax_nnz, 123)
        self.assertEqual(solver.mjw_data.efc.J.shape, (2, 1, 123))

    def test_njmax_nnz_rejects_invalid_values(self):
        """Reject invalid sparse Jacobian capacities."""
        model = self._create_multiworld_model(world_count=1)

        for value in (True, 1.5, "123"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "njmax_nnz must be an integer or None"):
                    SolverMuJoCo(model, njmax_nnz=value)

        with self.assertRaisesRegex(ValueError, "njmax_nnz must be non-negative"):
            SolverMuJoCo(model, njmax_nnz=-1)

    def test_njmax_nnz_rejects_zero_sparse_capacity(self):
        """Reject zero sparse capacity whether or not a limit is initially active."""
        for limit_lower, limit_upper, initial_required_nnz in ((-1.0, 1.0, 0), (1.0, 2.0, 1)):
            with self.subTest(initial_required_nnz=initial_required_nnz):
                builder = newton.ModelBuilder()
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                joint = builder.add_joint_revolute(
                    parent=-1,
                    child=body,
                    limit_lower=limit_lower,
                    limit_upper=limit_upper,
                )
                builder.add_articulation([joint])

                with self.assertRaisesRegex(
                    ValueError,
                    rf"njmax_nnz=0 is too small:.*initial Jacobian contains {initial_required_nnz} nonzeros",
                ):
                    SolverMuJoCo(
                        builder.finalize(),
                        disable_contacts=True,
                        jacobian="sparse",
                        njmax_nnz=0,
                    )

    def test_njmax_nnz_rejects_capacity_below_initial_nnz(self):
        """Reject a positive sparse capacity below the initial Jacobian's requirement."""
        builder = newton.ModelBuilder()
        for _ in range(2):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_revolute(parent=-1, child=body, limit_lower=1.0, limit_upper=2.0)
            builder.add_articulation([joint])

        with self.assertRaisesRegex(
            ValueError,
            r"njmax_nnz=1 is too small:.*capacity of at least 2; the initial Jacobian contains 2 nonzeros",
        ):
            SolverMuJoCo(
                builder.finalize(),
                disable_contacts=True,
                jacobian="sparse",
                njmax_nnz=1,
            )

    def test_njmax_nnz_rejects_dense_initial_jacobian_in_sparse_storage(self):
        """Reject an undersized capacity when MuJoCo's dense Jacobian is stored sparsely by MuJoCo Warp."""
        builder = newton.ModelBuilder()
        for joint_index in range(33):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            if joint_index < 2:
                joint = builder.add_joint_revolute(parent=-1, child=body, limit_lower=1.0, limit_upper=2.0)
            else:
                joint = builder.add_joint_revolute(parent=-1, child=body)
            builder.add_articulation([joint])
        model = builder.finalize()

        solver = SolverMuJoCo(
            model,
            disable_contacts=True,
            jacobian="auto",
            njmax_nnz=2,
        )
        mujoco, _ = SolverMuJoCo.import_mujoco()
        from mujoco_warp._src.io import is_sparse as is_mujoco_warp_sparse

        self.assertFalse(mujoco.mj_isSparse(solver.mj_model))
        self.assertTrue(is_mujoco_warp_sparse(solver.mj_model))

        with self.assertRaisesRegex(
            ValueError,
            r"njmax_nnz=1 is too small:.*capacity of at least 2; the initial Jacobian contains 2 nonzeros",
        ):
            SolverMuJoCo(
                model,
                disable_contacts=True,
                jacobian="auto",
                njmax_nnz=1,
            )

    def test_impratio_multiworld_conversion(self):
        """
        Verify that impratio custom attribute with WORLD frequency:
        1. Is properly registered and exists on the model.
        2. The array has correct shape (one value per world).
        3. Different per-world values are stored correctly in the Newton model.
        4. Solver expands per-world values to MuJoCo Warp.
        """
        world_count = 3
        model = self._create_multiworld_model(world_count)

        # Verify the custom attribute is registered and exists on the model
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "impratio"))

        # Verify the array has correct shape (one value per world)
        impratio = model.mujoco.impratio.numpy()
        self.assertEqual(len(impratio), world_count, "impratio array should have one entry per world")

        # Set different impratio values per world
        initial_impratio = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        model.mujoco.impratio.assign(initial_impratio)

        # Verify all per-world values are stored correctly in Newton model
        updated_impratio = model.mujoco.impratio.numpy()
        for world_idx in range(world_count):
            self.assertAlmostEqual(
                updated_impratio[world_idx],
                initial_impratio[world_idx],
                places=4,
                msg=f"Newton model impratio[{world_idx}] should be {initial_impratio[world_idx]}",
            )

        # Create solver without constructor override
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo Warp model has per-world impratio_invsqrt values
        mjw_impratio_invsqrt = solver.mjw_model.opt.impratio_invsqrt.numpy()
        self.assertEqual(
            len(mjw_impratio_invsqrt),
            world_count,
            f"MuJoCo Warp opt.impratio_invsqrt should have {world_count} values (one per world)",
        )

        # Verify each world has the correct impratio_invsqrt value (1/sqrt(impratio))
        for world_idx in range(world_count):
            expected_invsqrt = 1.0 / np.sqrt(initial_impratio[world_idx])
            self.assertAlmostEqual(
                mjw_impratio_invsqrt[world_idx],
                expected_invsqrt,
                places=4,
                msg=f"MuJoCo Warp impratio_invsqrt[{world_idx}] should be {expected_invsqrt}",
            )

    def test_impratio_invalid_values_guarded(self):
        """
        Verify that zero or negative impratio values are guarded against
        to prevent NaN/Inf in opt_impratio_invsqrt computation.
        """
        world_count = 3
        model = self._create_multiworld_model(world_count)

        # Set impratio with invalid values: 0, negative, and positive
        initial_impratio = np.array([0.0, -1.0, 2.0], dtype=np.float32)
        model.mujoco.impratio.assign(initial_impratio)

        # Create solver - should not crash or produce NaN/Inf
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo Warp model has valid impratio_invsqrt values
        mjw_impratio_invsqrt = solver.mjw_model.opt.impratio_invsqrt.numpy()
        self.assertEqual(len(mjw_impratio_invsqrt), world_count)

        # World 0 (impratio=0): should keep MuJoCo default (not update)
        self.assertFalse(
            np.isnan(mjw_impratio_invsqrt[0]),
            "impratio=0 should not produce NaN",
        )
        self.assertFalse(
            np.isinf(mjw_impratio_invsqrt[0]),
            "impratio=0 should not produce Inf",
        )

        # World 1 (impratio=-1): should keep MuJoCo default (not update)
        self.assertFalse(
            np.isnan(mjw_impratio_invsqrt[1]),
            "impratio=-1 should not produce NaN",
        )
        self.assertFalse(
            np.isinf(mjw_impratio_invsqrt[1]),
            "impratio=-1 should not produce Inf",
        )

        # World 2 (impratio=2): should compute correctly
        expected_invsqrt = 1.0 / np.sqrt(2.0)
        self.assertAlmostEqual(
            mjw_impratio_invsqrt[2],
            expected_invsqrt,
            places=4,
            msg=f"impratio=2.0 should produce valid impratio_invsqrt={expected_invsqrt}",
        )

    def test_options_use_custom_attributes(self):
        """Resolve enum, iteration, and shared numeric options from custom attributes."""
        model = self._create_multiworld_model(world_count=3)
        model.mujoco.solver.assign(np.array([1], dtype=np.int32))  # CG
        model.mujoco.integrator.assign(np.array([0], dtype=np.int32))  # Euler
        model.mujoco.cone.assign(np.array([1], dtype=np.int32))  # elliptic
        model.mujoco.jacobian.assign(np.array([1], dtype=np.int32))  # sparse
        model.mujoco.iterations.assign(np.array([150], dtype=np.int32))
        model.mujoco.ls_iterations.assign(np.array([75], dtype=np.int32))
        ccd_iterations = model.mujoco.ccd_iterations.numpy()
        sdf_iterations = model.mujoco.sdf_iterations.numpy()
        sdf_initpoints = model.mujoco.sdf_initpoints.numpy()
        self.assertEqual(len(ccd_iterations), 1, "ONCE frequency should have single value")
        self.assertEqual(len(sdf_iterations), 1, "ONCE frequency should have single value")
        self.assertEqual(len(sdf_initpoints), 1, "ONCE frequency should have single value")
        model.mujoco.ccd_iterations.assign(np.array([25], dtype=np.int32))
        model.mujoco.sdf_iterations.assign(np.array([20], dtype=np.int32))
        model.mujoco.sdf_initpoints.assign(np.array([50], dtype=np.int32))

        solver = SolverMuJoCo(model, disable_contacts=True)
        mujoco = SolverMuJoCo._mujoco
        self.assertEqual(
            solver.mj_model.opt.solver, mujoco.mjtSolver.mjSOL_CG, "Should use custom attribute CG, not Newton default"
        )
        self.assertEqual(
            solver.mj_model.opt.integrator,
            mujoco.mjtIntegrator.mjINT_EULER,
            "Should use custom attribute Euler, not Newton default implicitfast",
        )
        self.assertEqual(
            solver.mj_model.opt.cone,
            mujoco.mjtCone.mjCONE_ELLIPTIC,
            "Should use custom attribute elliptic, not Newton default pyramidal",
        )
        self.assertEqual(
            solver.mj_model.opt.jacobian,
            mujoco.mjtJacobian.mjJAC_SPARSE,
            "Should use custom attribute sparse, not Newton default auto",
        )
        self.assertEqual(solver.mj_model.opt.iterations, 150, "Should use custom attribute 150, not default 100")
        self.assertEqual(solver.mj_model.opt.ls_iterations, 75, "Should use custom attribute 75, not default 50")
        self.assertEqual(solver.mj_model.opt.ccd_iterations, 25)
        self.assertEqual(solver.mj_model.opt.sdf_iterations, 20)
        self.assertEqual(solver.mj_model.opt.sdf_initpoints, 50)

    def test_options_constructor_override(self):
        """Give constructor options precedence over authored values in every world."""
        world_count = 2
        model = self._create_multiworld_model(world_count)
        model.mujoco.impratio.assign(np.array([1.5, 1.5], dtype=np.float32))
        model.mujoco.tolerance.assign(np.array([1e-6, 1e-7], dtype=np.float32))
        model.mujoco.ls_tolerance.assign(np.array([0.01, 0.02], dtype=np.float32))
        model.mujoco.ccd_tolerance.assign(np.array([1e-6, 1e-7], dtype=np.float32))
        model.mujoco.density.assign(np.array([0.0, 0.0], dtype=np.float32))
        model.mujoco.viscosity.assign(np.array([0.0, 0.0], dtype=np.float32))
        model.mujoco.ccd_iterations.assign(np.array([25], dtype=np.int32))
        model.mujoco.sdf_iterations.assign(np.array([20], dtype=np.int32))
        model.mujoco.sdf_initpoints.assign(np.array([50], dtype=np.int32))
        model.mujoco.jacobian.assign(np.array([1], dtype=np.int32))
        model.mujoco.iterations.assign(np.array([150], dtype=np.int32))
        model.mujoco.ls_iterations.assign(np.array([75], dtype=np.int32))

        # Nonzero density/viscosity require fluid dynamics, which MuJoCo Warp does not support.
        solver = SolverMuJoCo(
            model,
            impratio=3.0,
            tolerance=1e-5,
            ls_tolerance=0.001,
            ccd_tolerance=1e-4,
            density=0.0,
            viscosity=0.0,
            ccd_iterations=100,
            sdf_iterations=30,
            sdf_initpoints=80,
            jacobian="dense",
            iterations=5,
            ls_iterations=3,
            disable_contacts=True,
        )
        mjw_impratio_invsqrt = solver.mjw_model.opt.impratio_invsqrt.numpy()
        mjw_tolerance = solver.mjw_model.opt.tolerance.numpy()
        mjw_ls_tolerance = solver.mjw_model.opt.ls_tolerance.numpy()
        mjw_ccd_tolerance = solver.mjw_model.opt.ccd_tolerance.numpy()
        mjw_density = solver.mjw_model.opt.density.numpy()
        mjw_viscosity = solver.mjw_model.opt.viscosity.numpy()
        self.assertEqual(len(mjw_impratio_invsqrt), world_count)
        self.assertEqual(len(mjw_tolerance), world_count)
        self.assertEqual(len(mjw_ls_tolerance), world_count)
        self.assertEqual(len(mjw_ccd_tolerance), world_count)
        self.assertEqual(len(mjw_density), world_count)
        self.assertEqual(len(mjw_viscosity), world_count)
        expected_impratio_invsqrt = 1.0 / np.sqrt(3.0)
        for world_idx in range(world_count):
            self.assertAlmostEqual(
                mjw_impratio_invsqrt[world_idx],
                expected_impratio_invsqrt,
                places=4,
                msg=f"impratio_invsqrt[{world_idx}] should be {expected_impratio_invsqrt}",
            )
            self.assertAlmostEqual(
                mjw_tolerance[world_idx], 1e-5, places=10, msg=f"tolerance[{world_idx}] should be 1e-5"
            )
            self.assertAlmostEqual(
                mjw_ls_tolerance[world_idx], 0.001, places=6, msg=f"ls_tolerance[{world_idx}] should be 0.001"
            )
            self.assertAlmostEqual(
                mjw_ccd_tolerance[world_idx], 1e-4, places=10, msg=f"ccd_tolerance[{world_idx}] should be 1e-4"
            )
            self.assertAlmostEqual(mjw_density[world_idx], 0.0, places=6, msg=f"density[{world_idx}] should be 0.0")
            self.assertAlmostEqual(
                mjw_viscosity[world_idx], 0.0, places=10, msg=f"viscosity[{world_idx}] should be 0.0"
            )
        self.assertEqual(solver.mj_model.opt.ccd_iterations, 100, "Constructor should override custom attribute")
        self.assertEqual(solver.mj_model.opt.sdf_iterations, 30, "Constructor should override custom attribute")
        self.assertEqual(solver.mj_model.opt.sdf_initpoints, 80, "Constructor should override custom attribute")
        self.assertEqual(solver.mj_model.opt.jacobian, SolverMuJoCo._mujoco.mjtJacobian.mjJAC_DENSE)
        self.assertEqual(solver.mj_model.opt.iterations, 5, "Constructor value should override custom attribute")
        self.assertEqual(solver.mj_model.opt.ls_iterations, 3, "Constructor value should override custom attribute")

    def test_options_use_defaults(self):
        """
        Verify that solver, integrator, cone, and jacobian use Newton defaults
        when no constructor parameter or custom attribute is provided.
        """
        # Create model WITHOUT registering custom attributes
        builder = newton.ModelBuilder()
        pendulum = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body=pendulum, hx=0.05, hy=0.05, hz=0.05)
        joint = builder.add_joint_revolute(parent=-1, child=pendulum, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([joint])
        model = builder.finalize()

        # Create solver without specifying enum options - should use Newton defaults
        solver = SolverMuJoCo(model, disable_contacts=True)
        mujoco = SolverMuJoCo._mujoco

        # Verify Newton defaults are used
        # Newton defaults: solver=Newton(2), integrator=implicitfast(3), cone=pyramidal(0), jacobian=auto(2)
        self.assertEqual(
            solver.mj_model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON, "Should use Newton default (Newton solver)"
        )
        self.assertEqual(
            solver.mj_model.opt.integrator,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            "Should use Newton default (implicitfast)",
        )
        self.assertEqual(
            solver.mj_model.opt.cone, mujoco.mjtCone.mjCONE_PYRAMIDAL, "Should use Newton default (pyramidal)"
        )
        self.assertEqual(
            solver.mj_model.opt.jacobian, mujoco.mjtJacobian.mjJAC_AUTO, "Should use Newton default (auto)"
        )

        self.assertEqual(solver.mj_model.opt.iterations, 100, "Should use MuJoCo default (100)")
        self.assertEqual(solver.mj_model.opt.ls_iterations, 50, "Should use MuJoCo default (50)")

    def test_vector_options_multiworld_conversion(self):
        """
        Verify that vector options (wind, magnetic) with WORLD frequency:
        1. Are properly registered and exist on the model.
        2. Arrays have correct shape (one vec3 per world).
        3. Different per-world vector values are stored correctly.
        4. Solver expands per-world vectors to MuJoCo Warp.
        """
        world_count = 3
        model = self._create_multiworld_model(world_count)

        # Verify arrays have correct shape
        wind = model.mujoco.wind.numpy()
        magnetic = model.mujoco.magnetic.numpy()
        self.assertEqual(len(wind), world_count, "wind array should have one entry per world")
        self.assertEqual(len(magnetic), world_count, "magnetic array should have one entry per world")

        # Set different vector values per world
        initial_wind = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        initial_magnetic = np.array([[0.0, -0.5, 0.0], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
        model.mujoco.wind.assign(initial_wind)
        model.mujoco.magnetic.assign(initial_magnetic)

        # Verify values stored correctly
        updated_wind = model.mujoco.wind.numpy()
        updated_magnetic = model.mujoco.magnetic.numpy()
        for world_idx in range(world_count):
            self.assertTrue(
                np.allclose(updated_wind[world_idx], initial_wind[world_idx]),
                msg=f"Newton model wind[{world_idx}] should be {initial_wind[world_idx]}",
            )
            self.assertTrue(
                np.allclose(updated_magnetic[world_idx], initial_magnetic[world_idx]),
                msg=f"Newton model magnetic[{world_idx}] should be {initial_magnetic[world_idx]}",
            )

        # Create solver without constructor override
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo Warp has per-world vector values
        mjw_wind = solver.mjw_model.opt.wind.numpy()
        mjw_magnetic = solver.mjw_model.opt.magnetic.numpy()
        self.assertEqual(len(mjw_wind), world_count, f"MuJoCo Warp opt.wind should have {world_count} values")
        self.assertEqual(len(mjw_magnetic), world_count, f"MuJoCo Warp opt.magnetic should have {world_count} values")

        # Verify each world has correct values
        for world_idx in range(world_count):
            self.assertTrue(
                np.allclose(mjw_wind[world_idx], initial_wind[world_idx]),
                msg=f"MuJoCo Warp wind[{world_idx}] should be {initial_wind[world_idx]}",
            )
            self.assertTrue(
                np.allclose(mjw_magnetic[world_idx], initial_magnetic[world_idx]),
                msg=f"MuJoCo Warp magnetic[{world_idx}] should be {initial_magnetic[world_idx]}",
            )

    def test_disable_sensors_computes_rne_state_attributes(self):
        """Compute legacy body diagnostics even with sensors disabled."""
        self._check_disable_sensors_rne(legacy_state_attributes=True)

    def test_disable_sensors_computes_rne_observables(self):
        """Compute each RNE observable without requesting legacy state arrays."""
        for flag in (
            newton.solvers.SolverObservableFlags.BODY_QDD,
            newton.solvers.SolverObservableFlags.BODY_PARENT_F,
        ):
            with self.subTest(flag=flag):
                self._check_disable_sensors_rne(observable_flag=flag)

    def _check_disable_sensors_rne(self, legacy_state_attributes=False, observable_flag=None):
        """Compare diagnostics against a sensors-enabled reference.

        A horizontal pendulum released from rest has nonzero body acceleration
        and joint wrench, so stale zero-initialized outputs would not match the
        sensors-enabled reference.
        """
        results = {}
        attributes = ("body_qdd", "body_parent_f") if legacy_state_attributes else (observable_flag.value,)
        for disable_sensors in (False, True):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
            if legacy_state_attributes:
                with self.assertWarnsRegex(DeprecationWarning, r"ModelBuilder\.request_state_attributes.*1\.7"):
                    builder.request_state_attributes(*attributes)
            link = builder.add_link()
            builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
            joint = builder.add_joint_revolute(
                -1,
                link,
                child_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0), wp.quat_identity()),
                axis=wp.vec3(0.0, 1.0, 0.0),
            )
            builder.add_articulation([joint])
            model = builder.finalize()

            solver = SolverMuJoCo(model, disable_sensors=disable_sensors, disable_contacts=True)
            state_in, state_out = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
            observables = None
            if not legacy_state_attributes:
                observables = solver.observables({observable_flag})
                self.assertIsNone(state_out.body_qdd)
                self.assertIsNone(state_out.body_parent_f)
            solver.step(state_in, state_out, None, None, 1.0e-3, observables=observables)
            source = state_out if legacy_state_attributes else observables
            results[disable_sensors] = [getattr(source, attribute).numpy() for attribute in attributes]
            self.assertTrue(solver.mjw_model.opt.run_rne_postconstraint)

        for result, reference in zip(results[True], results[False], strict=True):
            self.assertGreater(np.linalg.norm(reference), 0.1)
            np.testing.assert_allclose(result, reference, rtol=1.0e-5, atol=1.0e-5)

    def test_disable_sensors_allows_unrelated_state_attributes(self):
        """Step with disabled sensors when no RNE-derived state attribute is requested."""
        model = self._create_multiworld_model(world_count=1)
        solver = SolverMuJoCo(model, disable_sensors=True)
        state_in, state_out = model.state(), model.state()
        solver.step(state_in, state_out, model.control(), None, 0.01)

    def test_selected_actuator_observable_skips_rne_request(self):
        """Select a custom actuator field without requesting allocated body diagnostics."""
        model = self._create_multiworld_model(world_count=1)
        solver = SolverMuJoCo(model, disable_sensors=True)
        flags = newton.solvers.SolverObservableFlags
        actuator_flag = SolverMuJoCo.ObservableFlags.QFRC_ACTUATOR
        observables = solver.observables({flags.BODY_QDD, actuator_flag})
        selected = observables.select({actuator_flag})
        self.assertIs(type(selected), SolverMuJoCo.Observables)
        self.assertIs(selected.qfrc_actuator, observables.qfrc_actuator)
        observables.body_qdd.fill_(wp.spatial_vector(-1.0))
        observables.qfrc_actuator.fill_(float("nan"))
        state_in, state_out = model.state(), model.state()
        solver.step(state_in, state_out, model.control(), None, 0.01, observables=selected)
        self.assertFalse(solver.mjw_model.opt.run_rne_postconstraint)
        np.testing.assert_array_equal(observables.body_qdd.numpy(), np.full((model.body_count, 6), -1.0))
        self.assertTrue(np.isfinite(observables.qfrc_actuator.numpy()).all())

    def test_native_cpu_observable_capabilities(self):
        """Reject unsupported body/contact exports while retaining native actuator forces."""
        model = self._create_multiworld_model(world_count=1)
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        for flag in newton.solvers.SolverObservableFlags:
            with self.subTest(flag=flag):
                self.assertNotIn(flag, solver.supported_observable_flags)
                with self.assertRaisesRegex(ValueError, "does not support"):
                    solver.observables({flag})
        flag = SolverMuJoCo.ObservableFlags.QFRC_ACTUATOR
        observables = solver.observables({flag})
        observables.qfrc_actuator.fill_(float("nan"))
        solver.mjw_data.qfrc_actuator.fill_(1234.0)
        solver.step(model.state(), model.state(), model.control(), None, 0.01, observables=observables)
        np.testing.assert_allclose(observables.qfrc_actuator.numpy(), solver.mj_data.qfrc_actuator, atol=1e-6)

    def test_enable_multiccd_default_off(self):
        """Verify that multi-CCD is disabled by default (Newton default differs from MuJoCo 3.8+)."""
        model = self._create_multiworld_model(world_count=1)
        solver = SolverMuJoCo(model)
        mujoco = SolverMuJoCo._mujoco
        self.assertTrue(
            solver.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD,
            "mjDSBL_MULTICCD should be set when enable_multiccd is not specified",
        )

    def test_enable_multiccd_passed_to_mujoco(self):
        """Verify that enable_multiccd clears mjDSBL_MULTICCD on the MuJoCo model."""
        model = self._create_multiworld_model(world_count=1)
        solver = SolverMuJoCo(model, enable_multiccd=True)
        mujoco = SolverMuJoCo._mujoco
        self.assertFalse(
            solver.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD,
            "mjDSBL_MULTICCD should not be set when enable_multiccd=True",
        )


class TestMuJoCoValidation(unittest.TestCase):
    """Test cases for SolverMuJoCo._validate_model_for_separate_worlds()."""

    def _create_homogeneous_model(self, world_count=2, with_ground_plane=True):
        """Create a valid homogeneous multi-world model for validation tests."""
        # Create a simple robot template (following pattern from working tests)
        template = newton.ModelBuilder()
        b1 = template.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        b2 = template.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        j1 = template.add_joint_revolute(-1, b1, axis=(0.0, 0.0, 1.0))
        j2 = template.add_joint_revolute(b1, b2, axis=(0.0, 0.0, 1.0))
        template.add_articulation([j1, j2])
        template.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        template.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)

        # Build main model using replicate (pattern from working tests)
        builder = newton.ModelBuilder()
        if with_ground_plane:
            builder.add_ground_plane()  # Global static shape
        builder.replicate(template, world_count)

        return builder.finalize()

    def test_valid_homogeneous_model_passes(self):
        """Test that a valid homogeneous model passes validation."""
        model = self._create_homogeneous_model(world_count=2, with_ground_plane=False)
        # Should not raise
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertIsNotNone(solver)

    def test_valid_model_with_global_shape_passes(self):
        """Test that a model with global static shapes (ground plane) passes validation."""
        model = self._create_homogeneous_model(world_count=2, with_ground_plane=True)
        # Should not raise - global shapes are allowed
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertIsNotNone(solver)

    def test_heterogeneous_body_count_fails(self):
        """Test that different body counts per world raises ValueError."""
        # Create two robots with different body counts
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        robot1.add_articulation([j1])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        j2 = robot2.add_joint_revolute(b1, b2)
        robot2.add_articulation([j1, j2])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot2.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)

        main = newton.ModelBuilder()
        main.add_world(robot1)  # 1 body
        main.add_world(robot2)  # 2 bodies
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("world 0 has 1 bodies", str(ctx.exception).lower())

    def test_heterogeneous_shape_count_fails(self):
        """Test that different shape counts per world raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        robot1.add_articulation([j1])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        robot2.add_articulation([j1])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot2.add_shape_sphere(b1, radius=0.05)  # Extra shape

        main = newton.ModelBuilder()
        main.add_world(robot1)  # 1 shape
        main.add_world(robot2)  # 2 shapes
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("world 0 has 1 shapes", str(ctx.exception).lower())

    def test_mismatched_joint_types_fails(self):
        """Test that different joint types at same position across worlds raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)  # Revolute joint
        robot1.add_articulation([j1])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_prismatic(-1, b1)  # Prismatic joint (different type)
        robot2.add_articulation([j1])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

        main = newton.ModelBuilder()
        main.add_world(robot1)
        main.add_world(robot2)
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("joint types mismatch at position", str(ctx.exception).lower())

    def test_mismatched_joint_dof_counts_fails(self):
        """Test that D6 joints with differing DOF layouts across worlds raises ValueError."""

        # Both worlds use D6 joints but differ in DOF count
        def make_robot(angular_axis_count):
            robot = newton.ModelBuilder()
            b1 = robot.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
            axes = [
                newton.ModelBuilder.JointDofConfig(axis=a) for a in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            ][:angular_axis_count]
            j1 = robot.add_joint_d6(-1, b1, linear_axes=[], angular_axes=axes)
            robot.add_articulation([j1])
            robot.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
            return robot

        main = newton.ModelBuilder()
        main.add_world(make_robot(2))
        main.add_world(make_robot(3))
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("joint dof counts mismatch at position", str(ctx.exception).lower())

    def test_mismatched_shape_types_fails(self):
        """Test that different shape types at same position across worlds raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        robot1.add_articulation([j1])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)  # Box

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        robot2.add_articulation([j1])
        robot2.add_shape_sphere(b1, radius=0.1)  # Sphere (different type)

        main = newton.ModelBuilder()
        main.add_world(robot1)
        main.add_world(robot2)
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("shape types mismatch at position", str(ctx.exception).lower())

    def test_global_body_fails(self):
        """Test that a body in global world (-1) raises ValueError."""
        builder = newton.ModelBuilder()

        # Add ground plane (allowed)
        builder.add_ground_plane()

        # Create a body in the default global world
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        # Need a joint to make this a valid model
        j1 = builder.add_joint_free(b1)
        builder.add_articulation([j1])

        # Add normal world content
        builder.begin_world()
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j2 = builder.add_joint_revolute(-1, b2)
        builder.add_articulation([j2])
        builder.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()

        builder.begin_world()
        b3 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j3 = builder.add_joint_revolute(-1, b3)
        builder.add_articulation([j3])
        builder.add_shape_box(b3, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()

        model = builder.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("global world (-1) cannot contain bodies", str(ctx.exception).lower())

    def test_global_joint_fails(self):
        """Test that a joint in global world (-1) raises ValueError."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add a body in the default global world with a joint
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = builder.add_joint_revolute(-1, b1)
        builder.add_articulation([j1])

        # Add normal world content
        builder.begin_world()
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j2 = builder.add_joint_revolute(-1, b2)
        builder.add_articulation([j2])
        builder.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()

        builder.begin_world()
        b3 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j3 = builder.add_joint_revolute(-1, b3)
        builder.add_articulation([j3])
        builder.add_shape_box(b3, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()

        model = builder.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        # Fails on global bodies first (bodies are checked before joints)
        self.assertIn("global world (-1) cannot contain", str(ctx.exception).lower())

    def test_single_world_model_skips_validation(self):
        """Test that single-world models skip validation (no homogeneity needed)."""
        model = self._create_homogeneous_model(world_count=1)

        # Should not raise - single world doesn't need homogeneity validation
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertIsNotNone(solver)

    def test_many_worlds_homogeneous_passes(self):
        """Test that a model with many homogeneous worlds passes validation."""
        model = self._create_homogeneous_model(world_count=10)
        # Should not raise
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertIsNotNone(solver)

    def test_heterogeneous_equality_constraint_count_fails(self):
        """Test that different equality constraint counts per world raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        j2 = robot1.add_joint_revolute(b1, b2)
        robot1.add_articulation([j1, j2])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot1.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        _add_equality_constraint(
            robot1, constraint_type=newton.solvers.SolverMuJoCo.EqType.WELD, body1=b1, body2=b2
        )  # 1 constraint

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        j2 = robot2.add_joint_revolute(b1, b2)
        robot2.add_articulation([j1, j2])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot2.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        # No constraints in robot2

        main = newton.ModelBuilder()
        main.add_world(robot1)  # 1 constraint
        main.add_world(robot2)  # 0 constraints
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("world 0 has 1 equality constraints", str(ctx.exception).lower())

    def test_mismatched_equality_constraint_types_fails(self):
        """Test that different constraint types at same position across worlds raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        j2 = robot1.add_joint_revolute(b1, b2)
        robot1.add_articulation([j1, j2])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot1.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        _add_equality_constraint(
            robot1, constraint_type=newton.solvers.SolverMuJoCo.EqType.WELD, body1=b1, body2=b2
        )  # WELD type

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        j2 = robot2.add_joint_revolute(b1, b2)
        robot2.add_articulation([j1, j2])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot2.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        _add_equality_constraint(
            robot2, constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT, body1=b1, body2=b2
        )  # CONNECT type (different)

        main = newton.ModelBuilder()
        main.add_world(robot1)
        main.add_world(robot2)
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("equality constraint types mismatch at position", str(ctx.exception).lower())

    def test_global_equality_constraint_fails(self):
        """Test that an equality constraint in global world (-1) raises ValueError."""
        # Create a model with a global equality constraint
        robot = newton.ModelBuilder()
        b1 = robot.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot.add_joint_revolute(-1, b1)
        j2 = robot.add_joint_revolute(b1, b2)
        robot.add_articulation([j1, j2])
        robot.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)

        main = newton.ModelBuilder()
        main.add_world(robot)
        main.add_world(robot)

        # Add a global equality constraint outside any world context
        # We need body indices in the main builder - use the first two bodies from world 0
        _add_equality_constraint(main, constraint_type=newton.solvers.SolverMuJoCo.EqType.WELD, body1=0, body2=1)

        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("global world (-1) cannot contain equality constraints", str(ctx.exception).lower())

    def test_body_missing_joint(self):
        """Ensure that each body has an incoming joint and these joints are part of an articulation."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        b0 = builder.add_link()
        b1 = builder.add_link()
        j0 = builder.add_joint_revolute(-1, b0)
        builder.add_joint_revolute(b0, b1)
        builder.add_articulation([j0])
        builder.end_world()
        # we forgot to add the second joint to the articulation
        # finalize() should now catch this and raise an error about orphan joints
        with self.assertRaises(ValueError) as ctx:
            builder.finalize()
        self.assertIn("not belonging to any articulation", str(ctx.exception))


class TestMuJoCoSolverDuplicateBodyNames(unittest.TestCase):
    def test_d6_single_linear_single_angular_joint_names_are_unique(self):
        """D6 joints with one slide and one hinge axis emit unique MuJoCo joint names."""
        builder = newton.ModelBuilder()
        link = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_d6(
            parent=-1,
            child=link,
            linear_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)],
            angular_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z)],
        )
        builder.add_articulation([joint])
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        mujoco = SolverMuJoCo._mujoco
        joint_names = [
            mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(solver.mj_model.njnt)
        ]

        self.assertEqual(joint_names, ["joint_1_lin", "joint_1_ang"])

    def test_body_actuator_with_duplicated_body_names(self):
        """Test that duplicated body names resolve correctly for BODY actuators."""
        mjcf = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="duplication_test">
   <worldbody>
     <body name="link">
       <geom type="sphere" size="0.1"/>
       <joint type="free"/>
     </body>
   </worldbody>
   <actuator>
     <general body="link" gainprm="1 0 0 0 0 0 0 0 0 0"/>
   </actuator>
 </mujoco>
"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        trnid = solver.mjw_model.actuator_trnid.numpy()
        np.testing.assert_equal(trnid[0][0], 1)
        np.testing.assert_equal(trnid[1][0], 2)

    def test_equality_connect_constraint_with_duplicated_body_names(self):
        """Test that duplicated body names resolve correctly for equality connect constraints."""
        mjcf = """<?xml version="1.0" encoding="utf-8"?>
  <mujoco model="equality_connect_constraint">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
  <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>
      </body>
    </body>
  </worldbody>

  <!-- Equality constraint tying the two bodies together -->
  <equality>
  <!-- type="connect" constrains body positions; here link1 follows link2 -->
    <connect name="body_couple" anchor="0 0 0"  body1="link1" body2="link2"/>
  </equality>
</mujoco>
"""

        # Add the mjcf three times so that we have duplicates of joint1, link1, joint2, link2.
        # We want the equality constraint coupling link1 and link2 to work for all duplicates.
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, ignore_inertial_definitions=False)
        builder.add_mjcf(mjcf, ignore_inertial_definitions=False)
        builder.add_mjcf(mjcf, ignore_inertial_definitions=False)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        joint_q_start = model.joint_q_start.numpy()
        slide_joints = [
            i for i, label in enumerate(model.joint_label) if label.endswith("/joint1") or label.endswith("/joint2")
        ]

        # Set joint1 (all of them) to have a non-zero speed.
        start_joint_q = state_0.joint_q.numpy()
        for joint_idx in slide_joints:
            start_joint_q[joint_q_start[joint_idx]] = 1.0 if model.joint_label[joint_idx].endswith("/joint1") else 0.0
        state_0.joint_q.assign(start_joint_q)

        # If the de-duplication is not working properly we will end up with the equality constraint
        # applied only to the first link1/link2 and not to the duplicates.
        # We'd therefore expect a joint speed of [0.5, 0.5, 1.0, 0.0, 1.0, 0.0]
        # If the de-duplication is working properly the outcome will be that all duplicates of
        # joint1/joint2 will have the same speed and the expected outcome will be
        # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        expected_joint_q = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        # Run the sim and test the joint speed outcome against the expected outcome.
        for _i in range(100):
            solver.step(state_0, state_1, control, contacts, 0.02)
            state_0, state_1 = state_1, state_0
        measured_joint_q = state_0.joint_q.numpy()
        for i, joint_idx in enumerate(slide_joints):
            measured = measured_joint_q[joint_q_start[joint_idx]]
            expected = expected_joint_q[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=3,
                msg=f"Expected joint_q value: {expected}, Measured value: {measured}",
            )

    def test_equality_weld_constraint_with_duplicated_body_names(self):
        """Test that duplicated body names resolve correctly for equality weld constraints."""
        mjcf = """<?xml version="1.0" encoding="utf-8"?>
    <mujoco model="equality_weld_constraint">
    <option timestep="0.002" gravity="0 0 0"/>

    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
     <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>

     <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>
      </body>
    </body>
  </worldbody>

    <!-- Equality constraint tying the two bodies together -->
  <equality>
    <!-- type="weld" constrains body positions; here link1 follows link2 -->
    <weld name="body_couple" anchor="0 0 0" torquescale="1.0" body1="link1" body2="link2"/>
  </equality>
</mujoco>
"""

        # Add the mjcf three times so that we have duplicates of joint1, link1, joint2, link2.
        # We want the equality weld constraint coupling link1 and link2 to work for all duplicates.
        builder = newton.ModelBuilder()
        for _i in range(0, 3):
            builder.add_mjcf(mjcf, ignore_inertial_definitions=False)
        model = builder.finalize()
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        joint_q_start = model.joint_q_start.numpy()
        slide_joints = [
            i for i, label in enumerate(model.joint_label) if label.endswith("/joint1") or label.endswith("/joint2")
        ]

        # Set joint1 (all of them) to have a non-zero speed.
        start_joint_q = state_0.joint_q.numpy()
        for joint_idx in slide_joints:
            start_joint_q[joint_q_start[joint_idx]] = 1.0 if model.joint_label[joint_idx].endswith("/joint1") else 0.0
        state_0.joint_q.assign(start_joint_q)

        # If the de-duplication is not working properly we will end up with the equality constraint
        # applied only to the first link1/link2 and not to the duplicates.
        # We'd therefore expect a joint speed of [0.5, 0.5, 1.0, 0.0, 1.0, 0.0]
        # If the de-duplication is working properly the outcome will be that all duplicates of
        # joint1/joint2 will have the same speed and the expected outcome will be
        # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        expected_joint_q = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        # Run the sim and test the joint speed outcome against the expected outcome.
        for _i in range(100):
            solver.step(state_0, state_1, control, contacts, 0.02)
            state_0, state_1 = state_1, state_0
        measured_joint_q = state_0.joint_q.numpy()
        for i, joint_idx in enumerate(slide_joints):
            measured = measured_joint_q[joint_q_start[joint_idx]]
            expected = expected_joint_q[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=3,
                msg=f"Expected joint_q value: {expected}, Measured value: {measured}",
            )

    def test_joint_drive_with_duplicated_body_names(self):
        """Test that duplicated body names resolve correctly for joint drive."""
        mjcf = """<?xml version="1.0" encoding="utf-8"?>
    <mujoco model="single_slide_example">
    <option timestep="0.001" gravity="0 0 0"/>

    <!-- Default visual and joint settings (optional) -->
    <default>
        <joint limited="true" range="-2 2" damping="1"/>
    </default>

    <worldbody>
        <!-- Root is fixed to the world: no joint on this body -->
        <body name="root" pos="0 0 0">
         <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>

        <!-- Child body connected by a slide joint along x -->
        <body name="child" pos="0 0 0">
            <joint name="slide_joint" type="slide" axis="1 0 0"/>
             <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>
        </body>
        </body>
    </worldbody>

    <actuator>
        <!-- Position actuator driving slide_joint toward q = 1.0 -->
        <position name="slide_drive"
                joint="slide_joint"
                kp="10"
                kv="2"
                ctrlrange="-0.2 0.2"
                gear="1"/>
    </actuator>
    </mujoco>
    """

        builder = newton.ModelBuilder()
        for _i in range(0, 3):
            builder.add_mjcf(mjcf, ignore_inertial_definitions=False)
        model = builder.finalize()
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        # Set joint1 (all of them) to start at 0.
        start_joint_q = [0.0, 0.0, 0.0]
        state_0.joint_q.assign(start_joint_q)

        # Set the drive targets.
        joint_target_pos = [0.2, 0.2, 0.2]
        control.joint_target_q.assign(joint_target_pos)
        expected_joint_q = [0.2, 0.2, 0.2]

        # Run the sim and test the joint speed outcome against the expected outcome.
        for _i in range(300):
            solver.step(state_0, state_1, control, contacts, 0.02)
            state_0, state_1 = state_1, state_0
        measured_joint_q = state_0.joint_q.numpy()
        for i in range(0, 3):
            measured = measured_joint_q[i]
            expected = expected_joint_q[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=3,
                msg=f"Expected joint_q value: {expected}, Measured value: {measured}",
            )


class TestMuJoCoLinesearchBlockDim(unittest.TestCase):
    def test_linesearch_block_dim_is_left_to_mujoco_warp(self):
        """Leave MuJoCo Warp's line-search block dimension untouched.

        Newton forced ``linesearch_iterative = 32`` after ``put_model()`` to work
        around a CUDA registers-per-block failure that MuJoCo Warp has since capped
        upstream. Upstream only assigns the field when ``nv > 500``, so a model over
        that threshold is the only one whose value differs from the default and
        therefore the only one that can detect the override coming back.
        """
        try:
            _, mujoco_warp = SolverMuJoCo.import_mujoco()
        except ImportError:
            self.skipTest("MuJoCo Warp not installed")

        builder = newton.ModelBuilder()
        builder.add_mjcf(_large_nv_mjcf("pyramidal"))
        solver = SolverMuJoCo(builder.finalize())

        self.assertGreater(solver.mj_model.nv, 500)
        # Compare against upstream's own choice rather than a hard-coded 256, so the
        # test keeps its meaning if MuJoCo Warp retunes the value.
        expected = mujoco_warp.put_model(solver.mj_model).block_dim.linesearch_iterative
        self.assertEqual(solver.mjw_model.block_dim.linesearch_iterative, expected)

    def test_large_elliptic_model_steps_on_cuda(self):
        """Step a large elliptic-cone model on CUDA without a kernel-launch failure.

        The sizing assertion above cannot catch a runtime fault: the block dimension
        only matters once the kernel launches. The original failure was CUDA error
        701 in the elliptic line-search kernel, so this runs that path on a model
        over the threshold. It exercises one architecture, not every architecture
        that originally hit the limit.
        """
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError:
            self.skipTest("MuJoCo Warp not installed")
        if not wp.get_device().is_cuda:
            self.skipTest("block dimensions only constrain CUDA kernel launches")

        builder = newton.ModelBuilder()
        builder.add_mjcf(_large_nv_mjcf("elliptic"))
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        self.assertGreater(solver.mj_model.nv, 500)

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        for _ in range(5):
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        # Launch failures surface asynchronously, so force them to be raised here.
        wp.synchronize()
