# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for elastic Coulomb default friction and legacy compatibility."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from projects.digital_shoe.friction_adapter import FrictionAdapter
from projects.digital_shoe.friction_parameter_adapter import FrictionParameterAdapter
from projects.digital_shoe.runtime import FoundationConfig, MidsoleFoundation, ShoeMaterial, SurroundConfig


class TestDigitalShoeFrictionDefault(unittest.TestCase):
    """Verify elastic Coulomb defaults, legacy compatibility, and adapter lifecycle."""

    @classmethod
    def setUpClass(cls):
        """Initialize Warp once for all test methods."""
        wp.init()

    def _build_foundation(self, device, friction_model="maxwell", stiffness=1000.0, mu=0.8, tau=None):
        """Construct a MidsoleFoundation under controlled test settings."""
        columns = 4
        anchors = np.zeros((columns, 3))
        anchors[:, 0] = np.arange(columns) * 0.001
        anchors[:, 2] = -0.02
        rest = np.full(columns, 0.02)
        neighbors = np.full((columns, 4), -1, dtype=np.int32)
        material = ShoeMaterial(
            instantaneous_shear_modulus_pa=74000.0,
            hyperfoam_exponent=0.22,
            equilibrium_fraction=0.7,
            pasternak_n_per_m=1500.0,
            maxwell_relaxation_time_s=0.05,
        )
        driven = np.ones(columns, dtype=bool)

        config = FoundationConfig(
            ground_height_m=0.0,
            friction_stiffness=stiffness,
            friction=10.0,
            mu=mu,
            friction_model=friction_model,
            friction_relaxation_time_s=tau,
        )
        surround = SurroundConfig(driven=driven, sweeps=1, carrier_bond=True)

        return MidsoleFoundation(
            anchors,
            np.zeros(columns),
            rest,
            np.full(columns, 1.0e-5),
            neighbors,
            0.005,
            material,
            np.arange(1),
            wp.zeros(1, dtype=wp.vec3, device=device),
            config,
            device,
            surround,
            world_count=1,
        )

    def _build_state(self, device):
        """Build carrier state penetrating ground with positive tangential velocity."""
        return SimpleNamespace(
            body_q=wp.array(
                [wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity())], dtype=wp.transform, device=device
            ),
            body_qd=wp.array([wp.spatial_vector(0.2, 0.0, 0.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector, device=device),
            body_f=wp.zeros(1, dtype=wp.spatial_vector, device=device),
        )

    def test_default_config_is_elastic_coulomb(self):
        """Verify default FoundationConfig selects area-scaled elastic Coulomb friction."""
        cfg = FoundationConfig()
        self.assertEqual(cfg.friction_model, "elastic_coulomb")
        self.assertIsNone(cfg.friction_relaxation_time_s)

    def test_default_foundation_auto_installs_maxwell_adapter(self):
        """Verify MidsoleFoundation auto-installs default Maxwell parameter adapter and exposes persistent arrays."""
        device = wp.get_device("cpu")
        foundation = self._build_foundation(device, friction_model="maxwell")

        self.assertIsNotNone(foundation.friction_solver)
        self.assertIsInstance(foundation.friction_solver, FrictionParameterAdapter)
        self.assertTrue(foundation.friction_solver.is_default)

        # Check FoundationParams values
        params = foundation.world_params.numpy()[0]
        self.assertEqual(int(params["friction_model"]), 1)
        self.assertAlmostEqual(float(params["friction_relaxation_time_s"]), 0.05, places=5)

        # Check persistent array exposure
        self.assertTrue(hasattr(foundation, "tangent_deflection"))
        self.assertTrue(hasattr(foundation, "tangent_maxwell_force"))
        self.assertIs(foundation.tangent_deflection, foundation.friction_solver.deflection)
        self.assertIs(foundation.tangent_maxwell_force, foundation.friction_solver.maxwell_force)

    def test_legacy_mode_leaves_friction_solver_none(self):
        """Verify legacy FoundationConfig leaves friction_solver None and provides zero aliases."""
        device = wp.get_device("cpu")
        foundation = self._build_foundation(device, friction_model="legacy")

        self.assertIsNone(foundation.friction_solver)
        params = foundation.world_params.numpy()[0]
        self.assertEqual(int(params["friction_model"]), 0)

        # Persistent zero aliases must exist even in legacy mode
        self.assertTrue(hasattr(foundation, "tangent_deflection"))
        self.assertTrue(hasattr(foundation, "tangent_maxwell_force"))
        np.testing.assert_array_equal(foundation.tangent_deflection.numpy(), np.zeros((4, 2)))
        np.testing.assert_array_equal(foundation.tangent_maxwell_force.numpy(), np.zeros((4, 2)))

    def test_normal_mechanics_unchanged_between_default_and_legacy(self):
        """Verify normal mechanics, compression, and vertical forces are identical between Maxwell and legacy."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            with self.subTest(device=str(device)):
                maxwell_fd = self._build_foundation(device, friction_model="maxwell", stiffness=1000.0)
                legacy_fd = self._build_foundation(device, friction_model="legacy", stiffness=1000.0)

                state_maxwell = self._build_state(device)
                state_legacy = self._build_state(device)

                dt = 0.001
                maxwell_fd.apply(state_maxwell, dt, clear_body_force=True)
                legacy_fd.apply(state_legacy, dt, clear_body_force=True)

                # Compression and normal force must be bit-level close
                np.testing.assert_allclose(
                    maxwell_fd.compression.numpy(), legacy_fd.compression.numpy(), rtol=1e-6, atol=1e-6
                )
                np.testing.assert_allclose(
                    maxwell_fd.normal_force.numpy(), legacy_fd.normal_force.numpy(), rtol=1e-6, atol=1e-6
                )
                self.assertAlmostEqual(state_maxwell.body_f.numpy()[0, 2], state_legacy.body_f.numpy()[0, 2], places=5)

                # Tangential response must differ (Maxwell vs legacy distinction)
                fx_maxwell = state_maxwell.body_f.numpy()[0, 0]
                fx_legacy = state_legacy.body_f.numpy()[0, 0]
                self.assertNotAlmostEqual(fx_maxwell, fx_legacy, places=3)

                # Maxwell branch force and deflection must be active on default
                self.assertGreater(np.linalg.norm(maxwell_fd.tangent_deflection.numpy()), 1e-6)
                self.assertGreater(np.linalg.norm(maxwell_fd.tangent_maxwell_force.numpy()), 1e-6)

    def test_explicit_adapter_replaces_auto_default(self):
        """Verify explicit FrictionAdapter replaces auto-default adapter and rejects second attachment."""
        device = wp.get_device("cpu")
        foundation = self._build_foundation(device, friction_model="maxwell")
        self.assertTrue(foundation.friction_solver.is_default)

        mobility = wp.array([np.eye(6, dtype=np.float32)], dtype=wp.spatial_matrix, device=device)
        explicit_adapter = FrictionAdapter(foundation, mobility, mode="deflection")

        self.assertIs(foundation.friction_solver, explicit_adapter)
        self.assertFalse(explicit_adapter.is_default)

        # Attempting second explicit attach must raise ValueError
        with self.assertRaises(ValueError):
            FrictionAdapter(foundation, mobility, mode="deflection")

        with self.assertRaises(ValueError):
            FrictionParameterAdapter(foundation, world_count=1)

    def test_explicit_parameter_adapter_replaces_auto_default(self):
        """Verify explicit FrictionParameterAdapter replaces auto-default adapter and rejects second attachment."""
        device = wp.get_device("cpu")
        foundation = self._build_foundation(device, friction_model="maxwell")
        self.assertTrue(foundation.friction_solver.is_default)

        explicit_adapter = FrictionParameterAdapter(foundation, world_count=1)
        self.assertIs(foundation.friction_solver, explicit_adapter)
        self.assertFalse(explicit_adapter.is_default)

        with self.assertRaises(ValueError):
            FrictionParameterAdapter(foundation, world_count=1)

    def test_detach_restores_default_maxwell_adapter(self):
        """Verify detach restores default Maxwell adapter when configured as Maxwell."""
        device = wp.get_device("cpu")
        foundation = self._build_foundation(device, friction_model="maxwell")

        explicit_adapter = FrictionParameterAdapter(foundation, world_count=1)
        self.assertFalse(foundation.friction_solver.is_default)

        # Detach with default restore_default=True
        explicit_adapter.detach()
        self.assertIsNotNone(foundation.friction_solver)
        self.assertTrue(foundation.friction_solver.is_default)

        # Detach with restore_default=False
        foundation.friction_solver.detach(restore_default=False)
        self.assertIsNone(foundation.friction_solver)

    def test_legacy_detach_does_not_install_maxwell(self):
        """Verify detach on legacy-configured foundation leaves friction_solver None."""
        device = wp.get_device("cpu")
        foundation = self._build_foundation(device, friction_model="legacy")
        self.assertIsNone(foundation.friction_solver)

        explicit_adapter = FrictionParameterAdapter(foundation, world_count=1)
        self.assertIs(foundation.friction_solver, explicit_adapter)

        explicit_adapter.detach()
        self.assertIsNone(foundation.friction_solver)

    def test_reset_clears_deflection_and_maxwell_force(self):
        """Verify foundation reset clears tangent deflection, maxwell force, and bristle states."""
        device = wp.get_device("cpu")
        foundation = self._build_foundation(device, friction_model="maxwell")
        state = self._build_state(device)

        foundation.apply(state, 0.001, clear_body_force=True)
        self.assertGreater(np.linalg.norm(foundation.tangent_deflection.numpy()), 1e-6)
        self.assertGreater(np.linalg.norm(foundation.tangent_maxwell_force.numpy()), 1e-6)

        foundation.reset()
        np.testing.assert_array_equal(foundation.tangent_deflection.numpy(), np.zeros((4, 2)))
        np.testing.assert_array_equal(foundation.tangent_maxwell_force.numpy(), np.zeros((4, 2)))
        np.testing.assert_array_equal(foundation.tangent_stuck.numpy(), np.zeros(4))

    def test_material_tau_update_propagates_to_params_and_adapter(self):
        """Verify set_world_material updates FoundationParams and default adapter tau."""
        device = wp.get_device("cpu")
        foundation = self._build_foundation(device, friction_model="maxwell")

        new_mat = ShoeMaterial(
            instantaneous_shear_modulus_pa=80000.0,
            hyperfoam_exponent=0.25,
            equilibrium_fraction=0.75,
            pasternak_n_per_m=1600.0,
            maxwell_relaxation_time_s=0.012,
        )
        foundation.set_world_material(0, new_mat)

        params = foundation.world_params.numpy()[0]
        self.assertAlmostEqual(float(params["friction_relaxation_time_s"]), 0.012, places=5)
        adapter_settings = foundation.friction_solver.settings.numpy()[0]
        self.assertAlmostEqual(float(adapter_settings[11]), 0.012, places=5)

    def test_zero_friction_bench_remains_strictly_zero(self):
        """Verify zero-friction bench setup produces strictly zero tangential ground reaction."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            with self.subTest(device=str(device)):
                foundation = self._build_foundation(device, friction_model="maxwell", stiffness=0.0, mu=0.0)
                self.assertIsNone(foundation.friction_solver)
                state = self._build_state(device)
                foundation.apply(state, 0.001, clear_body_force=True)

                forces = state.body_f.numpy()[0]
                self.assertEqual(forces[0], 0.0)
                self.assertEqual(forces[1], 0.0)
                self.assertGreater(forces[2], 0.0)
                np.testing.assert_array_equal(foundation.ground_force.numpy()[:, :2], np.zeros((4, 2)))

    def test_failed_validation_leaves_default_adapter_intact(self):
        """Verify failed adapter validation does not detach or corrupt the active default adapter."""
        device = wp.get_device("cpu")
        foundation = self._build_foundation(device, friction_model="maxwell")
        original_solver = foundation.friction_solver
        self.assertTrue(original_solver.is_default)

        # 1. Invalid world count on FrictionParameterAdapter
        with self.assertRaises(ValueError):
            FrictionParameterAdapter(foundation, world_count=foundation.world_count + 1)
        self.assertIs(foundation.friction_solver, original_solver)
        self.assertTrue(foundation.friction_solver.is_default)

        # 2. Invalid parameter settings on FrictionParameterAdapter
        with self.assertRaises(ValueError):
            FrictionParameterAdapter(
                foundation,
                world_count=foundation.world_count,
                initial_parameters=[[-1.0, 0.8, 1.0, 1.0, 0.0, 0.0005, 0.0]],
            )
        self.assertIs(foundation.friction_solver, original_solver)
        self.assertTrue(foundation.friction_solver.is_default)

        # 3. Invalid mobility / smoothing_speed on FrictionAdapter
        mobility = wp.array([np.eye(6, dtype=np.float32)], dtype=wp.spatial_matrix, device=device)
        with self.assertRaises(ValueError):
            FrictionAdapter(foundation, mobility, smoothing_speed=-1.0)
        self.assertIs(foundation.friction_solver, original_solver)
        self.assertTrue(foundation.friction_solver.is_default)


if __name__ == "__main__":
    unittest.main()
