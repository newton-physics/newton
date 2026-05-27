# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton.examples.contacts.example_hydro_shoe import (
    CONTACT_LAW_CALIBRATED_OGDEN,
    CONTACT_LAW_KIM_HYPERFOAM,
    CONTACT_LAW_KIM_LAYERED,
    KIM_HYPERFOAM_MATERIALS,
    accumulate_plate_bending_kernel,
    apply_shoe_body_force_kernel,
    evaluate_pressure_maps_kernel,
    finalize_plate_bending_kernel,
    integrate_shoe_foundation_kernel,
    set_kinematic_foot_state_kernel,
    update_shoe_stats_kernel,
    _kim_layered_pressure_pa,
    _kim_pressure_pa,
    _rotate_vec_by_quat,
)


class TestHydroShoeMaterial(unittest.TestCase):
    def test_kim_hyperfoam_presets_match_paper_table(self):
        peba = KIM_HYPERFOAM_MATERIALS["peba"]

        self.assertEqual(peba.name, "PEBA")
        self.assertAlmostEqual(peba.density_kg_m3, 90.0)
        self.assertAlmostEqual(peba.mu1_pa, 0.112e6)
        self.assertAlmostEqual(peba.alpha1, 5.05)
        self.assertAlmostEqual(peba.poisson_ratio, 0.30)

    def test_kim_pressure_is_zero_at_zero_and_monotonic(self):
        material = KIM_HYPERFOAM_MATERIALS["eva"]
        strain = np.array([0.0, 0.05, 0.10, 0.20], dtype=np.float32)

        pressure = _kim_pressure_pa(strain, material)

        self.assertAlmostEqual(float(pressure[0]), 0.0)
        self.assertTrue(np.all(np.diff(pressure) > 0.0))

    def test_layered_kim_pressure_matches_single_material_limit(self):
        material = KIM_HYPERFOAM_MATERIALS["peba"]
        strain = np.array([0.0, 0.05, 0.10, 0.20], dtype=np.float32)

        layered = _kim_layered_pressure_pa(strain, material, material, 0.4)
        single = _kim_pressure_pa(strain, material)

        np.testing.assert_allclose(layered, single, rtol=1.0e-5, atol=1.0e-7)

    def test_layered_kim_pressure_changes_with_lower_foam(self):
        strain = np.array([0.15], dtype=np.float32)
        peba = KIM_HYPERFOAM_MATERIALS["peba"]
        eva = KIM_HYPERFOAM_MATERIALS["eva"]
        tpu = KIM_HYPERFOAM_MATERIALS["tpu"]

        eva_stack = _kim_layered_pressure_pa(strain, peba, eva, 0.5)
        tpu_stack = _kim_layered_pressure_pa(strain, peba, tpu, 0.5)

        self.assertNotAlmostEqual(float(eva_stack[0]), float(tpu_stack[0]))

    def test_foundation_kernel_integrates_kim_pressure_wrench(self):
        material = KIM_HYPERFOAM_MATERIALS["peba"]
        device = "cpu"
        grid_xy = wp.array([(0.0, 0.0), (0.01, 0.0)], dtype=wp.vec2, device=device)
        top_m = wp.array([0.0, 0.0], dtype=float, device=device)
        foot_sole_z_m = wp.array([0.0, 0.0], dtype=float, device=device)
        valid = wp.array([1, 1], dtype=wp.int32, device=device)
        slack = wp.array([0.02, 0.02], dtype=float, device=device)
        params = wp.array(
            [material.bulk_modulus_pa, material.alpha1, 0.99, 0.0, 1.0, float(CONTACT_LAW_KIM_HYPERFOAM)],
            dtype=float,
            device=device,
        )
        wrench = wp.zeros(6, dtype=float, device=device)

        wp.launch(
            integrate_shoe_foundation_kernel,
            dim=2,
            inputs=[
                grid_xy,
                top_m,
                foot_sole_z_m,
                valid,
                slack,
                2,
                -0.005,
                0.0,
                0.01,
                params,
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wrench,
            ],
            device=device,
        )

        strain = 0.005 / 0.02
        expected_force = material.bulk_modulus_pa * strain ** (material.alpha1 - 1.0) * 0.01 * 0.01 * 2.0
        result = wrench.numpy()
        self.assertAlmostEqual(result[2], expected_force, delta=max(abs(expected_force) * 1.0e-5, 1.0e-6))
        self.assertAlmostEqual(result[4], -0.01 * expected_force * 0.5, delta=1.0e-6)

    def test_foundation_kernel_integrates_layered_kim_pressure(self):
        upper = KIM_HYPERFOAM_MATERIALS["peba"]
        lower = KIM_HYPERFOAM_MATERIALS["eva"]
        device = "cpu"
        grid_xy = wp.array([(0.0, 0.0)], dtype=wp.vec2, device=device)
        top_m = wp.array([0.0], dtype=float, device=device)
        foot_sole_z_m = wp.array([0.0], dtype=float, device=device)
        valid = wp.array([1], dtype=wp.int32, device=device)
        slack = wp.array([0.02], dtype=float, device=device)
        params = wp.array(
            [
                upper.bulk_modulus_pa,
                upper.alpha1,
                0.99,
                0.0,
                1.0,
                float(CONTACT_LAW_KIM_LAYERED),
                lower.bulk_modulus_pa,
                lower.alpha1,
                0.35,
            ],
            dtype=float,
            device=device,
        )
        wrench = wp.zeros(6, dtype=float, device=device)

        wp.launch(
            integrate_shoe_foundation_kernel,
            dim=1,
            inputs=[
                grid_xy,
                top_m,
                foot_sole_z_m,
                valid,
                slack,
                1,
                -0.005,
                0.0,
                0.01,
                params,
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wrench,
            ],
            device=device,
        )

        strain = np.array([0.005 / 0.02], dtype=np.float32)
        expected_stress = _kim_layered_pressure_pa(strain, upper, lower, 0.35)[0]
        self.assertAlmostEqual(wrench.numpy()[2], expected_stress * 0.01 * 0.01, delta=1.0e-5)

    def test_pressure_map_kernel_matches_kim_pressure_law(self):
        material = KIM_HYPERFOAM_MATERIALS["peba"]
        device = "cpu"
        top_m = wp.array([0.0, 0.0], dtype=float, device=device)
        bottom_m = wp.array([-0.02, -0.02], dtype=float, device=device)
        foot_sole_z_m = wp.array([0.0, 0.0], dtype=float, device=device)
        valid = wp.array([1, 0], dtype=wp.int32, device=device)
        slack = wp.array([0.02, 0.02], dtype=float, device=device)
        params = wp.array(
            [material.bulk_modulus_pa, material.alpha1, 0.99, 0.0, 1.0, float(CONTACT_LAW_KIM_HYPERFOAM)],
            dtype=float,
            device=device,
        )
        foot_disp = wp.zeros(2, dtype=float, device=device)
        foot_pressure = wp.zeros(2, dtype=float, device=device)
        ground_disp = wp.zeros(2, dtype=float, device=device)
        ground_pressure = wp.zeros(2, dtype=float, device=device)
        peak_foot_disp = wp.zeros(2, dtype=float, device=device)
        peak_foot_pressure = wp.zeros(2, dtype=float, device=device)
        peak_ground_disp = wp.zeros(2, dtype=float, device=device)
        peak_ground_pressure = wp.zeros(2, dtype=float, device=device)

        wp.launch(
            evaluate_pressure_maps_kernel,
            dim=2,
            inputs=[
                top_m,
                bottom_m,
                foot_sole_z_m,
                valid,
                slack,
                2,
                -0.005,
                0.0,
                0.0,
                params,
                foot_disp,
                foot_pressure,
                ground_disp,
                ground_pressure,
                peak_foot_disp,
                peak_foot_pressure,
                peak_ground_disp,
                peak_ground_pressure,
                1,
            ],
            device=device,
        )

        expected_foot_pressure_kpa = _kim_pressure_pa(np.array([0.25], dtype=np.float32), material)[0] * 0.001
        expected_ground_pressure_kpa = _kim_pressure_pa(np.array([0.99], dtype=np.float32), material)[0] * 0.001
        np.testing.assert_allclose(foot_disp.numpy(), [0.005, 0.0], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(ground_disp.numpy(), [0.02, 0.02], rtol=0.0, atol=1.0e-7)
        self.assertAlmostEqual(foot_pressure.numpy()[0], expected_foot_pressure_kpa, delta=1.0e-6)
        self.assertAlmostEqual(foot_pressure.numpy()[1], 0.0, delta=1.0e-7)
        np.testing.assert_allclose(
            ground_pressure.numpy(),
            [expected_ground_pressure_kpa, expected_ground_pressure_kpa],
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(peak_foot_disp.numpy(), foot_disp.numpy(), rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(peak_foot_pressure.numpy(), foot_pressure.numpy(), rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(peak_ground_disp.numpy(), ground_disp.numpy(), rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(peak_ground_pressure.numpy(), ground_pressure.numpy(), rtol=0.0, atol=1.0e-7)

    def test_apply_shoe_body_force_kernel_writes_spatial_wrench(self):
        device = "cpu"
        body_f = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        wrench = wp.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=float, device=device)
        plate_torque = wp.array([0.01, 0.02, 0.03], dtype=float, device=device)

        wp.launch(
            apply_shoe_body_force_kernel,
            dim=1,
            inputs=[body_f, 1, wrench, plate_torque, -9.0],
            device=device,
        )

        np.testing.assert_allclose(body_f.numpy()[0], np.zeros(6), atol=1.0e-7)
        np.testing.assert_allclose(body_f.numpy()[1], [1.0, 2.0, -6.0, 0.11, 0.22, 0.33], atol=1.0e-7)

    def test_kinematic_foot_state_kernel_writes_pose_and_velocity(self):
        device = "cpu"
        body_q = wp.array(
            [
                wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity()),
                wp.transform_identity(),
            ],
            dtype=wp.transform,
            device=device,
        )
        body_qd = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        joint_q = wp.zeros(14, dtype=float, device=device)
        joint_qd = wp.zeros(12, dtype=float, device=device)

        wp.launch(
            set_kinematic_foot_state_kernel,
            dim=1,
            inputs=[body_q, body_qd, joint_q, joint_qd, 1, 7, 6, 0.123, -0.456],
            device=device,
        )

        np.testing.assert_allclose(body_q.numpy()[0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], atol=1.0e-7)
        np.testing.assert_allclose(body_q.numpy()[1], [0.0, 0.0, 0.123, 0.0, 0.0, 0.0, 1.0], atol=1.0e-7)
        np.testing.assert_allclose(body_qd.numpy()[1], [0.0, 0.0, -0.456, 0.0, 0.0, 0.0], atol=1.0e-7)
        np.testing.assert_allclose(joint_q.numpy()[7:14], [0.0, 0.0, 0.123, 0.0, 0.0, 0.0, 1.0], atol=1.0e-7)
        np.testing.assert_allclose(joint_qd.numpy()[6:12], [0.0, 0.0, -0.456, 0.0, 0.0, 0.0], atol=1.0e-7)

    def test_update_shoe_stats_kernel_tracks_peak_force(self):
        device = "cpu"
        wrench = wp.array([0.0, 0.0, 12.0, 0.0, 0.0, 0.0], dtype=float, device=device)
        plate_torque = wp.array([0.0, 1.5, 0.0], dtype=float, device=device)
        stats = wp.zeros(3, dtype=float, device=device)

        wp.launch(update_shoe_stats_kernel, dim=1, inputs=[wrench, plate_torque, stats], device=device)
        wrench.assign(wp.array([0.0, 0.0, 8.0, 0.0, 0.0, 0.0], dtype=float, device=device))
        plate_torque.assign(wp.array([0.0, -2.0, 0.0], dtype=float, device=device))
        wp.launch(update_shoe_stats_kernel, dim=1, inputs=[wrench, plate_torque, stats], device=device)

        np.testing.assert_allclose(stats.numpy(), [8.0, 12.0, -2.0], atol=1.0e-7)

    def test_foundation_kernel_preserves_calibrated_ogden_law(self):
        device = "cpu"
        grid_xy = wp.array([(0.0, 0.0)], dtype=wp.vec2, device=device)
        top_m = wp.array([0.0], dtype=float, device=device)
        foot_sole_z_m = wp.array([0.0], dtype=float, device=device)
        valid = wp.array([1], dtype=wp.int32, device=device)
        slack = wp.array([0.02], dtype=float, device=device)
        params = wp.array(
            [1.0e6, 2.0, 0.65, 0.0, 1.0, float(CONTACT_LAW_CALIBRATED_OGDEN)],
            dtype=float,
            device=device,
        )
        wrench = wp.zeros(6, dtype=float, device=device)

        wp.launch(
            integrate_shoe_foundation_kernel,
            dim=1,
            inputs=[
                grid_xy,
                top_m,
                foot_sole_z_m,
                valid,
                slack,
                1,
                -0.005,
                0.0,
                0.01,
                params,
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(0.0, 0.0, 0.0),
                wrench,
            ],
            device=device,
        )

        strain = 0.005 / 0.02
        normalized = min(strain / 0.65, 0.999)
        expected_stress = 1.0e6 * ((1.0 - normalized) ** -2.0 - 1.0) / 2.0
        self.assertAlmostEqual(wrench.numpy()[2], expected_stress * 0.01 * 0.01, delta=1.0e-4)

    def test_quaternion_rotation_helper_for_com_offset(self):
        angle = np.pi / 2.0
        q = np.array([0.0, 0.0, np.sin(angle * 0.5), np.cos(angle * 0.5)])

        rotated = _rotate_vec_by_quat(np.array([1.0, 0.0, 0.0]), q)

        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1.0e-7)

    def test_plate_bending_kernels_compute_pitch_torque(self):
        device = "cpu"
        grid_xy = wp.array([(0.0, 0.0), (0.0, 1.0)], dtype=wp.vec2, device=device)
        top_m = wp.array([0.0, 0.0], dtype=float, device=device)
        foot_sole_z_m = wp.array([0.0, -0.01], dtype=float, device=device)
        valid = wp.array([1, 1], dtype=wp.int32, device=device)
        slack = wp.array([0.05, 0.05], dtype=float, device=device)
        accum = wp.zeros(6, dtype=float, device=device)
        torque = wp.zeros(3, dtype=float, device=device)
        plate_params = wp.array([33.0e9, 0.0015, 0.4, 0.07, 0.15, 0.05], dtype=float, device=device)

        wp.launch(
            accumulate_plate_bending_kernel,
            dim=2,
            inputs=[grid_xy, top_m, foot_sole_z_m, valid, slack, 2, 0.0, 0.0, 0.35, 0.65, accum],
            device=device,
        )
        wp.launch(finalize_plate_bending_kernel, dim=1, inputs=[accum, 0.0, plate_params, torque], device=device)

        plate_d = 33.0e9 * 0.0015**3 / (12.0 * (1.0 - 0.4**2))
        pitch_stiffness = plate_d * 0.07 / 0.15
        expected = -pitch_stiffness * np.arctan2(0.01, 1.0)
        self.assertAlmostEqual(torque.numpy()[1], expected, delta=abs(expected) * 1.0e-5)


if __name__ == "__main__":
    unittest.main()
