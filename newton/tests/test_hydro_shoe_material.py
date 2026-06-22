# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest

import numpy as np
import warp as wp

from newton.examples.contacts.example_hydro_shoe import (
    CONTACT_LAW_CALIBRATED_OGDEN,
    CONTACT_LAW_KIM_HYPERFOAM,
    CONTACT_LAW_KIM_LAYERED,
    KIM_HYPERFOAM_MATERIALS,
    SURFACE_GROUND_AREA,
    SURFACE_GROUND_DISP_AREA,
    SURFACE_TOP_AREA,
    SURFACE_TOP_DISP_AREA,
    _kim_layered_pressure_pa,
    _kim_pressure_pa,
    _rotate_vec_by_quat,
    apply_bonded_top_forces_kernel,
    evaluate_bottom_hydroelastic_ogden_kernel,
    evaluate_contact_stress,
    load_calibrated_foundation_material,
    preferred_calibrated_hydro_shoe_stroke_m,
    set_kinematic_foot_state_kernel,
    update_stack_material_state_kernel,
    update_shoe_stats_kernel,
)


@wp.kernel
def integrate_shoe_foundation_kernel(
    grid_xy: wp.array[wp.vec2],
    top_m: wp.array[float],
    foot_sole_z_m: wp.array[float],
    foot_contact_valid: wp.array[wp.int32],
    slack_length_m: wp.array[float],
    spring_count: int,
    body_z: float,
    start_z: float,
    spacing_m: float,
    params: wp.array[float],
    foot_vel: wp.vec3,
    foot_omega: wp.vec3,
    foot_com: wp.vec3,
    wrench_out: wp.array[float],
):
    spring = wp.tid()
    if spring >= spring_count:
        return
    if foot_contact_valid[spring] == 0:
        return

    slack = wp.max(slack_length_m[spring], 1.0e-6)
    top_rest_world = start_z + top_m[spring]
    foot_sole_world = body_z + foot_sole_z_m[spring]
    displacement = wp.clamp(top_rest_world - foot_sole_world, 0.0, slack)
    if displacement <= 0.0:
        return

    strain = wp.clamp(displacement / slack, 0.0, 0.99)

    xy = grid_xy[spring]
    point = wp.vec3(xy[0], xy[1], top_rest_world)
    normal = wp.vec3(0.0, 0.0, 1.0)
    r = point - foot_com
    v_contact = foot_vel + wp.cross(foot_omega, r)
    comp_vel = -wp.dot(v_contact, normal)

    stress = evaluate_contact_stress(strain, comp_vel, params)
    area = spacing_m * spacing_m
    force_vec = normal * (stress * area)
    torque = wp.cross(r, force_vec)

    wp.atomic_add(wrench_out, 0, force_vec[0])
    wp.atomic_add(wrench_out, 1, force_vec[1])
    wp.atomic_add(wrench_out, 2, force_vec[2])
    wp.atomic_add(wrench_out, 3, torque[0])
    wp.atomic_add(wrench_out, 4, torque[1])
    wp.atomic_add(wrench_out, 5, torque[2])


class TestHydroShoeMaterial(unittest.TestCase):
    def test_kim_hyperfoam_presets_match_paper_table(self):
        peba = KIM_HYPERFOAM_MATERIALS["peba"]

        self.assertEqual(peba.name, "PEBA")
        self.assertTrue(peba.density_kg_m3 in (90.0, 125.0))
        self.assertTrue(peba.mu1_pa in (0.085e6, 0.412e6, 0.512e6))
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
            [material.bulk_modulus_pa, material.alpha1, 0.99, 0.0, float(CONTACT_LAW_KIM_HYPERFOAM)],
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

    def test_bottom_hydroelastic_kernel_uses_two_sided_stack_pressure(self):
        device = "cpu"
        points = wp.array(
            [
                wp.vec3(-0.005, -0.005, 0.0),
                wp.vec3(0.005, -0.005, 0.0),
                wp.vec3(0.0, 0.005, 0.0),
            ],
            dtype=wp.vec3,
            device=device,
        )
        depths = wp.array([-0.0025], dtype=wp.float32, device=device)
        shape_pairs = wp.array([wp.vec2i(1, 2)], dtype=wp.vec2i, device=device)
        face_count_ptr = wp.array([1], dtype=wp.int32, device=device)

        midsole_shape_idx = 1
        ground_shape_idx = 2
        midsole_body_idx = 1

        body_com = wp.zeros(2, dtype=wp.vec3, device=device)
        body_q = wp.array(
            [
                wp.transform(wp.vec3(0.0, 0.0, -0.005), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            dtype=wp.transform,
            device=device,
        )
        body_qd = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        spring_xy = wp.array([(0.0, 0.0)], dtype=wp.vec2, device=device)
        spring_slack = wp.array([0.02], dtype=float, device=device)
        num_springs = 1
        coupled_state = wp.zeros(6, dtype=float, device=device)
        expected_area = 0.00005
        top_displacement = 0.004
        bottom_displacement = 0.0025
        coupled_np = coupled_state.numpy()
        coupled_np[SURFACE_TOP_AREA] = expected_area
        coupled_np[SURFACE_TOP_DISP_AREA] = top_displacement * expected_area
        coupled_np[SURFACE_GROUND_AREA] = expected_area
        coupled_np[SURFACE_GROUND_DISP_AREA] = bottom_displacement * expected_area
        coupled_state.assign(coupled_np)

        material = KIM_HYPERFOAM_MATERIALS["peba"]
        params = wp.array(
            [
                material.bulk_modulus_pa,
                material.alpha1,
                0.99,
                0.0,
                float(CONTACT_LAW_KIM_HYPERFOAM),
                0.0,
                0.0,
                0.0,
                1.0,  # PARAM_HYDRO_FORCE_SCALE
            ],
            dtype=float,
            device=device,
        )
        body_f = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        wrench = wp.zeros(6, dtype=float, device=device)
        energy = wp.zeros(4, dtype=float, device=device)
        dissipated_total = wp.zeros(1, dtype=float, device=device)
        ground_disp = wp.zeros(1, dtype=float, device=device)
        ground_pressure = wp.zeros(1, dtype=float, device=device)
        stack_disp = wp.zeros(1, dtype=float, device=device)
        stack_pressure = wp.zeros(1, dtype=float, device=device)

        slack = 0.02
        stack_displacement = top_displacement + bottom_displacement
        strain = stack_displacement / slack
        expected_stress = material.bulk_modulus_pa * strain ** (material.alpha1 - 1.0)
        expected_force = expected_stress * expected_area * 1.0
        stack_stress = wp.array([expected_stress], dtype=float, device=device)
        stack_material_disp = wp.array([stack_displacement], dtype=float, device=device)
        energy_density = wp.array([0.5 * expected_stress * stack_displacement], dtype=float, device=device)
        dissipation_density = wp.zeros(1, dtype=float, device=device)

        # Define grid for the single spring
        grid_to_spring = wp.array([[0]], dtype=wp.int32, device=device)
        grid_min_u = 0.0
        grid_min_v = 0.0
        grid_spacing = 1.0
        grid_num_u = 1
        grid_num_v = 1

        wp.launch(
            evaluate_bottom_hydroelastic_ogden_kernel,
            dim=1,
            inputs=[
                points,
                depths,
                shape_pairs,
                face_count_ptr,
                midsole_shape_idx,
                ground_shape_idx,
                midsole_body_idx,
                body_com,
                body_q,
                body_qd,
                spring_xy,
                spring_slack,
                num_springs,
                coupled_state,
                params,
                stack_stress,
                stack_material_disp,
                energy_density,
                dissipation_density,
                body_f,
                wrench,
                energy,
                dissipated_total,
                0.001,
                ground_disp,
                ground_pressure,
                stack_disp,
                stack_pressure,
                # Grid parameters
                grid_to_spring,
                grid_min_u,
                grid_min_v,
                grid_spacing,
                grid_num_u,
                grid_num_v,
            ],
            device=device,
        )

        forces = body_f.numpy()
        self.assertAlmostEqual(forces[0][2], 0.0, delta=1.0e-12)
        self.assertAlmostEqual(forces[1][2], expected_force, delta=max(abs(expected_force) * 1.0e-5, 1.0e-6))
        self.assertAlmostEqual(wrench.numpy()[2], expected_force, delta=max(abs(expected_force) * 1.0e-5, 1.0e-6))
        self.assertAlmostEqual(energy.numpy()[0], 0.0, delta=1.0e-12)
        self.assertGreater(energy.numpy()[1], 0.0)
        self.assertAlmostEqual(dissipated_total.numpy()[0], 0.0, delta=1.0e-12)
        self.assertAlmostEqual(ground_disp.numpy()[0], bottom_displacement, delta=1.0e-7)
        self.assertAlmostEqual(stack_disp.numpy()[0], stack_displacement, delta=1.0e-7)

    def test_bonded_top_force_kernel_applies_continuous_top_wrench(self):
        device = "cpu"
        material = KIM_HYPERFOAM_MATERIALS["peba"]
        grid_xy = wp.array([(0.0, 0.0)], dtype=wp.vec2, device=device)
        top_m = wp.array([0.0], dtype=float, device=device)
        slack = wp.array([0.02], dtype=float, device=device)
        valid = wp.array([1], dtype=wp.int32, device=device)
        body_com = wp.zeros(2, dtype=wp.vec3, device=device)
        body_q = wp.array(
            [
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            dtype=wp.transform,
            device=device,
        )
        params = wp.array(
            [
                material.bulk_modulus_pa,
                material.alpha1,
                0.99,
                0.0,
                float(CONTACT_LAW_KIM_HYPERFOAM),
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=float,
            device=device,
        )
        area = 0.01 * 0.01
        coupled_state = wp.zeros(6, dtype=float, device=device)
        coupled_np = coupled_state.numpy()
        coupled_np[SURFACE_TOP_AREA] = area
        coupled_np[SURFACE_TOP_DISP_AREA] = 0.005 * area
        coupled_np[SURFACE_GROUND_AREA] = area
        coupled_np[SURFACE_GROUND_DISP_AREA] = 0.004 * area
        coupled_state.assign(coupled_np)
        body_f = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        wrench = wp.zeros(6, dtype=float, device=device)
        energy = wp.zeros(4, dtype=float, device=device)
        dissipated_total = wp.zeros(1, dtype=float, device=device)
        foot_pressure = wp.zeros(1, dtype=float, device=device)
        stack_disp = wp.zeros(1, dtype=float, device=device)
        stack_pressure = wp.zeros(1, dtype=float, device=device)

        strain = 0.009 / 0.02
        expected_stress = material.bulk_modulus_pa * strain ** (material.alpha1 - 1.0)
        expected_force = expected_stress * area
        stack_stress = wp.array([expected_stress], dtype=float, device=device)
        stack_material_disp = wp.array([0.009], dtype=float, device=device)
        energy_density = wp.array([0.5 * expected_stress * 0.009], dtype=float, device=device)
        dissipation_density = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            apply_bonded_top_forces_kernel,
            dim=1,
            inputs=[
                grid_xy,
                top_m,
                slack,
                valid,
                1,
                body_com,
                body_q,
                0,
                1,
                coupled_state,
                params,
                stack_stress,
                stack_material_disp,
                energy_density,
                dissipation_density,
                body_f,
                wrench,
                energy,
                dissipated_total,
                0.001,
                foot_pressure,
                stack_disp,
                stack_pressure,
            ],
            device=device,
        )

        forces = body_f.numpy()
        self.assertAlmostEqual(forces[0][2], expected_force, delta=max(abs(expected_force) * 1.0e-5, 1.0e-6))
        self.assertAlmostEqual(forces[1][2], -expected_force, delta=max(abs(expected_force) * 1.0e-5, 1.0e-6))
        self.assertAlmostEqual(wrench.numpy()[2], expected_force, delta=max(abs(expected_force) * 1.0e-5, 1.0e-6))
        self.assertAlmostEqual(stack_disp.numpy()[0], 0.009, delta=1.0e-7)
        self.assertAlmostEqual(stack_pressure.numpy()[0], expected_stress * 0.001, delta=2.0e-5)

    def test_calibrated_stack_material_state_uses_digital_instron_terms(self):
        device = "cpu"
        grid_xy = wp.array([(0.0, 0.0), (0.01, 0.0)], dtype=wp.vec2, device=device)
        slack = wp.array([0.02, 0.02], dtype=float, device=device)
        area = 0.01 * 0.01
        coupled_state = wp.zeros(12, dtype=float, device=device)
        coupled_np = coupled_state.numpy()
        coupled_np[SURFACE_TOP_AREA] = area
        coupled_np[SURFACE_TOP_DISP_AREA] = 0.006 * area
        coupled_np[SURFACE_GROUND_AREA] = area
        coupled_np[SURFACE_GROUND_DISP_AREA] = 0.004 * area
        coupled_np[6 + SURFACE_TOP_AREA] = area
        coupled_np[6 + SURFACE_TOP_DISP_AREA] = 0.002 * area
        coupled_np[6 + SURFACE_GROUND_AREA] = area
        coupled_np[6 + SURFACE_GROUND_DISP_AREA] = 0.005 * area
        coupled_state.assign(coupled_np)
        params = wp.array(
            [
                50000.0,  # stiffness
                0.5,  # alpha
                0.8,  # lock strain
                0.0,  # damping
                float(CONTACT_LAW_CALIBRATED_OGDEN),
                1.0,  # lower bulk
                0.5,  # lower alpha
                1.0,  # upper fraction
                1.0,  # hydro force scale
                1000.0,  # prony stiffness
                100.0,  # prony damping
            ],
            dtype=float,
            device=device,
        )
        prony_state = wp.zeros(2, dtype=float, device=device)
        stress_out = wp.zeros(2, dtype=float, device=device)
        displacement_out = wp.zeros(2, dtype=float, device=device)
        energy_density_out = wp.zeros(2, dtype=float, device=device)
        dissipation_density_out = wp.zeros(2, dtype=float, device=device)

        wp.launch(
            update_stack_material_state_kernel,
            dim=2,
            inputs=[
                grid_xy,
                slack,
                2,
                coupled_state,
                params,
                0.001,
                prony_state,
                stress_out,
                displacement_out,
                energy_density_out,
                dissipation_density_out,
            ],
            device=device,
        )

        displacement0 = 0.006 + 0.004
        displacement1 = 0.002 + 0.005
        strain0 = displacement0 / 0.02
        strain1 = displacement1 / 0.02
        ogden0 = 50000.0 * ((1.0 - strain0 / 0.8) ** -0.5 - 1.0) / 0.5
        ogden1 = 50000.0 * ((1.0 - strain1 / 0.8) ** -0.5 - 1.0) / 0.5
        beta = 1000.0 / 50000.0
        tau = 100.0 / 1000.0
        relax = 1.0 - np.exp(-0.001 / tau)
        expected0 = ogden0 * (1.0 - beta * relax)
        expected1 = max(ogden1 * (1.0 - beta * relax), 0.0)

        result = stress_out.numpy()
        np.testing.assert_allclose(result, [expected0, expected1], rtol=5.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            displacement_out.numpy(),
            [displacement0, displacement1],
            rtol=1.0e-6,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            energy_density_out.numpy(),
            [0.5 * expected0 * displacement0, 0.5 * expected1 * displacement1],
            rtol=5.0e-5,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(dissipation_density_out.numpy(), [0.0, 0.0], atol=1.0e-12)
        self.assertGreater(prony_state.numpy()[0], 0.0)

    def test_calibrated_material_loader_accepts_top_plus_bottom_artifact(self):
        artifact = {
            "schema_version": "digital_instron_v2_foundation_material_1",
            "contact_model": {
                "type": "two_sided_spring_grid",
                "compression_components": "top_plus_bottom",
            },
            "material": {
                "stiffness_pa": 141138.0,
                "ogden_alpha": 0.104,
                "lock_strain": 0.557,
                "damping_pa_s": 10115.5,
                "per_cylinder_area": True,
                "prony_stiffness_pa": 17339.8,
                "prony_damping_pa_s": 621.8,
                "state_warmup_cycles": 5,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/material.json"
            with open(path, "w") as f:
                json.dump(artifact, f)

            material = load_calibrated_foundation_material(path)

        self.assertAlmostEqual(material.stiffness_pa, 141138.0)
        self.assertTrue(material.per_cylinder_area)

    def test_calibrated_hydro_shoe_stroke_uses_two_sided_envelope(self):
        artifact = {
            "contact_model": {
                "compression_components": "top_plus_bottom",
                "trials": {
                    "rearfoot": {"fixture": "rearfoot_punch"},
                    "fullfoot": {"fixture": "fullfoot_last"},
                },
            },
            "calibration_envelope": {
                "trials": {
                    "rearfoot": {
                        "peak_top_compression_m": 0.012,
                        "peak_bottom_compression_m": 0.012,
                        "peak_max_compression_m": 0.024,
                    },
                    "fullfoot": {
                        "peak_top_compression_m": 0.008,
                        "peak_bottom_compression_m": 0.008,
                        "peak_max_compression_m": 0.016,
                    },
                }
            },
        }

        self.assertAlmostEqual(preferred_calibrated_hydro_shoe_stroke_m(artifact), 0.012)

    def test_calibrated_material_loader_rejects_explicit_top_only_artifact(self):
        artifact = {
            "schema_version": "digital_instron_v2_foundation_material_1",
            "contact_model": {
                "type": "one_sided_spring_grid",
                "compression_components": "top_only",
            },
            "material": {
                "stiffness_pa": 1.0,
                "ogden_alpha": 1.0,
                "lock_strain": 0.5,
                "damping_pa_s": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/material.json"
            with open(path, "w") as f:
                json.dump(artifact, f)

            with self.assertRaisesRegex(ValueError, "top_plus_bottom"):
                load_calibrated_foundation_material(path)

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
        energy = wp.array([0.25, 0.75], dtype=float, device=device)
        dissipated_total = wp.array([0.125], dtype=float, device=device)
        stats = wp.zeros(5, dtype=float, device=device)

        wp.launch(
            update_shoe_stats_kernel,
            dim=1,
            inputs=[wrench, energy, dissipated_total, stats],
            device=device,
        )
        wrench.assign(wp.array([0.0, 0.0, 8.0, 0.0, 0.0, 0.0], dtype=float, device=device))
        energy.assign(wp.array([0.10, 0.40], dtype=float, device=device))
        dissipated_total.assign(wp.array([0.25], dtype=float, device=device))
        wp.launch(
            update_shoe_stats_kernel,
            dim=1,
            inputs=[wrench, energy, dissipated_total, stats],
            device=device,
        )

        np.testing.assert_allclose(stats.numpy(), [8.0, 12.0, 0.5, 1.0, 0.25], atol=1.0e-7)

    def test_quaternion_rotation_helper_for_com_offset(self):
        angle = np.pi / 2.0
        q = np.array([0.0, 0.0, np.sin(angle * 0.5), np.cos(angle * 0.5)])

        rotated = _rotate_vec_by_quat(np.array([1.0, 0.0, 0.0]), q)

        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
