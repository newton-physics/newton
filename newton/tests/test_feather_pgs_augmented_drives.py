# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs.kernels import (
    eval_rigid_tau,
    eval_rigid_tau_add,
    eval_rigid_tau_and_augmented_drives,
    prepare_augmented_joint_drives,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _prepare_augmented_joint_drives_by_dof


class TestFeatherPGSAugmentedDrives(unittest.TestCase):
    def test_split_augmented_drive_pipeline_matches_combined_kernel(self):
        """Match combined drive forces and clear articulations without active drives."""
        device = wp.get_device()
        articulation_start = wp.array([0, 2], dtype=wp.int32, device=device)
        articulation_joint_end = wp.array([2], dtype=wp.int32, device=device)
        articulation_dof_count = wp.array([2], dtype=wp.int32, device=device)
        joint_type = wp.array([JointType.REVOLUTE, JointType.REVOLUTE], dtype=wp.int32, device=device)
        joint_parent = wp.array([-1, 0], dtype=wp.int32, device=device)
        joint_child = wp.array([0, 1], dtype=wp.int32, device=device)
        joint_articulation = wp.zeros(2, dtype=wp.int32, device=device)
        joint_start = wp.array([0, 1], dtype=wp.int32, device=device)
        joint_dof_dim = wp.array([[0, 1], [0, 1]], dtype=wp.int32, device=device)
        joint_f = wp.array([0.4, -0.2], dtype=wp.float32, device=device)
        joint_q = wp.array([0.25, -0.35], dtype=wp.float32, device=device)
        joint_qd = wp.array([0.1, -0.2], dtype=wp.float32, device=device)
        spring_stiffness = wp.array([1.5, 0.75], dtype=wp.float32, device=device)
        spring_ref = wp.array([0.0, 0.1], dtype=wp.float32, device=device)
        damping = wp.array([0.2, 0.3], dtype=wp.float32, device=device)
        joint_S_s = wp.array(
            [wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] * 2,
            dtype=wp.spatial_vector,
            device=device,
        )
        body_fb_s = wp.array(
            [
                wp.spatial_vector(0.1, 0.2, 0.3, 1.0, 1.5, -0.5),
                wp.spatial_vector(-0.2, 0.4, -0.1, -0.5, 0.25, 0.75),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_f_ext = wp.array(
            [
                wp.spatial_vector(0.05, 0.0, -0.1, 0.0, 0.25, 0.0),
                wp.spatial_vector(0.0, -0.05, 0.1, 0.15, 0.0, -0.2),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_flags = wp.zeros(2, dtype=wp.int32, device=device)
        body_q = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device)
        body_com = wp.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=wp.vec3, device=device)
        articulation_origin = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        target_ke = wp.array([10.0, 20.0], dtype=wp.float32, device=device)
        target_kd = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
        target_q = wp.array([0.1, -0.1], dtype=wp.float32, device=device)
        target_qd = wp.array([0.0, 0.05], dtype=wp.float32, device=device)
        effort_limit = wp.array([1.25, 2.5], dtype=wp.float32, device=device)

        sequential_body_ft = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        sequential_tau = wp.zeros(2, dtype=wp.float32, device=device)
        wp.launch(
            eval_rigid_tau,
            dim=1,
            inputs=[
                articulation_start,
                articulation_joint_end,
                joint_type,
                joint_parent,
                joint_child,
                joint_articulation,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                spring_stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_fb_s,
                body_f_ext,
                body_flags,
                body_q,
                body_com,
                articulation_origin,
            ],
            outputs=[sequential_body_ft, sequential_tau],
            device=device,
        )

        combined_body_ft = wp.zeros_like(sequential_body_ft)
        combined_tau = wp.zeros_like(sequential_tau)
        combined_counts = wp.zeros(1, dtype=wp.int32, device=device)
        combined_indices = wp.zeros(2, dtype=wp.int32, device=device)
        combined_stiffness = wp.zeros(2, dtype=wp.float32, device=device)
        wp.launch(
            eval_rigid_tau_and_augmented_drives,
            dim=1,
            inputs=[
                articulation_start,
                articulation_joint_end,
                articulation_dof_count,
                joint_type,
                joint_parent,
                joint_child,
                joint_articulation,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                spring_stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_fb_s,
                body_f_ext,
                body_flags,
                body_q,
                body_com,
                articulation_origin,
                target_ke,
                target_kd,
                target_q,
                joint_start,
                target_qd,
                effort_limit,
                2,
                0.1,
            ],
            outputs=[
                combined_body_ft,
                combined_counts,
                combined_indices,
                combined_stiffness,
                combined_tau,
            ],
            device=device,
        )
        np.testing.assert_array_equal(combined_body_ft.numpy(), sequential_body_ft.numpy())
        expected_tau = sequential_tau.numpy() + np.array([-1.25, 2.5], dtype=np.float32)
        np.testing.assert_allclose(combined_tau.numpy(), expected_tau, rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(combined_counts.numpy(), [2])
        np.testing.assert_array_equal(combined_indices.numpy(), [0, 1])
        np.testing.assert_allclose(combined_stiffness.numpy(), [0.2, 0.4], rtol=0.0, atol=1.0e-7)

        prepared_counts = wp.zeros_like(combined_counts)
        prepared_indices = wp.zeros_like(combined_indices)
        prepared_stiffness = wp.zeros_like(combined_stiffness)
        prepared_tau = wp.zeros_like(combined_tau)
        wp.launch(
            prepare_augmented_joint_drives,
            dim=1,
            inputs=[
                articulation_start,
                articulation_dof_count,
                joint_type,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_q,
                joint_qd,
                target_ke,
                target_kd,
                target_q,
                joint_start,
                target_qd,
                effort_limit,
                2,
                0.1,
            ],
            outputs=[prepared_counts, prepared_indices, prepared_stiffness, prepared_tau],
            device=device,
        )
        np.testing.assert_array_equal(prepared_counts.numpy(), combined_counts.numpy())
        np.testing.assert_array_equal(prepared_indices.numpy(), combined_indices.numpy())
        np.testing.assert_array_equal(prepared_stiffness.numpy(), combined_stiffness.numpy())
        np.testing.assert_allclose(prepared_tau.numpy(), [-1.25, 2.5], rtol=0.0, atol=1.0e-6)

        topology_stiffness = wp.zeros_like(combined_stiffness)
        topology_tau = wp.zeros_like(combined_tau)
        wp.launch(
            _prepare_augmented_joint_drives_by_dof,
            dim=2,
            inputs=[
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                joint_q,
                joint_qd,
                target_ke,
                target_kd,
                target_q,
                target_qd,
                effort_limit,
                0.1,
            ],
            outputs=[topology_stiffness, topology_tau],
            device=device,
        )
        np.testing.assert_array_equal(topology_stiffness.numpy(), combined_stiffness.numpy())
        np.testing.assert_allclose(topology_tau.numpy(), [-1.25, 2.5], rtol=0.0, atol=1.0e-6)

        additive_body_ft = wp.zeros_like(combined_body_ft)
        wp.launch(
            eval_rigid_tau_add,
            dim=1,
            inputs=[
                articulation_start,
                articulation_joint_end,
                joint_type,
                joint_parent,
                joint_child,
                joint_articulation,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                spring_stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_fb_s,
                body_f_ext,
                body_flags,
                body_q,
                body_com,
                articulation_origin,
            ],
            outputs=[additive_body_ft, prepared_tau],
            device=device,
        )
        np.testing.assert_array_equal(additive_body_ft.numpy(), combined_body_ft.numpy())
        np.testing.assert_allclose(prepared_tau.numpy(), combined_tau.numpy(), rtol=0.0, atol=1.0e-6)

        passive_counts = wp.array([7], dtype=wp.int32, device=device)
        passive_tau = wp.array([3.0, -2.0], dtype=wp.float32, device=device)
        zero_target_ke = wp.zeros_like(target_ke)
        zero_target_kd = wp.zeros_like(target_kd)
        wp.launch(
            prepare_augmented_joint_drives,
            dim=1,
            inputs=[
                articulation_start,
                articulation_dof_count,
                joint_type,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_q,
                joint_qd,
                zero_target_ke,
                zero_target_kd,
                target_q,
                joint_start,
                target_qd,
                effort_limit,
                2,
                0.1,
            ],
            outputs=[passive_counts, prepared_indices, prepared_stiffness, passive_tau],
            device=device,
        )

        np.testing.assert_array_equal(passive_counts.numpy(), [0])
        np.testing.assert_array_equal(passive_tau.numpy(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
