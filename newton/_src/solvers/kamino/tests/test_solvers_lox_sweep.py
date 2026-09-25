# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX body-space projection sweeps."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat36f, mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox.jacobi import project_constraints_jacobi
from newton._src.solvers.kamino._src.solvers.lox.projection import (
    PROJECTION_STATUS_VALID,
    prepare_jacobi_projection_data,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _project_constraints_accelerated(
    projection_iterations,
    adapter,
    world_active,
    body_world,
    inverse_weight,
    projected_twist,
    theta,
    beta,
    restart_dot,
    **offsets,
):
    project_constraints_jacobi(
        projection_iterations,
        world_active,
        body_world,
        adapter.friction_world,
        adapter.friction_local,
        adapter.world_friction_count,
        adapter.friction_body_first,
        adapter.friction_body_second,
        adapter.friction_jacobian_first,
        adapter.friction_jacobian_second,
        adapter.friction_impulse_bound,
        adapter.friction_projection_delassus,
        adapter.contact_world,
        adapter.contact_local,
        adapter.world_contact_count,
        adapter.contact_body_first,
        adapter.contact_body_second,
        adapter.contact_jacobian_first,
        adapter.contact_jacobian_second,
        adapter.contact_bias,
        adapter.contact_friction,
        adapter.contact_projection_delassus,
        adapter.limit_world,
        adapter.limit_local,
        adapter.world_limit_count,
        adapter.limit_body_first,
        adapter.limit_body_second,
        adapter.limit_jacobian_first,
        adapter.limit_jacobian_second,
        adapter.limit_bias,
        adapter.limit_projection_delassus,
        inverse_weight,
        projected_twist,
        adapter.projection_twist_delta,
        adapter.contact_reaction,
        adapter.limit_reaction,
        adapter.friction_reaction,
        adapter.world_jacobi_projection_status,
        adapter.projection_status,
        accelerated=True,
        theta=theta,
        beta=beta,
        restart_dot=restart_dot,
        friction_trial=adapter.friction_acceleration_trial,
        friction_previous=adapter.friction_acceleration_previous,
        contact_trial=adapter.contact_acceleration_trial,
        contact_previous=adapter.contact_acceleration_previous,
        limit_trial=adapter.limit_acceleration_trial,
        limit_previous=adapter.limit_acceleration_previous,
        **offsets,
    )


class TestLOXSweep(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for block synchronization")
    def test_world_jacobi_matches_global_jacobi(self):
        """Match the occupancy-gated world Jacobi path to the global path."""
        device = wp.get_device("cuda:0")
        world_count = device.sm_count * 4
        world_ids = np.arange(world_count, dtype=np.int32)
        zeros = np.zeros(world_count, dtype=np.int32)
        ones = np.ones(world_count, dtype=np.int32)
        minus_ones = -ones

        body_world = wp.array(world_ids, dtype=wp.int32, device=device)
        world_active = wp.ones(world_count, dtype=wp.bool, device=device)
        world_offset = wp.array(world_ids, dtype=wp.int32, device=device)
        world_count_one = wp.array(ones, dtype=wp.int32, device=device)
        constraint_world = wp.array(world_ids, dtype=wp.int32, device=device)
        constraint_local = wp.array(zeros, dtype=wp.int32, device=device)
        body_first = wp.array(world_ids, dtype=wp.int32, device=device)
        body_second = wp.array(minus_ones, dtype=wp.int32, device=device)

        friction_jacobian_values = np.zeros((world_count, 6), dtype=np.float32)
        friction_jacobian_values[:, 0] = 1.0
        friction_jacobian_first = wp.array(friction_jacobian_values, dtype=vec6f, device=device)
        friction_jacobian_second = wp.zeros(world_count, dtype=vec6f, device=device)
        friction_impulse_bound = wp.full(world_count, 10.0, dtype=wp.float32, device=device)
        friction_delassus = wp.ones(world_count, dtype=wp.float32, device=device)

        contact_jacobian_values = np.zeros((world_count, 3, 6), dtype=np.float32)
        contact_jacobian_values[:, 0, 3] = 1.0
        contact_jacobian_values[:, 1, 4] = 1.0
        contact_jacobian_values[:, 2, 2] = 1.0
        contact_jacobian_first = wp.array(contact_jacobian_values, dtype=mat36f, device=device)
        contact_jacobian_second = wp.zeros(world_count, dtype=mat36f, device=device)
        contact_bias = wp.zeros(world_count, dtype=wp.vec3f, device=device)
        contact_friction = wp.zeros(world_count, dtype=wp.float32, device=device)
        contact_delassus = wp.array(
            np.repeat(np.eye(3, dtype=np.float32)[None, :, :], world_count, axis=0),
            dtype=wp.mat33f,
            device=device,
        )

        limit_jacobian_values = np.zeros((world_count, 6), dtype=np.float32)
        limit_jacobian_values[:, 1] = 1.0
        limit_jacobian_first = wp.array(limit_jacobian_values, dtype=vec6f, device=device)
        limit_jacobian_second = wp.zeros(world_count, dtype=vec6f, device=device)
        limit_bias = wp.zeros(world_count, dtype=wp.float32, device=device)
        limit_delassus = wp.ones(world_count, dtype=wp.float32, device=device)

        inverse_weight_values = np.repeat(np.eye(6, dtype=np.float32)[None, :, :], world_count, axis=0)
        inverse_weight_values[:, np.arange(6), np.arange(6)] = [2.0, 0.5, 3.0, 4.0, 5.0, 6.0]
        inverse_weight = wp.array(inverse_weight_values, dtype=mat66f, device=device)
        initial_twist = np.zeros((world_count, 6), dtype=np.float32)
        initial_twist[:, :3] = [-1.0, -2.0, -3.0]
        prepared_status = wp.full(world_count, PROJECTION_STATUS_VALID, dtype=wp.int32, device=device)

        def run(use_world_projection: bool):
            projected_twist = wp.array(initial_twist, dtype=vec6f, device=device)
            twist_delta = wp.zeros(world_count, dtype=vec6f, device=device)
            contact_reaction = wp.full(
                world_count,
                wp.vec3f(0.0, 0.0, 0.25),
                dtype=wp.vec3f,
                device=device,
            )
            limit_reaction = wp.full(world_count, 0.5, dtype=wp.float32, device=device)
            friction_reaction = wp.full(world_count, 0.1, dtype=wp.float32, device=device)
            world_status = wp.zeros(world_count, dtype=wp.int32, device=device)
            offsets = {}
            if use_world_projection:
                offsets = {
                    "world_body_offset": world_offset,
                    "world_body_count": world_count_one,
                    "world_friction_offset": world_offset,
                    "world_contact_offset": world_offset,
                    "world_limit_offset": world_offset,
                }
            project_constraints_jacobi(
                3,
                world_active,
                body_world,
                constraint_world,
                constraint_local,
                world_count_one,
                body_first,
                body_second,
                friction_jacobian_first,
                friction_jacobian_second,
                friction_impulse_bound,
                friction_delassus,
                constraint_world,
                constraint_local,
                world_count_one,
                body_first,
                body_second,
                contact_jacobian_first,
                contact_jacobian_second,
                contact_bias,
                contact_friction,
                contact_delassus,
                constraint_world,
                constraint_local,
                world_count_one,
                body_first,
                body_second,
                limit_jacobian_first,
                limit_jacobian_second,
                limit_bias,
                limit_delassus,
                inverse_weight,
                projected_twist,
                twist_delta,
                contact_reaction,
                limit_reaction,
                friction_reaction,
                prepared_status,
                world_status,
                **offsets,
            )
            return (
                projected_twist.numpy(),
                contact_reaction.numpy(),
                limit_reaction.numpy(),
                friction_reaction.numpy(),
                twist_delta.numpy(),
                world_status.numpy(),
            )

        global_result = run(False)
        world_result = run(True)
        for global_value, world_value in zip(global_result[:-1], world_result[:-1], strict=True):
            np.testing.assert_allclose(world_value, global_value, rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(world_result[-1], global_result[-1])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for block synchronization")
    def test_accelerated_jacobi_accepts_world_offsets(self):
        """Match accelerated Jacobi with and without world offsets."""
        device = wp.get_device("cuda:0")
        world_count = device.sm_count * 4
        world_ids = np.arange(world_count, dtype=np.int32)
        zeros = np.zeros(world_count, dtype=np.int32)
        ones = np.ones(world_count, dtype=np.int32)
        minus_ones = -ones
        initial_twist = np.repeat(
            np.asarray([[-1.0, 0.2, 0.1, 0.8, -0.4, 0.0]], dtype=np.float32),
            world_count,
            axis=0,
        )
        inverse_weight = wp.array(
            np.repeat(np.eye(6, dtype=np.float32)[None, :, :], world_count, axis=0),
            dtype=mat66f,
            device=device,
        )
        body_world = wp.array(world_ids, dtype=wp.int32, device=device)
        world_active = wp.ones(world_count, dtype=wp.bool, device=device)
        world_offset = wp.array(world_ids, dtype=wp.int32, device=device)
        world_count_one = wp.array(ones, dtype=wp.int32, device=device)

        def make_adapter():
            adapter = SimpleNamespace(
                device=device,
                friction_capacity=world_count,
                contact_capacity=world_count,
                limit_capacity=world_count,
            )
            adapter.friction_world = wp.array(world_ids, dtype=wp.int32, device=device)
            adapter.friction_local = wp.array(zeros, dtype=wp.int32, device=device)
            adapter.world_friction_count = wp.array(ones, dtype=wp.int32, device=device)
            adapter.friction_body_first = wp.array(world_ids, dtype=wp.int32, device=device)
            adapter.friction_body_second = wp.array(minus_ones, dtype=wp.int32, device=device)
            friction_jacobian = np.zeros((world_count, 6), dtype=np.float32)
            friction_jacobian[:, 3] = 1.0
            adapter.friction_jacobian_first = wp.array(friction_jacobian, dtype=vec6f, device=device)
            adapter.friction_jacobian_second = wp.zeros(world_count, dtype=vec6f, device=device)
            adapter.friction_impulse_bound = wp.full(world_count, 0.4, dtype=wp.float32, device=device)
            adapter.friction_projection_delassus = wp.ones(world_count, dtype=wp.float32, device=device)
            adapter.friction_reaction = wp.full(world_count, 0.1, dtype=wp.float32, device=device)
            adapter.friction_acceleration_trial = wp.zeros(world_count, dtype=wp.float32, device=device)
            adapter.friction_acceleration_previous = wp.zeros(world_count, dtype=wp.float32, device=device)
            adapter.friction_velocity = wp.zeros(world_count, dtype=wp.float32, device=device)

            adapter.contact_world = wp.array(world_ids, dtype=wp.int32, device=device)
            adapter.contact_local = wp.array(zeros, dtype=wp.int32, device=device)
            adapter.world_contact_count = wp.array(ones, dtype=wp.int32, device=device)
            adapter.contact_body_first = wp.array(world_ids, dtype=wp.int32, device=device)
            adapter.contact_body_second = wp.array(minus_ones, dtype=wp.int32, device=device)
            contact_jacobian = np.zeros((world_count, 3, 6), dtype=np.float32)
            contact_jacobian[:, 0, 1] = 1.0
            contact_jacobian[:, 1, 2] = 1.0
            contact_jacobian[:, 2, 0] = 1.0
            adapter.contact_jacobian_first = wp.array(contact_jacobian, dtype=mat36f, device=device)
            adapter.contact_jacobian_second = wp.zeros(world_count, dtype=mat36f, device=device)
            adapter.contact_bias = wp.full(
                world_count,
                wp.vec3f(0.0, 0.0, -0.2),
                dtype=wp.vec3f,
                device=device,
            )
            adapter.contact_friction = wp.full(world_count, 0.5, dtype=wp.float32, device=device)
            adapter.contact_projection_delassus = wp.array(
                np.repeat(np.eye(3, dtype=np.float32)[None, :, :], world_count, axis=0),
                dtype=wp.mat33f,
                device=device,
            )
            adapter.contact_reaction = wp.zeros(world_count, dtype=wp.vec3f, device=device)
            adapter.contact_acceleration_trial = wp.zeros(world_count, dtype=wp.vec3f, device=device)
            adapter.contact_acceleration_previous = wp.zeros(world_count, dtype=wp.vec3f, device=device)
            adapter.contact_velocity = wp.zeros(world_count, dtype=wp.vec3f, device=device)

            adapter.limit_world = wp.array(world_ids, dtype=wp.int32, device=device)
            adapter.limit_local = wp.array(zeros, dtype=wp.int32, device=device)
            adapter.world_limit_count = wp.array(ones, dtype=wp.int32, device=device)
            adapter.limit_body_first = wp.array(world_ids, dtype=wp.int32, device=device)
            adapter.limit_body_second = wp.array(minus_ones, dtype=wp.int32, device=device)
            limit_jacobian = np.zeros((world_count, 6), dtype=np.float32)
            limit_jacobian[:, 4] = 1.0
            adapter.limit_jacobian_first = wp.array(limit_jacobian, dtype=vec6f, device=device)
            adapter.limit_jacobian_second = wp.zeros(world_count, dtype=vec6f, device=device)
            adapter.limit_bias = wp.full(world_count, -0.1, dtype=wp.float32, device=device)
            adapter.limit_projection_delassus = wp.ones(world_count, dtype=wp.float32, device=device)
            adapter.limit_reaction = wp.zeros(world_count, dtype=wp.float32, device=device)
            adapter.limit_acceleration_trial = wp.zeros(world_count, dtype=wp.float32, device=device)
            adapter.limit_acceleration_previous = wp.zeros(world_count, dtype=wp.float32, device=device)
            adapter.limit_velocity = wp.zeros(world_count, dtype=wp.float32, device=device)
            adapter.world_jacobi_projection_status = wp.full(
                world_count,
                PROJECTION_STATUS_VALID,
                dtype=wp.int32,
                device=device,
            )
            adapter.projection_status = wp.zeros(world_count, dtype=wp.int32, device=device)
            adapter.projection_twist_delta = wp.zeros(world_count, dtype=vec6f, device=device)
            return adapter

        def run(use_world_projection: bool):
            adapter = make_adapter()
            projected_twist = wp.array(initial_twist, dtype=vec6f, device=device)
            theta = wp.ones(world_count, dtype=wp.float32, device=device)
            beta = wp.zeros(world_count, dtype=wp.float32, device=device)
            restart_dot = wp.zeros(world_count, dtype=wp.float32, device=device)
            offsets = {}
            if use_world_projection:
                offsets = {
                    "world_body_offset": world_offset,
                    "world_body_count": world_count_one,
                    "world_friction_offset": world_offset,
                    "world_contact_offset": world_offset,
                    "world_limit_offset": world_offset,
                }
            _project_constraints_accelerated(
                3,
                adapter,
                world_active,
                body_world,
                inverse_weight,
                projected_twist,
                theta,
                beta,
                restart_dot,
                **offsets,
            )
            return (
                projected_twist.numpy(),
                adapter.projection_twist_delta.numpy(),
                adapter.friction_reaction.numpy(),
                adapter.friction_acceleration_trial.numpy(),
                adapter.friction_acceleration_previous.numpy(),
                adapter.friction_velocity.numpy(),
                adapter.contact_reaction.numpy(),
                adapter.contact_acceleration_trial.numpy(),
                adapter.contact_acceleration_previous.numpy(),
                adapter.contact_velocity.numpy(),
                adapter.limit_reaction.numpy(),
                adapter.limit_acceleration_trial.numpy(),
                adapter.limit_acceleration_previous.numpy(),
                adapter.limit_velocity.numpy(),
                theta.numpy(),
                beta.numpy(),
                restart_dot.numpy(),
                adapter.projection_status.numpy(),
            )

        global_result = run(False)
        world_result = run(True)
        for global_value, world_value in zip(global_result[:-1], world_result[:-1], strict=True):
            np.testing.assert_allclose(world_value, global_value, rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(world_result[-1], global_result[-1])

    def test_mass_split_jacobi_scales_body_contributions(self):
        """Scale Jacobi body contributions by constraint multiplicity."""
        inverse_weight_values = np.repeat(np.eye(6, dtype=np.float32)[None, :, :], 2, axis=0)
        inverse_weight_values[:, 0, 0] = [2.0, 4.0]
        inverse_weight = wp.array(inverse_weight_values, dtype=mat66f, device=self.device)
        projected_twist = wp.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6], dtype=vec6f, device=self.device)
        twist_delta = wp.zeros(2, dtype=vec6f, device=self.device)
        body_world = wp.zeros(2, dtype=wp.int32, device=self.device)
        body_constraint_count = wp.array([2, 1], dtype=wp.int32, device=self.device)
        static_body_constraint_count = wp.zeros(2, dtype=wp.int32, device=self.device)

        contact_jacobian = np.zeros((2, 3, 6), dtype=np.float32)
        contact_jacobian[:, 0, 1] = 1.0
        contact_jacobian[:, 1, 2] = 1.0
        contact_jacobian[:, 2, 0] = 1.0
        contact_world = wp.zeros(2, dtype=wp.int32, device=self.device)
        contact_local = wp.array([0, 1], dtype=wp.int32, device=self.device)
        world_contact_count = wp.array([2], dtype=wp.int32, device=self.device)
        contact_body_first = wp.zeros(2, dtype=wp.int32, device=self.device)
        contact_body_second = wp.array([1, -1], dtype=wp.int32, device=self.device)
        contact_jacobian_first = wp.array(contact_jacobian, dtype=mat36f, device=self.device)
        contact_jacobian_second_values = np.zeros_like(contact_jacobian)
        contact_jacobian_second_values[0] = contact_jacobian[0]
        contact_jacobian_second = wp.array(contact_jacobian_second_values, dtype=mat36f, device=self.device)
        contact_bias = wp.zeros(2, dtype=wp.vec3f, device=self.device)
        contact_friction = wp.zeros(2, dtype=wp.float32, device=self.device)
        contact_delassus = wp.zeros(2, dtype=wp.mat33f, device=self.device)
        contact_reaction = wp.zeros(2, dtype=wp.vec3f, device=self.device)

        empty_int = wp.empty(0, dtype=wp.int32, device=self.device)
        empty_vec6 = wp.empty(0, dtype=vec6f, device=self.device)
        empty_float = wp.empty(0, dtype=wp.float32, device=self.device)
        world_limit_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        prepared_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        projection_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        world_active = wp.ones(1, dtype=wp.bool, device=self.device)

        prepare_jacobi_projection_data(
            empty_int,
            empty_int,
            world_limit_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            contact_world,
            contact_local,
            world_contact_count,
            contact_body_first,
            contact_body_second,
            contact_jacobian_first,
            contact_jacobian_second,
            contact_bias,
            contact_friction,
            empty_int,
            empty_int,
            world_limit_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            body_constraint_count,
            static_body_constraint_count,
            inverse_weight,
            empty_float,
            contact_delassus,
            empty_float,
            prepared_status,
        )
        project_constraints_jacobi(
            1,
            world_active,
            body_world,
            empty_int,
            empty_int,
            world_limit_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            empty_float,
            empty_float,
            contact_world,
            contact_local,
            world_contact_count,
            contact_body_first,
            contact_body_second,
            contact_jacobian_first,
            contact_jacobian_second,
            contact_bias,
            contact_friction,
            contact_delassus,
            empty_int,
            empty_int,
            world_limit_count,
            empty_int,
            empty_int,
            empty_vec6,
            empty_vec6,
            empty_float,
            empty_float,
            inverse_weight,
            projected_twist,
            twist_delta,
            contact_reaction,
            empty_float,
            empty_float,
            prepared_status,
            projection_status,
        )

        np.testing.assert_allclose(contact_delassus.numpy()[:, 2, 2], [8.0, 4.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(contact_reaction.numpy()[:, 2], [0.125, 0.25], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            projected_twist.numpy(),
            [[-0.25, 0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]],
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(projection_status.numpy(), [PROJECTION_STATUS_VALID])

    def test_accelerated_jacobi_projects_all_rigid_constraint_families(self):
        """Match one rigid Jacobi sweep with its first accelerated step."""
        device = self.device
        adapter = SimpleNamespace(device=device, friction_capacity=1, contact_capacity=1, limit_capacity=1)
        adapter.friction_world = wp.array([0], dtype=wp.int32, device=device)
        adapter.friction_local = wp.array([0], dtype=wp.int32, device=device)
        adapter.world_friction_count = wp.array([1], dtype=wp.int32, device=device)
        adapter.friction_body_first = wp.array([0], dtype=wp.int32, device=device)
        adapter.friction_body_second = wp.array([-1], dtype=wp.int32, device=device)
        adapter.friction_jacobian_first = wp.array([[0, 0, 0, 1, 0, 0]], dtype=vec6f, device=device)
        adapter.friction_jacobian_second = wp.zeros(1, dtype=vec6f, device=device)
        adapter.friction_impulse_bound = wp.array([0.4], dtype=wp.float32, device=device)
        adapter.friction_reaction = wp.array([0.1], dtype=wp.float32, device=device)
        adapter.friction_velocity = wp.zeros(1, dtype=wp.float32, device=device)
        adapter.friction_projection_delassus = wp.zeros(1, dtype=wp.float32, device=device)
        adapter.friction_acceleration_trial = wp.zeros(1, dtype=wp.float32, device=device)
        adapter.friction_acceleration_previous = wp.zeros(1, dtype=wp.float32, device=device)

        adapter.contact_world = wp.array([0], dtype=wp.int32, device=device)
        adapter.contact_local = wp.array([0], dtype=wp.int32, device=device)
        adapter.world_contact_count = wp.array([1], dtype=wp.int32, device=device)
        adapter.contact_body_first = wp.array([0], dtype=wp.int32, device=device)
        adapter.contact_body_second = wp.array([-1], dtype=wp.int32, device=device)
        contact_jacobian = np.zeros((1, 3, 6), dtype=np.float32)
        contact_jacobian[0, 0, 1] = 1.0
        contact_jacobian[0, 1, 2] = 1.0
        contact_jacobian[0, 2, 0] = 1.0
        adapter.contact_jacobian_first = wp.array(contact_jacobian, dtype=mat36f, device=device)
        adapter.contact_jacobian_second = wp.zeros(1, dtype=mat36f, device=device)
        adapter.contact_bias = wp.array([[0.0, 0.0, -0.2]], dtype=wp.vec3f, device=device)
        adapter.contact_friction = wp.array([0.5], dtype=wp.float32, device=device)
        adapter.contact_reaction = wp.zeros(1, dtype=wp.vec3f, device=device)
        adapter.contact_velocity = wp.zeros(1, dtype=wp.vec3f, device=device)
        adapter.contact_projection_delassus = wp.zeros(1, dtype=wp.mat33f, device=device)
        adapter.contact_acceleration_trial = wp.zeros(1, dtype=wp.vec3f, device=device)
        adapter.contact_acceleration_previous = wp.zeros(1, dtype=wp.vec3f, device=device)

        adapter.limit_world = wp.array([0], dtype=wp.int32, device=device)
        adapter.limit_local = wp.array([0], dtype=wp.int32, device=device)
        adapter.world_limit_count = wp.array([1], dtype=wp.int32, device=device)
        adapter.limit_body_first = wp.array([0], dtype=wp.int32, device=device)
        adapter.limit_body_second = wp.array([-1], dtype=wp.int32, device=device)
        adapter.limit_jacobian_first = wp.array([[0, 0, 0, 0, 1, 0]], dtype=vec6f, device=device)
        adapter.limit_jacobian_second = wp.zeros(1, dtype=vec6f, device=device)
        adapter.limit_bias = wp.array([-0.1], dtype=wp.float32, device=device)
        adapter.limit_reaction = wp.zeros(1, dtype=wp.float32, device=device)
        adapter.limit_velocity = wp.zeros(1, dtype=wp.float32, device=device)
        adapter.limit_projection_delassus = wp.zeros(1, dtype=wp.float32, device=device)
        adapter.limit_acceleration_trial = wp.zeros(1, dtype=wp.float32, device=device)
        adapter.limit_acceleration_previous = wp.zeros(1, dtype=wp.float32, device=device)
        adapter.world_jacobi_projection_status = wp.zeros(1, dtype=wp.int32, device=device)
        adapter.projection_status = wp.zeros(1, dtype=wp.int32, device=device)
        adapter.projection_twist_delta = wp.zeros(1, dtype=vec6f, device=device)

        inverse_weight_host = np.eye(6, dtype=np.float32)
        inverse_weight_host[0, 0] = 2.0
        inverse_weight_host[1, 1] = 1.5
        inverse_weight_host[2, 2] = 0.75
        inverse_weight_host[0, 1] = 0.35
        inverse_weight_host[1, 0] = 0.35
        inverse_weight = wp.array([inverse_weight_host], dtype=mat66f, device=device)
        prepare_jacobi_projection_data(
            adapter.friction_world,
            adapter.friction_local,
            adapter.world_friction_count,
            adapter.friction_body_first,
            adapter.friction_body_second,
            adapter.friction_jacobian_first,
            adapter.friction_jacobian_second,
            adapter.contact_world,
            adapter.contact_local,
            adapter.world_contact_count,
            adapter.contact_body_first,
            adapter.contact_body_second,
            adapter.contact_jacobian_first,
            adapter.contact_jacobian_second,
            adapter.contact_bias,
            adapter.contact_friction,
            adapter.limit_world,
            adapter.limit_local,
            adapter.world_limit_count,
            adapter.limit_body_first,
            adapter.limit_body_second,
            adapter.limit_jacobian_first,
            adapter.limit_jacobian_second,
            wp.array([3], dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            inverse_weight,
            adapter.friction_projection_delassus,
            adapter.contact_projection_delassus,
            adapter.limit_projection_delassus,
            adapter.world_jacobi_projection_status,
        )
        candidate = np.array([[-1.0, 0.2, 0.1, 0.8, -0.4, 0.0]], dtype=np.float32)
        world_active = wp.ones(1, dtype=wp.bool, device=device)
        body_world = wp.array([0], dtype=wp.int32, device=device)
        jacobi_projected_twist = wp.array(candidate, dtype=vec6f, device=device)
        jacobi_friction_reaction = wp.array([0.1], dtype=wp.float32, device=device)
        jacobi_contact_reaction = wp.zeros(1, dtype=wp.vec3f, device=device)
        jacobi_limit_reaction = wp.zeros(1, dtype=wp.float32, device=device)
        jacobi_status = wp.zeros(1, dtype=wp.int32, device=device)
        project_constraints_jacobi(
            1,
            world_active,
            body_world,
            adapter.friction_world,
            adapter.friction_local,
            adapter.world_friction_count,
            adapter.friction_body_first,
            adapter.friction_body_second,
            adapter.friction_jacobian_first,
            adapter.friction_jacobian_second,
            adapter.friction_impulse_bound,
            adapter.friction_projection_delassus,
            adapter.contact_world,
            adapter.contact_local,
            adapter.world_contact_count,
            adapter.contact_body_first,
            adapter.contact_body_second,
            adapter.contact_jacobian_first,
            adapter.contact_jacobian_second,
            adapter.contact_bias,
            adapter.contact_friction,
            adapter.contact_projection_delassus,
            adapter.limit_world,
            adapter.limit_local,
            adapter.world_limit_count,
            adapter.limit_body_first,
            adapter.limit_body_second,
            adapter.limit_jacobian_first,
            adapter.limit_jacobian_second,
            adapter.limit_bias,
            adapter.limit_projection_delassus,
            inverse_weight,
            jacobi_projected_twist,
            wp.zeros(1, dtype=vec6f, device=device),
            jacobi_contact_reaction,
            jacobi_limit_reaction,
            jacobi_friction_reaction,
            adapter.world_jacobi_projection_status,
            jacobi_status,
        )

        projected_twist = wp.array(candidate, dtype=vec6f, device=device)
        _project_constraints_accelerated(
            1,
            adapter,
            world_active,
            body_world,
            inverse_weight,
            projected_twist,
            wp.ones(1, dtype=wp.float32, device=device),
            wp.zeros(1, dtype=wp.float32, device=device),
            wp.zeros(1, dtype=wp.float32, device=device),
        )

        np.testing.assert_allclose(adapter.friction_reaction.numpy(), jacobi_friction_reaction.numpy(), atol=1.0e-6)
        np.testing.assert_allclose(adapter.contact_reaction.numpy(), jacobi_contact_reaction.numpy(), atol=1.0e-6)
        np.testing.assert_allclose(adapter.limit_reaction.numpy(), jacobi_limit_reaction.numpy(), atol=1.0e-6)
        np.testing.assert_allclose(projected_twist.numpy(), jacobi_projected_twist.numpy(), atol=2.0e-6)
        contact_reaction = adapter.contact_reaction.numpy()[0]
        self.assertGreaterEqual(float(contact_reaction[2]), 0.0)
        self.assertLessEqual(float(np.linalg.norm(contact_reaction[:2])), 0.5 * float(contact_reaction[2]) + 1.0e-6)
        self.assertGreaterEqual(float(adapter.limit_reaction.numpy()[0]), 0.0)
        self.assertLessEqual(abs(float(adapter.friction_reaction.numpy()[0])), 0.4 + 1.0e-6)
        expected = candidate[0].copy()
        expected += inverse_weight_host @ (
            float(adapter.friction_reaction.numpy()[0]) * adapter.friction_jacobian_first.numpy()[0]
        )
        expected += inverse_weight_host @ (contact_jacobian[0].T @ contact_reaction)
        expected += inverse_weight_host @ (
            float(adapter.limit_reaction.numpy()[0]) * adapter.limit_jacobian_first.numpy()[0]
        )
        np.testing.assert_allclose(projected_twist.numpy()[0], expected, atol=2.0e-6)
        np.testing.assert_array_equal(adapter.projection_status.numpy(), [PROJECTION_STATUS_VALID])


if __name__ == "__main__":
    unittest.main(verbosity=2)
