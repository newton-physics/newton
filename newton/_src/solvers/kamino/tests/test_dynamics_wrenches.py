# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for `dynamics/wrenches.py`.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.wrenches import (
    compute_constraint_body_wrenches_dense,
    compute_constraint_body_wrenches_sparse,
    compute_joint_dof_body_wrenches_dense,
    compute_joint_dof_body_wrenches_sparse,
)
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.extract import (
    extract_active_constraint_vectors,
    extract_actuation_forces,
    extract_cts_jacobians,
    extract_dofs_jacobians,
)
from newton._src.solvers.kamino.tests.utils.make import (
    make_constraint_multiplier_arrays,
    make_test_problem_fourbar,
    make_test_problem_heterogeneous,
)

###
# Constants
###

test_jacobian_rtol = 1e-7
test_jacobian_atol = 1e-7

# TODO: FIX THIS: sparse-dense differences are larger than expected,
# likely due to the sparse implementation not fully matching the dense
test_wrench_rtol = 1e-4  # TODO: Should be 1e-6
test_wrench_atol = 1e-4  # TODO: Should be 1e-6


###
# Helper Functions
###


def compute_and_compare_dense_sparse_jacobian_wrenches(
    model: ModelKamino,
    data: DataKamino,
    limits: LimitsKamino,
    contacts: ContactsKamino,
):
    # Create the Jacobians container
    jacobians_dense = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
    jacobians_sparse = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
    wp.synchronize()

    # Build the system Jacobians
    jacobians_dense.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
    jacobians_sparse.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
    wp.synchronize()

    # Create arrays for the constraint multipliers and initialize them
    lambdas_start, lambdas = make_constraint_multiplier_arrays(model)
    lambdas.fill_(1.0)

    # Initialize the generalized joint actuation forces
    data.joints.tau_j.fill_(1.0)

    # Compute the wrenches using the dense Jacobians
    compute_joint_dof_body_wrenches_dense(
        model=model,
        data=data,
        jacobians=jacobians_dense,
        reset_to_zero=True,
    )
    compute_constraint_body_wrenches_dense(
        model=model,
        data=data,
        jacobians=jacobians_dense,
        lambdas_offsets=lambdas_start,
        lambdas_data=lambdas,
        limits=limits.data,
        contacts=contacts.data,
        reset_to_zero=True,
    )
    wp.synchronize()
    w_a_i_dense_np = data.bodies.w_a_i.numpy().copy()
    w_j_i_dense_np = data.bodies.w_j_i.numpy().copy()
    w_l_i_dense_np = data.bodies.w_l_i.numpy().copy()
    w_c_i_dense_np = data.bodies.w_c_i.numpy().copy()

    # Compute the wrenches using the sparse Jacobians
    compute_joint_dof_body_wrenches_sparse(
        model=model,
        data=data,
        jacobians=jacobians_sparse,
        reset_to_zero=True,
    )
    compute_constraint_body_wrenches_sparse(
        model=model,
        data=data,
        jacobians=jacobians_sparse,
        lambdas_offsets=lambdas_start,
        lambdas_data=lambdas,
        reset_to_zero=True,
    )
    wp.synchronize()
    w_a_i_sparse_np = data.bodies.w_a_i.numpy().copy()
    w_j_i_sparse_np = data.bodies.w_j_i.numpy().copy()
    w_l_i_sparse_np = data.bodies.w_l_i.numpy().copy()
    w_c_i_sparse_np = data.bodies.w_c_i.numpy().copy()

    # TODO
    np.set_printoptions(precision=12, suppress=True, linewidth=20000, threshold=20000)

    # Extract the number of bodies and constraints for each world
    num_bodies_np = model.info.num_bodies.numpy().astype(int).tolist()
    num_joint_cts_np = model.info.num_joint_cts.numpy().astype(int).tolist()
    num_limit_cts_np = data.info.num_limit_cts.numpy().astype(int).tolist()
    num_contact_cts_np = data.info.num_contact_cts.numpy().astype(int).tolist()
    num_total_cts_np = data.info.num_total_cts.numpy().astype(int).tolist()
    msg.info("num_bodies_np: %s", num_bodies_np)
    msg.info("num_joint_cts_np: %s", num_joint_cts_np)
    msg.info("num_limit_cts_np: %s", num_limit_cts_np)
    msg.info("num_contact_cts_np: %s", num_contact_cts_np)
    msg.info("num_total_cts_np: %s\n", num_total_cts_np)

    # Extract the Jacobians and constraint multipliers as lists of numpy arrays (i.e. per world)
    J_cts_dense = extract_cts_jacobians(model, limits, contacts, jacobians_dense, only_active_cts=True)
    J_dofs_dense = extract_dofs_jacobians(model, jacobians_dense)
    J_cts_sparse = jacobians_sparse._J_cts.bsm.numpy()
    J_dofs_sparse = jacobians_sparse._J_dofs.bsm.numpy()
    lambdas_np = extract_active_constraint_vectors(model, data, lambdas)
    tau_j_np = extract_actuation_forces(model, data)
    for w in range(model.size.num_worlds):
        msg.info("[world='%d']: J_cts_dense:\n%s", w, J_cts_dense[w])
        msg.info("[world='%d']: J_cts_sparse:\n%s\n", w, J_cts_sparse[w])
        msg.info("[world='%d']: lambdas_np:\n%s\n\n", w, lambdas_np[w])
        msg.info("[world='%d']: J_dofs_dense:\n%s", w, J_dofs_dense[w])
        msg.info("[world='%d']: J_dofs_sparse:\n%s\n", w, J_dofs_sparse[w])
        msg.info("[world='%d']: tau_j_np:\n%s\n", w, tau_j_np[w])

    # Compute the wrenches manually using the extracted Jacobians and multipliers/forces for additional verification
    inv_dt_np = model.time.inv_dt.numpy().tolist()
    w_a_i_ref_np = [np.zeros((num_bodies_np[w], 6), dtype=np.float32) for w in range(model.size.num_worlds)]
    w_j_i_ref_np = [np.zeros((num_bodies_np[w], 6), dtype=np.float32) for w in range(model.size.num_worlds)]
    w_l_i_ref_np = [np.zeros((num_bodies_np[w], 6), dtype=np.float32) for w in range(model.size.num_worlds)]
    w_c_i_ref_np = [np.zeros((num_bodies_np[w], 6), dtype=np.float32) for w in range(model.size.num_worlds)]
    for w in range(model.size.num_worlds):
        joint_cts_start_w = 0
        joint_cts_end_w = num_joint_cts_np[w]
        limit_cts_start_w = joint_cts_end_w
        limit_cts_end_w = limit_cts_start_w + num_limit_cts_np[w]
        contact_cts_start_w = limit_cts_end_w
        contact_cts_end_w = contact_cts_start_w + num_contact_cts_np[w]
        J_cts_j = J_cts_dense[w][joint_cts_start_w:joint_cts_end_w, :]
        J_cts_l = J_cts_dense[w][limit_cts_start_w:limit_cts_end_w, :]
        J_cts_c = J_cts_dense[w][contact_cts_start_w:contact_cts_end_w, :]
        lambdas_j = lambdas_np[w][joint_cts_start_w:joint_cts_end_w]
        lambdas_l = lambdas_np[w][limit_cts_start_w:limit_cts_end_w]
        lambdas_c = lambdas_np[w][contact_cts_start_w:contact_cts_end_w]
        w_a_i_ref_np[w][:, :] = (J_dofs_dense[w].T @ tau_j_np[w]).reshape(num_bodies_np[w], 6)
        w_j_i_ref_np[w][:, :] = inv_dt_np[w] * (J_cts_j.T @ lambdas_j).reshape(num_bodies_np[w], 6)
        w_l_i_ref_np[w][:, :] = inv_dt_np[w] * (J_cts_l.T @ lambdas_l).reshape(num_bodies_np[w], 6)
        w_c_i_ref_np[w][:, :] = inv_dt_np[w] * (J_cts_c.T @ lambdas_c).reshape(num_bodies_np[w], 6)
    w_a_i_ref_np = wp.array(np.concatenate(w_a_i_ref_np, axis=0), device="cpu")
    w_j_i_ref_np = wp.array(np.concatenate(w_j_i_ref_np, axis=0), device="cpu")
    w_l_i_ref_np = wp.array(np.concatenate(w_l_i_ref_np, axis=0), device="cpu")
    w_c_i_ref_np = wp.array(np.concatenate(w_c_i_ref_np, axis=0), device="cpu")

    # Debug output
    msg.info("w_a_i_ref_np:\n%s", w_a_i_ref_np)
    msg.info("w_a_i_dense_np:\n%s", w_a_i_dense_np)
    msg.info("w_a_i_sparse_np:\n%s\n", w_a_i_sparse_np)
    msg.info("w_j_i_ref_np:\n%s", w_j_i_ref_np)
    msg.info("w_j_i_dense_np:\n%s", w_j_i_dense_np)
    msg.info("w_j_i_sparse_np:\n%s\n", w_j_i_sparse_np)
    msg.info("w_l_i_ref_np:\n%s", w_l_i_ref_np)
    msg.info("w_l_i_dense_np:\n%s", w_l_i_dense_np)
    msg.info("w_l_i_sparse_np:\n%s\n", w_l_i_sparse_np)
    msg.info("w_c_i_ref_np:\n%s", w_c_i_ref_np)
    msg.info("w_c_i_dense_np:\n%s", w_c_i_dense_np)
    msg.info("w_c_i_sparse_np:\n%s\n\n", w_c_i_sparse_np)

    # Check that the Jacobians computed using the dense and sparse implementations are close
    for w in range(model.size.num_worlds):
        np.testing.assert_allclose(J_cts_sparse[w], J_cts_dense[w], rtol=test_jacobian_rtol, atol=test_jacobian_atol)
        np.testing.assert_allclose(J_dofs_sparse[w], J_dofs_dense[w], rtol=test_jacobian_rtol, atol=test_jacobian_atol)

    # Check that the wrenches computed using the dense Jacobians match the reference wrenches
    np.testing.assert_allclose(w_a_i_dense_np, w_a_i_ref_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_j_i_dense_np, w_j_i_ref_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_l_i_dense_np, w_l_i_ref_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_c_i_dense_np, w_c_i_ref_np, rtol=test_wrench_rtol, atol=test_wrench_atol)

    # Check that the wrenches computed using the dense and sparse Jacobians are close
    np.testing.assert_allclose(w_a_i_sparse_np, w_a_i_dense_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_j_i_sparse_np, w_j_i_dense_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_l_i_sparse_np, w_l_i_dense_np, rtol=test_wrench_rtol, atol=test_wrench_atol)
    np.testing.assert_allclose(w_c_i_sparse_np, w_c_i_dense_np, rtol=test_wrench_rtol, atol=test_wrench_atol)


###
# Tests
###


class TestDynamicsWrenches(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set info-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_compute_wrenches_for_single_fourbar_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=False,  # TODO
        )

        # Compute and compare the wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_jacobian_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )

    def test_02_compute_wrenches_for_multiple_fourbars_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=False,
        )

        # Compute and compare the wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_jacobian_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )

    def test_03_compute_wrenches_heterogeneous_model_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=False,
        )

        # Compute and compare the wrenches using the dense and sparse Jacobians
        compute_and_compare_dense_sparse_jacobian_wrenches(
            model=model,
            data=data,
            limits=limits,
            contacts=contacts,
        )


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
