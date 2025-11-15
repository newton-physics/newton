# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KAMINO: UNIT TESTS: GENERAL UTILITIES
"""

import numpy as np
import warp as wp
from warp.context import Devicelike

from newton._src.solvers.kamino.core.bodies import update_body_inertias
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.geometry.detector import CollisionDetector
from newton._src.solvers.kamino.kinematics.constraints import make_unilateral_constraints_info, update_constraints_info
from newton._src.solvers.kamino.kinematics.jacobians import DenseSystemJacobians
from newton._src.solvers.kamino.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino.kinematics.limits import Limits
from newton._src.solvers.kamino.simulation.simulator import Simulator

###
# Helper functions
###


def make_simulator(nw: int, model_build_func, device=None) -> Simulator:
    # Create a model builder
    builder = ModelBuilder()

    # Construct a specific test model in the builder
    _, _, _ = model_build_func(builder)

    # Create additional builders and add them to the main builder
    for _ in range(nw - 1):
        other = ModelBuilder()
        _, _, _ = model_build_func(other)
        builder.add_builder(other)

    sim = Simulator(builder=builder, device=device)

    # Run only the parts of the simulation pipeline that are needed for the test
    sim._collide()
    sim._forward_intermediate()
    sim._forward_kinematics()

    # Create the ready-to-use Simulator instance
    # NOTE: We have already called the necessary forward
    # methods to update the internal data containers
    return sim


def make_generalized_mass_matrices(model: Model, data: ModelData) -> list[np.ndarray]:
    # Extract the masses and inertias as numpy arrays
    m_i = model.bodies.m_i.numpy()
    I_i = data.bodies.I_i.numpy()

    # Initialize a list to hold the generalized mass matrices
    M_np: list[np.ndarray] = []

    # Iterate over each world in the model and construct the generalized mass matrix
    num_worlds = model.info.num_worlds
    for w in range(num_worlds):
        nb = model.worlds[w].num_bodies
        bio = model.worlds[w].bodies_idx_offset
        M = np.zeros((6 * nb, 6 * nb), dtype=np.float32)
        for i in range(nb):
            start = 6 * i
            M[start : start + 3, start : start + 3] = m_i[bio + i] * np.eye(3)  # Linear part
            M[start + 3 : start + 6, start + 3 : start + 6] = I_i[bio + i]  # Angular part
        M_np.append(M)

    # Return the list of generalized mass matrices
    return M_np


def make_inverse_generalized_mass_matrices(model: Model, data: ModelData) -> list[np.ndarray]:
    # Extract the inverse masses and inertias as numpy arrays
    inv_m_i = model.bodies.inv_m_i.numpy()
    inv_I_i = data.bodies.inv_I_i.numpy()

    # Initialize a list to hold the inverse generalized mass matrices
    invM_np: list[np.ndarray] = []

    # Iterate over each world in the model and construct the inverse generalized mass matrix
    num_worlds = model.info.num_worlds
    for w in range(num_worlds):
        nb = model.worlds[w].num_bodies
        bio = model.worlds[w].bodies_idx_offset
        invM = np.zeros((6 * nb, 6 * nb), dtype=np.float32)
        for i in range(nb):
            start = 6 * i
            invM[start : start + 3, start : start + 3] = inv_m_i[bio + i] * np.eye(3)  # Linear part
            invM[start + 3 : start + 6, start + 3 : start + 6] = inv_I_i[bio + i]  # Angular part
        invM_np.append(invM)

    # Return the list of inverse generalized mass matrices
    return invM_np


def make_containers(
    builder: ModelBuilder, max_world_contacts: int = 0, dt: float = 0.001, device: Devicelike = None
) -> tuple[Model, ModelData, Limits, CollisionDetector, DenseSystemJacobians]:
    # Create the model from the builder
    model = builder.finalize(device=device)

    # Configure model time-steps
    model.time.dt.fill_(wp.float32(dt))
    model.time.inv_dt.fill_(wp.float32(1.0 / dt))

    # Create a model data container
    data = model.data(device=device)

    # Create the limits container
    limits = Limits(builder=builder, device=device)

    # Create the collision detector
    detector = CollisionDetector(builder=builder, default_max_contacts=max_world_contacts, device=device)

    # Construct the unilateral constraints members in the model info
    make_unilateral_constraints_info(model, data, limits, detector.contacts, device=device)

    # Create the Jacobians container
    jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=detector.contacts, device=device)

    # Return the model, data, detector, and jacobians
    return model, data, limits, detector, jacobians


def update_containers(
    model: Model, data: ModelData, limits: Limits, detector: CollisionDetector, jacobians: DenseSystemJacobians
) -> None:
    # Update body inertias according to the current state of the bodies
    update_body_inertias(model=model.bodies, data=data.bodies)
    wp.synchronize()

    # Update joint states according to the state of the bodies
    compute_joints_data(model=model, data=data, q_j_ref=wp.zeros_like(data.joints.q_j))
    wp.synchronize()

    # Run joint-limit detection to generate active limits
    limits.detect(model, data=data)
    wp.synchronize()

    # Run collision detection to generate active contacts
    detector.collide(model, data=data)
    wp.synchronize()

    # Update the constraint state info
    update_constraints_info(model=model, data=data)
    wp.synchronize()

    # Build the dense system Jacobians
    jacobians.build(model=model, data=data, limits=limits.data, contacts=detector.contacts.data)
    wp.synchronize()
