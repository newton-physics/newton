# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dual-solution reconstruction of LOX outputs for Kamino's solution metrics.

LOX solves a primal splitting of the step. To evaluate Kamino's dual NCP and VI
metrics, :class:`DualSolution` packs LOX's force-valued joint, limit, and
contact reactions into the impulse-space layout of the dual problem and
evaluates the measured constraint velocity at the final body velocity.

Contacts whose LOX law departs from the canonical hard-contact model
(compliant contacts, and contacts with angular friction under spatial
friction) are masked: their reactions and velocities are zeroed so that they
drop out of the NCP and VI residuals. Metrics that couple all rows
of a world, such as ``r_v_plus`` and the objectives, still include the other
rows' coupling to the masked contacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core.types import vec6f

if TYPE_CHECKING:
    from ....config import LOXSolverConfig
    from ...core.data import DataKamino
    from ...core.model import ModelKamino
    from ...core.state import StateKamino
    from ...dynamics.dual import DualProblem
    from ...geometry.contacts import ContactsKamino
    from ...kinematics.jacobians import SparseSystemJacobians
    from ...kinematics.limits import LimitsKamino

###
# Module interface
###

__all__ = [
    "DualSolution",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _pack_joint_constraint_reactions(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    model_joint_wid: wp.array[wp.int32],
    model_joint_num_dynamic_cts: wp.array[wp.int32],
    model_joint_num_kinematic_cts: wp.array[wp.int32],
    model_joint_num_friction_cts: wp.array[wp.int32],
    model_joint_dynamic_cts_offset: wp.array[wp.int32],
    model_joint_kinematic_cts_offset: wp.array[wp.int32],
    model_joint_friction_cts_offset: wp.array[wp.int32],
    model_joint_dynamic_cts_offset_total_cts: wp.array[wp.int32],
    model_joint_kinematic_cts_offset_total_cts: wp.array[wp.int32],
    model_joint_friction_cts_offset_total_cts: wp.array[wp.int32],
    data_joints_lambda_dyn_j: wp.array[wp.float32],
    data_joints_lambda_kin_j: wp.array[wp.float32],
    data_joints_lambda_f_j: wp.array[wp.float32],
    # Outputs:
    constraint_impulse: wp.array[wp.float32],
):
    """Pack force-valued joint reactions into the dual impulse layout."""
    # Retrieve the joint index from the thread index
    jid = wp.tid()

    # Retrieve the time step of the joint's world
    wid = model_joint_wid[jid]
    dt = model_time_dt[wid]

    # Retrieve the joint constraint sizes and offsets
    num_dyn_cts_j = model_joint_num_dynamic_cts[jid]
    num_kin_cts_j = model_joint_num_kinematic_cts[jid]
    num_f_cts_j = model_joint_num_friction_cts[jid]
    dyn_cio_j = model_joint_dynamic_cts_offset[jid]
    kin_cio_j = model_joint_kinematic_cts_offset[jid]
    f_cio_j = model_joint_friction_cts_offset[jid]
    dyn_vio_j = model_joint_dynamic_cts_offset_total_cts[jid]
    kin_vio_j = model_joint_kinematic_cts_offset_total_cts[jid]
    f_vio_j = model_joint_friction_cts_offset_total_cts[jid]

    # Convert the reaction forces to impulses
    for row in range(num_dyn_cts_j):
        constraint_impulse[dyn_vio_j + row] = dt * data_joints_lambda_dyn_j[dyn_cio_j + row]
    for row in range(num_kin_cts_j):
        constraint_impulse[kin_vio_j + row] = dt * data_joints_lambda_kin_j[kin_cio_j + row]
    for row in range(num_f_cts_j):
        constraint_impulse[f_vio_j + row] = dt * data_joints_lambda_f_j[f_cio_j + row]


@wp.kernel
def _pack_limit_constraint_reactions(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    model_info_total_cts_offset: wp.array[wp.int32],
    data_info_limit_cts_group_offset: wp.array[wp.int32],
    limit_model_num_limits: wp.array[wp.int32],
    limit_wid: wp.array[wp.int32],
    limit_lid: wp.array[wp.int32],
    limit_reaction: wp.array[wp.float32],
    # Outputs:
    constraint_impulse: wp.array[wp.float32],
):
    """Pack force-valued limit reactions into the dual impulse layout."""
    # Retrieve the thread index as the limit index
    lid = wp.tid()

    # Skip if lid is greater than the number of limits active in the model
    if lid >= limit_model_num_limits[0]:
        return

    # Convert the reaction force to an impulse
    wid = limit_wid[lid]
    vio_l = model_info_total_cts_offset[wid] + data_info_limit_cts_group_offset[wid] + limit_lid[lid]
    constraint_impulse[vio_l] = model_time_dt[wid] * limit_reaction[lid]


@wp.kernel
def _pack_contact_constraint_reactions(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    model_info_total_cts_offset: wp.array[wp.int32],
    data_info_contact_cts_group_offset: wp.array[wp.int32],
    contact_model_num_contacts: wp.array[wp.int32],
    contact_wid: wp.array[wp.int32],
    contact_cid: wp.array[wp.int32],
    contact_reaction: wp.array[wp.vec3f],
    contact_angular_friction: wp.array[wp.vec2f],
    contact_stiffness: wp.array[wp.float32],
    use_spatial_friction: wp.bool,
    # Outputs:
    constraint_impulse: wp.array[wp.float32],
    constraint_velocity: wp.array[wp.float32],
):
    """Pack force-valued contact reactions into the dual impulse layout, masking extended-law contacts."""
    # Retrieve the thread index as the contact index
    cid = wp.tid()

    # Skip if cid is greater than the number of contacts active in the model
    if cid >= contact_model_num_contacts[0]:
        return

    wid = contact_wid[cid]
    vio_k = model_info_total_cts_offset[wid] + data_info_contact_cts_group_offset[wid] + 3 * contact_cid[cid]

    # Zero the rows of the contacts outside the hard-contact model
    masked = wp.bool(False)
    if contact_stiffness:
        masked = contact_stiffness[cid] > 0.0
    if use_spatial_friction:
        angular = contact_angular_friction[cid]
        masked = masked or angular[0] != 0.0 or angular[1] != 0.0
    if masked:
        for i in range(3):
            constraint_impulse[vio_k + i] = 0.0
            constraint_velocity[vio_k + i] = 0.0
        return

    # Convert the reaction force to an impulse
    lambda_k = model_time_dt[wid] * contact_reaction[cid]
    for i in range(3):
        constraint_impulse[vio_k + i] = lambda_k[i]


@wp.kernel
def _initialize_constraint_velocity(
    # Inputs:
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_v_b: wp.array[wp.float32],
    # Outputs:
    constraint_velocity: wp.array[wp.float32],
):
    # Retrieve the world and constraint row indices from the 2D thread grid
    wid, row = wp.tid()

    # Initialize the constraint velocity with the velocity bias
    if row < problem_dim[wid]:
        vio_row = problem_vio[wid] + row
        constraint_velocity[vio_row] = problem_v_b[vio_row]


@wp.kernel
def _accumulate_constraint_velocity(
    # Inputs:
    model_info_bodies_offset: wp.array[wp.int32],
    state_bodies_u_i_p: wp.array[wp.spatial_vectorf],
    data_bodies_u_i: wp.array[wp.spatial_vectorf],
    jacobian_cts_num_nzb: wp.array[wp.int32],
    jacobian_cts_nzb_start: wp.array[wp.int32],
    jacobian_cts_nzb_coords: wp.array2d[wp.int32],
    jacobian_cts_nzb_values: wp.array[vec6f],
    problem_dim: wp.array[wp.int32],
    problem_vio: wp.array[wp.int32],
    problem_v_i: wp.array[wp.float32],
    # Outputs:
    constraint_velocity: wp.array[wp.float32],
):
    # Retrieve the world and non-zero block indices from the 2D thread grid
    wid, tid = wp.tid()

    # Skip if the block index exceeds the number of non-zero blocks of the world
    if tid >= jacobian_cts_num_nzb[wid]:
        return

    # Retrieve the constraint row and body of the non-zero block
    nzb = jacobian_cts_nzb_start[wid] + tid
    coords = jacobian_cts_nzb_coords[nzb]
    row = coords[0]
    if row >= problem_dim[wid]:
        return
    vio_row = problem_vio[wid] + row
    bid = model_info_bodies_offset[wid] + coords[1] // 6

    # Accumulate the block contribution: J_i u_i + v_i * J_i u_i_p
    J_i = jacobian_cts_nzb_values[nzb]
    v = wp.dot(J_i, data_bodies_u_i[bid])
    v += problem_v_i[vio_row] * wp.dot(J_i, state_bodies_u_i_p[bid])
    wp.atomic_add(constraint_velocity, vio_row, v)


###
# Interfaces
###


class DualSolution:
    """Dual solution reconstructed from LOX's force-valued constraint outputs.

    The reconstruction reads the reactions written by LOX and the analysis dual
    problem, which must have been built at the LOX linearization before the
    solve. The velocity is the measured constraint velocity at the final body
    velocity, so metrics must be evaluated with ``use_solution_velocity=True``.
    """

    def __init__(self, model: ModelKamino, config: LOXSolverConfig):
        """Allocates the dual solution of a model.

        Args:
            model: The model containing the time-invariant simulation data.
            config: The LOX solver config, selecting the contact law.
        """
        self._config = config
        with wp.ScopedDevice(model.device):
            self.sigma = wp.zeros(model.size.num_worlds, dtype=wp.vec2f)
            """Delassus regularization of each world, zero for LOX."""
            self.lambdas = wp.zeros(model.size.sum_of_max_total_cts, dtype=wp.float32)
            """Constraint impulses in the dual problem layout; zero on masked contacts."""
            self.v_plus = wp.zeros(model.size.sum_of_max_total_cts, dtype=wp.float32)
            """Measured post-event constraint velocities; zero on masked contacts."""

    ###
    # Public API
    ###

    def build(
        self,
        model: ModelKamino,
        data: DataKamino,
        state_p: StateKamino,
        problem: DualProblem,
        jacobians: SparseSystemJacobians,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ) -> None:
        """Reconstruct the dual solution after a LOX solve.

        Args:
            model: The model containing the time-invariant simulation data.
            data: The model data holding the final body state and joint reactions.
            state_p: The state at the beginning of the time step.
            problem: The analysis dual problem built before the LOX solve.
            jacobians: The sparse system Jacobians used by the LOX solve.
            limits: Active joint-limit constraints.
            contacts: Active contact constraints.
        """
        jacobian = jacobians._J_cts.bsm

        self.lambdas.zero_()
        self.v_plus.zero_()
        if model.size.sum_of_num_joints > 0:
            wp.launch(
                _pack_joint_constraint_reactions,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    model.time.dt,
                    model.joints.wid,
                    model.joints.num_dynamic_cts,
                    model.joints.num_kinematic_cts,
                    model.joints.num_friction_cts,
                    model.joints.dynamic_cts_offset,
                    model.joints.kinematic_cts_offset,
                    model.joints.friction_cts_offset,
                    model.joints.dynamic_cts_offset_total_cts,
                    model.joints.kinematic_cts_offset_total_cts,
                    model.joints.friction_cts_offset_total_cts,
                    data.joints.lambda_dyn_j,
                    data.joints.lambda_kin_j,
                    data.joints.lambda_f_j,
                ],
                outputs=[self.lambdas],
                device=model.device,
            )
        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                _pack_limit_constraint_reactions,
                dim=limits.model_max_limits_host,
                inputs=[
                    model.time.dt,
                    model.info.total_cts_offset,
                    data.info.limit_cts_group_offset,
                    limits.model_active_limits,
                    limits.wid,
                    limits.lid,
                    limits.reaction,
                ],
                outputs=[self.lambdas],
                device=model.device,
            )

        # Evaluate the measured constraint velocity: v = v_b + J u + v_i * J u_p
        wp.launch(
            _initialize_constraint_velocity,
            dim=(model.size.num_worlds, model.size.max_of_max_total_cts),
            inputs=[problem.data.dim, problem.data.vio, problem.data.v_b],
            outputs=[self.v_plus],
            device=model.device,
        )
        wp.launch(
            _accumulate_constraint_velocity,
            dim=(model.size.num_worlds, jacobian.max_of_num_nzb),
            inputs=[
                model.info.bodies_offset,
                state_p.u_i,
                data.bodies.u_i,
                jacobian.num_nzb,
                jacobian.nzb_start,
                jacobian.nzb_coords,
                jacobian.nzb_values,
                problem.data.dim,
                problem.data.vio,
                problem.data.v_i,
            ],
            outputs=[self.v_plus],
            device=model.device,
        )

        # Pack the contacts last, since masking also zeroes their velocity rows
        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                _pack_contact_constraint_reactions,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    model.time.dt,
                    model.info.total_cts_offset,
                    data.info.contact_cts_group_offset,
                    contacts.model_active_contacts,
                    contacts.wid,
                    contacts.cid,
                    contacts.reaction,
                    contacts.angular_friction,
                    contacts.stiffness if self._config.contact_compliance else None,
                    self._config.contact_spatial_friction,
                ],
                outputs=[self.lambdas, self.v_plus],
                device=model.device,
            )
