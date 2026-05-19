# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the model container of Kamino."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

# Newton imports
from .....sim import Model

# Kamino imports
from .bodies import RigidBodiesData, RigidBodiesModel
from .control import ControlKamino
from .conversions import (
    convert_entity_local_transforms,
    convert_geometries,
    convert_joints,
    convert_rigid_bodies,
)
from .data import DataKamino, DataKaminoInfo
from .geometry import GeometriesData, GeometriesModel
from .gravity import GravityModel
from .joints import (
    JointsData,
    JointsModel,
)
from .materials import MaterialManager, MaterialPairsModel, MaterialsModel
from .size import SizeKamino
from .state import StateKamino
from .time import TimeData, TimeModel
from .types import float32, int32, mat33f, transformf, vec6f

###
# Module interface
###

__all__ = [
    "ModelKamino",
    "ModelKaminoInfo",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@dataclass
class ModelKaminoInfo:
    """
    A container to hold the time-invariant information and meta-data of a model.
    """

    ###
    # Host-side Summary Counts
    ###

    num_worlds: int = 0
    """The number of worlds represented in the model."""

    ###
    # Entity Counts
    ###

    num_bodies: wp.array | None = None
    """
    The number of bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joints: wp.array | None = None
    """
    The number of joints in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_passive_joints: wp.array | None = None
    """
    The number of passive joints in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_actuated_joints: wp.array | None = None
    """
    The number of actuated joints in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_dynamic_joints: wp.array | None = None
    """
    The number of dynamic joints in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_geoms: wp.array | None = None
    """
    The number of geometries in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_limits: wp.array | None = None
    """
    The maximum number of limits in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_contacts: wp.array | None = None
    """
    The maximum number of contacts in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # DoF Counts
    ###

    num_body_dofs: wp.array | None = None
    """
    The number of body DoFs of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joint_coords: wp.array | None = None
    """
    The number of joint coordinates of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joint_dofs: wp.array | None = None
    """
    The number of joint DoFs of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_passive_joint_coords: wp.array | None = None
    """
    The number of passive joint coordinates of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_passive_joint_dofs: wp.array | None = None
    """
    The number of passive joint DoFs of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_actuated_joint_coords: wp.array | None = None
    """
    The number of actuated joint coordinates of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_actuated_joint_dofs: wp.array | None = None
    """
    The number of actuated joint DoFs of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Constraint Counts
    ###

    # TODO: We could make this a vec2i to store dynamic
    # and kinematic joint constraint counts separately
    num_joint_cts: wp.array | None = None
    """
    The number of joint constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joint_dynamic_cts: wp.array | None = None
    """
    The number of dynamic joint constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    num_joint_kinematic_cts: wp.array | None = None
    """
    The number of kinematic joint constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_limit_cts: wp.array | None = None
    """
    The maximum number of active limit constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_contact_cts: wp.array | None = None
    """
    The maximum number of active contact constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    max_total_cts: wp.array | None = None
    """
    The maximum total number of active constraints of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Entity Offsets
    ###

    bodies_offset: wp.array | None = None
    """
    The body index offset of each world w.r.t the model.\n
    Shape of ``(num_worlds + 1,)`` and type :class:`int`.
    The last entry is the total bodies count, so that the per-world
    bodies count is encoded as ``bodies_offset[w+1] - bodies_offset[w]``.
    """

    joints_offset: wp.array | None = None
    """
    The joint index offset of each world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    geoms_offset: wp.array | None = None
    """
    The geom index offset of each world w.r.t. the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    limits_offset: wp.array | None = None
    """
    The limit index offset of each world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    contacts_offset: wp.array | None = None
    """
    The contact index offset of world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    unilaterals_offset: wp.array | None = None
    """
    The index offset of the unilaterals (limits + contacts) block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # DoF Offsets
    ###

    body_dofs_offset: wp.array | None = None
    """
    The index offset of the body DoF block of each world.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_coords_offset: wp.array | None = None
    """
    The index offset of the joint coordinates block of each world.\n
    Used to index into arrays that contain flattened joint coordinate-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_dofs_offset: wp.array | None = None
    """
    The index offset of the joint DoF block of each world.\n
    Used to index into arrays that contain flattened joint DoF-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_passive_coords_offset: wp.array | None = None
    """
    The index offset of the passive joint coordinates block of each world.\n
    Used to index into arrays that contain flattened passive joint coordinate-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_passive_dofs_offset: wp.array | None = None
    """
    The index offset of the passive joint DoF block of each world.\n
    Used to index into arrays that contain flattened passive joint DoF-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_actuated_coords_offset: wp.array | None = None
    """
    The index offset of the actuated joint coordinates block of each world.\n
    Used to index into arrays that contain flattened actuated joint coordinate-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_actuated_dofs_offset: wp.array | None = None
    """
    The index offset of the actuated joint DoF block of each world.\n
    Used to index into arrays that contain flattened actuated joint DoF-sized data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Constraint Offsets
    ###

    joint_cts_offset: wp.array | None = None
    """
    The index offset of the joint constraints block of each world.\n
    Used to index into arrays that contain flattened and
    concatenated dynamic and kinematic joint constraint data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_dynamic_cts_offset: wp.array | None = None
    """
    The index offset of the dynamic joint constraints block of each world.\n
    Used to index into arrays that contain flattened dynamic joint constraint data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_kinematic_cts_offset: wp.array | None = None
    """
    The index offset of the kinematic joint constraints block of each world.\n
    Used to index into arrays that contain flattened kinematic joint constraint data.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    # TODO: We could make this an array of vec5i and store the absolute
    #  startindex of each constraint group in the constraint array `lambda`:
    # - [0]: total_cts_offset
    # - [1]: joint_dynamic_cts_group_offset
    # - [2]: joint_kinematic_cts_group_offset
    # - [3]: limit_cts_group_offset
    # - [4]: contact_cts_group_offset
    # TODO: We could then provide helper functions to get the start-end of each block
    total_cts_offset: wp.array | None = None
    """
    The index offset of the total constraints block of each world.\n
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.\n

    This offset should be used together with:
    - joint_dynamic_cts_group_offset
    - joint_kinematic_cts_group_offset
    - limit_cts_group_offset
    - contact_cts_group_offset

    Example:
    ```
    # To index into the dynamic joint constraint reactions of world `w`:
    world_cts_start = model_info.total_cts_offset[w]
    local_joint_dynamic_cts_start = model_info.joint_dynamic_cts_group_offset[w]
    local_joint_kinematic_cts_start = model_info.joint_kinematic_cts_group_offset[w]
    local_limit_cts_start = model_info.limit_cts_group_offset[w]
    local_contact_cts_start = model_info.contact_cts_group_offset[w]

    # Now compute the starting index of each constraint group within the total constraints block of world `w`:
    world_dynamic_joint_cts_start = world_cts_start + local_joint_dynamic_cts_start
    world_kinematic_joint_cts_start = world_cts_start + local_joint_kinematic_cts_start
    world_limit_cts_start = world_cts_start + local_limit_cts_start
    world_contact_cts_start = world_cts_start + local_contact_cts_start
    ```

    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_dynamic_cts_group_offset: wp.array | None = None
    """
    The index offset of the dynamic joint constraints group within the constraints block of each world.\n
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    joint_kinematic_cts_group_offset: wp.array | None = None
    """
    The index offset of the kinematic joint constraints group within the constraints block of each world.\n
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Base Properties
    ###

    base_body_index: wp.array | None = None
    """
    The index of the base body assigned in each world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    base_joint_index: wp.array | None = None
    """
    The index of the base joint assigned in each world w.r.t the model.\n
    Shape of ``(num_worlds,)`` and type :class:`int`.
    """

    ###
    # Inertial Properties
    ###

    mass_min: wp.array | None = None
    """
    Smallest mass amongst all bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`float`.
    """

    mass_max: wp.array | None = None
    """
    Largest mass amongst all bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`float`.
    """

    mass_total: wp.array | None = None
    """
    Total mass over all bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`float`.
    """

    inertia_total: wp.array | None = None
    """
    Total diagonal inertia over all bodies in each world.\n
    Shape of ``(num_worlds,)`` and type :class:`float`.
    """


@dataclass
class ModelKamino:
    """
    A container to hold the time-invariant system model data.
    """

    _model: Model | None = None
    """The base :class:`newton.Model` instance from which this :class:`kamino.ModelKamino` was created."""

    _device: wp.DeviceLike | None = None
    """The Warp device on which the model data is allocated."""

    _requires_grad: bool = False
    """Whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""

    size: SizeKamino | None = None
    """
    Host-side cache of the model summary sizes.\n
    This is used for memory allocations and kernel thread dimensions.
    """

    info: ModelKaminoInfo | None = None
    """The model info container holding the information and meta-data of the model."""

    time: TimeModel | None = None
    """The time model container holding time-step of each world."""

    gravity: GravityModel | None = None
    """The gravity model container holding the gravity configurations for each world."""

    bodies: RigidBodiesModel | None = None
    """The rigid bodies model container holding all rigid body entities in the model."""

    joints: JointsModel | None = None
    """The joints model container holding all joint entities in the model."""

    geoms: GeometriesModel | None = None
    """The geometries model container holding all geometry entities in the model."""

    materials: MaterialsModel | None = None
    """
    The materials model container holding all material entities in the model.\n
    The materials data is currently defined globally to be shared by all worlds.
    """

    material_pairs: MaterialPairsModel | None = None
    """
    The material pairs model container holding all material pairs in the model.\n
    The material-pairs data is currently defined globally to be shared by all worlds.
    """

    ###
    # Properties
    ###

    @property
    def device(self) -> wp.DeviceLike:
        """The Warp device on which the model data is allocated."""
        return self._device

    @property
    def requires_grad(self) -> bool:
        """Whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""
        return self._requires_grad

    ###
    # Factories
    ###

    def data(
        self,
        unilateral_cts: bool = False,
        joint_wrenches: bool = False,
        requires_grad: bool = False,
        device: wp.DeviceLike = None,
    ) -> DataKamino:
        """
        Creates a model data container with the initial state of the model entities.

        Parameters:
            unilateral_cts (`bool`, optional):
                Whether to include unilateral constraints (limits and contacts) in the model data. Defaults to `True`.
            joint_wrenches (`bool`, optional):
                Whether to include joint wrenches in the model data. Defaults to `False`.
            requires_grad (`bool`, optional):
                Whether the model data should require gradients. Defaults to `False`.
            device (`wp.DeviceLike`, optional):
                The device to create the model data on. If not specified, the model's device is used.
                Defaults to `None`. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Retrieve entity counts
        nw = self.size.num_worlds
        nb = self.size.sum_of_num_bodies
        nj = self.size.sum_of_num_joints
        ng = self.size.sum_of_num_geoms

        # Retrieve the joint coordinate, DoF and constraint counts
        njcoords = self.size.sum_of_num_joint_coords
        njdofs = self.size.sum_of_num_joint_dofs
        njcts = self.size.sum_of_num_joint_cts
        njdyncts = self.size.sum_of_num_dynamic_joint_cts
        njkincts = self.size.sum_of_num_kinematic_joint_cts

        # Construct the model data on the specified device
        with wp.ScopedDevice(device=device):
            # Create a new model data info with the total constraint
            # counts initialized to the joint constraints count
            info = DataKaminoInfo(
                num_total_cts=wp.clone(self.info.num_joint_cts),
                num_limits=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                num_contacts=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                num_limit_cts=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                num_contact_cts=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                limit_cts_group_offset=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
                contact_cts_group_offset=wp.zeros(shape=nw, dtype=int32) if unilateral_cts else None,
            )

            # Construct the time data with the initial step and time set to zero for all worlds
            time = TimeData(
                steps=wp.zeros(shape=nw, dtype=int32, requires_grad=requires_grad),
                time=wp.zeros(shape=nw, dtype=float32, requires_grad=requires_grad),
            )

            # Construct the rigid bodies data from the model's initial state
            bodies = RigidBodiesData(
                num_bodies=nb,
                I_i=wp.zeros(shape=nb, dtype=mat33f, requires_grad=requires_grad),
                inv_I_i=wp.zeros(shape=nb, dtype=mat33f, requires_grad=requires_grad),
                q_i=wp.clone(self.bodies.q_i_0, requires_grad=requires_grad),
                u_i=wp.clone(self.bodies.u_i_0, requires_grad=requires_grad),
                w_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_a_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_j_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_l_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_c_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_e_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
            )

            # Construct the joints data from the model's initial state
            joints = JointsData(
                num_joints=nj,
                p_j=wp.zeros(shape=nj, dtype=transformf, requires_grad=requires_grad),
                q_j=wp.zeros(shape=njcoords, dtype=float32, requires_grad=requires_grad),
                q_j_p=wp.zeros(shape=njcoords, dtype=float32, requires_grad=requires_grad),
                dq_j=wp.zeros(shape=njdofs, dtype=float32, requires_grad=requires_grad),
                tau_j=wp.zeros(shape=njdofs, dtype=float32, requires_grad=requires_grad),
                r_j=wp.zeros(shape=njkincts, dtype=float32, requires_grad=requires_grad),
                dr_j=wp.zeros(shape=njkincts, dtype=float32, requires_grad=requires_grad),
                lambda_j=wp.zeros(shape=njcts, dtype=float32, requires_grad=requires_grad),
                m_j=wp.zeros(shape=njdyncts, dtype=float32, requires_grad=requires_grad),
                inv_m_j=wp.zeros(shape=njdyncts, dtype=float32, requires_grad=requires_grad),
                dq_b_j=wp.zeros(shape=njdyncts, dtype=float32, requires_grad=requires_grad),
                # TODO: Should we make these optional and only include them when implicit joints are present?
                q_j_ref=wp.clone(self.joints.q_j_0, requires_grad=requires_grad),
                dq_j_ref=wp.clone(self.joints.dq_j_0, requires_grad=requires_grad),
                tau_j_ref=wp.zeros(shape=njdofs, dtype=float32, requires_grad=requires_grad),
                j_w_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad) if joint_wrenches else None,
                j_w_c_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad) if joint_wrenches else None,
                j_w_a_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad) if joint_wrenches else None,
                j_w_l_j=wp.zeros(shape=nj, dtype=vec6f, requires_grad=requires_grad) if joint_wrenches else None,
            )

            # Construct the geometries data from the model's initial state
            geoms = GeometriesData(
                num_geoms=ng,
                pose=wp.zeros(shape=ng, dtype=transformf, requires_grad=requires_grad),
            )

        # Assemble and return the new data container
        return DataKamino(
            info=info,
            time=time,
            bodies=bodies,
            joints=joints,
            geoms=geoms,
        )

    def state(self, requires_grad: bool = False, device: wp.DeviceLike = None) -> StateKamino:
        """
        Creates state container initialized to the initial body state defined in the model.

        Parameters:
            requires_grad (`bool`, optional):
                Whether the state should require gradients. Defaults to `False`.
            device (`wp.DeviceLike`, optional):
                The device to create the state on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Create a new state container with the initial state of the model entities on the specified device
        with wp.ScopedDevice(device=device):
            state = StateKamino(
                q_i=wp.clone(self.bodies.q_i_0, requires_grad=requires_grad),
                u_i=wp.clone(self.bodies.u_i_0, requires_grad=requires_grad),
                w_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_i_e=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                q_j=wp.clone(self.joints.q_j_0, requires_grad=requires_grad),
                q_j_p=wp.clone(self.joints.q_j_0, requires_grad=requires_grad),
                dq_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
                lambda_j=wp.zeros(shape=self.size.sum_of_num_joint_cts, dtype=float32, requires_grad=requires_grad),
            )

        # Return the constructed state container
        return state

    def control(self, requires_grad: bool = False, device: wp.DeviceLike = None) -> ControlKamino:
        """
        Creates a control container with all values initialized to zeros.

        Parameters:
            requires_grad (`bool`, optional):
                Whether the control container should require gradients. Defaults to `False`.
            device (`wp.DeviceLike`, optional):
                The device to create the control container on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Create a new control container on the specified device
        with wp.ScopedDevice(device=device):
            control = ControlKamino(
                tau_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
                q_j_ref=wp.clone(self.joints.q_j_0, requires_grad=requires_grad),
                dq_j_ref=wp.clone(self.joints.dq_j_0, requires_grad=requires_grad),
                tau_j_ref=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
            )

        # Post-processing to finalize the control container
        # NOTE: This is currently necessary to handle the case when
        # the total number of joint coordinates and DoFs differ, in
        # which case a temporary buffer is allocated for the conversion.
        control.finalize(self, device=device)

        # Return the constructed control container
        return control

    @staticmethod
    def from_newton(model: Model, overwrite_source_model: bool = True) -> ModelKamino:
        """
        Finalizes the :class:`ModelKamino` from an existing instance of :class:`newton.Model`.

        Args:
            model:
                The source :class:`newton.Model` instance to be converted.
            overwrite_source_model:
                Whether to overwrite the source model with
                the converted model. Defaults to `True`.
        """

        # Ensure the base model is valid
        if model is None:
            raise ValueError("Cannot finalize ModelKamino from a None newton.Model instance.")
        elif not isinstance(model, Model):
            raise TypeError("Cannot finalize ModelKamino from an invalid newton.Model instance.")

        # Single-world Newton models may have world index -1 (unassigned).
        # Normalize to 0 so downstream world-based grouping works correctly.
        if model.world_count == 1:
            for attr, start_attr in (
                ("body_world", "body_world_start"),
                ("joint_world", "joint_world_start"),
                ("shape_world", "shape_world_start"),
            ):
                arr = getattr(model, attr)
                arr_np = arr.numpy()
                if np.any(arr_np < 0):
                    arr_np[arr_np < 0] = 0
                    arr.assign(arr_np)
                    # Update world start indices
                    arr_start = getattr(model, start_attr)
                    arr_start_np = arr_start.numpy()
                    arr_start_np[0] = 0
                    arr_start_np[-2] = arr_start_np[-1]
                    arr_start.assign(arr_start_np)

        # ---------------------------------------------------------------------------
        # Pre-processing: absorb non-identity joint_X_c rotations into child body
        # frames so that Kamino sees aligned joint frames on both sides.
        #
        # Kamino's constraint system assumes a single joint frame X_j valid for both
        # the base (parent) and follower (child) bodies.  At q = 0 it requires
        #   q_base^{-1} * q_follower = identity
        # Newton, however, allows different parent / child joint-frame orientations
        # via joint_X_p and joint_X_c.  At q = 0 Newton's FK gives:
        #   q_follower = q_parent * q_pj * inv(q_cj)
        # so q_base^{-1} * q_follower = q_pj * inv(q_cj) which is generally not
        # identity.
        #
        # To fix this we apply a per-body correction rotation q_corr = q_cj * inv(q_pj)
        # (applied on the right) to each child body's frame:
        #   q_body_new = q_body_old * q_corr
        # This makes q_base^{-1} * q_follower_new = identity at q = 0, and the joint
        # rotation axis R(q_pj) * axis is preserved.
        #
        # All body-local quantities (CoM, inertia, shapes) are re-expressed in the
        # rotated frame, and downstream joint_X_p transforms are updated to account
        # for the parent body's frame change.
        # ---------------------------------------------------------------------------
        converted = convert_entity_local_transforms(model)
        # ----------------------------------------------------------------------------

        # Unpack converted quantities
        body_q = converted["body_q"]
        body_qd = converted["body_qd"]
        body_com = converted["body_com"]
        body_inertia = converted["body_inertia"]
        body_inv_inertia = converted["body_inv_inertia"]
        shape_transform = converted["shape_transform"]
        joint_X_p = converted["joint_X_p"]
        joint_X_c = converted["joint_X_c"]

        # Initialize materials manager
        materials_manager = MaterialManager()

        ###
        # Model Attributes
        ###

        # Initialize SizeKamino object, to be completed by helper functions
        model_size = SizeKamino(num_worlds=model.world_count)

        # Construct the model entities from the newton.Model instance
        with wp.ScopedDevice(device=model.device):
            # Per-world heterogeneous model info, to be completed by helper functions
            model_info = ModelKaminoInfo(num_worlds=model.world_count)

            # Per-world time
            model_time = TimeModel(
                dt=wp.zeros(shape=(model.world_count,), dtype=float32),
                inv_dt=wp.zeros(shape=(model.world_count,), dtype=float32),
            )

            # Per-world gravity
            model_gravity = GravityModel.from_newton(model)

            # Bodies
            model_bodies = convert_rigid_bodies(
                model, model_size, model_info, body_com, body_q, body_qd, body_inertia, body_inv_inertia
            )

            # Joints
            model_joints = convert_joints(model, model_size, model_info, body_com, joint_X_p, joint_X_c)

            # Geometries
            model_geoms = convert_geometries(
                model=model,
                model_size=model_size,
                model_bodies=model_bodies,
                materials_manager=materials_manager,
                shape_transform=shape_transform,
            )

            # Materials
            model_materials = materials_manager.make_materials_model()
            model_material_pairs = materials_manager.make_material_pairs_model()

        ###
        # Post-processing
        ###

        # Modify the model's body COM and shape transform properties in-place to convert from body-frame-relative
        # NOTE: These are modified only so that the visualizer correctly
        # shows the shape poses, joints frames and body inertial properties
        if overwrite_source_model:
            wp.copy(model.body_com, body_com)
            wp.copy(model.body_inertia, body_inertia)
            wp.copy(model.shape_transform, shape_transform)
            wp.copy(model.joint_X_p, joint_X_p)
            wp.copy(model.joint_X_c, joint_X_c)

        # Construct and return the new ModelKamino instance
        return ModelKamino(
            _model=model,
            _device=model.device,
            _requires_grad=model.requires_grad,
            size=model_size,
            info=model_info,
            time=model_time,
            gravity=model_gravity,
            bodies=model_bodies,
            joints=model_joints,
            geoms=model_geoms,
            materials=model_materials,
            material_pairs=model_material_pairs,
        )
