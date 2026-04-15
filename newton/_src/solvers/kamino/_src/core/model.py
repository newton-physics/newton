# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the model container of Kamino."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

# Newton imports
from .....geometry import GeoType, ShapeFlags
from .....sim import JointTargetMode, JointType, Model

# Kamino imports
from .bodies import RigidBodiesData, RigidBodiesModel, convert_geom_offset_origin_to_com
from .control import ControlKamino
from .conversions import (
    compute_required_contact_capacity,
    convert_entity_local_transforms,
)
from .data import DataKamino, DataKaminoInfo
from .geometry import GeometriesData, GeometriesModel
from .gravity import GravityModel
from .joints import (
    JOINT_DQMAX,
    JOINT_QMAX,
    JOINT_QMIN,
    JOINT_TAUMAX,
    JointActuationType,
    JointDoFType,
    JointsData,
    JointsModel,
)
from .materials import MaterialDescriptor, MaterialManager, MaterialPairsModel, MaterialsModel
from .size import SizeKamino
from .state import StateKamino
from .time import TimeData, TimeModel
from .types import float32, int32, mat33f, transformf, vec2i, vec3f, vec4f, vec6f

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
    Shape of ``(num_worlds,)`` and type :class:`int`.
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
                q_j_ref=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
                dq_j_ref=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
                tau_j_ref=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=float32, requires_grad=requires_grad),
            )

        # Return the constructed control container
        return control

    @staticmethod
    def from_newton(model: Model) -> ModelKamino:
        """
        Finalizes the :class:`ModelKamino` from an existing instance of :class:`newton.Model`.
        """
        # Ensure the base model is valid
        if model is None:
            raise ValueError("Cannot finalize ModelKamino from a None newton.Model instance.")
        elif not isinstance(model, Model):
            raise TypeError("Cannot finalize ModelKamino from an invalid newton.Model instance.")

        # Single-world Newton models may have world index -1 (unassigned).
        # Normalize to 0 so downstream world-based grouping works correctly.
        if model.world_count == 1:
            for attr in ("body_world", "joint_world", "shape_world"):
                arr = getattr(model, attr)
                arr_np = arr.numpy()
                if np.any(arr_np < 0):
                    arr_np[arr_np < 0] = 0
                    arr.assign(arr_np)

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

        def _to_wpq(q):
            return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        def _compute_entity_indices_wrt_world(entity_world: wp.array) -> np.ndarray:
            wid_np = entity_world.numpy()
            eid_np = np.zeros_like(wid_np)
            for e in range(wid_np.size):
                eid_np[e] = np.sum(wid_np[:e] == wid_np[e])
            return eid_np

        def _compute_num_entities_per_world(entity_world: wp.array, num_worlds: int) -> np.ndarray:
            wid_np = entity_world.numpy()
            counts = np.zeros(num_worlds, dtype=int)
            for w in range(num_worlds):
                counts[w] = np.sum(wid_np == w)
            return counts

        # Compute the entity indices of each body w.r.t the corresponding world
        body_bid_np = _compute_entity_indices_wrt_world(model.body_world)
        joint_jid_np = _compute_entity_indices_wrt_world(model.joint_world)
        shape_sid_np = _compute_entity_indices_wrt_world(model.shape_world)

        # Compute the number of entities per world
        num_bodies_np = _compute_num_entities_per_world(model.body_world, model.world_count)
        num_joints_np = _compute_num_entities_per_world(model.joint_world, model.world_count)
        num_shapes_np = _compute_num_entities_per_world(model.shape_world, model.world_count)

        # Compute body coord/DoF counts per world
        num_body_dofs_np = num_bodies_np * 6

        # Compute joint coord/DoF/constraint counts per world
        num_passive_joints_np = np.zeros((model.world_count,), dtype=int)
        num_actuated_joints_np = np.zeros((model.world_count,), dtype=int)
        num_dynamic_joints_np = np.zeros((model.world_count,), dtype=int)
        num_joint_coords_np = np.zeros((model.world_count,), dtype=int)
        num_joint_dofs_np = np.zeros((model.world_count,), dtype=int)
        num_joint_passive_coords_np = np.zeros((model.world_count,), dtype=int)
        num_joint_passive_dofs_np = np.zeros((model.world_count,), dtype=int)
        num_joint_actuated_coords_np = np.zeros((model.world_count,), dtype=int)
        num_joint_actuated_dofs_np = np.zeros((model.world_count,), dtype=int)
        num_joint_cts_np = np.zeros((model.world_count,), dtype=int)
        num_joint_dynamic_cts_np = np.zeros((model.world_count,), dtype=int)
        num_joint_kinematic_cts_np = np.zeros((model.world_count,), dtype=int)

        # TODO
        joint_dof_type_np = np.zeros((model.joint_count,), dtype=int)
        joint_act_type_np = np.zeros((model.joint_count,), dtype=int)
        joint_B_r_Bj_np = np.zeros((model.joint_count, 3), dtype=float)
        joint_F_r_Fj_np = np.zeros((model.joint_count, 3), dtype=float)
        joint_X_j_np = np.zeros((model.joint_count, 9), dtype=float)
        joint_num_coords_np = np.zeros((model.joint_count,), dtype=int)
        joint_num_dofs_np = np.zeros((model.joint_count,), dtype=int)
        joint_num_cts_np = np.zeros((model.joint_count,), dtype=int)
        joint_num_dynamic_cts_np = np.zeros((model.joint_count,), dtype=int)
        joint_num_kinematic_cts_np = np.zeros((model.joint_count,), dtype=int)
        joint_coord_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_dofs_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_actuated_coord_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_actuated_dofs_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_passive_coord_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_passive_dofs_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_cts_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_dynamic_cts_start_np = np.zeros((model.joint_count,), dtype=int)
        joint_kinematic_cts_start_np = np.zeros((model.joint_count,), dtype=int)

        # Unpack converted quantities
        body_q_np = converted["body_q"]
        body_qd_np = converted["body_qd"]
        body_com_np = converted["body_com"]
        body_inertia_np = converted["body_inertia"]
        body_inv_inertia_np = converted["body_inv_inertia"]
        shape_transform_np = converted["shape_transform"]
        joint_X_p_np = converted["joint_X_p"]
        joint_X_c_np = converted["joint_X_c"]

        # TODO
        joint_wid_np: np.ndarray = model.joint_world.numpy().copy()
        joint_type_np: np.ndarray = model.joint_type.numpy().copy()
        joint_target_mode_np: np.ndarray = model.joint_target_mode.numpy().copy()
        joint_parent_np: np.ndarray = model.joint_parent.numpy().copy()
        joint_child_np: np.ndarray = model.joint_child.numpy().copy()
        joint_axis_np: np.ndarray = model.joint_axis.numpy().copy()
        joint_dof_dim_np: np.ndarray = model.joint_dof_dim.numpy().copy()
        joint_q_start_np: np.ndarray = model.joint_q_start.numpy().copy()
        joint_qd_start_np: np.ndarray = model.joint_qd_start.numpy().copy()
        joint_limit_lower_np: np.ndarray = model.joint_limit_lower.numpy().copy()
        joint_limit_upper_np: np.ndarray = model.joint_limit_upper.numpy().copy()
        joint_velocity_limit_np = model.joint_velocity_limit.numpy().copy()
        joint_effort_limit_np = model.joint_effort_limit.numpy().copy()
        joint_armature_np: np.ndarray = model.joint_armature.numpy().copy()
        joint_friction_np: np.ndarray = model.joint_friction.numpy().copy()
        joint_target_ke_np: np.ndarray = model.joint_target_ke.numpy().copy()
        joint_target_kd_np: np.ndarray = model.joint_target_kd.numpy().copy()

        for j in range(model.joint_count):
            # TODO
            wid_j = joint_wid_np[j]

            # TODO
            joint_coord_start_np[j] = num_joint_coords_np[wid_j]
            joint_dofs_start_np[j] = num_joint_dofs_np[wid_j]
            joint_actuated_coord_start_np[j] = num_joint_actuated_coords_np[wid_j]
            joint_actuated_dofs_start_np[j] = num_joint_actuated_dofs_np[wid_j]
            joint_passive_coord_start_np[j] = num_joint_passive_coords_np[wid_j]
            joint_passive_dofs_start_np[j] = num_joint_passive_dofs_np[wid_j]
            joint_cts_start_np[j] = num_joint_cts_np[wid_j]
            joint_dynamic_cts_start_np[j] = num_joint_dynamic_cts_np[wid_j]
            joint_kinematic_cts_start_np[j] = num_joint_kinematic_cts_np[wid_j]

            # TODO
            type_j = int(joint_type_np[j])
            dof_dim_j = (int(joint_dof_dim_np[j][0]), int(joint_dof_dim_np[j][1]))
            q_count_j = int(joint_q_start_np[j + 1] - joint_q_start_np[j])
            qd_count_j = int(joint_qd_start_np[j + 1] - joint_qd_start_np[j])
            limit_upper_j = joint_limit_upper_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]].astype(float)
            limit_lower_j = joint_limit_lower_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]].astype(float)
            dof_type_j = JointDoFType.from_newton(
                JointType(type_j), q_count_j, qd_count_j, dof_dim_j, limit_lower_j, limit_upper_j
            )

            # TODO
            ncoords_j = dof_type_j.num_coords
            ndofs_j = dof_type_j.num_dofs
            ncts_j = dof_type_j.num_cts

            # TODO
            joint_dof_type_np[j] = dof_type_j.value
            num_joint_coords_np[wid_j] += ncoords_j
            num_joint_dofs_np[wid_j] += ndofs_j
            joint_num_coords_np[j] = ncoords_j
            joint_num_dofs_np[j] = ndofs_j

            # TODO
            dofs_start_j = joint_qd_start_np[j]
            dof_axes_j = joint_axis_np[dofs_start_j : dofs_start_j + ndofs_j]
            joint_dofs_target_mode_j = joint_target_mode_np[dofs_start_j : dofs_start_j + ndofs_j]
            act_type_j = JointActuationType.from_newton(
                JointTargetMode(max(joint_dofs_target_mode_j) if len(joint_dofs_target_mode_j) > 0 else 0)
            )
            joint_act_type_np[j] = act_type_j.value

            # Infer if the joint requires dynamic constraints
            is_dynamic_j = False
            if ndofs_j > 0:
                a_j = joint_armature_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]]
                b_j = joint_friction_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]]
                ke_j = joint_target_ke_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]]
                kd_j = joint_target_kd_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]]
                a_j_min = float(a_j.min())
                b_j_min = float(b_j.min())
                ke_j_min = float(ke_j.min())
                kd_j_min = float(kd_j.min())
                a_j_max = float(a_j.max())
                b_j_max = float(b_j.max())
                ke_j_max = float(ke_j.max())
                kd_j_max = float(kd_j.max())
                if (a_j_min < 0.0) or (b_j_min < 0.0) or (ke_j_min < 0.0) or (kd_j_min < 0.0):
                    raise ValueError(
                        f"Joint {j} in world {wid_j} has negative armature, friction "
                        "or target stiffness/damping values, which is not supported."
                    )
                if (a_j_min < a_j_max) or (b_j_min < b_j_max) or (ke_j_min < ke_j_max) or (kd_j_min < kd_j_max):
                    raise ValueError(
                        f"Joint {j} in world {wid_j} has non-constant armature, friction "
                        "or target stiffness/damping values, which is not supported."
                    )
                is_dynamic_j = (a_j_max > 0.0) or (b_j_max > 0.0) or (ke_j_max > 0.0) or (kd_j_max > 0.0)

            # TODO
            if is_dynamic_j:
                joint_num_dynamic_cts_np[j] = ndofs_j
                joint_dynamic_cts_start_np[j] = num_joint_dynamic_cts_np[wid_j]
                num_joint_dynamic_cts_np[wid_j] += ndofs_j
                num_joint_cts_np[wid_j] += ndofs_j
                num_dynamic_joints_np[wid_j] += 1
            else:
                joint_dynamic_cts_start_np[j] = -1

            # TODO
            num_joint_cts_np[wid_j] += ncts_j
            num_joint_kinematic_cts_np[wid_j] += ncts_j
            if act_type_j > JointActuationType.PASSIVE:
                num_actuated_joints_np[wid_j] += 1
                num_joint_actuated_coords_np[wid_j] += ncoords_j
                num_joint_actuated_dofs_np[wid_j] += ndofs_j
                joint_passive_coord_start_np[j] = -1
                joint_passive_dofs_start_np[j] = -1
            else:
                num_passive_joints_np[wid_j] += 1
                num_joint_passive_coords_np[wid_j] += ncoords_j
                num_joint_passive_dofs_np[wid_j] += ndofs_j
                joint_actuated_coord_start_np[j] = -1
                joint_actuated_dofs_start_np[j] = -1
            joint_num_kinematic_cts_np[j] = ncts_j
            joint_num_cts_np[j] = joint_num_dynamic_cts_np[j] + joint_num_kinematic_cts_np[j]

            # TODO
            parent_bid = joint_parent_np[j]
            p_r_p_com = vec3f(body_com_np[parent_bid]) if parent_bid >= 0 else vec3f(0.0, 0.0, 0.0)
            c_r_c_com = vec3f(body_com_np[joint_child_np[j]])
            X_p_j = transformf(*joint_X_p_np[j, :])
            X_c_j = transformf(*joint_X_c_np[j, :])
            q_p_j = wp.transform_get_rotation(X_p_j)
            p_r_p_j = wp.transform_get_translation(X_p_j)
            c_r_c_j = wp.transform_get_translation(X_c_j)

            # TODO
            R_axis_j = JointDoFType.axes_matrix_from_joint_type(dof_type_j, dof_dim_j, dof_axes_j)
            B_r_Bj = p_r_p_j - p_r_p_com
            F_r_Fj = c_r_c_j - c_r_c_com
            X_j = wp.quat_to_matrix(q_p_j) @ R_axis_j
            joint_B_r_Bj_np[j, :] = B_r_Bj
            joint_F_r_Fj_np[j, :] = F_r_Fj
            joint_X_j_np[j, :] = X_j

        # Convert joint limits and effort/velocity limits to np.float32 and clip to supported ranges
        np.clip(a=joint_limit_lower_np, a_min=JOINT_QMIN, a_max=JOINT_QMAX, out=joint_limit_lower_np)
        np.clip(a=joint_limit_upper_np, a_min=JOINT_QMIN, a_max=JOINT_QMAX, out=joint_limit_upper_np)
        np.clip(a=joint_velocity_limit_np, a_min=-JOINT_DQMAX, a_max=JOINT_DQMAX, out=joint_velocity_limit_np)
        np.clip(a=joint_effort_limit_np, a_min=-JOINT_TAUMAX, a_max=JOINT_TAUMAX, out=joint_effort_limit_np)

        # Set up materials
        materials_manager = MaterialManager()
        default_material = materials_manager.materials[0]
        shape_friction_np = model.shape_material_mu.numpy().tolist()
        shape_restitution_np = model.shape_material_restitution.numpy().tolist()
        geom_material_np = np.zeros((model.shape_count,), dtype=int)
        # TODO: Integrate world index for shape material
        # shape_world_np = model.shape_world.numpy()
        material_param_indices: dict[tuple[float, float], int] = {}
        # Adding default material from material manager, making sure the values undergo the same
        # transformation as any material parameters in the Newton model (conversion to np.float32)
        default_mu = float(np.float32(default_material.static_friction))
        default_restitution = float(np.float32(default_material.restitution))
        material_param_indices[(default_mu, default_restitution)] = 0
        for s in range(model.shape_count):
            # Check if material with these parameters already exists
            material_desc = (shape_friction_np[s], shape_restitution_np[s])
            if material_desc in material_param_indices:
                material_id = material_param_indices[material_desc]
            else:
                material = MaterialDescriptor(
                    name=f"{model.shape_label[s]}_material",
                    restitution=shape_restitution_np[s],
                    static_friction=shape_friction_np[s],
                    dynamic_friction=shape_friction_np[s],
                    # wid=shape_world_np[s],
                )
                material_id = materials_manager.register(material)
                material_param_indices[material_desc] = material_id
            geom_material_np[s] = material_id

        # Store Newton shape type and scale directly (no Kamino conversion needed).
        # The geoms.type array stores GeoType values, and geoms.params stores
        # Newton's shape_scale convention (half-extents, half-heights, etc.)
        # padded to vec4f.  Plane shapes are special: params store the plane
        # normal (derived from the shape transform) instead of scale.
        shape_type_np = model.shape_type.numpy()
        shape_scale_np = model.shape_scale.numpy()
        shape_flags_np = model.shape_flags.numpy()
        geom_shape_collision_group_np = model.shape_collision_group.numpy()
        geom_shape_params_np = np.zeros((model.shape_count, 4), dtype=float)
        model_num_collidable_geoms = 0
        for s in range(model.shape_count):
            scale = shape_scale_np[s]
            geom_shape_params_np[s, 0] = scale[0]
            geom_shape_params_np[s, 1] = scale[1]
            geom_shape_params_np[s, 2] = scale[2]
            if (shape_flags_np[s] & ShapeFlags.COLLIDE_SHAPES) != 0 and geom_shape_collision_group_np[s] > 0:
                model_num_collidable_geoms += 1
            else:
                geom_material_np[s] = -1  # Ensure non-collidable geoms no material assigned

        # Plane normals: derive from the shape transform rotation (local Z-axis)
        # and store in params since Newton's shape_scale doesn't encode plane normals.
        for s in range(model.shape_count):
            if shape_type_np[s] == GeoType.PLANE:
                tf = shape_transform_np[s, :]
                q_rot = _to_wpq(np.array([tf[3], tf[4], tf[5], tf[6]]))
                normal = wp.quat_rotate(q_rot, vec3f(0.0, 0.0, 1.0))
                geom_shape_params_np[s, 0] = float(normal[0])
                geom_shape_params_np[s, 1] = float(normal[1])
                geom_shape_params_np[s, 2] = float(normal[2])
                geom_shape_params_np[s, 3] = 0.0

        # Compute total number of required contacts per world
        if model.rigid_contact_max > 0:
            model_min_contacts = int(model.rigid_contact_max)
            min_contacts_per_world = model.rigid_contact_max // model.world_count
            world_min_contacts = [min_contacts_per_world] * model.world_count
        else:
            model_min_contacts, world_min_contacts = compute_required_contact_capacity(model)

        # Compute offsets per world
        world_shape_offset_np = np.zeros((model.world_count,), dtype=int)
        world_body_offset_np = np.zeros((model.world_count,), dtype=int)
        world_body_dof_offset_np = np.zeros((model.world_count,), dtype=int)
        world_joint_offset_np = np.zeros((model.world_count,), dtype=int)
        world_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
        world_joint_dof_offset_np = np.zeros((model.world_count,), dtype=int)
        world_actuated_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
        world_actuated_joint_dofs_offset_np = np.zeros((model.world_count,), dtype=int)
        world_passive_joint_coord_offset_np = np.zeros((model.world_count,), dtype=int)
        world_passive_joint_dofs_offset_np = np.zeros((model.world_count,), dtype=int)
        world_joint_cts_offset_np = np.zeros((model.world_count,), dtype=int)
        world_joint_dynamic_cts_offset_np = np.zeros((model.world_count,), dtype=int)
        world_joint_kinematic_cts_offset_np = np.zeros((model.world_count,), dtype=int)

        for w in range(1, model.world_count):
            world_shape_offset_np[w] = world_shape_offset_np[w - 1] + num_shapes_np[w - 1]
            world_body_offset_np[w] = world_body_offset_np[w - 1] + num_bodies_np[w - 1]
            world_body_dof_offset_np[w] = world_body_dof_offset_np[w - 1] + num_body_dofs_np[w - 1]
            world_joint_offset_np[w] = world_joint_offset_np[w - 1] + num_joints_np[w - 1]
            world_joint_coord_offset_np[w] = world_joint_coord_offset_np[w - 1] + num_joint_coords_np[w - 1]
            world_joint_dof_offset_np[w] = world_joint_dof_offset_np[w - 1] + num_joint_dofs_np[w - 1]
            world_actuated_joint_coord_offset_np[w] = (
                world_actuated_joint_coord_offset_np[w - 1] + num_joint_actuated_coords_np[w - 1]
            )
            world_actuated_joint_dofs_offset_np[w] = (
                world_actuated_joint_dofs_offset_np[w - 1] + num_joint_actuated_dofs_np[w - 1]
            )
            world_passive_joint_coord_offset_np[w] = (
                world_passive_joint_coord_offset_np[w - 1] + num_joint_passive_coords_np[w - 1]
            )
            world_passive_joint_dofs_offset_np[w] = (
                world_passive_joint_dofs_offset_np[w - 1] + num_joint_passive_dofs_np[w - 1]
            )
            world_joint_cts_offset_np[w] = world_joint_cts_offset_np[w - 1] + num_joint_cts_np[w - 1]
            world_joint_dynamic_cts_offset_np[w] = (
                world_joint_dynamic_cts_offset_np[w - 1] + num_joint_dynamic_cts_np[w - 1]
            )
            world_joint_kinematic_cts_offset_np[w] = (
                world_joint_kinematic_cts_offset_np[w - 1] + num_joint_kinematic_cts_np[w - 1]
            )

        # Determine the base body and joint indices per world
        base_body_idx_np = np.full((model.world_count,), -1, dtype=int)
        base_joint_idx_np = np.full((model.world_count,), -1, dtype=int)
        body_world_np = model.body_world.numpy()
        joint_world_np = model.joint_world.numpy()
        body_world_start_np = model.body_world_start.numpy()

        # Check for articulations
        if model.articulation_count > 0:
            articulation_start_np = model.articulation_start.numpy()
            articulation_world_np = model.articulation_world.numpy()
            # For each articulation, assign its base body and joint to the corresponding world
            # NOTE: We only assign the first articulation found in each world
            for aid in range(model.articulation_count):
                wid = articulation_world_np[aid]
                base_joint = articulation_start_np[aid]
                base_body = joint_child_np[base_joint]
                if base_body_idx_np[wid] == -1 and base_joint_idx_np[wid] == -1:
                    base_body_idx_np[wid] = base_body
                    base_joint_idx_np[wid] = base_joint

        # Check for root joint (i.e. joint with no parent body (= -1))
        elif model.joint_count > 0:
            # TODO: How to handle no free joint being defined?
            # Create a list of joint indices with parent body == -1 for each world
            world_parent_joints: dict[int, list[int]] = {w: [] for w in range(model.world_count)}
            for j in range(model.joint_count):
                wid_j = joint_world_np[j]
                parent_j = joint_parent_np[j]
                if parent_j == -1:
                    world_parent_joints[wid_j].append(j)
            # For each world, assign the base body and joint based on the first joint with parent == -1,
            # If no joint with parent == -1 is found in a world, then assign the first body as base
            # If multiple joints with parent == -1 are found in a world, then assign the first one as the base
            for w in range(model.world_count):
                if len(world_parent_joints[w]) > 0:
                    j = world_parent_joints[w][0]
                    base_joint_idx_np[w] = j
                    base_body_idx_np[w] = int(joint_child_np[j])
                else:
                    base_body_idx_np[w] = int(body_world_start_np[w])
                    base_joint_idx_np[w] = -1

        # Fall-back: first body and joint in the world
        else:
            for w in range(model.world_count):
                # Base body: first body in the world
                for b in range(model.body_count):
                    if body_world_np[b] == w:
                        base_body_idx_np[w] = b
                        break
                # Base joint: first joint in the world
                for j in range(model.joint_count):
                    if joint_world_np[j] == w:
                        base_joint_idx_np[w] = j
                        break

        # Ensure that all worlds have a base body assigned
        for w in range(model.world_count):
            if base_body_idx_np[w] == -1:
                raise ValueError(f"World {w} does not have a base body assigned (index is -1).")

        # Construct per-world inertial summaries
        mass_min_np = np.zeros((model.world_count,), dtype=float)
        mass_max_np = np.zeros((model.world_count,), dtype=float)
        mass_total_np = np.zeros((model.world_count,), dtype=float)
        inertia_total_np = np.zeros((model.world_count,), dtype=float)
        body_mass_np = model.body_mass.numpy()
        for w in range(model.world_count):
            masses_w = []
            for b in range(model.body_count):
                if body_world_np[b] == w:
                    mass_b = body_mass_np[b]
                    masses_w.append(mass_b)
                    mass_total_np[w] += mass_b
                    inertia_total_np[w] += 3.0 * mass_b + body_inertia_np[b].diagonal().sum()
            mass_min_np[w] = min(masses_w)
            mass_max_np[w] = max(masses_w)

        # Construct the per-material and per-material-pair properties
        materials_rest = [materials_manager.restitution_vector()]
        materials_static_fric = [materials_manager.static_friction_vector()]
        materials_dynamic_fric = [materials_manager.dynamic_friction_vector()]
        mpairs_rest = [materials_manager.restitution_matrix()]
        mpairs_static_fric = [materials_manager.static_friction_matrix()]
        mpairs_dynamic_fric = [materials_manager.dynamic_friction_matrix()]

        # model.body_q stores body-origin world poses, but Kamino expects
        # COM world poses (joint attachment vectors are COM-relative).
        q_i_0_np = np.empty((model.body_count, 7), dtype=np.float32)
        for i in range(model.body_count):
            pos = body_q_np[i, :3]
            rot = wp.quatf(*body_q_np[i, 3:7])
            com_world = pos + np.array(wp.quat_rotate(rot, wp.vec3f(*body_com_np[i])))
            q_i_0_np[i, :3] = com_world
            q_i_0_np[i, 3:7] = body_q_np[i, 3:7]

        ###
        # Model Attributes
        ###

        # Construct SizeKamino from the newton.Model instance
        model_size = SizeKamino(
            num_worlds=model.world_count,
            sum_of_num_bodies=int(num_bodies_np.sum()),
            max_of_num_bodies=int(num_bodies_np.max()),
            sum_of_num_joints=int(num_joints_np.sum()),
            max_of_num_joints=int(num_joints_np.max()),
            sum_of_num_passive_joints=int(num_passive_joints_np.sum()),
            max_of_num_passive_joints=int(num_passive_joints_np.max()),
            sum_of_num_actuated_joints=int(num_actuated_joints_np.sum()),
            max_of_num_actuated_joints=int(num_actuated_joints_np.max()),
            sum_of_num_dynamic_joints=int(num_dynamic_joints_np.sum()),
            max_of_num_dynamic_joints=int(num_dynamic_joints_np.max()),
            sum_of_num_geoms=int(num_shapes_np.sum()),
            max_of_num_geoms=int(num_shapes_np.max()),
            sum_of_num_materials=materials_manager.num_materials,
            max_of_num_materials=materials_manager.num_materials,
            sum_of_num_material_pairs=materials_manager.num_material_pairs,
            max_of_num_material_pairs=materials_manager.num_material_pairs,
            sum_of_num_body_dofs=int(num_body_dofs_np.sum()),
            max_of_num_body_dofs=int(num_body_dofs_np.max()),
            sum_of_num_joint_coords=int(num_joint_coords_np.sum()),
            max_of_num_joint_coords=int(num_joint_coords_np.max()),
            sum_of_num_joint_dofs=int(num_joint_dofs_np.sum()),
            max_of_num_joint_dofs=int(num_joint_dofs_np.max()),
            sum_of_num_passive_joint_coords=int(num_joint_passive_coords_np.sum()),
            max_of_num_passive_joint_coords=int(num_joint_passive_coords_np.max()),
            sum_of_num_passive_joint_dofs=int(num_joint_passive_dofs_np.sum()),
            max_of_num_passive_joint_dofs=int(num_joint_passive_dofs_np.max()),
            sum_of_num_actuated_joint_coords=int(num_joint_actuated_coords_np.sum()),
            max_of_num_actuated_joint_coords=int(num_joint_actuated_coords_np.max()),
            sum_of_num_actuated_joint_dofs=int(num_joint_actuated_dofs_np.sum()),
            max_of_num_actuated_joint_dofs=int(num_joint_actuated_dofs_np.max()),
            sum_of_num_joint_cts=int(num_joint_cts_np.sum()),
            max_of_num_joint_cts=int(num_joint_cts_np.max()),
            sum_of_num_dynamic_joint_cts=int(num_joint_dynamic_cts_np.sum()),
            max_of_num_dynamic_joint_cts=int(num_joint_dynamic_cts_np.max()),
            sum_of_num_kinematic_joint_cts=int(num_joint_kinematic_cts_np.sum()),
            max_of_num_kinematic_joint_cts=int(num_joint_kinematic_cts_np.max()),
            sum_of_max_total_cts=int(num_joint_cts_np.sum()),
            max_of_max_total_cts=int(num_joint_cts_np.max()),
        )

        # Construct the model entities from the newton.Model instance
        with wp.ScopedDevice(device=model.device):
            # Per-world heterogeneous model info
            model_info = ModelKaminoInfo(
                num_worlds=model.world_count,
                num_bodies=wp.array(num_bodies_np, dtype=int32),
                num_joints=wp.array(num_joints_np, dtype=int32),
                num_passive_joints=wp.array(num_passive_joints_np, dtype=int32),
                num_actuated_joints=wp.array(num_actuated_joints_np, dtype=int32),
                num_dynamic_joints=wp.array(num_dynamic_joints_np, dtype=int32),
                num_geoms=wp.array(num_shapes_np, dtype=int32),
                num_body_dofs=wp.array(num_body_dofs_np, dtype=int32),
                num_joint_coords=wp.array(num_joint_coords_np, dtype=int32),
                num_joint_dofs=wp.array(num_joint_dofs_np, dtype=int32),
                num_passive_joint_coords=wp.array(num_joint_passive_coords_np, dtype=int32),
                num_passive_joint_dofs=wp.array(num_joint_passive_dofs_np, dtype=int32),
                num_actuated_joint_coords=wp.array(num_joint_actuated_coords_np, dtype=int32),
                num_actuated_joint_dofs=wp.array(num_joint_actuated_dofs_np, dtype=int32),
                num_joint_cts=wp.array(num_joint_cts_np, dtype=int32),
                num_joint_dynamic_cts=wp.array(num_joint_dynamic_cts_np, dtype=int32),
                num_joint_kinematic_cts=wp.array(num_joint_kinematic_cts_np, dtype=int32),
                bodies_offset=wp.array(world_body_offset_np, dtype=int32),
                joints_offset=wp.array(world_joint_offset_np, dtype=int32),
                geoms_offset=wp.array(world_shape_offset_np, dtype=int32),
                body_dofs_offset=wp.array(world_body_dof_offset_np, dtype=int32),
                joint_coords_offset=wp.array(world_joint_coord_offset_np, dtype=int32),
                joint_dofs_offset=wp.array(world_joint_dof_offset_np, dtype=int32),
                joint_passive_coords_offset=wp.array(world_passive_joint_coord_offset_np, dtype=int32),
                joint_passive_dofs_offset=wp.array(world_passive_joint_dofs_offset_np, dtype=int32),
                joint_actuated_coords_offset=wp.array(world_actuated_joint_coord_offset_np, dtype=int32),
                joint_actuated_dofs_offset=wp.array(world_actuated_joint_dofs_offset_np, dtype=int32),
                joint_cts_offset=wp.array(world_joint_cts_offset_np, dtype=int32),
                joint_dynamic_cts_offset=wp.array(world_joint_dynamic_cts_offset_np, dtype=int32),
                joint_kinematic_cts_offset=wp.array(world_joint_kinematic_cts_offset_np, dtype=int32),
                base_body_index=wp.array(base_body_idx_np, dtype=int32),
                base_joint_index=wp.array(base_joint_idx_np, dtype=int32),
                mass_min=wp.array(mass_min_np, dtype=float32),
                mass_max=wp.array(mass_max_np, dtype=float32),
                mass_total=wp.array(mass_total_np, dtype=float32),
                inertia_total=wp.array(inertia_total_np, dtype=float32),
            )

            # Per-world time
            model_time = TimeModel(
                dt=wp.zeros(shape=(model.world_count,), dtype=float32),
                inv_dt=wp.zeros(shape=(model.world_count,), dtype=float32),
            )

            # Per-world gravity
            model_gravity = GravityModel.from_newton(model)

            # Bodies
            model_bodies = RigidBodiesModel(
                num_bodies=model.body_count,
                label=model.body_label,
                wid=model.body_world,
                bid=wp.array(body_bid_np, dtype=int32),  # TODO: Remove
                m_i=model.body_mass,
                inv_m_i=model.body_inv_mass,
                i_r_com_i=wp.array(body_com_np, dtype=vec3f),
                i_I_i=wp.array(body_inertia_np, dtype=mat33f),
                inv_i_I_i=wp.array(body_inv_inertia_np, dtype=mat33f),
                q_i_0=wp.array(q_i_0_np, dtype=wp.transformf),
                u_i_0=wp.array(body_qd_np, dtype=vec6f),
            )

            # Joints
            model_joints = JointsModel(
                num_joints=model.joint_count,
                label=model.joint_label,
                wid=model.joint_world,
                jid=wp.array(joint_jid_np, dtype=int32),  # TODO: Remove
                dof_type=wp.array(joint_dof_type_np, dtype=int32),
                act_type=wp.array(joint_act_type_np, dtype=int32),
                bid_B=model.joint_parent,
                bid_F=model.joint_child,
                B_r_Bj=wp.array(joint_B_r_Bj_np, dtype=wp.vec3f),
                F_r_Fj=wp.array(joint_F_r_Fj_np, dtype=wp.vec3f),
                X_j=wp.array(joint_X_j_np.reshape((model.joint_count, 3, 3)), dtype=wp.mat33f),
                q_j_min=wp.array(joint_limit_lower_np, dtype=float32),
                q_j_max=wp.array(joint_limit_upper_np, dtype=float32),
                dq_j_max=wp.array(joint_velocity_limit_np, dtype=float32),
                tau_j_max=wp.array(joint_effort_limit_np, dtype=float32),
                a_j=model.joint_armature,
                b_j=model.joint_friction,  # TODO: Is this the right attribute?
                k_p_j=model.joint_target_ke,
                k_d_j=model.joint_target_kd,
                q_j_0=model.joint_q,
                dq_j_0=model.joint_qd,
                num_coords=wp.array(joint_num_coords_np, dtype=int32),
                num_dofs=wp.array(joint_num_dofs_np, dtype=int32),
                num_cts=wp.array(joint_num_cts_np, dtype=int32),
                num_dynamic_cts=wp.array(joint_num_dynamic_cts_np, dtype=int32),
                num_kinematic_cts=wp.array(joint_num_kinematic_cts_np, dtype=int32),
                coords_offset=wp.array(joint_coord_start_np, dtype=int32),
                dofs_offset=wp.array(joint_dofs_start_np, dtype=int32),
                passive_coords_offset=wp.array(joint_passive_coord_start_np, dtype=int32),
                passive_dofs_offset=wp.array(joint_passive_dofs_start_np, dtype=int32),
                actuated_coords_offset=wp.array(joint_actuated_coord_start_np, dtype=int32),
                actuated_dofs_offset=wp.array(joint_actuated_dofs_start_np, dtype=int32),
                cts_offset=wp.array(joint_cts_start_np, dtype=int32),
                dynamic_cts_offset=wp.array(joint_dynamic_cts_start_np, dtype=int32),
                kinematic_cts_offset=wp.array(joint_kinematic_cts_start_np, dtype=int32),
            )

            # Geometries
            model_geoms = GeometriesModel(
                num_geoms=model.shape_count,
                num_collidable=model_num_collidable_geoms,
                num_collidable_pairs=model.shape_contact_pair_count,
                num_excluded_pairs=len(model.shape_collision_filter_pairs),
                model_minimum_contacts=model_min_contacts,
                world_minimum_contacts=world_min_contacts,
                label=model.shape_label,
                wid=model.shape_world,
                gid=wp.array(shape_sid_np, dtype=int32),  # TODO: Remove
                bid=model.shape_body,
                type=wp.array(shape_type_np, dtype=int32),
                flags=model.shape_flags,
                ptr=model.shape_source_ptr,
                params=wp.array(geom_shape_params_np, dtype=vec4f),
                offset=wp.zeros_like(model.shape_transform),
                material=wp.array(geom_material_np, dtype=int32),
                group=model.shape_collision_group,
                gap=model.shape_gap,
                margin=model.shape_margin,
                collidable_pairs=model.shape_contact_pairs,
                excluded_pairs=wp.array(sorted(model.shape_collision_filter_pairs), dtype=vec2i),
                # Mesh / heightfield data pass-through from Newton model
                heightfield_index=model.shape_heightfield_index,
                heightfield_data=model.heightfield_data,
                heightfield_elevations=model.heightfield_elevations,
                collision_aabb_lower=model.shape_collision_aabb_lower,
                collision_aabb_upper=model.shape_collision_aabb_upper,
                voxel_resolution=model._shape_voxel_resolution,
                collision_radius=model.shape_collision_radius,
            )

            # Per-material properties
            model_materials = MaterialsModel(
                num_materials=model_size.sum_of_num_materials,
                restitution=wp.array(materials_rest[0], dtype=float32),
                static_friction=wp.array(materials_static_fric[0], dtype=float32),
                dynamic_friction=wp.array(materials_dynamic_fric[0], dtype=float32),
            )

            # Per-material-pair properties
            model_material_pairs = MaterialPairsModel(
                num_material_pairs=model_size.sum_of_num_material_pairs,
                restitution=wp.array(mpairs_rest[0], dtype=float32),
                static_friction=wp.array(mpairs_static_fric[0], dtype=float32),
                dynamic_friction=wp.array(mpairs_dynamic_fric[0], dtype=float32),
            )

        ###
        # Post-processing
        ###

        # Modify the model's body COM and shape transform properties in-place to convert from body-frame-relative
        # NOTE: These are modified only so that the visualizer correctly
        # shows the shape poses, joints frames and body inertial properties
        model.body_com.assign(body_com_np)
        model.body_inertia.assign(body_inertia_np)
        model.shape_transform.assign(shape_transform_np)
        model.joint_X_p.assign(joint_X_p_np)
        model.joint_X_c.assign(joint_X_c_np)

        # Convert shape offsets from body-frame-relative to COM-relative
        convert_geom_offset_origin_to_com(
            model_bodies.i_r_com_i,
            model.shape_body,
            wp.array(shape_transform_np, dtype=wp.transformf, device=model.device),
            model_geoms.offset,
        )

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
