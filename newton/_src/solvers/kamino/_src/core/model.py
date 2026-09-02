# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines the model container of Kamino."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

# Newton imports
from .....sim import Model
from ....coupled.model_view import ModelView

# Kamino imports
from .bodies import RigidBodiesData, RigidBodiesModel
from .control import ControlKamino
from .conversions import (
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
    """A container holding the time-invariant information and metadata of a model.

    Kamino lays out the constraints of each world in the following order::

        | dynamic | kinematic | Coulomb friction | effort limits | position limits | contacts |

    where all constraints except contacts are associated with joints.

    These constraint types form groups:

    - **Bilateral:** dynamic and kinematic constraints. These are equality
      constraints without a projection operator.
    - **Bounded:** Coulomb joint friction and effort limit constraints. They have box
      constraints on the Lagrange multipliers.
    - **Unilateral:** position-limit and contact constraints. Their multipliers
      are projected onto the nonnegative half-line and Coulomb cone, respectively.

    All constraints that have a projection operator, i.e. bounded and
    unilateral, are considered **Inequality** constraints.

    The sizes and topologies of the bilateral and bounded groups are fixed when
    the model is constructed. The unilateral group instead reserves capacity for
    position limits and contacts; the active constraints in this group may change
    during the simulation.
    """

    ###
    # Host-side Summary Counts
    ###

    num_worlds: int = 0
    """The number of worlds represented in the model."""

    ###
    # Entity Counts
    ###

    num_bodies: wp.array[wp.int32] | None = None
    """
    The number of bodies in each world.
    Shape of ``(num_worlds,)``.
    """

    num_joints: wp.array[wp.int32] | None = None
    """
    The number of joints in each world.
    Shape of ``(num_worlds,)``.
    """

    num_passive_joints: wp.array[wp.int32] | None = None
    """
    The number of passive joints in each world.
    Shape of ``(num_worlds,)``.
    """

    num_actuated_joints: wp.array[wp.int32] | None = None
    """
    The number of actuated joints in each world.
    Shape of ``(num_worlds,)``.
    """

    num_dynamic_joints: wp.array[wp.int32] | None = None
    """
    The number of dynamic joints in each world.
    Shape of ``(num_worlds,)``.
    """

    num_geoms: wp.array[wp.int32] | None = None
    """
    The number of geometries in each world.
    Shape of ``(num_worlds,)``.
    """

    max_limits: wp.array[wp.int32] | None = None
    """
    The maximum number of limits in each world.
    Shape of ``(num_worlds,)``.
    """

    max_contacts: wp.array[wp.int32] | None = None
    """
    The maximum number of contacts in each world.
    Shape of ``(num_worlds,)``.
    """

    ###
    # DoF Counts
    ###

    num_body_dofs: wp.array[wp.int32] | None = None
    """
    The number of body DoFs of each world.
    Shape of ``(num_worlds,)``.
    """

    num_joint_coords: wp.array[wp.int32] | None = None
    """
    The number of joint coordinates of each world.
    Shape of ``(num_worlds,)``.
    """

    num_joint_dofs: wp.array[wp.int32] | None = None
    """
    The number of joint DoFs of each world.
    Shape of ``(num_worlds,)``.
    """

    num_passive_joint_coords: wp.array[wp.int32] | None = None
    """
    The number of passive joint coordinates of each world.
    Shape of ``(num_worlds,)``.
    """

    num_passive_joint_dofs: wp.array[wp.int32] | None = None
    """
    The number of passive joint DoFs of each world.
    Shape of ``(num_worlds,)``.
    """

    num_actuated_joint_coords: wp.array[wp.int32] | None = None
    """
    The number of actuated joint coordinates of each world.
    Shape of ``(num_worlds,)``.
    """

    num_actuated_joint_dofs: wp.array[wp.int32] | None = None
    """
    The number of actuated joint DoFs of each world.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Constraint Counts
    ###

    # TODO: We could make this a wp.vec2i to store dynamic
    # and kinematic joint constraint counts separately
    num_joint_bilateral_cts: wp.array[wp.int32] | None = None
    """
    The number of bilateral joint constraints of each world.
    Shape of ``(num_worlds,)``.
    """

    num_joint_dynamic_cts: wp.array[wp.int32] | None = None
    """
    The number of dynamic joint constraints of each world.
    Shape of ``(num_worlds,)``.
    """

    num_joint_kinematic_cts: wp.array[wp.int32] | None = None
    """
    The number of kinematic joint constraints of each world.
    Shape of ``(num_worlds,)``.
    """

    num_joint_bounded_cts: wp.array[wp.int32] | None = None
    """
    The number of bounded-multiplier constraint rows of each world.
    Shape of ``(num_worlds,)``.
    """

    num_joint_friction_cts: wp.array[wp.int32] | None = None
    """
    The number of Coulomb joint friction constraint rows of each world.
    Shape of ``(num_worlds,)``.
    """

    num_joint_effort_cts: wp.array[wp.int32] | None = None
    """The number of effort-limit implicit-PD constraint rows in each world."""

    max_limit_cts: wp.array[wp.int32] | None = None
    """
    The maximum number of active limit constraints of each world.
    Shape of ``(num_worlds,)``.
    """

    max_contact_cts: wp.array[wp.int32] | None = None
    """
    The maximum number of active contact constraints of each world.
    Shape of ``(num_worlds,)``.
    """

    max_total_cts: wp.array[wp.int32] | None = None
    """
    The maximum total number of active constraints of each world.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Entity Offsets
    ###

    bodies_offset: wp.array[wp.int32] | None = None
    """
    The body index offset of each world w.r.t the model.
    Shape of ``(num_worlds + 1,)``.
    The last entry is the total bodies count, so that the per-world
    bodies count is encoded as ``bodies_offset[w+1] - bodies_offset[w]``.
    """

    joints_offset: wp.array[wp.int32] | None = None
    """
    The joint index offset of each world w.r.t the model.
    Shape of ``(num_worlds,)``.
    """

    geoms_offset: wp.array[wp.int32] | None = None
    """
    The geom index offset of each world w.r.t. the model.
    Shape of ``(num_worlds,)``.
    """

    limits_offset: wp.array[wp.int32] | None = None
    """
    The limit index offset of each world w.r.t the model.
    Shape of ``(num_worlds,)``.
    """

    contacts_offset: wp.array[wp.int32] | None = None
    """
    The contact index offset of world w.r.t the model.
    Shape of ``(num_worlds,)``.
    """

    inequalities_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the inequalities (bounded-multiplier, limits, and contacts) block of each world.
    Shape of ``(num_worlds,)``.
    """

    ###
    # DoF Offsets
    ###

    body_dofs_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the body DoF block of each world.
    Shape of ``(num_worlds,)``.
    """

    joint_coords_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the joint coordinates block of each world.
    Used to index into arrays that contain flattened joint coordinate-sized data.
    Shape of ``(num_worlds,)``.
    """

    joint_dofs_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the joint DoF block of each world.
    Used to index into arrays that contain flattened joint DoF-sized data.
    Shape of ``(num_worlds,)``.
    """

    joint_passive_coords_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the passive joint coordinates block of each world.
    Used to index into arrays that contain flattened passive joint coordinate-sized data.
    Shape of ``(num_worlds,)``.
    """

    joint_passive_dofs_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the passive joint DoF block of each world.
    Used to index into arrays that contain flattened passive joint DoF-sized data.
    Shape of ``(num_worlds,)``.
    """

    joint_actuated_coords_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the actuated joint coordinates block of each world.
    Used to index into arrays that contain flattened actuated joint coordinate-sized data.
    Shape of ``(num_worlds,)``.
    """

    joint_actuated_dofs_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the actuated joint DoF block of each world.
    Used to index into arrays that contain flattened actuated joint DoF-sized data.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Constraint Offsets
    ###

    joint_bilateral_cts_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the bilateral joint constraints block of each world.
    Used to index into arrays that contain flattened and
    concatenated dynamic and kinematic joint constraint data.
    Shape of ``(num_worlds,)``.
    """

    joint_dynamic_cts_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the dynamic joint constraints block of each world.
    Used to index into arrays that contain flattened dynamic joint constraint data.
    Shape of ``(num_worlds,)``.
    """

    joint_kinematic_cts_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the kinematic joint constraints block of each world.
    Used to index into arrays that contain flattened kinematic joint constraint data.
    Shape of ``(num_worlds,)``.
    """

    joint_bounded_cts_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the bounded-multiplier joint constraints block of each world.
    Used to index into arrays that contain flattened bounded-multiplier joint constraint data.
    Shape of ``(num_worlds,)``.
    """

    joint_friction_cts_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the Coulomb joint friction constraint block of each world.
    Used to index into arrays that contain flattened friction constraint data.
    Shape of ``(num_worlds,)``.
    """

    joint_effort_cts_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the effort limited implicit-PD actuator constraint block of each world.
    Used to index into arrays that contain flattened effort limited implicit-PD actuator constraint data.
    Shape of ``(num_worlds,)``.
    """

    # TODO: We could make this an array of vec7i and store the absolute
    #  startindex of each constraint group in the constraint array `lambda`:
    # - [0]: total_cts_offset
    # - [1]: joint_dynamic_cts_group_offset
    # - [2]: joint_kinematic_cts_group_offset
    # - [3]: joint_friction_cts_group_offset
    # - [4]: joint_effort_cts_group_offset
    # - [5]: limit_cts_group_offset
    # - [6]: contact_cts_group_offset
    # TODO: We could then provide helper functions to get the start-end of each block
    total_cts_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the total constraints block of each world.
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.

    This offset should be used together with:
    - joint_dynamic_cts_group_offset
    - joint_kinematic_cts_group_offset
    - joint_friction_cts_group_offset
    - joint_effort_cts_group_offset
    - limit_cts_group_offset
    - contact_cts_group_offset

    Example:
    ```
    # To index into the dynamic joint constraint reactions of world `w`:
    world_cts_start = model_info.total_cts_offset[w]
    local_joint_dynamic_cts_start = model_info.joint_dynamic_cts_group_offset[w]
    local_joint_kinematic_cts_start = model_info.joint_kinematic_cts_group_offset[w]
    local_joint_friction_cts_start = model_info.joint_friction_cts_group_offset[w]
    local_joint_effort_cts_start = model_info.joint_effort_cts_group_offset[w]
    local_limit_cts_start = model_info.limit_cts_group_offset[w]
    local_contact_cts_start = model_info.contact_cts_group_offset[w]

    # Now compute the starting index of each constraint group within the total constraints block of world `w`:
    world_dynamic_joint_cts_start = world_cts_start + local_joint_dynamic_cts_start
    world_kinematic_joint_cts_start = world_cts_start + local_joint_kinematic_cts_start
    world_friction_cts_start = world_cts_start + local_joint_friction_cts_start
    world_effort_cts_start = world_cts_start + local_joint_effort_cts_start
    world_limit_cts_start = world_cts_start + local_limit_cts_start
    world_contact_cts_start = world_cts_start + local_contact_cts_start
    ```

    Shape of ``(num_worlds,)``.
    """

    joint_dynamic_cts_group_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the dynamic joint constraints group within the constraints block of each world.
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.
    Shape of ``(num_worlds,)``.
    """

    joint_kinematic_cts_group_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the kinematic joint constraints group within the constraints block of each world.
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.
    Shape of ``(num_worlds,)``.
    """

    joint_bounded_cts_group_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the joint bounded constraint group within each world's total constraint block.
    Shape of ``(num_worlds,)``.
    """

    joint_friction_cts_group_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the Coulomb joint friction constraint group within each world's total constraint block.
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.
    Shape of ``(num_worlds,)``.
    """

    joint_effort_cts_group_offset: wp.array[wp.int32] | None = None
    """
    The index offset of the effort-limit implicit-PD constraint group within each world's constraint block.
    Used to index into constraint-space arrays, e.g. constraint residuals and reactions.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Base Properties
    ###

    base_body_index: wp.array[wp.int32] | None = None
    """
    The index of the base body assigned in each world w.r.t the model (-1 if not assigned).
    Must be assigned to enable setting the base pose/velocity in reset operations.
    If a base joint is also assigned, must be the follower body of that joint.
    Shape of ``(num_worlds,)``.
    """

    base_joint_index: wp.array[wp.int32] | None = None
    """
    The index of the base joint assigned in each world w.r.t the model (-1 if not assigned).
    If assigned, reset operations will interpret the base pose/velocity in the base joint frame.
    If assigned, must be a unary free joint.
    Shape of ``(num_worlds,)``.
    """

    ###
    # Host-side Metadata
    ###

    has_world_without_base_body: bool = False
    """
    Host-side flag to indicate whether any world could not be assigned a free-floating base body.
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
    Host-side cache of the model summary sizes.
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
    The materials model container holding all material entities in the model.
    The materials data is currently defined globally to be shared by all worlds.
    """

    material_pairs: MaterialPairsModel | None = None
    """
    The material pairs model container holding all material pairs in the model.
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

    @property
    def use_coord_layout_targets(self) -> bool:
        """Target-layout snapshot. Returns the wrapped
        :class:`newton.Model`'s snapshot when this ``ModelKamino`` was built
        via :meth:`from_newton`; falls back to the live module global
        :data:`newton.use_coord_layout_targets` for native Kamino models built
        through :class:`ModelBuilderKamino` (no wrapped Newton model).
        """
        if self._model is not None:
            return self._model.use_coord_layout_targets
        import newton  # noqa: PLC0415

        return newton.use_coord_layout_targets

    ###
    # Factories
    ###

    def data(
        self,
        joint_wrenches: bool = False,
        requires_grad: bool = False,
        device: wp.DeviceLike = None,
    ) -> DataKamino:
        """
        Creates a model data container with the initial state of the model entities.

        Args:
            joint_wrenches: Whether to include joint wrenches in the model data. Defaults to ``False``.
            requires_grad: Whether the model data should require gradients. Defaults to ``False``.
            device: The device to create the model data on. If not specified, the model's device is used.
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
        njdyncts = self.size.sum_of_num_dynamic_joint_cts
        njkincts = self.size.sum_of_num_kinematic_joint_cts
        njfccts = self.size.sum_of_num_friction_joint_cts
        njeccts = self.size.sum_of_num_effort_joint_cts

        # Construct the model data on the specified device
        with wp.ScopedDevice(device=device):
            # Create a new model data info with the total constraint
            # counts initialized to the joint + bounded constraints count
            info = DataKaminoInfo(
                num_total_cts=wp.array(
                    self.info.num_joint_bilateral_cts.numpy() + self.info.num_joint_bounded_cts.numpy(),
                    dtype=wp.int32,
                    device=device,
                ),
            )

            # Construct the time data with the initial step and time set to zero for all worlds
            time = TimeData(
                steps=wp.zeros(shape=nw, dtype=wp.int32, requires_grad=requires_grad),
                time=wp.zeros(shape=nw, dtype=wp.float32, requires_grad=requires_grad),
            )

            # Construct the rigid bodies data from the model's initial state
            bodies = RigidBodiesData(
                num_bodies=nb,
                I_i=wp.zeros(shape=nb, dtype=wp.mat33f, requires_grad=requires_grad),
                inv_I_i=wp.zeros(shape=nb, dtype=wp.mat33f, requires_grad=requires_grad),
                q_i=wp.clone(self.bodies.q_i_0, requires_grad=requires_grad),
                u_i=wp.clone(self.bodies.u_i_0, requires_grad=requires_grad),
                w_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_a_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_j_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_f_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_l_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_c_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
                w_e_i=wp.zeros_like(self.bodies.u_i_0, requires_grad=requires_grad),
            )

            # Construct the joints data from the model's initial state
            joints = JointsData(
                num_joints=nj,
                p_j=wp.zeros(shape=nj, dtype=wp.transformf, requires_grad=requires_grad),
                q_j=wp.zeros(shape=njcoords, dtype=wp.float32, requires_grad=requires_grad),
                q_j_p=wp.zeros(shape=njcoords, dtype=wp.float32, requires_grad=requires_grad),
                dq_j=wp.zeros(shape=njdofs, dtype=wp.float32, requires_grad=requires_grad),
                tau_j=wp.zeros(shape=njdofs, dtype=wp.float32, requires_grad=requires_grad),
                r_j=wp.zeros(shape=njkincts, dtype=wp.float32, requires_grad=requires_grad),
                dr_j=wp.zeros(shape=njkincts, dtype=wp.float32, requires_grad=requires_grad),
                lambda_kin_j=wp.zeros(shape=njkincts, dtype=wp.float32, requires_grad=requires_grad),
                lambda_dyn_j=wp.zeros(shape=njdyncts, dtype=wp.float32, requires_grad=requires_grad),
                lambda_f_j=wp.zeros(shape=njfccts, dtype=wp.float32, requires_grad=requires_grad),
                lambda_tau_j=wp.zeros(shape=njeccts, dtype=wp.float32, requires_grad=requires_grad),
                m_j=wp.zeros(shape=njdyncts, dtype=wp.float32, requires_grad=requires_grad),
                inv_m_j=wp.zeros(shape=njdyncts, dtype=wp.float32, requires_grad=requires_grad),
                dq_b_j=wp.zeros(shape=njdyncts, dtype=wp.float32, requires_grad=requires_grad),
                inv_m_a=wp.zeros(shape=njeccts, dtype=wp.float32, requires_grad=requires_grad),
                dq_b_a=wp.zeros(shape=njeccts, dtype=wp.float32, requires_grad=requires_grad),
                bound_a=wp.zeros(shape=njeccts, dtype=wp.float32, requires_grad=requires_grad),
                # TODO: Should we make these optional and only include them when implicit joints are present?
                q_j_ref=wp.clone(self.joints.q_j_0, requires_grad=requires_grad),
                dq_j_ref=wp.clone(self.joints.dq_j_0, requires_grad=requires_grad),
                tau_j_ref=wp.zeros(shape=njdofs, dtype=wp.float32, requires_grad=requires_grad),
                j_w_j=wp.zeros(shape=nj, dtype=wp.spatial_vectorf, requires_grad=requires_grad)
                if joint_wrenches
                else None,
                j_w_c_j=wp.zeros(shape=nj, dtype=wp.spatial_vectorf, requires_grad=requires_grad)
                if joint_wrenches
                else None,
                j_w_f_j=wp.zeros(shape=nj, dtype=wp.spatial_vectorf, requires_grad=requires_grad)
                if joint_wrenches
                else None,
                j_w_a_j=wp.zeros(shape=nj, dtype=wp.spatial_vectorf, requires_grad=requires_grad)
                if joint_wrenches
                else None,
                j_w_l_j=wp.zeros(shape=nj, dtype=wp.spatial_vectorf, requires_grad=requires_grad)
                if joint_wrenches
                else None,
            )

            # Construct the geometries data from the model's initial state
            geoms = GeometriesData(
                num_geoms=ng,
                pose=wp.zeros(shape=ng, dtype=wp.transformf, requires_grad=requires_grad),
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

        Args:
            requires_grad: Whether the state should require gradients. Defaults to ``False``.
            device: The device to create the state on. If not specified, the model's device is used.
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
                dq_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=wp.float32, requires_grad=requires_grad),
                lambda_kin_j=wp.zeros(
                    shape=self.size.sum_of_num_kinematic_joint_cts, dtype=wp.float32, requires_grad=requires_grad
                ),
                lambda_dyn_j=wp.zeros(
                    shape=self.size.sum_of_num_dynamic_joint_cts, dtype=wp.float32, requires_grad=requires_grad
                ),
                lambda_f_j=wp.zeros(
                    shape=self.size.sum_of_num_friction_joint_cts, dtype=wp.float32, requires_grad=requires_grad
                ),
                lambda_tau_j=wp.zeros(
                    shape=self.size.sum_of_num_effort_joint_cts, dtype=wp.float32, requires_grad=requires_grad
                ),
            )

        # Return the constructed state container
        return state

    def control(self, requires_grad: bool = False, device: wp.DeviceLike = None) -> ControlKamino:
        """
        Creates a control container with all values initialized to zeros.

        Args:
            requires_grad: Whether the control container should require gradients. Defaults to ``False``.
            device: The device to create the control container on. If not specified, the model's device is used.
        """
        # If no device is specified, use the model's device
        if device is None:
            device = self.device

        # Create a new control container on the specified device
        with wp.ScopedDevice(device=device):
            control = ControlKamino(
                tau_j=wp.zeros(shape=self.size.sum_of_num_joint_dofs, dtype=wp.float32, requires_grad=requires_grad),
                q_j_ref=wp.clone(self.joints.q_j_0, requires_grad=requires_grad),
                dq_j_ref=wp.clone(self.joints.dq_j_0, requires_grad=requires_grad),
                tau_j_ref=wp.zeros(
                    shape=self.size.sum_of_num_joint_dofs, dtype=wp.float32, requires_grad=requires_grad
                ),
            )

        # Post-processing to finalize the control container
        # NOTE: This is currently necessary to handle the case when
        # the total number of joint coordinates and DoFs differ, in
        # which case a temporary buffer is allocated for the conversion.
        control.finalize(self, device=device)

        # Return the constructed control container
        return control

    @staticmethod
    def from_newton(model: Model | ModelView) -> ModelKamino:
        """
        Finalizes the :class:`ModelKamino` from an existing instance of :class:`newton.Model`.

        Args:
            model: The source :class:`newton.Model` instance to be converted.

        Returns:
            Kamino model converted from the input Newton model.
        """

        # Ensure the base model is valid. Coupled solvers pass ModelView
        # instances, which are intentionally accepted alongside full Models.
        if model is None:
            raise ValueError("ModelKamino.from_newton() requires a newton.Model or ModelView instance, got None.")
        if not isinstance(model, (Model, ModelView)):
            raise TypeError(
                f"ModelKamino.from_newton() requires a newton.Model or ModelView instance, got {type(model).__name__}."
            )

        # Normalize conversion-only grouping metadata for single-world models.
        conversion_model = model
        if model.world_count == 1:
            conversion_model = ModelView(model, "kamino_worlds")
            has_dedicated_global_gravity = model.gravity.shape[0] > model.world_count
            for attr, start_attr in (
                ("body_world", "body_world_start"),
                ("joint_world", "joint_world_start"),
                ("shape_world", "shape_world_start"),
            ):
                arr = getattr(model, attr)
                arr_np = arr.numpy()
                if np.any(arr_np < 0):
                    # Preserve body -1 only when it selects a dedicated global gravity entry.
                    if attr != "body_world" or not has_dedicated_global_gravity:
                        arr_np = arr_np.copy()
                        arr_np[arr_np < 0] = 0
                        setattr(conversion_model, attr, wp.array(arr_np, dtype=wp.int32, device=model.device))
                    # Update world start indices
                    arr_start = getattr(model, start_attr)
                    arr_start_np = arr_start.numpy().copy()
                    arr_start_np[0] = 0
                    arr_start_np[-2] = arr_start_np[-1]
                    setattr(
                        conversion_model,
                        start_attr,
                        wp.array(arr_start_np, dtype=wp.int32, device=model.device),
                    )

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
                dt=wp.zeros(shape=(model.world_count,), dtype=wp.float32),
                inv_dt=wp.zeros(shape=(model.world_count,), dtype=wp.float32),
            )

            # Per-world gravity
            model_gravity = GravityModel.from_newton(model)

            # Bodies
            model_bodies = convert_rigid_bodies(conversion_model, model_size, model_info)

            # Joints
            model_joints = convert_joints(
                conversion_model,
                model_size,
                model_info,
                model_bodies,
            )

            # Geometries
            model_geoms = convert_geometries(
                model=conversion_model,
                model_size=model_size,
                model_bodies=model_bodies,
                materials_manager=materials_manager,
            )

            # Materials
            model_materials = materials_manager.make_materials_model()
            model_material_pairs = materials_manager.make_material_pairs_model()

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
