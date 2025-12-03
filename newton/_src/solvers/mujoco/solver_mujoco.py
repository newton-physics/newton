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

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...core.types import nparray, override
from ...geometry import MESH_MAXHULLVERT, GeoType, ShapeFlags
from ...sim import (
    JOINT_LIMIT_UNLIMITED,
    Contacts,
    Control,
    EqType,
    JointType,
    Model,
    ModelAttributeAssignment,
    ModelAttributeFrequency,
    ModelBuilder,
    State,
    color_graph,
    plot_graph,
)
from ...utils import topological_sort
from ...utils.benchmark import event_scope
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .kernels import (
    apply_mjc_body_f_kernel,
    apply_mjc_control_kernel,
    apply_mjc_qfrc_kernel,
    convert_body_xforms_to_warp_kernel,
    convert_mj_coords_to_warp_kernel,
    convert_mjw_contact_to_warp_kernel,
    convert_newton_contacts_to_mjwarp_kernel,
    convert_warp_coords_to_mj_kernel,
    eval_articulation_fk,
    repeat_array_kernel,
    update_axis_properties_kernel,
    update_body_inertia_kernel,
    update_body_mass_ipos_kernel,
    update_geom_properties_kernel,
    update_joint_dof_properties_kernel,
    update_joint_transforms_kernel,
    update_model_properties_kernel,
    update_shape_mappings_kernel,
)

if TYPE_CHECKING:
    from mujoco import MjData, MjModel
    from mujoco_warp import Data as MjWarpData
    from mujoco_warp import Model as MjWarpModel
else:
    MjModel = object
    MjData = object
    MjWarpModel = object
    MjWarpData = object


class SolverMuJoCo(SolverBase):
    """
    This solver provides an interface to simulate physics using the `MuJoCo <https://github.com/google-deepmind/mujoco>`_ physics engine,
    optimized with GPU acceleration through `mujoco_warp <https://github.com/google-deepmind/mujoco_warp>`_. It supports both MuJoCo and
    mujoco_warp backends, enabling efficient simulation of articulated systems with
    contacts and constraints.

    .. note::

        - This solver requires `mujoco_warp`_ and its dependencies to be installed.
        - For installation instructions, see the `mujoco_warp`_ repository.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCo(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    Debugging
    ---------

    To debug the SolverMuJoCo, you can save the MuJoCo model that is created from the :class:`newton.Model` in the constructor of the SolverMuJoCo:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCo(model, save_to_mjcf="model.xml")

    This will save the MuJoCo model as an MJCF file, which can be opened in the MuJoCo simulator.

    It is also possible to visualize the simulation running in the SolverMuJoCo through MuJoCo's own viewer.
    This may help to debug the simulation and see how the MuJoCo model looks like when it is created from the Newton model.

    .. code-block:: python

        import newton

        solver = newton.solvers.SolverMuJoCo(model)

        for _ in range(num_frames):
            # step the solver
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

            solver.render_mujoco_viewer()
    """

    # Class variables to cache the imported modules
    _mujoco = None
    _mujoco_warp = None

    @classmethod
    def import_mujoco(cls):
        """Import the MuJoCo Warp dependencies and cache them as class variables."""
        if cls._mujoco is None or cls._mujoco_warp is None:
            try:
                import mujoco  # noqa: PLC0415
                import mujoco_warp  # noqa: PLC0415

                cls._mujoco = mujoco
                cls._mujoco_warp = mujoco_warp
            except ImportError as e:
                raise ImportError(
                    "MuJoCo backend not installed. Please refer to https://github.com/google-deepmind/mujoco_warp for installation instructions."
                ) from e
        return cls._mujoco, cls._mujoco_warp

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """
        Declare custom attributes to be allocated on the Model object within the ``mujoco`` namespace.
        Note that we declare all custom attributes with the :attr:`newton.ModelBuilder.CustomAttribute.usd_attribute_name` set to ``"mjc"`` here to leverage the MuJoCo USD schema
        where attributes are named ``"mjc:attr"`` rather than ``"newton:mujoco:attr"``.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="condim",
                frequency=ModelAttributeFrequency.SHAPE,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.int32,
                default=3,
                namespace="mujoco",
                usd_attribute_name="mjc:condim",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="limit_margin",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:margin",
                mjcf_attribute_name="margin",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solimplimit",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.types.vector(length=5, dtype=wp.float32),
                default=wp.types.vector(length=5, dtype=wp.float32)(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimplimit",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solreffriction",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.types.vector(length=2, dtype=wp.float32),
                default=wp.types.vector(length=2, dtype=wp.float32)(0.02, 1.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solreffriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solimpfriction",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.types.vector(length=5, dtype=wp.float32),
                default=wp.types.vector(length=5, dtype=wp.float32)(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimpfriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="gravcomp",
                frequency=ModelAttributeFrequency.BODY,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:gravcomp",
                mjcf_attribute_name="gravcomp",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_passive_stiffness",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:stiffness",
                mjcf_attribute_name="stiffness",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_passive_damping",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:damping",
                mjcf_attribute_name="damping",
            )
        )

    def __init__(
        self,
        model: Model,
        *,
        mjw_model: MjWarpModel | None = None,
        mjw_data: MjWarpData | None = None,
        separate_worlds: bool | None = None,
        njmax: int | None = None,
        nconmax: int | None = None,
        iterations: int = 20,
        ls_iterations: int = 10,
        solver: int | str = "cg",
        integrator: int | str = "implicitfast",
        cone: int | str = "pyramidal",
        impratio: float = 1.0,
        use_mujoco_cpu: bool = False,
        disable_contacts: bool = False,
        default_actuator_gear: float | None = None,
        actuator_gears: dict[str, float] | None = None,
        update_data_interval: int = 1,
        save_to_mjcf: str | None = None,
        ls_parallel: bool = False,
        use_mujoco_contacts: bool = True,
        tolerance: float = 1e-6,
        ls_tolerance: float = 0.01,
        include_sites: bool = True,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            mjw_model (MjWarpModel | None): Optional pre-existing MuJoCo Warp model. If provided with `mjw_data`, conversion from Newton model is skipped.
            mjw_data (MjWarpData | None): Optional pre-existing MuJoCo Warp data. If provided with `mjw_model`, conversion from Newton model is skipped.
            separate_worlds (bool | None): If True, each Newton world is mapped to a separate MuJoCo world. Defaults to `not use_mujoco_cpu`.
            njmax (int): Maximum number of constraints per world. If None, a default value is estimated from the initial state. Note that the larger of the user-provided value or the default value is used.
            nconmax (int | None): Number of contact points per world. If None, a default value is estimated from the initial state. Note that the larger of the user-provided value or the default value is used.
            iterations (int): Number of solver iterations.
            ls_iterations (int): Number of line search iterations for the solver.
            solver (int | str): Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants.
            integrator (int | str): Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants.
            cone (int | str): The type of contact friction cone. Can be "pyramidal", "elliptic", or their corresponding MuJoCo integer constants.
            impratio (float): Frictional-to-normal constraint impedance ratio.
            use_mujoco_cpu (bool): If True, use the MuJoCo-C CPU backend instead of `mujoco_warp`.
            disable_contacts (bool): If True, disable contact computation in MuJoCo.
            register_collision_groups (bool): If True, register collision groups from the Newton model in MuJoCo.
            default_actuator_gear (float | None): Default gear ratio for all actuators. Can be overridden by `actuator_gears`.
            actuator_gears (dict[str, float] | None): Dictionary mapping joint names to specific gear ratios, overriding the `default_actuator_gear`.
            update_data_interval (int): Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state. If 0, Data is never updated after initialization.
            save_to_mjcf (str | None): Optional path to save the generated MJCF model file.
            ls_parallel (bool): If True, enable parallel line search in MuJoCo. Defaults to False.
            use_mujoco_contacts (bool): If True, use the MuJoCo contact solver. If False, use the Newton contact solver (newton contacts must be passed in through the step function in that case).
            tolerance (float | None): Solver tolerance for early termination of the iterative solver. Defaults to 1e-6 and will be increased to 1e-6 by the MuJoCo solver if a smaller value is provided.
            ls_tolerance (float | None): Solver tolerance for early termination of the line search. Defaults to 0.01.
            include_sites (bool): If ``True`` (default), Newton shapes marked with ``ShapeFlags.SITE`` are exported as MuJoCo sites. Sites are non-colliding reference points used for sensor attachment, debugging, or as frames of reference. If ``False``, sites are skipped during export. Defaults to ``True``.
        """
        super().__init__(model)
        # Import and cache MuJoCo modules (only happens once per class)
        mujoco, _ = self.import_mujoco()

        if use_mujoco_cpu and not use_mujoco_contacts:
            print("Setting use_mujoco_contacts to False has no effect when use_mujoco_cpu is True")

        self.joint_mjc_dof_start: wp.array(dtype=wp.int32) | None = None
        """Mapping from Newton joint index to the start index of its joint axes in MuJoCo. Only defined for the joint indices of the first world in Newton, defaults to -1 otherwise. Shape [joint_count], dtype int32."""
        self.dof_to_mjc_joint: wp.array(dtype=wp.int32) | None = None
        """Mapping from Newton DOF index to MuJoCo joint index. Only defined for the first world in Newton. Shape [joint_dof_count // num_worlds], dtype int32."""
        self.mjc_axis_to_actuator: wp.array(dtype=int) | None = None
        """Mapping from Newton joint axis index to MJC actuator index. Shape [dof_count], dtype int32."""
        self.to_mjc_body_index: wp.array(dtype=int) | None = None
        """Mapping from MuJoCo body index to Newton body index (skip world body index -1). Shape [bodies_per_world], dtype int32."""
        self.newton_body_to_mocap_index: wp.array(dtype=int) | None = None
        """Mapping from Newton body index to MuJoCo mocap body index. -1 if body is not mocap. Shape [bodies_per_world], dtype int32."""
        self.to_newton_shape_index: wp.array(dtype=int, ndim=2) | None = None
        """Mapping from MuJoCo [worldid, geom index] to Newton shape index. This is used to map MuJoCo geoms to Newton shapes."""
        self.to_mjc_geom_index: wp.array(dtype=wp.vec2i) | None = None
        """Mapping from Newton shape index to MuJoCo [worldid, geom index]."""

        self.selected_shapes: wp.array(dtype=int) | None = None
        """Indices of Newton shapes that are used in the MuJoCo model (includes non-instantiated visual-only shapes) for the first world as a basis for replicating the nworlds worlds in MuJoCo Warp."""
        self.selected_joints: wp.array(dtype=int) | None = None
        """Indices of Newton joints that are used in the MuJoCo model for the first world as a basis for replicating the nworlds worlds in MuJoCo Warp."""
        self.selected_bodies: wp.array(dtype=int) | None = None
        """Indices of Newton bodies that are used in the MuJoCo model for the first world as a basis for replicating the nworlds worlds in MuJoCo Warp."""

        self._viewer = None
        """Instance of the MuJoCo viewer for debugging."""

        disableflags = 0
        if disable_contacts:
            disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        if mjw_model is not None and mjw_data is not None:
            self.mjw_model = mjw_model
            self.mjw_data = mjw_data
            self.use_mujoco_cpu = False
        else:
            self.use_mujoco_cpu = use_mujoco_cpu
            if separate_worlds is None:
                separate_worlds = not use_mujoco_cpu
            with wp.ScopedTimer("convert_model_to_mujoco", active=False):
                self._convert_to_mjc(
                    model,
                    disableflags=disableflags,
                    disable_contacts=disable_contacts,
                    separate_worlds=separate_worlds,
                    njmax=njmax,
                    nconmax=nconmax,
                    iterations=iterations,
                    ls_iterations=ls_iterations,
                    cone=cone,
                    impratio=impratio,
                    solver=solver,
                    integrator=integrator,
                    default_actuator_gear=default_actuator_gear,
                    actuator_gears=actuator_gears,
                    target_filename=save_to_mjcf,
                    ls_parallel=ls_parallel,
                    tolerance=tolerance,
                    ls_tolerance=ls_tolerance,
                    include_sites=include_sites,
                )
        self.update_data_interval = update_data_interval
        self._step = 0

        if self.mjw_model is not None:
            self.mjw_model.opt.run_collision_detection = use_mujoco_contacts

    @event_scope
    def mujoco_warp_step(self):
        self._mujoco_warp.step(self.mjw_model, self.mjw_data)

    @event_scope
    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        if self.use_mujoco_cpu:
            self.apply_mjc_control(self.model, state_in, control, self.mj_data)
            if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
                # XXX updating the mujoco state at every step may introduce numerical instability
                self.update_mjc_data(self.mj_data, self.model, state_in)
            self.mj_model.opt.timestep = dt
            self._mujoco.mj_step(self.mj_model, self.mj_data)
            self.update_newton_state(self.model, state_out, self.mj_data)
        else:
            self.apply_mjc_control(self.model, state_in, control, self.mjw_data)
            if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
                self.update_mjc_data(self.mjw_data, self.model, state_in)
            self.mjw_model.opt.timestep.fill_(dt)
            with wp.ScopedDevice(self.model.device):
                if self.mjw_model.opt.run_collision_detection:
                    self.mujoco_warp_step()
                else:
                    self.convert_contacts_to_mjwarp(self.model, state_in, contacts)
                    self.mujoco_warp_step()

            self.update_newton_state(self.model, state_out, self.mjw_data)
        self._step += 1
        return state_out

    def convert_contacts_to_mjwarp(self, model: Model, state_in: State, contacts: Contacts):
        bodies_per_world = self.model.body_count // self.model.num_worlds
        wp.launch(
            convert_newton_contacts_to_mjwarp_kernel,
            dim=(contacts.rigid_contact_max,),
            inputs=[
                state_in.body_q,
                model.shape_body,
                self.mjw_model.geom_condim,
                self.mjw_model.geom_priority,
                self.mjw_model.geom_solmix,
                self.mjw_model.geom_solref,
                self.mjw_model.geom_solimp,
                self.mjw_model.geom_friction,
                self.mjw_model.geom_margin,
                self.mjw_model.geom_gap,
                # Newton contacts
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                bodies_per_world,
                self.to_mjc_geom_index,
                # Mujoco warp contacts
                self.mjw_data.nacon,
                self.mjw_data.contact.dist,
                self.mjw_data.contact.pos,
                self.mjw_data.contact.frame,
                self.mjw_data.contact.includemargin,
                self.mjw_data.contact.friction,
                self.mjw_data.contact.solref,
                self.mjw_data.contact.solreffriction,
                self.mjw_data.contact.solimp,
                self.mjw_data.contact.dim,
                self.mjw_data.contact.geom,
                self.mjw_data.contact.worldid,
                # Data to clear
                self.mjw_data.nworld,
                self.mjw_data.ncollision,
            ],
        )

    @override
    def notify_model_changed(self, flags: int):
        if flags & SolverNotifyFlags.BODY_INERTIAL_PROPERTIES:
            self.update_model_inertial_properties()
        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self.update_joint_properties()
        if flags & SolverNotifyFlags.JOINT_DOF_PROPERTIES:
            self.update_joint_dof_properties()
        if flags & SolverNotifyFlags.SHAPE_PROPERTIES:
            self.update_geom_properties()
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self.update_model_properties()

    @staticmethod
    def _data_is_mjwarp(data):
        # Check if the data is a mujoco_warp Data object
        return hasattr(data, "nworld")

    def apply_mjc_control(self, model: Model, state: State, control: Control | None, mj_data: MjWarpData | MjData):
        if control is None or control.joint_f is None:
            if state.body_f is None:
                return
        is_mjwarp = SolverMuJoCo._data_is_mjwarp(mj_data)
        if is_mjwarp:
            ctrl = mj_data.ctrl
            qfrc = mj_data.qfrc_applied
            xfrc = mj_data.xfrc_applied
            nworld = mj_data.nworld
        else:
            ctrl = wp.zeros((1, len(mj_data.ctrl)), dtype=wp.float32, device=model.device)
            qfrc = wp.zeros((1, len(mj_data.qfrc_applied)), dtype=wp.float32, device=model.device)
            xfrc = wp.zeros((1, len(mj_data.xfrc_applied)), dtype=wp.spatial_vector, device=model.device)
            nworld = 1
        axes_per_world = model.joint_dof_count // nworld
        joints_per_world = model.joint_count // nworld
        bodies_per_world = model.body_count // nworld
        if control is not None:
            wp.launch(
                apply_mjc_control_kernel,
                dim=(nworld, axes_per_world),
                inputs=[
                    control.joint_target_pos,
                    control.joint_target_vel,
                    self.mjc_axis_to_actuator,
                    axes_per_world,
                ],
                outputs=[
                    ctrl,
                ],
                device=model.device,
            )
            wp.launch(
                apply_mjc_qfrc_kernel,
                dim=(nworld, joints_per_world),
                inputs=[
                    state.body_q,
                    control.joint_f,
                    model.joint_type,
                    model.body_com,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    joints_per_world,
                    bodies_per_world,
                ],
                outputs=[
                    qfrc,
                ],
                device=model.device,
            )

        if state.body_f is not None:
            wp.launch(
                apply_mjc_body_f_kernel,
                dim=(nworld, bodies_per_world),
                inputs=[
                    model.up_axis,
                    state.body_q,
                    state.body_f,
                    self.to_mjc_body_index,
                    bodies_per_world,
                ],
                outputs=[
                    xfrc,
                ],
                device=model.device,
            )
        if not is_mjwarp:
            mj_data.xfrc_applied = xfrc.numpy()
            mj_data.ctrl[:] = ctrl.numpy().flatten()
            mj_data.qfrc_applied[:] = qfrc.numpy()

    def update_mjc_data(self, mj_data: MjWarpData | MjData, model: Model, state: State | None = None):
        is_mjwarp = SolverMuJoCo._data_is_mjwarp(mj_data)
        if is_mjwarp:
            # we have an MjWarp Data object
            qpos = mj_data.qpos
            qvel = mj_data.qvel
            nworld = mj_data.nworld
        else:
            # we have an MjData object from Mujoco
            qpos = wp.empty((1, model.joint_coord_count), dtype=wp.float32, device=model.device)
            qvel = wp.empty((1, model.joint_dof_count), dtype=wp.float32, device=model.device)
            nworld = 1
        if state is None:
            joint_q = model.joint_q
            joint_qd = model.joint_qd
        else:
            joint_q = state.joint_q
            joint_qd = state.joint_qd
        joints_per_world = model.joint_count // nworld
        wp.launch(
            convert_warp_coords_to_mj_kernel,
            dim=(nworld, joints_per_world),
            inputs=[
                joint_q,
                joint_qd,
                joints_per_world,
                model.up_axis,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
            ],
            outputs=[qpos, qvel],
            device=model.device,
        )
        if not is_mjwarp:
            mj_data.qpos[:] = qpos.numpy().flatten()[: len(mj_data.qpos)]
            mj_data.qvel[:] = qvel.numpy().flatten()[: len(mj_data.qvel)]

    def update_newton_state(
        self,
        model: Model,
        state: State,
        mj_data: MjWarpData | MjData,
        eval_fk: bool = True,
    ):
        is_mjwarp = SolverMuJoCo._data_is_mjwarp(mj_data)
        if is_mjwarp:
            # we have an MjWarp Data object
            qpos = mj_data.qpos
            qvel = mj_data.qvel
            nworld = mj_data.nworld

            xpos = mj_data.xpos
            xquat = mj_data.xquat
        else:
            # we have an MjData object from Mujoco
            qpos = wp.array([mj_data.qpos], dtype=wp.float32, device=model.device)
            qvel = wp.array([mj_data.qvel], dtype=wp.float32, device=model.device)
            nworld = 1

            xpos = wp.array([mj_data.xpos], dtype=wp.vec3, device=model.device)
            xquat = wp.array([mj_data.xquat], dtype=wp.quat, device=model.device)
        joints_per_world = model.joint_count // nworld
        wp.launch(
            convert_mj_coords_to_warp_kernel,
            dim=(nworld, joints_per_world),
            inputs=[
                qpos,
                qvel,
                joints_per_world,
                int(model.up_axis),
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
            ],
            outputs=[state.joint_q, state.joint_qd],
            device=model.device,
        )

        if eval_fk:
            # custom forward kinematics for handling multi-dof joints
            wp.launch(
                kernel=eval_articulation_fk,
                dim=model.articulation_count,
                inputs=[
                    model.articulation_start,
                    state.joint_q,
                    state.joint_qd,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_dof_dim,
                    model.body_com,
                ],
                outputs=[
                    state.body_q,
                    state.body_qd,
                ],
                device=model.device,
            )
        else:
            bodies_per_world = model.body_count // model.num_worlds
            wp.launch(
                convert_body_xforms_to_warp_kernel,
                dim=(nworld, bodies_per_world),
                inputs=[
                    xpos,
                    xquat,
                    self.to_mjc_body_index,
                    bodies_per_world,
                ],
                outputs=[state.body_q],
                device=model.device,
            )

    @staticmethod
    def find_body_collision_filter_pairs(
        model: Model,
        selected_bodies: nparray,
        colliding_shapes: nparray,
    ):
        """For shape collision filter pairs, find body collision filter pairs that are contained within."""

        body_exclude_pairs = []
        shape_set = set(colliding_shapes)

        body_shapes = {}
        for body in selected_bodies:
            shapes = model.body_shapes[body]
            shapes = [s for s in shapes if s in shape_set]
            body_shapes[body] = shapes

        bodies_a, bodies_b = np.triu_indices(len(selected_bodies), k=1)
        for body_a, body_b in zip(bodies_a, bodies_b, strict=True):
            b1, b2 = selected_bodies[body_a], selected_bodies[body_b]
            shapes_1 = body_shapes[b1]
            shapes_2 = body_shapes[b2]
            excluded = True
            for shape_1 in shapes_1:
                for shape_2 in shapes_2:
                    if shape_1 > shape_2:
                        s1, s2 = shape_2, shape_1
                    else:
                        s1, s2 = shape_1, shape_2
                    if (s1, s2) not in model.shape_collision_filter_pairs:
                        excluded = False
                        break
            if excluded:
                body_exclude_pairs.append((b1, b2))
        return body_exclude_pairs

    @staticmethod
    def color_collision_shapes(
        model: Model, selected_shapes: nparray, visualize_graph: bool = False, shape_keys: list[str] | None = None
    ) -> nparray:
        """
        Find a graph coloring of the collision filter pairs in the model.
        Shapes within the same color cannot collide with each other.
        Shapes can only collide with shapes of different colors.

        Args:
            model (Model): The model to color the collision shapes of.
            selected_shapes (nparray): The indices of the collision shapes to color.
            visualize_graph (bool): Whether to visualize the graph coloring.
            shape_keys (list[str]): The keys of the shapes, only used for visualization.

        Returns:
            nparray: An integer array of shape (num_shapes,), where each element is the color of the corresponding shape.
        """
        # we first create a mapping from selected shape to local color shape index
        # to reduce the number of nodes in the graph to only the number of selected shapes
        # without any gaps between the indices (otherwise we have to allocate max(selected_shapes) + 1 nodes)
        to_color_shape_index = {}
        for i, shape in enumerate(selected_shapes):
            to_color_shape_index[shape] = i
        # find graph coloring of collision filter pairs
        num_shapes = len(selected_shapes)
        shape_a, shape_b = np.triu_indices(num_shapes, k=1)
        shape_collision_group_np = model.shape_collision_group.numpy()
        cgroup = [shape_collision_group_np[i] for i in selected_shapes]
        # edges representing colliding shape pairs
        graph_edges = [
            (i, j)
            for i, j in zip(shape_a, shape_b, strict=True)
            if (
                (selected_shapes[i], selected_shapes[j]) not in model.shape_collision_filter_pairs
                and (cgroup[i] == cgroup[j] or cgroup[i] == -1 or cgroup[j] == -1)
            )
        ]
        shape_color = np.zeros(model.shape_count, dtype=np.int32)
        if len(graph_edges) > 0:
            color_groups = color_graph(
                num_nodes=num_shapes,
                graph_edge_indices=wp.array(graph_edges, dtype=wp.int32),
                balance_colors=False,
            )
            num_colors = 0
            for group in color_groups:
                num_colors += 1
                shape_color[selected_shapes[group]] = num_colors
            if visualize_graph:
                plot_graph(
                    vertices=np.arange(num_shapes),
                    edges=graph_edges,
                    node_labels=[shape_keys[i] for i in selected_shapes] if shape_keys is not None else None,
                    node_colors=[shape_color[i] for i in selected_shapes],
                )

        return shape_color

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        # TODO: ensure that class invariants are preserved
        # TODO: fill actual contact arrays instead of creating new ones
        mj_data = self.mjw_data
        naconmax = mj_data.naconmax
        mj_contact = mj_data.contact

        contacts.rigid_contact_max = naconmax
        contacts.rigid_contact_count = mj_data.nacon
        contacts.position = mj_contact.pos
        contacts.separation = mj_contact.dist

        if not hasattr(contacts, "pair"):
            contacts.pair = wp.zeros(naconmax, dtype=wp.vec2i, device=self.model.device)

        if not hasattr(contacts, "normal"):
            contacts.normal = wp.zeros(naconmax, dtype=wp.vec3f, device=self.model.device)

        if not hasattr(contacts, "force"):
            contacts.force = wp.zeros(naconmax, dtype=wp.float32, device=self.model.device)

        wp.launch(
            convert_mjw_contact_to_warp_kernel,
            dim=mj_data.naconmax,
            inputs=[
                self.to_newton_shape_index,
                self.mjw_model.opt.cone == int(self._mujoco.mjtCone.mjCONE_PYRAMIDAL),
                mj_data.nacon,
                mj_contact.frame,
                mj_contact.dim,
                mj_contact.geom,
                mj_contact.efc_address,
                mj_contact.worldid,
                mj_data.efc.force,
            ],
            outputs=[
                contacts.pair,
                contacts.normal,
                contacts.force,
            ],
            device=self.model.device,
        )
        contacts.n_contacts = mj_data.nacon

    def _convert_to_mjc(
        self,
        model: Model,
        state: State | None = None,
        *,
        separate_worlds: bool = True,
        iterations: int = 20,
        ls_iterations: int = 10,
        njmax: int | None = None,  # number of constraints per world
        nconmax: int | None = None,
        solver: int | str = "cg",
        integrator: int | str = "implicitfast",
        disableflags: int = 0,
        disable_contacts: bool = False,
        impratio: float = 1.0,
        tolerance: float = 1e-6,
        ls_tolerance: float = 0.01,
        cone: int | str = "pyramidal",
        geom_solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0),
        target_filename: str | None = None,
        default_actuator_args: dict | None = None,
        default_actuator_gear: float | None = None,
        actuator_gears: dict[str, float] | None = None,
        actuated_axes: list[int] | None = None,
        skip_visual_only_geoms: bool = True,
        include_sites: bool = True,
        add_axes: bool = False,
        mesh_maxhullvert: int = MESH_MAXHULLVERT,
        ls_parallel: bool = False,
    ) -> tuple[MjWarpModel, MjWarpData, MjModel, MjData]:
        """
        Convert a Newton model and state to MuJoCo (Warp) model and data.

        Args:
            Model (newton.Model): The Newton model to convert.
            State (newton.State): The Newton state to convert.

        Returns:
            tuple[MjWarpModel, MjWarpData, MjModel, MjData]: A tuple containing the model and data objects for ``mujoco_warp`` and MuJoCo.
        """

        if not model.joint_count:
            raise ValueError("The model must have at least one joint to be able to convert it to MuJoCo.")

        # Validate that separate_worlds=False is only used with single world
        if not separate_worlds and model.num_worlds > 1:
            raise ValueError(
                f"separate_worlds=False is only supported for single-world models. "
                f"Got num_worlds={model.num_worlds}. Use separate_worlds=True for multi-world models."
            )

        mujoco, mujoco_warp = self.import_mujoco()

        actuator_args = {
            # "ctrllimited": True,
            # "ctrlrange": (-1.0, 1.0),
            "gear": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "trntype": mujoco.mjtTrn.mjTRN_JOINT,
            # motor actuation properties (already the default settings in Mujoco)
            "gainprm": [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "biasprm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dyntype": mujoco.mjtDyn.mjDYN_NONE,
            "gaintype": mujoco.mjtGain.mjGAIN_FIXED,
            "biastype": mujoco.mjtBias.mjBIAS_AFFINE,
        }
        if default_actuator_args is not None:
            actuator_args.update(default_actuator_args)
        if default_actuator_gear is not None:
            actuator_args["gear"][0] = default_actuator_gear
        if actuator_gears is None:
            actuator_gears = {}

        def _resolve_mj_opt(val, opts: dict[str, int], kind: str):
            if isinstance(val, str):
                key = val.strip().lower()
                try:
                    return opts[key]
                except KeyError as e:
                    options = "', '".join(sorted(opts))
                    raise ValueError(f"Unknown {kind} '{val}'. Valid options: '{options}'.") from e
            return val

        solver = _resolve_mj_opt(
            solver, {"cg": mujoco.mjtSolver.mjSOL_CG, "newton": mujoco.mjtSolver.mjSOL_NEWTON}, "solver"
        )
        integrator = _resolve_mj_opt(
            integrator,
            {
                "euler": mujoco.mjtIntegrator.mjINT_EULER,
                "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            },
            "integrator",
        )
        cone = _resolve_mj_opt(
            cone, {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}, "cone"
        )

        def quat_to_mjc(q):
            # convert from xyzw to wxyz
            return [q[3], q[0], q[1], q[2]]

        def quat_from_mjc(q):
            # convert from wxyz to xyzw
            return [q[1], q[2], q[3], q[0]]

        def fill_arr_from_dict(arr: nparray, d: dict[int, Any]):
            # fast way to fill an array from a dictionary
            # keys and values can also be tuples of integers
            keys = np.array(list(d.keys()), dtype=int)
            vals = np.array(list(d.values()), dtype=int)
            if keys.ndim == 1:
                arr[keys] = vals
            else:
                arr[tuple(keys.T)] = vals

        spec = mujoco.MjSpec()
        spec.option.disableflags = disableflags
        spec.option.gravity = np.array([*model.gravity.numpy()[0]])
        spec.option.solver = solver
        spec.option.integrator = integrator
        spec.option.iterations = iterations
        spec.option.ls_iterations = ls_iterations
        spec.option.cone = cone
        spec.option.impratio = impratio
        spec.option.tolerance = tolerance
        spec.option.ls_tolerance = ls_tolerance
        spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO

        defaults = spec.default
        if callable(defaults):
            defaults = defaults()
        defaults.geom.solref = (0.02, 1.0)
        defaults.geom.solimp = geom_solimp
        # defaults.geom.contype = 0
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO

        if add_axes:
            # add axes for debug visualization in MuJoCo viewer when loading the generated XML
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                name="axis_x",
                fromto=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                rgba=[1.0, 0.0, 0.0, 1.0],
                size=[0.01, 0.01, 0.01],
                contype=0,
                conaffinity=0,
            )
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                name="axis_y",
                fromto=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                rgba=[0.0, 1.0, 0.0, 1.0],
                size=[0.01, 0.01, 0.01],
                contype=0,
                conaffinity=0,
            )
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                name="axis_z",
                fromto=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                rgba=[0.0, 0.0, 1.0, 1.0],
                size=[0.01, 0.01, 0.01],
                contype=0,
                conaffinity=0,
            )

        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_parent_xform = model.joint_X_p.numpy()
        joint_child_xform = model.joint_X_c.numpy()
        joint_limit_lower = model.joint_limit_lower.numpy()
        joint_limit_upper = model.joint_limit_upper.numpy()
        joint_limit_ke = model.joint_limit_ke.numpy()
        joint_limit_kd = model.joint_limit_kd.numpy()
        joint_type = model.joint_type.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_target_kd = model.joint_target_kd.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_armature = model.joint_armature.numpy()
        joint_effort_limit = model.joint_effort_limit.numpy()
        # MoJoCo doesn't have velocity limit
        # joint_velocity_limit = model.joint_velocity_limit.numpy()
        joint_friction = model.joint_friction.numpy()
        joint_world = model.joint_world.numpy()
        body_mass = model.body_mass.numpy()
        body_inertia = model.body_inertia.numpy()
        body_com = model.body_com.numpy()
        body_world = model.body_world.numpy()
        shape_transform = model.shape_transform.numpy()
        shape_type = model.shape_type.numpy()
        shape_size = model.shape_scale.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_world = model.shape_world.numpy()
        shape_mu = model.shape_material_mu.numpy()
        shape_torsional_friction = model.shape_material_torsional_friction.numpy()
        shape_rolling_friction = model.shape_material_rolling_friction.numpy()

        # retrieve MuJoCo-specific attributes
        mujoco_attrs = getattr(model, "mujoco", None)

        def get_custom_attribute(name: str) -> nparray | None:
            if mujoco_attrs is None:
                return None
            attr = getattr(mujoco_attrs, name, None)
            if attr is None:
                return None
            return attr.numpy()

        shape_condim = get_custom_attribute("condim")
        joint_dof_limit_margin = get_custom_attribute("limit_margin")
        joint_solimp_limit = get_custom_attribute("solimplimit")
        joint_dof_solref = get_custom_attribute("solreffriction")
        joint_dof_solimp = get_custom_attribute("solimpfriction")
        joint_stiffness = get_custom_attribute("dof_passive_stiffness")
        joint_damping = get_custom_attribute("dof_passive_damping")

        eq_constraint_type = model.equality_constraint_type.numpy()
        eq_constraint_body1 = model.equality_constraint_body1.numpy()
        eq_constraint_body2 = model.equality_constraint_body2.numpy()
        eq_constraint_anchor = model.equality_constraint_anchor.numpy()
        eq_constraint_torquescale = model.equality_constraint_torquescale.numpy()
        eq_constraint_relpose = model.equality_constraint_relpose.numpy()
        eq_constraint_joint1 = model.equality_constraint_joint1.numpy()
        eq_constraint_joint2 = model.equality_constraint_joint2.numpy()
        eq_constraint_polycoef = model.equality_constraint_polycoef.numpy()
        eq_constraint_enabled = model.equality_constraint_enabled.numpy()
        eq_constraint_world = model.equality_constraint_world.numpy()

        INT32_MAX = np.iinfo(np.int32).max
        collision_mask_everything = INT32_MAX

        # mapping from joint axis to actuator index
        # axis_to_actuator[i, 0] = position actuator index
        # axis_to_actuator[i, 1] = velocity actuator index
        axis_to_actuator = np.zeros((model.joint_dof_count, 2), dtype=np.int32) - 1
        actuator_count = 0

        # supported non-fixed joint types in MuJoCo (fixed joints are handled by nesting bodies)
        supported_joint_types = {
            JointType.FREE,
            JointType.BALL,
            JointType.PRISMATIC,
            JointType.REVOLUTE,
            JointType.D6,
        }

        geom_type_mapping = {
            GeoType.SPHERE: mujoco.mjtGeom.mjGEOM_SPHERE,
            GeoType.PLANE: mujoco.mjtGeom.mjGEOM_PLANE,
            GeoType.CAPSULE: mujoco.mjtGeom.mjGEOM_CAPSULE,
            GeoType.CYLINDER: mujoco.mjtGeom.mjGEOM_CYLINDER,
            GeoType.BOX: mujoco.mjtGeom.mjGEOM_BOX,
            GeoType.ELLIPSOID: mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            GeoType.MESH: mujoco.mjtGeom.mjGEOM_MESH,
            GeoType.CONVEX_MESH: mujoco.mjtGeom.mjGEOM_MESH,
        }

        mj_bodies = [spec.worldbody]
        # mapping from Newton body id to MuJoCo body id
        body_mapping = {-1: 0}
        # mapping from Newton shape id to MuJoCo geom name
        shape_mapping = {}
        # track mocap index for each Newton body (dict: newton_body_id -> mocap_index)
        newton_body_to_mocap_index = {}
        # counter for assigning sequential mocap indices
        next_mocap_index = 0

        # ensure unique names
        body_name_counts = {}
        joint_names = {}

        # number of shapes which are replicated per world (excludes singular static shapes from a negative group)
        shape_range_len = 0

        if separate_worlds:
            # determine which shapes, bodies and joints belong to the first world
            # based on the shape world: we pick objects from the first world and global shapes
            non_negatives = shape_world[shape_world >= 0]
            if len(non_negatives) > 0:
                first_group = np.min(non_negatives)
                shape_range_len = len(np.where(shape_world == first_group)[0])
            else:
                first_group = -1
                shape_range_len = model.shape_count
            selected_shapes = np.where((shape_world == first_group) | (shape_world < 0))[0]
            selected_bodies = np.where((body_world == first_group) | (body_world < 0))[0]
            selected_joints = np.where((joint_world == first_group) | (joint_world < 0))[0]
            selected_constraints = np.where((eq_constraint_world == first_group) | (eq_constraint_world < 0))[0]
        else:
            # if we are not separating environments to worlds, we use all shapes, bodies, joints
            first_group = 0
            shape_range_len = model.shape_count

            # if we are not separating worlds, we use all shapes, bodies, joints, constraints
            selected_shapes = np.arange(model.shape_count, dtype=np.int32)
            selected_bodies = np.arange(model.body_count, dtype=np.int32)
            selected_joints = np.arange(model.joint_count, dtype=np.int32)
            selected_constraints = np.arange(model.equality_constraint_count, dtype=np.int32)

        # sort joints topologically depth-first since this is the order that will also be used
        # for placing bodies in the MuJoCo model
        joints_simple = list(zip(joint_parent[selected_joints], joint_child[selected_joints], strict=False))
        joint_order = topological_sort(joints_simple, use_dfs=True)
        if any(joint_order[i] != i for i in range(len(joints_simple))):
            warnings.warn(
                "Joint order is not in depth-first topological order while converting Newton model to MuJoCo, this may lead to diverging kinematics between MuJoCo and Newton.",
                stacklevel=2,
            )

        # find graph coloring of collision filter pairs
        # filter out shapes that are not colliding with anything
        colliding_shapes = selected_shapes[shape_flags[selected_shapes] & ShapeFlags.COLLIDE_SHAPES != 0]

        # number of shapes we are instantiating in MuJoCo (which will be replicated for the number of envs)
        colliding_shapes_per_world = len(colliding_shapes)

        # filter out non-colliding bodies using excludes
        body_filters = self.find_body_collision_filter_pairs(
            model,
            selected_bodies,
            colliding_shapes,
        )

        shape_color = self.color_collision_shapes(
            model, colliding_shapes, visualize_graph=False, shape_keys=model.shape_key
        )

        # store selected shapes, bodies, joints for later use in update_geom_properties
        self.selected_shapes = wp.array(selected_shapes, dtype=wp.int32, device=model.device)
        self.selected_joints = wp.array(selected_joints, dtype=wp.int32, device=model.device)
        self.selected_bodies = wp.array(selected_bodies, dtype=wp.int32, device=model.device)
        selected_shapes_set = set(selected_shapes)

        def add_geoms(newton_body_id: int):
            body = mj_bodies[body_mapping[newton_body_id]]
            shapes = model.body_shapes.get(newton_body_id)
            if not shapes:
                return
            for shape in shapes:
                if shape not in selected_shapes_set:
                    # skip shapes that are not selected for this world
                    continue
                # Skip visual-only geoms, but don't skip sites
                is_site = shape_flags[shape] & ShapeFlags.SITE
                if skip_visual_only_geoms and not is_site and not (shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES):
                    continue
                stype = shape_type[shape]
                name = f"{model.shape_key[shape]}_{shape}"

                if is_site:
                    if not include_sites:
                        continue

                    # Map unsupported site types to SPHERE
                    # MuJoCo sites only support: SPHERE, CAPSULE, CYLINDER, BOX
                    supported_site_types = {GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.BOX}
                    site_geom_type = stype if stype in supported_site_types else GeoType.SPHERE

                    tf = wp.transform(*shape_transform[shape])
                    site_params = {
                        "type": geom_type_mapping[site_geom_type],
                        "name": name,
                        "pos": tf.p,
                        "quat": quat_to_mjc(tf.q),
                    }

                    size = shape_size[shape]
                    # Ensure size is valid for the site type
                    if np.any(size > 0.0):
                        nonzero = size[size > 0.0][0]
                        size[size == 0.0] = nonzero
                        site_params["size"] = size
                    else:
                        site_params["size"] = [0.01, 0.01, 0.01]

                    if shape_flags[shape] & ShapeFlags.VISIBLE:
                        site_params["rgba"] = [0.0, 1.0, 0.0, 0.5]
                    else:
                        site_params["rgba"] = [0.0, 1.0, 0.0, 0.0]

                    body.add_site(**site_params)
                    continue

                if stype == GeoType.PLANE and newton_body_id != -1:
                    raise ValueError("Planes can only be attached to static bodies")
                geom_params = {
                    "type": geom_type_mapping[stype],
                    "name": name,
                }
                tf = wp.transform(*shape_transform[shape])
                if stype == GeoType.MESH or stype == GeoType.CONVEX_MESH:
                    mesh_src = model.shape_source[shape]
                    # use mesh-specific maxhullvert or fall back to the default
                    maxhullvert = getattr(mesh_src, "maxhullvert", mesh_maxhullvert)
                    # apply scaling
                    size = shape_size[shape]
                    vertices = mesh_src.vertices * size
                    spec.add_mesh(
                        name=name,
                        uservert=vertices.flatten(),
                        userface=mesh_src.indices.flatten(),
                        maxhullvert=maxhullvert,
                    )
                    geom_params["meshname"] = name
                geom_params["pos"] = tf.p
                geom_params["quat"] = quat_to_mjc(tf.q)
                size = shape_size[shape]
                if np.any(size > 0.0):
                    # duplicate nonzero entries at places where size is 0
                    nonzero = size[size > 0.0][0]
                    size[size == 0.0] = nonzero
                    geom_params["size"] = size
                else:
                    assert stype == GeoType.PLANE, "Only plane shapes are allowed to have a size of zero"
                    # planes are always infinite for collision purposes in mujoco
                    geom_params["size"] = [5.0, 5.0, 5.0]
                    # make ground plane blue in the MuJoCo viewer (only used for debugging)
                    geom_params["rgba"] = [0.0, 0.3, 0.6, 1.0]

                # encode collision filtering information
                if not (shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES):
                    # this shape is not colliding with anything
                    geom_params["contype"] = 0
                    geom_params["conaffinity"] = 0
                else:
                    color = shape_color[shape]
                    if color < 32:
                        contype = 1 << color
                        geom_params["contype"] = contype
                        # collide with anything except shapes from the same color
                        geom_params["conaffinity"] = collision_mask_everything & ~contype

                # set friction from Newton shape materials
                mu = shape_mu[shape]
                torsional = shape_torsional_friction[shape]
                rolling = shape_rolling_friction[shape]
                geom_params["friction"] = [
                    mu,
                    torsional,
                    rolling,
                ]
                if shape_condim is not None:
                    geom_params["condim"] = shape_condim[shape]

                body.add_geom(**geom_params)
                # store the geom name instead of assuming index
                shape_mapping[shape] = name

        # add static geoms attached to the worldbody
        add_geoms(-1)

        # Maps from Newton joint index (per-world/template) to MuJoCo DOF start index (per-world/template)
        # Only populated for template joints; in kernels, use joint_in_world to index
        joint_mjc_dof_start = np.full(len(selected_joints), -1, dtype=np.int32)

        # Maps from Newton DOF index to MuJoCo joint index (first world only)
        # Needed because jnt_solimp/jnt_solref are per-joint (not per-DOF) in MuJoCo
        dof_to_mjc_joint = np.full(model.joint_dof_count // model.num_worlds, -1, dtype=np.int32)

        # need to keep track of current dof and joint counts to make the indexing above correct
        num_dofs = 0
        num_mjc_joints = 0

        # add joints, bodies and geoms
        for ji in joint_order:
            parent, child = joints_simple[ji]
            if child in body_mapping:
                raise ValueError(f"Body {child} already exists in the mapping")

            # add body
            body_mapping[child] = len(mj_bodies)

            # use the correct global joint index
            j = selected_joints[ji]

            # check if fixed-base articulation
            fixed_base = False
            if parent == -1 and joint_type[j] == JointType.FIXED:
                fixed_base = True

            # this assumes that the joint position is 0
            tf = wp.transform(*joint_parent_xform[j])
            tf = tf * wp.transform_inverse(wp.transform(*joint_child_xform[j]))

            jc_xform = wp.transform(*joint_child_xform[j])
            joint_pos = jc_xform.p
            joint_rot = jc_xform.q

            # ensure unique body name
            name = model.body_key[child]
            if name not in body_name_counts:
                body_name_counts[name] = 1
            else:
                while name in body_name_counts:
                    body_name_counts[name] += 1
                    name = f"{name}_{body_name_counts[name]}"

            inertia = body_inertia[child]
            body = mj_bodies[body_mapping[parent]].add_body(
                name=name,
                pos=tf.p,
                quat=quat_to_mjc(tf.q),
                mass=body_mass[child],
                ipos=body_com[child, :],
                fullinertia=[inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]],
                explicitinertial=True,
                mocap=fixed_base,
            )
            mj_bodies.append(body)
            if fixed_base:
                newton_body_to_mocap_index[child] = next_mocap_index
                next_mocap_index += 1

            # add joint
            j_type = joint_type[j]
            qd_start = joint_qd_start[j]
            name = model.joint_key[j]
            if name not in joint_names:
                joint_names[name] = 1
            else:
                while name in joint_names:
                    joint_names[name] += 1
                    name = f"{name}_{joint_names[name]}"

            joint_mjc_dof_start[ji] = num_dofs

            if j_type == JointType.FREE:
                body.add_joint(
                    name=name,
                    type=mujoco.mjtJoint.mjJNT_FREE,
                    damping=0.0,
                    limited=False,
                )
                # For free joints, all 6 DOFs map to the same MuJoCo joint
                for i in range(6):
                    dof_to_mjc_joint[qd_start + i] = num_mjc_joints
                num_dofs += 6
                num_mjc_joints += 1
            elif j_type in supported_joint_types:
                lin_axis_count, ang_axis_count = joint_dof_dim[j]
                num_dofs += lin_axis_count + ang_axis_count

                # linear dofs
                for i in range(lin_axis_count):
                    ai = qd_start + i

                    axis = wp.quat_rotate(joint_rot, wp.vec3(*joint_axis[ai]))

                    joint_params = {
                        "armature": joint_armature[qd_start + i],
                        "pos": joint_pos,
                    }
                    # Set friction
                    joint_params["frictionloss"] = joint_friction[ai]
                    # Set margin if available
                    if joint_dof_limit_margin is not None:
                        joint_params["margin"] = joint_dof_limit_margin[ai]
                    if joint_stiffness is not None:
                        joint_params["stiffness"] = joint_stiffness[ai]
                    if joint_damping is not None:
                        joint_params["damping"] = joint_damping[ai]
                    lower, upper = joint_limit_lower[ai], joint_limit_upper[ai]
                    if lower <= -JOINT_LIMIT_UNLIMITED and upper >= JOINT_LIMIT_UNLIMITED:
                        joint_params["limited"] = False
                    else:
                        joint_params["limited"] = True

                    # we're piping these through unconditionally even though they are only active with limited joints
                    joint_params["range"] = (lower, upper)
                    # Use negative convention for solref_limit: (-stiffness, -damping)
                    if joint_limit_ke[ai] > 0:
                        joint_params["solref_limit"] = (-joint_limit_ke[ai], -joint_limit_kd[ai])
                    if joint_solimp_limit is not None:
                        joint_params["solimp_limit"] = joint_solimp_limit[ai]
                    if joint_dof_solref is not None:
                        joint_params["solref_friction"] = joint_dof_solref[ai]
                    if joint_dof_solimp is not None:
                        joint_params["solimp_friction"] = joint_dof_solimp[ai]
                    axname = name
                    if lin_axis_count > 1 or ang_axis_count > 1:
                        axname += "_lin"
                    if lin_axis_count > 1:
                        axname += str(i)
                    body.add_joint(
                        name=axname,
                        type=mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=axis,
                        **joint_params,
                    )
                    # Map this DOF to the current MuJoCo joint index
                    dof_to_mjc_joint[ai] = num_mjc_joints
                    num_mjc_joints += 1

                    if actuated_axes is None or ai in actuated_axes:
                        kp = joint_target_ke[ai]
                        kd = joint_target_kd[ai]
                        effort_limit = joint_effort_limit[ai]
                        gear = actuator_gears.get(axname)
                        if gear is not None:
                            args = {}
                            args.update(actuator_args)
                            args["gear"] = [gear, 0.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            args = actuator_args
                        # forcerange is defined per actuator, meaning that P and D terms will be clamped separately in PD control and not their sum
                        # is there a similar attribute per joint dof?
                        args["forcerange"] = [-effort_limit, effort_limit]
                        args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                        spec.add_actuator(target=axname, **args)
                        axis_to_actuator[ai, 0] = actuator_count
                        actuator_count += 1

                        args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                        spec.add_actuator(target=axname, **args)
                        axis_to_actuator[ai, 1] = actuator_count
                        actuator_count += 1

                # angular dofs
                for i in range(lin_axis_count, lin_axis_count + ang_axis_count):
                    ai = qd_start + i

                    axis = wp.quat_rotate(joint_rot, wp.vec3(*joint_axis[ai]))

                    joint_params = {
                        "armature": joint_armature[qd_start + i],
                        "pos": joint_pos,
                    }
                    # Set friction
                    joint_params["frictionloss"] = joint_friction[ai]
                    # Set margin if available
                    if joint_dof_limit_margin is not None:
                        joint_params["margin"] = joint_dof_limit_margin[ai]
                    if joint_stiffness is not None:
                        joint_params["stiffness"] = joint_stiffness[ai]
                    if joint_damping is not None:
                        joint_params["damping"] = joint_damping[ai]
                    lower, upper = joint_limit_lower[ai], joint_limit_upper[ai]
                    if lower <= -JOINT_LIMIT_UNLIMITED and upper >= JOINT_LIMIT_UNLIMITED:
                        joint_params["limited"] = False
                    else:
                        joint_params["limited"] = True

                    # we're piping these through unconditionally even though they are only active with limited joints
                    joint_params["range"] = (np.rad2deg(lower), np.rad2deg(upper))
                    # Use negative convention for solref_limit: (-stiffness, -damping)
                    if joint_limit_ke[ai] > 0:
                        joint_params["solref_limit"] = (-joint_limit_ke[ai], -joint_limit_kd[ai])
                    if joint_solimp_limit is not None:
                        joint_params["solimp_limit"] = joint_solimp_limit[ai]
                    if joint_dof_solref is not None:
                        joint_params["solref_friction"] = joint_dof_solref[ai]
                    if joint_dof_solimp is not None:
                        joint_params["solimp_friction"] = joint_dof_solimp[ai]

                    axname = name
                    if lin_axis_count > 1 or ang_axis_count > 1:
                        axname += "_ang"
                    if ang_axis_count > 1:
                        axname += str(i - lin_axis_count)
                    body.add_joint(
                        name=axname,
                        type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=axis,
                        **joint_params,
                    )
                    # Map this DOF to the current MuJoCo joint index
                    dof_to_mjc_joint[ai] = num_mjc_joints
                    num_mjc_joints += 1

                    if actuated_axes is None or ai in actuated_axes:
                        kp = joint_target_ke[ai]
                        kd = joint_target_kd[ai]
                        effort_limit = joint_effort_limit[ai]
                        gear = actuator_gears.get(axname)
                        if gear is not None:
                            args = {}
                            args.update(actuator_args)
                            args["gear"] = [gear, 0.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            args = actuator_args
                        args["forcerange"] = [-effort_limit, effort_limit]
                        args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                        spec.add_actuator(target=axname, **args)
                        axis_to_actuator[ai, 0] = actuator_count
                        actuator_count += 1

                        args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                        spec.add_actuator(target=axname, **args)
                        axis_to_actuator[ai, 1] = actuator_count
                        actuator_count += 1

            elif j_type != JointType.FIXED:
                raise NotImplementedError(f"Joint type {j_type} is not supported yet")

            add_geoms(child)

        for i in selected_constraints:
            constraint_type = eq_constraint_type[i]
            if constraint_type == EqType.CONNECT:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_CONNECT
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.body_key[eq_constraint_body1[i]]
                eq.name2 = model.body_key[eq_constraint_body2[i]]
                eq.data[0:3] = eq_constraint_anchor[i]

            elif constraint_type == EqType.JOINT:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_JOINT)
                eq.type = mujoco.mjtEq.mjEQ_JOINT
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.joint_key[eq_constraint_joint1[i]]
                eq.name2 = model.joint_key[eq_constraint_joint2[i]]
                eq.data[0:5] = eq_constraint_polycoef[i]

            elif constraint_type == EqType.WELD:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.body_key[eq_constraint_body1[i]]
                eq.name2 = model.body_key[eq_constraint_body2[i]]
                cns_relpose = wp.transform(*eq_constraint_relpose[i])
                eq.data[0:3] = eq_constraint_anchor[i]
                eq.data[3:6] = wp.transform_get_translation(cns_relpose)
                eq.data[6:10] = wp.transform_get_rotation(cns_relpose)
                eq.data[10] = eq_constraint_torquescale[i]

        assert len(spec.geoms) == colliding_shapes_per_world, (
            "The number of geoms in the MuJoCo model does not match the number of colliding shapes in the Newton model."
        )

        # add contact exclusions between bodies to ensure parent <> child collisions are ignored
        # even when one of the bodies is static
        for b1, b2 in body_filters:
            mb1, mb2 = body_mapping[b1], body_mapping[b2]
            spec.add_exclude(bodyname1=spec.bodies[mb1].name, bodyname2=spec.bodies[mb2].name)

        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.update_mjc_data(self.mj_data, model, state)

        # fill some MjWarp model fields that are outdated after update_mjc_data.
        # just setting qpos0 to d.qpos leads to weird behavior here, needs
        # to be investigated.

        mujoco.mj_forward(self.mj_model, self.mj_data)

        if target_filename:
            with open(target_filename, "w") as f:
                f.write(spec.to_xml())
                print(f"Saved mujoco model to {os.path.abspath(target_filename)}")

        # now that the model is compiled, get the actual geom indices and compute
        # shape transform corrections
        shape_to_geom_idx = {}
        geom_to_shape_idx = {}
        for shape, geom_name in shape_mapping.items():
            geom_idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_idx >= 0:
                shape_to_geom_idx[shape] = geom_idx
                geom_to_shape_idx[geom_idx] = shape

        with wp.ScopedDevice(model.device):
            # create the MuJoCo Warp model
            self.mjw_model = mujoco_warp.put_model(self.mj_model)

            # patch mjw_model with mesh_pos if it doesn't have it
            if not hasattr(self.mjw_model, "mesh_pos"):
                self.mjw_model.mesh_pos = wp.array(self.mj_model.mesh_pos, dtype=wp.vec3)

            # build the geom index mappings now that we have the actual indices
            # geom_to_shape_idx maps from MuJoCo geom index to absolute Newton shape index.
            # Convert non-static shapes to template-relative indices for the kernel.
            geom_to_shape_idx_np = np.full((self.mj_model.ngeom,), -1, dtype=np.int32)

            # Find the minimum shape index for the first non-static group to use as the base
            first_env_shapes = np.where(shape_world == first_group)[0]
            first_env_shape_base = int(np.min(first_env_shapes)) if len(first_env_shapes) > 0 else 0

            # Per-geom static mask (True if static, False otherwise)
            geom_is_static_np = np.zeros((self.mj_model.ngeom,), dtype=bool)

            for geom_idx, abs_shape_idx in geom_to_shape_idx.items():
                if shape_world[abs_shape_idx] < 0:
                    # Static shape - use absolute index and mark mask
                    geom_to_shape_idx_np[geom_idx] = abs_shape_idx
                    geom_is_static_np[geom_idx] = True
                else:
                    # Non-static shape - convert to template-relative offset from first env base
                    geom_to_shape_idx_np[geom_idx] = abs_shape_idx - first_env_shape_base

            geom_to_shape_idx_wp = wp.array(geom_to_shape_idx_np, dtype=wp.int32)
            geom_is_static_wp = wp.array(geom_is_static_np, dtype=bool)

            # use the actual number of geoms from the MuJoCo model
            self.to_newton_shape_index = wp.full((model.num_worlds, self.mj_model.ngeom), -1, dtype=wp.int32)

            # create mapping from Newton shape index to MuJoCo [world, geom index]
            self.to_mjc_geom_index = wp.full(model.shape_count, -1, dtype=wp.vec2i)

            # mapping from Newton joint index to the start index of its joint axes in MuJoCo
            self.joint_mjc_dof_start = wp.array(joint_mjc_dof_start, dtype=wp.int32)
            # mapping from Newton DOF index to MuJoCo joint index
            self.dof_to_mjc_joint = wp.array(dof_to_mjc_joint, dtype=wp.int32, device=model.device)

            if self.mjw_model.geom_pos.size:
                wp.launch(
                    update_shape_mappings_kernel,
                    dim=(self.model.num_worlds, self.mj_model.ngeom),
                    inputs=[
                        geom_to_shape_idx_wp,
                        geom_is_static_wp,
                        shape_range_len,
                        first_env_shape_base,
                    ],
                    outputs=[
                        self.to_mjc_geom_index,
                        self.to_newton_shape_index,
                    ],
                )

            # mapping from Newton joint axis index to MJC actuator index
            # mjc_axis_to_actuator[i, 0] = position actuator index
            # mjc_axis_to_actuator[i, 1] = velocity actuator index
            self.mjc_axis_to_actuator = wp.array2d(axis_to_actuator, dtype=wp.int32)
            # mapping from MJC body index to Newton body index (skip world index -1)
            to_mjc_body_index = np.fromiter(body_mapping.keys(), dtype=int)[1:] + 1
            self.to_mjc_body_index = wp.array(to_mjc_body_index, dtype=wp.int32)

            # create mapping from Newton body index to mocap index (-1 if not mocap)
            newton_body_indices = np.fromiter(body_mapping.keys(), dtype=int)[1:]
            body_to_mocap = np.array(
                [newton_body_to_mocap_index.get(idx, -1) for idx in newton_body_indices], dtype=np.int32
            )
            self.newton_body_to_mocap_index = wp.array(body_to_mocap, dtype=wp.int32)

            # set mjwarp-only settings
            self.mjw_model.opt.ls_parallel = ls_parallel

            if separate_worlds:
                nworld = model.num_worlds
            else:
                nworld = 1

            # TODO find better heuristics to determine nconmax and njmax
            if disable_contacts:
                nconmax = 0
            else:
                if nconmax is not None:
                    rigid_contact_max = nconmax
                    if rigid_contact_max < self.mj_data.ncon:
                        warnings.warn(
                            f"[WARNING] Value for nconmax is changed from {nconmax} to {self.mj_data.ncon} following an MjWarp requirement.",
                            stacklevel=2,
                        )
                        nconmax = self.mj_data.ncon
                    else:
                        nconmax = rigid_contact_max
                else:
                    nconmax = self.mj_data.ncon

            if njmax is not None:
                if njmax < self.mj_data.nefc:
                    warnings.warn(
                        f"[WARNING] Value for njmax is changed from {njmax} to {self.mj_data.nefc} following an MjWarp requirement.",
                        stacklevel=2,
                    )
                    njmax = self.mj_data.nefc
            else:
                njmax = self.mj_data.nefc

            self.mjw_data = mujoco_warp.put_data(
                self.mj_model,
                self.mj_data,
                nworld=nworld,
                nconmax=nconmax,
                njmax=njmax,
            )

            # expand model fields that can be expanded:
            self.expand_model_fields(self.mjw_model, nworld)

            # so far we have only defined the first world,
            # now complete the data from the Newton model
            self.notify_model_changed(SolverNotifyFlags.ALL)

    def expand_model_fields(self, mj_model: MjWarpModel, nworld: int):
        if nworld == 1:
            return

        model_fields_to_expand = {
            # "qpos0",
            # "qpos_spring",
            "body_pos",
            "body_quat",
            "body_ipos",
            "body_iquat",
            "body_mass",
            # "body_subtreemass",
            "body_inertia",
            # "body_invweight0",
            "body_gravcomp",
            "jnt_solref",
            "jnt_solimp",
            "jnt_pos",
            "jnt_axis",
            "jnt_stiffness",
            "jnt_range",
            # "jnt_actfrcrange",
            "jnt_margin",  # corresponds to newton custom attribute "limit_margin"
            "dof_armature",
            "dof_damping",
            # "dof_invweight0",
            "dof_frictionloss",
            "dof_solimp",
            "dof_solref",
            # "geom_matid",
            # "geom_solmix",
            "geom_solref",
            # "geom_solimp",
            "geom_size",
            "geom_rbound",
            "geom_pos",
            "geom_quat",
            "geom_friction",
            # "geom_margin",
            # "geom_gap",
            # "geom_rgba",
            # "site_pos",
            # "site_quat",
            # "cam_pos",
            # "cam_quat",
            # "cam_poscom0",
            # "cam_pos0",
            # "cam_mat0",
            # "light_pos",
            # "light_dir",
            # "light_poscom0",
            # "light_pos0",
            # "eq_solref",
            # "eq_solimp",
            # "eq_data",
            # "actuator_dynprm",
            "actuator_gainprm",
            "actuator_biasprm",
            # "actuator_ctrlrange",
            "actuator_forcerange",
            # "actuator_actrange",
            # "actuator_gear",
            # "pair_solref",
            # "pair_solreffriction",
            # "pair_solimp",
            # "pair_margin",
            # "pair_gap",
            # "pair_friction",
            # "tendon_solref_lim",
            # "tendon_solimp_lim",
            # "tendon_range",
            # "tendon_margin",
            # "tendon_length0",
            # "tendon_invweight0",
            # "mat_rgba",
        }

        def tile(x: wp.array):
            # Create new array with same shape but first dim multiplied by nworld
            new_shape = list(x.shape)
            new_shape[0] = nworld
            wp_array = {1: wp.array, 2: wp.array2d, 3: wp.array3d, 4: wp.array4d}[len(new_shape)]
            dst = wp_array(shape=new_shape, dtype=x.dtype, device=x.device)

            # Flatten arrays for kernel
            src_flat = x.flatten()
            dst_flat = dst.flatten()

            # Launch kernel to repeat data - one thread per destination element
            n_elems_per_world = dst_flat.shape[0] // nworld
            wp.launch(
                repeat_array_kernel,
                dim=dst_flat.shape[0],
                inputs=[src_flat, n_elems_per_world],
                outputs=[dst_flat],
                device=x.device,
            )
            return dst

        for field in mj_model.__dataclass_fields__:
            if field in model_fields_to_expand:
                array = getattr(mj_model, field)
                setattr(mj_model, field, tile(array))

    def update_model_inertial_properties(self):
        if self.model.body_count == 0:
            return

        bodies_per_world = self.model.body_count // self.model.num_worlds

        # Get gravcomp if available
        mujoco_attrs = getattr(self.model, "mujoco", None)
        gravcomp = getattr(mujoco_attrs, "gravcomp", None) if mujoco_attrs is not None else None

        wp.launch(
            update_body_mass_ipos_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_com,
                self.model.body_mass,
                gravcomp,
                bodies_per_world,
                self.model.up_axis,
                self.to_mjc_body_index,
            ],
            outputs=[
                self.mjw_model.body_ipos,
                self.mjw_model.body_mass,
                self.mjw_model.body_gravcomp,
            ],
            device=self.model.device,
        )

        wp.launch(
            update_body_inertia_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_inertia,
                bodies_per_world,
                self.to_mjc_body_index,
            ],
            outputs=[self.mjw_model.body_inertia, self.mjw_model.body_iquat],
            device=self.model.device,
        )

    def update_joint_dof_properties(self):
        """Update all joint DOF properties including effort limits, friction, armature, solimplimit, solref, passive stiffness and damping, and joint limit ranges in the MuJoCo model."""
        if self.model.joint_dof_count == 0:
            return

        joints_per_world = self.model.joint_count // self.model.num_worlds
        dofs_per_world = self.model.joint_dof_count // self.model.num_worlds

        # Update actuator force ranges (effort limits) if actuators exist
        if self.mjc_axis_to_actuator is not None:
            wp.launch(
                update_axis_properties_kernel,
                dim=self.model.joint_dof_count,
                inputs=[
                    self.model.joint_target_ke,
                    self.model.joint_target_kd,
                    self.model.joint_effort_limit,
                    self.mjc_axis_to_actuator,
                    dofs_per_world,
                ],
                outputs=[
                    self.mjw_model.actuator_biasprm,
                    self.mjw_model.actuator_gainprm,
                    self.mjw_model.actuator_forcerange,
                ],
                device=self.model.device,
            )

        # Update DOF properties (armature, friction, passive stiffness and damping, and solimplimit) with proper DOF mapping
        mujoco_attrs = getattr(self.model, "mujoco", None)
        solimplimit = getattr(mujoco_attrs, "solimplimit", None) if mujoco_attrs is not None else None
        dof_solref = getattr(mujoco_attrs, "solreffriction", None) if mujoco_attrs is not None else None
        dof_solimp = getattr(mujoco_attrs, "solimpfriction", None) if mujoco_attrs is not None else None
        joint_dof_limit_margin = getattr(mujoco_attrs, "limit_margin", None) if mujoco_attrs is not None else None
        joint_stiffness = getattr(mujoco_attrs, "dof_passive_stiffness", None) if mujoco_attrs is not None else None
        joint_damping = getattr(mujoco_attrs, "dof_passive_damping", None) if mujoco_attrs is not None else None
        wp.launch(
            update_joint_dof_properties_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.joint_mjc_dof_start,
                self.dof_to_mjc_joint,
                self.model.joint_armature,
                self.model.joint_friction,
                self.model.joint_limit_ke,
                self.model.joint_limit_kd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                solimplimit,
                dof_solref,
                dof_solimp,
                joint_stiffness,
                joint_damping,
                joint_dof_limit_margin,
                joints_per_world,
            ],
            outputs=[
                self.mjw_model.dof_armature,
                self.mjw_model.dof_frictionloss,
                self.mjw_model.jnt_solimp,
                self.mjw_model.dof_solref,
                self.mjw_model.dof_solimp,
                self.mjw_model.jnt_solref,
                self.mjw_model.jnt_stiffness,
                self.mjw_model.dof_damping,
                self.mjw_model.jnt_margin,
                self.mjw_model.jnt_range,
            ],
            device=self.model.device,
        )

    def update_joint_properties(self):
        """Update joint properties including joint positions, joint axes, and relative body transforms in the MuJoCo model."""
        if self.model.joint_count == 0:
            return

        joints_per_world = self.model.joint_count // self.model.num_worlds

        # Update joint positions, joint axes, and relative body transforms
        wp.launch(
            update_joint_transforms_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.model.joint_X_p,
                self.model.joint_X_c,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
                self.model.joint_axis,
                self.model.joint_child,
                self.model.joint_type,
                self.dof_to_mjc_joint,
                self.to_mjc_body_index,
                self.newton_body_to_mocap_index,
                joints_per_world,
            ],
            outputs=[
                self.mjw_model.jnt_pos,
                self.mjw_model.jnt_axis,
                self.mjw_model.body_pos,
                self.mjw_model.body_quat,
                self.mjw_data.mocap_pos,
                self.mjw_data.mocap_quat,
            ],
            device=self.model.device,
        )

    def update_geom_properties(self):
        """Update geom properties including collision radius, friction, and contact parameters in the MuJoCo model."""

        # Get number of geoms and worlds from MuJoCo model
        num_geoms = self.mj_model.ngeom
        if num_geoms == 0:
            return

        num_worlds = self.model.num_worlds

        wp.launch(
            update_geom_properties_kernel,
            dim=(num_worlds, num_geoms),
            inputs=[
                self.model.shape_collision_radius,
                self.model.shape_material_mu,
                self.model.shape_material_ke,
                self.model.shape_material_kd,
                self.model.shape_scale,
                self.model.shape_transform,
                self.to_newton_shape_index,
                self.mjw_model.geom_type,
                self._mujoco.mjtGeom.mjGEOM_MESH,
                self.mjw_model.geom_dataid,
                self.mjw_model.mesh_pos,
                self.mjw_model.mesh_quat,
                self.model.shape_material_torsional_friction,
                self.model.shape_material_rolling_friction,
            ],
            outputs=[
                self.mjw_model.geom_rbound,
                self.mjw_model.geom_friction,
                self.mjw_model.geom_solref,
                self.mjw_model.geom_size,
                self.mjw_model.geom_pos,
                self.mjw_model.geom_quat,
            ],
            device=self.model.device,
        )

    def update_model_properties(self):
        """Update model properties including gravity in the MuJoCo model."""
        if self.use_mujoco_cpu:
            self.mj_model.opt.gravity[:] = np.array([*self.model.gravity.numpy()[0]])
        else:
            if hasattr(self, "mjw_data"):
                wp.launch(
                    kernel=update_model_properties_kernel,
                    dim=self.mjw_data.nworld,
                    inputs=[
                        self.model.gravity,
                    ],
                    outputs=[
                        self.mjw_model.opt.gravity,
                    ],
                    device=self.model.device,
                )

    def render_mujoco_viewer(
        self,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
        show_contact_points: bool = True,
        show_contact_forces: bool = False,
        show_transparent_geoms: bool = True,
    ):
        """Create and synchronize the MuJoCo viewer.
        The viewer will be created if it is not already open.

        .. note::

            The MuJoCo viewer only supports rendering Newton models with a single world,
            unless :attr:`use_mujoco_cpu` is :obj:`True` or the solver was initialized with
            :attr:`separate_worlds` set to :obj:`False`.

            The MuJoCo viewer is only meant as a debugging tool.

        Args:
            show_left_ui: Whether to show the left UI.
            show_right_ui: Whether to show the right UI.
            show_contact_points: Whether to show contact points.
            show_contact_forces: Whether to show contact forces.
            show_transparent_geoms: Whether to show transparent geoms.
        """
        if self._viewer is None:
            import mujoco  # noqa: PLC0415
            import mujoco.viewer  # noqa: PLC0415

            # make the headlights brighter to improve visibility
            # in the MuJoCo viewer
            self.mj_model.vis.headlight.ambient[:] = [0.3, 0.3, 0.3]
            self.mj_model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
            self.mj_model.vis.headlight.specular[:] = [0.9, 0.9, 0.9]

            self._viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, show_left_ui=show_left_ui, show_right_ui=show_right_ui
            )
            # Enter the context manager to keep the viewer alive
            self._viewer.__enter__()

            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contact_points
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_contact_forces
            self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = show_transparent_geoms

        if self._viewer.is_running():
            if not self.use_mujoco_cpu:
                self._mujoco_warp.get_data_into(self.mj_data, self.mj_model, self.mjw_data)

            self._viewer.sync()

    def close_mujoco_viewer(self):
        """Close the MuJoCo viewer if it exists."""
        if hasattr(self, "_viewer") and self._viewer is not None:
            try:
                self._viewer.__exit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._viewer = None

    def __del__(self):
        """Cleanup method to close the viewer when the solver is destroyed."""
        self.close_mujoco_viewer()
