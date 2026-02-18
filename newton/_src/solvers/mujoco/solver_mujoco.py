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
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...core.types import MAXVAL, nparray, override, vec5, vec10
from ...geometry import GeoType, Mesh, ShapeFlags
from ...sim import (
    ActuatorMode,
    Contacts,
    Control,
    EqType,
    JointType,
    Model,
    ModelBuilder,
    State,
)
from ...sim.graph_coloring import color_graph, plot_graph
from ...utils import topological_sort
from ...utils.benchmark import event_scope
from ...utils.import_utils import string_to_warp
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .kernels import (
    apply_mjc_body_f_kernel,
    apply_mjc_control_kernel,
    apply_mjc_free_joint_f_to_body_f_kernel,
    apply_mjc_qfrc_kernel,
    convert_body_xforms_to_warp_kernel,
    convert_mj_coords_to_warp_kernel,
    convert_mjw_contacts_to_newton_kernel,
    convert_newton_contacts_to_mjwarp_kernel,
    convert_rigid_forces_from_mj_kernel,
    convert_solref,
    convert_warp_coords_to_mj_kernel,
    create_inverse_shape_mapping_kernel,
    eval_articulation_fk,
    repeat_array_kernel,
    update_axis_properties_kernel,
    update_body_inertia_kernel,
    update_body_mass_ipos_kernel,
    update_ctrl_direct_actuator_properties_kernel,
    update_dof_properties_kernel,
    update_eq_data_and_active_kernel,
    update_eq_properties_kernel,
    update_geom_properties_kernel,
    update_jnt_properties_kernel,
    update_joint_transforms_kernel,
    update_mimic_eq_data_and_active_kernel,
    update_mocap_transforms_kernel,
    update_model_properties_kernel,
    update_pair_properties_kernel,
    update_shape_mappings_kernel,
    update_solver_options_kernel,
    update_tendon_properties_kernel,
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

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency


class SolverMuJoCo(SolverBase):
    """
    This solver provides an interface to simulate physics using the `MuJoCo <https://github.com/google-deepmind/mujoco>`_ physics engine,
    optimized with GPU acceleration through `mujoco_warp <https://github.com/google-deepmind/mujoco_warp>`_. It supports both MuJoCo and
    mujoco_warp backends, enabling efficient simulation of articulated systems with
    contacts and constraints.

    .. note::

        - This solver requires `mujoco_warp`_ and its dependencies to be installed.
        - For installation instructions, see the `mujoco_warp`_ repository.
        - ``shape_collision_radius`` from Newton models is not used by MuJoCo. Instead, MuJoCo computes
          bounding sphere radii (``geom_rbound``) internally based on the geometry definition.

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

    class CtrlSource(IntEnum):
        """Control source for MuJoCo actuators.

        Determines where an actuator gets its control input from:

        - :attr:`JOINT_TARGET`: Maps from Newton's :attr:`~newton.Control.joint_target_pos`/:attr:`~newton.Control.joint_target_vel` arrays
        - :attr:`CTRL_DIRECT`: Uses ``control.mujoco.ctrl`` directly (for MuJoCo-native control)
        """

        JOINT_TARGET = 0
        CTRL_DIRECT = 1

    class CtrlType(IntEnum):
        """Control type for MuJoCo actuators.

        For :attr:`~newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET` mode, determines which target array to read from:

        - :attr:`POSITION`: Maps from :attr:`~newton.Control.joint_target_pos`, syncs gains from
          :attr:`~newton.Control.joint_target_ke`. For :attr:`~newton.ActuatorMode.POSITION`-only actuators,
          also syncs damping from :attr:`~newton.Control.joint_target_kd`. For
          :attr:`~newton.ActuatorMode.POSITION_VELOCITY` mode, kd is handled by the separate velocity actuator.
        - :attr:`VELOCITY`: Maps from :attr:`~newton.Control.joint_target_vel`, syncs gains from :attr:`~newton.Control.joint_target_kd`
        - :attr:`GENERAL`: Used with :attr:`~newton.solvers.SolverMuJoCo.CtrlSource.CTRL_DIRECT` mode for motor/general actuators
        """

        POSITION = 0
        VELOCITY = 1
        GENERAL = 2

    # Class variables to cache the imported modules
    _mujoco = None
    _mujoco_warp = None

    @classmethod
    def import_mujoco(cls):
        """Import the MuJoCo Warp dependencies and cache them as class variables."""
        if cls._mujoco is None or cls._mujoco_warp is None:
            try:
                with warnings.catch_warnings():
                    # Set a filter to make all ImportWarnings "always" appear
                    # This is useful to debug import errors on Windows, for example
                    warnings.simplefilter("always", category=ImportWarning)

                    import mujoco
                    import mujoco_warp

                    cls._mujoco = mujoco
                    cls._mujoco_warp = mujoco_warp
            except ImportError as e:
                raise ImportError(
                    "MuJoCo backend not installed. Please refer to https://github.com/google-deepmind/mujoco_warp for installation instructions."
                ) from e
        return cls._mujoco, cls._mujoco_warp

    @staticmethod
    def _parse_integrator(value: str | int, context: dict[str, Any] | None = None) -> int:
        """Parse integrator option: Euler=0, RK4=1, implicit=2, implicitfast=3."""
        if not isinstance(value, str):
            return int(value)
        mapping = {"euler": 0, "rk4": 1, "implicit": 2, "implicitfast": 3}
        lower_value = value.lower().strip()
        if lower_value in mapping:
            return mapping[lower_value]
        return int(value)

    @staticmethod
    def _parse_solver(value: str | int, context: dict[str, Any] | None = None) -> int:
        """Parse solver option: CG=1, Newton=2. PGS (0) is not supported."""
        if not isinstance(value, str):
            return int(value)
        mapping = {"cg": 1, "newton": 2}
        lower_value = value.lower().strip()
        if lower_value in mapping:
            return mapping[lower_value]
        return int(value)

    @staticmethod
    def _parse_cone(value: str | int, context: dict[str, Any] | None = None) -> int:
        """Parse cone option: pyramidal=0, elliptic=1."""
        if not isinstance(value, str):
            return int(value)
        mapping = {"pyramidal": 0, "elliptic": 1}
        lower_value = value.lower().strip()
        if lower_value in mapping:
            return mapping[lower_value]
        return int(value)

    @staticmethod
    def _parse_jacobian(value: str | int, context: dict[str, Any] | None = None) -> int:
        """Parse jacobian option: dense=0, sparse=1, auto=2."""
        if not isinstance(value, str):
            return int(value)
        mapping = {"dense": 0, "sparse": 1, "auto": 2}
        lower_value = value.lower().strip()
        if lower_value in mapping:
            return mapping[lower_value]
        return int(value)

    @staticmethod
    def _angle_value_transformer(value: str, context: dict[str, Any] | None) -> float:
        """Transform angle values from MJCF, converting deg to rad for angular joints.

        For attributes like springref and ref that represent angles,
        parses the string value and multiplies by pi/180 when use_degrees=True and joint is angular.
        """
        parsed = string_to_warp(value, wp.float32, 0.0)
        if context is not None:
            joint_type = context.get("joint_type")
            use_degrees = context.get("use_degrees", False)
            is_angular = joint_type in ["hinge", "ball"]
            if is_angular and use_degrees:
                return parsed * (np.pi / 180)
        return parsed

    @staticmethod
    def _per_angle_value_transformer(value: str, context: dict[str, Any] | None) -> float:
        """Transform per-angle values from MJCF, converting Nm/deg to Nm/rad for angular joints.

        For attributes like stiffness (Nm/rad) and damping (Nm·s/rad) that have angle in the denominator,
        parses the string value and multiplies by 180/pi when use_degrees=True and joint is angular.
        """
        parsed = string_to_warp(value, wp.float32, 0.0)
        if context is not None:
            joint_type = context.get("joint_type")
            use_degrees = context.get("use_degrees", False)
            is_angular = joint_type in ["hinge", "ball"]
            if is_angular and use_degrees:
                return parsed * (180 / np.pi)
        return parsed

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
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=3,
                namespace="mujoco",
                usd_attribute_name="mjc:condim",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="geom_priority",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                usd_attribute_name="mjc:priority",
                mjcf_attribute_name="priority",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="geom_solimp",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimp",
                mjcf_attribute_name="solimp",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="geom_solmix",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0,
                namespace="mujoco",
                usd_attribute_name="mjc:solmix",
                mjcf_attribute_name="solmix",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="geom_gap",
                frequency=AttributeFrequency.SHAPE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:gap",
                mjcf_attribute_name="gap",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="limit_margin",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
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
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimplimit",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solreffriction",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solreffriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solimpfriction",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimpfriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="gravcomp",
                frequency=AttributeFrequency.BODY,
                assignment=AttributeAssignment.MODEL,
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
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:stiffness",
                mjcf_attribute_name="stiffness",
                mjcf_value_transformer=cls._per_angle_value_transformer,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_passive_damping",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:damping",
                mjcf_attribute_name="damping",
                mjcf_value_transformer=cls._per_angle_value_transformer,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_springref",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:springref",
                mjcf_attribute_name="springref",
                mjcf_value_transformer=cls._angle_value_transformer,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_ref",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:ref",
                mjcf_attribute_name="ref",
                mjcf_value_transformer=cls._angle_value_transformer,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="jnt_actgravcomp",
                frequency=AttributeFrequency.JOINT_DOF,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.bool,
                default=False,
                namespace="mujoco",
                usd_attribute_name="mjc:actuatorgravcomp",
                mjcf_attribute_name="actuatorgravcomp",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="eq_solref",
                frequency=AttributeFrequency.EQUALITY_CONSTRAINT,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solref",
                mjcf_attribute_name="solref",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="eq_solimp",
                frequency=AttributeFrequency.EQUALITY_CONSTRAINT,
                assignment=AttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimp",
                mjcf_attribute_name="solimp",
            )
        )
        # Solver options (frequency WORLD for per-world values)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="impratio",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0,
                namespace="mujoco",
                usd_attribute_name="mjc:option:impratio",
                mjcf_attribute_name="impratio",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tolerance",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-8,
                namespace="mujoco",
                usd_attribute_name="mjc:option:tolerance",
                mjcf_attribute_name="tolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ls_tolerance",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.01,
                namespace="mujoco",
                usd_attribute_name="mjc:option:ls_tolerance",
                mjcf_attribute_name="ls_tolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ccd_tolerance",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1e-6,
                namespace="mujoco",
                usd_attribute_name="mjc:option:ccd_tolerance",
                mjcf_attribute_name="ccd_tolerance",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="density",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:option:density",
                mjcf_attribute_name="density",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="viscosity",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:option:viscosity",
                mjcf_attribute_name="viscosity",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="wind",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0, 0.0, 0.0),
                namespace="mujoco",
                usd_attribute_name="mjc:option:wind",
                mjcf_attribute_name="wind",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="magnetic",
                frequency=AttributeFrequency.WORLD,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0, -0.5, 0.0),
                namespace="mujoco",
                usd_attribute_name="mjc:option:magnetic",
                mjcf_attribute_name="magnetic",
            )
        )

        # Solver options (frequency ONCE for single value shared across all worlds)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="iterations",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=100,
                namespace="mujoco",
                usd_attribute_name="mjc:option:iterations",
                mjcf_attribute_name="iterations",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ls_iterations",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=50,
                namespace="mujoco",
                usd_attribute_name="mjc:option:ls_iterations",
                mjcf_attribute_name="ls_iterations",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ccd_iterations",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=35,  # MuJoCo default
                namespace="mujoco",
                usd_attribute_name="mjc:option:ccd_iterations",
                mjcf_attribute_name="ccd_iterations",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="sdf_iterations",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=10,
                namespace="mujoco",
                usd_attribute_name="mjc:option:sdf_iterations",
                mjcf_attribute_name="sdf_iterations",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="sdf_initpoints",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=40,
                namespace="mujoco",
                usd_attribute_name="mjc:option:sdf_initpoints",
                mjcf_attribute_name="sdf_initpoints",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="integrator",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=3,  # Newton default: implicitfast (not MuJoCo's 0/Euler)
                namespace="mujoco",
                usd_attribute_name="mjc:option:integrator",
                mjcf_attribute_name="integrator",
                mjcf_value_transformer=cls._parse_integrator,
                usd_value_transformer=cls._parse_integrator,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solver",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,  # Newton
                namespace="mujoco",
                usd_attribute_name="mjc:option:solver",
                mjcf_attribute_name="solver",
                mjcf_value_transformer=cls._parse_solver,
                usd_value_transformer=cls._parse_solver,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="cone",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,  # pyramidal
                namespace="mujoco",
                usd_attribute_name="mjc:option:cone",
                mjcf_attribute_name="cone",
                mjcf_value_transformer=cls._parse_cone,
                usd_value_transformer=cls._parse_cone,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="jacobian",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,  # auto
                namespace="mujoco",
                usd_attribute_name="mjc:option:jacobian",
                mjcf_attribute_name="jacobian",
                mjcf_value_transformer=cls._parse_jacobian,
                usd_value_transformer=cls._parse_jacobian,
            )
        )

        # --- Pair attributes (from MJCF <pair> tag) ---
        # Explicit contact pairs with custom properties. Only pairs from the template world are used.
        # These are parsed automatically from MJCF <contact><pair> elements.
        # All pair attributes share the "pair" custom frequency (resolves to "mujoco:pair" via namespace).
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_world",
                frequency="pair",  # Resolves to "mujoco:pair" via namespace
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                references="world",
                # No mjcf_attribute_name - this is set automatically during parsing
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_geom1",
                frequency="pair",
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="shape",
                mjcf_attribute_name="geom1",  # Maps to shape index via geom name lookup
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_geom2",
                frequency="pair",
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="shape",
                mjcf_attribute_name="geom2",  # Maps to shape index via geom name lookup
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_condim",
                frequency="pair",
                dtype=wp.int32,
                default=3,
                namespace="mujoco",
                mjcf_attribute_name="condim",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_solref",
                frequency="pair",
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="solref",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_solreffriction",
                frequency="pair",
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="solreffriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_solimp",
                frequency="pair",
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                mjcf_attribute_name="solimp",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_margin",
                frequency="pair",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="margin",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_gap",
                frequency="pair",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="gap",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_friction",
                frequency="pair",
                dtype=vec5,
                default=vec5(1.0, 1.0, 0.005, 0.0001, 0.0001),
                namespace="mujoco",
                mjcf_attribute_name="friction",
            )
        )

        # --- MuJoCo General Actuator attributes (mujoco:actuator frequency) ---
        # These are used for general/motor actuators parsed from MJCF
        # All actuator attributes share the "actuator" custom frequency (resolves to "mujoco:actuator" via namespace)
        # Note: actuator_trnid[0] stores the target index, actuator_trntype determines its meaning (joint/tendon/site)
        def parse_trntype(s: str, _context: dict[str, Any] | None = None) -> int:
            return {"joint": 0, "jointinparent": 1, "tendon": 2, "site": 3, "body": 4, "slidercrank": 5}.get(
                s.lower(), 0
            )

        def parse_dyntype(s: str, _context: dict[str, Any] | None = None) -> int:
            return {"none": 0, "integrator": 1, "filter": 2, "filterexact": 3, "muscle": 4, "user": 5}.get(s.lower(), 0)

        def parse_gaintype(s: str, _context: dict[str, Any] | None = None) -> int:
            return {"fixed": 0, "affine": 1, "muscle": 2, "user": 3}.get(s.lower(), 0)

        def parse_biastype(s: str, _context: dict[str, Any] | None = None) -> int:
            return {"none": 0, "affine": 1, "muscle": 2, "user": 3}.get(s.lower(), 0)

        def parse_bool_int(s: str, _context: dict[str, Any] | None = None) -> int:
            """Parse MJCF boolean values to int (0 or 1)."""
            s = s.strip().lower()
            return 1 if s in ("true", "1") else 0

        def parse_limited(value: str, context: dict[str, Any] | None = None) -> int:
            """Parse MuJoCo limited attribute: false=0, true=1, auto=2."""
            v = value.lower().strip()
            if v in ("false", "0"):
                return 0
            if v in ("true", "1"):
                return 1
            if v in ("auto", "2"):
                return 2
            return int(value)

        def parse_presence(_value: str, _context: dict[str, Any] | None = None) -> int:
            """Return 1 to indicate the attribute was explicitly present in the MJCF."""
            return 1

        # Compiler option (frequency ONCE for single value shared across all worlds)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="autolimits",
                frequency=AttributeFrequency.ONCE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=1,  # MuJoCo default: true
                namespace="mujoco",
                mjcf_value_transformer=parse_bool_int,
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_trntype",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,  # TrnType.JOINT
                namespace="mujoco",
                mjcf_attribute_name="trntype",
                mjcf_value_transformer=parse_trntype,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_dyntype",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,  # DynType.NONE
                namespace="mujoco",
                mjcf_attribute_name="dyntype",
                mjcf_value_transformer=parse_dyntype,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_gaintype",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,  # GainType.FIXED
                namespace="mujoco",
                mjcf_attribute_name="gaintype",
                mjcf_value_transformer=parse_gaintype,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_biastype",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,  # BiasType.NONE
                namespace="mujoco",
                mjcf_attribute_name="biastype",
                mjcf_value_transformer=parse_biastype,
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_trnid",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2i,
                default=wp.vec2i(-1, -1),
                namespace="mujoco",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_world",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="world",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_ctrllimited",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,  # 0=false, 1=true, 2=auto
                namespace="mujoco",
                mjcf_attribute_name="ctrllimited",
                mjcf_value_transformer=parse_limited,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_forcelimited",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,  # 0=false, 1=true, 2=auto
                namespace="mujoco",
                mjcf_attribute_name="forcelimited",
                mjcf_value_transformer=parse_limited,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_ctrlrange",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(-1.0, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="ctrlrange",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_has_ctrlrange",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                mjcf_attribute_name="ctrlrange",
                mjcf_value_transformer=parse_presence,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_forcerange",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(-1.0, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="forcerange",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_has_forcerange",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                mjcf_attribute_name="forcerange",
                mjcf_value_transformer=parse_presence,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_gear",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.types.vector(length=6, dtype=wp.float32),
                default=wp.types.vector(length=6, dtype=wp.float32)(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="gear",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_dynprm",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=vec10,
                default=vec10(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="dynprm",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_gainprm",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=vec10,
                default=vec10(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="gainprm",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_biasprm",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=vec10,
                default=vec10(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="biasprm",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_actlimited",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=2,  # 0=false, 1=true, 2=auto
                namespace="mujoco",
                mjcf_attribute_name="actlimited",
                mjcf_value_transformer=parse_limited,
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_actrange",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="actrange",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_has_actrange",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                mjcf_attribute_name="actrange",
                mjcf_value_transformer=parse_presence,
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_actdim",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                mjcf_attribute_name="actdim",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="actuator_actearly",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                mjcf_attribute_name="actearly",
                mjcf_value_transformer=parse_bool_int,
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ctrl",
                frequency="actuator",
                assignment=AttributeAssignment.CONTROL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ctrl_source",
                frequency="actuator",
                assignment=AttributeAssignment.MODEL,
                dtype=wp.int32,
                default=int(SolverMuJoCo.CtrlSource.CTRL_DIRECT),
                namespace="mujoco",
            )
        )

        # --- Fixed Tendon attributes (variable-length, from MJCF <tendon><fixed> tag) ---
        # Fixed tendons compute length as a linear combination of joint positions.
        # Only tendons from the template world are used; MuJoCo replicates them across worlds.

        # Tendon-level attributes (one per tendon)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_world",
                frequency="tendon",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                references="world",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_stiffness",
                frequency="tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="stiffness",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_damping",
                frequency="tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="damping",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_frictionloss",
                frequency="tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="frictionloss",
            )
        )

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_limited",
                frequency="tendon",
                dtype=wp.int32,
                default=2,  # 0=false, 1=true, 2=auto
                namespace="mujoco",
                mjcf_attribute_name="limited",
                mjcf_value_transformer=parse_limited,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_range",
                frequency="tendon",
                dtype=wp.vec2,
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="range",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_margin",
                frequency="tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="margin",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_solref_limit",
                frequency="tendon",
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="solreflimit",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_solimp_limit",
                frequency="tendon",
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                mjcf_attribute_name="solimplimit",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_solref_friction",
                frequency="tendon",
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                mjcf_attribute_name="solreffriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_solimp_friction",
                frequency="tendon",
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                mjcf_attribute_name="solimpfriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_armature",
                frequency="tendon",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="armature",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_springlength",
                frequency="tendon",
                dtype=wp.vec2,
                default=wp.vec2(-1.0, -1.0),  # -1 means use default (model length)
                namespace="mujoco",
                mjcf_attribute_name="springlength",
            )
        )
        # Addressing into joint arrays (one per tendon)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_joint_adr",
                frequency="tendon",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
                references="mujoco:tendon_joint",  # Offset by joint entry count during merge
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_joint_num",
                frequency="tendon",
                dtype=wp.int32,
                default=0,
                namespace="mujoco",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_actuator_force_range",
                frequency="tendon",
                dtype=wp.vec2,
                default=wp.vec2(0.0, 0.0),
                namespace="mujoco",
                mjcf_attribute_name="actuatorfrcrange",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_actuator_force_limited",
                frequency="tendon",
                dtype=wp.int32,
                default=2,  # 0=false, 1=true, 2=auto
                namespace="mujoco",
                mjcf_attribute_name="actuatorfrclimited",
                mjcf_value_transformer=parse_limited,
            )
        )
        # Tendon names (string attribute - stored as list[str], not warp array)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_key",
                frequency="tendon",
                dtype=str,
                default="",
                namespace="mujoco",
                mjcf_attribute_name="name",
            )
        )

        # Joint arrays (one entry per joint in a fixed tendon's linear combination)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_joint",
                frequency="tendon_joint",
                dtype=wp.int32,
                default=-1,
                namespace="mujoco",
                references="joint",  # Offset by joint count during merge
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tendon_coef",
                frequency="tendon_joint",
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                mjcf_attribute_name="coef",
            )
        )

    def _init_pairs(self, model: Model, spec, shape_mapping: dict[int, str], template_world: int) -> None:
        """
        Initialize MuJoCo contact pairs from custom attributes.

        Only pairs belonging to the template world are added to the MuJoCo spec.
        MuJoCo will replicate these pairs across all worlds automatically.

        Args:
            model: The Newton model.
            spec: The MuJoCo spec to add pairs to.
            shape_mapping: Mapping from Newton shape index to MuJoCo geom name.
            template_world: The world index to use as the template (typically first_group).
        """
        pair_count = model.custom_frequency_counts.get("mujoco:pair", 0)
        if pair_count == 0:
            return

        mujoco_attrs = model.mujoco

        def get_numpy(name):
            attr = getattr(mujoco_attrs, name, None)
            return attr.numpy() if attr is not None else None

        pair_world = get_numpy("pair_world")
        pair_geom1 = get_numpy("pair_geom1")
        pair_geom2 = get_numpy("pair_geom2")
        if pair_world is None or pair_geom1 is None or pair_geom2 is None:
            return

        pair_condim = get_numpy("pair_condim")
        pair_solref = get_numpy("pair_solref")
        pair_solreffriction = get_numpy("pair_solreffriction")
        pair_solimp = get_numpy("pair_solimp")
        pair_margin = get_numpy("pair_margin")
        pair_gap = get_numpy("pair_gap")
        pair_friction = get_numpy("pair_friction")

        for i in range(pair_count):
            # Only include pairs from the template world
            if int(pair_world[i]) != template_world:
                continue

            # Map Newton shape indices to MuJoCo geom names
            newton_shape1 = int(pair_geom1[i])
            newton_shape2 = int(pair_geom2[i])

            # Skip invalid pairs
            if newton_shape1 < 0 or newton_shape2 < 0:
                continue

            geom_name1 = shape_mapping.get(newton_shape1)
            geom_name2 = shape_mapping.get(newton_shape2)

            if geom_name1 is None or geom_name2 is None:
                warnings.warn(
                    f"Skipping pair {i}: Newton shapes ({newton_shape1}, {newton_shape2}) "
                    f"not found in MuJoCo shape mapping.",
                    stacklevel=2,
                )
                continue

            # Build pair kwargs
            pair_kwargs: dict[str, Any] = {
                "geomname1": geom_name1,
                "geomname2": geom_name2,
            }

            if pair_condim is not None:
                pair_kwargs["condim"] = int(pair_condim[i])
            if pair_solref is not None:
                pair_kwargs["solref"] = pair_solref[i].tolist()
            if pair_solreffriction is not None:
                pair_kwargs["solreffriction"] = pair_solreffriction[i].tolist()
            if pair_solimp is not None:
                pair_kwargs["solimp"] = pair_solimp[i].tolist()
            if pair_margin is not None:
                pair_kwargs["margin"] = float(pair_margin[i])
            if pair_gap is not None:
                pair_kwargs["gap"] = float(pair_gap[i])
            if pair_friction is not None:
                pair_kwargs["friction"] = pair_friction[i].tolist()

            spec.add_pair(**pair_kwargs)

    @staticmethod
    def _validate_tendon_attributes(model: Model) -> tuple[int, int]:
        """
        Validate that all tendon attributes have consistent lengths.

        Args:
            model: The Newton model to validate.

        Returns:
            tuple[int, int]: (tendon_count, wrap_count) - number of tendons and total wraps defined.

        Raises:
            ValueError: If tendon attributes have inconsistent lengths.
        """
        mujoco_attrs = getattr(model, "mujoco", None)
        if mujoco_attrs is None:
            return 0, 0

        # Tendon-level attributes
        tendon_attr_names = [
            "tendon_world",
            "tendon_stiffness",
            "tendon_damping",
            "tendon_frictionloss",
            "tendon_limited",
            "tendon_range",
            "tendon_margin",
            "tendon_actuator_force_limited",
            "tendon_actuator_force_range",
            "tendon_solref_limit",
            "tendon_solimp_limit",
            "tendon_solref_friction",
            "tendon_solimp_friction",
            "tendon_springlength",
            "tendon_armature",
            "tendon_joint_adr",
            "tendon_joint_num",
        ]

        # If the list above has N parameters then each tendon should have exactly N parameters.
        # Count the number of parameters that we have for each tendon.
        # Each entry in the array of counts should be N.
        # We can then extract the number of unique entries in our array of counts.
        # The number of unique entries should be 1 because every entry should be N.
        # If the number of unique entries is not 1 then we are missing an attribute on at least one tendon.
        tendon_lengths: dict[str, int] = {}
        for name in tendon_attr_names:
            attr = getattr(mujoco_attrs, name, None)
            if attr is not None:
                tendon_lengths[name] = len(attr)
        if not tendon_lengths:
            return 0, 0
        # Check all tendon-level lengths are the same
        unique_tendon_lengths = set(tendon_lengths.values())
        if len(unique_tendon_lengths) > 1:
            raise ValueError(
                f"MuJoCo tendon attributes have inconsistent lengths: {tendon_lengths}. "
                "All tendon-level attributes must have the same number of elements."
            )

        # Compute the number of tendons.
        tendon_count = next(iter(unique_tendon_lengths))

        # Attributes per joint in the tendon that allow the tendon length to
        # be calculated as a linear sum of coefficient and joint position.
        # For each joint in a tendon (specified by joint index) there must be a corresponding coefficient.
        joint_attr_names = ["tendon_joint", "tendon_coef"]
        joint_lengths: dict[str, int] = {}
        for name in joint_attr_names:
            attr = getattr(mujoco_attrs, name, None)
            if attr is not None:
                joint_lengths[name] = len(attr)
        if not joint_lengths:
            return tendon_count, 0
        # Check all joint-level lengths are the same
        unique_joint_lengths = set(joint_lengths.values())
        if len(unique_joint_lengths) > 1:
            raise ValueError(
                f"MuJoCo tendon joint attributes have inconsistent lengths: {joint_lengths}. "
                "All joint-level attributes must have the same number of elements."
            )

        # Count the length of the array of all joint indices and the
        # length of the array of all coefficients.
        joint_entry_count = next(iter(unique_joint_lengths))

        return tendon_count, joint_entry_count

    def _init_tendons(
        self, model: Model, spec, joint_mapping: dict[int, str], template_world: int
    ) -> tuple[list[int], list[str]]:
        """
        Initialize MuJoCo fixed tendons from custom attributes.

        Only tendons belonging to the template world are added to the MuJoCo spec.
        MuJoCo will replicate these tendons across all worlds automatically.

        Args:
            model: The Newton model.
            spec: The MuJoCo spec to add tendons to.
            joint_mapping: Mapping from Newton joint index to MuJoCo joint name.
            template_world: The world index to use as the template (typically first_group).

        Returns:
            tuple[list[int], list[str]]: Tuple of (Newton tendon indices, MuJoCo tendon names).
        """

        # Count the number of tendons (tendon_count)
        # Count the length of the arrays that contains the joint indices of all tendons (joint_entry_count)
        tendon_count, joint_entry_count = self._validate_tendon_attributes(model)
        if tendon_count == 0:
            return [], []

        mujoco_attrs = model.mujoco

        # Get tendon arrays
        tendon_world = mujoco_attrs.tendon_world.numpy()
        tendon_stiffness = getattr(mujoco_attrs, "tendon_stiffness", None)
        tendon_stiffness_np = tendon_stiffness.numpy() if tendon_stiffness is not None else None
        tendon_damping = getattr(mujoco_attrs, "tendon_damping", None)
        tendon_damping_np = tendon_damping.numpy() if tendon_damping is not None else None
        tendon_frictionloss = getattr(mujoco_attrs, "tendon_frictionloss", None)
        tendon_frictionloss_np = tendon_frictionloss.numpy() if tendon_frictionloss is not None else None
        tendon_limited = getattr(mujoco_attrs, "tendon_limited", None)
        tendon_limited_np = tendon_limited.numpy() if tendon_limited is not None else None
        tendon_range = getattr(mujoco_attrs, "tendon_range", None)
        tendon_range_np = tendon_range.numpy() if tendon_range is not None else None
        tendon_actuator_force_limited = getattr(mujoco_attrs, "tendon_actuator_force_limited", None)
        tendon_actuator_force_limited_np = (
            tendon_actuator_force_limited.numpy() if tendon_actuator_force_limited is not None else None
        )
        tendon_actuator_force_range = getattr(mujoco_attrs, "tendon_actuator_force_range", None)
        tendon_actuator_force_range_np = (
            tendon_actuator_force_range.numpy() if tendon_actuator_force_range is not None else None
        )
        tendon_margin = getattr(mujoco_attrs, "tendon_margin", None)
        tendon_margin_np = tendon_margin.numpy() if tendon_margin is not None else None
        tendon_solref_limit = getattr(mujoco_attrs, "tendon_solref_limit", None)
        tendon_solref_limit_np = tendon_solref_limit.numpy() if tendon_solref_limit is not None else None
        tendon_solimp_limit = getattr(mujoco_attrs, "tendon_solimp_limit", None)
        tendon_solimp_limit_np = tendon_solimp_limit.numpy() if tendon_solimp_limit is not None else None
        tendon_solref_friction = getattr(mujoco_attrs, "tendon_solref_friction", None)
        tendon_solref_friction_np = tendon_solref_friction.numpy() if tendon_solref_friction is not None else None
        tendon_solimp_friction = getattr(mujoco_attrs, "tendon_solimp_friction", None)
        tendon_solimp_friction_np = tendon_solimp_friction.numpy() if tendon_solimp_friction is not None else None
        tendon_armature = getattr(mujoco_attrs, "tendon_armature", None)
        tendon_armature_np = tendon_armature.numpy() if tendon_armature is not None else None
        tendon_springlength = getattr(mujoco_attrs, "tendon_springlength", None)
        tendon_springlength_np = tendon_springlength.numpy() if tendon_springlength is not None else None
        tendon_joint_adr = mujoco_attrs.tendon_joint_adr.numpy()
        tendon_joint_num = mujoco_attrs.tendon_joint_num.numpy()

        # Get joint arrays (for the linear combination)
        tendon_joint = mujoco_attrs.tendon_joint.numpy() if joint_entry_count > 0 else None
        tendon_coef = mujoco_attrs.tendon_coef.numpy() if joint_entry_count > 0 else None

        model_joint_type_np = model.joint_type.numpy()

        # Track which Newton tendon indices are added to MuJoCo and their names
        selected_tendons: list[int] = []
        tendon_names: list[str] = []

        for i in range(tendon_count):
            # Only include tendons from the template world or global tendons (world < 0)
            tw = int(tendon_world[i])
            if tw != template_world and tw >= 0:
                continue

            # Track this tendon
            selected_tendons.append(i)

            # Create tendon with a unique name
            tendon_name = f"tendon_{i}"
            tendon_names.append(tendon_name)
            t = spec.add_tendon()
            t.name = tendon_name

            # Set tendon properties
            if tendon_stiffness_np is not None:
                t.stiffness = float(tendon_stiffness_np[i])
            if tendon_damping_np is not None:
                t.damping = float(tendon_damping_np[i])
            if tendon_frictionloss_np is not None:
                t.frictionloss = float(tendon_frictionloss_np[i])
            if tendon_limited_np is not None:
                t.limited = int(tendon_limited_np[i])
            if tendon_range_np is not None:
                t.range = tendon_range_np[i].tolist()
            if tendon_actuator_force_limited_np is not None:
                t.actfrclimited = int(tendon_actuator_force_limited_np[i])
            if tendon_actuator_force_range_np is not None:
                t.actfrcrange = tendon_actuator_force_range_np[i].tolist()
            if tendon_margin_np is not None:
                t.margin = float(tendon_margin_np[i])
            if tendon_armature_np is not None:
                t.armature = float(tendon_armature_np[i])
            if tendon_solref_limit_np is not None:
                t.solref_limit = tendon_solref_limit_np[i].tolist()
            if tendon_solimp_limit_np is not None:
                t.solimp_limit = tendon_solimp_limit_np[i].tolist()
            if tendon_solref_friction_np is not None:
                t.solref_friction = tendon_solref_friction_np[i].tolist()
            if tendon_solimp_friction_np is not None:
                t.solimp_friction = tendon_solimp_friction_np[i].tolist()
            if tendon_springlength_np is not None:
                val = tendon_springlength_np[i]
                has_automatic_length_computation = val[0] == -1.0
                has_dead_zone = val[1] >= val[0]
                if has_automatic_length_computation:
                    # The spring length is automatically computed from
                    # start state. In this mode it is not possible to
                    # author a dead zone.
                    t.springlength[0] = -1.0
                    t.springlength[1] = -1.0
                elif has_dead_zone:
                    # Set a finite dead zone.
                    t.springlength[0] = val[0]
                    t.springlength[1] = val[1]
                else:
                    # Set a dead zone of zero width
                    t.springlength[0] = val[0]
                    t.springlength[1] = val[0]

            # Add joints for this fixed tendon's linear combination
            joint_start = int(tendon_joint_adr[i])
            joint_num = int(tendon_joint_num[i])

            for j in range(joint_start, joint_start + joint_num):
                if tendon_joint is None or tendon_coef is None:
                    break

                newton_joint = int(tendon_joint[j])
                coef = float(tendon_coef[j])

                if newton_joint < 0:
                    warnings.warn(
                        f"Skipping joint entry {j} for tendon {i}: invalid joint index {newton_joint}.",
                        stacklevel=2,
                    )
                    continue

                if model_joint_type_np[newton_joint] == JointType.D6:
                    warnings.warn(
                        f"Skipping joint entry {j} for tendon {i}: invalid D6 joint type {newton_joint}.",
                        stacklevel=2,
                    )
                    continue

                joint_name = joint_mapping.get(newton_joint)
                if joint_name is None:
                    warnings.warn(
                        f"Skipping joint entry {j} for tendon {i}: Newton joint {newton_joint} "
                        f"not found in MuJoCo joint mapping.",
                        stacklevel=2,
                    )
                    continue

                t.wrap_joint(joint_name, coef)

        return selected_tendons, tendon_names

    def _init_actuators(
        self,
        model: Model,
        spec,
        template_world: int,
        actuator_args: dict,
        mjc_actuator_ctrl_source_list: list[int],
        mjc_actuator_to_newton_idx_list: list[int],
        dof_to_mjc_joint: np.ndarray,
        mjc_joint_names: list[str],
        selected_tendons: list[int],
        mjc_tendon_names: list[str],
    ) -> int:
        """Initialize MuJoCo general actuators from custom attributes.

        Only processes CTRL_DIRECT actuators (motor, general, etc.) from the
        mujoco:actuator custom attributes. JOINT_TARGET actuators (position/velocity)
        are handled separately in the joint iteration loop.

        For CTRL_DIRECT actuators targeting joints, this method uses the DOF index
        stored in actuator_trnid (see import_mjcf.py) to look up the correct MuJoCo
        joint name. This is necessary because Newton may combine multiple MJCF joints
        into one, but MuJoCo needs the specific joint name (e.g., "joint_ang1" not "joint").

        Args:
            model: The Newton model.
            spec: The MuJoCo spec to add actuators to.
            template_world: The world index to use as the template.
            actuator_args: Default actuator arguments.
            mjc_actuator_ctrl_source_list: List to append control sources to.
            mjc_actuator_to_newton_idx_list: List to append Newton indices to.
            dof_to_mjc_joint: Mapping from Newton DOF index to MuJoCo joint index.
                Used to resolve CTRL_DIRECT joint actuators to their MuJoCo targets.
            mjc_joint_names: List of MuJoCo joint names indexed by MuJoCo joint index.
                Used together with dof_to_mjc_joint to get the correct joint name.

        Returns:
            int: Number of actuators added.
        """
        import mujoco

        mujoco_attrs = getattr(model, "mujoco", None)
        mujoco_actuator_count = model.custom_frequency_counts.get("mujoco:actuator", 0)

        if mujoco_actuator_count == 0 or mujoco_attrs is None or not hasattr(mujoco_attrs, "actuator_trnid"):
            return 0

        actuator_count = 0

        # actuator_trnid[:,0] is the target index, actuator_trntype determines its meaning
        actuator_trnid = mujoco_attrs.actuator_trnid.numpy()
        trntype_arr = mujoco_attrs.actuator_trntype.numpy() if hasattr(mujoco_attrs, "actuator_trntype") else None
        ctrl_source_arr = mujoco_attrs.ctrl_source.numpy() if hasattr(mujoco_attrs, "ctrl_source") else None
        actuator_world_arr = mujoco_attrs.actuator_world.numpy() if hasattr(mujoco_attrs, "actuator_world") else None
        # Pre-fetch range/limited arrays to avoid per-element .numpy() calls
        has_ctrlrange_arr = (
            mujoco_attrs.actuator_has_ctrlrange.numpy() if hasattr(mujoco_attrs, "actuator_has_ctrlrange") else None
        )
        ctrlrange_arr = mujoco_attrs.actuator_ctrlrange.numpy() if hasattr(mujoco_attrs, "actuator_ctrlrange") else None
        ctrllimited_arr = (
            mujoco_attrs.actuator_ctrllimited.numpy() if hasattr(mujoco_attrs, "actuator_ctrllimited") else None
        )
        has_forcerange_arr = (
            mujoco_attrs.actuator_has_forcerange.numpy() if hasattr(mujoco_attrs, "actuator_has_forcerange") else None
        )
        forcerange_arr = (
            mujoco_attrs.actuator_forcerange.numpy() if hasattr(mujoco_attrs, "actuator_forcerange") else None
        )
        forcelimited_arr = (
            mujoco_attrs.actuator_forcelimited.numpy() if hasattr(mujoco_attrs, "actuator_forcelimited") else None
        )
        has_actrange_arr = (
            mujoco_attrs.actuator_has_actrange.numpy() if hasattr(mujoco_attrs, "actuator_has_actrange") else None
        )
        actrange_arr = mujoco_attrs.actuator_actrange.numpy() if hasattr(mujoco_attrs, "actuator_actrange") else None
        actlimited_arr = (
            mujoco_attrs.actuator_actlimited.numpy() if hasattr(mujoco_attrs, "actuator_actlimited") else None
        )

        for mujoco_act_idx in range(mujoco_actuator_count):
            # Skip JOINT_TARGET actuators - they're already added via joint_act_mode path
            if ctrl_source_arr is not None:
                ctrl_source = int(ctrl_source_arr[mujoco_act_idx])
                if ctrl_source == SolverMuJoCo.CtrlSource.JOINT_TARGET:
                    continue  # Already handled in joint iteration

            # Only include actuators from the first world (template) or global actuators
            if actuator_world_arr is not None:
                actuator_world = int(actuator_world_arr[mujoco_act_idx])
                if actuator_world != template_world and actuator_world != -1:
                    continue  # Skip actuators from other worlds

            target_idx = int(actuator_trnid[mujoco_act_idx, 0])

            # Determine target type from trntype (0=JOINT, 1=TENDON, 2=SITE, etc.)
            trntype = int(trntype_arr[mujoco_act_idx]) if trntype_arr is not None else 0

            if trntype == 0:  # TrnType.JOINT
                # For CTRL_DIRECT joint actuators, actuator_trnid stores a DOF index
                # (not a Newton joint index). This allows us to find the specific MuJoCo
                # joint when Newton has combined multiple MJCF joints into one.
                dof_idx = target_idx
                dofs_per_world = len(dof_to_mjc_joint)
                if dof_idx < 0 or dof_idx >= dofs_per_world:
                    if wp.config.verbose:
                        print(f"Warning: MuJoCo actuator {mujoco_act_idx} has invalid DOF target {dof_idx}")
                    continue
                mjc_joint_idx = dof_to_mjc_joint[dof_idx]
                if mjc_joint_idx < 0 or mjc_joint_idx >= len(mjc_joint_names):
                    if wp.config.verbose:
                        print(f"Warning: MuJoCo actuator {mujoco_act_idx} DOF {dof_idx} not mapped to MuJoCo joint")
                    continue
                target_name = mjc_joint_names[mjc_joint_idx]
            elif trntype == 2:  # TrnType.TENDON
                try:
                    mjc_tendon_idx = selected_tendons.index(target_idx)
                    target_name = mjc_tendon_names[mjc_tendon_idx]
                except (ValueError, IndexError):
                    if wp.config.verbose:
                        print(f"Warning: MuJoCo actuator {mujoco_act_idx} references tendon {target_idx} not in MuJoCo")
                    continue
            elif trntype == 4:  # TrnType.BODY
                if target_idx < 0 or target_idx >= len(model.body_key):
                    if wp.config.verbose:
                        print(f"Warning: MuJoCo actuator {mujoco_act_idx} has invalid body target {target_idx}")
                    continue
                target_name = model.body_key[target_idx]
            else:
                # TODO: Support site, slidercrank, and jointinparent transmission types
                if wp.config.verbose:
                    print(f"Warning: MuJoCo actuator {mujoco_act_idx} has unsupported trntype {trntype}")
                continue

            general_args = dict(actuator_args)

            # Get custom attributes for this MuJoCo actuator
            if hasattr(mujoco_attrs, "actuator_gainprm"):
                gainprm = mujoco_attrs.actuator_gainprm.numpy()[mujoco_act_idx]
                general_args["gainprm"] = list(gainprm)  # All 10 elements
            if hasattr(mujoco_attrs, "actuator_biasprm"):
                biasprm = mujoco_attrs.actuator_biasprm.numpy()[mujoco_act_idx]
                general_args["biasprm"] = list(biasprm)  # All 10 elements
            if hasattr(mujoco_attrs, "actuator_dynprm"):
                dynprm = mujoco_attrs.actuator_dynprm.numpy()[mujoco_act_idx]
                general_args["dynprm"] = list(dynprm)  # All 10 elements
            if hasattr(mujoco_attrs, "actuator_gear"):
                gear_arr = mujoco_attrs.actuator_gear.numpy()[mujoco_act_idx]
                general_args["gear"] = list(gear_arr)
            # Only pass range to MuJoCo when explicitly set in MJCF (has_*range flags),
            # so MuJoCo can correctly resolve auto-limited flags via spec.compiler.autolimits.
            if has_ctrlrange_arr is not None and has_ctrlrange_arr[mujoco_act_idx]:
                general_args["ctrlrange"] = tuple(ctrlrange_arr[mujoco_act_idx])
            if ctrllimited_arr is not None:
                general_args["ctrllimited"] = int(ctrllimited_arr[mujoco_act_idx])
            if has_forcerange_arr is not None and has_forcerange_arr[mujoco_act_idx]:
                general_args["forcerange"] = tuple(forcerange_arr[mujoco_act_idx])
            if forcelimited_arr is not None:
                general_args["forcelimited"] = int(forcelimited_arr[mujoco_act_idx])
            if has_actrange_arr is not None and has_actrange_arr[mujoco_act_idx]:
                general_args["actrange"] = tuple(actrange_arr[mujoco_act_idx])
            if actlimited_arr is not None:
                general_args["actlimited"] = int(actlimited_arr[mujoco_act_idx])
            if hasattr(mujoco_attrs, "actuator_actearly"):
                actearly = mujoco_attrs.actuator_actearly.numpy()[mujoco_act_idx]
                general_args["actearly"] = bool(actearly)
            if hasattr(mujoco_attrs, "actuator_actdim"):
                actdim = mujoco_attrs.actuator_actdim.numpy()[mujoco_act_idx]
                if actdim >= 0:  # -1 means auto
                    general_args["actdim"] = int(actdim)
            if hasattr(mujoco_attrs, "actuator_dyntype"):
                dyntype = int(mujoco_attrs.actuator_dyntype.numpy()[mujoco_act_idx])
                general_args["dyntype"] = dyntype
            if hasattr(mujoco_attrs, "actuator_gaintype"):
                gaintype = int(mujoco_attrs.actuator_gaintype.numpy()[mujoco_act_idx])
                general_args["gaintype"] = gaintype
            if hasattr(mujoco_attrs, "actuator_biastype"):
                biastype = int(mujoco_attrs.actuator_biastype.numpy()[mujoco_act_idx])
                general_args["biastype"] = biastype

            # Map trntype integer to MuJoCo enum and override default in general_args
            trntype_enum = {
                0: mujoco.mjtTrn.mjTRN_JOINT,
                1: mujoco.mjtTrn.mjTRN_JOINTINPARENT,
                2: mujoco.mjtTrn.mjTRN_TENDON,
                3: mujoco.mjtTrn.mjTRN_SITE,
                4: mujoco.mjtTrn.mjTRN_BODY,
                5: mujoco.mjtTrn.mjTRN_SLIDERCRANK,
            }.get(trntype, mujoco.mjtTrn.mjTRN_JOINT)
            general_args["trntype"] = trntype_enum
            spec.add_actuator(target=target_name, **general_args)
            # CTRL_DIRECT actuators - store MJCF-order index into control.mujoco.ctrl
            # mujoco_act_idx is the index in Newton's mujoco:actuator frequency (MJCF order)
            mjc_actuator_ctrl_source_list.append(1)  # CTRL_DIRECT
            mjc_actuator_to_newton_idx_list.append(mujoco_act_idx)
            actuator_count += 1

        return actuator_count

    def __init__(
        self,
        model: Model,
        *,
        mjw_model: MjWarpModel | None = None,
        mjw_data: MjWarpData | None = None,
        separate_worlds: bool | None = None,
        njmax: int | None = None,
        nconmax: int | None = None,
        iterations: int | None = None,
        ls_iterations: int | None = None,
        ccd_iterations: int | None = None,
        sdf_iterations: int | None = None,
        sdf_initpoints: int | None = None,
        solver: int | str | None = None,
        integrator: int | str | None = None,
        cone: int | str | None = None,
        jacobian: int | str | None = None,
        impratio: float | None = None,
        tolerance: float | None = None,
        ls_tolerance: float | None = None,
        ccd_tolerance: float | None = None,
        density: float | None = None,
        viscosity: float | None = None,
        wind: tuple | None = None,
        magnetic: tuple | None = None,
        use_mujoco_cpu: bool = False,
        disable_contacts: bool = False,
        default_actuator_gear: float | None = None,
        actuator_gears: dict[str, float] | None = None,
        update_data_interval: int = 1,
        save_to_mjcf: str | None = None,
        ls_parallel: bool = False,
        use_mujoco_contacts: bool = True,
        include_sites: bool = True,
    ):
        """
        Solver options (e.g., ``impratio``) follow this resolution priority:

        1. **Constructor argument** - If provided, same value is used for all worlds.
        2. **Newton model custom attribute** (``model.mujoco.<option>``) - Supports per-world values.
        3. **MuJoCo default** - Used if neither of the above is set.

        Args:
            model (Model): the model to be simulated.
            mjw_model (MjWarpModel | None): Optional pre-existing MuJoCo Warp model. If provided with `mjw_data`, conversion from Newton model is skipped.
            mjw_data (MjWarpData | None): Optional pre-existing MuJoCo Warp data. If provided with `mjw_model`, conversion from Newton model is skipped.
            separate_worlds (bool | None): If True, each Newton world is mapped to a separate MuJoCo world. Defaults to `not use_mujoco_cpu`.
            njmax (int | None): Maximum number of constraints per world. If None, a default value is estimated from the initial state. Note that the larger of the user-provided value or the default value is used.
            nconmax (int | None): Number of contact points per world. If None, a default value is estimated from the initial state. Note that the larger of the user-provided value or the default value is used.
            iterations (int | None): Number of solver iterations. If None, uses model custom attribute or MuJoCo's default (100).
            ls_iterations (int | None): Number of line search iterations for the solver. If None, uses model custom attribute or MuJoCo's default (50).
            ccd_iterations (int | None): Maximum CCD iterations. If None, uses model custom attribute or MuJoCo's default (35).
            sdf_iterations (int | None): Maximum SDF iterations. If None, uses model custom attribute or MuJoCo's default (10).
            sdf_initpoints (int | None): Number of SDF initialization points. If None, uses model custom attribute or MuJoCo's default (40).
            solver (int | str): Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants. If None, uses model custom attribute or Newton's default ("newton").
            integrator (int | str): Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants. If None, uses model custom attribute or Newton's default ("implicitfast").
            cone (int | str): The type of contact friction cone. Can be "pyramidal", "elliptic", or their corresponding MuJoCo integer constants. If None, uses model custom attribute or Newton's default ("pyramidal").
            jacobian (int | str | None): Jacobian computation method. Can be "dense", "sparse", or "auto", or their corresponding MuJoCo integer constants. If None, uses model custom attribute or MuJoCo's default ("auto").
            impratio (float | None): Frictional-to-normal constraint impedance ratio. If None, uses model custom attribute or MuJoCo's default (1.0).
            tolerance (float | None): Solver tolerance for early termination. If None, uses model custom attribute or MuJoCo's default (1e-8).
            ls_tolerance (float | None): Line search tolerance for early termination. If None, uses model custom attribute or MuJoCo's default (0.01).
            ccd_tolerance (float | None): Continuous collision detection tolerance. If None, uses model custom attribute or MuJoCo's default (1e-6).
            density (float | None): Medium density for lift and drag forces. If None, uses model custom attribute or MuJoCo's default (0.0).
            viscosity (float | None): Medium viscosity for lift and drag forces. If None, uses model custom attribute or MuJoCo's default (0.0).
            wind (tuple | None): Wind velocity vector (x, y, z) for lift and drag forces. If None, uses model custom attribute or MuJoCo's default (0, 0, 0).
            magnetic (tuple | None): Global magnetic flux vector (x, y, z). If None, uses model custom attribute or MuJoCo's default (0, -0.5, 0).
            use_mujoco_cpu (bool): If True, use the MuJoCo-C CPU backend instead of `mujoco_warp`.
            disable_contacts (bool): If True, disable contact computation in MuJoCo.
            register_collision_groups (bool): If True, register collision groups from the Newton model in MuJoCo.
            default_actuator_gear (float | None): Default gear ratio for all actuators. Can be overridden by `actuator_gears`.
            actuator_gears (dict[str, float] | None): Dictionary mapping joint names to specific gear ratios, overriding the `default_actuator_gear`.
            update_data_interval (int): Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state. If 0, Data is never updated after initialization.
            save_to_mjcf (str | None): Optional path to save the generated MJCF model file.
            ls_parallel (bool): If True, enable parallel line search in MuJoCo. Defaults to False.
            use_mujoco_contacts (bool): If True, use the MuJoCo contact solver. If False, use the Newton contact solver (newton contacts must be passed in through the step function in that case).
            include_sites (bool): If ``True`` (default), Newton shapes marked with ``ShapeFlags.SITE`` are exported as MuJoCo sites. Sites are non-colliding reference points used for sensor attachment, debugging, or as frames of reference. If ``False``, sites are skipped during export. Defaults to ``True``.
        """
        super().__init__(model)

        # Import and cache MuJoCo modules (only happens once per class)
        mujoco, _ = self.import_mujoco()

        # --- New unified mappings: MuJoCo[world, entity] -> Newton[entity] ---
        self.mjc_body_to_newton: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, body] to Newton body index. Shape [nworld, nbody], dtype int32."""
        self.mjc_geom_to_newton_shape: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, geom] to Newton shape index. Shape [nworld, ngeom], dtype int32."""
        self.mjc_jnt_to_newton_jnt: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, joint] to Newton joint index. Shape [nworld, njnt], dtype int32."""
        self.mjc_jnt_to_newton_dof: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, joint] to Newton DOF index. Shape [nworld, njnt], dtype int32."""
        self.mjc_dof_to_newton_dof: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, dof] to Newton DOF index. Shape [nworld, nv], dtype int32."""
        self.mjc_actuator_ctrl_source: wp.array(dtype=wp.int32) | None = None
        """Control source for each MuJoCo actuator.

        Values: 0=JOINT_TARGET (uses joint_target_pos/vel), 1=CTRL_DIRECT (uses mujoco.ctrl)
        Shape [nu], dtype int32."""
        self.mjc_actuator_to_newton_idx: wp.array(dtype=wp.int32) | None = None
        """Mapping from MuJoCo actuator to Newton index.

        For JOINT_TARGET: sign-encoded DOF index (>=0: position, -1: unmapped, <=-2: velocity with -(idx+2))
        For CTRL_DIRECT: MJCF-order index into control.mujoco.ctrl array

        Shape [nu], dtype int32."""
        self.mjc_mocap_to_newton_jnt: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, mocap] to Newton joint index. Shape [nworld, nmocap], dtype int32."""
        self.mjc_eq_to_newton_eq: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, eq] to Newton equality constraint index.

        Corresponds to the equality constraints that are created in MuJoCo from Newton's equality constraints.
        A value of -1 indicates that the MuJoCo equality constraint has been created from a Newton joint, see :attr:`mjc_eq_to_newton_jnt`
        for the corresponding joint index.

        Shape [nworld, neq], dtype int32."""
        self.mjc_eq_to_newton_jnt: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, eq] to Newton joint index.

        Corresponds to the equality constraints that are created in MuJoCo from Newton joints that have no associated articulation,
        i.e. where :attr:`newton.Model.joint_articulation` is -1 for the joint which results in 2 equality constraints being created in MuJoCo.
        A value of -1 indicates that the MuJoCo equality constraint is not associated with a Newton joint but an explicitly created Newton equality constraint,
        see :attr:`mjc_eq_to_newton_eq` for the corresponding equality constraint index.

        Shape [nworld, neq], dtype int32."""
        self.mjc_eq_to_newton_mimic: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, eq] to Newton mimic constraint index.

        Corresponds to the equality constraints that are created in MuJoCo from Newton's mimic constraints.
        A value of -1 indicates that the MuJoCo equality constraint is not associated with a Newton mimic constraint.

        Shape [nworld, neq], dtype int32."""
        self.mjc_tendon_to_newton_tendon: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, tendon] to Newton tendon index.

        Shape [nworld, ntendon], dtype int32."""
        self.body_free_qd_start: wp.array(dtype=wp.int32) | None = None
        """Per-body mapping to the free-joint qd_start index (or -1 if not free)."""

        # --- Conditional/lazy mappings ---
        self.newton_shape_to_mjc_geom: wp.array(dtype=wp.int32) | None = None
        """Inverse mapping from Newton shape index to MuJoCo geom index. Only created when use_mujoco_contacts=False. Shape [nshape], dtype int32."""

        # --- Helper arrays for actuator types ---

        # --- Internal state for mapping creation ---
        self._shapes_per_world: int = 0
        """Number of shapes per world (for computing Newton shape indices from template)."""
        self._first_env_shape_base: int = 0
        """Base shape index for the first environment."""

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
                separate_worlds = not use_mujoco_cpu and model.world_count > 1
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
                    ccd_iterations=ccd_iterations,
                    sdf_iterations=sdf_iterations,
                    sdf_initpoints=sdf_initpoints,
                    cone=cone,
                    jacobian=jacobian,
                    impratio=impratio,
                    tolerance=tolerance,
                    ls_tolerance=ls_tolerance,
                    ccd_tolerance=ccd_tolerance,
                    density=density,
                    viscosity=viscosity,
                    wind=wind,
                    magnetic=magnetic,
                    solver=solver,
                    integrator=integrator,
                    default_actuator_gear=default_actuator_gear,
                    actuator_gears=actuator_gears,
                    target_filename=save_to_mjcf,
                    ls_parallel=ls_parallel,
                    include_sites=include_sites,
                )
        self.update_data_interval = update_data_interval
        self._step = 0

        # Check if dof_ref is used - if so, we use MuJoCo's FK (eval_fk=False)
        # because ref is only handled by MuJoCo via qpos0
        mujoco_attrs = getattr(model, "mujoco", None)
        dof_ref = getattr(mujoco_attrs, "dof_ref", None) if mujoco_attrs is not None else None
        self._has_ref = dof_ref is not None and np.any(dof_ref.numpy() != 0.0)

        if self.mjw_model is not None:
            self.mjw_model.opt.run_collision_detection = use_mujoco_contacts

    @event_scope
    def mujoco_warp_step(self):
        self._mujoco_warp.step(self.mjw_model, self.mjw_data)

    @event_scope
    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        # When ref is used, we rely on MuJoCo's FK (eval_fk=False) because ref is handled by MuJoCo via qpos0
        eval_fk = not self._has_ref

        if self.use_mujoco_cpu:
            self.apply_mjc_control(self.model, state_in, control, self.mj_data)
            if self.update_data_interval > 0 and self._step % self.update_data_interval == 0:
                # XXX updating the mujoco state at every step may introduce numerical instability
                self.update_mjc_data(self.mj_data, self.model, state_in)
            self.mj_model.opt.timestep = dt
            self._mujoco.mj_step(self.mj_model, self.mj_data)
            self.update_newton_state(self.model, state_out, self.mj_data, eval_fk=eval_fk)
        else:
            self.enable_rne_postconstraint(state_out)
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

            self.update_newton_state(self.model, state_out, self.mjw_data, eval_fk=eval_fk)
        self._step += 1
        return state_out

    def enable_rne_postconstraint(self, state_out: State):
        """Request computation of RNE forces if required for state fields."""
        rne_postconstraint_fields = {"body_qdd", "body_parent_f"}
        # TODO: handle use_mujoco_cpu
        m = self.mjw_model
        if m.sensor_rne_postconstraint:
            return
        if any(getattr(state_out, field) is not None for field in rne_postconstraint_fields):
            # required for cfrc_ext, cfrc_int, cacc
            if wp.config.verbose:
                print("Setting model.sensor_rne_postconstraint True")
            m.sensor_rne_postconstraint = True

    def convert_contacts_to_mjwarp(self, model: Model, state_in: State, contacts: Contacts):
        # Ensure the inverse shape mapping exists (lazy creation)
        if self.newton_shape_to_mjc_geom is None:
            self._create_inverse_shape_mapping()

        bodies_per_world = self.model.body_count // self.model.world_count
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
                contacts.rigid_contact_stiffness,
                contacts.rigid_contact_damping,
                contacts.rigid_contact_friction,
                model.shape_thickness,
                bodies_per_world,
                self.newton_shape_to_mjc_geom,
                # Mujoco warp contacts
                self.mjw_data.naconmax,
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
            device=model.device,
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
            self.update_pair_properties()
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self.update_model_properties()
        if flags & SolverNotifyFlags.CONSTRAINT_PROPERTIES:
            self.update_eq_properties()
            self.update_mimic_eq_properties()
        if flags & SolverNotifyFlags.TENDON_PROPERTIES:
            self.update_tendon_properties()
        if flags & SolverNotifyFlags.ACTUATOR_PROPERTIES:
            self.update_actuator_properties()

    def _create_inverse_shape_mapping(self):
        """
        Create the inverse shape mapping (Newton shape -> MuJoCo [world, geom]).
        This is lazily created only when use_mujoco_contacts=False.
        """
        nworld = self.mjc_geom_to_newton_shape.shape[0]
        ngeom = self.mjc_geom_to_newton_shape.shape[1]

        # Create the inverse mapping array
        self.newton_shape_to_mjc_geom = wp.full(self.model.shape_count, -1, dtype=wp.int32, device=self.model.device)

        # Launch kernel to populate the inverse mapping
        wp.launch(
            create_inverse_shape_mapping_kernel,
            dim=(nworld, ngeom),
            inputs=[
                self.mjc_geom_to_newton_shape,
            ],
            outputs=[
                self.newton_shape_to_mjc_geom,
            ],
            device=self.model.device,
        )

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
        joints_per_world = model.joint_count // nworld
        if control is not None:
            # Use instance arrays (built during MuJoCo model construction)
            if self.mjc_actuator_ctrl_source is not None and self.mjc_actuator_to_newton_idx is not None:
                nu = self.mjc_actuator_ctrl_source.shape[0]
                dofs_per_world = model.joint_dof_count // nworld if nworld > 0 else model.joint_dof_count

                # Get mujoco.ctrl (None if not available - won't be accessed if no CTRL_DIRECT actuators)
                mujoco_ctrl_ns = getattr(control, "mujoco", None)
                mujoco_ctrl = getattr(mujoco_ctrl_ns, "ctrl", None) if mujoco_ctrl_ns is not None else None
                ctrls_per_world = mujoco_ctrl.shape[0] // nworld if mujoco_ctrl is not None and nworld > 0 else 0

                wp.launch(
                    apply_mjc_control_kernel,
                    dim=(nworld, nu),
                    inputs=[
                        self.mjc_actuator_ctrl_source,
                        self.mjc_actuator_to_newton_idx,
                        control.joint_target_pos,
                        control.joint_target_vel,
                        mujoco_ctrl,
                        dofs_per_world,
                        ctrls_per_world,
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
                    control.joint_f,
                    model.joint_type,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    joints_per_world,
                ],
                outputs=[
                    qfrc,
                ],
                device=model.device,
            )

        if state.body_f is not None:
            # Launch over MuJoCo bodies
            nbody = self.mjc_body_to_newton.shape[1]
            wp.launch(
                apply_mjc_body_f_kernel,
                dim=(nworld, nbody),
                inputs=[
                    self.mjc_body_to_newton,
                    state.body_f,
                ],
                outputs=[
                    xfrc,
                ],
                device=model.device,
            )
        if control is not None and control.joint_f is not None:
            # Free/DISTANCE joint forces are applied via xfrc_applied to preserve COM-wrench semantics.
            nbody = self.mjc_body_to_newton.shape[1]
            wp.launch(
                apply_mjc_free_joint_f_to_body_f_kernel,
                dim=(nworld, nbody),
                inputs=[
                    self.mjc_body_to_newton,
                    self.body_free_qd_start,
                    control.joint_f,
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
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_child,
                model.body_com,
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
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_child,
                model.body_com,
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
                    model.joint_articulation,
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
            # Launch over MuJoCo bodies
            nbody = self.mjc_body_to_newton.shape[1]
            wp.launch(
                convert_body_xforms_to_warp_kernel,
                dim=(nworld, nbody),
                inputs=[
                    self.mjc_body_to_newton,
                    xpos,
                    xquat,
                ],
                outputs=[state.body_q],
                device=model.device,
            )

        # Update rigid force fields on state.
        if state.body_qdd is not None or state.body_parent_f is not None:
            # Launch over MuJoCo bodies
            nbody = self.mjc_body_to_newton.shape[1]
            wp.launch(
                convert_rigid_forces_from_mj_kernel,
                (nworld, nbody),
                inputs=[
                    self.mjc_body_to_newton,
                    self.mjw_model.body_rootid,
                    self.mjw_model.opt.gravity,
                    self.mjw_data.xipos,
                    self.mjw_data.subtree_com,
                    self.mjw_data.cacc,
                    self.mjw_data.cvel,
                    self.mjw_data.cfrc_int,
                ],
                outputs=[state.body_qdd, state.body_parent_f],
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
                (min(selected_shapes[i], selected_shapes[j]), max(selected_shapes[i], selected_shapes[j]))
                not in model.shape_collision_filter_pairs
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

    def get_max_contact_count(self) -> int:
        """Return the maximum number of rigid contacts that can be generated by MuJoCo."""
        if self.use_mujoco_cpu:
            raise NotImplementedError()
        return self.mjw_data.naconmax

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Update `contacts` from MuJoCo contacts when running with ``use_mujoco_contacts``."""
        if self.use_mujoco_cpu:
            raise NotImplementedError()

        # TODO: ensure that class invariants are preserved
        # TODO: fill actual contact arrays instead of creating new ones
        mj_data = self.mjw_data
        mj_contact = mj_data.contact

        if mj_data.naconmax > contacts.rigid_contact_max:
            raise ValueError(
                f"MuJoCo naconmax ({mj_data.naconmax}) exceeds contacts.rigid_contact_max "
                f"({contacts.rigid_contact_max}). Create Contacts with at least "
                f"rigid_contact_max={mj_data.naconmax}."
            )

        wp.launch(
            convert_mjw_contacts_to_newton_kernel,
            dim=mj_data.naconmax,
            inputs=[
                self.mjc_geom_to_newton_shape,
                self.mjc_body_to_newton,
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
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
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
        separate_worlds: bool | None = None,
        iterations: int | None = None,
        ls_iterations: int | None = None,
        ccd_iterations: int | None = None,
        sdf_iterations: int | None = None,
        sdf_initpoints: int | None = None,
        njmax: int | None = None,  # number of constraints per world
        nconmax: int | None = None,
        solver: int | str | None = None,
        integrator: int | str | None = None,
        disableflags: int = 0,
        disable_contacts: bool = False,
        impratio: float | None = None,
        tolerance: float | None = None,
        ls_tolerance: float | None = None,
        ccd_tolerance: float | None = None,
        density: float | None = None,
        viscosity: float | None = None,
        wind: tuple | None = None,
        magnetic: tuple | None = None,
        cone: int | str | None = None,
        jacobian: int | str | None = None,
        target_filename: str | None = None,
        default_actuator_args: dict | None = None,
        default_actuator_gear: float | None = None,
        actuator_gears: dict[str, float] | None = None,
        actuated_axes: list[int] | None = None,
        skip_visual_only_geoms: bool = True,
        include_sites: bool = True,
        mesh_maxhullvert: int | None = None,
        ls_parallel: bool = False,
    ) -> tuple[MjWarpModel, MjWarpData, MjModel, MjData]:
        """
        Convert a Newton model and state to MuJoCo (Warp) model and data.

        Solver options (e.g., ``impratio``) follow this resolution priority:

        1. **Constructor argument** - If provided, same value is used for all worlds.
        2. **Newton model custom attribute** (``model.mujoco.<option>``) - Supports per-world values.
        3. **MuJoCo default** - Used if neither of the above is set.

        Args:
            model: The Newton model to convert.
            state: The Newton state to convert (optional).
            separate_worlds: If True, each world is a separate MuJoCo simulation. If None, defaults to True for GPU mode (not use_mujoco_cpu).
            iterations: Maximum solver iterations. If None, uses model custom attribute or MuJoCo's default (100).
            ls_iterations: Maximum line search iterations. If None, uses model custom attribute or MuJoCo's default (50).
            njmax: Maximum number of constraints per world.
            nconmax: Maximum number of contacts.
            solver: Constraint solver type ("cg" or "newton"). If None, uses model custom attribute or Newton's default ("newton").
            integrator: Integration method ("euler", "rk4", "implicit", "implicitfast"). If None, uses model custom attribute or Newton's default ("implicitfast").
            disableflags: MuJoCo disable flags bitmask.
            disable_contacts: If True, disable contact computation.
            impratio: Impedance ratio for contacts. If None, uses model custom attribute or MuJoCo default (1.0).
            tolerance: Solver tolerance. If None, uses model custom attribute or MuJoCo default (1e-8).
            ls_tolerance: Line search tolerance. If None, uses model custom attribute or MuJoCo default (0.01).
            ccd_tolerance: CCD tolerance. If None, uses model custom attribute or MuJoCo default (1e-6).
            density: Medium density. If None, uses model custom attribute or MuJoCo default (0.0).
            viscosity: Medium viscosity. If None, uses model custom attribute or MuJoCo default (0.0).
            wind: Wind velocity vector (x, y, z). If None, uses model custom attribute or MuJoCo default (0, 0, 0).
            magnetic: Magnetic flux vector (x, y, z). If None, uses model custom attribute or MuJoCo default (0, -0.5, 0).
            cone: Friction cone type ("pyramidal" or "elliptic"). If None, uses model custom attribute or Newton's default ("pyramidal").
            jacobian: Jacobian computation method ("dense", "sparse", or "auto"). If None, uses model custom attribute or MuJoCo default ("auto").
            target_filename: Optional path to save generated MJCF file.
            default_actuator_args: Default actuator parameters.
            default_actuator_gear: Default actuator gear ratio.
            actuator_gears: Per-actuator gear ratios by name.
            actuated_axes: List of DOF indices to actuate.
            skip_visual_only_geoms: If True, skip geoms that are visual-only.
            include_sites: If True, include sites in the model.
            mesh_maxhullvert: Maximum vertices for convex hull meshes.
            ls_parallel: If True, enable parallel line search.

        Returns:
            tuple[MjWarpModel, MjWarpData, MjModel, MjData]: Model and data objects for
                ``mujoco_warp`` and MuJoCo.
        """
        if mesh_maxhullvert is None:
            mesh_maxhullvert = Mesh.MAX_HULL_VERTICES

        if not model.joint_count:
            raise ValueError("The model must have at least one joint to be able to convert it to MuJoCo.")

        # Set default for separate_worlds if None
        if separate_worlds is None:
            separate_worlds = True

        # Validate that separate_worlds=False is only used with single world
        if not separate_worlds and model.world_count > 1:
            raise ValueError(
                f"separate_worlds=False is only supported for single-world models. "
                f"Got world_count={model.world_count}. Use separate_worlds=True for multi-world models."
            )

        # Validate model compatibility with separate_worlds mode
        if separate_worlds:
            self._validate_model_for_separate_worlds(model)

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

        # Convert string enum values to integers using the static parser methods
        # (these methods handle both string and int inputs)
        # Only convert if not None - will check custom attributes later if None
        if solver is not None:
            solver = self._parse_solver(solver)
        if integrator is not None:
            integrator = self._parse_integrator(integrator)
        if cone is not None:
            cone = self._parse_cone(cone)
        if jacobian is not None:
            jacobian = self._parse_jacobian(jacobian)

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

        # Solver option resolution priority (highest to lowest):
        #   1. Constructor argument (e.g., impratio=5.0) - same value for all worlds
        #   2. Newton model custom attribute (model.mujoco.<option>) - supports per-world values
        #   3. MuJoCo default

        # Track which WORLD frequency options were overridden by constructor
        overridden_options = set()

        # Get mujoco custom attributes once
        mujoco_attrs = getattr(model, "mujoco", None)

        # Helper to resolve scalar option value
        def resolve_option(name: str, constructor_value):
            """Resolve scalar option from constructor > model attribute > None (use MuJoCo default)."""
            if constructor_value is not None:
                overridden_options.add(name)
                return constructor_value
            if mujoco_attrs and hasattr(mujoco_attrs, name):
                # Read from index 0 (template world) for initialization
                return float(getattr(mujoco_attrs, name).numpy()[0])
            return None

        # Helper to resolve vector option value
        def resolve_vector_option(name: str, constructor_value):
            """Resolve vector option from constructor > model attribute > None (use MuJoCo default)."""
            if constructor_value is not None:
                overridden_options.add(name)
                return constructor_value
            if mujoco_attrs and hasattr(mujoco_attrs, name):
                # Read from index 0 (template world) for initialization
                vec = getattr(mujoco_attrs, name).numpy()[0]
                return tuple(vec)
            return None

        # Resolve all WORLD frequency scalar options
        impratio = resolve_option("impratio", impratio)
        tolerance = resolve_option("tolerance", tolerance)
        ls_tolerance = resolve_option("ls_tolerance", ls_tolerance)
        ccd_tolerance = resolve_option("ccd_tolerance", ccd_tolerance)
        density = resolve_option("density", density)
        viscosity = resolve_option("viscosity", viscosity)

        # Resolve WORLD frequency vector options
        wind = resolve_vector_option("wind", wind)
        magnetic = resolve_vector_option("magnetic", magnetic)

        # Resolve ONCE frequency numeric options from custom attributes if not provided
        if iterations is None and mujoco_attrs and hasattr(mujoco_attrs, "iterations"):
            iterations = int(mujoco_attrs.iterations.numpy()[0])
        if ls_iterations is None and mujoco_attrs and hasattr(mujoco_attrs, "ls_iterations"):
            ls_iterations = int(mujoco_attrs.ls_iterations.numpy()[0])
        if ccd_iterations is None and mujoco_attrs and hasattr(mujoco_attrs, "ccd_iterations"):
            ccd_iterations = int(mujoco_attrs.ccd_iterations.numpy()[0])
        if sdf_iterations is None and mujoco_attrs and hasattr(mujoco_attrs, "sdf_iterations"):
            sdf_iterations = int(mujoco_attrs.sdf_iterations.numpy()[0])
        if sdf_initpoints is None and mujoco_attrs and hasattr(mujoco_attrs, "sdf_initpoints"):
            sdf_initpoints = int(mujoco_attrs.sdf_initpoints.numpy()[0])

        # Set defaults for numeric options if still None (use MuJoCo defaults)
        if iterations is None:
            iterations = 100
        if ls_iterations is None:
            ls_iterations = 50

        # Resolve ONCE frequency enum options from custom attributes if not provided
        if solver is None and mujoco_attrs and hasattr(mujoco_attrs, "solver"):
            solver = int(mujoco_attrs.solver.numpy()[0])
        if integrator is None and mujoco_attrs and hasattr(mujoco_attrs, "integrator"):
            integrator = int(mujoco_attrs.integrator.numpy()[0])
        if cone is None and mujoco_attrs and hasattr(mujoco_attrs, "cone"):
            cone = int(mujoco_attrs.cone.numpy()[0])
        if jacobian is None and mujoco_attrs and hasattr(mujoco_attrs, "jacobian"):
            jacobian = int(mujoco_attrs.jacobian.numpy()[0])

        # Set defaults for enum options if still None (use Newton defaults, not MuJoCo defaults)
        if solver is None:
            solver = mujoco.mjtSolver.mjSOL_NEWTON  # Newton default (not CG)
        if integrator is None:
            integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # Newton default (not Euler)
        if cone is None:
            cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        if jacobian is None:
            jacobian = mujoco.mjtJacobian.mjJAC_AUTO

        spec = mujoco.MjSpec()
        spec.option.disableflags = disableflags
        spec.option.gravity = np.array([*model.gravity.numpy()[0]])
        spec.option.solver = solver
        spec.option.integrator = integrator
        spec.option.iterations = iterations
        spec.option.ls_iterations = ls_iterations
        spec.option.cone = cone
        spec.option.jacobian = jacobian

        # Set ONCE frequency numeric options (use MuJoCo defaults if None)
        if ccd_iterations is not None:
            spec.option.ccd_iterations = ccd_iterations
        if sdf_iterations is not None:
            spec.option.sdf_iterations = sdf_iterations
        if sdf_initpoints is not None:
            spec.option.sdf_initpoints = sdf_initpoints

        # Set WORLD frequency options (use MuJoCo defaults if None)
        if impratio is not None:
            spec.option.impratio = impratio
        if tolerance is not None:
            spec.option.tolerance = tolerance
        if ls_tolerance is not None:
            spec.option.ls_tolerance = ls_tolerance
        if ccd_tolerance is not None:
            spec.option.ccd_tolerance = ccd_tolerance
        if density is not None:
            spec.option.density = density
        if viscosity is not None:
            spec.option.viscosity = viscosity
        if wind is not None:
            spec.option.wind = np.array(wind)
        if magnetic is not None:
            spec.option.magnetic = np.array(magnetic)

        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO
        if mujoco_attrs and hasattr(mujoco_attrs, "autolimits"):
            spec.compiler.autolimits = bool(mujoco_attrs.autolimits.numpy()[0])

        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_articulation = model.joint_articulation.numpy()
        joint_parent_xform = model.joint_X_p.numpy()
        joint_child_xform = model.joint_X_c.numpy()
        joint_limit_lower = model.joint_limit_lower.numpy()
        joint_limit_upper = model.joint_limit_upper.numpy()
        joint_limit_ke = model.joint_limit_ke.numpy()
        joint_limit_kd = model.joint_limit_kd.numpy()
        joint_type = model.joint_type.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_armature = model.joint_armature.numpy()
        joint_effort_limit = model.joint_effort_limit.numpy()
        # Per-DOF actuator arrays
        joint_act_mode = model.joint_act_mode.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_target_kd = model.joint_target_kd.numpy()
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
        shape_ke = model.shape_material_ke.numpy()
        shape_kd = model.shape_material_kd.numpy()
        shape_mu_torsional = model.shape_material_mu_torsional.numpy()
        shape_mu_rolling = model.shape_material_mu_rolling.numpy()
        shape_thickness = model.shape_thickness.numpy()

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
        shape_priority = get_custom_attribute("geom_priority")
        shape_geom_solimp = get_custom_attribute("geom_solimp")
        shape_geom_solmix = get_custom_attribute("geom_solmix")
        shape_geom_gap = get_custom_attribute("geom_gap")
        joint_dof_limit_margin = get_custom_attribute("limit_margin")
        joint_solimp_limit = get_custom_attribute("solimplimit")
        joint_dof_solref = get_custom_attribute("solreffriction")
        joint_dof_solimp = get_custom_attribute("solimpfriction")
        joint_stiffness = get_custom_attribute("dof_passive_stiffness")
        joint_damping = get_custom_attribute("dof_passive_damping")
        joint_actgravcomp = get_custom_attribute("jnt_actgravcomp")
        joint_springref = get_custom_attribute("dof_springref")
        joint_ref = get_custom_attribute("dof_ref")

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
        eq_constraint_solref = get_custom_attribute("eq_solref")

        # Read mimic constraint arrays
        mimic_joint0 = model.constraint_mimic_joint0.numpy()
        mimic_joint1 = model.constraint_mimic_joint1.numpy()
        mimic_coef0 = model.constraint_mimic_coef0.numpy()
        mimic_coef1 = model.constraint_mimic_coef1.numpy()
        mimic_enabled = model.constraint_mimic_enabled.numpy()
        mimic_world = model.constraint_mimic_world.numpy()

        INT32_MAX = np.iinfo(np.int32).max
        collision_mask_everything = INT32_MAX

        # mapping from joint axis to actuator index
        # axis_to_actuator[i, 0] = position actuator index
        # axis_to_actuator[i, 1] = velocity actuator index
        axis_to_actuator = np.zeros((model.joint_dof_count, 2), dtype=np.int32) - 1
        actuator_count = 0

        # Track actuator mapping as they're created (indexed by MuJoCo actuator order)
        # ctrl_source: 0=JOINT_TARGET, 1=CTRL_DIRECT
        # to_newton_idx: for JOINT_TARGET: >=0 position DOF, -1 unmapped, <=-2 velocity (DOF = -(val+2))
        #                for CTRL_DIRECT: MJCF-order index into control.mujoco.ctrl
        mjc_actuator_ctrl_source_list: list[int] = []
        mjc_actuator_to_newton_idx_list: list[int] = []

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
            GeoType.HFIELD: mujoco.mjtGeom.mjGEOM_HFIELD,
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
        # Store mapping from Newton joint index to MuJoCo joint name
        joint_mapping = {}
        # track mocap index for each Newton body (dict: newton_body_id -> mocap_index)
        newton_body_to_mocap_index = {}
        # counter for assigning sequential mocap indices
        next_mocap_index = 0

        # ensure unique names
        body_name_counts = {}
        joint_names = {}

        if separate_worlds:
            # determine which shapes, bodies and joints belong to the first world
            # based on the body world indices: we pick objects from the first world and global shapes
            non_negatives = body_world[body_world >= 0]
            if len(non_negatives) > 0:
                first_world = np.min(non_negatives)
            else:
                first_world = -1
            selected_shapes = np.where((shape_world == first_world) | (shape_world < 0))[0].astype(np.int32)
            selected_bodies = np.where((body_world == first_world) | (body_world < 0))[0].astype(np.int32)
            selected_joints = np.where((joint_world == first_world) | (joint_world < 0))[0].astype(np.int32)
            selected_constraints = np.where((eq_constraint_world == first_world) | (eq_constraint_world < 0))[0].astype(
                np.int32
            )
            selected_mimic_constraints = np.where((mimic_world == first_world) | (mimic_world < 0))[0].astype(np.int32)
        else:
            # if we are not separating environments to worlds, we use all shapes, bodies, joints
            first_world = 0

            # if we are not separating worlds, we use all shapes, bodies, joints, constraints
            selected_shapes = np.arange(model.shape_count, dtype=np.int32)
            selected_bodies = np.arange(model.body_count, dtype=np.int32)
            selected_joints = np.arange(model.joint_count, dtype=np.int32)
            selected_constraints = np.arange(model.equality_constraint_count, dtype=np.int32)
            selected_mimic_constraints = np.arange(model.constraint_mimic_count, dtype=np.int32)

        # get the shapes for the first environment
        first_env_shapes = np.where(shape_world == first_world)[0]

        # split joints into loop and non-loop joints (loop joints will be instantiated separately as equality constraints)
        joints_loop = selected_joints[joint_articulation[selected_joints] == -1]
        joints_non_loop = selected_joints[joint_articulation[selected_joints] >= 0]
        # sort joints topologically depth-first since this is the order that will also be used
        # for placing bodies in the MuJoCo model
        joints_simple = [(joint_parent[i], joint_child[i]) for i in joints_non_loop]
        joint_order = topological_sort(joints_simple, use_dfs=True, custom_indices=joints_non_loop)
        if any(joint_order[i] != joints_non_loop[i] for i in range(len(joints_simple))):
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
                if stype == GeoType.HFIELD:
                    # Retrieve heightfield source
                    hfield_src = model.shape_source[shape]
                    if hfield_src is None:
                        if wp.config.verbose:
                            print(f"Warning: Heightfield shape {shape} has no source data, skipping")
                        continue

                    # Convert Newton heightfield to MuJoCo format
                    # MuJoCo size: (size_x, size_y, size_z, size_base) — all must be positive
                    # Our data is normalized [0,1], height range = max_z - min_z
                    # We set size_base to eps (MuJoCo requires positive) and shift the
                    # geom origin by min_z so the lowest point is at the right world Z.
                    eps = 1e-4
                    mj_size_z = max(hfield_src.max_z - hfield_src.min_z, eps)
                    mj_size = (hfield_src.hx, hfield_src.hy, mj_size_z, eps)
                    elevation_data = hfield_src.data.flatten()

                    hfield_name = f"{model.shape_key[shape]}_{shape}"
                    spec.add_hfield(
                        name=hfield_name,
                        nrow=hfield_src.nrow,
                        ncol=hfield_src.ncol,
                        size=mj_size,
                        userdata=elevation_data,
                    )

                    geom_params["hfieldname"] = hfield_name

                    # Shift geom origin so data=0 maps to min_z in world space
                    tf = wp.transform(
                        wp.vec3(tf.p[0], tf.p[1], tf.p[2] + hfield_src.min_z),
                        tf.q,
                    )
                elif stype == GeoType.MESH or stype == GeoType.CONVEX_MESH:
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
                torsional = shape_mu_torsional[shape]
                rolling = shape_mu_rolling[shape]
                geom_params["friction"] = [
                    mu,
                    torsional,
                    rolling,
                ]

                # set solref from shape stiffness and damping
                geom_params["solref"] = convert_solref(float(shape_ke[shape]), float(shape_kd[shape]), 1.0, 1.0)

                if shape_condim is not None:
                    geom_params["condim"] = shape_condim[shape]
                if shape_priority is not None:
                    geom_params["priority"] = shape_priority[shape]
                if shape_geom_solimp is not None:
                    geom_params["solimp"] = shape_geom_solimp[shape]
                if shape_geom_solmix is not None:
                    geom_params["solmix"] = shape_geom_solmix[shape]
                if shape_geom_gap is not None:
                    geom_params["gap"] = shape_geom_gap[shape]

                geom_params["margin"] = float(shape_thickness[shape])

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
        dof_to_mjc_joint = np.full(model.joint_dof_count // model.world_count, -1, dtype=np.int32)

        # This is needed for CTRL_DIRECT actuators targeting joints within combined Newton joints.
        mjc_joint_names: list[str] = []

        # need to keep track of current dof and joint counts to make the indexing above correct
        num_dofs = 0
        num_mjc_joints = 0

        # add joints, bodies and geoms
        for j in joint_order:
            parent, child = int(joint_parent[j]), int(joint_child[j])
            if child in body_mapping:
                raise ValueError(f"Body {child} already exists in the mapping")

            # add body
            body_mapping[child] = len(mj_bodies)

            # check if fixed-base articulation
            fixed_base = False
            if parent == -1 and joint_type[j] == JointType.FIXED:
                fixed_base = True

            # this assumes that the joint position is 0
            tf = wp.transform(*joint_parent_xform[j])
            child_xform = wp.transform(*joint_child_xform[j])
            tf = tf * wp.transform_inverse(child_xform)

            joint_pos = child_xform.p
            joint_rot = child_xform.q

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

            # Store mapping from Newton joint index to MuJoCo joint name
            joint_mapping[j] = name

            joint_mjc_dof_start[j] = num_dofs

            if j_type == JointType.FREE:
                body.add_joint(
                    name=name,
                    type=mujoco.mjtJoint.mjJNT_FREE,
                    damping=0.0,
                    limited=False,
                )
                mjc_joint_names.append(name)
                # For free joints, all 6 DOFs map to the same MuJoCo joint
                for i in range(6):
                    dof_to_mjc_joint[qd_start + i] = num_mjc_joints
                num_dofs += 6
                num_mjc_joints += 1
            elif j_type == JointType.BALL:
                body.add_joint(
                    name=name,
                    type=mujoco.mjtJoint.mjJNT_BALL,
                    axis=wp.quat_rotate(joint_rot, wp.vec3(1.0, 0.0, 0.0)),
                    pos=joint_pos,
                    damping=0.0,
                    limited=False,
                    armature=joint_armature[qd_start],
                    frictionloss=joint_friction[qd_start],
                )
                mjc_joint_names.append(name)
                # For ball joints, all 3 DOFs map to the same MuJoCo joint
                for i in range(3):
                    dof_to_mjc_joint[qd_start + i] = num_mjc_joints
                num_dofs += 3
                num_mjc_joints += 1
                # Add actuators for the ball joint using per-DOF arrays
                for i in range(3):
                    ai = qd_start + i
                    mode = joint_act_mode[ai]

                    if (actuated_axes is None or ai in actuated_axes) and mode != int(ActuatorMode.NONE):
                        kp = joint_target_ke[ai]
                        kd = joint_target_kd[ai]
                        effort_limit = joint_effort_limit[ai]
                        gear = actuator_gears.get(name)
                        args = {}
                        args.update(actuator_args)
                        args["gear"] = [0.0] * 6
                        if gear is not None:
                            args["gear"][i] = gear
                        else:
                            args["gear"][i] = 1.0
                        args["forcerange"] = [-effort_limit, effort_limit]

                        template_dof = ai
                        # Add position actuator if mode includes position
                        if mode == ActuatorMode.POSITION:
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=name, **args)
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            actuator_count += 1
                        elif mode == ActuatorMode.POSITION_VELOCITY:
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=name, **args)
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            actuator_count += 1

                        # Add velocity actuator if mode includes velocity
                        if mode in (ActuatorMode.VELOCITY, ActuatorMode.POSITION_VELOCITY):
                            args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=name, **args)
                            axis_to_actuator[ai, 1] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(-(template_dof + 2))  # negative = velocity
                            actuator_count += 1
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
                    if joint_actgravcomp is not None:
                        joint_params["actgravcomp"] = joint_actgravcomp[ai]
                    lower, upper = joint_limit_lower[ai], joint_limit_upper[ai]
                    if lower <= -MAXVAL and upper >= MAXVAL:
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
                    # Use actfrcrange to clamp total actuator force (P+D sum) on this joint
                    if actuated_axes is None or ai in actuated_axes:
                        effort_limit = joint_effort_limit[ai]
                        joint_params["actfrclimited"] = True
                        joint_params["actfrcrange"] = (-effort_limit, effort_limit)

                    if joint_springref is not None:
                        joint_params["springref"] = joint_springref[ai]
                    if joint_ref is not None:
                        joint_params["ref"] = joint_ref[ai]

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
                    mjc_joint_names.append(axname)
                    # Map this DOF to the current MuJoCo joint index
                    dof_to_mjc_joint[ai] = num_mjc_joints
                    num_mjc_joints += 1

                    mode = joint_act_mode[ai]
                    if (actuated_axes is None or ai in actuated_axes) and mode != int(ActuatorMode.NONE):
                        kp = joint_target_ke[ai]
                        kd = joint_target_kd[ai]
                        gear = actuator_gears.get(axname)
                        if gear is not None:
                            args = {}
                            args.update(actuator_args)
                            args["gear"] = [gear, 0.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            args = actuator_args

                        template_dof = ai
                        # Add position actuator if mode includes position
                        if mode == ActuatorMode.POSITION:
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **args)
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            actuator_count += 1
                        elif mode == ActuatorMode.POSITION_VELOCITY:
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **args)
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            actuator_count += 1

                        # Add velocity actuator if mode includes velocity
                        if mode in (ActuatorMode.VELOCITY, ActuatorMode.POSITION_VELOCITY):
                            args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **args)
                            axis_to_actuator[ai, 1] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(-(template_dof + 2))  # negative = velocity
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
                        joint_params["stiffness"] = joint_stiffness[ai] * (np.pi / 180)
                    if joint_damping is not None:
                        joint_params["damping"] = joint_damping[ai] * (np.pi / 180)
                    if joint_actgravcomp is not None:
                        joint_params["actgravcomp"] = joint_actgravcomp[ai]
                    lower, upper = joint_limit_lower[ai], joint_limit_upper[ai]
                    if lower <= -MAXVAL and upper >= MAXVAL:
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
                    # Use actfrcrange to clamp total actuator force (P+D sum) on this joint
                    if actuated_axes is None or ai in actuated_axes:
                        effort_limit = joint_effort_limit[ai]
                        joint_params["actfrclimited"] = True
                        joint_params["actfrcrange"] = (-effort_limit, effort_limit)

                    if joint_springref is not None:
                        joint_params["springref"] = np.rad2deg(joint_springref[ai])
                    if joint_ref is not None:
                        joint_params["ref"] = np.rad2deg(joint_ref[ai])

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
                    mjc_joint_names.append(axname)
                    # Map this DOF to the current MuJoCo joint index
                    dof_to_mjc_joint[ai] = num_mjc_joints
                    num_mjc_joints += 1

                    mode = joint_act_mode[ai]
                    if (actuated_axes is None or ai in actuated_axes) and mode != int(ActuatorMode.NONE):
                        kp = joint_target_ke[ai]
                        kd = joint_target_kd[ai]
                        gear = actuator_gears.get(axname)
                        if gear is not None:
                            args = {}
                            args.update(actuator_args)
                            args["gear"] = [gear, 0.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            args = actuator_args

                        template_dof = ai
                        # Add position actuator if mode includes position
                        if mode == ActuatorMode.POSITION:
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, -kp, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **args)
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            actuator_count += 1
                        elif mode == ActuatorMode.POSITION_VELOCITY:
                            args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **args)
                            axis_to_actuator[ai, 0] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(template_dof)  # positive = position
                            actuator_count += 1

                        # Add velocity actuator if mode includes velocity
                        if mode in (ActuatorMode.VELOCITY, ActuatorMode.POSITION_VELOCITY):
                            args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                            spec.add_actuator(target=axname, **args)
                            axis_to_actuator[ai, 1] = actuator_count
                            mjc_actuator_ctrl_source_list.append(0)  # JOINT_TARGET
                            mjc_actuator_to_newton_idx_list.append(-(template_dof + 2))  # negative = velocity
                            actuator_count += 1

                        # Note: MuJoCo general actuators are handled separately via custom attributes

            elif j_type != JointType.FIXED:
                raise NotImplementedError(f"Joint type {j_type} is not supported yet")

            add_geoms(child)

        def get_body_name(body_idx: int) -> str:
            """Get body name, handling world body (-1) correctly."""
            if body_idx == -1:
                return "world"
            return model.body_key[body_idx]

        for i in selected_constraints:
            constraint_type = eq_constraint_type[i]
            if constraint_type == EqType.CONNECT:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_CONNECT
                eq.active = eq_constraint_enabled[i]
                eq.name1 = get_body_name(eq_constraint_body1[i])
                eq.name2 = get_body_name(eq_constraint_body2[i])
                eq.data[0:3] = eq_constraint_anchor[i]
                if eq_constraint_solref is not None:
                    eq.solref = eq_constraint_solref[i]

            elif constraint_type == EqType.JOINT:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_JOINT)
                eq.type = mujoco.mjtEq.mjEQ_JOINT
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.joint_key[eq_constraint_joint1[i]]
                eq.name2 = model.joint_key[eq_constraint_joint2[i]]
                eq.data[0:5] = eq_constraint_polycoef[i]
                if eq_constraint_solref is not None:
                    eq.solref = eq_constraint_solref[i]

            elif constraint_type == EqType.WELD:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.active = eq_constraint_enabled[i]
                eq.name1 = get_body_name(eq_constraint_body1[i])
                eq.name2 = get_body_name(eq_constraint_body2[i])
                cns_relpose = wp.transform(*eq_constraint_relpose[i])
                eq.data[0:3] = eq_constraint_anchor[i]
                eq.data[3:6] = wp.transform_get_translation(cns_relpose)
                eq.data[6:10] = wp.transform_get_rotation(cns_relpose)
                eq.data[10] = eq_constraint_torquescale[i]
                if eq_constraint_solref is not None:
                    eq.solref = eq_constraint_solref[i]

        # add connect constraints for joints that are excluded from the articulation
        # (the UsdPhysics way of defining loop closures)
        mjc_eq_to_newton_jnt = {}
        for j in joints_loop:
            eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
            eq.type = mujoco.mjtEq.mjEQ_CONNECT
            eq.active = True
            eq.name1 = model.body_key[joint_parent[j]]
            eq.name2 = model.body_key[joint_child[j]]
            eq.data[0:3] = joint_parent_xform[j][:3]
            mjc_eq_to_newton_jnt[eq.id] = j

            eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
            eq.type = mujoco.mjtEq.mjEQ_CONNECT
            eq.active = True
            eq.name1 = model.body_key[joint_child[j]]
            eq.name2 = model.body_key[joint_parent[j]]
            eq.data[0:3] = joint_child_xform[j][:3]
            mjc_eq_to_newton_jnt[eq.id] = j

        # add mimic constraints as mjEQ_JOINT equality constraints
        mjc_eq_to_newton_mimic_dict = {}
        for i in selected_mimic_constraints:
            j0 = mimic_joint0[i]  # follower
            j1 = mimic_joint1[i]  # leader

            # check that both joints exist in the MuJoCo joint mapping
            j0_name = joint_mapping.get(j0)
            j1_name = joint_mapping.get(j1)
            if j0_name is None or j1_name is None:
                warnings.warn(
                    f"Skipping mimic constraint {i}: follower joint {j0} or leader joint {j1} "
                    f"not found in MuJoCo joint mapping.",
                    stacklevel=2,
                )
                continue

            # mjEQ_JOINT only works with scalar joints (hinge/slide)
            j0_type = joint_type[j0]
            j1_type = joint_type[j1]
            if j0_type not in (JointType.REVOLUTE, JointType.PRISMATIC):
                warnings.warn(
                    f"Skipping mimic constraint {i}: follower joint {j0} has unsupported type "
                    f"{JointType(j0_type).name} for mjEQ_JOINT (only REVOLUTE and PRISMATIC supported).",
                    stacklevel=2,
                )
                continue
            if j1_type not in (JointType.REVOLUTE, JointType.PRISMATIC):
                warnings.warn(
                    f"Skipping mimic constraint {i}: leader joint {j1} has unsupported type "
                    f"{JointType(j1_type).name} for mjEQ_JOINT (only REVOLUTE and PRISMATIC supported).",
                    stacklevel=2,
                )
                continue

            eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_JOINT)
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.active = bool(mimic_enabled[i])
            eq.name1 = j0_name  # follower (constrained joint)
            eq.name2 = j1_name  # leader (driving joint)
            # polycoef: data[0] + data[1]*q2 + data[2]*q2^2 + ... - q1 = 0
            # mimic: q1 = coef0 + coef1*q2
            eq.data[0] = float(mimic_coef0[i])
            eq.data[1] = float(mimic_coef1[i])
            eq.data[2] = 0.0
            eq.data[3] = 0.0
            eq.data[4] = 0.0
            mjc_eq_to_newton_mimic_dict[eq.id] = i

        if skip_visual_only_geoms and len(spec.geoms) != colliding_shapes_per_world:
            raise ValueError(
                "The number of geoms in the MuJoCo model does not match the number of colliding shapes in the Newton model."
            )

        if len(spec.bodies) != len(selected_bodies) + 1:  # +1 for the world body
            raise ValueError(
                "The number of bodies in the MuJoCo model does not match the number of selected bodies in the Newton model. "
                "Make sure that each body has an incoming joint and that the joints are part of an articulation."
            )

        # add contact exclusions between bodies to ensure parent <> child collisions are ignored
        # even when one of the bodies is static
        for b1, b2 in body_filters:
            mb1, mb2 = body_mapping[b1], body_mapping[b2]
            spec.add_exclude(bodyname1=spec.bodies[mb1].name, bodyname2=spec.bodies[mb2].name)

        # add explicit contact pairs from custom attributes
        self._init_pairs(model, spec, shape_mapping, first_world)

        selected_tendons, mjc_tendon_names = self._init_tendons(model, spec, joint_mapping, first_world)

        # Process MuJoCo general actuators (motor, general, etc.) from custom attributes
        actuator_count += self._init_actuators(
            model,
            spec,
            first_world,
            actuator_args,
            mjc_actuator_ctrl_source_list,
            mjc_actuator_to_newton_idx_list,
            dof_to_mjc_joint,
            mjc_joint_names,
            selected_tendons,
            mjc_tendon_names,
        )

        # Convert actuator mapping lists to warp arrays
        if mjc_actuator_ctrl_source_list:
            self.mjc_actuator_ctrl_source = wp.array(
                np.array(mjc_actuator_ctrl_source_list, dtype=np.int32),
                dtype=wp.int32,
                device=model.device,
            )
            self.mjc_actuator_to_newton_idx = wp.array(
                np.array(mjc_actuator_to_newton_idx_list, dtype=np.int32),
                dtype=wp.int32,
                device=model.device,
            )
        else:
            self.mjc_actuator_ctrl_source = None
            self.mjc_actuator_to_newton_idx = None

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

            # Determine nworld for mapping dimensions
            nworld = model.world_count if separate_worlds else 1

            # --- Create unified mappings: MuJoCo[world, entity] -> Newton[entity] ---

            # Build geom -> shape mapping
            # geom_to_shape_idx maps from MuJoCo geom index to absolute Newton shape index.
            # Convert non-static shapes to template-relative indices for the kernel.
            geom_to_shape_idx_np = np.full((self.mj_model.ngeom,), -1, dtype=np.int32)

            # Find the minimum shape index for the first non-static group to use as the base
            first_env_shape_base = int(np.min(first_env_shapes)) if len(first_env_shapes) > 0 else 0

            # Store for lazy inverse creation
            self._shapes_per_world = len(first_env_shapes)
            self._first_env_shape_base = first_env_shape_base

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

            # Create mjc_geom_to_newton_shape: MuJoCo[world, geom] -> Newton shape
            self.mjc_geom_to_newton_shape = wp.full((nworld, self.mj_model.ngeom), -1, dtype=wp.int32)

            if self.mjw_model.geom_pos.size:
                wp.launch(
                    update_shape_mappings_kernel,
                    dim=(nworld, self.mj_model.ngeom),
                    inputs=[
                        geom_to_shape_idx_wp,
                        geom_is_static_wp,
                        self._shapes_per_world,
                        first_env_shape_base,
                    ],
                    outputs=[
                        self.mjc_geom_to_newton_shape,
                    ],
                    device=model.device,
                )

            # Create mjc_body_to_newton: MuJoCo[world, body] -> Newton body
            # body_mapping is {newton_body_id: mjc_body_id}, we need to invert it
            # and expand to 2D for all worlds
            nbody = self.mj_model.nbody
            bodies_per_world = model.body_count // model.world_count
            mjc_body_to_newton_np = np.full((nworld, nbody), -1, dtype=np.int32)
            for newton_body, mjc_body in body_mapping.items():
                if newton_body >= 0:  # Skip world body (-1 -> 0)
                    newton_body_in_world = newton_body % bodies_per_world
                    for w in range(nworld):
                        mjc_body_to_newton_np[w, mjc_body] = w * bodies_per_world + newton_body_in_world
            self.mjc_body_to_newton = wp.array(mjc_body_to_newton_np, dtype=wp.int32)

            # Common variables for mapping creation
            njnt = self.mj_model.njnt
            joints_per_world = model.joint_count // model.world_count
            dofs_per_world = model.joint_dof_count // model.world_count

            # Map each Newton body to the qd_start of its free/DISTANCE joint (or -1).
            # Use selected_joints as the template and tile offsets across worlds.
            joint_type_np = model.joint_type.numpy()
            joint_child_np = model.joint_child.numpy()
            joint_qd_start_np = model.joint_qd_start.numpy()

            template_joint_types = joint_type_np[selected_joints]
            free_mask = np.isin(template_joint_types, (JointType.FREE, JointType.DISTANCE))
            body_free_qd_start_np = np.full(model.body_count, -1, dtype=np.int32)
            if np.any(free_mask):
                template_children = joint_child_np[selected_joints] % bodies_per_world
                template_qd_start = joint_qd_start_np[selected_joints] % dofs_per_world
                child_free = template_children[free_mask]
                qd_start_free = template_qd_start[free_mask]
                world_body_offsets = (np.arange(model.world_count, dtype=np.int32) * bodies_per_world)[:, None]
                world_qd_offsets = (np.arange(model.world_count, dtype=np.int32) * dofs_per_world)[:, None]
                body_indices = (child_free[None, :] + world_body_offsets).ravel()
                qd_starts = (qd_start_free[None, :] + world_qd_offsets).ravel()
                body_free_qd_start_np[body_indices] = qd_starts

            self.body_free_qd_start = wp.array(body_free_qd_start_np, dtype=wp.int32)

            # Create mjc_mocap_to_newton_jnt: MuJoCo[world, mocap] -> Newton joint index
            # Mocap bodies are created from fixed-base articulations (FIXED joint to world)
            # In MuJoCo, they have body_mocapid >= 0 and no joint
            nmocap = self.mj_model.nmocap
            if nmocap > 0:
                mjc_mocap_to_newton_jnt_np = np.full((nworld, nmocap), -1, dtype=np.int32)
                body_mocapid = self.mj_model.body_mocapid
                # For each MuJoCo body that is a mocap body, find the corresponding Newton joint
                for mjc_body in range(nbody):
                    mocap_idx = body_mocapid[mjc_body]
                    if mocap_idx >= 0:
                        # Find the Newton body for this MuJoCo body
                        newton_body = mjc_body_to_newton_np[0, mjc_body]
                        if newton_body >= 0:
                            newton_body_template = newton_body % bodies_per_world
                            # Find the Newton joint that has this body as child
                            for j in range(joints_per_world):
                                if joint_child[j] == newton_body_template:
                                    for w in range(nworld):
                                        mjc_mocap_to_newton_jnt_np[w, mocap_idx] = w * joints_per_world + j
                                    break
                self.mjc_mocap_to_newton_jnt = wp.array(mjc_mocap_to_newton_jnt_np, dtype=wp.int32)

            # Create mjc_jnt_to_newton_jnt: MuJoCo[world, joint] -> Newton joint index
            # selected_joints[idx] is the Newton template joint index
            mjc_jnt_to_newton_jnt_np = np.full((nworld, njnt), -1, dtype=np.int32)
            # Invert dof_to_mjc_joint to get mjc_jnt -> template_dof, then find the joint
            for template_dof, mjc_jnt in enumerate(dof_to_mjc_joint):
                if mjc_jnt >= 0:
                    # Find which Newton template joint contains this DOF
                    # This is the first DOF of the joint, so we can search for it
                    for _ji, j in enumerate(selected_joints):
                        j_dof_start = joint_qd_start[j] % dofs_per_world
                        j_lin_count, j_ang_count = joint_dof_dim[j]
                        j_dof_end = j_dof_start + j_lin_count + j_ang_count
                        if j_dof_start <= template_dof < j_dof_end:
                            for w in range(nworld):
                                mjc_jnt_to_newton_jnt_np[w, mjc_jnt] = w * joints_per_world + j
                            break
            self.mjc_jnt_to_newton_jnt = wp.array(mjc_jnt_to_newton_jnt_np, dtype=wp.int32)

            # Create mjc_jnt_to_newton_dof: MuJoCo[world, joint] -> Newton DOF start
            # joint_mjc_dof_start[template_joint] -> mjc_dof_start
            # dof_to_mjc_joint[template_dof] -> mjc_joint
            mjc_jnt_to_newton_dof_np = np.full((nworld, njnt), -1, dtype=np.int32)
            for template_dof, mjc_jnt in enumerate(dof_to_mjc_joint):
                if mjc_jnt >= 0:
                    for w in range(nworld):
                        mjc_jnt_to_newton_dof_np[w, mjc_jnt] = w * dofs_per_world + template_dof
            self.mjc_jnt_to_newton_dof = wp.array(mjc_jnt_to_newton_dof_np, dtype=wp.int32)

            # Create mjc_dof_to_newton_dof: MuJoCo[world, dof] -> Newton DOF
            nv = self.mj_model.nv  # Number of DOFs in MuJoCo
            mjc_dof_to_newton_dof_np = np.full((nworld, nv), -1, dtype=np.int32)
            # joint_mjc_dof_start tells us where each Newton template joint's DOFs start in MuJoCo
            for j, mjc_dof_start in enumerate(joint_mjc_dof_start):
                if mjc_dof_start >= 0:
                    newton_dof_start = joint_qd_start[j]
                    lin_count, ang_count = joint_dof_dim[j]
                    total_dofs = lin_count + ang_count
                    for d in range(total_dofs):
                        mjc_dof = mjc_dof_start + d
                        template_newton_dof = (newton_dof_start % dofs_per_world) + d
                        for w in range(nworld):
                            mjc_dof_to_newton_dof_np[w, mjc_dof] = w * dofs_per_world + template_newton_dof
            self.mjc_dof_to_newton_dof = wp.array(mjc_dof_to_newton_dof_np, dtype=wp.int32)

            # Create mjc_eq_to_newton_eq: MuJoCo[world, eq] -> Newton equality constraint
            # selected_constraints[idx] is the Newton template constraint index
            neq = self.mj_model.neq
            eq_constraints_per_world = model.equality_constraint_count // model.world_count
            mjc_eq_to_newton_eq_np = np.full((nworld, neq), -1, dtype=np.int32)
            mjc_eq_to_newton_jnt_np = np.full((nworld, neq), -1, dtype=np.int32)
            for mjc_eq, newton_eq in enumerate(selected_constraints):
                template_eq = newton_eq % eq_constraints_per_world if eq_constraints_per_world > 0 else newton_eq
                for w in range(nworld):
                    mjc_eq_to_newton_eq_np[w, mjc_eq] = w * eq_constraints_per_world + template_eq
            for mjc_eq, newton_jnt in mjc_eq_to_newton_jnt.items():
                for w in range(nworld):
                    mjc_eq_to_newton_jnt_np[w, mjc_eq] = w * joints_per_world + newton_jnt
            self.mjc_eq_to_newton_eq = wp.array(mjc_eq_to_newton_eq_np, dtype=wp.int32)
            self.mjc_eq_to_newton_jnt = wp.array(mjc_eq_to_newton_jnt_np, dtype=wp.int32)

            # Create mjc_eq_to_newton_mimic: MuJoCo[world, eq] -> Newton mimic constraint
            mimic_per_world = (
                model.constraint_mimic_count // model.world_count
                if model.world_count > 0
                else model.constraint_mimic_count
            )
            mjc_eq_to_newton_mimic_np = np.full((nworld, neq), -1, dtype=np.int32)
            for mjc_eq, newton_mimic in mjc_eq_to_newton_mimic_dict.items():
                template_mimic = newton_mimic % mimic_per_world if mimic_per_world > 0 else newton_mimic
                for w in range(nworld):
                    mjc_eq_to_newton_mimic_np[w, mjc_eq] = w * mimic_per_world + template_mimic
            self.mjc_eq_to_newton_mimic = wp.array(mjc_eq_to_newton_mimic_np, dtype=wp.int32)

            # Create mjc_tendon_to_newton_tendon: MuJoCo[world, tendon] -> Newton tendon
            # selected_tendons[idx] is the Newton template tendon index
            ntendon = self.mj_model.ntendon
            if ntendon > 0:
                # Get tendon count per world from custom attributes
                mujoco_attrs = getattr(model, "mujoco", None)
                tendon_world = getattr(mujoco_attrs, "tendon_world", None) if mujoco_attrs else None
                if tendon_world is not None:
                    total_tendons = len(tendon_world)
                    tendons_per_world = total_tendons // model.world_count if model.world_count > 0 else total_tendons
                else:
                    tendons_per_world = ntendon
                mjc_tendon_to_newton_tendon_np = np.full((nworld, ntendon), -1, dtype=np.int32)
                for mjc_tendon, newton_tendon in enumerate(selected_tendons):
                    template_tendon = newton_tendon % tendons_per_world if tendons_per_world > 0 else newton_tendon
                    for w in range(nworld):
                        mjc_tendon_to_newton_tendon_np[w, mjc_tendon] = w * tendons_per_world + template_tendon
                self.mjc_tendon_to_newton_tendon = wp.array(mjc_tendon_to_newton_tendon_np, dtype=wp.int32)

            # set mjwarp-only settings
            self.mjw_model.opt.ls_parallel = ls_parallel

            if separate_worlds:
                nworld = model.world_count
            else:
                nworld = 1

            # TODO find better heuristics to determine nconmax and njmax
            if disable_contacts:
                nconmax = 0
            elif nconmax is not None and nconmax < self.mj_data.ncon:
                warnings.warn(
                    f"[WARNING] Value for nconmax is changed from {nconmax} to {self.mj_data.ncon} following an MjWarp requirement.",
                    stacklevel=2,
                )
                nconmax = self.mj_data.ncon

            if njmax is not None and njmax < self.mj_data.nefc:
                warnings.warn(
                    f"[WARNING] Value for njmax is changed from {njmax} to {self.mj_data.nefc} following an MjWarp requirement.",
                    stacklevel=2,
                )
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

            # update solver options from Newton model (only if not overridden by constructor)
            self._update_solver_options(overridden_options=overridden_options)

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
            "body_subtreemass",  # Derived from body_mass, computed by set_const_fixed
            "body_inertia",
            "body_invweight0",  # Derived from inertia, computed by set_const_0
            "body_gravcomp",
            "jnt_solref",
            "jnt_solimp",
            "jnt_pos",
            "jnt_axis",
            "jnt_stiffness",
            "jnt_range",
            "jnt_actfrcrange",  # joint-level actuator force range (effort limit)
            "jnt_margin",  # corresponds to newton custom attribute "limit_margin"
            "dof_armature",
            "dof_damping",
            "dof_invweight0",  # Derived from inertia, computed by set_const_0
            "dof_frictionloss",
            "dof_solimp",
            "dof_solref",
            # "geom_matid",
            "geom_solmix",
            "geom_solref",
            "geom_solimp",
            "geom_size",
            "geom_rbound",
            "geom_pos",
            "geom_quat",
            "geom_friction",
            "geom_margin",
            "geom_gap",
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
            "eq_solref",
            "eq_solimp",
            "eq_data",
            # "actuator_dynprm",
            "actuator_gainprm",
            "actuator_biasprm",
            # "actuator_ctrlrange",
            # "actuator_forcerange",  # No longer used - force clamping via jnt_actfrcrange
            # "actuator_actrange",
            # "actuator_gear",
            "pair_solref",
            "pair_solreffriction",
            "pair_solimp",
            "pair_margin",
            "pair_gap",
            "pair_friction",
            "tendon_world",
            "tendon_solref_lim",
            "tendon_solimp_lim",
            "tendon_solref_fri",
            "tendon_solimp_fri",
            "tendon_range",
            "tendon_actfrcrange",
            "tendon_margin",
            "tendon_stiffness",
            "tendon_damping",
            "tendon_armature",
            "tendon_frictionloss",
            "tendon_lengthspring",
            "tendon_length0",  # Derived from tendon config, computed by set_const_0
            "tendon_invweight0",  # Derived from inertia, computed by set_const_0
            # "mat_rgba",
        }

        # Solver option fields to expand (nested in mj_model.opt)
        opt_fields_to_expand = {
            # "timestep",  # Excluded: conflicts with step() function parameter
            "impratio_invsqrt",
            "tolerance",
            "ls_tolerance",
            "ccd_tolerance",
            "density",
            "viscosity",
            "gravity",
            "wind",
            "magnetic",
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

        for field in mj_model.opt.__dataclass_fields__:
            if field in opt_fields_to_expand:
                array = getattr(mj_model.opt, field)
                setattr(mj_model.opt, field, tile(array))

    def _update_solver_options(self, overridden_options: set[str] | None = None):
        """Update WORLD frequency solver options from Newton model to MuJoCo Warp.

        Called after tile() to update per-world option arrays in mjw_model.opt.
        Only updates WORLD frequency options; ONCE frequency options are already
        set on MjSpec before put_model() and shared across all worlds.

        Args:
            overridden_options: Set of option names that were overridden by constructor.
                These options should not be updated from model custom attributes.
        """
        if overridden_options is None:
            overridden_options = set()

        mujoco_attrs = getattr(self.model, "mujoco", None)
        nworld = self.model.world_count

        # Helper to get WORLD frequency option array or None
        def get_option(name: str):
            if name in overridden_options or not mujoco_attrs or not hasattr(mujoco_attrs, name):
                return None
            return getattr(mujoco_attrs, name)

        # Get all WORLD frequency scalar arrays
        newton_impratio = get_option("impratio")
        newton_tolerance = get_option("tolerance")
        newton_ls_tolerance = get_option("ls_tolerance")
        newton_ccd_tolerance = get_option("ccd_tolerance")
        newton_density = get_option("density")
        newton_viscosity = get_option("viscosity")

        # Get WORLD frequency vector arrays
        newton_wind = get_option("wind")
        newton_magnetic = get_option("magnetic")

        # Skip kernel if all options are None
        if all(
            x is None
            for x in [
                newton_impratio,
                newton_tolerance,
                newton_ls_tolerance,
                newton_ccd_tolerance,
                newton_density,
                newton_viscosity,
                newton_wind,
                newton_magnetic,
            ]
        ):
            return

        wp.launch(
            update_solver_options_kernel,
            dim=nworld,
            inputs=[
                newton_impratio,
                newton_tolerance,
                newton_ls_tolerance,
                newton_ccd_tolerance,
                newton_density,
                newton_viscosity,
                newton_wind,
                newton_magnetic,
            ],
            outputs=[
                self.mjw_model.opt.impratio_invsqrt,
                self.mjw_model.opt.tolerance,
                self.mjw_model.opt.ls_tolerance,
                self.mjw_model.opt.ccd_tolerance,
                self.mjw_model.opt.density,
                self.mjw_model.opt.viscosity,
                self.mjw_model.opt.wind,
                self.mjw_model.opt.magnetic,
            ],
            device=self.model.device,
        )

    def update_model_inertial_properties(self):
        if self.model.body_count == 0:
            return

        # Get gravcomp if available
        mujoco_attrs = getattr(self.model, "mujoco", None)
        gravcomp = getattr(mujoco_attrs, "gravcomp", None) if mujoco_attrs is not None else None

        # Launch over MuJoCo bodies [nworld, nbody]
        nworld = self.mjc_body_to_newton.shape[0]
        nbody = self.mjc_body_to_newton.shape[1]

        wp.launch(
            update_body_mass_ipos_kernel,
            dim=(nworld, nbody),
            inputs=[
                self.mjc_body_to_newton,
                self.model.body_com,
                self.model.body_mass,
                gravcomp,
                self.model.up_axis,
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
            dim=(nworld, nbody),
            inputs=[
                self.mjc_body_to_newton,
                self.model.body_inertia,
            ],
            outputs=[self.mjw_model.body_inertia, self.mjw_model.body_iquat],
            device=self.model.device,
        )

        # Recompute derived quantities after mass/inertia changes.
        # set_const computes:
        # - body_subtreemass: mass of body and all descendants (depends on body_mass)
        # - dof_invweight0, body_invweight0, tendon_invweight0: inverse inertias
        # - cam_pos0, light_pos0, actuator_acc0: other derived quantities
        self._mujoco_warp.set_const(self.mjw_model, self.mjw_data)

    def update_joint_dof_properties(self):
        """Update all joint DOF properties including effort limits, friction, armature, solimplimit, solref, passive stiffness and damping, and joint limit ranges in the MuJoCo model."""
        if self.model.joint_dof_count == 0:
            return

        # Update actuator gains for JOINT_TARGET mode actuators
        if self.mjc_actuator_ctrl_source is not None and self.mjc_actuator_to_newton_idx is not None:
            nu = self.mjc_actuator_ctrl_source.shape[0]
            nworld = self.mjw_model.actuator_biasprm.shape[0]
            dofs_per_world = self.model.joint_dof_count // nworld if nworld > 0 else self.model.joint_dof_count

            wp.launch(
                update_axis_properties_kernel,
                dim=(nworld, nu),
                inputs=[
                    self.mjc_actuator_ctrl_source,
                    self.mjc_actuator_to_newton_idx,
                    self.model.joint_target_ke,
                    self.model.joint_target_kd,
                    self.model.joint_act_mode,
                    dofs_per_world,
                ],
                outputs=[
                    self.mjw_model.actuator_biasprm,
                    self.mjw_model.actuator_gainprm,
                ],
                device=self.model.device,
            )

        # Update DOF properties (armature, friction, damping, solimp, solref) - iterate over MuJoCo DOFs
        mujoco_attrs = getattr(self.model, "mujoco", None)
        joint_damping = getattr(mujoco_attrs, "dof_passive_damping", None) if mujoco_attrs is not None else None
        dof_solimp = getattr(mujoco_attrs, "solimpfriction", None) if mujoco_attrs is not None else None
        dof_solref = getattr(mujoco_attrs, "solreffriction", None) if mujoco_attrs is not None else None

        nworld = self.mjc_dof_to_newton_dof.shape[0]
        nv = self.mjc_dof_to_newton_dof.shape[1]
        wp.launch(
            update_dof_properties_kernel,
            dim=(nworld, nv),
            inputs=[
                self.mjc_dof_to_newton_dof,
                self.model.joint_armature,
                self.model.joint_friction,
                joint_damping,
                dof_solimp,
                dof_solref,
            ],
            outputs=[
                self.mjw_model.dof_armature,
                self.mjw_model.dof_frictionloss,
                self.mjw_model.dof_damping,
                self.mjw_model.dof_solimp,
                self.mjw_model.dof_solref,
            ],
            device=self.model.device,
        )

        # Update joint properties (limits, stiffness, solref, solimp) - iterate over MuJoCo joints
        solimplimit = getattr(mujoco_attrs, "solimplimit", None) if mujoco_attrs is not None else None
        joint_dof_limit_margin = getattr(mujoco_attrs, "limit_margin", None) if mujoco_attrs is not None else None
        joint_stiffness = getattr(mujoco_attrs, "dof_passive_stiffness", None) if mujoco_attrs is not None else None

        njnt = self.mjc_jnt_to_newton_dof.shape[1]
        wp.launch(
            update_jnt_properties_kernel,
            dim=(nworld, njnt),
            inputs=[
                self.mjc_jnt_to_newton_dof,
                self.model.joint_limit_ke,
                self.model.joint_limit_kd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.model.joint_effort_limit,
                solimplimit,
                joint_stiffness,
                joint_dof_limit_margin,
            ],
            outputs=[
                self.mjw_model.jnt_solimp,
                self.mjw_model.jnt_solref,
                self.mjw_model.jnt_stiffness,
                self.mjw_model.jnt_margin,
                self.mjw_model.jnt_range,
                self.mjw_model.jnt_actfrcrange,
            ],
            device=self.model.device,
        )

        # Recompute derived quantities after dof_armature changes.
        # set_const computes:
        # - dof_invweight0, body_invweight0, tendon_invweight0: inverse inertias
        # - body_subtreemass: mass of body and all descendants
        # - cam_pos0, light_pos0, actuator_acc0: other derived quantities
        self._mujoco_warp.set_const(self.mjw_model, self.mjw_data)

    def update_joint_properties(self):
        """Update joint properties including joint positions, joint axes, and relative body transforms in the MuJoCo model."""
        if self.model.joint_count == 0:
            return

        # Update mocap body transforms first (they have no MuJoCo joints)
        if self.mjc_mocap_to_newton_jnt is not None:
            nworld = self.mjc_mocap_to_newton_jnt.shape[0]
            nmocap = self.mjc_mocap_to_newton_jnt.shape[1]
            wp.launch(
                update_mocap_transforms_kernel,
                dim=(nworld, nmocap),
                inputs=[
                    self.mjc_mocap_to_newton_jnt,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                ],
                outputs=[
                    self.mjw_data.mocap_pos,
                    self.mjw_data.mocap_quat,
                ],
                device=self.model.device,
            )

        # Update joint positions, joint axes, and relative body transforms
        # Iterates over MuJoCo joints [world, jnt]
        if self.mjc_jnt_to_newton_jnt is not None and self.mjc_jnt_to_newton_jnt.shape[1] > 0:
            nworld = self.mjc_jnt_to_newton_jnt.shape[0]
            njnt = self.mjc_jnt_to_newton_jnt.shape[1]

            wp.launch(
                update_joint_transforms_kernel,
                dim=(nworld, njnt),
                inputs=[
                    self.mjc_jnt_to_newton_jnt,
                    self.mjc_jnt_to_newton_dof,
                    self.mjw_model.jnt_bodyid,
                    self.mjw_model.jnt_type,
                    # Newton model data (joint-indexed)
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                    # Newton model data (DOF-indexed)
                    self.model.joint_axis,
                ],
                outputs=[
                    self.mjw_model.jnt_pos,
                    self.mjw_model.jnt_axis,
                    self.mjw_model.body_pos,
                    self.mjw_model.body_quat,
                ],
                device=self.model.device,
            )

    def update_geom_properties(self):
        """Update geom properties including collision radius, friction, and contact parameters in the MuJoCo model."""

        # Get number of geoms and worlds from MuJoCo model
        num_geoms = self.mj_model.ngeom
        if num_geoms == 0:
            return

        world_count = self.mjc_geom_to_newton_shape.shape[0]

        # Get custom attribute for geom_solimp and geom_solmix
        mujoco_attrs = getattr(self.model, "mujoco", None)
        shape_geom_solimp = getattr(mujoco_attrs, "geom_solimp", None) if mujoco_attrs is not None else None
        shape_geom_solmix = getattr(mujoco_attrs, "geom_solmix", None) if mujoco_attrs is not None else None
        shape_geom_gap = getattr(mujoco_attrs, "geom_gap", None) if mujoco_attrs is not None else None

        wp.launch(
            update_geom_properties_kernel,
            dim=(world_count, num_geoms),
            inputs=[
                self.model.shape_material_mu,
                self.model.shape_material_ke,
                self.model.shape_material_kd,
                self.model.shape_scale,
                self.model.shape_transform,
                self.mjc_geom_to_newton_shape,
                self.mjw_model.geom_type,
                self._mujoco.mjtGeom.mjGEOM_MESH,
                self.mjw_model.geom_dataid,
                self.mjw_model.mesh_pos,
                self.mjw_model.mesh_quat,
                self.model.shape_material_mu_torsional,
                self.model.shape_material_mu_rolling,
                shape_geom_solimp,
                shape_geom_solmix,
                shape_geom_gap,
                self.model.shape_thickness,
            ],
            outputs=[
                self.mjw_model.geom_friction,
                self.mjw_model.geom_solref,
                self.mjw_model.geom_size,
                self.mjw_model.geom_pos,
                self.mjw_model.geom_quat,
                self.mjw_model.geom_solimp,
                self.mjw_model.geom_solmix,
                self.mjw_model.geom_gap,
                self.mjw_model.geom_margin,
            ],
            device=self.model.device,
        )

    def update_pair_properties(self):
        """Update MuJoCo contact pair properties from Newton custom attributes.

        Updates the randomizable pair properties (solref, solreffriction, solimp,
        margin, gap, friction) for explicit contact pairs defined in the model.
        """
        if self.use_mujoco_cpu:
            return  # CPU mode not supported for pair runtime updates

        npair = self.mj_model.npair
        if npair == 0:
            return

        # Get custom attributes for pair properties
        mujoco_attrs = getattr(self.model, "mujoco", None)
        if mujoco_attrs is None:
            return

        pair_solref = getattr(mujoco_attrs, "pair_solref", None)
        pair_solreffriction = getattr(mujoco_attrs, "pair_solreffriction", None)
        pair_solimp = getattr(mujoco_attrs, "pair_solimp", None)
        pair_margin = getattr(mujoco_attrs, "pair_margin", None)
        pair_gap = getattr(mujoco_attrs, "pair_gap", None)
        pair_friction = getattr(mujoco_attrs, "pair_friction", None)

        # Only launch kernel if at least one attribute is defined
        if any(
            attr is not None
            for attr in [pair_solref, pair_solreffriction, pair_solimp, pair_margin, pair_gap, pair_friction]
        ):
            # Compute pairs_per_world from Newton custom attributes
            pair_world_attr = getattr(mujoco_attrs, "pair_world", None)
            if pair_world_attr is not None:
                total_pairs = len(pair_world_attr)
                pairs_per_world = total_pairs // self.model.world_count
            else:
                pairs_per_world = npair

            world_count = self.mjw_data.nworld

            wp.launch(
                update_pair_properties_kernel,
                dim=(world_count, npair),
                inputs=[
                    pairs_per_world,
                    pair_solref,
                    pair_solreffriction,
                    pair_solimp,
                    pair_margin,
                    pair_gap,
                    pair_friction,
                ],
                outputs=[
                    self.mjw_model.pair_solref,
                    self.mjw_model.pair_solreffriction,
                    self.mjw_model.pair_solimp,
                    self.mjw_model.pair_margin,
                    self.mjw_model.pair_gap,
                    self.mjw_model.pair_friction,
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

    def update_eq_properties(self):
        """Update equality constraint properties in the MuJoCo model.

        Updates:

        - eq_solref/eq_solimp from MuJoCo custom attributes (if set)
        - eq_data from Newton's equality_constraint_anchor, equality_constraint_relpose,
          equality_constraint_polycoef, equality_constraint_torquescale
        - eq_active from Newton's equality_constraint_enabled

        .. note::

            Note this update only affects the equality constraints explicitly defined in Newton,
            not the equality constraints defined for joints that are excluded from articulations
            (i.e. joints that have joint_articulation == -1, for example loop-closing joints).
            Equality constraints for these joints are defined after the regular equality constraints
            in the MuJoCo model."""
        if self.model.equality_constraint_count == 0:
            return

        neq = self.mj_model.neq
        if neq == 0:
            return

        world_count = self.mjc_eq_to_newton_eq.shape[0]

        # Get custom attributes for eq_solref and eq_solimp
        mujoco_attrs = getattr(self.model, "mujoco", None)
        eq_solref = getattr(mujoco_attrs, "eq_solref", None) if mujoco_attrs is not None else None
        eq_solimp = getattr(mujoco_attrs, "eq_solimp", None) if mujoco_attrs is not None else None

        if eq_solref is not None or eq_solimp is not None:
            wp.launch(
                update_eq_properties_kernel,
                dim=(world_count, neq),
                inputs=[
                    self.mjc_eq_to_newton_eq,
                    eq_solref,
                    eq_solimp,
                ],
                outputs=[
                    self.mjw_model.eq_solref,
                    self.mjw_model.eq_solimp,
                ],
                device=self.model.device,
            )

        # Update eq_data and eq_active from Newton equality constraint properties
        wp.launch(
            update_eq_data_and_active_kernel,
            dim=(world_count, neq),
            inputs=[
                self.mjc_eq_to_newton_eq,
                self.model.equality_constraint_type,
                self.model.equality_constraint_anchor,
                self.model.equality_constraint_relpose,
                self.model.equality_constraint_polycoef,
                self.model.equality_constraint_torquescale,
                self.model.equality_constraint_enabled,
            ],
            outputs=[
                self.mjw_model.eq_data,
                self.mjw_data.eq_active,
            ],
            device=self.model.device,
        )

    def update_mimic_eq_properties(self):
        """Update mimic constraint properties in the MuJoCo model.

        Updates:

        - eq_data from Newton's constraint_mimic_coef0, constraint_mimic_coef1
        - eq_active from Newton's constraint_mimic_enabled

        Maps mimic constraints to MuJoCo mjEQ_JOINT equality constraints
        using the polycoef representation: q1 = coef0 + coef1 * q2.
        """
        if self.model.constraint_mimic_count == 0 or self.mjc_eq_to_newton_mimic is None:
            return

        neq = self.mj_model.neq
        if neq == 0:
            return

        world_count = self.mjc_eq_to_newton_mimic.shape[0]

        wp.launch(
            update_mimic_eq_data_and_active_kernel,
            dim=(world_count, neq),
            inputs=[
                self.mjc_eq_to_newton_mimic,
                self.model.constraint_mimic_coef0,
                self.model.constraint_mimic_coef1,
                self.model.constraint_mimic_enabled,
            ],
            outputs=[
                self.mjw_model.eq_data,
                self.mjw_data.eq_active,
            ],
            device=self.model.device,
        )

    def update_tendon_properties(self):
        """Update fixed tendon properties in the MuJoCo model.

        Updates tendon stiffness, damping, frictionloss, range, margin, solref, solimp,
        armature, and actfrcrange from Newton custom attributes.
        """
        if self.mjc_tendon_to_newton_tendon is None:
            return

        ntendon = self.mj_model.ntendon
        if ntendon == 0:
            return

        world_count = self.mjc_tendon_to_newton_tendon.shape[0]

        # Get custom attributes for tendons
        mujoco_attrs = getattr(self.model, "mujoco", None)
        if mujoco_attrs is None:
            return

        # Get tendon custom attributes (may be None if not defined)
        # Note: tendon_springlength is NOT updated at runtime because it has special
        # initialization semantics in MuJoCo (value -1.0 means auto-compute from initial state).
        tendon_stiffness = getattr(mujoco_attrs, "tendon_stiffness", None)
        tendon_damping = getattr(mujoco_attrs, "tendon_damping", None)
        tendon_frictionloss = getattr(mujoco_attrs, "tendon_frictionloss", None)
        tendon_range = getattr(mujoco_attrs, "tendon_range", None)
        tendon_margin = getattr(mujoco_attrs, "tendon_margin", None)
        tendon_solref_limit = getattr(mujoco_attrs, "tendon_solref_limit", None)
        tendon_solimp_limit = getattr(mujoco_attrs, "tendon_solimp_limit", None)
        tendon_solref_friction = getattr(mujoco_attrs, "tendon_solref_friction", None)
        tendon_solimp_friction = getattr(mujoco_attrs, "tendon_solimp_friction", None)
        tendon_armature = getattr(mujoco_attrs, "tendon_armature", None)
        tendon_actfrcrange = getattr(mujoco_attrs, "tendon_actuator_force_range", None)

        wp.launch(
            update_tendon_properties_kernel,
            dim=(world_count, ntendon),
            inputs=[
                self.mjc_tendon_to_newton_tendon,
                tendon_stiffness,
                tendon_damping,
                tendon_frictionloss,
                tendon_range,
                tendon_margin,
                tendon_solref_limit,
                tendon_solimp_limit,
                tendon_solref_friction,
                tendon_solimp_friction,
                tendon_armature,
                tendon_actfrcrange,
            ],
            outputs=[
                self.mjw_model.tendon_stiffness,
                self.mjw_model.tendon_damping,
                self.mjw_model.tendon_frictionloss,
                self.mjw_model.tendon_range,
                self.mjw_model.tendon_margin,
                self.mjw_model.tendon_solref_lim,
                self.mjw_model.tendon_solimp_lim,
                self.mjw_model.tendon_solref_fri,
                self.mjw_model.tendon_solimp_fri,
                self.mjw_model.tendon_armature,
                self.mjw_model.tendon_actfrcrange,
            ],
            device=self.model.device,
        )

    def update_actuator_properties(self):
        """Update CTRL_DIRECT actuator properties (gainprm, biasprm) in the MuJoCo model.

        Only updates actuators that use CTRL_DIRECT mode. JOINT_TARGET actuators are
        updated via update_joint_dof_properties() using joint_target_ke/kd.
        """
        if self.mjc_actuator_ctrl_source is None or self.mjc_actuator_to_newton_idx is None:
            return

        nu = self.mjc_actuator_ctrl_source.shape[0]
        if nu == 0:
            return

        mujoco_attrs = getattr(self.model, "mujoco", None)
        if mujoco_attrs is None:
            return

        actuator_gainprm = getattr(mujoco_attrs, "actuator_gainprm", None)
        actuator_biasprm = getattr(mujoco_attrs, "actuator_biasprm", None)
        if actuator_gainprm is None or actuator_biasprm is None:
            return

        nworld = self.mjw_model.actuator_biasprm.shape[0]
        actuators_per_world = actuator_gainprm.shape[0] // nworld if nworld > 0 else actuator_gainprm.shape[0]

        wp.launch(
            update_ctrl_direct_actuator_properties_kernel,
            dim=(nworld, nu),
            inputs=[
                self.mjc_actuator_ctrl_source,
                self.mjc_actuator_to_newton_idx,
                actuator_gainprm,
                actuator_biasprm,
                actuators_per_world,
            ],
            outputs=[
                self.mjw_model.actuator_gainprm,
                self.mjw_model.actuator_biasprm,
            ],
            device=self.model.device,
        )

    def _validate_model_for_separate_worlds(self, model: Model) -> None:
        """Validate that the Newton model is compatible with MuJoCo's separate_worlds mode.

        MuJoCo's separate_worlds mode creates identical copies of a single MuJoCo model
        for each Newton world. This requires:
        1. All worlds have the same number of bodies, joints, shapes, and equality constraints
        2. Entity types match across corresponding entities in each world
        3. Global world (-1) only contains static shapes (no bodies, joints, or constraints)

        Args:
            model: The Newton model to validate.

        Raises:
            ValueError: If the model is not compatible with separate_worlds mode.
        """
        world_count = model.world_count

        # Check that we have at least one world
        if world_count == 0:
            raise ValueError(
                "SolverMuJoCo with separate_worlds=True requires at least one world (world_count >= 1). "
                "Found world_count=0 (all entities in global world -1)."
            )

        body_world = model.body_world.numpy()
        joint_world = model.joint_world.numpy()
        shape_world = model.shape_world.numpy()
        eq_constraint_world = model.equality_constraint_world.numpy()

        # --- Check global world restrictions (always, regardless of world_count) ---
        # No bodies in global world
        global_bodies = np.where(body_world == -1)[0]
        if len(global_bodies) > 0:
            body_names = [model.body_key[i] for i in global_bodies[:3]]
            msg = f"Global world (-1) cannot contain bodies. Found {len(global_bodies)} body(ies) with world == -1"
            if body_names:
                msg += f": {body_names}"
            raise ValueError(msg)

        # No joints in global world
        global_joints = np.where(joint_world == -1)[0]
        if len(global_joints) > 0:
            joint_names = [model.joint_key[i] for i in global_joints[:3]]
            msg = f"Global world (-1) cannot contain joints. Found {len(global_joints)} joint(s) with world == -1"
            if joint_names:
                msg += f": {joint_names}"
            raise ValueError(msg)

        # No equality constraints in global world
        global_constraints = np.where(eq_constraint_world == -1)[0]
        if len(global_constraints) > 0:
            msg = f"Global world (-1) cannot contain equality constraints. Found {len(global_constraints)} constraint(s) with world == -1"
            raise ValueError(msg)

        # No mimic constraints in global world
        mimic_world = model.constraint_mimic_world.numpy()
        global_mimic = np.where(mimic_world == -1)[0]
        if len(global_mimic) > 0:
            msg = f"Global world (-1) cannot contain mimic constraints. Found {len(global_mimic)} constraint(s) with world == -1"
            raise ValueError(msg)

        # Skip homogeneity checks for single-world models
        if world_count <= 1:
            return

        # --- Check entity count homogeneity ---
        # Count entities per world (excluding global shapes)
        non_global_shapes = shape_world[shape_world >= 0]

        for entity_name, world_arr in [
            ("bodies", body_world),
            ("joints", joint_world),
            ("shapes", non_global_shapes),
            ("equality constraints", eq_constraint_world),
            ("mimic constraints", mimic_world),
        ]:
            # Use bincount for O(n) counting instead of O(n * world_count) loop
            if len(world_arr) == 0:
                continue
            counts = np.bincount(world_arr.astype(np.int64), minlength=world_count)
            # Vectorized check: all counts must equal the first
            if not np.all(counts == counts[0]):
                # Find first mismatch for error message (only on failure path)
                expected = counts[0]
                mismatched = np.where(counts != expected)[0]
                w = mismatched[0]
                raise ValueError(
                    f"SolverMuJoCo requires homogeneous worlds. "
                    f"World 0 has {expected} {entity_name}, but world {w} has {counts[w]}."
                )

        # --- Check type matching across worlds (vectorized) ---
        # Load type arrays lazily - only when needed for validation
        joints_per_world = model.joint_count // world_count
        if joints_per_world > 0:
            joint_type = model.joint_type.numpy()
            joint_types_2d = joint_type.reshape(world_count, joints_per_world)
            # Vectorized mismatch check: compare all rows to first row
            mismatches = joint_types_2d != joint_types_2d[0]
            if np.any(mismatches):
                # Find first mismatch position using vectorized operations
                j = np.argmax(np.any(mismatches, axis=0))
                types = joint_types_2d[:, j]
                raise ValueError(
                    f"SolverMuJoCo requires homogeneous worlds. "
                    f"Joint types mismatch at position {j}: world 0 has type {types[0]}, "
                    f"but other worlds have types {types[1:].tolist()}."
                )

        # Only check non-global shapes
        shapes_per_world = len(non_global_shapes) // world_count if world_count > 0 else 0
        if shapes_per_world > 0:
            shape_type = model.shape_type.numpy()
            # Get shape types for non-global shapes only
            non_global_shape_types = shape_type[shape_world >= 0]
            shape_types_2d = non_global_shape_types.reshape(world_count, shapes_per_world)
            # Vectorized mismatch check
            mismatches = shape_types_2d != shape_types_2d[0]
            if np.any(mismatches):
                s = np.argmax(np.any(mismatches, axis=0))
                types = shape_types_2d[:, s]
                raise ValueError(
                    f"SolverMuJoCo requires homogeneous worlds. "
                    f"Shape types mismatch at position {s}: world 0 has type {types[0]}, "
                    f"but other worlds have types {types[1:].tolist()}."
                )

        constraints_per_world = model.equality_constraint_count // world_count if world_count > 0 else 0
        if constraints_per_world > 0:
            eq_constraint_type = model.equality_constraint_type.numpy()
            constraint_types_2d = eq_constraint_type.reshape(world_count, constraints_per_world)
            # Vectorized mismatch check
            mismatches = constraint_types_2d != constraint_types_2d[0]
            if np.any(mismatches):
                c = np.argmax(np.any(mismatches, axis=0))
                types = constraint_types_2d[:, c]
                raise ValueError(
                    f"SolverMuJoCo requires homogeneous worlds. "
                    f"Equality constraint types mismatch at position {c}: world 0 has type {types[0]}, "
                    f"but other worlds have types {types[1:].tolist()}."
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
            import mujoco
            import mujoco.viewer

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
