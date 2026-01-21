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

from ...core.types import nparray, override, vec5
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
    _create_inverse_shape_mapping_kernel,
    apply_mjc_body_f_kernel,
    apply_mjc_control_kernel,
    apply_mjc_qfrc_kernel,
    convert_body_xforms_to_warp_kernel,
    convert_mj_coords_to_warp_kernel,
    convert_mjw_contact_to_warp_kernel,
    convert_newton_contacts_to_mjwarp_kernel,
    convert_rigid_forces_from_mj_kernel,
    convert_solref,
    convert_warp_coords_to_mj_kernel,
    eval_articulation_fk,
    repeat_array_kernel,
    update_axis_properties_kernel,
    update_body_inertia_kernel,
    update_body_mass_ipos_kernel,
    update_dof_properties_kernel,
    update_eq_properties_kernel,
    update_geom_properties_kernel,
    update_jnt_properties_kernel,
    update_joint_transforms_kernel,
    update_mocap_transforms_kernel,
    update_model_properties_kernel,
    update_shape_mappings_kernel,
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
                with warnings.catch_warnings():
                    # Set a filter to make all ImportWarnings "always" appear
                    # This is useful to debug import errors on Windows, for example
                    warnings.simplefilter("always", category=ImportWarning)

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
                name="geom_priority",
                frequency=ModelAttributeFrequency.SHAPE,
                assignment=ModelAttributeAssignment.MODEL,
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
                frequency=ModelAttributeFrequency.SHAPE,
                assignment=ModelAttributeAssignment.MODEL,
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
                frequency=ModelAttributeFrequency.SHAPE,
                assignment=ModelAttributeAssignment.MODEL,
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
                frequency=ModelAttributeFrequency.SHAPE,
                assignment=ModelAttributeAssignment.MODEL,
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
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimplimit",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solreffriction",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.vec2,
                default=wp.vec2(0.02, 1.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solreffriction",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="solimpfriction",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
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
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_springref",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:springref",
                mjcf_attribute_name="springref",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="dof_ref",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mujoco",
                usd_attribute_name="mjc:ref",
                mjcf_attribute_name="ref",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="jnt_actgravcomp",
                frequency=ModelAttributeFrequency.JOINT_DOF,
                assignment=ModelAttributeAssignment.MODEL,
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
                frequency=ModelAttributeFrequency.EQUALITY_CONSTRAINT,
                assignment=ModelAttributeAssignment.MODEL,
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
                frequency=ModelAttributeFrequency.EQUALITY_CONSTRAINT,
                assignment=ModelAttributeAssignment.MODEL,
                dtype=vec5,
                default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
                namespace="mujoco",
                usd_attribute_name="mjc:solimp",
                mjcf_attribute_name="solimp",
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

        def parse_limited(value: str) -> int:
            """Parse MuJoCo limited attribute: false=0, true=1, auto=2."""
            v = value.lower().strip()
            if v in ("false", "0"):
                return 0
            if v in ("true", "1"):
                return 1
            if v in ("auto", "2"):
                return 2
            return int(value)

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
        # We will not test that  the number of joint indices of all tendons must be equal to the number of
        # coefficients of all tendons.
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

    def _init_tendons(self, model: Model, spec, joint_mapping: dict[int, str], template_world: int) -> list[int]:
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
            list[int]: List of Newton tendon indices that were added to MuJoCo (in order).
        """

        # Count the number of tendons (tendon_count)
        # Count the length of the arrays that contains the joint indices of all tendons (joint_entry_count)
        tendon_count, joint_entry_count = self._validate_tendon_attributes(model)
        if tendon_count == 0:
            return []

        mujoco_attrs = model.mujoco

        # Get tendon arrays
        tendon_world = mujoco_attrs.tendon_world.numpy()
        tendon_stiffness = getattr(mujoco_attrs, "tendon_stiffness", None)
        tendon_damping = getattr(mujoco_attrs, "tendon_damping", None)
        tendon_frictionloss = getattr(mujoco_attrs, "tendon_frictionloss", None)
        tendon_limited = getattr(mujoco_attrs, "tendon_limited", None)
        tendon_range = getattr(mujoco_attrs, "tendon_range", None)
        tendon_actuator_force_limited = getattr(mujoco_attrs, "tendon_actuator_force_limited", None)
        tendon_actuator_force_range = getattr(mujoco_attrs, "tendon_actuator_force_range", None)
        tendon_margin = getattr(mujoco_attrs, "tendon_margin", None)
        tendon_solref_limit = getattr(mujoco_attrs, "tendon_solref_limit", None)
        tendon_solimp_limit = getattr(mujoco_attrs, "tendon_solimp_limit", None)
        tendon_solref_friction = getattr(mujoco_attrs, "tendon_solref_friction", None)
        tendon_solimp_friction = getattr(mujoco_attrs, "tendon_solimp_friction", None)
        tendon_armature = getattr(mujoco_attrs, "tendon_armature", None)
        tendon_springlength = getattr(mujoco_attrs, "tendon_springlength", None)
        tendon_joint_adr = mujoco_attrs.tendon_joint_adr.numpy()
        tendon_joint_num = mujoco_attrs.tendon_joint_num.numpy()

        # Get joint arrays (for the linear combination)
        tendon_joint = mujoco_attrs.tendon_joint.numpy() if joint_entry_count > 0 else None
        tendon_coef = mujoco_attrs.tendon_coef.numpy() if joint_entry_count > 0 else None

        # Track which Newton tendon indices are added to MuJoCo
        selected_tendons: list[int] = []

        for i in range(tendon_count):
            # Only include tendons from the template world or global tendons (world < 0)
            tw = int(tendon_world[i])
            if tw != template_world and tw >= 0:
                continue

            # Track this tendon
            selected_tendons.append(i)

            # Create tendon
            t = spec.add_tendon()

            # Set tendon properties
            if tendon_stiffness is not None:
                t.stiffness = float(tendon_stiffness.numpy()[i])
            if tendon_damping is not None:
                t.damping = float(tendon_damping.numpy()[i])
            if tendon_frictionloss is not None:
                t.frictionloss = float(tendon_frictionloss.numpy()[i])
            if tendon_limited is not None:
                t.limited = int(tendon_limited.numpy()[i])
            if tendon_range is not None:
                t.range = tendon_range.numpy()[i].tolist()
            if tendon_actuator_force_limited is not None:
                t.actfrclimited = int(tendon_actuator_force_limited.numpy()[i])
            if tendon_actuator_force_range is not None:
                t.actfrcrange = tendon_actuator_force_range.numpy()[i].tolist()
            if tendon_margin is not None:
                t.margin = float(tendon_margin.numpy()[i])
            if tendon_armature is not None:
                t.armature = float(tendon_armature.numpy()[i])
            if tendon_solref_limit is not None:
                t.solref_limit = tendon_solref_limit.numpy()[i].tolist()
            if tendon_solimp_limit is not None:
                t.solimp_limit = tendon_solimp_limit.numpy()[i].tolist()
            if tendon_solref_friction is not None:
                t.solref_friction = tendon_solref_friction.numpy()[i].tolist()
            if tendon_solimp_friction is not None:
                t.solimp_friction = tendon_solimp_friction.numpy()[i].tolist()
            if tendon_springlength is not None:
                val = tendon_springlength.numpy()[i]
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

                joint_name = joint_mapping.get(newton_joint)
                if joint_name is None:
                    warnings.warn(
                        f"Skipping joint entry {j} for tendon {i}: Newton joint {newton_joint} "
                        f"not found in MuJoCo joint mapping.",
                        stacklevel=2,
                    )
                    continue

                t.wrap_joint(joint_name, coef)

        return selected_tendons

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
        self.mjc_actuator_to_newton_axis: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Maps MuJoCo[world, actuator] -> Newton axis (DOF) index with actuator type encoded in sign.

        Encoding:
        - Positive value: position actuator, newton_axis = value
        - Negative value (not -1): velocity actuator, newton_axis = -(value + 2)
        - Value of -1: unmapped actuator

        Shape [nworld, nu], dtype int32."""
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
        self.mjc_tendon_to_newton_tendon: wp.array(dtype=wp.int32, ndim=2) | None = None
        """Mapping from MuJoCo [world, tendon] to Newton tendon index.

        Shape [nworld, ntendon], dtype int32."""

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
                separate_worlds = not use_mujoco_cpu and model.num_worlds > 1
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
            if wp.config.verbose:
                print("Setting model.sensor_rne_postconstraint True")
            m.sensor_rne_postconstraint = True
            # required for cfrc_ext, cfrc_int, cacc

    def convert_contacts_to_mjwarp(self, model: Model, state_in: State, contacts: Contacts):
        # Ensure the inverse shape mapping exists (lazy creation)
        if self.newton_shape_to_mjc_geom is None:
            self._create_inverse_shape_mapping()

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
                contacts.rigid_contact_stiffness,
                contacts.rigid_contact_damping,
                contacts.rigid_contact_friction,
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
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self.update_model_properties()
        if flags & SolverNotifyFlags.EQUALITY_CONSTRAINT_PROPERTIES:
            self.update_eq_properties()
        if flags & SolverNotifyFlags.FIXED_TENDON_PROPERTIES:
            self.update_tendon_properties()

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
            _create_inverse_shape_mapping_kernel,
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
        bodies_per_world = model.body_count // nworld
        if control is not None:
            # Launch over MuJoCo actuators
            nu = self.mjc_actuator_to_newton_axis.shape[1]
            wp.launch(
                apply_mjc_control_kernel,
                dim=(nworld, nu),
                inputs=[
                    self.mjc_actuator_to_newton_axis,
                    control.joint_target_pos,
                    control.joint_target_vel,
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
                    # self.mjw_data.cfrc_int,
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
                self.mjc_geom_to_newton_shape,
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
        target_filename: str | None = None,
        default_actuator_args: dict | None = None,
        default_actuator_gear: float | None = None,
        actuator_gears: dict[str, float] | None = None,
        actuated_axes: list[int] | None = None,
        skip_visual_only_geoms: bool = True,
        include_sites: bool = True,
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

        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO

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
        joint_target_kd = model.joint_target_kd.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_armature = model.joint_armature.numpy()
        joint_effort_limit = model.joint_effort_limit.numpy()
        # MoJoCo doesn't have velocity limit
        # joint_velocity_limit = model.joint_velocity_limit.numpy()
        joint_friction = model.joint_friction.numpy()
        joint_world = model.joint_world.numpy()
        joint_q = model.joint_q.numpy()
        joint_q_start = model.joint_q_start.numpy()
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
        else:
            # if we are not separating environments to worlds, we use all shapes, bodies, joints
            first_world = 0

            # if we are not separating worlds, we use all shapes, bodies, joints, constraints
            selected_shapes = np.arange(model.shape_count, dtype=np.int32)
            selected_bodies = np.arange(model.body_count, dtype=np.int32)
            selected_joints = np.arange(model.joint_count, dtype=np.int32)
            selected_constraints = np.arange(model.equality_constraint_count, dtype=np.int32)

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
                # For ball joints, all 3 DOFs map to the same MuJoCo joint
                for i in range(3):
                    dof_to_mjc_joint[qd_start + i] = num_mjc_joints
                num_dofs += 3
                num_mjc_joints += 1
                # Add actuators for the ball joint
                for i in range(3):
                    ai = qd_start + i

                    if actuated_axes is None or ai in actuated_axes:
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
                        args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                        spec.add_actuator(target=name, **args)
                        axis_to_actuator[ai, 0] = actuator_count
                        actuator_count += 1

                        args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                        spec.add_actuator(target=name, **args)
                        axis_to_actuator[ai, 1] = actuator_count
                        actuator_count += 1
            elif j_type in supported_joint_types:
                lin_axis_count, ang_axis_count = joint_dof_dim[j]
                num_dofs += lin_axis_count + ang_axis_count

                # linear dofs
                for i in range(lin_axis_count):
                    ai = qd_start + i

                    axis = wp.quat_rotate(joint_rot, wp.vec3(*joint_axis[ai]))

                    # Get initial joint position for ref parameter
                    q_start = joint_q_start[j]
                    ref_value = joint_q[q_start + i]

                    joint_params = {
                        "armature": joint_armature[qd_start + i],
                        "pos": joint_pos,
                        # "ref": float(ref_value),
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
                    # Map this DOF to the current MuJoCo joint index
                    dof_to_mjc_joint[ai] = num_mjc_joints
                    num_mjc_joints += 1

                    if actuated_axes is None or ai in actuated_axes:
                        kp = joint_target_ke[ai]
                        kd = joint_target_kd[ai]
                        gear = actuator_gears.get(axname)
                        if gear is not None:
                            args = {}
                            args.update(actuator_args)
                            args["gear"] = [gear, 0.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            args = actuator_args
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

                    # Get initial joint position for ref parameter (convert radians to degrees for hinge)
                    q_start = joint_q_start[j]
                    ref_value = np.rad2deg(joint_q[q_start + i])

                    joint_params = {
                        "armature": joint_armature[qd_start + i],
                        "pos": joint_pos,
                        # "ref": float(ref_value),
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
                        gear = actuator_gears.get(axname)
                        if gear is not None:
                            args = {}
                            args.update(actuator_args)
                            args["gear"] = [gear, 0.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            args = actuator_args
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

        # add fixed tendons from custom attributes
        selected_tendons = self._init_tendons(model, spec, joint_mapping, first_world)

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
            nworld = model.num_worlds if separate_worlds else 1

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
            bodies_per_world = model.body_count // model.num_worlds
            mjc_body_to_newton_np = np.full((nworld, nbody), -1, dtype=np.int32)
            for newton_body, mjc_body in body_mapping.items():
                if newton_body >= 0:  # Skip world body (-1 -> 0)
                    newton_body_in_world = newton_body % bodies_per_world
                    for w in range(nworld):
                        mjc_body_to_newton_np[w, mjc_body] = w * bodies_per_world + newton_body_in_world
            self.mjc_body_to_newton = wp.array(mjc_body_to_newton_np, dtype=wp.int32)

            # Common variables for mapping creation
            njnt = self.mj_model.njnt
            joints_per_world = model.joint_count // model.num_worlds
            dofs_per_world = model.joint_dof_count // model.num_worlds

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

            # Create mjc_actuator_to_newton_axis: MuJoCo[world, actuator] -> Newton axis
            # axis_to_actuator[newton_axis, 0/1] -> mjc_actuator (pos/vel)
            nu = self.mj_model.nu  # Number of actuators
            mjc_actuator_to_newton_axis_np = np.full((nworld, nu), -1, dtype=np.int32)
            for newton_axis in range(axis_to_actuator.shape[0]):
                template_axis = newton_axis % dofs_per_world
                for act_type in range(2):  # 0=pos, 1=vel
                    mjc_act = axis_to_actuator[newton_axis, act_type]
                    if mjc_act >= 0:
                        for w in range(nworld):
                            world_newton_axis = w * dofs_per_world + template_axis
                            if act_type == 0:
                                # Position actuator: store as positive value
                                mjc_actuator_to_newton_axis_np[w, mjc_act] = world_newton_axis
                            else:
                                # Velocity actuator: encode as -(newton_axis + 2)
                                # Using +2 offset so that newton_axis=0 encodes as -2, leaving -1 for "unmapped"
                                mjc_actuator_to_newton_axis_np[w, mjc_act] = -(world_newton_axis + 2)
            self.mjc_actuator_to_newton_axis = wp.array(mjc_actuator_to_newton_axis_np, dtype=wp.int32)

            # Create mjc_eq_to_newton_eq: MuJoCo[world, eq] -> Newton equality constraint
            # selected_constraints[idx] is the Newton template constraint index
            neq = self.mj_model.neq
            eq_constraints_per_world = model.equality_constraint_count // model.num_worlds
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

            # Create mjc_tendon_to_newton_tendon: MuJoCo[world, tendon] -> Newton tendon
            # selected_tendons[idx] is the Newton template tendon index
            ntendon = self.mj_model.ntendon
            if ntendon > 0:
                # Get tendon count per world from custom attributes
                mujoco_attrs = getattr(model, "mujoco", None)
                tendon_world = getattr(mujoco_attrs, "tendon_world", None) if mujoco_attrs else None
                if tendon_world is not None:
                    total_tendons = len(tendon_world)
                    tendons_per_world = total_tendons // model.num_worlds if model.num_worlds > 0 else total_tendons
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
                nworld = model.num_worlds
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
            "jnt_actfrcrange",  # joint-level actuator force range (effort limit)
            "jnt_margin",  # corresponds to newton custom attribute "limit_margin"
            "dof_armature",
            "dof_damping",
            # "dof_invweight0",
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
            # "geom_margin",
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
            # "eq_data",
            # "actuator_dynprm",
            "actuator_gainprm",
            "actuator_biasprm",
            # "actuator_ctrlrange",
            # "actuator_forcerange",  # No longer used - force clamping via jnt_actfrcrange
            # "actuator_actrange",
            # "actuator_gear",
            # "pair_solref",
            # "pair_solreffriction",
            # "pair_solimp",
            # "pair_margin",
            # "pair_gap",
            # "pair_friction",
            "tendon_world",
            # "tendon_adr",                 # Topological, not replicated
            # "tendon_num",                 # Topological, not replicated
            # "tendon_limited",             # Topological, not replicated
            # "tendon_actfrclimited",       # Topological, not replicated
            # "tendon_jnt_adr",             # Topological, not replicated
            # "tendon_limited_adr",         # Topological, not replicated
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
            # "tendon_length0",            # Autocomputed and auto-replicated by mujoco
            # "tendon_invweight0",         # Autocomputed and auto-replicated by mujoco
            # "mat_rgba",
        }

        # Fields in mj_model.opt to expand
        opt_fields_to_expand = {
            "gravity",
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

    def update_joint_dof_properties(self):
        """Update all joint DOF properties including effort limits, friction, armature, solimplimit, solref, passive stiffness and damping, and joint limit ranges in the MuJoCo model."""
        if self.model.joint_dof_count == 0:
            return

        # Update actuator force ranges (effort limits) if actuators exist
        if self.mjc_actuator_to_newton_axis is not None:
            nworld = self.mjc_actuator_to_newton_axis.shape[0]
            nu = self.mjc_actuator_to_newton_axis.shape[1]
            wp.launch(
                update_axis_properties_kernel,
                dim=(nworld, nu),
                inputs=[
                    self.mjc_actuator_to_newton_axis,
                    self.model.joint_target_ke,
                    self.model.joint_target_kd,
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

        num_worlds = self.mjc_geom_to_newton_shape.shape[0]

        # Get custom attribute for geom_solimp and geom_solmix
        mujoco_attrs = getattr(self.model, "mujoco", None)
        shape_geom_solimp = getattr(mujoco_attrs, "geom_solimp", None) if mujoco_attrs is not None else None
        shape_geom_solmix = getattr(mujoco_attrs, "geom_solmix", None) if mujoco_attrs is not None else None
        shape_geom_gap = getattr(mujoco_attrs, "geom_gap", None) if mujoco_attrs is not None else None

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
                self.mjc_geom_to_newton_shape,
                self.mjw_model.geom_type,
                self._mujoco.mjtGeom.mjGEOM_MESH,
                self.mjw_model.geom_dataid,
                self.mjw_model.mesh_pos,
                self.mjw_model.mesh_quat,
                self.model.shape_material_torsional_friction,
                self.model.shape_material_rolling_friction,
                shape_geom_solimp,
                shape_geom_solmix,
                shape_geom_gap,
            ],
            outputs=[
                self.mjw_model.geom_rbound,
                self.mjw_model.geom_friction,
                self.mjw_model.geom_solref,
                self.mjw_model.geom_size,
                self.mjw_model.geom_pos,
                self.mjw_model.geom_quat,
                self.mjw_model.geom_solimp,
                self.mjw_model.geom_solmix,
                self.mjw_model.geom_gap,
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
        """Update equality constraint properties including solref in the MuJoCo model.

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

        num_worlds = self.mjc_eq_to_newton_eq.shape[0]

        # Get custom attributes for eq_solref and eq_solimp
        mujoco_attrs = getattr(self.model, "mujoco", None)
        eq_solref = getattr(mujoco_attrs, "eq_solref", None) if mujoco_attrs is not None else None
        eq_solimp = getattr(mujoco_attrs, "eq_solimp", None) if mujoco_attrs is not None else None

        if eq_solref is not None or eq_solimp is not None:
            wp.launch(
                update_eq_properties_kernel,
                dim=(num_worlds, neq),
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

    def update_tendon_properties(self):
        """Update fixed tendon properties in the MuJoCo model.

        Updates tendon stiffness, damping, frictionloss, range, margin, solref, solimp,
        springlength, armature, and actfrcrange from Newton custom attributes.
        """
        if self.mjc_tendon_to_newton_tendon is None:
            return

        ntendon = self.mj_model.ntendon
        if ntendon == 0:
            return

        num_worlds = self.mjc_tendon_to_newton_tendon.shape[0]

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
            dim=(num_worlds, ntendon),
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
        num_worlds = model.num_worlds

        # Check that we have at least one world
        if num_worlds == 0:
            raise ValueError(
                "SolverMuJoCo with separate_worlds=True requires at least one world (num_worlds >= 1). "
                "Found num_worlds=0 (all entities in global world -1)."
            )

        body_world = model.body_world.numpy()
        joint_world = model.joint_world.numpy()
        shape_world = model.shape_world.numpy()
        eq_constraint_world = model.equality_constraint_world.numpy()

        # --- Check global world restrictions (always, regardless of num_worlds) ---
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

        # Skip homogeneity checks for single-world models
        if num_worlds <= 1:
            return

        # --- Check entity count homogeneity ---
        # Count entities per world (excluding global shapes)
        non_global_shapes = shape_world[shape_world >= 0]

        for entity_name, world_arr in [
            ("bodies", body_world),
            ("joints", joint_world),
            ("shapes", non_global_shapes),
            ("equality constraints", eq_constraint_world),
        ]:
            # Count per world and check all worlds have same count as world 0
            counts = [np.sum(world_arr == w) for w in range(num_worlds)]
            expected = counts[0]
            for w in range(1, num_worlds):
                if counts[w] != expected:
                    raise ValueError(
                        f"SolverMuJoCo requires homogeneous worlds. "
                        f"World 0 has {expected} {entity_name}, but world {w} has {counts[w]}."
                    )

        # --- Check type matching across worlds (vectorized) ---
        # For entities that must have matching types across worlds
        joint_type = model.joint_type.numpy()
        shape_type = model.shape_type.numpy()
        eq_constraint_type = model.equality_constraint_type.numpy()

        joints_per_world = model.joint_count // num_worlds
        if joints_per_world > 0:
            joint_types_2d = joint_type.reshape(num_worlds, joints_per_world)
            if not np.all(joint_types_2d == joint_types_2d[0]):
                # Find first mismatch for error message
                for j in range(joints_per_world):
                    types = joint_types_2d[:, j]
                    if not np.all(types == types[0]):
                        raise ValueError(
                            f"SolverMuJoCo requires homogeneous worlds. "
                            f"Joint types mismatch at position {j}: world 0 has type {types[0]}, "
                            f"but other worlds have types {types[1:].tolist()}."
                        )

        # Only check non-global shapes
        shapes_per_world = len(non_global_shapes) // num_worlds if num_worlds > 0 else 0
        if shapes_per_world > 0:
            # Get shape types for non-global shapes only
            non_global_shape_types = shape_type[shape_world >= 0]
            shape_types_2d = non_global_shape_types.reshape(num_worlds, shapes_per_world)
            if not np.all(shape_types_2d == shape_types_2d[0]):
                # Find first mismatch for error message
                for s in range(shapes_per_world):
                    types = shape_types_2d[:, s]
                    if not np.all(types == types[0]):
                        raise ValueError(
                            f"SolverMuJoCo requires homogeneous worlds. "
                            f"Shape types mismatch at position {s}: world 0 has type {types[0]}, "
                            f"but other worlds have types {types[1:].tolist()}."
                        )

        constraints_per_world = model.equality_constraint_count // num_worlds if num_worlds > 0 else 0
        if constraints_per_world > 0:
            constraint_types_2d = eq_constraint_type.reshape(num_worlds, constraints_per_world)
            if not np.all(constraint_types_2d == constraint_types_2d[0]):
                # Find first mismatch for error message
                for c in range(constraints_per_world):
                    types = constraint_types_2d[:, c]
                    if not np.all(types == types[0]):
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
