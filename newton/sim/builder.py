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

"""A module for building Newton models."""

from __future__ import annotations

import copy
import ctypes
import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from newton.core.spatial import quat_between_axes
from newton.core.types import (
    Axis,
    AxisType,
    Devicelike,
    Mat33,
    Quat,
    Sequence,
    Transform,
    Vec3,
    Vec4,
    axis_to_vec3,
    flag_to_int,
    nparray,
)
from newton.geometry import (
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CONE,
    GEO_CYLINDER,
    GEO_MESH,
    GEO_NONE,
    GEO_PLANE,
    GEO_SDF,
    GEO_SPHERE,
    PARTICLE_FLAG_ACTIVE,
    SDF,
    SHAPE_FLAG_COLLIDE_PARTICLES,
    SHAPE_FLAG_COLLIDE_SHAPES,
    SHAPE_FLAG_VISIBLE,
    Mesh,
    compute_shape_inertia,
    compute_shape_radius,
    transform_inertia,
)

from .graph_coloring import ColoringAlgorithm, color_trimesh, combine_independent_particle_coloring
from .joints import (
    JOINT_BALL,
    JOINT_D6,
    JOINT_DISTANCE,
    JOINT_FIXED,
    JOINT_FREE,
    JOINT_MODE_NONE,
    JOINT_MODE_TARGET_POSITION,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    get_joint_dof_count,
)
from .model import Model


class ModelBuilder:
    """A helper class for building simulation models at runtime.

    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
    and builds the scene representation using standard Python data structures (lists),
    this means it is not differentiable. Once :meth:`finalize`
    has been called the ModelBuilder transfers all data to Warp tensors and returns
    an object that may be used for simulation.

    Example
    -------

    .. testcode::

        import newton
        from newton.solvers import XPBDSolver

        builder = newton.ModelBuilder()

        # anchor point (zero mass)
        builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

        # build chain
        for i in range(1, 10):
            builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
            builder.add_spring(i - 1, i, 1.0e3, 0.0, 0)

        # create model
        model = builder.finalize()

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        solver = XPBDSolver(model)

        for i in range(10):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, control, contacts, dt=1.0 / 60.0)
            state_0, state_1 = state_1, state_0

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if
        desired.
    """

    @dataclass
    class ShapeConfig:
        """
        Represents the properties of a collision shape used in simulation.
        """

        density: float = 1000.0
        """The density of the shape material."""
        ke: float = 1.0e5
        """The contact elastic stiffness."""
        kd: float = 1000.0
        """The contact damping stiffness."""
        kf: float = 1000.0
        """The contact friction stiffness."""
        ka: float = 0.0
        """The contact adhesion distance."""
        mu: float = 0.5
        """The coefficient of friction."""
        restitution: float = 0.0
        """The coefficient of restitution."""
        thickness: float = 1e-5
        """The thickness of the shape."""
        is_solid: bool = True
        """Indicates whether the shape is solid or hollow. Defaults to True."""
        collision_group: int = -1
        """The collision group ID for the shape. Defaults to -1."""
        collision_filter_parent: bool = True
        """Whether to inherit collision filtering from the parent. Defaults to True."""
        has_shape_collision: bool = True
        """Whether the shape can collide with other shapes. Defaults to True."""
        has_particle_collision: bool = True
        """Whether the shape can collide with particles. Defaults to True."""
        is_visible: bool = True
        """Indicates whether the shape is visible in the simulation. Defaults to True."""

        @property
        def flags(self) -> int:
            """Returns the flags for the shape."""

            shape_flags = int(SHAPE_FLAG_VISIBLE) if self.is_visible else 0
            shape_flags |= int(SHAPE_FLAG_COLLIDE_SHAPES) if self.has_shape_collision else 0
            shape_flags |= int(SHAPE_FLAG_COLLIDE_PARTICLES) if self.has_particle_collision else 0
            return shape_flags

        @flags.setter
        def flags(self, value: int):
            """Sets the flags for the shape."""

            self.is_visible = bool(value & SHAPE_FLAG_VISIBLE)
            self.has_shape_collision = bool(value & SHAPE_FLAG_COLLIDE_SHAPES)
            self.has_particle_collision = bool(value & SHAPE_FLAG_COLLIDE_PARTICLES)

        def copy(self) -> ShapeConfig:
            return copy.copy(self)

    class JointDofConfig:
        """
        Describes a joint axis (a single degree of freedom) that can have limits and be driven towards a target.
        """

        def __init__(
            self,
            axis: AxisType | Vec3 = Axis.X,
            limit_lower: float = -1e6,
            limit_upper: float = 1e6,
            limit_ke: float = 1e4,
            limit_kd: float = 1e1,
            target: float = 0.0,
            target_ke: float = 0.0,
            target_kd: float = 0.0,
            mode: int = JOINT_MODE_TARGET_POSITION,
            armature: float = 1e-2,
            effort_limit: float = 1e6,
            velocity_limit: float = 1e6,
            friction: float = 0.0,
        ):
            self.axis = wp.normalize(axis_to_vec3(axis))
            """The 3D axis that this JointDofConfig object describes."""
            self.limit_lower = limit_lower
            """The lower position limit of the joint axis. Defaults to -1e6."""
            self.limit_upper = limit_upper
            """The upper position limit of the joint axis. Defaults to 1e6."""
            self.limit_ke = limit_ke
            """The elastic stiffness of the joint axis limits. Defaults to 1e4."""
            self.limit_kd = limit_kd
            """The damping stiffness of the joint axis limits. Defaults to 1e1."""
            self.target = target
            """The target position or velocity (depending on the mode) of the joint axis.
            If `mode` is `JOINT_MODE_TARGET_POSITION` and the initial `target` is outside the limits,
            it defaults to the midpoint of `limit_lower` and `limit_upper`. Otherwise, defaults to 0.0."""
            self.target_ke = target_ke
            """The proportional gain of the target drive PD controller. Defaults to 0.0."""
            self.target_kd = target_kd
            """The derivative gain of the target drive PD controller. Defaults to 0.0."""
            self.mode = mode
            """The mode of the joint axis (e.g., `JOINT_MODE_TARGET_POSITION` or `JOINT_MODE_TARGET_VELOCITY`). Defaults to `JOINT_MODE_TARGET_POSITION`."""
            self.armature = armature
            """Artificial inertia added around the joint axis. Defaults to 1e-2."""
            self.effort_limit = effort_limit
            """Maximum effort (force or torque) the joint axis can exert. Defaults to 1e6."""
            self.velocity_limit = velocity_limit
            """Maximum velocity the joint axis can achieve. Defaults to 1e6."""
            self.friction = friction
            """Friction coefficient for the joint axis. Defaults to 0.0."""

            if self.mode == JOINT_MODE_TARGET_POSITION and (
                self.target > self.limit_upper or self.target < self.limit_lower
            ):
                self.target = 0.5 * (self.limit_lower + self.limit_upper)

        @classmethod
        def create_unlimited(cls, axis: AxisType | Vec3) -> ModelBuilder.JointDofConfig:
            """Creates a JointDofConfig with no limits."""
            return ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=-1e6,
                limit_upper=1e6,
                target=0.0,
                target_ke=0.0,
                target_kd=0.0,
                armature=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
                mode=JOINT_MODE_NONE,
            )

    def __init__(self, up_axis: AxisType = Axis.Z, gravity: float = -9.81):
        self.num_envs = 0

        # region defaults
        self.default_shape_cfg = ModelBuilder.ShapeConfig()
        self.default_joint_cfg = ModelBuilder.JointDofConfig()

        # Default particle settings
        self.default_particle_radius = 0.1

        # Default triangle soft mesh settings
        self.default_tri_ke = 100.0
        self.default_tri_ka = 100.0
        self.default_tri_kd = 10.0
        self.default_tri_drag = 0.0
        self.default_tri_lift = 0.0

        # Default distance constraint properties
        self.default_spring_ke = 100.0
        self.default_spring_kd = 0.0

        # Default edge bending properties
        self.default_edge_ke = 100.0
        self.default_edge_kd = 0.0

        # Default body settings
        self.default_body_armature = 0.0
        # endregion

        # particles
        self.particle_q = []
        self.particle_qd = []
        self.particle_mass = []
        self.particle_radius = []
        self.particle_flags = []
        self.particle_max_velocity = 1e5
        self.particle_color_groups: list[nparray] = []

        # shapes (each shape has an entry in these arrays)
        self.shape_key = []  # shape keys
        # transform from shape to body
        self.shape_transform = []
        # maps from shape index to body index
        self.shape_body = []
        self.shape_flags = []
        self.shape_geo_type = []
        self.shape_geo_scale = []
        self.shape_geo_src = []
        self.shape_geo_is_solid = []
        self.shape_geo_thickness = []
        self.shape_material_ke = []
        self.shape_material_kd = []
        self.shape_material_kf = []
        self.shape_material_ka = []
        self.shape_material_mu = []
        self.shape_material_restitution = []
        # collision groups within collisions are handled
        self.shape_collision_group = []
        self.shape_collision_group_map = {}
        self.last_collision_group = 0
        # radius to use for broadphase collision checking
        self.shape_collision_radius = []

        # filtering to ignore certain collision pairs
        self.shape_collision_filter_pairs = set()

        # springs
        self.spring_indices = []
        self.spring_rest_length = []
        self.spring_stiffness = []
        self.spring_damping = []
        self.spring_control = []

        # triangles
        self.tri_indices = []
        self.tri_poses = []
        self.tri_activations = []
        self.tri_materials = []
        self.tri_areas = []

        # edges (bending)
        self.edge_indices = []
        self.edge_rest_angle = []
        self.edge_rest_length = []
        self.edge_bending_properties = []

        # tetrahedra
        self.tet_indices = []
        self.tet_poses = []
        self.tet_activations = []
        self.tet_materials = []

        # muscles
        self.muscle_start = []
        self.muscle_params = []
        self.muscle_activations = []
        self.muscle_bodies = []
        self.muscle_points = []

        # rigid bodies
        self.body_mass = []
        self.body_inertia = []
        self.body_inv_mass = []
        self.body_inv_inertia = []
        self.body_com = []
        self.body_q = []
        self.body_qd = []
        self.body_key = []
        self.body_shapes = {}  # mapping from body to shapes

        # rigid joints
        self.joint_parent = []  # index of the parent body                      (constant)
        self.joint_parents = {}  # mapping from joint to parent bodies
        self.joint_child = []  # index of the child body                       (constant)
        self.joint_axis = []  # joint axis in child joint frame               (constant)
        self.joint_X_p = []  # frame of joint in parent                      (constant)
        self.joint_X_c = []  # frame of child com (in child coordinates)     (constant)
        self.joint_q = []
        self.joint_qd = []
        self.joint_f = []

        self.joint_type = []
        self.joint_key = []
        self.joint_armature = []
        self.joint_target_ke = []
        self.joint_target_kd = []
        self.joint_dof_mode = []
        self.joint_limit_lower = []
        self.joint_limit_upper = []
        self.joint_limit_ke = []
        self.joint_limit_kd = []
        self.joint_target = []
        self.joint_effort_limit = []
        self.joint_velocity_limit = []
        self.joint_friction = []

        self.joint_twist_lower = []
        self.joint_twist_upper = []

        self.joint_enabled = []

        self.joint_q_start = []
        self.joint_qd_start = []
        self.joint_dof_dim = []

        self.articulation_start = []
        self.articulation_key = []

        self.joint_dof_count = 0
        self.joint_coord_count = 0

        self.up_axis: Axis = Axis.from_any(up_axis)
        self.gravity: float = gravity

        # contacts to be generated within the given distance margin to be generated at
        # every simulation substep (can be 0 if only one PBD solver iteration is used)
        self.rigid_contact_margin = 0.1
        # torsional friction coefficient (only considered by XPBD so far)
        self.rigid_contact_torsional_friction = 0.5
        # rolling friction coefficient (only considered by XPBD so far)
        self.rigid_contact_rolling_friction = 0.001

        # number of rigid contact points to allocate in the model during self.finalize() per environment
        # if setting is None, the number of worst-case number of contacts will be calculated in self.finalize()
        self.num_rigid_contacts_per_env = None

    @property
    def up_vector(self) -> Vec3:
        """Computes the 3D up vector from :attr:`up_axis`."""
        return axis_to_vec3(self.up_axis)

    @up_vector.setter
    def up_vector(self, _):
        raise AttributeError(
            "The 'up_vector' property is read-only and cannot be set. Instead, use 'up_axis' to set the up axis."
        )

    # region counts
    @property
    def shape_count(self):
        return len(self.shape_geo_type)

    @property
    def body_count(self):
        return len(self.body_q)

    @property
    def joint_count(self):
        return len(self.joint_type)

    @property
    def particle_count(self):
        return len(self.particle_q)

    @property
    def tri_count(self):
        return len(self.tri_poses)

    @property
    def tet_count(self):
        return len(self.tet_poses)

    @property
    def edge_count(self):
        return len(self.edge_rest_angle)

    @property
    def spring_count(self):
        return len(self.spring_rest_length)

    @property
    def muscle_count(self):
        return len(self.muscle_start)

    @property
    def articulation_count(self):
        return len(self.articulation_start)

    # endregion

    def add_articulation(self, key: str | None = None):
        # an articulation is a set of contiguous bodies bodies from articulation_start[i] to articulation_start[i+1]
        # these are used for computing forward kinematics e.g.:
        # articulations are automatically 'closed' when calling finalize
        self.articulation_start.append(self.joint_count)
        self.articulation_key.append(key or f"articulation_{self.articulation_count}")

    def add_builder(
        self,
        builder: ModelBuilder,
        xform: Transform | None = None,
        update_num_env_count: bool = True,
        separate_collision_group: bool = True,
    ):
        """Copies the data from `builder`, another `ModelBuilder` to this `ModelBuilder`.

        Args:
            builder (ModelBuilder): a model builder to add model data from.
            xform (Transform): offset transform applied to root bodies.
            update_num_env_count (bool): if True, the number of environments is incremented by 1.
            separate_collision_group (bool): if True, the shapes from the articulations in `builder` will all be put into a single new collision group, otherwise, only the shapes in collision group > -1 will be moved to a new group.
        """

        if builder.up_axis != self.up_axis:
            raise ValueError("Cannot add a builder with a different up axis.")

        # explicitly resolve the transform multiplication function to avoid
        # repeatedly resolving builtin overloads during shape transformation
        transform_mul_cfunc = wp.context.runtime.core.wp_builtin_mul_transformf_transformf

        # dispatches two transform multiplies to the native implementation
        def transform_mul(a, b):
            out = wp.transformf()
            transform_mul_cfunc(a, b, ctypes.byref(out))
            return out

        start_particle_idx = self.particle_count
        if builder.particle_count:
            self.particle_max_velocity = builder.particle_max_velocity
            if xform is not None:
                pos_offset = wp.transform_get_translation(xform)
            else:
                pos_offset = np.zeros(3)
            self.particle_q.extend((np.array(builder.particle_q) + pos_offset).tolist())
            # other particle attributes are added below

        if builder.spring_count:
            self.spring_indices.extend((np.array(builder.spring_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.edge_count:
            # Update edge indices by adding offset, preserving -1 values
            edge_indices = np.array(builder.edge_indices, dtype=np.int32)
            mask = edge_indices != -1
            edge_indices[mask] += start_particle_idx
            self.edge_indices.extend(edge_indices.tolist())
        if builder.tri_count:
            self.tri_indices.extend((np.array(builder.tri_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.tet_count:
            self.tet_indices.extend((np.array(builder.tet_indices, dtype=np.int32) + start_particle_idx).tolist())

        builder_coloring_translated = [group + start_particle_idx for group in builder.particle_color_groups]
        self.particle_color_groups = combine_independent_particle_coloring(
            self.particle_color_groups, builder_coloring_translated
        )

        start_body_idx = self.body_count
        start_shape_idx = self.shape_count
        for s, b in enumerate(builder.shape_body):
            if b > -1:
                new_b = b + start_body_idx
                self.shape_body.append(new_b)
                self.shape_transform.append(builder.shape_transform[s])
            else:
                self.shape_body.append(-1)
                # apply offset transform to root bodies
                if xform is not None:
                    self.shape_transform.append(transform_mul(xform, wp.transform(*builder.shape_transform[s])))
                else:
                    self.shape_transform.append(builder.shape_transform[s])

        for b, shapes in builder.body_shapes.items():
            self.body_shapes[b + start_body_idx] = [s + start_shape_idx for s in shapes]

        if builder.joint_count:
            joint_X_p = copy.deepcopy(builder.joint_X_p)
            joint_q = copy.deepcopy(builder.joint_q)
            if xform is not None:
                for i in range(len(joint_X_p)):
                    if builder.joint_type[i] == JOINT_FREE:
                        qi = builder.joint_q_start[i]
                        xform_prev = wp.transform(joint_q[qi : qi + 3], joint_q[qi + 3 : qi + 7])
                        tf = transform_mul(xform, xform_prev)
                        joint_q[qi : qi + 3] = tf.p
                        joint_q[qi + 3 : qi + 7] = tf.q
                    elif builder.joint_parent[i] == -1:
                        joint_X_p[i] = transform_mul(xform, wp.transform(*joint_X_p[i]))
            self.joint_X_p.extend(joint_X_p)
            self.joint_q.extend(joint_q)

            # offset the indices
            self.articulation_start.extend([a + self.joint_count for a in builder.articulation_start])
            self.joint_parent.extend([p + self.body_count if p != -1 else -1 for p in builder.joint_parent])
            self.joint_child.extend([c + self.body_count for c in builder.joint_child])

            self.joint_q_start.extend([c + self.joint_coord_count for c in builder.joint_q_start])
            self.joint_qd_start.extend([c + self.joint_dof_count for c in builder.joint_qd_start])

        for i in range(builder.body_count):
            if xform is not None:
                self.body_q.append(transform_mul(xform, wp.transform(*builder.body_q[i])))
            else:
                self.body_q.append(builder.body_q[i])

        # apply collision group
        if separate_collision_group:
            self.shape_collision_group.extend([self.last_collision_group + 1 for _ in builder.shape_collision_group])
        else:
            self.shape_collision_group.extend(
                [(g + self.last_collision_group if g > -1 else -1) for g in builder.shape_collision_group]
            )
        shape_count = self.shape_count
        for i, j in builder.shape_collision_filter_pairs:
            self.shape_collision_filter_pairs.add((i + shape_count, j + shape_count))
        for group, shapes in builder.shape_collision_group_map.items():
            if separate_collision_group:
                extend_group = self.last_collision_group + 1
            else:
                extend_group = group + self.last_collision_group if group > -1 else -1

            if extend_group not in self.shape_collision_group_map:
                self.shape_collision_group_map[extend_group] = []

            self.shape_collision_group_map[extend_group].extend([s + shape_count for s in shapes])

        # update last collision group counter
        if separate_collision_group:
            self.last_collision_group += 1
        elif builder.last_collision_group > -1:
            self.last_collision_group += builder.last_collision_group

        more_builder_attrs = [
            "articulation_key",
            "body_inertia",
            "body_mass",
            "body_inv_inertia",
            "body_inv_mass",
            "body_com",
            "body_qd",
            "body_key",
            "joint_type",
            "joint_enabled",
            "joint_X_c",
            "joint_armature",
            "joint_axis",
            "joint_dof_dim",
            "joint_dof_mode",
            "joint_key",
            "joint_qd",
            "joint_f",
            "joint_target",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_target_ke",
            "joint_target_kd",
            "joint_effort_limit",
            "joint_velocity_limit",
            "joint_friction",
            "shape_key",
            "shape_flags",
            "shape_geo_type",
            "shape_geo_scale",
            "shape_geo_src",
            "shape_geo_is_solid",
            "shape_geo_thickness",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_ka",
            "shape_material_mu",
            "shape_material_restitution",
            "shape_collision_radius",
            "particle_qd",
            "particle_mass",
            "particle_radius",
            "particle_flags",
            "edge_rest_angle",
            "edge_rest_length",
            "edge_bending_properties",
            "spring_rest_length",
            "spring_stiffness",
            "spring_damping",
            "spring_control",
            "tri_poses",
            "tri_activations",
            "tri_materials",
            "tri_areas",
            "tet_poses",
            "tet_activations",
            "tet_materials",
        ]

        for attr in more_builder_attrs:
            getattr(self, attr).extend(getattr(builder, attr))

        self.joint_dof_count += builder.joint_dof_count
        self.joint_coord_count += builder.joint_coord_count

        if update_num_env_count:
            self.num_envs += 1

    def add_body(
        self,
        xform: Transform | None = None,
        armature: float | None = None,
        com: Vec3 | None = None,
        I_m: Mat33 | None = None,
        mass: float = 0.0,
        key: str | None = None,
    ) -> int:
        """Adds a rigid body to the model.

        Args:
            xform: The location of the body in the world frame.
            armature: Artificial inertia added to the body. If None, the default value from :attr:`default_body_armature` is used.
            com: The center of mass of the body w.r.t its origin. If None, the center of mass is assumed to be at the origin.
            I_m: The 3x3 inertia tensor of the body (specified relative to the center of mass). If None, the inertia tensor is assumed to be zero.
            mass: Mass of the body.
            key: Key of the body (optional).

        Returns:
            The index of the body in the model.

        Note:
            If the mass is zero then the body is treated as kinematic with no dynamics.

        """

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
        if com is None:
            com = wp.vec3()
        if I_m is None:
            I_m = wp.mat33()

        body_id = len(self.body_mass)

        # body data
        if armature is None:
            armature = self.default_body_armature
        inertia = I_m + wp.mat33(np.eye(3)) * armature
        self.body_inertia.append(inertia)
        self.body_mass.append(mass)
        self.body_com.append(com)

        if mass > 0.0:
            self.body_inv_mass.append(1.0 / mass)
        else:
            self.body_inv_mass.append(0.0)

        if any(x for x in inertia):
            self.body_inv_inertia.append(wp.inverse(inertia))
        else:
            self.body_inv_inertia.append(inertia)

        self.body_q.append(xform)
        self.body_qd.append(wp.spatial_vector())

        self.body_key.append(key or f"body_{body_id}")
        self.body_shapes[body_id] = []
        return body_id

    # region joints

    def add_joint(
        self,
        joint_type: wp.constant,
        parent: int,
        child: int,
        linear_axes: list[JointDofConfig] | None = None,
        angular_axes: list[JointDofConfig] | None = None,
        key: str | None = None,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """
        Generic method to add any type of joint to this ModelBuilder.

        Args:
            joint_type (constant): The type of joint to add (see :ref:'joint-types').
            parent (int): The index of the parent body (-1 is the world).
            child (int): The index of the child body.
            linear_axes (list(:class:`JointDofConfig`)): The linear axes (see :class:`JointDofConfig`) of the joint.
            angular_axes (list(:class:`JointDofConfig`)): The angular axes (see :class:`JointDofConfig`) of the joint.
            key (str): The key of the joint (optional).
            parent_xform (Transform): The transform of the joint in the parent body's local frame. If None, the identity transform is used.
            child_xform (Transform): The transform of the joint in the child body's local frame. If None, the identity transform is used.
            collision_filter_parent (bool): Whether to filter collisions between shapes of the parent and child bodies.
            enabled (bool): Whether the joint is enabled (not considered by :class:`FeatherstoneSolver`).

        Returns:
            The index of the added joint.
        """
        if linear_axes is None:
            linear_axes = []
        if angular_axes is None:
            angular_axes = []

        if parent_xform is None:
            parent_xform = wp.transform()
        else:
            parent_xform = wp.transform(*parent_xform)
        if child_xform is None:
            child_xform = wp.transform()
        else:
            child_xform = wp.transform(*child_xform)

        if len(self.articulation_start) == 0:
            # automatically add an articulation if none exists
            self.add_articulation()
        self.joint_type.append(joint_type)
        self.joint_parent.append(parent)
        if child not in self.joint_parents:
            self.joint_parents[child] = [parent]
        else:
            self.joint_parents[child].append(parent)
        self.joint_child.append(child)
        self.joint_X_p.append(wp.transform(parent_xform))
        self.joint_X_c.append(wp.transform(child_xform))
        self.joint_key.append(key or f"joint_{self.joint_count}")
        self.joint_dof_dim.append((len(linear_axes), len(angular_axes)))
        self.joint_enabled.append(enabled)

        def add_axis_dim(dim: ModelBuilder.JointDofConfig):
            self.joint_axis.append(dim.axis)
            self.joint_dof_mode.append(dim.mode)
            self.joint_target.append(dim.target)
            self.joint_target_ke.append(dim.target_ke)
            self.joint_target_kd.append(dim.target_kd)
            self.joint_limit_ke.append(dim.limit_ke)
            self.joint_limit_kd.append(dim.limit_kd)
            self.joint_armature.append(dim.armature)
            self.joint_effort_limit.append(dim.effort_limit)
            self.joint_velocity_limit.append(dim.velocity_limit)
            self.joint_friction.append(dim.friction)
            if np.isfinite(dim.limit_lower):
                self.joint_limit_lower.append(dim.limit_lower)
            else:
                self.joint_limit_lower.append(-1e6)
            if np.isfinite(dim.limit_upper):
                self.joint_limit_upper.append(dim.limit_upper)
            else:
                self.joint_limit_upper.append(1e6)

        for dim in linear_axes:
            add_axis_dim(dim)
        for dim in angular_axes:
            add_axis_dim(dim)

        dof_count, coord_count = get_joint_dof_count(joint_type, len(linear_axes) + len(angular_axes))

        for _ in range(coord_count):
            self.joint_q.append(0.0)
        for _ in range(dof_count):
            self.joint_qd.append(0.0)
            self.joint_f.append(0.0)

        if joint_type == JOINT_FREE or joint_type == JOINT_DISTANCE or joint_type == JOINT_BALL:
            # ensure that a valid quaternion is used for the angular dofs
            self.joint_q[-1] = 1.0

        self.joint_q_start.append(self.joint_coord_count)
        self.joint_qd_start.append(self.joint_dof_count)

        self.joint_dof_count += dof_count
        self.joint_coord_count += coord_count

        if collision_filter_parent and parent > -1:
            for child_shape in self.body_shapes[child]:
                for parent_shape in self.body_shapes[parent]:
                    self.shape_collision_filter_pairs.add((parent_shape, child_shape))

        return self.joint_count - 1

    def add_joint_revolute(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        axis: AxisType | Vec3 | JointDofConfig | None = None,
        target: float | None = None,
        target_ke: float | None = None,
        target_kd: float | None = None,
        mode: int | None = None,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        limit_ke: float | None = None,
        limit_kd: float | None = None,
        armature: float | None = None,
        effort_limit: float | None = None,
        velocity_limit: float | None = None,
        friction: float | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a revolute (hinge) joint to the model. It has one degree of freedom.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            axis (AxisType | Vec3 | JointDofConfig): The axis of rotation in the parent body's local frame, can be a :class:`JointDofConfig` object whose settings will be used instead of the other arguments.
            target: The target angle (in radians) or target velocity of the joint.
            target_ke: The stiffness of the joint target.
            target_kd: The damping of the joint target.
            mode: The control mode of the joint. If None, the default value from :attr:`default_joint_control_mode` is used.
            limit_lower: The lower limit of the joint. If None, the default value from :attr:`default_joint_limit_lower` is used.
            limit_upper: The upper limit of the joint. If None, the default value from :attr:`default_joint_limit_upper` is used.
            limit_ke: The stiffness of the joint limit. If None, the default value from :attr:`default_joint_limit_ke` is used.
            limit_kd: The damping of the joint limit. If None, the default value from :attr:`default_joint_limit_kd` is used.
            armature: Artificial inertia added around the joint axis. If None, the default value from :attr:`default_joint_armature` is used.
            effort_limit: Maximum effort (force/torque) the joint axis can exert. If None, the default value from :attr:`default_joint_cfg.effort_limit` is used.
            velocity_limit: Maximum velocity the joint axis can achieve. If None, the default value from :attr:`default_joint_cfg.velocity_limit` is used.
            friction: Friction coefficient for the joint axis. If None, the default value from :attr:`default_joint_cfg.friction` is used.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """

        if axis is None:
            axis = self.default_joint_cfg.axis
        if isinstance(axis, ModelBuilder.JointDofConfig):
            ax = axis
        else:
            ax = ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=limit_lower if limit_lower is not None else self.default_joint_cfg.limit_lower,
                limit_upper=limit_upper if limit_upper is not None else self.default_joint_cfg.limit_upper,
                target=target if target is not None else self.default_joint_cfg.target,
                target_ke=target_ke if target_ke is not None else self.default_joint_cfg.target_ke,
                target_kd=target_kd if target_kd is not None else self.default_joint_cfg.target_kd,
                mode=mode if mode is not None else self.default_joint_cfg.mode,
                limit_ke=limit_ke if limit_ke is not None else self.default_joint_cfg.limit_ke,
                limit_kd=limit_kd if limit_kd is not None else self.default_joint_cfg.limit_kd,
                armature=armature if armature is not None else self.default_joint_cfg.armature,
                effort_limit=effort_limit if effort_limit is not None else self.default_joint_cfg.effort_limit,
                velocity_limit=velocity_limit if velocity_limit is not None else self.default_joint_cfg.velocity_limit,
                friction=friction if friction is not None else self.default_joint_cfg.friction,
            )
        return self.add_joint(
            JOINT_REVOLUTE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            angular_axes=[ax],
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_prismatic(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        axis: AxisType | Vec3 | JointDofConfig = Axis.X,
        target: float | None = None,
        target_ke: float | None = None,
        target_kd: float | None = None,
        mode: int | None = None,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        limit_ke: float | None = None,
        limit_kd: float | None = None,
        armature: float | None = None,
        effort_limit: float | None = None,
        velocity_limit: float | None = None,
        friction: float | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a prismatic (sliding) joint to the model. It has one degree of freedom.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            axis (AxisType | Vec3 | JointDofConfig): The axis of rotation in the parent body's local frame, can be a :class:`JointDofConfig` object whose settings will be used instead of the other arguments.
            target: The target angle (in radians) or target velocity of the joint.
            target_ke: The stiffness of the joint target.
            target_kd: The damping of the joint target.
            mode: The control mode of the joint. If None, the default value from :attr:`default_joint_control_mode` is used.
            limit_lower: The lower limit of the joint. If None, the default value from :attr:`default_joint_limit_lower` is used.
            limit_upper: The upper limit of the joint. If None, the default value from :attr:`default_joint_limit_upper` is used.
            limit_ke: The stiffness of the joint limit. If None, the default value from :attr:`default_joint_limit_ke` is used.
            limit_kd: The damping of the joint limit. If None, the default value from :attr:`default_joint_limit_kd` is used.
            armature: Artificial inertia added around the joint axis. If None, the default value from :attr:`default_joint_armature` is used.
            effort_limit: Maximum effort (force) the joint axis can exert. If None, the default value from :attr:`default_joint_cfg.effort_limit` is used.
            velocity_limit: Maximum velocity the joint axis can achieve. If None, the default value from :attr:`default_joint_cfg.velocity_limit` is used.
            friction: Friction coefficient for the joint axis. If None, the default value from :attr:`default_joint_cfg.friction` is used.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """

        if axis is None:
            axis = self.default_joint_cfg.axis
        if isinstance(axis, ModelBuilder.JointDofConfig):
            ax = axis
        else:
            ax = ModelBuilder.JointDofConfig(
                axis=axis,
                limit_lower=limit_lower if limit_lower is not None else self.default_joint_cfg.limit_lower,
                limit_upper=limit_upper if limit_upper is not None else self.default_joint_cfg.limit_upper,
                target=target if target is not None else self.default_joint_cfg.target,
                target_ke=target_ke if target_ke is not None else self.default_joint_cfg.target_ke,
                target_kd=target_kd if target_kd is not None else self.default_joint_cfg.target_kd,
                mode=mode if mode is not None else self.default_joint_cfg.mode,
                limit_ke=limit_ke if limit_ke is not None else self.default_joint_cfg.limit_ke,
                limit_kd=limit_kd if limit_kd is not None else self.default_joint_cfg.limit_kd,
                armature=armature if armature is not None else self.default_joint_cfg.armature,
                effort_limit=effort_limit if effort_limit is not None else self.default_joint_cfg.effort_limit,
                velocity_limit=velocity_limit if velocity_limit is not None else self.default_joint_cfg.velocity_limit,
                friction=friction if friction is not None else self.default_joint_cfg.friction,
            )
        return self.add_joint(
            JOINT_PRISMATIC,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[ax],
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_ball(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a ball (spherical) joint to the model. Its position is defined by a 4D quaternion (xyzw) and its velocity is a 3D vector.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """

        return self.add_joint(
            JOINT_BALL,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_fixed(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a fixed (static) joint to the model. It has no degrees of freedom.
        See :meth:`collapse_fixed_joints` for a helper function that removes these fixed joints and merges the connecting bodies to simplify the model and improve stability.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint

        """

        return self.add_joint(
            JOINT_FIXED,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_free(
        self,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        parent: int = -1,
        key: str | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a free joint to the model.
        It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation) and 6 velocity degrees of freedom (see :ref:`Twist conventions in Newton <Twist conventions>`).
        The positional dofs are initialized by the child body's transform (see :attr:`body_q` and the ``xform`` argument to :meth:`add_body`).

        Args:
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            parent: The index of the parent body (-1 by default to use the world frame, e.g. to make the child body and its children a floating-base mechanism).
            key: The key of the joint.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """

        joint_id = self.add_joint(
            JOINT_FREE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
            linear_axes=[
                ModelBuilder.JointDofConfig.create_unlimited(Axis.X),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            angular_axes=[
                ModelBuilder.JointDofConfig.create_unlimited(Axis.X),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
        )
        q_start = self.joint_q_start[joint_id]
        # set the positional dofs to the child body's transform
        self.joint_q[q_start : q_start + 7] = list(self.body_q[child])
        return joint_id

    def add_joint_distance(
        self,
        parent: int,
        child: int,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        min_distance: float = -1.0,
        max_distance: float = 1.0,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a distance joint to the model. The distance joint constraints the distance between the joint anchor points on the two bodies (see :ref:`FK-IK`) it connects to the interval [`min_distance`, `max_distance`].
        It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation) and 6 velocity degrees of freedom (first 3 angular and then 3 linear velocity dimensions).

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            parent_xform (Transform): The transform of the joint in the parent body's local frame.
            child_xform (Transform): The transform of the joint in the child body's local frame.
            min_distance: The minimum distance between the bodies (no limit if negative).
            max_distance: The maximum distance between the bodies (no limit if negative).
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        .. note:: Distance joints are currently only supported in the :class:`newton.solvers.XPBDSolver`.

        """

        ax = ModelBuilder.JointDofConfig(
            axis=(1.0, 0.0, 0.0),
            limit_lower=min_distance,
            limit_upper=max_distance,
        )
        return self.add_joint(
            JOINT_DISTANCE,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[
                ax,
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            angular_axes=[
                ModelBuilder.JointDofConfig.create_unlimited(Axis.X),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Y),
                ModelBuilder.JointDofConfig.create_unlimited(Axis.Z),
            ],
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    def add_joint_d6(
        self,
        parent: int,
        child: int,
        linear_axes: Sequence[JointDofConfig] | None = None,
        angular_axes: Sequence[JointDofConfig] | None = None,
        key: str | None = None,
        parent_xform: Transform | None = None,
        child_xform: Transform | None = None,
        collision_filter_parent: bool = True,
        enabled: bool = True,
    ) -> int:
        """Adds a generic joint with custom linear and angular axes. The number of axes determines the number of degrees of freedom of the joint.

        Args:
            parent: The index of the parent body.
            child: The index of the child body.
            linear_axes: A list of linear axes.
            angular_axes: A list of angular axes.
            key: The key of the joint.
            parent_xform (Transform): The transform of the joint in the parent body's local frame
            child_xform (Transform): The transform of the joint in the child body's local frame
            armature: Artificial inertia added around the joint axes. If None, the default value from :attr:`default_joint_armature` is used.
            collision_filter_parent: Whether to filter collisions between shapes of the parent and child bodies.
            enabled: Whether the joint is enabled.

        Returns:
            The index of the added joint.

        """
        if linear_axes is None:
            linear_axes = []
        if angular_axes is None:
            angular_axes = []

        return self.add_joint(
            JOINT_D6,
            parent,
            child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=list(linear_axes),
            angular_axes=list(angular_axes),
            key=key,
            collision_filter_parent=collision_filter_parent,
            enabled=enabled,
        )

    # endregion

    def plot_articulation(
        self,
        show_body_keys=True,
        show_joint_keys=True,
        show_joint_types=True,
        plot_shapes=True,
        show_shape_types=True,
        show_legend=True,
    ):
        """
        Visualizes the model's articulation graph using matplotlib and networkx.
        Uses the spring layout algorithm from networkx to arrange the nodes.
        Bodies are shown as orange squares, shapes are shown as blue circles.

        Args:
            show_body_keys (bool): Whether to show the body keys or indices
            show_joint_keys (bool): Whether to show the joint keys or indices
            show_joint_types (bool): Whether to show the joint types
            plot_shapes (bool): Whether to render the shapes connected to the rigid bodies
            show_shape_types (bool): Whether to show the shape geometry types
            show_legend (bool): Whether to show a legend
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import networkx as nx  # noqa: PLC0415

        def joint_type_str(type):
            if type == JOINT_FREE:
                return "free"
            elif type == JOINT_BALL:
                return "ball"
            elif type == JOINT_PRISMATIC:
                return "prismatic"
            elif type == JOINT_REVOLUTE:
                return "revolute"
            elif type == JOINT_D6:
                return "D6"
            elif type == JOINT_FIXED:
                return "fixed"
            elif type == JOINT_DISTANCE:
                return "distance"
            return "unknown"

        def shape_type_str(type):
            if type == GEO_SPHERE:
                return "sphere"
            if type == GEO_BOX:
                return "box"
            if type == GEO_CAPSULE:
                return "capsule"
            if type == GEO_CYLINDER:
                return "cylinder"
            if type == GEO_CONE:
                return "cone"
            if type == GEO_MESH:
                return "mesh"
            if type == GEO_SDF:
                return "sdf"
            if type == GEO_PLANE:
                return "plane"
            if type == GEO_NONE:
                return "none"
            return "unknown"

        if show_body_keys:
            vertices = ["world", *self.body_key]
        else:
            vertices = ["-1"] + [str(i) for i in range(self.body_count)]
        if plot_shapes:
            for i in range(self.shape_count):
                shape_label = f"shape_{i}"
                if show_shape_types:
                    shape_label += f"\n({shape_type_str(self.shape_geo_type[i])})"
                vertices.append(shape_label)
        edges = []
        edge_labels = []
        for i in range(self.joint_count):
            edge = (self.joint_child[i] + 1, self.joint_parent[i] + 1)
            edges.append(edge)
            if show_joint_keys:
                joint_label = self.joint_key[i]
            else:
                joint_label = str(i)
            if show_joint_types:
                joint_label += f"\n({joint_type_str(self.joint_type[i])})"
            edge_labels.append(joint_label)

        if plot_shapes:
            for i in range(self.shape_count):
                edges.append((len(self.body_key) + i + 1, self.shape_body[i] + 1))

        # plot graph
        G = nx.Graph()
        for i in range(len(vertices)):
            G.add_node(i, label=vertices[i])
        for i in range(len(edges)):
            label = edge_labels[i] if i < len(edge_labels) else ""
            G.add_edge(edges[i][0], edges[i][1], label=label)
        pos = nx.spring_layout(G)
        nx.draw_networkx_edges(G, pos, node_size=0, edgelist=edges[: self.joint_count])
        # render body vertices
        draw_args = {"node_size": 100}
        bodies = nx.subgraph(G, list(range(self.body_count + 1)))
        nx.draw_networkx_nodes(bodies, pos, node_color="orange", node_shape="s", **draw_args)
        if plot_shapes:
            # render shape vertices
            shapes = nx.subgraph(G, list(range(self.body_count + 1, len(vertices))))
            nx.draw_networkx_nodes(shapes, pos, node_color="skyblue", **draw_args)
            nx.draw_networkx_edges(
                G, pos, node_size=0, edgelist=edges[self.joint_count :], edge_color="gray", style="dashed"
            )
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=6, bbox={"alpha": 0.6, "color": "w", "lw": 0}
        )
        # add node labels
        nx.draw_networkx_labels(G, pos, dict(enumerate(vertices)), font_size=6)
        if show_legend:
            plt.plot([], [], "s", color="orange", label="body")
            plt.plot([], [], "k-", label="joint")
            if plot_shapes:
                plt.plot([], [], "o", color="skyblue", label="shape")
                plt.plot([], [], "k--", label="shape-body connection")
            plt.legend(loc="upper left", fontsize=6)
        plt.show()

    def collapse_fixed_joints(self, verbose=wp.config.verbose):
        """Removes fixed joints from the model and merges the bodies they connect. This is useful for simplifying the model for faster and more stable simulation."""

        body_data = {}
        body_children = {-1: []}
        visited = {}
        merged_body_data = {}
        for i in range(self.body_count):
            key = self.body_key[i]
            body_data[i] = {
                "shapes": self.body_shapes[i],
                "q": self.body_q[i],
                "qd": self.body_qd[i],
                "mass": self.body_mass[i],
                "inertia": wp.mat33(*self.body_inertia[i]),
                "inv_mass": self.body_inv_mass[i],
                "inv_inertia": self.body_inv_inertia[i],
                "com": self.body_com[i],
                "key": key,
                "original_id": i,
            }
            visited[i] = False
            body_children[i] = []

        joint_data = {}
        for i in range(self.joint_count):
            key = self.joint_key[i]
            parent = self.joint_parent[i]
            child = self.joint_child[i]
            body_children[parent].append(child)

            q_start = self.joint_q_start[i]
            qd_start = self.joint_qd_start[i]
            if i < self.joint_count - 1:
                q_dim = self.joint_q_start[i + 1] - q_start
                qd_dim = self.joint_qd_start[i + 1] - qd_start
            else:
                q_dim = len(self.joint_q) - q_start
                qd_dim = len(self.joint_qd) - qd_start

            data = {
                "type": self.joint_type[i],
                "q": self.joint_q[q_start : q_start + q_dim],
                "qd": self.joint_qd[qd_start : qd_start + qd_dim],
                "armature": self.joint_armature[qd_start : qd_start + qd_dim],
                "q_start": q_start,
                "qd_start": qd_start,
                "key": key,
                "parent_xform": wp.transform_expand(self.joint_X_p[i]),
                "child_xform": wp.transform_expand(self.joint_X_c[i]),
                "enabled": self.joint_enabled[i],
                "axes": [],
                "axis_dim": self.joint_dof_dim[i],
                "parent": parent,
                "child": child,
                "original_id": i,
            }
            num_lin_axes, num_ang_axes = self.joint_dof_dim[i]
            for j in range(qd_start, qd_start + num_lin_axes + num_ang_axes):
                data["axes"].append(
                    {
                        "axis": self.joint_axis[j],
                        "axis_mode": self.joint_dof_mode[j],
                        "target_ke": self.joint_target_ke[j],
                        "target_kd": self.joint_target_kd[j],
                        "limit_ke": self.joint_limit_ke[j],
                        "limit_kd": self.joint_limit_kd[j],
                        "limit_lower": self.joint_limit_lower[j],
                        "limit_upper": self.joint_limit_upper[j],
                        "act": self.joint_target[j],
                    }
                )

            joint_data[(parent, child)] = data

        # sort body children so we traverse the tree in the same order as the bodies are listed
        for children in body_children.values():
            children.sort(key=lambda x: body_data[x]["original_id"])

        retained_joints = []
        retained_bodies = []
        body_remap = {-1: -1}
        body_merged_parent = {}
        body_merged_transform = {}

        # depth first search over the joint graph
        def dfs(parent_body: int, child_body: int, incoming_xform: wp.transform, last_dynamic_body: int):
            nonlocal visited
            nonlocal retained_joints
            nonlocal retained_bodies
            nonlocal body_data

            joint = joint_data[(parent_body, child_body)]
            if joint["type"] == JOINT_FIXED:
                joint_xform = joint["parent_xform"] * wp.transform_inverse(joint["child_xform"])
                incoming_xform = incoming_xform * joint_xform
                parent_key = self.body_key[parent_body] if parent_body > -1 else "world"
                child_key = self.body_key[child_body]
                last_dynamic_body_key = self.body_key[last_dynamic_body] if last_dynamic_body > -1 else "world"
                if verbose:
                    print(
                        f"Remove fixed joint {joint['key']} between {parent_key} and {child_key}, "
                        f"merging {child_key} into {last_dynamic_body_key}"
                    )
                child_id = body_data[child_body]["original_id"]
                relative_xform = incoming_xform
                merged_body_data[self.body_key[child_body]] = {
                    "relative_xform": relative_xform,
                    "parent_body": self.body_key[parent_body],
                }
                body_merged_parent[child_body] = last_dynamic_body
                body_merged_transform[child_body] = incoming_xform
                for shape in self.body_shapes[child_id]:
                    self.shape_transform[shape] = incoming_xform * self.shape_transform[shape]
                    if verbose:
                        print(
                            f"  Shape {shape} moved to body {last_dynamic_body_key} with transform {self.shape_transform[shape]}"
                        )
                    if last_dynamic_body > -1:
                        self.shape_body[shape] = body_data[last_dynamic_body]["id"]
                        body_data[last_dynamic_body]["shapes"].append(shape)
                    else:
                        self.shape_body[shape] = -1
                        if -1 not in self.body_shapes:
                            self.body_shapes[-1] = []
                        self.body_shapes[-1].append(shape)

                if last_dynamic_body > -1:
                    source_m = body_data[last_dynamic_body]["mass"]
                    source_com = body_data[last_dynamic_body]["com"]
                    # add inertia to last_dynamic_body
                    m = body_data[child_body]["mass"]
                    com = wp.transform_point(incoming_xform, body_data[child_body]["com"])
                    inertia = body_data[child_body]["inertia"]
                    body_data[last_dynamic_body]["inertia"] += transform_inertia(
                        m, inertia, incoming_xform.p, incoming_xform.q
                    )
                    body_data[last_dynamic_body]["mass"] += m
                    body_data[last_dynamic_body]["com"] = (m * com + source_m * source_com) / (m + source_m)
                    # indicate to recompute inverse mass, inertia for this body
                    body_data[last_dynamic_body]["inv_mass"] = None
            else:
                joint["parent_xform"] = incoming_xform * joint["parent_xform"]
                joint["parent"] = last_dynamic_body
                last_dynamic_body = child_body
                incoming_xform = wp.transform()
                retained_joints.append(joint)
                new_id = len(retained_bodies)
                body_data[child_body]["id"] = new_id
                retained_bodies.append(child_body)
                for shape in body_data[child_body]["shapes"]:
                    self.shape_body[shape] = new_id

            visited[parent_body] = True
            if visited[child_body] or child_body not in body_children:
                return
            for child in body_children[child_body]:
                if not visited[child]:
                    dfs(child_body, child, incoming_xform, last_dynamic_body)

        for body in body_children[-1]:
            if not visited[body]:
                dfs(-1, body, wp.transform(), -1)

        # repopulate the model
        self.body_key.clear()
        self.body_q.clear()
        self.body_qd.clear()
        self.body_mass.clear()
        self.body_inertia.clear()
        self.body_com.clear()
        self.body_inv_mass.clear()
        self.body_inv_inertia.clear()
        if -1 in self.body_shapes:
            static_shapes = self.body_shapes[-1]
            self.body_shapes.clear()
            # restore static shapes
            self.body_shapes[-1] = static_shapes
        for i in retained_bodies:
            body = body_data[i]
            new_id = len(self.body_key)
            body_remap[body["original_id"]] = new_id
            self.body_key.append(body["key"])
            self.body_q.append(list(body["q"]))
            self.body_qd.append(list(body["qd"]))
            m = body["mass"]
            inertia = body["inertia"]
            self.body_mass.append(m)
            self.body_inertia.append(inertia)
            self.body_com.append(body["com"])
            if body["inv_mass"] is None:
                # recompute inverse mass and inertia
                if m > 0.0:
                    self.body_inv_mass.append(1.0 / m)
                    self.body_inv_inertia.append(wp.inverse(inertia))
                else:
                    self.body_inv_mass.append(0.0)
                    self.body_inv_inertia.append(wp.mat33(0.0))
            else:
                self.body_inv_mass.append(body["inv_mass"])
                self.body_inv_inertia.append(body["inv_inertia"])
            self.body_shapes[new_id] = body["shapes"]

        # sort joints so they appear in the same order as before
        retained_joints.sort(key=lambda x: x["original_id"])

        joint_remap = {}
        for i, joint in enumerate(retained_joints):
            joint_remap[joint["original_id"]] = i
        # update articulation_start
        for i, old_i in enumerate(self.articulation_start):
            start_i = old_i
            while start_i not in joint_remap:
                start_i += 1
                if start_i >= self.joint_count:
                    break
            self.articulation_start[i] = joint_remap.get(start_i, start_i)
        # remove empty articulation starts, i.e. where the start and end are the same
        self.articulation_start = list(set(self.articulation_start))

        self.joint_key.clear()
        self.joint_type.clear()
        self.joint_parent.clear()
        self.joint_child.clear()
        self.joint_q.clear()
        self.joint_qd.clear()
        self.joint_q_start.clear()
        self.joint_qd_start.clear()
        self.joint_enabled.clear()
        self.joint_armature.clear()
        self.joint_X_p.clear()
        self.joint_X_c.clear()
        self.joint_axis.clear()
        self.joint_dof_mode.clear()
        self.joint_target_ke.clear()
        self.joint_target_kd.clear()
        self.joint_limit_lower.clear()
        self.joint_limit_upper.clear()
        self.joint_limit_ke.clear()
        self.joint_limit_kd.clear()
        self.joint_dof_dim.clear()
        self.joint_target.clear()
        for joint in retained_joints:
            self.joint_key.append(joint["key"])
            self.joint_type.append(joint["type"])
            self.joint_parent.append(body_remap[joint["parent"]])
            self.joint_child.append(body_remap[joint["child"]])
            self.joint_q_start.append(len(self.joint_q))
            self.joint_qd_start.append(len(self.joint_qd))
            self.joint_q.extend(joint["q"])
            self.joint_qd.extend(joint["qd"])
            self.joint_armature.extend(joint["armature"])
            self.joint_enabled.append(joint["enabled"])
            self.joint_X_p.append(list(joint["parent_xform"]))
            self.joint_X_c.append(list(joint["child_xform"]))
            self.joint_dof_dim.append(joint["axis_dim"])
            for axis in joint["axes"]:
                self.joint_axis.append(axis["axis"])
                self.joint_dof_mode.append(axis["axis_mode"])
                self.joint_target_ke.append(axis["target_ke"])
                self.joint_target_kd.append(axis["target_kd"])
                self.joint_limit_lower.append(axis["limit_lower"])
                self.joint_limit_upper.append(axis["limit_upper"])
                self.joint_limit_ke.append(axis["limit_ke"])
                self.joint_limit_kd.append(axis["limit_kd"])
                self.joint_target.append(axis["act"])

        return {
            "body_remap": body_remap,
            "joint_remap": joint_remap,
            "body_merged_parent": body_merged_parent,
            "body_merged_transform": body_merged_transform,
            # TODO clean up this data
            "merged_body_data": merged_body_data,
        }

    # muscles
    def add_muscle(
        self, bodies: list[int], positions: list[Vec3], f0: float, lm: float, lt: float, lmax: float, pen: float
    ) -> float:
        """Adds a muscle-tendon activation unit.

        Args:
            bodies: A list of body indices for each waypoint
            positions: A list of positions of each waypoint in the body's local frame
            f0: Force scaling
            lm: Muscle length
            lt: Tendon length
            lmax: Maximally efficient muscle length

        Returns:
            The index of the muscle in the model

        .. note:: The simulation support for muscles is in progress and not yet fully functional.

        """

        n = len(bodies)

        self.muscle_start.append(len(self.muscle_bodies))
        self.muscle_params.append((f0, lm, lt, lmax, pen))
        self.muscle_activations.append(0.0)

        for i in range(n):
            self.muscle_bodies.append(bodies[i])
            self.muscle_points.append(positions[i])

        # return the index of the muscle
        return len(self.muscle_start) - 1

    # region shapes

    def add_shape(
        self,
        body: int,
        type: int,
        xform: Transform | None = None,
        cfg: ShapeConfig | None = None,
        scale: Vec3 | None = None,
        src: SDF | Mesh | Any | None = None,
        is_static: bool = False,
        key: str | None = None,
    ) -> int:
        """Adds a generic collision shape to the model.

        This is the base method for adding shapes; prefer using specific helpers like :meth:`add_shape_sphere` where possible.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body (e.g., static environment geometry).
            type (int): The geometry type of the shape (e.g., `GEO_BOX`, `GEO_SPHERE`).
            xform (Transform | None): The transform of the shape in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            scale (Vec3 | None): The scale of the geometry. The interpretation depends on the shape type. Defaults to `(1.0, 1.0, 1.0)` if `None`.
            src (SDF | Mesh | Any | None): The source geometry data, e.g., a :class:`Mesh` object for `GEO_MESH` or an :class:`SDF` object for `GEO_SDF`. Defaults to `None`.
            is_static (bool): If `True`, the shape will have zero mass, and its density property in `cfg` will be effectively ignored for mass calculation. Typically used for fixed, non-movable collision geometry. Defaults to `False`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated (e.g., "shape_N"). Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """
        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
        if cfg is None:
            cfg = self.default_shape_cfg
        if scale is None:
            scale = (1.0, 1.0, 1.0)
        self.shape_body.append(body)
        shape = self.shape_count
        if body in self.body_shapes:
            # no contacts between shapes of the same body
            for same_body_shape in self.body_shapes[body]:
                self.shape_collision_filter_pairs.add((same_body_shape, shape))
            self.body_shapes[body].append(shape)
        else:
            self.body_shapes[body] = [shape]
        self.shape_key.append(key or f"shape_{shape}")
        self.shape_transform.append(xform)
        self.shape_flags.append(cfg.flags)
        self.shape_geo_type.append(type)
        self.shape_geo_scale.append((scale[0], scale[1], scale[2]))
        self.shape_geo_src.append(src)
        self.shape_geo_thickness.append(cfg.thickness)
        self.shape_geo_is_solid.append(cfg.is_solid)
        self.shape_material_ke.append(cfg.ke)
        self.shape_material_kd.append(cfg.kd)
        self.shape_material_kf.append(cfg.kf)
        self.shape_material_ka.append(cfg.ka)
        self.shape_material_mu.append(cfg.mu)
        self.shape_material_restitution.append(cfg.restitution)
        self.shape_collision_group.append(cfg.collision_group)
        if cfg.collision_group not in self.shape_collision_group_map:
            self.shape_collision_group_map[cfg.collision_group] = []
        self.last_collision_group = max(self.last_collision_group, cfg.collision_group)
        self.shape_collision_group_map[cfg.collision_group].append(shape)
        self.shape_collision_radius.append(compute_shape_radius(type, scale, src))
        if cfg.collision_filter_parent and body > -1 and body in self.joint_parents:
            for parent_body in self.joint_parents[body]:
                if parent_body > -1:
                    for parent_shape in self.body_shapes[parent_body]:
                        self.shape_collision_filter_pairs.add((parent_shape, shape))

        if not is_static and cfg.density > 0.0:
            (m, c, I) = compute_shape_inertia(type, scale, src, cfg.density, cfg.is_solid, cfg.thickness)
            com_body = wp.transform_point(xform, c)
            self._update_body_mass(body, m, I, com_body, xform.q)
        return shape

    def add_shape_plane(
        self,
        plane: Vec4 | None = (0.0, 0.0, 1.0, 0.0),
        xform: Transform | None = None,
        width: float = 10.0,
        length: float = 10.0,
        body: int = -1,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """
        Adds a plane collision shape to the model.

        If `xform` is provided, it directly defines the plane's position and orientation. The plane's collision normal
        is assumed to be along the local Y-axis of this `xform`.
        If `xform` is `None`, it will be derived from the `plane` equation `a*x + b*y + c*z + d = 0`.
        Plane shapes added via this method are always static (massless).

        Args:
            plane (Vec4 | None): The plane equation `(a, b, c, d)`. If `xform` is `None`, this defines the plane.
                The normal is `(a,b,c)` and `d` is the offset. Defaults to `(0.0, 1.0, 0.0, 0.0)` (an XZ ground plane at Y=0) if `xform` is also `None`.
            xform (Transform | None): The transform of the plane in the world or parent body's frame. If `None`, transform is derived from `plane`. Defaults to `None`.
            width (float): The visual/collision extent of the plane along its local X-axis. If `0.0`, considered infinite for collision. Defaults to `10.0`.
            length (float): The visual/collision extent of the plane along its local Z-axis. If `0.0`, considered infinite for collision. Defaults to `10.0`.
            body (int): The index of the parent body this shape belongs to. Use -1 for world-static planes. Defaults to `-1`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """
        if xform is None:
            assert plane is not None, "Either xform or plane must be provided"
            # compute position and rotation from plane equation
            normal = np.array(plane[:3])
            normal /= np.linalg.norm(normal)
            pos = plane[3] * normal
            if np.allclose(normal, (0.0, 1.0, 0.0)):
                # no rotation necessary
                rot = wp.quat_identity()
            else:
                c = np.cross(normal, (0.0, 1.0, 0.0))
                angle = np.arcsin(np.linalg.norm(c))
                axis = np.abs(c) / np.linalg.norm(c)
                rot = wp.quat_from_axis_angle(wp.vec3(*axis), wp.float32(angle))
            xform = wp.transform(pos, rot)
        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(width, length, 0.0)
        return self.add_shape(
            body=body,
            type=GEO_PLANE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            is_static=True,
            key=key,
        )

    def add_ground_plane(
        self,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a ground plane collision shape to the model.

        Args:
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """
        return self.add_shape_plane(
            plane=(*self.up_vector, 0.0),
            width=0.0,
            length=0.0,
            cfg=cfg,
            key=key or "ground_plane",
        )

    def add_shape_sphere(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a sphere collision shape to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the sphere in the parent body's local frame. The sphere is centered at this transform's position. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the sphere. Defaults to `1.0`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if cfg is None:
            cfg = self.default_shape_cfg
        scale: Any = wp.vec3(radius, 0.0, 0.0)
        return self.add_shape(
            body=body,
            type=GEO_SPHERE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_box(
        self,
        body: int,
        xform: Transform | None = None,
        hx: float = 0.5,
        hy: float = 0.5,
        hz: float = 0.5,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a box collision shape to a body.

        The box is centered at its local origin as defined by `xform`.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the box in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            hx (float): The half-extent of the box along its local X-axis. Defaults to `0.5`.
            hy (float): The half-extent of the box along its local Y-axis. Defaults to `0.5`.
            hz (float): The half-extent of the box along its local Z-axis. Defaults to `0.5`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(hx, hy, hz)
        return self.add_shape(
            body=body,
            type=GEO_BOX,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_capsule(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        half_height: float = 0.5,
        up_axis: AxisType = Axis.Z,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a capsule collision shape to a body.

        The capsule is centered at its local origin as defined by `xform`. Its length extends along the specified `up_axis`.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the capsule in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the capsule's hemispherical ends and its cylindrical segment. Defaults to `1.0`.
            half_height (float): The half-length of the capsule's central cylindrical segment (excluding the hemispherical ends). Defaults to `0.5`.
            up_axis (AxisType): The local axis of the capsule along which its length is aligned (typically `Axis.X`, `Axis.Y`, or `Axis.Z`). Defaults to `Axis.Y`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
        # up axis is always +Y for capsules
        q = quat_between_axes(Axis.Y, up_axis)
        xform = wp.transform(xform.p, xform.q * q)

        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(radius, half_height, 0.0)
        return self.add_shape(
            body=body,
            type=GEO_CAPSULE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_cylinder(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        half_height: float = 0.5,
        up_axis: AxisType = Axis.Z,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a cylinder collision shape to a body.

        The cylinder is centered at its local origin as defined by `xform`. Its length extends along the specified `up_axis`.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the cylinder in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the cylinder. Defaults to `1.0`.
            half_height (float): The half-length of the cylinder along the `up_axis`. Defaults to `0.5`.
            up_axis (AxisType): The local axis of the cylinder along which its length is aligned (e.g., `Axis.X`, `Axis.Y`, `Axis.Z`). Defaults to `Axis.Y`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
        # up axis is always +Y for cylinders
        q = quat_between_axes(Axis.Y, up_axis)
        xform = wp.transform(xform.p, xform.q * q)

        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(radius, half_height, 0.0)
        return self.add_shape(
            body=body,
            type=GEO_CYLINDER,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_cone(
        self,
        body: int,
        xform: Transform | None = None,
        radius: float = 1.0,
        half_height: float = 0.5,
        up_axis: AxisType = Axis.Z,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a cone collision shape to a body.

        The cone's base is centered at its local origin as defined by `xform` and it extends along the specified `up_axis` towards its apex.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the cone in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            radius (float): The radius of the cone's base. Defaults to `1.0`.
            half_height (float): The half-height of the cone (distance from the center of the base to the apex along the `up_axis`). Defaults to `0.5`.
            up_axis (AxisType): The local axis of the cone along which its height is aligned, pointing from base to apex (e.g., `Axis.X`, `Axis.Y`, `Axis.Z`). Defaults to `Axis.Y`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
        # up axis is always +Y for cones
        q = quat_between_axes(Axis.Y, up_axis)
        xform = wp.transform(xform.p, xform.q * q)

        if cfg is None:
            cfg = self.default_shape_cfg
        scale = wp.vec3(radius, half_height, 0.0)
        return self.add_shape(
            body=body,
            type=GEO_CONE,
            xform=xform,
            cfg=cfg,
            scale=scale,
            key=key,
        )

    def add_shape_mesh(
        self,
        body: int,
        xform: Transform | None = None,
        mesh: Mesh | None = None,
        scale: Vec3 | None = None,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a triangle mesh collision shape to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the mesh in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            mesh (Mesh | None): The :class:`Mesh` object containing the vertex and triangle data. Defaults to `None`.
            scale (Vec3 | None): The scale of the mesh. Defaults to `None`, in which case the scale is `(1.0, 1.0, 1.0)`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """

        if cfg is None:
            cfg = self.default_shape_cfg
        return self.add_shape(
            body=body,
            type=GEO_MESH,
            xform=xform,
            cfg=cfg,
            scale=scale,
            src=mesh,
            key=key,
        )

    def add_shape_sdf(
        self,
        body: int,
        xform: Transform | None = None,
        sdf: SDF | None = None,
        cfg: ShapeConfig | None = None,
        key: str | None = None,
    ) -> int:
        """Adds a signed distance field (SDF) collision shape to a body.

        Args:
            body (int): The index of the parent body this shape belongs to. Use -1 for shapes not attached to any specific body.
            xform (Transform | None): The transform of the SDF in the parent body's local frame. If `None`, the identity transform `wp.transform()` is used. Defaults to `None`.
            sdf (SDF | None): The :class:`SDF` object representing the signed distance field. Defaults to `None`.
            cfg (ShapeConfig | None): The configuration for the shape's physical and collision properties. If `None`, :attr:`default_shape_cfg` is used. Defaults to `None`.
            key (str | None): An optional unique key for identifying the shape. If `None`, a default key is automatically generated. Defaults to `None`.

        Returns:
            int: The index of the newly added shape.
        """
        if cfg is None:
            cfg = self.default_shape_cfg
        return self.add_shape(
            body=body,
            type=GEO_SDF,
            xform=xform,
            cfg=cfg,
            src=sdf,
            key=key,
        )

    # endregion

    # particles
    def add_particle(
        self,
        pos: Vec3,
        vel: Vec3,
        mass: float,
        radius: float | None = None,
        flags: wp.uint32 = PARTICLE_FLAG_ACTIVE,
    ) -> int:
        """Adds a single particle to the model.

        Args:
            pos: The initial position of the particle.
            vel: The initial velocity of the particle.
            mass: The mass of the particle.
            radius: The radius of the particle used in collision handling. If None, the radius is set to the default value (:attr:`default_particle_radius`).
            flags: The flags that control the dynamical behavior of the particle, see PARTICLE_FLAG_* constants.

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that is not subject to dynamics.

        Returns:
            The index of the particle in the system.
        """
        self.particle_q.append(pos)
        self.particle_qd.append(vel)
        self.particle_mass.append(mass)
        if radius is None:
            radius = self.default_particle_radius
        self.particle_radius.append(radius)
        self.particle_flags.append(flags)

        particle_id = self.particle_count - 1

        return particle_id

    def add_particles(
        self,
        pos: list[Vec3],
        vel: list[Vec3],
        mass: list[float],
        radius: list[float] | None = None,
        flags: list[wp.uint32] | None = None,
    ):
        """Adds a group particles to the model.

        Args:
            pos: The initial positions of the particle.
            vel: The initial velocities of the particle.
            mass: The mass of the particles.
            radius: The radius of the particles used in collision handling. If None, the radius is set to the default value (:attr:`default_particle_radius`).
            flags: The flags that control the dynamical behavior of the particles, see PARTICLE_FLAG_* constants.

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that is not subject to dynamics.
        """
        self.particle_q.extend(pos)
        self.particle_qd.extend(vel)
        self.particle_mass.extend(mass)
        if radius is None:
            radius = [self.default_particle_radius] * len(pos)
        if flags is None:
            flags = [PARTICLE_FLAG_ACTIVE] * len(pos)
        self.particle_radius.extend(radius)
        self.particle_flags.extend(flags)

    def add_spring(self, i: int, j, ke: float, kd: float, control: float):
        """Adds a spring between two particles in the system

        Args:
            i: The index of the first particle
            j: The index of the second particle
            ke: The elastic stiffness of the spring
            kd: The damping stiffness of the spring
            control: The actuation level of the spring

        Note:
            The spring is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """
        self.spring_indices.append(i)
        self.spring_indices.append(j)
        self.spring_stiffness.append(ke)
        self.spring_damping.append(kd)
        self.spring_control.append(control)

        # compute rest length
        p = self.particle_q[i]
        q = self.particle_q[j]

        delta = np.subtract(p, q)
        l = np.sqrt(np.dot(delta, delta))

        self.spring_rest_length.append(l)

    def add_triangle(
        self,
        i: int,
        j: int,
        k: int,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
    ) -> float:
        """Adds a triangular FEM element between three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        parameters specified on the model. See model.tri_ke, model.tri_kd.

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle

        Return:
            The area of the triangle

        Note:
            The triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.
        """
        # TODO: Expose elastic parameters on a per-element basis
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift

        # compute basis for 2D rest pose
        p = self.particle_q[i]
        q = self.particle_q[j]
        r = self.particle_q[k]

        qp = q - p
        rp = r - p

        # construct basis aligned with the triangle
        n = wp.normalize(wp.cross(qp, rp))
        e1 = wp.normalize(qp)
        e2 = wp.normalize(wp.cross(n, e1))

        R = np.array((e1, e2))
        M = np.array((qp, rp))

        D = R @ M.T

        area = np.linalg.det(D) / 2.0

        if area <= 0.0:
            print("inverted or degenerate triangle element")
            return 0.0
        else:
            inv_D = np.linalg.inv(D)

            self.tri_indices.append((i, j, k))
            self.tri_poses.append(inv_D.tolist())
            self.tri_activations.append(0.0)
            self.tri_materials.append((tri_ke, tri_ka, tri_kd, tri_drag, tri_lift))
            self.tri_areas.append(area)
            return area

    def add_triangles(
        self,
        i: list[int],
        j: list[int],
        k: list[int],
        tri_ke: list[float] | None = None,
        tri_ka: list[float] | None = None,
        tri_kd: list[float] | None = None,
        tri_drag: list[float] | None = None,
        tri_lift: list[float] | None = None,
    ) -> list[float]:
        """Adds triangular FEM elements between groups of three particles in the system.

        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
        Parameters specified on the model. See model.tri_ke, model.tri_kd.

        Args:
            i: The indices of the first particle
            j: The indices of the second particle
            k: The indices of the third particle

        Return:
            The areas of the triangles

        Note:
            A triangle is created with a rest-length based on the distance
            between the particles in their initial configuration.

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q)[i]
        q = np.array(self.particle_q)[j]
        r = np.array(self.particle_q)[k]

        qp = q - p
        rp = r - p

        def normalized(a):
            l = np.linalg.norm(a, axis=-1, keepdims=True)
            l[l == 0] = 1.0
            return a / l

        n = normalized(np.cross(qp, rp))
        e1 = normalized(qp)
        e2 = normalized(np.cross(n, e1))

        R = np.concatenate((e1[..., None], e2[..., None]), axis=-1)
        M = np.concatenate((qp[..., None], rp[..., None]), axis=-1)

        D = np.matmul(R.transpose(0, 2, 1), M)

        areas = np.linalg.det(D) / 2.0
        areas[areas < 0.0] = 0.0
        valid_inds = (areas > 0.0).nonzero()[0]
        if len(valid_inds) < len(areas):
            print("inverted or degenerate triangle elements")

        D[areas == 0.0] = np.eye(2)[None, ...]
        inv_D = np.linalg.inv(D)

        inds = np.concatenate((i[valid_inds, None], j[valid_inds, None], k[valid_inds, None]), axis=-1)

        self.tri_indices.extend(inds.tolist())
        self.tri_poses.extend(inv_D[valid_inds].tolist())
        self.tri_activations.extend([0.0] * len(valid_inds))

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(areas)
            return arr

        tri_ke = init_if_none(tri_ke, self.default_tri_ke)
        tri_ka = init_if_none(tri_ka, self.default_tri_ka)
        tri_kd = init_if_none(tri_kd, self.default_tri_kd)
        tri_drag = init_if_none(tri_drag, self.default_tri_drag)
        tri_lift = init_if_none(tri_lift, self.default_tri_lift)

        self.tri_materials.extend(
            zip(
                np.array(tri_ke)[valid_inds],
                np.array(tri_ka)[valid_inds],
                np.array(tri_kd)[valid_inds],
                np.array(tri_drag)[valid_inds],
                np.array(tri_lift)[valid_inds],
            )
        )
        areas = areas.tolist()
        self.tri_areas.extend(areas)
        return areas

    def add_tetrahedron(
        self, i: int, j: int, k: int, l: int, k_mu: float = 1.0e3, k_lambda: float = 1.0e3, k_damp: float = 0.0
    ) -> float:
        """Adds a tetrahedral FEM element between four particles in the system.

        Tetrahedra are modeled as viscoelastic elements with a NeoHookean energy
        density based on [Smith et al. 2018].

        Args:
            i: The index of the first particle
            j: The index of the second particle
            k: The index of the third particle
            l: The index of the fourth particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The element's damping stiffness

        Return:
            The volume of the tetrahedron

        Note:
            The tetrahedron is created with a rest-pose based on the particle's initial configuration

        """
        # compute basis for 2D rest pose
        p = np.array(self.particle_q[i])
        q = np.array(self.particle_q[j])
        r = np.array(self.particle_q[k])
        s = np.array(self.particle_q[l])

        qp = q - p
        rp = r - p
        sp = s - p

        Dm = np.array((qp, rp, sp)).T
        volume = np.linalg.det(Dm) / 6.0

        if volume <= 0.0:
            print("inverted tetrahedral element")
        else:
            inv_Dm = np.linalg.inv(Dm)

            self.tet_indices.append((i, j, k, l))
            self.tet_poses.append(inv_Dm.tolist())
            self.tet_activations.append(0.0)
            self.tet_materials.append((k_mu, k_lambda, k_damp))

        return volume

    def add_edge(
        self,
        i: int,
        j: int,
        k: int,
        l: int,
        rest: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
    ) -> None:
        """Adds a bending edge element between two adjacent triangles in the cloth mesh, defined by four vertices.

        The bending energy model follows the discrete shell formulation from [Grinspun et al. 2003].
        The bending stiffness is controlled by the `edge_ke` parameter, and the bending damping by the `edge_kd` parameter.

        Args:
            i: The index of the first particle, i.e., opposite vertex 0
            j: The index of the second particle, i.e., opposite vertex 1
            k: The index of the third particle, i.e., vertex 0
            l: The index of the fourth particle, i.e., vertex 1
            rest: The rest angle across the edge in radians, if not specified it will be computed
            edge_ke: The bending stiffness coefficient
            edge_kd: The bending damping coefficient

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counterclockwise
            winding: (i, k, l), (j, l, k).

        """
        edge_ke = edge_ke if edge_ke is not None else self.default_edge_ke
        edge_kd = edge_kd if edge_kd is not None else self.default_edge_kd

        # compute rest angle
        x3 = self.particle_q[k]
        x4 = self.particle_q[l]
        if rest is None:
            rest = 0.0
            if i != -1 and j != -1:
                x1 = self.particle_q[i]
                x2 = self.particle_q[j]

                n1 = wp.normalize(wp.cross(x3 - x1, x4 - x1))
                n2 = wp.normalize(wp.cross(x4 - x2, x3 - x2))
                e = wp.normalize(x4 - x3)

                cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
                sin_theta = np.dot(np.cross(n1, n2), e)
                rest = math.atan2(sin_theta, cos_theta)

        self.edge_indices.append((i, j, k, l))
        self.edge_rest_angle.append(rest)
        self.edge_rest_length.append(wp.length(x4 - x3))
        self.edge_bending_properties.append((edge_ke, edge_kd))

    def add_edges(
        self,
        i,
        j,
        k,
        l,
        rest: list[float] | None = None,
        edge_ke: list[float] | None = None,
        edge_kd: list[float] | None = None,
    ) -> None:
        """Adds bending edge elements between two adjacent triangles in the cloth mesh, defined by four vertices.

        The bending energy model follows the discrete shell formulation from [Grinspun et al. 2003].
        The bending stiffness is controlled by the `edge_ke` parameter, and the bending damping by the `edge_kd` parameter.

        Args:
            i: The index of the first particle, i.e., opposite vertex 0
            j: The index of the second particle, i.e., opposite vertex 1
            k: The index of the third particle, i.e., vertex 0
            l: The index of the fourth particle, i.e., vertex 1
            rest: The rest angles across the edges in radians, if not specified they will be computed
            edge_ke: The bending stiffness coefficient
            edge_kd: The bending damping coefficient

        Note:
            The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
            vertices indexed by 'i' and 'j'. This defines two connected triangles with counterclockwise
            winding: (i, k, l), (j, l, k).

        """
        x3 = np.array(self.particle_q)[k]
        x4 = np.array(self.particle_q)[l]
        if rest is None:
            rest = np.zeros_like(i, dtype=float)
            valid_mask = (i != -1) & (j != -1)

            # compute rest angle
            x1_valid = np.array(self.particle_q)[i[valid_mask]]
            x2_valid = np.array(self.particle_q)[j[valid_mask]]
            x3_valid = np.array(self.particle_q)[k[valid_mask]]
            x4_valid = np.array(self.particle_q)[l[valid_mask]]

            def normalized(a):
                l = np.linalg.norm(a, axis=-1, keepdims=True)
                l[l == 0] = 1.0
                return a / l

            n1 = normalized(np.cross(x3_valid - x1_valid, x4_valid - x1_valid))
            n2 = normalized(np.cross(x4_valid - x2_valid, x3_valid - x2_valid))
            e = normalized(x4_valid - x3_valid)

            def dot(a, b):
                return (a * b).sum(axis=-1)

            cos_theta = np.clip(dot(n1, n2), -1.0, 1.0)
            sin_theta = dot(np.cross(n1, n2), e)
            rest[valid_mask] = np.arctan2(sin_theta, cos_theta)

        inds = np.concatenate((i[:, None], j[:, None], k[:, None], l[:, None]), axis=-1)

        self.edge_indices.extend(inds.tolist())
        self.edge_rest_angle.extend(rest.tolist())
        self.edge_rest_length.extend(np.linalg.norm(x4 - x3, axis=1).tolist())

        def init_if_none(arr, defaultValue):
            if arr is None:
                return [defaultValue] * len(i)
            return arr

        edge_ke = init_if_none(edge_ke, self.default_edge_ke)
        edge_kd = init_if_none(edge_kd, self.default_edge_kd)

        self.edge_bending_properties.extend(zip(edge_ke, edge_kd))

    def add_cloth_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        cell_x: float,
        cell_y: float,
        mass: float,
        reverse_winding: bool = False,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
        add_springs: bool = False,
        spring_ke: float | None = None,
        spring_kd: float | None = None,
        particle_radius: float | None = None,
    ):
        """Helper to create a regular planar cloth grid

        Creates a rectangular grid of particles with FEM triangles and bending elements
        automatically.

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            mass: The mass of each particle
            reverse_winding: Flip the winding of the mesh
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
        """

        def grid_index(x, y, dim_x):
            return y * dim_x + x

        indices, vertices = [], []
        for y in range(0, dim_y + 1):
            for x in range(0, dim_x + 1):
                local_pos = wp.vec3(x * cell_x, y * cell_y, 0.0)
                vertices.append(local_pos)
                if x > 0 and y > 0:
                    v0 = grid_index(x - 1, y - 1, dim_x + 1)
                    v1 = grid_index(x, y - 1, dim_x + 1)
                    v2 = grid_index(x, y, dim_x + 1)
                    v3 = grid_index(x - 1, y, dim_x + 1)
                    if reverse_winding:
                        indices.extend([v0, v1, v2])
                        indices.extend([v0, v2, v3])
                    else:
                        indices.extend([v0, v1, v3])
                        indices.extend([v1, v2, v3])

        start_vertex = len(self.particle_q)

        total_mass = mass * (dim_x + 1) * (dim_x + 1)
        total_area = cell_x * cell_y * dim_x * dim_y
        density = total_mass / total_area

        self.add_cloth_mesh(
            pos=pos,
            rot=rot,
            scale=1.0,
            vel=vel,
            vertices=vertices,
            indices=indices,
            density=density,
            edge_callback=None,
            face_callback=None,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            tri_drag=tri_drag,
            tri_lift=tri_lift,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
            add_springs=add_springs,
            spring_ke=spring_ke,
            spring_kd=spring_kd,
            particle_radius=particle_radius,
        )

        vertex_id = 0
        for y in range(dim_y + 1):
            for x in range(dim_x + 1):
                particle_mass = mass
                particle_flag = PARTICLE_FLAG_ACTIVE

                if (
                    (x == 0 and fix_left)
                    or (x == dim_x and fix_right)
                    or (y == 0 and fix_bottom)
                    or (y == dim_y and fix_top)
                ):
                    particle_flag = wp.uint32(int(particle_flag) & ~int(PARTICLE_FLAG_ACTIVE))
                    particle_mass = 0.0

                self.particle_flags[start_vertex + vertex_id] = particle_flag
                self.particle_mass[start_vertex + vertex_id] = particle_mass
                vertex_id = vertex_id + 1

    def add_cloth_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: list[Vec3],
        indices: list[int],
        density: float,
        edge_callback=None,
        face_callback=None,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
        edge_ke: float | None = None,
        edge_kd: float | None = None,
        add_springs: bool = False,
        spring_ke: float | None = None,
        spring_kd: float | None = None,
        particle_radius: float | None = None,
    ) -> None:
        """Helper to create a cloth model from a regular triangle mesh

        Creates one FEM triangle element and one bending element for every face
        and edge in the input triangle mesh

        Args:
            pos: The position of the cloth in world space
            rot: The orientation of the cloth in world space
            vel: The velocity of the cloth in world space
            vertices: A list of vertex positions
            indices: A list of triangle indices, 3 entries per-face
            density: The density per-area of the mesh
            edge_callback: A user callback when an edge is created
            face_callback: A user callback when a face is created
            particle_radius: The particle_radius which controls particle based collisions.
        Note:

            The mesh should be two manifold.
        """
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift
        edge_ke = edge_ke if edge_ke is not None else self.default_edge_ke
        edge_kd = edge_kd if edge_kd is not None else self.default_edge_kd
        spring_ke = spring_ke if spring_ke is not None else self.default_spring_ke
        spring_kd = spring_kd if spring_kd is not None else self.default_spring_kd
        particle_radius = particle_radius if particle_radius is not None else self.default_particle_radius

        num_verts = int(len(vertices))
        num_tris = int(len(indices) / 3)

        start_vertex = len(self.particle_q)
        start_tri = len(self.tri_indices)

        # particles
        # for v in vertices:
        #     p = wp.quat_rotate(rot, v * scale) + pos
        #     self.add_particle(p, vel, 0.0, radius=particle_radius)
        vertices_np = np.array(vertices) * scale
        rot_mat_np = np.array(wp.quat_to_matrix(rot), dtype=np.float32).reshape(3, 3)
        verts_3d_np = np.dot(vertices_np, rot_mat_np.T) + pos
        self.add_particles(
            verts_3d_np.tolist(), [vel] * num_verts, mass=[0.0] * num_verts, radius=[particle_radius] * num_verts
        )

        # triangles
        inds = start_vertex + np.array(indices)
        inds = inds.reshape(-1, 3)
        areas = self.add_triangles(
            inds[:, 0],
            inds[:, 1],
            inds[:, 2],
            [tri_ke] * num_tris,
            [tri_ka] * num_tris,
            [tri_kd] * num_tris,
            [tri_drag] * num_tris,
            [tri_lift] * num_tris,
        )

        for t in range(num_tris):
            area = areas[t]

            self.particle_mass[inds[t, 0]] += density * area / 3.0
            self.particle_mass[inds[t, 1]] += density * area / 3.0
            self.particle_mass[inds[t, 2]] += density * area / 3.0

        end_tri = len(self.tri_indices)

        adj = wp.utils.MeshAdjacency(self.tri_indices[start_tri:end_tri], end_tri - start_tri)

        edge_indices = np.fromiter(
            (x for e in adj.edges.values() for x in (e.o0, e.o1, e.v0, e.v1)),
            int,
        ).reshape(-1, 4)
        self.add_edges(
            edge_indices[:, 0],
            edge_indices[:, 1],
            edge_indices[:, 2],
            edge_indices[:, 3],
            edge_ke=[edge_ke] * len(edge_indices),
            edge_kd=[edge_kd] * len(edge_indices),
        )

        if add_springs:
            spring_indices = set()
            for i, j, k, l in edge_indices:
                spring_indices.add((min(k, l), max(k, l)))
                if i != -1:
                    spring_indices.add((min(i, k), max(i, k)))
                    spring_indices.add((min(i, l), max(i, l)))
                if j != -1:
                    spring_indices.add((min(j, k), max(j, k)))
                    spring_indices.add((min(j, l), max(j, l)))
                if i != -1 and j != -1:
                    spring_indices.add((min(i, j), max(i, j)))

            for i, j in spring_indices:
                self.add_spring(i, j, spring_ke, spring_kd, control=0.0)

    def add_particle_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        mass: float,
        jitter: float,
        radius_mean: float | None = None,
        radius_std: float = 0.0,
    ):
        radius_mean = radius_mean if radius_mean is not None else self.default_particle_radius

        rng = np.random.default_rng(42)
        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    p = wp.quat_rotate(rot, v) + pos + wp.vec3(rng.random(3) * jitter)

                    if radius_std > 0.0:
                        r = radius_mean + rng.standard_normal() * radius_std
                    else:
                        r = radius_mean
                    self.add_particle(p, vel, m, r)

    def add_soft_grid(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        fix_left: bool = False,
        fix_right: bool = False,
        fix_top: bool = False,
        fix_bottom: bool = False,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
    ):
        """Helper to create a rectangular tetrahedral FEM grid

        Creates a regular grid of FEM tetrahedra and surface triangles. Useful for example
        to create beams and sheets. Each hexahedral cell is decomposed into 5
        tetrahedral elements.

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            dim_x_: The number of rectangular cells along the x-axis
            dim_y: The number of rectangular cells along the y-axis
            dim_z: The number of rectangular cells along the z-axis
            cell_x: The width of each cell in the x-direction
            cell_y: The width of each cell in the y-direction
            cell_z: The width of each cell in the z-direction
            density: The density of each particle
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
            fix_left: Make the left-most edge of particles kinematic (fixed in place)
            fix_right: Make the right-most edge of particles kinematic
            fix_top: Make the top-most edge of particles kinematic
            fix_bottom: Make the bottom-most edge of particles kinematic
        """
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift

        start_vertex = len(self.particle_q)

        mass = cell_x * cell_y * cell_z * density

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    if fix_left and x == 0:
                        m = 0.0

                    if fix_right and x == dim_x:
                        m = 0.0

                    if fix_top and y == dim_y:
                        m = 0.0

                    if fix_bottom and y == 0:
                        m = 0.0

                    p = wp.quat_rotate(rot, v) + pos

                    self.add_particle(p, vel, m)

        # dict of open faces
        faces = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i: int, j: int, k: int, l: int):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v0 = grid_index(x, y, z) + start_vertex
                    v1 = grid_index(x + 1, y, z) + start_vertex
                    v2 = grid_index(x + 1, y, z + 1) + start_vertex
                    v3 = grid_index(x, y, z + 1) + start_vertex
                    v4 = grid_index(x, y + 1, z) + start_vertex
                    v5 = grid_index(x + 1, y + 1, z) + start_vertex
                    v6 = grid_index(x + 1, y + 1, z + 1) + start_vertex
                    v7 = grid_index(x, y + 1, z + 1) + start_vertex

                    if (x & 1) ^ (y & 1) ^ (z & 1):
                        add_tet(v0, v1, v4, v3)
                        add_tet(v2, v3, v6, v1)
                        add_tet(v5, v4, v1, v6)
                        add_tet(v7, v6, v3, v4)
                        add_tet(v4, v1, v6, v3)

                    else:
                        add_tet(v1, v2, v5, v0)
                        add_tet(v3, v0, v7, v2)
                        add_tet(v4, v7, v0, v5)
                        add_tet(v6, v5, v2, v7)
                        add_tet(v5, v2, v7, v0)

        # add triangles
        for _k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

    def add_soft_mesh(
        self,
        pos: Vec3,
        rot: Quat,
        scale: float,
        vel: Vec3,
        vertices: list[Vec3],
        indices: list[int],
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        tri_ke: float | None = None,
        tri_ka: float | None = None,
        tri_kd: float | None = None,
        tri_drag: float | None = None,
        tri_lift: float | None = None,
    ) -> None:
        """Helper to create a tetrahedral model from an input tetrahedral mesh

        Args:
            pos: The position of the solid in world space
            rot: The orientation of the solid in world space
            vel: The velocity of the solid in world space
            vertices: A list of vertex positions, array of 3D points
            indices: A list of tetrahedron indices, 4 entries per-element, flattened array
            density: The density per-area of the mesh
            k_mu: The first elastic Lame parameter
            k_lambda: The second elastic Lame parameter
            k_damp: The damping stiffness
        """
        tri_ke = tri_ke if tri_ke is not None else self.default_tri_ke
        tri_ka = tri_ka if tri_ka is not None else self.default_tri_ka
        tri_kd = tri_kd if tri_kd is not None else self.default_tri_kd
        tri_drag = tri_drag if tri_drag is not None else self.default_tri_drag
        tri_lift = tri_lift if tri_lift is not None else self.default_tri_lift

        num_tets = int(len(indices) / 4)

        start_vertex = len(self.particle_q)

        # dict of open faces
        faces = {}

        def add_face(i, j, k):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        pos = wp.vec3(pos[0], pos[1], pos[2])
        # add particles
        for v in vertices:
            p = wp.quat_rotate(rot, wp.vec3(v[0], v[1], v[2]) * scale) + pos

            self.add_particle(p, vel, 0.0)

        # add tetrahedra
        for t in range(num_tets):
            v0 = start_vertex + indices[t * 4 + 0]
            v1 = start_vertex + indices[t * 4 + 1]
            v2 = start_vertex + indices[t * 4 + 2]
            v3 = start_vertex + indices[t * 4 + 3]

            volume = self.add_tetrahedron(v0, v1, v2, v3, k_mu, k_lambda, k_damp)

            # distribute volume fraction to particles
            if volume > 0.0:
                self.particle_mass[v0] += density * volume / 4.0
                self.particle_mass[v1] += density * volume / 4.0
                self.particle_mass[v2] += density * volume / 4.0
                self.particle_mass[v3] += density * volume / 4.0

                # build open faces
                add_face(v0, v2, v1)
                add_face(v1, v2, v3)
                add_face(v0, v1, v3)
                add_face(v0, v3, v2)

        # add triangles
        for _k, v in faces.items():
            try:
                self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
            except np.linalg.LinAlgError:
                continue

    # incrementally updates rigid body mass with additional mass and inertia expressed at a local to the body
    def _update_body_mass(self, i, m, I, p, q):
        if i == -1:
            return

        # find new COM
        new_mass = self.body_mass[i] + m

        if new_mass == 0.0:  # no mass
            return

        new_com = (self.body_com[i] * self.body_mass[i] + p * m) / new_mass

        # shift inertia to new COM
        com_offset = new_com - self.body_com[i]
        shape_offset = new_com - p

        new_inertia = transform_inertia(
            self.body_mass[i], self.body_inertia[i], com_offset, wp.quat_identity()
        ) + transform_inertia(m, I, shape_offset, q)

        self.body_mass[i] = new_mass
        self.body_inertia[i] = new_inertia
        self.body_com[i] = new_com

        if new_mass > 0.0:
            self.body_inv_mass[i] = 1.0 / new_mass
        else:
            self.body_inv_mass[i] = 0.0

        if any(x for x in new_inertia):
            self.body_inv_inertia[i] = wp.inverse(new_inertia)
        else:
            self.body_inv_inertia[i] = new_inertia

    def set_coloring(self, particle_color_groups):
        """
        Sets coloring information with user-provided coloring.

        Args:
            particle_color_groups: A list of list or `np.array` with `dtype`=`int`. The length of the list is the number of colors
                and each list or `np.array` contains the indices of vertices with this color.
        """
        particle_color_groups = [
            color_group if isinstance(color_group, np.ndarray) else np.array(color_group)
            for color_group in particle_color_groups
        ]
        self.particle_color_groups = particle_color_groups

    def color(
        self,
        include_bending=False,
        balance_colors=True,
        target_max_min_color_ratio=1.1,
        coloring_algorithm=ColoringAlgorithm.MCS,
    ):
        """
        Runs coloring algorithm to generate coloring information.

        Args:
            include_bending_energy: Whether to consider bending energy for trimeshes in the coloring process. If set to `True`, the generated
                graph will contain all the edges connecting o1 and o2; otherwise, the graph will be equivalent to the trimesh.
            balance_colors: Whether to apply the color balancing algorithm to balance the size of each color
            target_max_min_color_ratio: the color balancing algorithm will stop when the ratio between the largest color and
                the smallest color reaches this value
            algorithm: Value should be an enum type of ColoringAlgorithm, otherwise it will raise an error. ColoringAlgorithm.mcs means using the MCS coloring algorithm,
                while ColoringAlgorithm.ordered_greedy means using the degree-ordered greedy algorithm. The MCS algorithm typically generates 30% to 50% fewer colors
                compared to the ordered greedy algorithm, while maintaining the same linear complexity. Although MCS has a constant overhead that makes it about twice
                as slow as the greedy algorithm, it produces significantly better coloring results. We recommend using MCS, especially if coloring is only part of the
                preprocessing.

        Note:

            References to the coloring algorithm:

            MCS: Pereira, F. M. Q., & Palsberg, J. (2005, November). Register allocation via coloring of chordal graphs. In Asian Symposium on Programming Languages and Systems (pp. 315-329). Berlin, Heidelberg: Springer Berlin Heidelberg.

            Ordered Greedy: Ton-That, Q. M., Kry, P. G., & Andrews, S. (2023). Parallel block Neo-Hookean XPBD using graph clustering. Computers & Graphics, 110, 1-10.

        """
        # ignore bending energy if it is too small
        edge_indices = np.array(self.edge_indices)

        self.particle_color_groups = color_trimesh(
            len(self.particle_q),
            edge_indices,
            include_bending,
            algorithm=coloring_algorithm,
            balance_colors=balance_colors,
            target_max_min_color_ratio=target_max_min_color_ratio,
        )

    def finalize(self, device: Devicelike | None = None, requires_grad: bool = False) -> Model:
        """Convert this builder object to a concrete model for simulation.

        After building simulation elements this method should be called to transfer
        all data to device memory ready for simulation.

        Args:
            device: The simulation device to use, e.g.: 'cpu', 'cuda'
            requires_grad: Whether to enable gradient computation for the model

        Returns:

            A model object.
        """
        from .collide import count_rigid_contact_points  # noqa: PLC0415

        # ensure the env count is set correctly
        self.num_envs = max(1, self.num_envs)

        # construct particle inv masses
        ms = np.array(self.particle_mass, dtype=np.float32)
        # static particles (with zero mass) have zero inverse mass
        particle_inv_mass = np.divide(1.0, ms, out=np.zeros_like(ms), where=ms != 0.0)

        with wp.ScopedDevice(device):
            # -------------------------------------
            # construct Model (non-time varying) data

            m = Model(device)
            m.requires_grad = requires_grad

            m.num_envs = self.num_envs

            # ---------------------
            # particles

            # state (initial)
            m.particle_q = wp.array(self.particle_q, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_qd = wp.array(self.particle_qd, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_mass = wp.array(self.particle_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_inv_mass = wp.array(particle_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_radius = wp.array(self.particle_radius, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_flags = wp.array([flag_to_int(f) for f in self.particle_flags], dtype=wp.uint32)
            m.particle_max_radius = np.max(self.particle_radius) if len(self.particle_radius) > 0 else 0.0
            m.particle_max_velocity = self.particle_max_velocity

            particle_colors = np.empty(self.particle_count, dtype=int)
            for color in range(len(self.particle_color_groups)):
                particle_colors[self.particle_color_groups[color]] = color
            m.particle_colors = wp.array(particle_colors, dtype=int)
            m.particle_color_groups = [wp.array(group, dtype=int) for group in self.particle_color_groups]

            # hash-grid for particle interactions
            m.particle_grid = wp.HashGrid(128, 128, 128)

            # ---------------------
            # collision geometry

            m.shape_key = self.shape_key
            m.shape_transform = wp.array(self.shape_transform, dtype=wp.transform, requires_grad=requires_grad)
            m.shape_body = wp.array(self.shape_body, dtype=wp.int32)
            m.shape_flags = wp.array(self.shape_flags, dtype=wp.uint32)
            m.body_shapes = self.body_shapes

            # build list of ids for geometry sources (meshes, sdfs)
            geo_sources = []
            finalized_meshes = {}  # do not duplicate meshes
            for geo in self.shape_geo_src:
                geo_hash = hash(geo)  # avoid repeated hash computations
                if geo:
                    if geo_hash not in finalized_meshes:
                        finalized_meshes[geo_hash] = geo.finalize(device=device)
                    geo_sources.append(finalized_meshes[geo_hash])
                else:
                    # add null pointer
                    geo_sources.append(0)

            m.shape_geo.type = wp.array(self.shape_geo_type, dtype=wp.int32)
            m.shape_geo.source = wp.array(geo_sources, dtype=wp.uint64)
            m.shape_geo.scale = wp.array(self.shape_geo_scale, dtype=wp.vec3, requires_grad=requires_grad)
            m.shape_geo.is_solid = wp.array(self.shape_geo_is_solid, dtype=wp.bool)
            m.shape_geo.thickness = wp.array(self.shape_geo_thickness, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_collision_radius = wp.array(
                self.shape_collision_radius, dtype=wp.float32, requires_grad=requires_grad
            )

            m.shape_geo_src = self.shape_geo_src  # used for rendering

            m.shape_materials.ke = wp.array(self.shape_material_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.kd = wp.array(self.shape_material_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.kf = wp.array(self.shape_material_kf, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.ka = wp.array(self.shape_material_ka, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.mu = wp.array(self.shape_material_mu, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.restitution = wp.array(
                self.shape_material_restitution, dtype=wp.float32, requires_grad=requires_grad
            )

            m.shape_collision_filter_pairs = self.shape_collision_filter_pairs
            m.shape_collision_group = self.shape_collision_group
            m.shape_collision_group_map = self.shape_collision_group_map

            # ---------------------
            # springs

            m.spring_indices = wp.array(self.spring_indices, dtype=wp.int32)
            m.spring_rest_length = wp.array(self.spring_rest_length, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_stiffness = wp.array(self.spring_stiffness, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_damping = wp.array(self.spring_damping, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_control = wp.array(self.spring_control, dtype=wp.float32, requires_grad=requires_grad)

            # ---------------------
            # triangles

            m.tri_indices = wp.array(self.tri_indices, dtype=wp.int32)
            m.tri_poses = wp.array(self.tri_poses, dtype=wp.mat22, requires_grad=requires_grad)
            m.tri_activations = wp.array(self.tri_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tri_materials = wp.array(self.tri_materials, dtype=wp.float32, requires_grad=requires_grad)
            m.tri_areas = wp.array(self.tri_areas, dtype=wp.float32, requires_grad=requires_grad)

            # ---------------------
            # edges

            m.edge_indices = wp.array(self.edge_indices, dtype=wp.int32)
            m.edge_rest_angle = wp.array(self.edge_rest_angle, dtype=wp.float32, requires_grad=requires_grad)
            m.edge_rest_length = wp.array(self.edge_rest_length, dtype=wp.float32, requires_grad=requires_grad)
            m.edge_bending_properties = wp.array(
                self.edge_bending_properties, dtype=wp.float32, requires_grad=requires_grad
            )

            # ---------------------
            # tetrahedra

            m.tet_indices = wp.array(self.tet_indices, dtype=wp.int32)
            m.tet_poses = wp.array(self.tet_poses, dtype=wp.mat33, requires_grad=requires_grad)
            m.tet_activations = wp.array(self.tet_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tet_materials = wp.array(self.tet_materials, dtype=wp.float32, requires_grad=requires_grad)

            # -----------------------
            # muscles

            # close the muscle waypoint indices
            muscle_start = copy.copy(self.muscle_start)
            muscle_start.append(len(self.muscle_bodies))

            m.muscle_start = wp.array(muscle_start, dtype=wp.int32)
            m.muscle_params = wp.array(self.muscle_params, dtype=wp.float32, requires_grad=requires_grad)
            m.muscle_bodies = wp.array(self.muscle_bodies, dtype=wp.int32)
            m.muscle_points = wp.array(self.muscle_points, dtype=wp.vec3, requires_grad=requires_grad)
            m.muscle_activations = wp.array(self.muscle_activations, dtype=wp.float32, requires_grad=requires_grad)

            # --------------------------------------
            # rigid bodies

            m.body_q = wp.array(self.body_q, dtype=wp.transform, requires_grad=requires_grad)
            m.body_qd = wp.array(self.body_qd, dtype=wp.spatial_vector, requires_grad=requires_grad)
            m.body_inertia = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
            m.body_inv_inertia = wp.array(self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad)
            m.body_mass = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.body_inv_mass = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.body_com = wp.array(self.body_com, dtype=wp.vec3, requires_grad=requires_grad)
            m.body_key = self.body_key

            # joints
            m.joint_type = wp.array(self.joint_type, dtype=wp.int32)
            m.joint_parent = wp.array(self.joint_parent, dtype=wp.int32)
            m.joint_child = wp.array(self.joint_child, dtype=wp.int32)
            m.joint_X_p = wp.array(self.joint_X_p, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_X_c = wp.array(self.joint_X_c, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_dof_dim = wp.array(np.array(self.joint_dof_dim), dtype=wp.int32, ndim=2)
            m.joint_axis = wp.array(self.joint_axis, dtype=wp.vec3, requires_grad=requires_grad)
            m.joint_q = wp.array(self.joint_q, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_qd = wp.array(self.joint_qd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_key = self.joint_key
            # compute joint ancestors
            child_to_joint = {}
            for i, child in enumerate(self.joint_child):
                child_to_joint[child] = i
            parent_joint = []
            for parent in self.joint_parent:
                parent_joint.append(child_to_joint.get(parent, -1))
            m.joint_ancestor = wp.array(parent_joint, dtype=wp.int32)

            # dynamics properties
            m.joint_armature = wp.array(self.joint_armature, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_ke = wp.array(self.joint_target_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_kd = wp.array(self.joint_target_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_dof_mode = wp.array(self.joint_dof_mode, dtype=wp.int32)
            m.joint_target = wp.array(self.joint_target, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_f = wp.array(self.joint_f, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_effort_limit = wp.array(self.joint_effort_limit, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_velocity_limit = wp.array(self.joint_velocity_limit, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_friction = wp.array(self.joint_friction, dtype=wp.float32, requires_grad=requires_grad)

            m.joint_limit_lower = wp.array(self.joint_limit_lower, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_upper = wp.array(self.joint_limit_upper, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_ke = wp.array(self.joint_limit_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_kd = wp.array(self.joint_limit_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_enabled = wp.array(self.joint_enabled, dtype=wp.int32)

            # 'close' the start index arrays with a sentinel value
            joint_q_start = copy.copy(self.joint_q_start)
            joint_q_start.append(self.joint_coord_count)
            joint_qd_start = copy.copy(self.joint_qd_start)
            joint_qd_start.append(self.joint_dof_count)
            articulation_start = copy.copy(self.articulation_start)
            articulation_start.append(self.joint_count)

            m.joint_q_start = wp.array(joint_q_start, dtype=wp.int32)
            m.joint_qd_start = wp.array(joint_qd_start, dtype=wp.int32)
            m.articulation_start = wp.array(articulation_start, dtype=wp.int32)
            m.articulation_key = self.articulation_key

            # counts
            m.joint_count = self.joint_count
            m.joint_dof_count = self.joint_dof_count
            m.joint_coord_count = self.joint_coord_count
            m.particle_count = len(self.particle_q)
            m.body_count = len(self.body_q)
            m.shape_count = len(self.shape_geo_type)
            m.tri_count = len(self.tri_poses)
            m.tet_count = len(self.tet_poses)
            m.edge_count = len(self.edge_rest_angle)
            m.spring_count = len(self.spring_rest_length)
            m.muscle_count = len(self.muscle_start)
            m.articulation_count = len(self.articulation_start)

            self.find_shape_contact_pairs(m)
            m.rigid_contact_max = count_rigid_contact_points(m)

            m.rigid_contact_torsional_friction = self.rigid_contact_torsional_friction
            m.rigid_contact_rolling_friction = self.rigid_contact_rolling_friction

            # enable ground plane
            m.gravity = np.array(self.up_vector, dtype=wp.float32) * self.gravity
            m.up_axis = self.up_axis
            m.up_vector = np.array(self.up_vector, dtype=wp.float32)

            return m

    def find_shape_contact_pairs(self, model: Model):
        # find potential contact pairs based on collision groups and collision mask (pairwise filtering)
        filters = copy.copy(self.shape_collision_filter_pairs)
        contact_pairs = []
        # iterate over collision groups (islands)
        for group, shapes in self.shape_collision_group_map.items():
            for s1, s2 in itertools.combinations(shapes, 2):
                if not (self.shape_flags[s1] & int(SHAPE_FLAG_COLLIDE_SHAPES)):
                    continue
                if not (self.shape_flags[s2] & int(SHAPE_FLAG_COLLIDE_SHAPES)):
                    continue

                # Ensure canonical order (smaller_element, larger_element)
                shape_a, shape_b = min(s1, s2), max(s1, s2)

                if (shape_a, shape_b) not in filters:
                    contact_pairs.append((shape_a, shape_b))
                    filters.add((shape_a, shape_b))
            if group != -1 and -1 in self.shape_collision_group_map:
                # shapes with collision group -1 collide with all other shapes
                for s1, s2 in itertools.product(shapes, self.shape_collision_group_map[-1]):
                    if not (self.shape_flags[s1] & int(SHAPE_FLAG_COLLIDE_SHAPES)):
                        continue
                    if not (self.shape_flags[s2] & int(SHAPE_FLAG_COLLIDE_SHAPES)):
                        continue
                    shape_a, shape_b = min(s1, s2), max(s1, s2)
                    if (shape_a, shape_b) not in filters:
                        contact_pairs.append((shape_a, shape_b))
                        filters.add((shape_a, shape_b))
        model.shape_contact_pairs = wp.array(np.array(contact_pairs), dtype=wp.vec2i, device=model.device)
        model.shape_contact_pair_count = len(contact_pairs)
