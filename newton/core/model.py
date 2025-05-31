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

"""Implementation of the Newton model class."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton.geometry import CollisionPipeline

from .control import Control
from .state import State
from .types import Devicelike, ShapeGeometry, ShapeMaterials


class Model:
    """Holds the definition of the simulation model

    This class holds the non-time varying description of the system, i.e.:
    all geometry, constraints, and parameters used to describe the simulation.

    Note:
        It is strongly recommended to use the ModelBuilder to construct a
        simulation rather than creating your own Model object directly,
        however it is possible to do so if desired.
    """

    def __init__(self, device: Devicelike | None = None):
        """
        Initializes the Model object.

        Args:
            device (wp.Device): Device on which the Model's data will be allocated.
        """
        self.requires_grad = False
        """Indicates whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""
        self.num_envs = 0
        """Number of articulation environments that were added to the ModelBuilder via `add_builder`."""

        self.particle_q = None
        """Particle positions, shape [particle_count, 3], float."""
        self.particle_qd = None
        """Particle velocities, shape [particle_count, 3], float."""
        self.particle_mass = None
        """Particle mass, shape [particle_count], float."""
        self.particle_inv_mass = None
        """Particle inverse mass, shape [particle_count], float."""
        self.particle_radius = None
        """Particle radius, shape [particle_count], float."""
        self.particle_max_radius = 0.0
        """Maximum particle radius (useful for HashGrid construction)."""
        self.particle_ke = 1.0e3
        """Particle normal contact stiffness (used by :class:`~newton.solvers.SemiImplicitSolver`). Default is 1.0e3."""
        self.particle_kd = 1.0e2
        """Particle normal contact damping (used by :class:`~newton.solvers.SemiImplicitSolver`). Default is 1.0e2."""
        self.particle_kf = 1.0e2
        """Particle friction force stiffness (used by :class:`~newton.solvers.SemiImplicitSolver`). Default is 1.0e2."""
        self.particle_mu = 0.5
        """Particle friction coefficient. Default is 0.5."""
        self.particle_cohesion = 0.0
        """Particle cohesion strength. Default is 0.0."""
        self.particle_adhesion = 0.0
        """Particle adhesion strength. Default is 0.0."""
        self.particle_grid = None
        """HashGrid instance used for accelerated simulation of particle interactions."""
        self.particle_flags = None
        """Particle enabled state, shape [particle_count], bool."""
        self.particle_max_velocity = 1e5
        """Maximum particle velocity (to prevent instability). Default is 1e5."""

        self.shape_key = []
        """List of keys for each shape."""
        self.shape_transform = None
        """Rigid shape transforms, shape [shape_count, 7], float."""
        self.shape_body = None
        """Rigid shape body index, shape [shape_count], int."""
        self.shape_flags = None
        """Rigid shape flags, shape [shape_count], uint32."""
        self.body_shapes = {}
        """Mapping from body index to list of attached shape indices."""
        self.shape_materials = ShapeMaterials()
        """Rigid shape contact materials."""
        self.shape_geo = ShapeGeometry()
        """Shape geometry properties (geo type, scale, thickness, etc.)."""
        self.shape_geo_src = []
        """List of source geometry objects (e.g., `wp.Mesh`, `SDF`) used for rendering and broadphase."""
        self.geo_meshes = []
        """List of finalized `wp.Mesh` objects."""
        self.geo_sdfs = []
        """List of finalized `SDF` objects."""

        self.shape_collision_group = []
        """Collision group of each shape, shape [shape_count], int."""
        self.shape_collision_group_map = {}
        """Mapping from collision group to list of shape indices."""
        self.shape_collision_filter_pairs = set()
        """Pairs of shape indices that should not collide."""
        self.shape_collision_radius = None
        """Collision radius of each shape used for bounding sphere broadphase collision checking, shape [shape_count], float."""
        self.shape_contact_pairs = None
        """Pairs of shape indices that may collide, shape [contact_pair_count, 2], int."""
        self.shape_contact_pair_count = 0
        """Number of shape contact pairs."""

        self.spring_indices = None
        """Particle spring indices, shape [spring_count*2], int."""
        self.spring_rest_length = None
        """Particle spring rest length, shape [spring_count], float."""
        self.spring_stiffness = None
        """Particle spring stiffness, shape [spring_count], float."""
        self.spring_damping = None
        """Particle spring damping, shape [spring_count], float."""
        self.spring_control = None
        """Particle spring activation, shape [spring_count], float."""
        self.spring_constraint_lambdas = None
        """Lagrange multipliers for spring constraints (internal use)."""

        self.tri_indices = None
        """Triangle element indices, shape [tri_count*3], int."""
        self.tri_poses = None
        """Triangle element rest pose, shape [tri_count, 2, 2], float."""
        self.tri_activations = None
        """Triangle element activations, shape [tri_count], float."""
        self.tri_materials = None
        """Triangle element materials, shape [tri_count, 5], float."""
        self.tri_areas = None
        """Triangle element rest areas, shape [tri_count], float."""

        self.edge_indices = None
        """Bending edge indices, shape [edge_count*4], int, each row is [o0, o1, v1, v2], where v1, v2 are on the edge."""
        self.edge_rest_angle = None
        """Bending edge rest angle, shape [edge_count], float."""
        self.edge_rest_length = None
        """Bending edge rest length, shape [edge_count], float."""
        self.edge_bending_properties = None
        """Bending edge stiffness and damping parameters, shape [edge_count, 2], float."""
        self.edge_constraint_lambdas = None
        """Lagrange multipliers for edge constraints (internal use)."""

        self.tet_indices = None
        """Tetrahedral element indices, shape [tet_count*4], int."""
        self.tet_poses = None
        """Tetrahedral rest poses, shape [tet_count, 3, 3], float."""
        self.tet_activations = None
        """Tetrahedral volumetric activations, shape [tet_count], float."""
        self.tet_materials = None
        """Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`, shape [tet_count, 3]."""

        self.muscle_start = None
        """Start index of the first muscle point per muscle, shape [muscle_count], int."""
        self.muscle_params = None
        """Muscle parameters, shape [muscle_count, 5], float."""
        self.muscle_bodies = None
        """Body indices of the muscle waypoints, int."""
        self.muscle_points = None
        """Local body offset of the muscle waypoints, float."""
        self.muscle_activations = None
        """Muscle activations, shape [muscle_count], float."""

        self.body_q = None
        """Poses of rigid bodies used for state initialization, shape [body_count, 7], float."""
        self.body_qd = None
        """Velocities of rigid bodies used for state initialization, shape [body_count, 6], float."""
        self.body_com = None
        """Rigid body center of mass (in local frame), shape [body_count, 3], float."""
        self.body_inertia = None
        """Rigid body inertia tensor (relative to COM), shape [body_count, 3, 3], float."""
        self.body_inv_inertia = None
        """Rigid body inverse inertia tensor (relative to COM), shape [body_count, 3, 3], float."""
        self.body_mass = None
        """Rigid body mass, shape [body_count], float."""
        self.body_inv_mass = None
        """Rigid body inverse mass, shape [body_count], float."""
        self.body_key = []
        """Rigid body keys, shape [body_count], str."""

        self.joint_q = None
        """Generalized joint positions used for state initialization, shape [joint_coord_count], float."""
        self.joint_qd = None
        """Generalized joint velocities used for state initialization, shape [joint_dof_count], float."""
        self.joint_f = None
        """Generalized joint forces used for state initialization, shape [joint_dof_count], float."""
        self.joint_target = None
        """Generalized joint target inputs, shape [joint_axis_count], float."""
        self.joint_type = None
        """Joint type, shape [joint_count], int."""
        self.joint_parent = None
        """Joint parent body indices, shape [joint_count], int."""
        self.joint_child = None
        """Joint child body indices, shape [joint_count], int."""
        self.joint_ancestor = None
        """Maps from joint index to the index of the joint that has the current joint parent body as child (-1 if no such joint ancestor exists), shape [joint_count], int."""
        self.joint_X_p = None
        """Joint transform in parent frame, shape [joint_count, 7], float."""
        self.joint_X_c = None
        """Joint mass frame in child frame, shape [joint_count, 7], float."""
        self.joint_axis = None
        """Joint axis in child frame, shape [joint_axis_count, 3], float."""
        self.joint_armature = None
        """Armature for each joint axis (used by :class:`~newton.solvers.MuJoCoSolver` and :class:`~newton.solvers.FeatherstoneSolver`), shape [joint_dof_count], float."""
        self.joint_target_ke = None
        """Joint stiffness, shape [joint_axis_count], float."""
        self.joint_target_kd = None
        """Joint damping, shape [joint_axis_count], float."""
        self.joint_axis_start = None
        """Start index of the first axis per joint, shape [joint_count], int."""
        self.joint_axis_dim = None
        """Number of linear and angular axes per joint, shape [joint_count, 2], int."""
        self.joint_axis_mode = None
        """Joint axis mode, shape [joint_axis_count], int."""
        self.joint_enabled = None
        """Controls which joint is simulated (bodies become disconnected if False), shape [joint_count], int."""
        self.joint_limit_lower = None
        """Joint lower position limits, shape [joint_axis_count], float."""
        self.joint_limit_upper = None
        """Joint upper position limits, shape [joint_axis_count], float."""
        self.joint_limit_ke = None
        """Joint position limit stiffness (used by :class:`~newton.solvers.SemiImplicitSolver` and :class:`~newton.solvers.FeatherstoneSolver`), shape [joint_axis_count], float."""
        self.joint_limit_kd = None
        """Joint position limit damping (used by :class:`~newton.solvers.SemiImplicitSolver` and :class:`~newton.solvers.FeatherstoneSolver`), shape [joint_axis_count], float."""
        self.joint_twist_lower = None
        """Joint lower twist limit, shape [joint_count], float."""
        self.joint_twist_upper = None
        """Joint upper twist limit, shape [joint_count], float."""
        self.joint_q_start = None
        """Start index of the first position coordinate per joint (note the last value is an additional sentinel entry to allow for querying the q dimensionality of joint i via ``joint_q_start[i+1] - joint_q_start[i]``), shape [joint_count + 1], int."""
        self.joint_qd_start = None
        """Start index of the first velocity coordinate per joint (note the last value is an additional sentinel entry to allow for querying the qd dimensionality of joint i via ``joint_qd_start[i+1] - joint_qd_start[i]``), shape [joint_count + 1], int."""
        self.joint_key = []
        """Joint keys, shape [joint_count], str."""
        self.articulation_start = None
        """Articulation start index, shape [articulation_count], int."""
        self.articulation_key = []
        """Articulation keys, shape [articulation_count], str."""

        self.soft_contact_radius = 0.2
        """Contact radius used by :class:`~newton.solvers.VBDSolver` for self-collisions. Default is 0.2."""
        self.soft_contact_margin = 0.2
        """Contact margin for generation of soft contacts. Default is 0.2."""
        self.soft_contact_ke = 1.0e3
        """Stiffness of soft contacts (used by :class:`~newton.solvers.SemiImplicitSolver` and :class:`~newton.solvers.FeatherstoneSolver`). Default is 1.0e3."""
        self.soft_contact_kd = 10.0
        """Damping of soft contacts (used by :class:`~newton.solvers.SemiImplicitSolver` and :class:`~newton.solvers.FeatherstoneSolver`). Default is 10.0."""
        self.soft_contact_kf = 1.0e3
        """Stiffness of friction force in soft contacts (used by :class:`~newton.solvers.SemiImplicitSolver` and :class:`~newton.solvers.FeatherstoneSolver`). Default is 1.0e3."""
        self.soft_contact_mu = 0.5
        """Friction coefficient of soft contacts. Default is 0.5."""
        self.soft_contact_restitution = 0.0
        """Restitution coefficient of soft contacts (used by :class:`XPBDSolver`). Default is 0.0."""

        self.rigid_contact_torsional_friction = 0.0
        """Torsional friction coefficient for rigid body contacts (used by :class:`XPBDSolver`)."""
        self.rigid_contact_rolling_friction = 0.0
        """Rolling friction coefficient for rigid body contacts (used by :class:`XPBDSolver`)."""

        # toggles ground contact for all shapes
        self.up_vector = np.array((0.0, 0.0, 1.0))
        """Up vector of the world, shape [3], float."""
        self.up_axis = 2
        """Up axis, 0 for x, 1 for y, 2 for z."""
        self.gravity = np.array((0.0, 0.0, -9.81))
        """Gravity vector, shape [3], float."""

        self.particle_count = 0
        """Total number of particles in the system."""
        self.body_count = 0
        """Total number of bodies in the system."""
        self.shape_count = 0
        """Total number of shapes in the system."""
        self.joint_count = 0
        """Total number of joints in the system."""
        self.joint_axis_count = 0
        """Total number of joint axes in the system."""
        self.tri_count = 0
        """Total number of triangles in the system."""
        self.tet_count = 0
        """Total number of tetrahedra in the system."""
        self.edge_count = 0
        """Total number of edges in the system."""
        self.spring_count = 0
        """Total number of springs in the system."""
        self.muscle_count = 0
        """Total number of muscles in the system."""
        self.articulation_count = 0
        """Total number of articulations in the system."""
        self.joint_dof_count = 0
        """Total number of velocity degrees of freedom of all joints in the system."""
        self.joint_coord_count = 0
        """Total number of position degrees of freedom of all joints in the system."""

        # indices of particles sharing the same color
        self.particle_color_groups = []
        """The coloring of all the particles, used by :class:`~newton.solvers.VBDSolver` for Gauss-Seidel iteration. Each array contains indices of particles sharing the same color."""
        # the color of each particles
        self.particle_colors = None
        """Contains the color assignment for every particle."""

        self.device = wp.get_device(device)
        """Device on which the Model was allocated."""

    def state(self, requires_grad: bool | None = None) -> State:
        """Returns a state object for the model

        The returned state will be initialized with the initial configuration given in
        the model description.

        Args:
            requires_grad (bool): Manual overwrite whether the state variables should have `requires_grad` enabled (defaults to `None` to use the model's setting :attr:`requires_grad`)

        Returns:
            State: The state object
        """

        s = State()
        if requires_grad is None:
            requires_grad = self.requires_grad

        # particles
        if self.particle_count:
            s.particle_q = wp.clone(self.particle_q, requires_grad=requires_grad)
            s.particle_qd = wp.clone(self.particle_qd, requires_grad=requires_grad)
            s.particle_f = wp.zeros_like(self.particle_qd, requires_grad=requires_grad)

        # articulations
        if self.body_count:
            s.body_q = wp.clone(self.body_q, requires_grad=requires_grad)
            s.body_qd = wp.clone(self.body_qd, requires_grad=requires_grad)
            s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        if self.joint_count:
            s.joint_q = wp.clone(self.joint_q, requires_grad=requires_grad)
            s.joint_qd = wp.clone(self.joint_qd, requires_grad=requires_grad)

        return s

    def control(self, requires_grad: bool | None = None, clone_variables: bool = True) -> Control:
        """
        Returns a control object for the model.

        The returned control object will be initialized with the control inputs given in the model description.

        Args:
            requires_grad (bool): Manual overwrite whether the control variables should have `requires_grad` enabled (defaults to `None` to use the model's setting :attr:`requires_grad`)
            clone_variables (bool): Whether to clone the control inputs or use the original data

        Returns:
            Control: The control object
        """
        c = Control()
        if requires_grad is None:
            requires_grad = self.requires_grad
        if clone_variables:
            if self.joint_count:
                c.joint_target = wp.clone(self.joint_target, requires_grad=requires_grad)
                c.joint_f = wp.clone(self.joint_f, requires_grad=requires_grad)
            if self.tri_count:
                c.tri_activations = wp.clone(self.tri_activations, requires_grad=requires_grad)
            if self.tet_count:
                c.tet_activations = wp.clone(self.tet_activations, requires_grad=requires_grad)
            if self.muscle_count:
                c.muscle_activations = wp.clone(self.muscle_activations, requires_grad=requires_grad)
        else:
            c.joint_target = self.joint_target
            c.joint_f = self.joint_f
            c.tri_activations = self.tri_activations
            c.tet_activations = self.tet_activations
            c.muscle_activations = self.muscle_activations
        return c

    @wp.kernel
    def _compute_shape_world_transforms(
        shape_transform: wp.array(dtype=wp.transform),
        shape_body: wp.array(dtype=int),
        body_q: wp.array(dtype=wp.transform),
        # outputs
        shape_world_transform: wp.array(dtype=wp.transform),
    ):
        """Compute world-space transforms for shapes by concatenating local shape
        transforms with body transforms.

        Args:
            shape_transform: Local shape transforms in body frame,
                shape [shape_count, 7]
            shape_body: Body index for each shape, shape [shape_count]
            body_q: Body transforms in world frame, shape [body_count, 7]
            shape_world_transform: Output world transforms for shapes,
                shape [shape_count, 7]
        """
        shape_idx = wp.tid()

        # Get the local shape transform
        X_bs = shape_transform[shape_idx]

        # Get the body index for this shape
        body_idx = shape_body[shape_idx]

        # If shape is attached to a body (body_idx >= 0), concatenate transforms
        if body_idx >= 0:
            # Get the body transform in world space
            X_wb = body_q[body_idx]

            # Concatenate: world_transform = body_transform * shape_transform
            X_ws = wp.transform_multiply(X_wb, X_bs)
            shape_world_transform[shape_idx] = X_ws
        else:
            # Shape is not attached to a body (static shape), use local
            # transform as world transform
            shape_world_transform[shape_idx] = X_bs

    def collide(
        self: Model,
        state: State,
        requires_grad: bool | None = None,
        edge_sdf_iter: int = 10,
        iterate_mesh_vertices: bool = True,
        max_contacts_per_pair: int = 10,
        soft_contact_margin: float = 0.0,
        rigid_contact_margin: float = 0.0,
    ) -> Contacts:
        """Generate contact points for the particles and rigid bodies in the
        model for use in contact-dynamics kernels.

        Args:
            state: The state of the model.
            requires_grad: Whether to duplicate contact arrays for gradient
                computation (if ``None``, uses ``self.requires_grad``).
            edge_sdf_iter: Number of search iterations for finding closest
                contact points between edges and SDF.
            iterate_mesh_vertices: Whether to iterate over all vertices of a
                mesh for contact generation (used for capsule/box <> mesh
                collision).
            max_contacts_per_pair: Maximum number of contacts per shape pair.
            soft_contact_margin: Margin for soft contact generation.
            rigid_contact_margin: Margin for rigid contact generation.

        Returns:
            Contact: The contact object containing collision information.
        """
        if requires_grad is None:
            requires_grad = self.requires_grad

        if not hasattr(self, "_collision_pipeline"):
            self._collision_pipeline = CollisionPipeline(
                self.shape_count,
                self.shape_contact_pairs,
                max_contacts_per_pair,
                rigid_contact_margin,
                self.particle_count * self.shape_count,
                soft_contact_margin,
                edge_sdf_iter,
                iterate_mesh_vertices,
                requires_grad,
            )

        # update any additional parameters
        self._collision_pipeline.rigid_contact_margin = rigid_contact_margin
        self._collision_pipeline.soft_contact_margin = soft_contact_margin
        self._collision_pipeline.edge_sdf_iter = edge_sdf_iter
        self._collision_pipeline.iterate_mesh_vertices = iterate_mesh_vertices

        return self._collision_pipeline.collide(
            self.shape_geo.type,
            self.shape_geo.is_solid,
            self.shape_geo.thickness,
            self.shape_geo.source,
            self.shape_geo.scale,
            self.shape_geo.filter,
            self.shape_collision_radius,
            self.shape_body,
            self.shape_transform,
            state.body_q,
            state.particle_q,
            self.particle_radius,
            self.particle_flags,
        )
