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

from .control import Control
from .state import State
from .types import ModelShapeGeometry, ModelShapeMaterials


class Model:
    """Holds the definition of the simulation model

    This class holds the non-time varying description of the system, i.e.:
    all geometry, constraints, and parameters used to describe the simulation.

    Attributes:
        requires_grad (float): Indicates whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled
        num_envs (int): Number of articulation environments that were added to the ModelBuilder via `add_builder`

        particle_q (array): Particle positions, shape [particle_count, 3], float
        particle_qd (array): Particle velocities, shape [particle_count, 3], float
        particle_mass (array): Particle mass, shape [particle_count], float
        particle_inv_mass (array): Particle inverse mass, shape [particle_count], float
        particle_radius (array): Particle radius, shape [particle_count], float
        particle_max_radius (float): Maximum particle radius (useful for HashGrid construction)
        particle_ke (array): Particle normal contact stiffness (used by :class:`SemiImplicitIntegrator`), shape [particle_count], float
        particle_kd (array): Particle normal contact damping (used by :class:`SemiImplicitIntegrator`), shape [particle_count], float
        particle_kf (array): Particle friction force stiffness (used by :class:`SemiImplicitIntegrator`), shape [particle_count], float
        particle_mu (array): Particle friction coefficient, shape [particle_count], float
        particle_cohesion (array): Particle cohesion strength, shape [particle_count], float
        particle_adhesion (array): Particle adhesion strength, shape [particle_count], float
        particle_grid (HashGrid): HashGrid instance used for accelerated simulation of particle interactions
        particle_flags (array): Particle enabled state, shape [particle_count], bool
        particle_max_velocity (float): Maximum particle velocity (to prevent instability)

        shape_transform (array): Rigid shape transforms, shape [shape_count, 7], float
        shape_flags (array): Rigid shape flags, shape [shape_count], uint32
        shape_body (array): Rigid shape body index, shape [shape_count], int
        body_shapes (dict): Mapping from body index to list of attached shape indices
        shape_materials (ModelShapeMaterials): Rigid shape contact materials, shape [shape_count], float
        shape_shape_geo (ModelShapeGeometry): Shape geometry properties (geo type, scale, thickness, etc.), shape [shape_count, 3], float
        shape_geo_src (list): List of `wp.Mesh` instances used for rendering of mesh geometry

        shape_collision_group (list): Collision group of each shape, shape [shape_count], int
        shape_collision_group_map (dict): Mapping from collision group to list of shape indices
        shape_collision_filter_pairs (set): Pairs of shape indices that should not collide
        shape_collision_radius (array): Collision radius of each shape used for bounding sphere broadphase collision checking, shape [shape_count], float
        shape_contact_pairs (array): Pairs of shape indices that may collide, shape [contact_pair_count, 2], int
        shape_ground_contact_pairs (array): Pairs of shape, ground indices that may collide, shape [ground_contact_pair_count, 2], int

        spring_indices (array): Particle spring indices, shape [spring_count*2], int
        spring_rest_length (array): Particle spring rest length, shape [spring_count], float
        spring_stiffness (array): Particle spring stiffness, shape [spring_count], float
        spring_damping (array): Particle spring damping, shape [spring_count], float
        spring_control (array): Particle spring activation, shape [spring_count], float

        tri_indices (array): Triangle element indices, shape [tri_count*3], int
        tri_poses (array): Triangle element rest pose, shape [tri_count, 2, 2], float
        tri_activations (array): Triangle element activations, shape [tri_count], float
        tri_materials (array): Triangle element materials, shape [tri_count, 5], float
        tri_areas (array): Triangle element rest areas, shape [tri_count], float

        edge_indices (array): Bending edge indices, shape [edge_count*4], int, each row is [o0, o1, v1, v2], where v1, v2 are on the edge
        edge_rest_angle (array): Bending edge rest angle, shape [edge_count], float
        edge_rest_length (array): Bending edge rest length, shape [edge_count], float
        edge_bending_properties (array): Bending edge stiffness and damping parameters, shape [edge_count, 2], float

        tet_indices (array): Tetrahedral element indices, shape [tet_count*4], int
        tet_poses (array): Tetrahedral rest poses, shape [tet_count, 3, 3], float
        tet_activations (array): Tetrahedral volumetric activations, shape [tet_count], float
        tet_materials (array): Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`, shape [tet_count, 3]

        muscle_start (array): Start index of the first muscle point per muscle, shape [muscle_count], int
        muscle_params (array): Muscle parameters, shape [muscle_count, 5], float
        muscle_bodies (array): Body indices of the muscle waypoints, int
        muscle_points (array): Local body offset of the muscle waypoints, float
        muscle_activations (array): Muscle activations, shape [muscle_count], float

        body_q (array): Poses of rigid bodies used for state initialization, shape [body_count, 7], float
        body_qd (array): Velocities of rigid bodies used for state initialization, shape [body_count, 6], float
        body_com (array): Rigid body center of mass (in local frame), shape [body_count, 7], float
        body_inertia (array): Rigid body inertia tensor (relative to COM), shape [body_count, 3, 3], float
        body_inv_inertia (array): Rigid body inverse inertia tensor (relative to COM), shape [body_count, 3, 3], float
        body_mass (array): Rigid body mass, shape [body_count], float
        body_inv_mass (array): Rigid body inverse mass, shape [body_count], float
        body_key (list): Rigid body keys, shape [body_count], str

        joint_q (array): Generalized joint positions used for state initialization, shape [joint_coord_count], float
        joint_qd (array): Generalized joint velocities used for state initialization, shape [joint_dof_count], float
        joint_act (array): Generalized joint control inputs, shape [joint_axis_count], float
        joint_type (array): Joint type, shape [joint_count], int
        joint_parent (array): Joint parent body indices, shape [joint_count], int
        joint_child (array): Joint child body indices, shape [joint_count], int
        joint_ancestor (array): Maps from joint index to the index of the joint that has the current joint parent body as child (-1 if no such joint ancestor exists), shape [joint_count], int
        joint_X_p (array): Joint transform in parent frame, shape [joint_count, 7], float
        joint_X_c (array): Joint mass frame in child frame, shape [joint_count, 7], float
        joint_axis (array): Joint axis in child frame, shape [joint_axis_count, 3], float
        joint_armature (array): Armature for each joint axis (only used by :class:`FeatherstoneIntegrator`), shape [joint_dof_count], float
        joint_target_ke (array): Joint stiffness, shape [joint_axis_count], float
        joint_target_kd (array): Joint damping, shape [joint_axis_count], float
        joint_axis_start (array): Start index of the first axis per joint, shape [joint_count], int
        joint_axis_dim (array): Number of linear and angular axes per joint, shape [joint_count, 2], int
        joint_axis_mode (array): Joint axis mode, shape [joint_axis_count], int.
        joint_linear_compliance (array): Joint linear compliance, shape [joint_count], float
        joint_angular_compliance (array): Joint linear compliance, shape [joint_count], float
        joint_enabled (array): Controls which joint is simulated (bodies become disconnected if False), shape [joint_count], int
        joint_limit_lower (array): Joint lower position limits, shape [joint_axis_count], float
        joint_limit_upper (array): Joint upper position limits, shape [joint_axis_count], float
        joint_limit_ke (array): Joint position limit stiffness (used by the Euler integrators), shape [joint_axis_count], float
        joint_limit_kd (array): Joint position limit damping (used by the Euler integrators), shape [joint_axis_count], float
        joint_twist_lower (array): Joint lower twist limit, shape [joint_count], float
        joint_twist_upper (array): Joint upper twist limit, shape [joint_count], float
        joint_q_start (array): Start index of the first position coordinate per joint (note the last value is an additional sentinel entry to allow for querying the q dimensionality of joint i via ``joint_q_start[i+1] - joint_q_start[i]``), shape [joint_count + 1], int
        joint_qd_start (array): Start index of the first velocity coordinate per joint (note the last value is an additional sentinel entry to allow for querying the qd dimensionality of joint i via ``joint_qd_start[i+1] - joint_qd_start[i]``), shape [joint_count + 1], int
        articulation_start (array): Articulation start index, shape [articulation_count], int
        joint_key (list): Joint keys, shape [joint_count], str

        soft_contact_radius (float): Contact radius used for self-collisions in the VBD integrator.
        soft_contact_margin (float): Contact margin for generation of soft contacts
        soft_contact_ke (float): Stiffness of soft contacts (used by the Euler integrators)
        soft_contact_kd (float): Damping of soft contacts (used by the Euler integrators)
        soft_contact_kf (float): Stiffness of friction force in soft contacts (used by the Euler integrators)
        soft_contact_mu (float): Friction coefficient of soft contacts
        soft_contact_restitution (float): Restitution coefficient of soft contacts (used by :class:`XPBDSolver`)

        soft_contact_count (array): Number of active particle-shape contacts, shape [1], int
        soft_contact_particle (array), Index of particle per soft contact point, shape [soft_contact_max], int
        soft_contact_shape (array), Index of shape per soft contact point, shape [soft_contact_max], int
        soft_contact_body_pos (array), Positional offset of soft contact point in body frame, shape [soft_contact_max], vec3
        soft_contact_body_vel (array), Linear velocity of soft contact point in body frame, shape [soft_contact_max], vec3
        soft_contact_normal (array), Contact surface normal of soft contact point in world space, shape [soft_contact_max], vec3
        soft_contact_tids (array), Thread indices of the soft contact points, shape [soft_contact_max], int

        rigid_contact_max (int): Maximum number of potential rigid body contact points to generate ignoring the `rigid_mesh_contact_max` limit.
        rigid_contact_max_limited (int): Maximum number of potential rigid body contact points to generate respecting the `rigid_mesh_contact_max` limit.
        rigid_mesh_contact_max (int): Maximum number of rigid body contact points to generate per mesh (0 = unlimited, default)
        rigid_contact_margin (float): Contact margin for generation of rigid body contacts
        rigid_contact_torsional_friction (float): Torsional friction coefficient for rigid body contacts (used by :class:`XPBDSolver`)
        rigid_contact_rolling_friction (float): Rolling friction coefficient for rigid body contacts (used by :class:`XPBDSolver`)

        rigid_contact_count (array): Number of active shape-shape contacts, shape [1], int
        rigid_contact_point0 (array): Contact point relative to frame of body 0, shape [rigid_contact_max], vec3
        rigid_contact_point1 (array): Contact point relative to frame of body 1, shape [rigid_contact_max], vec3
        rigid_contact_offset0 (array): Contact offset due to contact thickness relative to body 0, shape [rigid_contact_max], vec3
        rigid_contact_offset1 (array): Contact offset due to contact thickness relative to body 1, shape [rigid_contact_max], vec3
        rigid_contact_normal (array): Contact normal in world space, shape [rigid_contact_max], vec3
        rigid_contact_thickness (array): Total contact thickness, shape [rigid_contact_max], float
        rigid_contact_shape0 (array): Index of shape 0 per contact, shape [rigid_contact_max], int
        rigid_contact_shape1 (array): Index of shape 1 per contact, shape [rigid_contact_max], int
        rigid_contact_tids (array): Triangle indices of the contact points, shape [rigid_contact_max], int
        rigid_contact_pairwise_counter (array): Pairwise counter for contact generation, shape [rigid_contact_max], int
        rigid_contact_broad_shape0 (array): Broadphase shape index of shape 0 per contact, shape [rigid_contact_max], int
        rigid_contact_broad_shape1 (array): Broadphase shape index of shape 1 per contact, shape [rigid_contact_max], int
        rigid_contact_point_id (array): Contact point ID, shape [rigid_contact_max], int
        rigid_contact_point_limit (array): Contact point limit, shape [rigid_contact_max], int

        ground (bool): Whether the ground plane and ground contacts are enabled
        ground_plane (array): Ground plane 3D normal and offset, shape [4], float
        up_vector (np.ndarray): Up vector of the world, shape [3], float
        up_axis (int): Up axis, 0 for x, 1 for y, 2 for z
        gravity (np.ndarray): Gravity vector, shape [3], float

        particle_count (int): Total number of particles in the system
        body_count (int): Total number of bodies in the system
        shape_count (int): Total number of shapes in the system
        joint_count (int): Total number of joints in the system
        tri_count (int): Total number of triangles in the system
        tet_count (int): Total number of tetrahedra in the system
        edge_count (int): Total number of edges in the system
        spring_count (int): Total number of springs in the system
        contact_count (int): Total number of contacts in the system
        muscle_count (int): Total number of muscles in the system
        articulation_count (int): Total number of articulations in the system
        joint_dof_count (int): Total number of velocity degrees of freedom of all joints in the system
        joint_coord_count (int): Total number of position degrees of freedom of all joints in the system

        particle_color_groups (list of array): The coloring of all the particles, used for VBD's Gauss-Seidel iteration. Each array contains indices of particles sharing the same color.
        particle_colors (array): Contains the color assignment for every particle

        device (wp.Device): Device on which the Model was allocated

    Note:
        It is strongly recommended to use the ModelBuilder to construct a simulation rather
        than creating your own Model object directly, however it is possible to do so if
        desired.
    """

    def __init__(self, device=None):
        self.requires_grad = False
        self.num_envs = 0

        self.particle_q = None
        self.particle_qd = None
        self.particle_mass = None
        self.particle_inv_mass = None
        self.particle_radius = None
        self.particle_max_radius = 0.0
        self.particle_ke = 1.0e3
        self.particle_kd = 1.0e2
        self.particle_kf = 1.0e2
        self.particle_mu = 0.5
        self.particle_cohesion = 0.0
        self.particle_adhesion = 0.0
        self.particle_grid = None
        self.particle_flags = None
        self.particle_max_velocity = 1e5

        self.shape_key = []
        self.shape_transform = None
        self.shape_body = None
        self.shape_flags = None
        self.body_shapes = {}
        self.shape_materials = ModelShapeMaterials()
        self.shape_geo = ModelShapeGeometry()
        self.shape_geo_src = []
        self.geo_meshes = []
        self.geo_sdfs = []
        self.ground_plane_params = {}

        self.shape_collision_group = []
        self.shape_collision_group_map = {}
        self.shape_collision_filter_pairs = set()
        self.shape_collision_radius = None
        self.shape_contact_pairs = None
        self.shape_ground_contact_pairs = None
        self.shape_contact_pair_count = 0
        self.shape_ground_contact_pair_count = 0

        self.spring_indices = None
        self.spring_rest_length = None
        self.spring_stiffness = None
        self.spring_damping = None
        self.spring_control = None
        self.spring_constraint_lambdas = None

        self.tri_indices = None
        self.tri_poses = None
        self.tri_activations = None
        self.tri_materials = None
        self.tri_areas = None

        self.edge_indices = None
        self.edge_rest_angle = None
        self.edge_rest_length = None
        self.edge_bending_properties = None
        self.edge_constraint_lambdas = None

        self.tet_indices = None
        self.tet_poses = None
        self.tet_activations = None
        self.tet_materials = None

        self.muscle_start = None
        self.muscle_params = None
        self.muscle_bodies = None
        self.muscle_points = None
        self.muscle_activations = None

        self.body_q = None
        self.body_qd = None
        self.body_com = None
        self.body_inertia = None
        self.body_inv_inertia = None
        self.body_mass = None
        self.body_inv_mass = None
        self.body_key = []
        
        self._joint_q = None
        self._joint_q_dirty = False
        self.joint_qd = None
        self.joint_act = None
        self.joint_type = None
        self.joint_parent = None
        self.joint_child = None
        self.joint_ancestor = None
        self.joint_X_p = None
        self.joint_X_c = None
        self.joint_axis = None
        self.joint_armature = None
        self.joint_target_ke = None
        self.joint_target_kd = None
        self.joint_axis_start = None
        self.joint_axis_dim = None
        self.joint_axis_mode = None
        self.joint_linear_compliance = None
        self.joint_angular_compliance = None
        self.joint_enabled = None
        self.joint_limit_lower = None
        self.joint_limit_upper = None
        self.joint_limit_ke = None
        self.joint_limit_kd = None
        self.joint_twist_lower = None
        self.joint_twist_upper = None
        self.joint_q_start = None
        self.joint_qd_start = None
        self.joint_key = []
        self.articulation_start = None
        self.articulation_key = []

        self.soft_contact_radius = 0.2
        self.soft_contact_margin = 0.2
        self.soft_contact_ke = 1.0e3
        self.soft_contact_kd = 10.0
        self.soft_contact_kf = 1.0e3
        self.soft_contact_mu = 0.5
        self.soft_contact_restitution = 0.0

        self.soft_contact_count = 0
        self.soft_contact_particle = None
        self.soft_contact_shape = None
        self.soft_contact_body_pos = None
        self.soft_contact_body_vel = None
        self.soft_contact_normal = None
        self.soft_contact_tids = None

        self.rigid_contact_max = 0
        self.rigid_contact_max_limited = 0
        self.rigid_mesh_contact_max = 0
        self.rigid_contact_margin = 0.0
        self.rigid_contact_torsional_friction = 0.0
        self.rigid_contact_rolling_friction = 0.0
        self.enable_tri_collisions = False
        self.contact_particle_pairs = None
        self.contact_shape_pairs = None

        self.rigid_contact_count = None
        self.rigid_contact_point0 = None
        self.rigid_contact_point1 = None
        self.rigid_contact_offset0 = None
        self.rigid_contact_offset1 = None
        self.rigid_contact_normal = None
        self.rigid_contact_thickness = None
        self.rigid_contact_shape0 = None
        self.rigid_contact_shape1 = None
        self.rigid_contact_tids = None
        self.rigid_contact_pairwise_counter = None
        self.rigid_contact_broad_shape0 = None
        self.rigid_contact_broad_shape1 = None
        self.rigid_contact_point_id = None
        self.rigid_contact_point_limit = None

        # toggles ground contact for all shapes
        self.ground = True
        self.ground_plane = None
        self.up_vector = np.array((0.0, 1.0, 0.0))
        self.up_axis = 1
        self.gravity = np.array((0.0, -9.80665, 0.0))

        self.particle_count = 0
        self.body_count = 0
        self.shape_count = 0
        self.joint_count = 0
        self.joint_axis_count = 0
        self.tri_count = 0
        self.tet_count = 0
        self.edge_count = 0
        self.spring_count = 0
        self.muscle_count = 0
        self.articulation_count = 0
        self.joint_dof_count = 0
        self.joint_coord_count = 0

        # indices of particles sharing the same color
        self.particle_color_groups = []
        # the color of each particles
        self.particle_colors = None

        self.device = wp.get_device(device)

    @property
    def joint_q(self):
        return self._joint_q

    @joint_q.setter
    def joint_q(self, value):
        self._joint_q = value
        self._joint_q_dirty = True

    def state(self, requires_grad=None) -> State:
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

    def control(self, requires_grad=None, clone_variables=True) -> Control:
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
                c.joint_act = wp.clone(self.joint_act, requires_grad=requires_grad)
            if self.tri_count:
                c.tri_activations = wp.clone(self.tri_activations, requires_grad=requires_grad)
            if self.tet_count:
                c.tet_activations = wp.clone(self.tet_activations, requires_grad=requires_grad)
            if self.muscle_count:
                c.muscle_activations = wp.clone(self.muscle_activations, requires_grad=requires_grad)
        else:
            c.joint_act = self.joint_act
            c.tri_activations = self.tri_activations
            c.tet_activations = self.tet_activations
            c.muscle_activations = self.muscle_activations
        return c

    def _allocate_soft_contacts(self, target, count, requires_grad=False):
        with wp.ScopedDevice(self.device):
            target.soft_contact_count = wp.zeros(1, dtype=wp.int32)
            target.soft_contact_particle = wp.zeros(count, dtype=int)
            target.soft_contact_shape = wp.zeros(count, dtype=int)
            target.soft_contact_body_pos = wp.zeros(count, dtype=wp.vec3, requires_grad=requires_grad)
            target.soft_contact_body_vel = wp.zeros(count, dtype=wp.vec3, requires_grad=requires_grad)
            target.soft_contact_normal = wp.zeros(count, dtype=wp.vec3, requires_grad=requires_grad)
            target.soft_contact_tids = wp.zeros(self.particle_count * (self.shape_count - 1), dtype=int)

    def allocate_soft_contacts(self, count, requires_grad=False):
        self._allocate_soft_contacts(self, count, requires_grad)

    def count_contact_points(self):
        """
        Counts the maximum number of rigid contact points that need to be allocated.
        This function returns two values corresponding to the maximum number of potential contacts
        excluding the limiting from `Model.rigid_mesh_contact_max` and the maximum number of
        contact points that may be generated when considering the `Model.rigid_mesh_contact_max` limit.

        :returns:
            - potential_count (int): Potential number of contact points
            - actual_count (int): Actual number of contact points
        """
        from newton.collision.collide import count_contact_points

        # calculate the potential number of shape pair contact points
        contact_count = wp.zeros(2, dtype=wp.int32, device=self.device)
        wp.launch(
            kernel=count_contact_points,
            dim=self.shape_contact_pair_count,
            inputs=[
                self.shape_contact_pairs,
                self.shape_geo,
                self.rigid_mesh_contact_max,
            ],
            outputs=[contact_count],
            device=self.device,
            record_tape=False,
        )
        # count ground contacts
        wp.launch(
            kernel=count_contact_points,
            dim=self.shape_ground_contact_pair_count,
            inputs=[
                self.shape_ground_contact_pairs,
                self.shape_geo,
                self.rigid_mesh_contact_max,
            ],
            outputs=[contact_count],
            device=self.device,
            record_tape=False,
        )
        counts = contact_count.numpy()
        potential_count = int(counts[0])
        actual_count = int(counts[1])
        return potential_count, actual_count

    def allocate_rigid_contacts(self, target=None, count=None, limited_contact_count=None, requires_grad=False):
        if count is not None:
            # potential number of contact points to consider
            self.rigid_contact_max = count
        if limited_contact_count is not None:
            self.rigid_contact_max_limited = limited_contact_count
        if target is None:
            target = self

        with wp.ScopedDevice(self.device):
            # serves as counter of the number of active contact points
            target.rigid_contact_count = wp.zeros(1, dtype=wp.int32)
            # contact point ID within the (shape_a, shape_b) contact pair
            target.rigid_contact_point_id = wp.zeros(self.rigid_contact_max, dtype=wp.int32)
            # position of contact point in body 0's frame before the integration step
            target.rigid_contact_point0 = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # position of contact point in body 1's frame before the integration step
            target.rigid_contact_point1 = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # moment arm before the integration step resulting from thickness displacement added to contact point 0 in body 0's frame (used in XPBD contact friction handling)
            target.rigid_contact_offset0 = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # moment arm before the integration step resulting from thickness displacement added to contact point 1 in body 1's frame (used in XPBD contact friction handling)
            target.rigid_contact_offset1 = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # contact normal in world frame
            target.rigid_contact_normal = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.vec3, requires_grad=requires_grad
            )
            # combined thickness of both shapes
            target.rigid_contact_thickness = wp.zeros(
                self.rigid_contact_max_limited, dtype=wp.float32, requires_grad=requires_grad
            )
            # ID of the first shape in the contact pair
            target.rigid_contact_shape0 = wp.zeros(self.rigid_contact_max_limited, dtype=wp.int32)
            # ID of the second shape in the contact pair
            target.rigid_contact_shape1 = wp.zeros(self.rigid_contact_max_limited, dtype=wp.int32)

            # shape IDs of potential contact pairs found during broadphase
            target.rigid_contact_broad_shape0 = wp.zeros(self.rigid_contact_max, dtype=wp.int32)
            target.rigid_contact_broad_shape1 = wp.zeros(self.rigid_contact_max, dtype=wp.int32)

            if self.rigid_mesh_contact_max > 0:
                # add additional buffers to track how many contact points are generated per contact pair
                # (significantly increases memory usage, only enable if mesh contacts need to be pruned)
                if self.shape_count >= 46340:
                    # clip the number of potential contacts to avoid signed 32-bit integer overflow
                    # i.e. when the number of shapes exceeds sqrt(2**31 - 1)
                    max_pair_count = 2**31 - 1
                else:
                    max_pair_count = self.shape_count * self.shape_count
                # maximum number of contact points per contact pair
                target.rigid_contact_point_limit = wp.zeros(max_pair_count, dtype=wp.int32)
                # currently found contacts per contact pair
                target.rigid_contact_pairwise_counter = wp.zeros(max_pair_count, dtype=wp.int32)
            else:
                target.rigid_contact_point_limit = None
                target.rigid_contact_pairwise_counter = None

            # ID of thread that found the current contact point
            target.rigid_contact_tids = wp.zeros(self.rigid_contact_max, dtype=wp.int32)

    @property
    def soft_contact_max(self):
        """Maximum number of soft contacts that can be registered"""
        if self.soft_contact_particle is None:
            return 0
        return len(self.soft_contact_particle)
