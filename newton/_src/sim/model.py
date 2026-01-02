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

from enum import IntEnum

import numpy as np
import warp as wp

from ..core.types import Devicelike
from .contacts import Contacts
from .control import Control
from .state import State


class ModelAttributeAssignment(IntEnum):
    """Enumeration of attribute assignment categories.

    Defines which component of the simulation system owns and manages specific attributes.
    This categorization determines where custom attributes are attached during simulation
    object creation (Model, State, Control, or Contacts).
    """

    MODEL = 0
    """Model attributes are attached to the Model object."""
    STATE = 1
    """State attributes are attached to the State object."""
    CONTROL = 2
    """Control attributes are attached to the Control object."""
    CONTACT = 3
    """Contact attributes are attached to the Contacts object."""


class ModelAttributeFrequency(IntEnum):
    """Enumeration of attribute frequency categories.

    Defines the dimensional structure and indexing pattern for custom attributes.
    This determines how many elements an attribute array should have and how it
    should be indexed in relation to the model's entities such as joints, bodies, shapes, etc.
    """

    ONCE = 0
    """Attribute frequency is a single value."""
    JOINT = 1
    """Attribute frequency follows the number of joints (see :attr:`~newton.Model.joint_count`)."""
    JOINT_DOF = 2
    """Attribute frequency follows the number of joint degrees of freedom (see :attr:`~newton.Model.joint_dof_count`)."""
    JOINT_COORD = 3
    """Attribute frequency follows the number of joint positional coordinates (see :attr:`~newton.Model.joint_coord_count`)."""
    BODY = 4
    """Attribute frequency follows the number of bodies (see :attr:`~newton.Model.body_count`)."""
    SHAPE = 5
    """Attribute frequency follows the number of shapes (see :attr:`~newton.Model.shape_count`)."""
    ARTICULATION = 6
    """Attribute frequency follows the number of articulations (see :attr:`~newton.Model.articulation_count`)."""
    EQUALITY_CONSTRAINT = 7
    """Attribute frequency follows the number of equality constraints (see :attr:`~newton.Model.equality_constraint_count`)."""


class AttributeNamespace:
    """
    A container for namespaced custom attributes.

    Custom attributes are stored as regular instance attributes on this object,
    allowing hierarchical organization of related properties.
    """

    def __init__(self, name: str):
        """Initialize the namespace container.

        Args:
            name: The name of the namespace
        """
        self._name = name

    def __repr__(self):
        """Return a string representation showing the namespace and its attributes."""
        # List all public attributes (not starting with _)
        attrs = [k for k in self.__dict__ if not k.startswith("_")]
        return f"AttributeNamespace('{self._name}', attributes={attrs})"


class Model:
    """
    Represents the static (non-time-varying) definition of a simulation model in Newton.

    The Model class encapsulates all geometry, constraints, and parameters that describe a physical system
    for simulation. It is designed to be constructed via the ModelBuilder, which handles the correct
    initialization and population of all fields.

    Key Features:
        - Stores all static data for simulation: particles, rigid bodies, joints, shapes, soft/rigid elements, etc.
        - Supports grouping of entities by world using world indices (e.g., `particle_world`, `body_world`, etc.).
          - Index -1: global entities shared across all worlds.
          - Indices 0, 1, 2, ...: world-specific entities.
        - Grouping enables:
          - Collision detection optimization (e.g., separating worlds)
          - Visualization (e.g., spatially separating worlds)
          - Parallel processing of independent worlds

    Note:
        It is strongly recommended to use the :class:`ModelBuilder` to construct a Model.
        Direct instantiation and manual population of Model fields is possible but discouraged.
    """

    def __init__(self, device: Devicelike | None = None):
        """
        Initialize a Model object.

        Args:
            device (wp.Device, optional): Device on which the Model's data will be allocated.
        """
        self.requires_grad = False
        """Whether the model was finalized (see :meth:`ModelBuilder.finalize`) with gradient computation enabled."""
        self.num_worlds = 0
        """Number of worlds added to the ModelBuilder."""

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
        """Particle normal contact stiffness (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_kd = 1.0e2
        """Particle normal contact damping (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_kf = 1.0e2
        """Particle friction force stiffness (used by :class:`~newton.solvers.SolverSemiImplicit`)."""
        self.particle_mu = 0.5
        """Particle friction coefficient."""
        self.particle_cohesion = 0.0
        """Particle cohesion strength."""
        self.particle_adhesion = 0.0
        """Particle adhesion strength."""
        self.particle_grid = None
        """HashGrid instance for accelerated simulation of particle interactions."""
        self.particle_flags = None
        """Particle enabled state, shape [particle_count], int."""
        self.particle_max_velocity = 1e5
        """Maximum particle velocity (to prevent instability)."""
        self.particle_world = None
        """World index for each particle, shape [particle_count], int. -1 for global."""

        self.shape_key = []
        """List of keys for each shape."""
        self.shape_transform = None
        """Rigid shape transforms, shape [shape_count, 7], float."""
        self.shape_body = None
        """Rigid shape body index, shape [shape_count], int."""
        self.shape_flags = None
        """Rigid shape flags, shape [shape_count], int."""
        self.body_shapes = {}
        """Mapping from body index to list of attached shape indices."""

        # Shape material properties
        self.shape_material_ke = None
        """Shape contact elastic stiffness, shape [shape_count], float."""
        self.shape_material_kd = None
        """Shape contact damping stiffness, shape [shape_count], float."""
        self.shape_material_kf = None
        """Shape contact friction stiffness, shape [shape_count], float."""
        self.shape_material_ka = None
        """Shape contact adhesion distance, shape [shape_count], float."""
        self.shape_material_mu = None
        """Shape coefficient of friction, shape [shape_count], float."""
        self.shape_material_restitution = None
        """Shape coefficient of restitution, shape [shape_count], float."""
        self.shape_material_torsional_friction = None
        """Shape torsional friction coefficient (resistance to spinning at contact point), shape [shape_count], float."""
        self.shape_material_rolling_friction = None
        """Shape rolling friction coefficient (resistance to rolling motion), shape [shape_count], float."""
        self.shape_material_k_hydro = None
        """Shape hydroelastic stiffness coefficient, shape [shape_count], float."""
        self.shape_contact_margin = None
        """Shape contact margin for collision detection, shape [shape_count], float."""

        # Shape geometry properties
        self.shape_type = None
        """Shape geometry type, shape [shape_count], int32."""
        self.shape_is_solid = None
        """Whether shape is solid or hollow, shape [shape_count], bool."""
        self.shape_thickness = None
        """Shape thickness, shape [shape_count], float."""
        self.shape_source = []
        """List of source geometry objects (e.g., :class:`~newton.Mesh`, :class:`~newton.SDF`) used for rendering and broadphase, shape [shape_count]."""
        self.shape_source_ptr = None
        """Geometry source pointer to be used inside the Warp kernels which can be generated by finalizing the geometry objects, see for example :meth:`newton.Mesh.finalize`, shape [shape_count], uint64."""
        self.shape_scale = None
        """Shape 3D scale, shape [shape_count, 3], float."""
        self.shape_filter = None
        """Shape filter group, shape [shape_count], int."""

        self.shape_collision_group = None
        """Collision group of each shape, shape [shape_count], int. Array populated during finalization."""
        self.shape_collision_filter_pairs: set[tuple[int, int]] = set()
        """Pairs of shape indices that should not collide."""
        self.shape_collision_radius = None
        """Collision radius for bounding sphere broadphase, shape [shape_count], float."""
        self.shape_contact_pairs = None
        """Pairs of shape indices that may collide, shape [contact_pair_count, 2], int."""
        self.shape_contact_pair_count = 0
        """Number of shape contact pairs."""
        self.shape_world = None
        """World index for each shape, shape [shape_count], int. -1 for global."""

        # Mesh SDF storage
        self.shape_sdf_data = None
        """Array of SDFData structs for mesh shapes, shape [shape_count]. Contains sparse and coarse SDF pointers, extents, and voxel sizes. Empty array if there are no colliding meshes."""
        self.shape_sdf_volume = []
        """List of sparse SDF volume references for mesh shapes, shape [shape_count]. None for non-mesh shapes. Empty if there are no colliding meshes. Kept for reference counting."""
        self.shape_sdf_coarse_volume = []
        """List of coarse SDF volume references for mesh shapes, shape [shape_count]. None for non-mesh shapes. Empty if there are no colliding meshes. Kept for reference counting."""

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
        """Bending edge stiffness and damping, shape [edge_count, 2], float."""
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
        """Rigid body poses for state initialization, shape [body_count, 7], float."""
        self.body_qd = None
        """Rigid body velocities for state initialization, shape [body_count, 6], float."""
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
        self.body_world = None
        """World index for each body, shape [body_count], int. Global entities have index -1."""

        self.joint_q = None
        """Generalized joint positions for state initialization, shape [joint_coord_count], float."""
        self.joint_qd = None
        """Generalized joint velocities for state initialization, shape [joint_dof_count], float."""
        self.joint_f = None
        """Generalized joint forces for state initialization, shape [joint_dof_count], float."""
        self.joint_target_pos = None
        """Generalized joint position targets, shape [joint_dof_count], float."""
        self.joint_target_vel = None
        """Generalized joint velocity targets, shape [joint_dof_count], float."""
        self.joint_type = None
        """Joint type, shape [joint_count], int."""
        self.joint_articulation = None
        """Joint articulation index (-1 if not in any articulation), shape [joint_count], int."""
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
        """Joint axis in child frame, shape [joint_dof_count, 3], float."""
        self.joint_armature = None
        """Armature for each joint axis (used by :class:`~newton.solvers.SolverMuJoCo` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_target_ke = None
        """Joint stiffness, shape [joint_dof_count], float."""
        self.joint_target_kd = None
        """Joint damping, shape [joint_dof_count], float."""
        self.joint_effort_limit = None
        """Joint effort (force/torque) limits, shape [joint_dof_count], float."""
        self.joint_velocity_limit = None
        """Joint velocity limits, shape [joint_dof_count], float."""
        self.joint_friction = None
        """Joint friction coefficient, shape [joint_dof_count], float."""
        self.joint_dof_dim = None
        """Number of linear and angular dofs per joint, shape [joint_count, 2], int."""
        self.joint_enabled = None
        """Controls which joint is simulated (bodies become disconnected if False, only supported by :class:`~newton.solvers.SolverXPBD` and :class:`~newton.solvers.SolverSemiImplicit`), shape [joint_count], bool."""
        self.joint_limit_lower = None
        """Joint lower position limits, shape [joint_dof_count], float."""
        self.joint_limit_upper = None
        """Joint upper position limits, shape [joint_dof_count], float."""
        self.joint_limit_ke = None
        """Joint position limit stiffness (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_limit_kd = None
        """Joint position limit damping (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`), shape [joint_dof_count], float."""
        self.joint_twist_lower = None
        """Joint lower twist limit, shape [joint_count], float."""
        self.joint_twist_upper = None
        """Joint upper twist limit, shape [joint_count], float."""
        self.joint_q_start = None
        """Start index of the first position coordinate per joint (last value is a sentinel for dimension queries), shape [joint_count + 1], int."""
        self.joint_qd_start = None
        """Start index of the first velocity coordinate per joint (last value is a sentinel for dimension queries), shape [joint_count + 1], int."""
        self.joint_key = []
        """Joint keys, shape [joint_count], str."""
        self.joint_world = None
        """World index for each joint, shape [joint_count], int. -1 for global."""
        self.articulation_start = None
        """Articulation start index, shape [articulation_count], int."""
        self.articulation_key = []
        """Articulation keys, shape [articulation_count], str."""
        self.articulation_world = None
        """World index for each articulation, shape [articulation_count], int. -1 for global."""
        self.max_joints_per_articulation = 0
        """Maximum number of joints in any articulation (used for IK kernel dimensioning)."""

        self.soft_contact_ke = 1.0e3
        """Stiffness of soft contacts (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_kd = 10.0
        """Damping of soft contacts (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_kf = 1.0e3
        """Stiffness of friction force in soft contacts (used by :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverFeatherstone`)."""
        self.soft_contact_mu = 0.5
        """Friction coefficient of soft contacts."""
        self.soft_contact_restitution = 0.0
        """Restitution coefficient of soft contacts (used by :class:`SolverXPBD`)."""

        self.rigid_contact_max = 0
        """Number of potential contact points between rigid bodies."""

        self.up_vector = np.array((0.0, 0.0, 1.0))
        """Up vector of the world, shape [3], float."""
        self.up_axis = 2
        """Up axis: 0 for x, 1 for y, 2 for z."""
        self.gravity = None
        """Gravity vector, shape [1], dtype vec3."""

        self.equality_constraint_type = None
        """Type of equality constraint, shape [equality_constraint_count], int."""
        self.equality_constraint_body1 = None
        """First body index, shape [equality_constraint_count], int."""
        self.equality_constraint_body2 = None
        """Second body index, shape [equality_constraint_count], int."""
        self.equality_constraint_anchor = None
        """Anchor point on first body, shape [equality_constraint_count, 3], float."""
        self.equality_constraint_torquescale = None
        """Torque scale, shape [equality_constraint_count], float."""
        self.equality_constraint_relpose = None
        """Relative pose, shape [equality_constraint_count, 7], float."""
        self.equality_constraint_joint1 = None
        """First joint index, shape [equality_constraint_count], int."""
        self.equality_constraint_joint2 = None
        """Second joint index, shape [equality_constraint_count], int."""
        self.equality_constraint_polycoef = None
        """Polynomial coefficients, shape [equality_constraint_count, 2], float."""
        self.equality_constraint_key = []
        """Constraint name/key, shape [equality_constraint_count], str."""
        self.equality_constraint_enabled = None
        """Whether constraint is active, shape [equality_constraint_count], bool."""
        self.equality_constraint_world = None
        """World index for each constraint, shape [equality_constraint_count], int."""

        self.particle_count = 0
        """Total number of particles in the system."""
        self.body_count = 0
        """Total number of bodies in the system."""
        self.shape_count = 0
        """Total number of shapes in the system."""
        self.joint_count = 0
        """Total number of joints in the system."""
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
        """Total number of velocity degrees of freedom of all joints. Equals the number of joint axes."""
        self.joint_coord_count = 0
        """Total number of position degrees of freedom of all joints."""
        self.equality_constraint_count = 0
        """Total number of equality constraints in the system."""

        # indices of particles sharing the same color
        self.particle_color_groups = []
        """Coloring of all particles for Gauss-Seidel iteration (see :class:`~newton.solvers.SolverVBD`). Each array contains indices of particles sharing the same color."""
        self.particle_colors = None
        """Color assignment for every particle."""

        self.body_color_groups = []
        """Coloring of all rigid bodies for Gauss-Seidel iteration (see :class:`~newton.solvers.SolverVBD`). Each array contains indices of bodies sharing the same color."""
        self.body_colors = None
        """Color assignment for every rigid body."""

        self.device = wp.get_device(device)
        """Device on which the Model was allocated."""

        self.attribute_frequency = {}
        """Classifies each attribute using ModelAttributeFrequency enum values (per body, per joint, per DOF, etc.)."""

        self.attribute_assignment = {}
        """Assignment for custom attributes using ModelAttributeAssignment enum values.
        If an attribute is not in this dictionary, it is assumed to be a Model attribute (assignment=ModelAttributeAssignment.MODEL)."""

        # attributes per body
        self.attribute_frequency["body_q"] = ModelAttributeFrequency.BODY
        self.attribute_frequency["body_qd"] = ModelAttributeFrequency.BODY
        self.attribute_frequency["body_com"] = ModelAttributeFrequency.BODY
        self.attribute_frequency["body_inertia"] = ModelAttributeFrequency.BODY
        self.attribute_frequency["body_inv_inertia"] = ModelAttributeFrequency.BODY
        self.attribute_frequency["body_mass"] = ModelAttributeFrequency.BODY
        self.attribute_frequency["body_inv_mass"] = ModelAttributeFrequency.BODY
        self.attribute_frequency["body_f"] = ModelAttributeFrequency.BODY

        # attributes per joint
        self.attribute_frequency["joint_type"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_parent"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_child"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_ancestor"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_articulation"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_X_p"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_X_c"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_dof_dim"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_enabled"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_twist_lower"] = ModelAttributeFrequency.JOINT
        self.attribute_frequency["joint_twist_upper"] = ModelAttributeFrequency.JOINT

        # attributes per joint coord
        self.attribute_frequency["joint_q"] = ModelAttributeFrequency.JOINT_COORD

        # attributes per joint dof
        self.attribute_frequency["joint_qd"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_f"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_armature"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_target_pos"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_target_vel"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_axis"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_target_ke"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_target_kd"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_limit_lower"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_limit_upper"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_limit_ke"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_limit_kd"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_effort_limit"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_friction"] = ModelAttributeFrequency.JOINT_DOF
        self.attribute_frequency["joint_velocity_limit"] = ModelAttributeFrequency.JOINT_DOF

        # attributes per shape
        self.attribute_frequency["shape_transform"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_body"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_flags"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_ke"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_kd"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_kf"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_ka"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_mu"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_restitution"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_torsional_friction"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_rolling_friction"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_material_k_hydro"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_contact_margin"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_type"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_is_solid"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_thickness"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_source_ptr"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_scale"] = ModelAttributeFrequency.SHAPE
        self.attribute_frequency["shape_filter"] = ModelAttributeFrequency.SHAPE

    def state(self, requires_grad: bool | None = None) -> State:
        """
        Create and return a new :class:`State` object for this model.

        The returned state is initialized with the initial configuration from the model description.

        Args:
            requires_grad (bool, optional): Whether the state variables should have `requires_grad` enabled.
                If None, uses the model's :attr:`requires_grad` setting.

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

        # rigid bodies
        if self.body_count:
            s.body_q = wp.clone(self.body_q, requires_grad=requires_grad)
            s.body_qd = wp.clone(self.body_qd, requires_grad=requires_grad)
            s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)

        # joints
        if self.joint_count:
            s.joint_q = wp.clone(self.joint_q, requires_grad=requires_grad)
            s.joint_qd = wp.clone(self.joint_qd, requires_grad=requires_grad)

        # attach custom attributes with assignment==STATE
        self._add_custom_attributes(s, ModelAttributeAssignment.STATE, requires_grad=requires_grad)

        return s

    def control(self, requires_grad: bool | None = None, clone_variables: bool = True) -> Control:
        """
        Create and return a new :class:`Control` object for this model.

        The returned control object is initialized with the control inputs from the model description.

        Args:
            requires_grad (bool, optional): Whether the control variables should have `requires_grad` enabled.
                If None, uses the model's :attr:`requires_grad` setting.
            clone_variables (bool): If True, clone the control input arrays; if False, use references.

        Returns:
            Control: The initialized control object.
        """
        c = Control()
        if requires_grad is None:
            requires_grad = self.requires_grad
        if clone_variables:
            if self.joint_count:
                c.joint_target_pos = wp.clone(self.joint_target_pos, requires_grad=requires_grad)
                c.joint_target_vel = wp.clone(self.joint_target_vel, requires_grad=requires_grad)
                c.joint_f = wp.clone(self.joint_f, requires_grad=requires_grad)
            if self.tri_count:
                c.tri_activations = wp.clone(self.tri_activations, requires_grad=requires_grad)
            if self.tet_count:
                c.tet_activations = wp.clone(self.tet_activations, requires_grad=requires_grad)
            if self.muscle_count:
                c.muscle_activations = wp.clone(self.muscle_activations, requires_grad=requires_grad)
        else:
            c.joint_target_pos = self.joint_target_pos
            c.joint_target_vel = self.joint_target_vel
            c.joint_f = self.joint_f
            c.tri_activations = self.tri_activations
            c.tet_activations = self.tet_activations
            c.muscle_activations = self.muscle_activations
        # attach custom attributes with assignment==CONTROL
        self._add_custom_attributes(
            c, ModelAttributeAssignment.CONTROL, requires_grad=requires_grad, clone_arrays=clone_variables
        )
        return c

    def set_gravity(self, gravity: tuple[float, float, float] | list[float] | wp.vec3) -> None:
        """
        Set gravity for runtime modification.

        Args:
            gravity: Gravity vector as a tuple, list, or wp.vec3.
                    Common values: (0, 0, -9.81) for Z-up, (0, -9.81, 0) for Y-up.

        Note:
            After calling this method, you should notify solvers via
            `solver.notify_model_changed(SolverNotifyFlags.MODEL_PROPERTIES)`.
        """
        if self.gravity is None:
            raise RuntimeError(
                "Model gravity not initialized. Ensure the model was created via ModelBuilder.finalize()"
            )

        if isinstance(gravity, tuple | list):
            self.gravity.assign([wp.vec3(gravity[0], gravity[1], gravity[2])])
        else:
            self.gravity.assign([gravity])

    def collide(
        self: Model,
        state: State,
        collision_pipeline: CollisionPipeline | None = None,
        rigid_contact_max_per_pair: int | None = None,
        soft_contact_max: int | None = None,
        soft_contact_margin: float = 0.01,
        edge_sdf_iter: int = 10,
        requires_grad: bool | None = None,
    ) -> Contacts:
        """
        Generate contact points for the particles and rigid bodies in the model.

        This method produces a :class:`Contacts` object containing collision/contact information
        for use in contact-dynamics kernels.

        Args:
            state (State): The current state of the model.
            collision_pipeline (CollisionPipeline, optional): Collision pipeline to use for contact generation.
                If not provided, a new one will be created if it hasn't been constructed before for this model.
            rigid_contact_max_per_pair (int, optional): Maximum number of rigid contacts per shape pair.
                If None, a kernel is launched to count the number of possible contacts.
            soft_contact_max (int, optional): Maximum number of soft contacts.
                If None, a kernel is launched to count the number of possible contacts.
            soft_contact_margin (float, optional): Margin for soft contact generation. Default is 0.01.
            edge_sdf_iter (int, optional): Number of search iterations for finding closest contact points between edges and SDF. Default is 10.
            requires_grad (bool, optional): Whether to duplicate contact arrays for gradient computation. If None, uses :attr:`Model.requires_grad`.

        Returns:
            Contacts: The contact object containing collision information.

        Note:
            Rigid contact margins are controlled per-shape via :attr:`Model.shape_contact_margin`, which is populated
            from ``ShapeConfig.contact_margin`` during model building. If a shape doesn't specify a contact margin,
            it defaults to ``builder.rigid_contact_margin``. To adjust contact margins, set them before calling
            :meth:`ModelBuilder.finalize`.
        """
        from .collide import CollisionPipeline  # noqa: PLC0415

        if requires_grad is None:
            requires_grad = self.requires_grad

        if collision_pipeline is not None:
            self._collision_pipeline = collision_pipeline
        elif not hasattr(self, "_collision_pipeline"):
            self._collision_pipeline = CollisionPipeline.from_model(
                model=self,
                rigid_contact_max_per_pair=rigid_contact_max_per_pair,
                soft_contact_max=soft_contact_max,
                soft_contact_margin=soft_contact_margin,
                edge_sdf_iter=edge_sdf_iter,
                requires_grad=requires_grad,
            )

        # update any additional parameters
        self._collision_pipeline.soft_contact_margin = soft_contact_margin
        self._collision_pipeline.edge_sdf_iter = edge_sdf_iter

        contacts = self._collision_pipeline.collide(self, state)
        # attach custom attributes with assignment==CONTACT
        self._add_custom_attributes(contacts, ModelAttributeAssignment.CONTACT, requires_grad=requires_grad)
        return contacts

    def _add_custom_attributes(
        self,
        destination: object,
        assignment: ModelAttributeAssignment,
        requires_grad: bool = False,
        clone_arrays: bool = True,
    ) -> None:
        """
        Add custom attributes of a specific assignment type to a destination object.

        Args:
            destination: The object to add attributes to (State, Control, or Contacts)
            assignment: The assignment type to filter attributes by
            requires_grad: Whether cloned arrays should have requires_grad enabled
            clone_arrays: Whether to clone wp.arrays (True) or use references (False)
        """
        for full_name, _freq in self.attribute_frequency.items():
            if self.attribute_assignment.get(full_name, ModelAttributeAssignment.MODEL) != assignment:
                continue

            # Parse namespace from full_name (format: "namespace:attr_name" or "attr_name")
            if ":" in full_name:
                namespace, attr_name = full_name.split(":", 1)
                # Get source from namespaced location on model
                ns_obj = getattr(self, namespace, None)
                if ns_obj is None:
                    raise AttributeError(f"Namespace '{namespace}' does not exist on the model")
                src = getattr(ns_obj, attr_name, None)
                if src is None:
                    raise AttributeError(
                        f"Attribute '{namespace}.{attr_name}' is registered but does not exist on the model"
                    )
                # Create namespace on destination if it doesn't exist
                if not hasattr(destination, namespace):
                    setattr(destination, namespace, AttributeNamespace(namespace))
                dest = getattr(destination, namespace)
            else:
                # Non-namespaced attribute - add directly to destination
                attr_name = full_name
                src = getattr(self, attr_name, None)
                if src is None:
                    raise AttributeError(
                        f"Attribute '{attr_name}' is registered in attribute_frequency but does not exist on the model"
                    )
                dest = destination

            # Add attribute to the determined destination (either destination or dest_ns)
            if isinstance(src, wp.array):
                if clone_arrays:
                    setattr(dest, attr_name, wp.clone(src, requires_grad=requires_grad))
                else:
                    setattr(dest, attr_name, src)
            else:
                setattr(dest, attr_name, src)

    def add_attribute(
        self,
        name: str,
        attrib: wp.array,
        frequency: ModelAttributeFrequency,
        assignment: ModelAttributeAssignment | None = None,
        namespace: str | None = None,
    ):
        """
        Add a custom attribute to the model.

        Args:
            name (str): Name of the attribute.
            attrib (wp.array): The array to add as an attribute.
            frequency (ModelAttributeFrequency): The frequency of the attribute using ModelAttributeFrequency enum.
            assignment (ModelAttributeAssignment, optional): The assignment category using ModelAttributeAssignment enum.
                Determines which object will hold the attribute.
            namespace (str, optional): Namespace for the attribute.
                If None, attribute is added directly to the assignment object (e.g., model.attr_name).
                If specified, attribute is added to a namespace object (e.g., model.namespace_name.attr_name).

        Raises:
            TypeError: If the attribute is not a wp.array.
            AttributeError: If the attribute already exists or is on the wrong device.
        """
        if not isinstance(attrib, wp.array):
            raise TypeError(f"Attribute '{name}' must be a wp.array")
        if attrib.device != self.device:
            raise AttributeError(f"Attribute '{name}' device mismatch (model={self.device}, got={attrib.device})")

        # Handle namespaced attributes
        if namespace:
            # Create namespace object if it doesn't exist
            if not hasattr(self, namespace):
                setattr(self, namespace, AttributeNamespace(namespace))

            ns_obj = getattr(self, namespace)
            if hasattr(ns_obj, name):
                raise AttributeError(f"Attribute already exists: {namespace}.{name}")

            setattr(ns_obj, name, attrib)
            full_name = f"{namespace}:{name}"
        else:
            # Add directly to model
            if hasattr(self, name):
                raise AttributeError(f"Attribute already exists: {name}")
            setattr(self, name, attrib)
            full_name = name

        self.attribute_frequency[full_name] = frequency
        if assignment is not None:
            self.attribute_assignment[full_name] = assignment

    def get_attribute_frequency(self, name: str) -> ModelAttributeFrequency:
        """
        Get the frequency of an attribute.

        Args:
            name (str): Name of the attribute.

        Returns:
            ModelAttributeFrequency: The frequency of the attribute as an enum value.

        Raises:
            AttributeError: If the attribute frequency is not known.
        """
        frequency = self.attribute_frequency.get(name)
        if frequency is None:
            raise AttributeError(f"Attribute frequency of '{name}' is not known")
        return frequency
