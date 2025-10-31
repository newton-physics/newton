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

"""
KAMINO: Constrained Rigid Multi-Body Model Builder
"""

from __future__ import annotations

import uuid  # TODO: remove this import

import numpy as np
import warp as wp
from warp.context import Devicelike

from .bodies import RigidBodiesModel, RigidBodyDescriptor
from .geometry import (
    CollisionGeometriesModel,
    CollisionGeometryDescriptor,
    GeometriesModel,
    GeometryDescriptor,
)
from .gravity import GravityDescriptor, GravityModel
from .joints import (
    JointActuationType,
    JointDescriptor,
    JointDoFType,
    JointsModel,
)
from .materials import MaterialDescriptor, MaterialManager, MaterialPairProperties, MaterialPairsModel
from .math import FLOAT32_EPS, FLOAT32_MAX, FLOAT32_MIN
from .model import Model, ModelInfo
from .shapes import ShapeDescriptorType, ShapeType
from .time import TimeModel
from .types import Axis, float32, int32, mat33f, transformf, uint32, vec3f, vec4f, vec6f
from .world import WorldDescriptor

###
# Module interface
###

__all__ = [
    "ModelBuilder",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Containers
###


class ModelBuilder:
    """
    A class to facilitate construction of simulation models.
    """

    def __init__(self):
        # Meta-data
        self._num_worlds: int = 1
        self._device: Devicelike = None
        self._requires_grad: bool = False

        # Declare and initialize counters
        self._num_bodies: int = 0
        self._num_joints: int = 0
        self._num_cgeoms: int = 0
        self._num_pgeoms: int = 0
        self._num_bdofs: int = 0
        self._num_jcoords: int = 0
        self._num_jdofs: int = 0
        self._num_jpcoords: int = 0
        self._num_jpdofs: int = 0
        self._num_jacoords: int = 0
        self._num_jadofs: int = 0
        self._num_jcts: int = 0

        # Model descriptors
        self._up_axes: list[Axis] = [Axis.Z]
        self._worlds: list[WorldDescriptor] = [WorldDescriptor()]
        self._gravity: list[GravityDescriptor] = [GravityDescriptor()]
        self._materials: list[MaterialManager] = [MaterialManager()]
        self._bodies: list[RigidBodyDescriptor] = []
        self._joints: list[JointDescriptor] = []
        self._pgeoms: list[GeometryDescriptor] = []
        self._cgeoms: list[CollisionGeometryDescriptor] = []

        # Initialize the default world descriptor with the default material
        self.world.add_material(self.materials.default)

    @property
    def num_worlds(self) -> int:
        return self._num_worlds

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def num_collision_geoms(self) -> int:
        return self._num_cgeoms

    @property
    def num_physical_geoms(self) -> int:
        return self._num_pgeoms

    @property
    def num_materials(self) -> int:
        return self._materials[0].num_materials

    @property
    def num_body_dofs(self) -> int:
        return self._num_bdofs

    @property
    def num_joint_dofs(self) -> int:
        return self._num_jdofs

    @property
    def num_passive_joint_dofs(self) -> int:
        return self._num_jpdofs

    @property
    def num_actuated_joint_dofs(self) -> int:
        return self._num_jadofs

    @property
    def num_joint_cts(self) -> int:
        return self._num_jcts

    @property
    def world(self) -> WorldDescriptor:
        return self._worlds[0]

    @property
    def up_axis(self) -> Axis:
        return self._up_axes[0]

    @property
    def gravity(self) -> GravityDescriptor:
        return self._gravity[0]

    @property
    def materials(self) -> MaterialManager:
        return self._materials[0]

    @property
    def bodies(self) -> list[RigidBodyDescriptor]:
        return self._bodies

    @property
    def joints(self) -> list[JointDescriptor]:
        return self._joints

    @property
    def collision_geoms(self) -> list[CollisionGeometryDescriptor]:
        return self._cgeoms

    @property
    def physical_geoms(self) -> list[GeometryDescriptor]:
        return self._pgeoms

    ###
    # Internal Functions
    ###

    @staticmethod
    def _compute_inverse_inertia(i_I_i: mat33f) -> mat33f:
        """
        Computes the inverse inertia matrix from the given inertia matrix.

        Args:
            i_I_i (mat33f): The inertia matrix to be inverted.

        Returns:
            mat33f: The inverse inertia matrix.
        """
        return mat33f(np.linalg.inv(np.ndarray(buffer=i_I_i, shape=(3, 3), dtype=np.float32)))

    @staticmethod
    def _check_body_inertia(m_i: float, i_I_i: mat33f):
        """
        Checks if the body inertia is valid.

        Args:
            i_I_i (mat33f): The inertia matrix to be checked.

        Raises:
            ValueError: If the inertia matrix is not symmetric of positive definite.
        """
        # Convert to numpy array for easier checks
        i_I_i_np = np.ndarray(buffer=i_I_i, shape=(3, 3), dtype=np.float32)

        # Perform checks on the inertial properties
        if m_i <= 0.0:
            raise ValueError(f"Invalid body mass: {m_i}. Must be greater than 0.0")
        if not np.allclose(i_I_i_np, i_I_i_np.T, atol=float(FLOAT32_EPS)):
            raise ValueError(f"Invalid body inertia matrix:\n{i_I_i}\nMust be symmetric.")
        if not np.all(np.linalg.eigvals(i_I_i_np) > 0.0):
            raise ValueError(f"Invalid body inertia matrix:\n{i_I_i}\nMust be positive definite.")

    @staticmethod
    def _check_body_pose(q_i: transformf):
        """
        Checks if the body pose is valid.

        Args:
            q_i_0 (transformf): The pose of the body to be checked.

        Raises:
            ValueError: If the body pose is not a valid transformation.
        """
        if not isinstance(q_i, transformf):
            raise TypeError(f"Invalid body pose type: {type(q_i)}. Must be `transformf`.")

        # Extract the orientation quaternion
        if not np.isclose(wp.length(q_i.q), 1.0, atol=float(FLOAT32_EPS)):
            raise ValueError(f"Invalid body pose orientation quaternion: {q_i.q}. Must be a unit quaternion.")

    def _check_world_index(self, world_index: int) -> WorldDescriptor:
        """
        Checks if the provided world index is valid.

        Args:
            world_index (int): The index of the world to be checked.

        Raises:
            ValueError: If the world index is out of range.
        """
        if world_index < 0 or world_index >= self._num_worlds:
            raise ValueError(f"Invalid world index (wid): {world_index}. Must be between 0 and {self._num_worlds - 1}.")
        return self._worlds[world_index]

    @staticmethod
    def _check_limits(
        limits: list[float] | float | None, num_dofs: int, default: float = float(FLOAT32_MAX)
    ) -> list[float]:
        """
        Processes a specified limit value to ensure it is a list of floats.

        Notes:
        - If the input is None, a list of default values is returned.
        - If the input is a single float, it is converted to a list of the specified length.
        - If the input is an empty list, a list of default values is returned.
        - If the input is a non-empty list, it is validated to ensure it contains only floats and matches the specified length.

        Args:
            limits (List[float] | float | None): The limits to be processed.
            num_dofs (int): The number of degrees of freedom to determine the length of the output list.
            default (float): The default value to use if limits is None or an empty list.

        Returns:
            List[float]: The processed list of limits.

        Raises:
            ValueError: If the length of the limits list does not match num_dofs.
            TypeError: If the limits list contains non-float types.
        """
        if limits is None:
            return [float(default) for _ in range(num_dofs)]

        if isinstance(limits, float):
            if limits == np.inf:
                return [float(FLOAT32_MAX) for _ in range(num_dofs)]
            elif limits == -np.inf:
                return [float(FLOAT32_MIN) for _ in range(num_dofs)]
            else:
                return [limits] * num_dofs

        if isinstance(limits, list):
            if len(limits) == 0:
                return [float(default) for _ in range(num_dofs)]

            if len(limits) != num_dofs:
                raise ValueError(f"Invalid limits length: {len(limits)} != {num_dofs}")

            if all(isinstance(x, float) for x in limits):
                return limits
            else:
                raise TypeError(
                    f"Invalid limits type: All entries must be `float`, but got:\n{[type(x) for x in limits]}"
                )

    ###
    # Public API
    ###

    def set_up_axis(self, axis: Axis, world_index: int = 0):
        # Check if the world index is valid
        self._check_world_index(world_index)

        # Check if the axis is valid
        if not isinstance(axis, Axis):
            raise TypeError(f"ModelBuilder: Invalid axis type: {type(axis)}. Must be `Axis`.")

        # Set the new up axis
        self._up_axes[world_index] = axis

    def set_gravity(self, gravity: GravityDescriptor, world_index: int = 0):
        # Check if the world index is valid
        self._check_world_index(world_index)

        # Check if the gravity descriptor is valid
        if not isinstance(gravity, GravityDescriptor):
            raise TypeError(f"Invalid gravity descriptor type: {type(gravity)}. Must be `GravityDescriptor`.")

        # Set the new gravity configurations
        self._gravity[world_index] = gravity

    def add_body(
        self,
        m_i: float,
        i_I_i: mat33f,
        q_i_0: transformf,
        u_i_0: vec6f,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Get current bid from the number of bodies
        bid = world.num_bodies

        # Generate identifiers if not provided
        if name is None:
            name = f"body_{bid}"
        if uid is None:
            uid = str(uuid.uuid4())

        # Check if the body name and UID are unique
        if name in world.body_names:
            raise ValueError(f"Body name '{name}' already exists.")
        if uid in world.body_uids:
            raise ValueError(f"Body UID '{uid}' already exists.")

        # Check if body properties are valid
        self._check_body_inertia(m_i, i_I_i)
        self._check_body_pose(q_i_0)

        # Create the body model descriptor
        body = RigidBodyDescriptor()
        body.wid = world_index
        body.name = name
        body.uid = uid
        body.bid = bid
        body.m_i = m_i
        body.i_I_i = i_I_i
        body.q_i_0 = q_i_0
        body.u_i_0 = u_i_0

        # Append body model data
        world.add_body(body)
        self._bodies.append(body)

        # Update counter
        self._num_bodies += 1
        self._num_bdofs += 6

        # Return the new body index
        return bid

    def add_rigid_body_descriptor(self, descriptor: RigidBodyDescriptor, world_index: int = 0) -> int:
        """
        Add a body to the model by descriptor.
        """
        # Check if the descriptor is valid
        if not isinstance(descriptor, RigidBodyDescriptor):
            raise TypeError(f"Invalid body descriptor type: {type(descriptor)}. Must be `RigidBodyDescriptor`.")

        # TODO: this seems wasteful to unpack and re-pack the descriptor, how can we avoid this?
        return self.add_body(
            m_i=descriptor.m_i,
            i_I_i=descriptor.i_I_i,
            q_i_0=descriptor.q_i_0,
            u_i_0=descriptor.u_i_0,
            name=descriptor.name,
            uid=descriptor.uid,
            world_index=world_index,
        )

    def add_joint(
        self,
        act_type: JointActuationType,
        dof_type: JointDoFType,
        bid_B: int,
        bid_F: int,
        B_r_Bj: vec3f,
        F_r_Fj: vec3f,
        X_j: mat33f,
        q_j_min: list[float] | float | None = None,
        q_j_max: list[float] | float | None = None,
        dq_j_max: list[float] | float | None = None,
        tau_j_max: list[float] | float | None = None,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Check if the actuation type is valid
        if not isinstance(act_type, JointActuationType):
            raise TypeError(f"Invalid actuation type: {act_type}. Must be `JointActuationType`.")

        # Check if the DoF type is valid
        if not isinstance(dof_type, JointDoFType):
            raise TypeError(f"Invalid DoF type: {dof_type}. Must be `JointDoFType`.")

        # Check if the body indices are valid
        if bid_B < 0 and bid_F < 0:
            raise ValueError(
                f"Invalid body indices: bid_B={bid_B}, bid_F={bid_F}:\n\
                - ==-1 indicates the world body, >=0 indicates finite rigid bodies\n\
                - Base BIDs must be in [-1, {world.num_bodies - 1}]\n\
                - Follower BIDs must be in [0, {world.num_bodies - 1}]"
            )
        if bid_B >= world.num_bodies or bid_F >= world.num_bodies:
            raise ValueError(
                f"Invalid body indices: bid_B={bid_B}, bid_F={bid_F}.\n\
                - ==-1 indicates the world body, >=0 indicates finite rigid bodies\n\
                - Base BIDs must be in [-1, {world.num_bodies - 1}]\n\
                - Follower BIDs must be in [0, {world.num_bodies - 1}]"
            )

        # Get current bid from the number of bodies
        jid = world.num_joints

        # Generate identifiers if not provided
        if name is None:
            name = f"joint_{jid}"
        if uid is None:
            uid = str(uuid.uuid4())

        # Check if the body name and UID are unique
        if name in world.joint_names:
            raise ValueError(f"Joint name '{name}' already exists.")
        if uid in world.joint_uids:
            raise ValueError(f"Joint UID '{uid}' already exists.")

        # Create the joint model descriptor
        joint = JointDescriptor()
        joint.wid = world_index
        joint.name = name
        joint.uid = uid
        joint.jid = jid
        joint.act_type = act_type
        joint.dof_type = dof_type
        joint.bid_B = bid_B
        joint.bid_F = bid_F
        joint.B_r_Bj = B_r_Bj
        joint.F_r_Fj = F_r_Fj
        joint.X_j = X_j

        # Retrieve joint dimensions from the DoF type
        joint.num_coords = dof_type.num_coords
        joint.num_dofs = dof_type.num_dofs
        joint.num_cts = dof_type.num_cts

        # Set the index offsets w.r.t the world
        joint.coords_offset = world.num_joint_coords
        joint.dofs_offset = world.num_joint_dofs
        joint.cts_offset = world.num_joint_cts
        # NOTE: passive/actuated index offsets are set to -1 if not belonging to the respective category
        # TODO: Is there a better way to handle this?
        joint.passive_coords_offset = world.num_passive_joint_coords if act_type == JointActuationType.PASSIVE else -1
        joint.passive_dofs_offset = world.num_passive_joint_dofs if act_type == JointActuationType.PASSIVE else -1
        joint.actuated_coords_offset = world.num_actuated_joint_coords if act_type > JointActuationType.PASSIVE else -1
        joint.actuated_dofs_offset = world.num_actuated_joint_dofs if act_type > JointActuationType.PASSIVE else -1

        # Set default values for joint limits if not provided
        q_j_min = self._check_limits(q_j_min, joint.num_coords, float(FLOAT32_MIN))
        q_j_max = self._check_limits(q_j_max, joint.num_coords, float(FLOAT32_MAX))
        dq_j_max = self._check_limits(dq_j_max, joint.num_dofs, float(FLOAT32_MAX))
        tau_j_max = self._check_limits(tau_j_max, joint.num_dofs, float(FLOAT32_MAX))

        # Store the joint limits
        joint.q_j_min = q_j_min
        joint.q_j_max = q_j_max
        joint.dq_j_max = dq_j_max
        joint.tau_j_max = tau_j_max

        # Append joint model data
        world.add_joint(joint)
        self._joints.append(joint)

        # Update counter
        self._num_joints += 1
        self._num_jcoords += joint.num_coords
        self._num_jdofs += joint.num_dofs
        self._num_jpcoords += joint.num_coords if act_type == JointActuationType.PASSIVE else 0
        self._num_jpdofs += joint.num_dofs if act_type == JointActuationType.PASSIVE else 0
        self._num_jacoords += joint.num_coords if act_type > JointActuationType.PASSIVE else 0
        self._num_jadofs += joint.num_dofs if act_type > JointActuationType.PASSIVE else 0
        self._num_jcts += joint.num_cts

        # Return the new joint index
        return jid

    def add_joint_descriptor(self, descriptor: JointDescriptor, world_index: int = 0) -> int:
        """Add a joint to the model by descriptor."""
        # Check if the descriptor is valid
        if not isinstance(descriptor, JointDescriptor):
            raise TypeError(f"Invalid joint descriptor type: {type(descriptor)}. Must be `JointDescriptor`.")
        # TODO: this seems wasteful to unpack and re-pack the descriptor, how can we avoid this?
        return self.add_joint(
            act_type=descriptor.act_type,
            dof_type=descriptor.dof_type,
            bid_B=descriptor.bid_B,
            bid_F=descriptor.bid_F,
            B_r_Bj=descriptor.B_r_Bj,
            F_r_Fj=descriptor.F_r_Fj,
            X_j=descriptor.X_j,
            q_j_min=descriptor.q_j_min,
            q_j_max=descriptor.q_j_max,
            dq_j_max=descriptor.dq_j_max,
            tau_j_max=descriptor.tau_j_max,
            name=descriptor.name,
            uid=descriptor.uid,
            world_index=world_index,
        )

    def add_collision_layer(self, name: str, world_index: int = 0):
        """Add a new collision geometry layer to the model."""
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Append the new layer name to the list of layers if it does not already exist
        if name not in world.collision_geometry_layers:
            world.collision_geometry_layers.append(name)

    def add_physical_layer(self, name: str, world_index: int = 0):
        """Add a new physical geometry layer to the model."""
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Check if the layer name already exists
        if name in world.physical_geometry_layers:
            raise ValueError(f"Layer name '{name}' already exists.")

        # Append the new layer name to the list of layers
        world.physical_geometry_layers.append(name)

    def add_collision_geometry(
        self,
        body_id: int = -1,
        shape: ShapeDescriptorType | None = None,
        layer: int | str = 0,
        offset: transformf | None = None,
        material: str | int | None = None,
        max_contacts: int = 0,
        group: int = 1,
        collides: int = 1,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Get current bid from the number of bodies
        cgid = world.num_collision_geoms

        # Generate identifiers if not provided
        if name is None:
            name = f"cgeom_{cgid}"
        if uid is None:
            uid = str(uuid.uuid4())

        # Set the default material if not provided
        if material is None:
            material = self.materials.default.name

        # Check if the body name and UID are unique
        name_exists = name in world.collision_geom_names
        uid_exists = uid in world.collision_geom_uids
        if name_exists and uid_exists:
            raise ValueError(f"Geometry name '{name}' and UID '{uid}' already exists.")

        # Retrieve the layer ID
        layer_id = 0
        if isinstance(layer, str):
            if layer not in world.collision_geometry_layers:
                raise ValueError(f"Layer name '{layer}' not found.")
            layer_id = world.collision_geometry_layers.index(layer)
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(world.collision_geometry_layers):
                raise ValueError(f"Layer index '{layer}' out of range.")
            layer_id = layer

        # Create the collision geometry descriptor
        geom = CollisionGeometryDescriptor()
        geom.wid = world_index
        geom.name = name
        geom.uid = uid
        geom.gid = cgid
        geom.bid = body_id
        geom.lid = layer_id
        geom.offset = offset if offset is not None else transformf()
        geom.shape = shape
        geom.mid = self.materials.index(material)
        geom.group = group
        geom.collides = collides
        geom.max_contacts = max_contacts

        # Append body model data
        world.add_cgeom(geom)
        self._cgeoms.append(geom)

        # Update counters
        self._num_cgeoms += 1

        # Return the new geometry index
        return cgid

    def add_collision_geometry_descriptor(self, descriptor: CollisionGeometryDescriptor, world_index: int = 0) -> int:
        """Add a collision geometry to the model by descriptor."""
        # Check if the descriptor is valid
        if not isinstance(descriptor, CollisionGeometryDescriptor):
            raise TypeError(
                f"Invalid collision geometry descriptor type: {type(descriptor)}. "
                "Must be `CollisionGeometryDescriptor`."
            )
        # TODO: this seems wasteful to unpack and re-pack the descriptor, how can we avoid this?
        return self.add_collision_geometry(
            body_id=descriptor.bid,
            shape=descriptor.shape,
            layer=descriptor.lid,
            offset=descriptor.offset,
            material=descriptor.mid,
            max_contacts=descriptor.max_contacts,
            group=descriptor.group,
            collides=descriptor.collides,
            name=descriptor.name,
            uid=descriptor.uid,
            world_index=world_index,
        )

    def add_physical_geometry(
        self,
        body_id: int,
        shape: ShapeDescriptorType,
        layer: int | str = 0,
        offset: transformf | None = None,
        name: str | None = None,
        uid: str | None = None,
        world_index: int = 0,
    ) -> int:
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Get current bid from the number of bodies
        pgid = world.num_physical_geoms

        # Generate identifiers if not provided
        if name is None:
            name = f"pgeom_{pgid}"
        if uid is None:
            uid = str(uuid.uuid4())

        # Check if the body name and UID are unique
        name_exists = name in world.physical_geom_names
        uid_exists = uid in world.physical_geom_uids
        if name_exists and uid_exists:
            raise ValueError(f"Geometry name '{name}' and UID '{uid}' already exists.")

        # Retrieve the layer ID
        layer_id = 0
        if isinstance(layer, str):
            if layer not in world.physical_geometry_layers:
                raise ValueError(f"Layer name '{layer}' not found.")
            layer_id = world.physical_geometry_layers.index(layer)
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(world.physical_geometry_layers):
                raise ValueError(f"Layer index '{layer}' out of range.")
            layer_id = layer

        # Create the collision geometry descriptor
        geom = GeometryDescriptor()
        geom.wid = world_index
        geom.name = name
        geom.uid = uid
        geom.gid = pgid
        geom.bid = body_id
        geom.lid = layer_id
        geom.offset = offset if offset is not None else transformf()
        geom.shape = shape

        # Append body model data
        world.add_pgeom(geom)
        self._pgeoms.append(geom)

        # Update counters
        self._num_pgeoms += 1

        # Return the new geometry index
        return pgid

    def add_physical_geometry_descriptor(self, descriptor: GeometryDescriptor, world_index: int = 0) -> int:
        """Add a physical geometry to the model by descriptor."""
        # Check if the descriptor is valid
        if not isinstance(descriptor, GeometryDescriptor):
            raise TypeError(
                f"Invalid physical geometry descriptor type: {type(descriptor)}. Must be `GeometryDescriptor`."
            )

        # TODO: this seems wasteful to unpack and re-pack the descriptor, how can we avoid this?
        return self.add_physical_geometry(
            body_id=descriptor.bid,
            shape=descriptor.shape,
            layer=descriptor.lid,
            offset=descriptor.offset,
            name=descriptor.name,
            uid=descriptor.uid,
            world_index=world_index,
        )

    def set_default_material(self, material: MaterialDescriptor, world_index: int = 0):
        """
        Sets the default material for the model.
        Raises an error if the material is not registered.
        """
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Check if the material is valid
        if not isinstance(material, MaterialDescriptor):
            raise TypeError(f"Invalid material type: {type(material)}. Must be `MaterialDescriptor`.")

        # Set the default material in the material manager
        self.materials.default = material

        # Reset the default material of the world
        world.set_material(material, 0)

    def add_material(self, material: MaterialDescriptor, world_index: int = 0) -> int:
        # Check if the world index is valid
        world = self._check_world_index(world_index)

        # Check if the material is valid
        if not isinstance(material, MaterialDescriptor):
            raise TypeError(f"Invalid material type: {type(material)}. Must be `MaterialDescriptor`.")

        # Register the material in the material manager
        world.add_material(material)
        return self._materials[world_index].register(material)

    def set_material_pair(
        self,
        first: int | str | MaterialDescriptor,
        second: int | str | MaterialDescriptor,
        material_pair: MaterialPairProperties,
        world_index: int = 0,
    ):
        # Check if the world index is valid
        self._check_world_index(world_index)

        # Extract the material names if arguments are descriptors
        first_id = first.name if isinstance(first, MaterialDescriptor) else first
        second_id = second.name if isinstance(second, MaterialDescriptor) else second

        # Register the material pair in the material manager
        self._materials[world_index].configure_pair(first=first_id, second=second_id, material_pair=material_pair)

    def add_builder(self, other: ModelBuilder):
        """
        Extends the contents of the current ModelBuilder with those of another.

        Each builder represents a distinct world, and this method allows for the
        combination of multiple worlds into a single model. The method ensures that the
        indices of the elements in the other builder are adjusted to account for the
        existing elements in the current builder, preventing any index conflicts.

        Arguments
        ----
            other (`ModelBuilder`): The other ModelBuilder instance to be added to the current one.
        """
        # Check if the other builder is of valid type
        if not isinstance(other, ModelBuilder):
            raise ValueError(f"Invalid builder type: {type(other)}. Must be a ModelBuilder instance.")

        # Offset the indices of the other builders elements
        for body in other.bodies:
            body.wid = self._num_worlds
        for joint in other.joints:
            joint.wid = self._num_worlds
            if joint.bid_B >= 0:
                joint.bid_B += self._num_bodies
            joint.bid_F += self._num_bodies
        for cgeom in other.collision_geoms:
            cgeom.wid = self._num_worlds
            if cgeom.bid >= 0:
                cgeom.bid += self._num_bodies
        for pgeom in other.physical_geoms:
            pgeom.wid = self._num_worlds
            if pgeom.bid >= 0:
                pgeom.bid += self._num_bodies

        # Set the element index offsets in the world descriptor of the other builder
        other.world.bodies_idx_offset += self._num_bodies
        other.world.joints_idx_offset += self._num_joints
        other.world.collision_geoms_idx_offset += self._num_cgeoms
        other.world.physical_geoms_idx_offset += self._num_pgeoms
        other.world.body_dofs_idx_offset += self._num_bdofs
        other.world.joint_coords_idx_offset += self._num_jcoords
        other.world.joint_dofs_idx_offset += self._num_jdofs
        other.world.passive_joint_coords_idx_offset += self._num_jpcoords
        other.world.passive_joint_dofs_idx_offset += self._num_jpdofs
        other.world.actuated_joint_coords_idx_offset += self._num_jacoords
        other.world.actuated_joint_dofs_idx_offset += self._num_jadofs
        other.world.joint_cts_idx_offset += self._num_jcts

        # Update counters
        self._num_bodies += len(other.bodies)
        self._num_joints += len(other.joints)
        self._num_cgeoms += len(other.collision_geoms)
        self._num_pgeoms += len(other.physical_geoms)
        self._num_bdofs += 6 * len(other.bodies)
        self._num_jcoords += sum([j.num_coords for j in other.joints])
        self._num_jdofs += sum([j.num_dofs for j in other.joints])
        self._num_jpcoords += sum([j.num_coords for j in other.joints if j.act_type == JointActuationType.PASSIVE])
        self._num_jpdofs += sum([j.num_dofs for j in other.joints if j.act_type == JointActuationType.PASSIVE])
        self._num_jacoords += sum([j.num_coords for j in other.joints if j.act_type > JointActuationType.PASSIVE])
        self._num_jadofs += sum([j.num_dofs for j in other.joints if j.act_type > JointActuationType.PASSIVE])
        self._num_jcts += sum([j.num_cts for j in other.joints])

        # Append the new model info
        self._worlds.append(other.world)

        # Append the new gravity descriptor
        self._gravity.append(other.gravity)

        # Append the new material manager
        self._materials.append(other.materials)

        # Append the new model elements
        self._bodies.extend(other.bodies)
        self._joints.extend(other.joints)
        self._cgeoms.extend(other.collision_geoms)
        self._pgeoms.extend(other.physical_geoms)

        # Increment the number of worlds
        self._num_worlds += 1

    def required_contact_capacity(self):
        # First check if there are any collision geometries
        has_cgeoms = False
        for world in self._worlds:
            if world.num_collision_geoms > 0:
                has_cgeoms = True
                break

        # If there are no collision geometries indicate this `-1`s
        if not has_cgeoms:
            # return -1, [-1] * self.num_worlds
            return 0, [0] * self.num_worlds

        # Else proceed to calculate the maximum number of contacts
        model_max_contacts = 0
        world_max_contacts = []

        # Calculate the maximum number of contacts for the model and each world
        for w in range(len(self._worlds)):
            world_max_contacts.append(0)
            for cgeom_maxnc in self._worlds[w].collision_geometry_max_contacts:
                model_max_contacts += cgeom_maxnc
                world_max_contacts[w] += cgeom_maxnc

        # Handle the case where there is only one model descriptor but multiple worlds are set
        if len(self._worlds) == 1 and self.num_worlds > 1:
            world_max_contacts = [model_max_contacts] * self.num_worlds
            model_max_contacts *= self.num_worlds

        # Return the total number of contacts and the per-world max contacts
        return model_max_contacts, world_max_contacts

    def finalize(self, device: Devicelike = None, requires_grad: bool = False) -> Model:
        # Number of model worlds
        num_worlds = len(self._worlds)

        ###
        # Model data collection
        ###

        # Initialize the info data collections
        info_nb = []
        info_nj = []
        info_njp = []
        info_nja = []
        info_ncg = []
        info_npg = []
        info_nbd = []
        info_njq = []
        info_njd = []
        info_njpq = []
        info_njpd = []
        info_njaq = []
        info_njad = []
        info_njc = []
        info_bio = []
        info_jio = []
        info_bdio = []
        info_jqio = []
        info_jdio = []
        info_jpqio = []
        info_jpdio = []
        info_jaqio = []
        info_jadio = []
        info_jcio = []
        info_mass_min = []
        info_mass_max = []
        info_mass_total = []
        info_inertia_total = []

        # Initialize the gravity data collections
        gravity_g_dir_acc = []
        gravity_vector = []

        # Initialize the body data collections
        bodies_wid = []
        bodies_bid = []
        bodies_m_i = []
        bodies_inv_m_i = []
        bodies_i_I_i = []
        bodies_inv_i_I_i = []
        bodies_q_i_0 = []
        bodies_u_i_0 = []

        # Initialize the joint data collections
        joints_wid = []
        joints_jid = []
        joints_dofid = []
        joints_actid = []
        joints_c_j = []
        joints_d_j = []
        joints_m_j = []
        joints_qio = []
        joints_dio = []
        joints_cio = []
        joints_pqio = []
        joints_pdio = []
        joints_aqio = []
        joints_adio = []
        joints_bid_B = []
        joints_bid_F = []
        joints_B_r_Bj = []
        joints_F_r_Fj = []
        joints_X_j = []
        joints_q_j_min = []
        joints_q_j_max = []
        joints_qd_j_max = []
        joints_tau_j_max = []

        # Initialize the collision geometry data collections
        cgeoms_wid = []
        cgeoms_gid = []
        cgeoms_lid = []
        cgeoms_bid = []
        cgeoms_sid = []
        cgeoms_ptr = []
        cgeoms_params = []
        cgeoms_offset = []
        cgeoms_group = []
        cgeoms_collides = []
        cgeoms_mid = []

        # Initialize the physical geometry data collections
        pgeoms_wid = []
        pgeoms_gid = []
        pgeoms_lid = []
        pgeoms_bid = []
        pgeoms_sid = []
        pgeoms_ptr = []
        pgeoms_params = []
        pgeoms_offset = []

        # Initialize the material-pair data collections
        mpairs_rest = []
        mpairs_static_fric = []
        mpairs_dynamic_fric = []

        # A helper function to collect model info data
        def collect_model_info_data():
            for world in self._worlds:
                # First collect the immutable counts and
                # index offsets for bodies and joints
                info_nb.append(world.num_bodies)
                info_nj.append(world.num_joints)
                info_njp.append(world.num_passive_joints)
                info_nja.append(world.num_actuated_joints)
                info_ncg.append(world.num_collision_geoms)
                info_npg.append(world.num_physical_geoms)
                info_nbd.append(world.num_body_dofs)
                info_njq.append(world.num_joint_coords)
                info_njd.append(world.num_joint_dofs)
                info_njpq.append(world.num_passive_joint_coords)
                info_njpd.append(world.num_passive_joint_dofs)
                info_njaq.append(world.num_actuated_joint_coords)
                info_njad.append(world.num_actuated_joint_dofs)
                info_njc.append(world.num_joint_cts)
                info_bio.append(world.bodies_idx_offset)
                info_jio.append(world.joints_idx_offset)

                # Collect the model mass and inertia data
                info_mass_min.append(world.mass_min)
                info_mass_max.append(world.mass_max)
                info_mass_total.append(world.mass_total)
                info_inertia_total.append(world.inertia_total)

            # Collect the index offsets for bodies and joints
            for world in self._worlds:
                info_bdio.append(world.body_dofs_idx_offset)
                info_jqio.append(world.joint_coords_idx_offset)
                info_jdio.append(world.joint_dofs_idx_offset)
                info_jpqio.append(world.passive_joint_coords_idx_offset)
                info_jpdio.append(world.passive_joint_dofs_idx_offset)
                info_jaqio.append(world.actuated_joint_coords_idx_offset)
                info_jadio.append(world.actuated_joint_dofs_idx_offset)
                info_jcio.append(world.joint_cts_idx_offset)

        # A helper function to collect model gravity data
        def collect_gravity_model_data():
            for w in range(num_worlds):
                gravity_g_dir_acc.append(self._gravity[w].dir_accel())
                gravity_vector.append(self._gravity[w].vector())

        # A helper function to collect model bodies data
        def collect_body_model_data():
            for body in self._bodies:
                bodies_wid.append(body.wid)
                bodies_bid.append(body.bid)
                bodies_m_i.append(body.m_i)
                bodies_inv_m_i.append(1.0 / body.m_i)
                bodies_i_I_i.append(body.i_I_i)
                bodies_inv_i_I_i.append(wp.inverse(body.i_I_i))
                bodies_q_i_0.append(body.q_i_0)
                bodies_u_i_0.append(body.u_i_0)

        # A helper function to collect model joints data
        def collect_joint_model_data():
            for joint in self._joints:
                joints_wid.append(joint.wid)
                joints_jid.append(joint.jid)
                joints_dofid.append(joint.dof_type.value)
                joints_actid.append(joint.act_type.value)
                joints_c_j.append(joint.num_coords)
                joints_d_j.append(joint.num_dofs)
                joints_m_j.append(joint.num_cts)
                joints_qio.append(joint.coords_offset)
                joints_dio.append(joint.dofs_offset)
                joints_pqio.append(joint.passive_coords_offset)
                joints_pdio.append(joint.passive_dofs_offset)
                joints_aqio.append(joint.actuated_coords_offset)
                joints_adio.append(joint.actuated_dofs_offset)
                joints_cio.append(joint.cts_offset)
                joints_bid_B.append(joint.bid_B)
                joints_bid_F.append(joint.bid_F)
                joints_B_r_Bj.append(joint.B_r_Bj)
                joints_F_r_Fj.append(joint.F_r_Fj)
                joints_X_j.append(joint.X_j)
                joints_q_j_min.extend(joint.q_j_min)
                joints_q_j_max.extend(joint.q_j_max)
                joints_qd_j_max.extend(joint.dq_j_max)
                joints_tau_j_max.extend(joint.tau_j_max)

        # A helper function to create geometry pointers
        # NOTE: This also finalizes the mesh/SDF/HField data on the device
        def make_geometry_source_pointer(geom: GeometryDescriptor, mesh_geoms: dict, device) -> int:
            # Append to data pointers array of the shape has a Mesh, SDF or HField source
            if geom.shape.type in (ShapeType.MESH, ShapeType.CONVEX, ShapeType.HFIELD, ShapeType.SDF):
                geom_hash = hash(geom)  # avoid repeated hash computations
                # If the geometry has a Mesh, SDF or HField source,
                # finalize it and retrieve the mesh pointer/index
                if geom_hash not in mesh_geoms:
                    mesh_geoms[geom_hash] = geom.shape.data.finalize(device=device)
                # Return the mesh data pointer/index
                return mesh_geoms[geom_hash]
            # Otherwise, append a null (i.e. zero-valued) pointer
            else:
                return 0

        # A helper function to collect model collision geometries data
        def collect_collision_geometry_model_data():
            cgeom_meshes = {}
            for geom in self._cgeoms:
                cgeoms_wid.append(geom.wid)
                cgeoms_gid.append(geom.gid)
                cgeoms_lid.append(geom.lid)
                cgeoms_bid.append(geom.bid)
                cgeoms_sid.append(geom.shape.type.value)
                cgeoms_params.append(geom.shape.paramsvec)
                cgeoms_offset.append(geom.offset)
                cgeoms_mid.append(geom.mid)
                cgeoms_group.append(geom.group)
                cgeoms_collides.append(geom.collides)
                cgeoms_ptr.append(make_geometry_source_pointer(geom, cgeom_meshes, device))

        # A helper function to collect model physical geometries data
        def collect_physical_geometry_model_data():
            pgeom_meshes = {}
            for geom in self._pgeoms:
                pgeoms_wid.append(geom.wid)
                pgeoms_gid.append(geom.gid)
                pgeoms_lid.append(geom.lid)
                pgeoms_bid.append(geom.bid)
                pgeoms_sid.append(geom.shape.type.value)
                pgeoms_params.append(geom.shape.paramsvec)
                pgeoms_offset.append(geom.offset)
                pgeoms_ptr.append(make_geometry_source_pointer(geom, pgeom_meshes, device))

        # A helper function to collect model material-pairs data
        def collect_material_pairs_model_data():
            for mm in self._materials:
                mpairs_rest.append(mm.restitution_matrix())
                mpairs_static_fric.append(mm.static_friction_matrix())
                mpairs_dynamic_fric.append(mm.dynamic_friction_matrix())

        # Collect model data
        collect_model_info_data()
        collect_gravity_model_data()
        collect_body_model_data()
        collect_joint_model_data()
        collect_collision_geometry_model_data()
        collect_physical_geometry_model_data()
        collect_material_pairs_model_data()

        ###
        # Model construction
        ###

        # Create the model
        model = Model()

        # Configure model properties
        model.device = device
        model.requires_grad = requires_grad

        # Store the model builder info list as the model descriptors
        # NOTE This caches the info of each model on the host side
        model.worlds = self._worlds

        # Set the host-side model size

        # Compute the sum/max of model entities
        model.size.num_worlds = num_worlds
        model.size.sum_of_num_bodies = self._num_bodies
        model.size.max_of_num_bodies = max([world.num_bodies for world in self._worlds])
        model.size.sum_of_num_joints = self._num_joints
        model.size.max_of_num_joints = max([world.num_joints for world in self._worlds])
        model.size.sum_of_num_passive_joints = sum([world.num_passive_joints for world in self._worlds])
        model.size.max_of_num_passive_joints = max([world.num_passive_joints for world in self._worlds])
        model.size.sum_of_num_actuated_joints = sum([world.num_actuated_joints for world in self._worlds])
        model.size.max_of_num_actuated_joints = max([world.num_actuated_joints for world in self._worlds])
        model.size.sum_of_num_collision_geoms = self._num_cgeoms
        model.size.max_of_num_collision_geoms = max([world.num_collision_geoms for world in self._worlds])
        model.size.sum_of_num_physical_geoms = self._num_pgeoms
        model.size.max_of_num_physical_geoms = max([world.num_physical_geoms for world in self._worlds])
        model.size.sum_of_num_material_pairs = sum([world.num_material_pairs for world in self._worlds])
        model.size.max_of_num_material_pairs = max([world.num_material_pairs for world in self._worlds])

        # Compute the sum/max of model DoFs and constraints
        model.size.sum_of_num_body_dofs = self._num_bdofs
        model.size.max_of_num_body_dofs = max([world.num_body_dofs for world in self._worlds])
        model.size.sum_of_num_joint_coords = self._num_jcoords
        model.size.max_of_num_joint_coords = max([world.num_joint_coords for world in self._worlds])
        model.size.sum_of_num_joint_dofs = self._num_jdofs
        model.size.max_of_num_joint_dofs = max([world.num_joint_dofs for world in self._worlds])
        model.size.sum_of_num_passive_joint_coords = self._num_jpcoords
        model.size.max_of_num_passive_joint_coords = max([world.num_passive_joint_coords for world in self._worlds])
        model.size.sum_of_num_passive_joint_dofs = self._num_jpdofs
        model.size.max_of_num_passive_joint_dofs = max([world.num_passive_joint_dofs for world in self._worlds])
        model.size.sum_of_num_actuated_joint_coords = self._num_jacoords
        model.size.max_of_num_actuated_joint_coords = max([world.num_actuated_joint_coords for world in self._worlds])
        model.size.sum_of_num_actuated_joint_dofs = self._num_jadofs
        model.size.max_of_num_actuated_joint_dofs = max([world.num_actuated_joint_dofs for world in self._worlds])
        model.size.sum_of_num_joint_cts = self._num_jcts
        model.size.max_of_num_joint_cts = max([world.num_joint_cts for world in self._worlds])

        # Initialize unilateral counts (limits, and contacts) to zero
        model.size.sum_of_max_limits = 0
        model.size.max_of_max_limits = 0
        model.size.sum_of_max_contacts = 0
        model.size.max_of_max_contacts = 0
        model.size.sum_of_max_unilaterals = 0
        model.size.max_of_max_unilaterals = 0

        # Initialize total constraint counts to the same as the joint constraint counts
        model.size.sum_of_max_total_cts = model.size.sum_of_num_joint_cts
        model.size.max_of_max_total_cts = model.size.max_of_num_joint_cts

        # Construct model info data
        model.info = ModelInfo()

        # Allocate the model data on the target device
        with wp.ScopedDevice(device):
            # Create the immutable model info arrays from the collected data
            model.info.num_worlds = num_worlds
            model.info.num_bodies = wp.array(info_nb, dtype=int32)
            model.info.num_joints = wp.array(info_nj, dtype=int32)
            model.info.num_passive_joints = wp.array(info_njp, dtype=int32)
            model.info.num_actuated_joints = wp.array(info_nja, dtype=int32)
            model.info.num_collision_geoms = wp.array(info_ncg, dtype=int32)
            model.info.num_physical_geoms = wp.array(info_npg, dtype=int32)
            model.info.num_body_dofs = wp.array(info_nbd, dtype=int32)
            model.info.num_joint_coords = wp.array(info_njq, dtype=int32)
            model.info.num_joint_dofs = wp.array(info_njd, dtype=int32)
            model.info.num_passive_joint_coords = wp.array(info_njpq, dtype=int32)
            model.info.num_passive_joint_dofs = wp.array(info_njpd, dtype=int32)
            model.info.num_actuated_joint_coords = wp.array(info_njaq, dtype=int32)
            model.info.num_actuated_joint_dofs = wp.array(info_njad, dtype=int32)
            model.info.num_joint_cts = wp.array(info_njc, dtype=int32)
            model.info.bodies_offset = wp.array(info_bio, dtype=int32)
            model.info.joints_offset = wp.array(info_jio, dtype=int32)
            model.info.body_dofs_offset = wp.array(info_bdio, dtype=int32)
            model.info.joint_coords_offset = wp.array(info_jqio, dtype=int32)
            model.info.joint_dofs_offset = wp.array(info_jdio, dtype=int32)
            model.info.joint_passive_coords_offset = wp.array(info_jpqio, dtype=int32)
            model.info.joint_passive_dofs_offset = wp.array(info_jpdio, dtype=int32)
            model.info.joint_actuated_coords_offset = wp.array(info_jaqio, dtype=int32)
            model.info.joint_actuated_dofs_offset = wp.array(info_jadio, dtype=int32)
            model.info.joint_cts_offset = wp.array(info_jcio, dtype=int32)
            model.info.mass_min = wp.array(info_mass_min, dtype=float32)
            model.info.mass_max = wp.array(info_mass_max, dtype=float32)
            model.info.mass_total = wp.array(info_mass_total, dtype=float32)
            model.info.inertia_total = wp.array(info_inertia_total, dtype=float32)
            # TODO: Should we also pre-allocate the model info for max limits, contacts and total here?

            # Create the model time data
            model.time = TimeModel()
            model.time.dt = wp.zeros(num_worlds, dtype=float32)
            model.time.inv_dt = wp.zeros(num_worlds, dtype=float32)

            # Construct model gravity data
            model.gravity = GravityModel()
            model.gravity.g_dir_acc = wp.array(gravity_g_dir_acc, dtype=vec4f)
            model.gravity.vector = wp.array(gravity_vector, dtype=vec4f, requires_grad=requires_grad)

            # Create the bodies model
            model.bodies = RigidBodiesModel()
            model.bodies.num_bodies = model.size.sum_of_num_bodies
            model.bodies.wid = wp.array(bodies_wid, dtype=int32)
            model.bodies.bid = wp.array(bodies_bid, dtype=int32)
            model.bodies.m_i = wp.array(bodies_m_i, dtype=float32, requires_grad=requires_grad)
            model.bodies.inv_m_i = wp.array(bodies_inv_m_i, dtype=float32, requires_grad=requires_grad)
            model.bodies.i_I_i = wp.array(bodies_i_I_i, dtype=mat33f, requires_grad=requires_grad)
            model.bodies.inv_i_I_i = wp.array(bodies_inv_i_I_i, dtype=mat33f, requires_grad=requires_grad)
            model.bodies.q_i_0 = wp.array(bodies_q_i_0, dtype=transformf, requires_grad=requires_grad)
            model.bodies.u_i_0 = wp.array(bodies_u_i_0, dtype=vec6f, requires_grad=requires_grad)

            # Create the joints model
            model.joints = JointsModel()
            model.joints.num_joints = model.size.sum_of_num_joints
            model.joints.wid = wp.array(joints_wid, dtype=int32)
            model.joints.jid = wp.array(joints_jid, dtype=int32)
            model.joints.dof_type = wp.array(joints_dofid, dtype=int32)
            model.joints.act_type = wp.array(joints_actid, dtype=int32)
            model.joints.num_coords = wp.array(joints_c_j, dtype=int32)
            model.joints.num_dofs = wp.array(joints_d_j, dtype=int32)
            model.joints.num_cts = wp.array(joints_m_j, dtype=int32)
            model.joints.coords_offset = wp.array(joints_qio, dtype=int32)
            model.joints.dofs_offset = wp.array(joints_dio, dtype=int32)
            model.joints.passive_coords_offset = wp.array(joints_pqio, dtype=int32)
            model.joints.passive_dofs_offset = wp.array(joints_pdio, dtype=int32)
            model.joints.actuated_coords_offset = wp.array(joints_aqio, dtype=int32)
            model.joints.actuated_dofs_offset = wp.array(joints_adio, dtype=int32)
            model.joints.cts_offset = wp.array(joints_cio, dtype=int32)
            model.joints.bid_B = wp.array(joints_bid_B, dtype=int32)
            model.joints.bid_F = wp.array(joints_bid_F, dtype=int32)
            model.joints.B_r_Bj = wp.array(joints_B_r_Bj, dtype=vec3f, requires_grad=requires_grad)
            model.joints.F_r_Fj = wp.array(joints_F_r_Fj, dtype=vec3f, requires_grad=requires_grad)
            model.joints.X_j = wp.array(joints_X_j, dtype=mat33f, requires_grad=requires_grad)
            model.joints.q_j_min = wp.array(joints_q_j_min, dtype=float32, requires_grad=requires_grad)
            model.joints.q_j_max = wp.array(joints_q_j_max, dtype=float32, requires_grad=requires_grad)
            model.joints.dq_j_max = wp.array(joints_qd_j_max, dtype=float32, requires_grad=requires_grad)
            model.joints.tau_j_max = wp.array(joints_tau_j_max, dtype=float32, requires_grad=requires_grad)

            # Create the collision geometries model
            model.cgeoms = CollisionGeometriesModel()
            model.cgeoms.num_geoms = model.size.sum_of_num_collision_geoms
            model.cgeoms.wid = wp.array(cgeoms_wid, dtype=int32)
            model.cgeoms.gid = wp.array(cgeoms_gid, dtype=int32)
            model.cgeoms.lid = wp.array(cgeoms_lid, dtype=int32)
            model.cgeoms.bid = wp.array(cgeoms_bid, dtype=int32)
            model.cgeoms.sid = wp.array(cgeoms_sid, dtype=int32)
            model.cgeoms.ptr = wp.array(cgeoms_ptr, dtype=wp.uint64)
            model.cgeoms.params = wp.array(cgeoms_params, dtype=vec4f, requires_grad=requires_grad)
            model.cgeoms.offset = wp.array(cgeoms_offset, dtype=transformf, requires_grad=requires_grad)
            model.cgeoms.mid = wp.array(cgeoms_mid, dtype=int32)
            model.cgeoms.group = wp.array(cgeoms_group, dtype=uint32)
            model.cgeoms.collides = wp.array(cgeoms_collides, dtype=uint32)

            # Create the physical geometries model
            model.pgeoms = GeometriesModel()
            model.pgeoms.num_geoms = model.size.sum_of_num_physical_geoms
            model.pgeoms.wid = wp.array(pgeoms_wid, dtype=int32)
            model.pgeoms.gid = wp.array(pgeoms_gid, dtype=int32)
            model.pgeoms.lid = wp.array(pgeoms_lid, dtype=int32)
            model.pgeoms.bid = wp.array(pgeoms_bid, dtype=int32)
            model.pgeoms.sid = wp.array(pgeoms_sid, dtype=int32)
            model.pgeoms.ptr = wp.array(pgeoms_ptr, dtype=wp.uint64)
            model.pgeoms.params = wp.array(pgeoms_params, dtype=vec4f, requires_grad=requires_grad)
            model.pgeoms.offset = wp.array(pgeoms_offset, dtype=transformf, requires_grad=requires_grad)

            # Create the material pairs model
            model.mpairs = MaterialPairsModel()
            model.mpairs.num_pairs = model.size.sum_of_num_material_pairs
            model.mpairs.restitution = wp.array(mpairs_rest, dtype=float32)
            model.mpairs.static_friction = wp.array(mpairs_static_fric, dtype=float32)
            model.mpairs.dynamic_friction = wp.array(mpairs_dynamic_fric, dtype=float32)

        # Return the constructed model data container
        return model
