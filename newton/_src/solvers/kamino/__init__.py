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

"""KAMINO"""

from .core import math
from .core.bodies import RigidBodiesData, RigidBodiesModel, RigidBodyDescriptor
from .core.builder import (
    ModelBuilder,
    WorldDescriptor,
)
from .core.control import (
    Control,
)
from .core.geometry import (
    CollisionGeometriesData,
    CollisionGeometriesModel,
    CollisionGeometryDescriptor,
    GeometriesData,
    GeometriesModel,
    GeometryDescriptor,
)
from .core.gravity import (
    GRAVITY_ACCEL_DEFAULT,
    GRAVITY_DIREC_DEFAULT,
    GRAVITY_NAME_DEFAULT,
    GravityDescriptor,
    GravityModel,
)
from .core.joints import JointActuationType, JointDescriptor, JointDoFType, JointsData, JointsModel
from .core.materials import (
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    MaterialDescriptor,
    MaterialManager,
    MaterialPairProperties,
    MaterialPairsModel,
)
from .core.model import (
    Model,
    ModelData,
    ModelDataInfo,
    ModelInfo,
)
from .core.shapes import (
    BoxShape,
    CapsuleShape,
    ConeShape,
    CylinderShape,
    EllipsoidShape,
    EmptyShape,
    MeshShape,
    PlaneShape,
    SDFShape,
    ShapeDescriptor,
    ShapeType,
    SphereShape,
)
from .core.state import (
    State,
)
from .core.time import (
    TimeData,
    TimeModel,
)
from .core.types import (
    ArrayLike,
    FloatArrayLike,
    IntArrayLike,
    Mat33,
    Quat,
    Transform,
    Vec3,
    Vec4,
    Vec6,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    mat22f,
    mat26f,
    mat33f,
    mat36f,
    mat44f,
    mat46f,
    mat56f,
    mat62f,
    mat63f,
    mat64f,
    mat65f,
    mat66f,
    transformf,
    uint8,
    uint16,
    uint32,
    uint64,
    vec2f,
    vec2i,
    vec3f,
    vec4f,
    vec6f,
    vec6i,
    vec7f,
    vec14f,
)
from .dynamics.delassus import (
    DelassusOperator,
)
from .dynamics.dual import (
    DualProblem,
    DualProblemData,
)
from .geometry.collisions import Collisions, CollisionsData, CollisionsModel
from .geometry.contacts import (
    Contacts,
    ContactsData,
)
from .geometry.detector import (
    CollisionDetector,
)
from .kinematics.jacobians import (
    DenseSystemJacobians,
    DenseSystemJacobiansData,
)
from .simulation import (
    Simulator,
    SimulatorData,
    SimulatorSettings,
)
from .solvers import (
    PADMMSettings,
    PADMMSolver,
)
from .utils.io import hdf5, usd

###
# Package interface
###

__all__ = [
    "DEFAULT_FRICTION",
    "DEFAULT_RESTITUTION",
    "GRAVITY_ACCEL_DEFAULT",
    "GRAVITY_DIREC_DEFAULT",
    "GRAVITY_NAME_DEFAULT",
    "ArrayLike",
    "BoxShape",
    "CapsuleShape",
    "CollisionDetector",
    "CollisionGeometriesData",
    "CollisionGeometriesModel",
    "CollisionGeometryDescriptor",
    "Collisions",
    "CollisionsData",
    "CollisionsModel",
    "ConeShape",
    "Contacts",
    "ContactsData",
    "Control",
    "CylinderShape",
    "DelassusOperator",
    "DelassusOperatorData",
    "DenseSystemJacobians",
    "DenseSystemJacobiansData",
    "DualProblem",
    "DualProblemData",
    "EllipsoidShape",
    "EmptyShape",
    "FloatArrayLike",
    "ForwardKinematicsSolver",
    "GeometriesData",
    "GeometriesModel",
    "GeometryDescriptor",
    "GravityDescriptor",
    "GravityModel",
    "IntArrayLike",
    "JointActuationType",
    "JointDescriptor",
    "JointDoFType",
    "JointsData",
    "JointsModel",
    "Mat33",
    "MaterialDescriptor",
    "MaterialManager",
    "MaterialPairProperties",
    "MaterialPairsModel",
    "MeshShape",
    "Model",
    "ModelBuilder",
    "ModelData",
    "ModelDataInfo",
    "ModelInfo",
    "PADMMSettings",
    "PADMMSolver",
    "PlaneShape",
    "Quat",
    "RigidBodiesData",
    "RigidBodiesModel",
    "RigidBodyDescriptor",
    "SDFShape",
    "ShapeDescriptor",
    "ShapeType",
    "Simulator",
    "SimulatorData",
    "SimulatorSettings",
    "SphereShape",
    "State",
    "TimeData",
    "TimeModel",
    "Transform",
    "Vec3",
    "Vec4",
    "Vec6",
    "WorldDescriptor",
    "__version__",
    "float16",
    "float32",
    "float64",
    "hdf5",
    "int8",
    "int16",
    "int32",
    "int64",
    "mat22f",
    "mat26f",
    "mat33f",
    "mat36f",
    "mat44f",
    "mat46f",
    "mat56f",
    "mat62f",
    "mat63f",
    "mat64f",
    "mat65f",
    "mat66f",
    "math",
    "transformf",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "usd",
    "vec2f",
    "vec2i",
    "vec3f",
    "vec4f",
    "vec6f",
    "vec6i",
    "vec7f",
    "vec14f",
]
