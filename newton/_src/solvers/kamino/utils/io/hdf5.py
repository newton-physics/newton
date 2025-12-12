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
KAMINO: UTILS: Input/Output: HDF5
"""

import ctypes

import h5py
import numpy as np

from ...core.builder import WorldDescriptor
from ...core.joints import JointActuationType, JointDoFType
from ...simulation.simulator import Simulator

###
# Helper functions
###

# TODO: Add helper to check that 'dataset' is a valid HDF5 dataset


def joint_dof_name_from_id(dofid: int) -> str:
    name = "Unknown"
    if dofid == JointDoFType.FREE:
        name = "Free"
    elif dofid == JointDoFType.REVOLUTE:
        name = "Revolute"
    elif dofid == JointDoFType.PRISMATIC:
        name = "Prismatic"
    elif dofid == JointDoFType.CYLINDRICAL:
        name = "Cylindrical"
    elif dofid == JointDoFType.UNIVERSAL:
        name = "Universal"
    elif dofid == JointDoFType.SPHERICAL:
        name = "Spherical"
    elif dofid == JointDoFType.GIMBAL:
        name = "Gimbal"
    elif dofid == JointDoFType.CARTESIAN:
        name = "Cartesian"
    elif dofid == JointDoFType.FIXED:
        name = "Fixed"
    return name


def joint_type_name_from_id(typeid: int) -> str:
    name = "Unknown"
    if typeid == JointActuationType.PASSIVE:
        name = "Passive"
    elif typeid == JointActuationType.FORCE:
        name = "ForceControlled"
    return name


###
# Containers
###


# NumPy-based container for the RigidBody data loaded from HDF5
class RigidBodyData:
    def __init__(self, dataset=None, dtype=np.float64):
        self.name: str | None = None
        self.uid: str | None = None
        self.m_i: float = 0.0
        self.i_I_i: np.ndarray = np.zeros((3, 3), dtype=dtype)
        self.s_i_0: np.ndarray = np.zeros((14,), dtype=dtype)
        self.s_i: np.ndarray = np.zeros((14,), dtype=dtype)
        self.w_i: np.ndarray = np.zeros((6,), dtype=dtype)
        self.w_a_i: np.ndarray = np.zeros((6,), dtype=dtype)
        self.w_j_i: np.ndarray = np.zeros((6,), dtype=dtype)
        self.w_l_i: np.ndarray = np.zeros((6,), dtype=dtype)
        self.w_c_i: np.ndarray = np.zeros((6,), dtype=dtype)
        self.w_e_i: np.ndarray = np.zeros((6,), dtype=dtype)
        if dataset is not None:
            self.load(dataset, dtype)

    def __repr__(self):
        return f"RigidBodyData(\
            \nname={self.name},\
            \nuid={self.uid},\
            \nm_i={self.m_i},\
            \ni_I_i=\n{self.i_I_i},\
            \ns_i_0={self.s_i_0},\
            \ns_i={self.s_i},\
            \nw_i={self.w_i},\
            \nw_a_i={self.w_a_i},\
            \nw_j_i={self.w_j_i},\
            \nw_l_i={self.w_l_i},\
            \nw_c_i={self.w_c_i},\
            \nw_e_i={self.w_e_i})"

    def load(self, dataset, dtype=np.float64):
        # Load the data from the dataset
        self.name = dataset["name"][()].decode("UTF-8")
        self.uid = dataset["uid"][()].decode("UTF-8")
        self.m_i = dataset["m_i"][()].astype(dtype)
        self.i_I_i[:] = dataset["i_I_i"][:].astype(dtype)
        self.s_i_0[:] = dataset["s_i_0"][:].astype(dtype)
        self.s_i[:] = dataset["s_i"][:].astype(dtype)
        self.w_i[:] = dataset["w_i"][:].astype(dtype)
        self.w_a_i[:] = dataset["w_a_i"][:].astype(dtype)
        self.w_j_i[:] = dataset["w_j_i"][:].astype(dtype)
        self.w_l_i[:] = dataset["w_l_i"][:].astype(dtype)
        self.w_c_i[:] = dataset["w_c_i"][:].astype(dtype)
        self.w_e_i[:] = dataset["w_e_i"][:].astype(dtype)

    def store(self, dataset, namespace: str = ""):
        # Store the data into the dataset
        dataset[namespace + "/name"] = self.name.encode("UTF-8")
        dataset[namespace + "/uid"] = self.uid.encode("UTF-8")
        dataset[namespace + "/m_i"] = ctypes.c_double(self.m_i)
        dataset[namespace + "/i_I_i"] = self.i_I_i.astype(np.float64)
        dataset[namespace + "/s_i_0"] = self.s_i_0.astype(np.float64)
        dataset[namespace + "/s_i"] = self.s_i.astype(np.float64)
        dataset[namespace + "/w_i"] = self.w_i.astype(np.float64)
        dataset[namespace + "/w_a_i"] = self.w_a_i.astype(np.float64)
        dataset[namespace + "/w_j_i"] = self.w_j_i.astype(np.float64)
        dataset[namespace + "/w_l_i"] = self.w_l_i.astype(np.float64)
        dataset[namespace + "/w_c_i"] = self.w_c_i.astype(np.float64)
        dataset[namespace + "/w_e_i"] = self.w_e_i.astype(np.float64)


# NumPy-based container for the Joint data loaded from HDF5
class JointData:
    def __init__(self, dataset=None, dtype=np.float64, itype=np.int32):
        self.name: str | None = None
        self.uid: str | None = None
        self.dofs: str | None = None
        self.type: str | None = None
        self.base_id: int = -1
        self.follower_id: int = -1
        self.frame: np.ndarray = np.zeros((15,), dtype=dtype)
        if dataset is not None:
            self.load(dataset, dtype, itype)

    def __repr__(self):
        return f"JointData(\
            \nname={self.name},\
            \nuid={self.uid},\
            \nframe=\n{self.frame},\
            \ndofs={self.dofs},\
            \ntype={self.type},\
            \nbase_id={self.base_id},\
            \nfollower_id={self.follower_id})"

    def load(self, dataset, dtype=np.float64, itype=np.int32):
        self.name = dataset["name"][()].decode("UTF-8")
        self.uid = dataset["uid"][()].decode("UTF-8")
        self.dofs = dataset["dofs"][()].decode("UTF-8")
        self.type = dataset["type"][()].decode("UTF-8")
        self.base_id = dataset["base_id"][()].astype(itype)
        self.follower_id = dataset["follower_id"][()].astype(itype)
        self.frame = dataset["frame"][:].astype(dtype)

    def store(self, dataset, namespace: str = ""):
        dataset[namespace + "/name"] = self.name.encode("UTF-8")
        dataset[namespace + "/uid"] = self.uid.encode("UTF-8")
        dataset[namespace + "/dofs"] = self.dofs.encode("UTF-8")
        dataset[namespace + "/type"] = self.type.encode("UTF-8")
        dataset[namespace + "/base_id"] = ctypes.c_int(self.base_id)
        dataset[namespace + "/follower_id"] = ctypes.c_int(self.follower_id)
        dataset[namespace + "/frame"] = self.frame.astype(np.float64)


class GravityData:
    def __init__(self, dataset=None, dtype=np.float64):
        self.enabled: bool = False
        self.acceleration: float = 0.0
        self.direction: np.ndarray = np.zeros((3,), dtype=dtype)
        self.vector: np.ndarray = np.zeros((3,), dtype=dtype)
        if dataset is not None:
            self.load(dataset, dtype)

    def __repr__(self):
        return f"GravityData(\
            \nenabled={self.enabled},\
            \nacceleration={self.acceleration},\
            \ndirection={self.direction},\
            \nvector={self.vector})"

    def load(self, dataset, dtype=np.float64):
        self.enabled = dataset["enabled"][()]
        self.acceleration = dataset["acceleration"][()].astype(dtype)
        self.direction[:] = dataset["direction"][:].astype(dtype)
        self.vector[:] = dataset["vector"][:].astype(dtype)

    def store(self, dataset, namespace: str = ""):
        dataset[namespace + "/enabled"] = self.enabled
        dataset[namespace + "/acceleration"] = ctypes.c_double(self.acceleration)
        dataset[namespace + "/direction"] = self.direction.astype(np.float64)
        dataset[namespace + "/vector"] = self.vector.astype(np.float64)


# NumPy-based container for the RigidBodySystemInfo data loaded from HDF5
class RigidBodySystemInfoData:
    def __init__(self, dataset=None, dtype=np.float64, itype=np.int32):
        self.nsd: int = 0
        self.nb: int = 0
        self.nbd: int = 0
        self.nj: int = 0
        self.njd: int = 0
        self.np: int = 0
        self.npd: int = 0
        self.nq: int = 0
        self.nqd: int = 0
        self.njdims: list[int] = []
        self.npdims: list[int] = []
        self.nqdims: list[int] = []
        self.body_names: list[str] = []
        self.joint_names: list[str] = []
        self.physical_geometry_layers: list[str] = []
        self.collision_geometry_layers: list[str] = []
        self.fixed_joint_names: list[str] = []
        self.passive_joint_names: list[str] = []
        self.actuated_joint_names: list[str] = []
        self.base_name: str = ""
        self.base_idx: int = -1
        self.grounding_name: str = ""
        self.grounding_idx: int = -1
        self.has_base: bool = False
        self.has_floating_base: bool = False
        self.has_grounding: bool = False
        self.has_passive_dofs: bool = False
        self.has_actuated_dofs: bool = False
        self.total_mass: float = 0.0
        self.total_diagonal_inertia: float = 0.0
        if dataset is not None:
            self.load(dataset, dtype, itype)

    def __repr__(self):
        return f"RigidBodySystemInfoData(\
            \nnsd={self.nsd}, \
            \nnb={self.nb}, \
            \nnbd={self.nbd}, \
            \nnj={self.nj}, \
            \nnjd={self.njd}, \
            \nnp={self.np}, \
            \nnpd={self.npd}, \
            \nnq={self.nq}, \
            \nnqd={self.nqd}, \
            \nnjdims={self.njdims}, \
            \nnpdims={self.npdims}, \
            \nnqdims={self.nqdims}, \
            \nbody_names={self.body_names}, \
            \njoint_names={self.joint_names}, \
            \nphysical_geometry_layers={self.physical_geometry_layers}, \
            \ncollision_geometry_layers={self.collision_geometry_layers}, \
            \nfixed_joint_names={self.fixed_joint_names}, \
            \npassive_joint_names={self.passive_joint_names}, \
            \nactuated_joint_names={self.actuated_joint_names}, \
            \nbase_name={self.base_name}, \
            \nbase_idx={self.base_idx}, \
            \ngrounding_name={self.grounding_name}, \
            \ngrounding_idx={self.grounding_idx}, \
            \nhas_base={self.has_base}, \
            \nhas_floating_base={self.has_floating_base}, \
            \nhas_grounding={self.has_grounding}, \
            \nhas_passive_dofs={self.has_passive_dofs}, \
            \nhas_actuated_dofs={self.has_actuated_dofs}, \
            \ntotal_mass={self.total_mass}, \
            \ntotal_diagonal_inertia={self.total_diagonal_inertia})"

    def load(self, dataset, dtype=np.float64, itype=np.int32):
        self.nsd = dataset["nsd"][()].astype(itype)
        self.nb = dataset["nb"][()].astype(itype)
        self.nbd = dataset["nbd"][()].astype(itype)
        self.nj = dataset["nj"][()].astype(itype)
        self.njd = dataset["njd"][()].astype(itype)
        self.np = dataset["np"][()].astype(itype)
        self.npd = dataset["npd"][()].astype(itype)
        self.nq = dataset["nq"][()].astype(itype)
        self.nqd = dataset["nqd"][()].astype(itype)
        self.njdims = dataset["njdims"][:].astype(itype)
        self.npdims = dataset["npdims"][:].astype(itype)
        self.nqdims = dataset["nqdims"][:].astype(itype)
        self.body_names = [str(s.decode("UTF-8")) for s in dataset["body_names"]]
        # self.joint_names = [str(s.decode('UTF-8')) for s in dataset['joint_names']]
        self.physical_geometry_layers = [str(s.decode("UTF-8")) for s in dataset["physical_geometry_layers"]]
        self.collision_geometry_layers = [str(s.decode("UTF-8")) for s in dataset["collision_geometry_layers"]]
        self.fixed_joint_names = [str(s.decode("UTF-8")) for s in dataset["fixed_joint_names"]]
        self.passive_joint_names = [str(s.decode("UTF-8")) for s in dataset["passive_joint_names"]]
        self.actuated_joint_names = [str(s.decode("UTF-8")) for s in dataset["actuated_joint_names"]]
        self.base_name = dataset["base_name"][()].decode("UTF-8")
        self.base_idx = dataset["base_idx"][()].astype(itype)
        self.grounding_name = dataset["grounding_name"][()].decode("UTF-8")
        self.grounding_idx = dataset["grounding_idx"][()].astype(itype)
        self.has_base = dataset["has_base"][()]
        self.has_floating_base = dataset["has_floating_base"][()]
        self.has_grounding = dataset["has_grounding"][()]
        self.has_passive_dofs = dataset["has_passive_dofs"][()]
        self.has_actuated_dofs = dataset["has_actuated_dofs"][()]
        self.total_mass = dataset["total_mass"][()].astype(dtype)
        self.total_diagonal_inertia = dataset["total_diagonal_inertia"][()].astype(dtype)

    def store(self, dataset, namespace: str = ""):
        dataset[namespace + "/nsd"] = ctypes.c_long(self.nsd)
        dataset[namespace + "/nb"] = ctypes.c_long(self.nb)
        dataset[namespace + "/nbd"] = ctypes.c_long(self.nbd)
        dataset[namespace + "/nj"] = ctypes.c_long(self.nj)
        dataset[namespace + "/njd"] = ctypes.c_long(self.njd)
        dataset[namespace + "/np"] = ctypes.c_long(self.np)
        dataset[namespace + "/npd"] = ctypes.c_long(self.npd)
        dataset[namespace + "/nq"] = ctypes.c_long(self.nq)
        dataset[namespace + "/nqd"] = ctypes.c_long(self.nqd)
        dataset[namespace + "/njdims"] = (ctypes.c_int * len(self.njdims))(*self.njdims)
        dataset[namespace + "/npdims"] = (ctypes.c_int * len(self.npdims))(*self.npdims)
        dataset[namespace + "/nqdims"] = (ctypes.c_int * len(self.nqdims))(*self.nqdims)
        dataset[namespace + "/body_names"] = [s.encode("UTF-8") for s in self.body_names]
        dataset[namespace + "/joint_names"] = [s.encode("UTF-8") for s in self.joint_names]
        dataset[namespace + "/physical_geometry_layers"] = [s.encode("UTF-8") for s in self.physical_geometry_layers]
        dataset[namespace + "/collision_geometry_layers"] = [s.encode("UTF-8") for s in self.collision_geometry_layers]
        dataset[namespace + "/fixed_joint_names"] = [s.encode("UTF-8") for s in self.fixed_joint_names]
        dataset[namespace + "/passive_joint_names"] = [s.encode("UTF-8") for s in self.passive_joint_names]
        dataset[namespace + "/actuated_joint_names"] = [s.encode("UTF-8") for s in self.actuated_joint_names]
        dataset[namespace + "/base_name"] = self.base_name.encode("UTF-8")
        dataset[namespace + "/base_idx"] = ctypes.c_long(self.base_idx)
        dataset[namespace + "/grounding_name"] = self.grounding_name.encode("UTF-8")
        dataset[namespace + "/grounding_idx"] = ctypes.c_long(self.grounding_idx)
        dataset[namespace + "/has_base"] = ctypes.c_bool(self.has_base)
        dataset[namespace + "/has_floating_base"] = ctypes.c_bool(self.has_floating_base)
        dataset[namespace + "/has_grounding"] = ctypes.c_bool(self.has_grounding)
        dataset[namespace + "/has_passive_dofs"] = ctypes.c_bool(self.has_passive_dofs)
        dataset[namespace + "/has_actuated_dofs"] = ctypes.c_bool(self.has_actuated_dofs)
        dataset[namespace + "/total_mass"] = ctypes.c_double(self.total_mass)
        dataset[namespace + "/total_diagonal_inertia"] = ctypes.c_double(self.total_diagonal_inertia)

    def from_descriptor(self, world: WorldDescriptor):
        self.nb = world.num_bodies
        self.nbd = world.num_body_dofs
        self.nj = world.num_joints
        self.njd = world.num_joint_cts
        self.np = world.num_passive_joints
        self.npd = world.num_passive_joint_dofs
        self.nq = world.num_actuated_joints
        self.nqd = world.num_actuated_joint_dofs
        self.nsd = self.nbd + self.njd
        self.njdims = world.joint_cts
        self.npdims = world.joint_passive_dofs
        self.nqdims = world.joint_actuated_dofs
        self.body_names = world.body_names
        self.joint_names = world.joint_names
        self.physical_geometry_layers = world.physical_geometry_layers
        self.collision_geometry_layers = world.collision_geometry_layers
        self.fixed_joint_names = world.fixed_joint_names
        self.passive_joint_names = world.passive_joint_names
        self.actuated_joint_names = world.actuated_joint_names
        self.base_name = world.base_name
        self.base_idx = world.base_idx
        self.grounding_name = world.grounding_name
        self.grounding_idx = world.grounding_idx
        self.has_base = world.has_base
        self.has_floating_base = not world.has_grounding  # TODO: Check this logic
        self.has_grounding = world.has_grounding
        self.has_passive_dofs = world.has_passive_dofs
        self.has_actuated_dofs = world.has_actuated_dofs
        self.total_mass = world.mass_total
        self.total_diagonal_inertia = world.inertia_total


# NumPy-based container for the RigidBodySystem data loaded from HDF5
class RigidBodySystemData:
    def __init__(self, dataset=None, dtype=np.float64, itype=np.int32):
        self._dtype = dtype
        self._itype = itype
        self.gravity = GravityData()
        self.info = RigidBodySystemInfoData()
        self.bodies: list[RigidBodyData] | None = None
        self.joints: list[JointData] = None
        self.p_design: np.ndarray | None = None
        self.p_control: np.ndarray | None = None
        self.state: np.ndarray | None = None
        self.state_p: np.ndarray | None = None
        self.state_0: np.ndarray | None = None
        self.state_0_pgo: np.ndarray | None = None
        self.state_0_cgo: np.ndarray | None = None
        self.lambda_j: np.ndarray | None = None
        self.lambda_l: np.ndarray | None = None
        self.lambda_c: np.ndarray | None = None
        self.w_a: np.ndarray | None = None
        self.w_j: np.ndarray | None = None
        self.w_l: np.ndarray | None = None
        self.w_c: np.ndarray | None = None
        self.p_j: np.ndarray | None = None
        self.p_j_0: np.ndarray | None = None
        self.p_j_min: np.ndarray | None = None
        self.p_j_max: np.ndarray | None = None
        self.dp_j: np.ndarray | None = None
        self.dp_j_0: np.ndarray | None = None
        self.q_j: np.ndarray | None = None
        self.q_j_0: np.ndarray | None = None
        self.q_j_min: np.ndarray | None = None
        self.q_j_max: np.ndarray | None = None
        self.dq_j: np.ndarray | None = None
        self.dq_j_0: np.ndarray | None = None
        self.tau_j: np.ndarray | None = None
        self.R_b_frame_to_com: np.ndarray | None = None
        self.r_b_frame_to_com: np.ndarray | None = None
        self.R_b: np.ndarray | None = None
        self.R_b_0: np.ndarray | None = None
        self.r_b: np.ndarray | None = None
        self.r_b_0: np.ndarray | None = None
        self.omega_b: np.ndarray | None = None
        self.omega_b_0: np.ndarray | None = None
        self.v_b: np.ndarray | None = None
        self.v_b_0: np.ndarray | None = None
        self.T_total: float = 0.0
        self.U_total: float = 0.0
        self.E_total: float = 0.0
        self.category: int = 0
        self.collides: int = 0
        # Construct the system data from the dataset if specified
        if dataset is not None:
            self.load(dataset, dtype, itype)

    def __repr__(self):
        return f"RigidBodySystemData(\ninfo={self.info}\nbodies={self.bodies}\njoints={self.joints})"

    def load(self, dataset, dtype=np.float64, itype=np.int32):
        self._dtype = dtype
        self._itype = itype
        self.gravity.load(dataset["gravity"], dtype)
        self.info.load(dataset["info"], dtype, itype)
        self.bodies = [RigidBodyData(dataset[f"bodies/{i}"], dtype) for i in range(self.info.nb)]
        self.joints = [JointData(dataset[f"joints/{j}"], dtype) for j in range(self.info.nj)]

    def store(self, dataset, namespace: str = ""):
        self.gravity.store(dataset, namespace=f"{namespace}/gravity")
        self.info.store(dataset, namespace=f"{namespace}/info")
        for i, body in enumerate(self.bodies):
            body.store(dataset, namespace=f"{namespace}/bodies/{i}")
        for j, joint in enumerate(self.joints):
            joint.store(dataset, namespace=f"{namespace}/joints/{j}")
        # Store additional system data
        dataset[namespace + "/state/p_design"] = self.p_design
        dataset[namespace + "/state/p_control"] = self.p_control
        dataset[namespace + "/state/state"] = self.state
        dataset[namespace + "/state/state_p"] = self.state_p
        dataset[namespace + "/state/state_0"] = self.state_0
        dataset[namespace + "/geometry/physical/state_0"] = self.state_0_pgo
        dataset[namespace + "/geometry/collision/state_0"] = self.state_0_cgo
        dataset[namespace + "/multipliers/lambda_j"] = self.lambda_j
        dataset[namespace + "/multipliers/lambda_l"] = self.lambda_l
        dataset[namespace + "/multipliers/lambda_c"] = self.lambda_c
        dataset[namespace + "/wrenches/w_a"] = self.w_a
        dataset[namespace + "/wrenches/w_j"] = self.w_j
        dataset[namespace + "/wrenches/w_l"] = self.w_l
        dataset[namespace + "/wrenches/w_c"] = self.w_c
        dataset[namespace + "/system/p_j"] = self.p_j
        dataset[namespace + "/system/p_j_0"] = self.p_j_0
        dataset[namespace + "/system/p_j_min"] = self.p_j_min
        dataset[namespace + "/system/p_j_max"] = self.p_j_max
        dataset[namespace + "/system/dp_j"] = self.dp_j
        dataset[namespace + "/system/dp_j_0"] = self.dp_j_0
        dataset[namespace + "/system/q_j"] = self.q_j
        dataset[namespace + "/system/q_j_0"] = self.q_j_0
        dataset[namespace + "/system/q_j_min"] = self.q_j_min
        dataset[namespace + "/system/q_j_max"] = self.q_j_max
        dataset[namespace + "/system/dq_j"] = self.dq_j
        dataset[namespace + "/system/dq_j_0"] = self.dq_j_0
        dataset[namespace + "/system/tau_j"] = self.tau_j
        dataset[namespace + "/system/R_b_frame_to_com"] = self.R_b_frame_to_com
        dataset[namespace + "/system/r_b_frame_to_com"] = self.r_b_frame_to_com
        dataset[namespace + "/system/R_b"] = self.R_b
        dataset[namespace + "/system/R_b_0"] = self.R_b_0
        dataset[namespace + "/system/r_b"] = self.r_b
        dataset[namespace + "/system/r_b_0"] = self.r_b_0
        dataset[namespace + "/system/omega_b"] = self.omega_b
        dataset[namespace + "/system/omega_b_0"] = self.omega_b_0
        dataset[namespace + "/system/v_b"] = self.v_b
        dataset[namespace + "/system/v_b_0"] = self.v_b_0
        dataset[namespace + "/energy/T_total"] = ctypes.c_double(self.T_total)
        dataset[namespace + "/energy/U_total"] = ctypes.c_double(self.U_total)
        dataset[namespace + "/energy/E_total"] = ctypes.c_double(self.E_total)
        dataset[namespace + "/collision/category"] = ctypes.c_ulong(self.category)
        dataset[namespace + "/collision/collides"] = ctypes.c_ulong(self.collides)

    def configure(self, simulator: Simulator, wid: int = 0, dtype=np.float64):
        # Configure gravity
        gravity_g_dir_acc = simulator.model.gravity.g_dir_acc.numpy().astype(self._dtype)
        gravity_vector = simulator.model.gravity.vector.numpy().astype(self._dtype)
        self.gravity.enabled = bool(gravity_vector[wid, 3])
        self.gravity.acceleration = float(gravity_g_dir_acc[wid, 3])
        self.gravity.direction[:] = gravity_g_dir_acc[wid, 0:3]
        self.gravity.vector[:] = gravity_vector[wid, 0:3]
        # Configure the system information
        self.info.from_descriptor(simulator.model.worlds[wid])
        # Construct element data containers
        if self.bodies is None:
            self.bodies = [RigidBodyData() for _ in range(self.info.nb)]
        if self.joints is None:
            self.joints = [JointData() for _ in range(self.info.nj)]
        # Initialize time-invariant model data
        body_m_i = simulator.model.bodies.m_i.numpy().astype(self._dtype)
        body_i_I_i = simulator.model.bodies.i_I_i.numpy().astype(self._dtype)
        body_q_i_0 = simulator.model.bodies.q_i_0.numpy().astype(self._dtype)
        # body_u_i_0 = simulator.model.bodies.u_i_0.numpy().astype(self._dtype)
        joint_bid_B = simulator.model.joints.bid_B.numpy().astype(self._itype)
        joint_bid_F = simulator.model.joints.bid_F.numpy().astype(self._itype)
        joint_dofid = simulator.model.joints.dof_type.numpy().astype(self._itype)
        joint_actid = simulator.model.joints.act_type.numpy().astype(self._itype)
        joint_B_r_Bj = simulator.model.joints.B_r_Bj.numpy().astype(self._dtype)
        joint_F_r_Fj = simulator.model.joints.F_r_Fj.numpy().astype(self._dtype)
        joint_X_j = simulator.model.joints.X_j.numpy().astype(self._dtype)
        for i, body in enumerate(self.bodies):
            body.name = self.info.body_names[i]
            body.uid = self.info.body_names[i]
            body.m_i = body_m_i[i]
            body.i_I_i[:] = body_i_I_i[i, :, :]
            body.s_i_0[0:7] = body_q_i_0[i, :]
            # body.s_i_0[7:14] = to_quaternion_derivative(body_u_i_0[wid, :], self.bodies[i].s_i_0[3:6])
        for j, joint in enumerate(self.joints):
            joint.name = self.info.joint_names[j]
            joint.uid = self.info.joint_names[j]
            joint.base_id = joint_bid_B[j]
            joint.follower_id = joint_bid_F[j]
            joint.dofs = joint_dof_name_from_id(joint_dofid[j])
            joint.type = joint_type_name_from_id(joint_actid[j])
            joint.frame[0:9] = joint_X_j[j, :, :].flatten()
            joint.frame[9:12] = joint_B_r_Bj[j, :]
            joint.frame[12:15] = joint_F_r_Fj[j, :]
        # Configure additional system data
        self.p_design = np.zeros((0,), dtype=self._dtype)
        self.p_control = np.zeros((self.info.nqd,), dtype=self._dtype)
        self.state = np.zeros((self.info.nsd,), dtype=self._dtype)
        self.state_p = np.zeros((self.info.nsd,), dtype=self._dtype)
        self.state_0 = np.zeros((self.info.nsd,), dtype=self._dtype)
        self.state_0_pgo = np.zeros((self.info.nsd,), dtype=self._dtype)
        self.state_0_cgo = np.zeros((self.info.nsd,), dtype=self._dtype)
        self.lambda_j = np.zeros((self.info.njd,), dtype=self._dtype)
        self.lambda_l = np.zeros((self.info.njd,), dtype=self._dtype)
        self.lambda_c = np.zeros((self.info.njd,), dtype=self._dtype)
        self.w_a = np.zeros((self.info.nbd,), dtype=self._dtype)
        self.w_j = np.zeros((self.info.nbd,), dtype=self._dtype)
        self.w_l = np.zeros((self.info.nbd,), dtype=self._dtype)
        self.w_c = np.zeros((self.info.nbd,), dtype=self._dtype)
        self.p_j = np.zeros((self.info.npd,), dtype=self._dtype)
        self.p_j_0 = np.zeros((self.info.npd,), dtype=self._dtype)
        self.p_j_min = np.zeros((self.info.npd,), dtype=self._dtype)
        self.p_j_max = np.zeros((self.info.npd,), dtype=self._dtype)
        self.dp_j = np.zeros((self.info.npd,), dtype=self._dtype)
        self.dp_j_0 = np.zeros((self.info.npd,), dtype=self._dtype)
        self.q_j = np.zeros((self.info.nqd,), dtype=self._dtype)
        self.q_j_0 = np.zeros((self.info.nqd,), dtype=self._dtype)
        self.q_j_min = np.zeros((self.info.nqd,), dtype=self._dtype)
        self.q_j_max = np.zeros((self.info.nqd,), dtype=self._dtype)
        self.dq_j = np.zeros((self.info.nqd,), dtype=self._dtype)
        self.dq_j_0 = np.zeros((self.info.nqd,), dtype=self._dtype)
        self.tau_j = np.zeros((self.info.nqd,), dtype=self._dtype)
        self.R_b_frame_to_com = np.zeros((3, 3), dtype=self._dtype)
        self.r_b_frame_to_com = np.zeros((3,), dtype=self._dtype)
        self.R_b = np.zeros((3, 3), dtype=self._dtype)
        self.R_b_0 = np.zeros((3, 3), dtype=self._dtype)
        self.r_b = np.zeros((3,), dtype=self._dtype)
        self.r_b_0 = np.zeros((3,), dtype=self._dtype)
        self.omega_b = np.zeros((3,), dtype=self._dtype)
        self.omega_b_0 = np.zeros((3,), dtype=self._dtype)
        self.v_b = np.zeros((3,), dtype=self._dtype)
        self.v_b_0 = np.zeros((3,), dtype=self._dtype)
        self.T_total = 0.0
        self.U_total = 0.0
        self.E_total = 0.0
        self.category = 0
        self.collides = 0

    def _to_quaternion_derivative(self, u_i: np.ndarray, q_i: np.ndarray, dtype=np.float64) -> np.ndarray:
        # TODO: Convert angular velocity to the quaternion derivative
        dq_i = np.zeros((7,), dtype=dtype)
        return dq_i

    def update_from(self, simulator: Simulator, wid: int = 0, dtype=np.float64, itype=np.int32):
        # TODO: add check to ensure the host-side data exists
        # Extract numpy arrays from the system data
        bodies_wid = simulator.model.bodies.wid.numpy().astype(itype)
        bodies_q_i = simulator.host.state.bodies.q_i.numpy().astype(dtype)
        bodies_u_i = simulator.host.state.bodies.u_i.numpy().astype(dtype)
        bodies_w_i = simulator.host.state.bodies.w_i.numpy().astype(dtype)
        bodies_w_a_i = simulator.host.state.bodies.w_a_i.numpy().astype(dtype)
        bodies_w_j_i = simulator.host.state.bodies.w_j_i.numpy().astype(dtype)
        bodies_w_l_i = simulator.host.state.bodies.w_l_i.numpy().astype(dtype)
        bodies_w_c_i = simulator.host.state.bodies.w_c_i.numpy().astype(dtype)
        bodies_w_e_i = simulator.host.state.bodies.w_e_i.numpy().astype(dtype)
        for i in range(self.info.nb):
            if bodies_wid[i] == wid:
                self.bodies[i].s_i[0:7] = bodies_q_i[i, :]
                self.bodies[i].s_i[7:14] = self._to_quaternion_derivative(
                    bodies_u_i[i, :], self.bodies[i].s_i[3:7], dtype=dtype
                )
                self.bodies[i].w_i = bodies_w_i[i, :]
                self.bodies[i].w_a_i = bodies_w_a_i[i, :]
                self.bodies[i].w_j_i = bodies_w_j_i[i, :]
                self.bodies[i].w_l_i = bodies_w_l_i[i, :]
                self.bodies[i].w_c_i = bodies_w_c_i[i, :]
                self.bodies[i].w_e_i = bodies_w_e_i[i, :]


# NumPy-based container for a single Contact loaded from HDF5
class ContactData:
    def __init__(self, dataset=None, dtype=np.float64, itype=np.int32):
        # Geometry/Shape info
        self.gid_A: int = -1
        self.bid_A: int = -1
        self.position_A: np.ndarray = np.zeros((3,), dtype=dtype)
        self.gid_B: int = -1
        self.bid_B: int = -1
        self.position_B: np.ndarray = np.zeros((3,), dtype=dtype)
        # Discrete contact info
        self.frame: np.ndarray = np.zeros((3, 3), dtype=dtype)
        self.position: np.ndarray = np.zeros((3,), dtype=dtype)
        self.normal: np.ndarray = np.zeros((3,), dtype=dtype)
        self.penetration: float = 0.0
        self.friction: float = 0.0
        self.restitutuon: float = 0.0
        # Conic info
        self.d_n: np.ndarray = np.zeros((4,), dtype=dtype)
        self.ratio: float = 0.0
        self.type: int = 0
        # Reaction info
        self.lambda_0: np.ndarray = np.zeros((3,), dtype=dtype)
        self.lambda_n: np.ndarray = np.zeros((3,), dtype=dtype)
        self.v_plus: np.ndarray = np.zeros((3,), dtype=dtype)
        self.mode: int = 1
        self.bilateral: bool = False
        # If a dataset is provided, load the data from it
        if dataset is not None:
            self.load(dataset, dtype, itype)

    def __repr__(self):
        return f"ContactData(\
            \ngid_A={self.gid_A},\
            \nbid_A={self.bid_A},\
            \nposition_A={self.position_A},\
            \ngid_B={self.gid_B},\
            \nbid_B={self.bid_B},\
            \nposition_B={self.position_B},\
            \nframe=\n{self.frame},\
            \nposition={self.position},\
            \nnormal={self.normal},\
            \npenetration={self.penetration},\
            \nfriction={self.friction},\
            \nrestitutuon={self.restitutuon},\
            \nd_n={self.d_n},\
            \nratio={self.ratio},\
            \ntype={self.type},\
            \nlambda_0={self.lambda_0},\
            \nlambda_n={self.lambda_n},\
            \nv_plus={self.v_plus},\
            \nmode={self.mode},\
            \nbilateral={self.bilateral})"

    def load(self, dataset, dtype=np.float64, itype=np.int32):
        self.gid_A = dataset["point/A/gid"][()].astype(itype)
        self.bid_A = dataset["point/A/bid"][()].astype(itype)
        self.position_A[:] = dataset["point/A/position"][:].astype(dtype)
        self.gid_B = dataset["point/B/gid"][()].astype(itype)
        self.bid_B = dataset["point/B/bid"][()].astype(itype)
        self.position_B[:] = dataset["point/B/position"][:].astype(dtype)
        self.position[:] = dataset["point/position"][:].astype(dtype)
        self.normal[:] = dataset["point/normal"][:].astype(dtype)
        self.penetration = dataset["point/penetration"][()].astype(dtype)
        self.friction = dataset["material/mu"][()].astype(dtype)
        self.restitutuon = dataset["material/epsilon"][()].astype(dtype)
        self.d_n[:] = dataset["conic/d_n"][:].astype(dtype)
        self.ratio = float(dataset["conic/ratio"][()])
        self.type = int(dataset["conic/type"][()])
        self.frame[:] = dataset["reaction/frame"][:, :].astype(dtype)
        self.lambda_0[:] = dataset["reaction/lambda_0"][:].astype(dtype)
        self.lambda_n[:] = dataset["reaction/lambda"][:].astype(dtype)
        self.v_plus[:] = dataset["reaction/v_plus"][:].astype(dtype)
        self.mode = int(dataset["reaction/mode"][()])
        self.bilateral = bool(dataset["reaction/bilateral"][()])

    def store(self, dataset, namespace: str = ""):
        dataset[namespace + "/point/A/gid"] = ctypes.c_long(self.gid_A)
        dataset[namespace + "/point/A/bid"] = ctypes.c_long(self.bid_A)
        dataset[namespace + "/point/A/position"] = self.position_A.astype(np.float64)
        dataset[namespace + "/point/B/gid"] = ctypes.c_long(self.gid_B)
        dataset[namespace + "/point/B/bid"] = ctypes.c_long(self.bid_B)
        dataset[namespace + "/point/B/position"] = self.position_B.astype(np.float64)
        dataset[namespace + "/point/position"] = self.position.astype(np.float64)
        dataset[namespace + "/point/normal"] = self.normal.astype(np.float64)
        dataset[namespace + "/point/penetration"] = self.penetration
        dataset[namespace + "/material/mu"] = self.friction
        dataset[namespace + "/material/epsilon"] = self.restitutuon
        dataset[namespace + "/reaction/frame"] = self.frame.astype(np.float64)
        dataset[namespace + "/reaction/lambda_0"] = self.lambda_0.astype(np.float64)
        dataset[namespace + "/reaction/lambda"] = self.lambda_n.astype(np.float64)
        dataset[namespace + "/reaction/v_plus"] = self.v_plus.astype(np.float64)
        dataset[namespace + "/reaction/mode"] = ctypes.c_int8(self.mode)
        dataset[namespace + "/reaction/bilateral"] = ctypes.c_bool(self.bilateral)
        dataset[namespace + "/conic/d_n"] = self.d_n.astype(np.float64)
        dataset[namespace + "/conic/ratio"] = ctypes.c_double(self.ratio)
        dataset[namespace + "/conic/type"] = ctypes.c_long(self.type)


# NumPy-based container for multiple Contact data loaded from HDF5
class ContactsData:
    def __init__(self, dataset=None, dtype=np.float64, itype=np.int32):
        self.ncontacts = None
        self.contacts = []
        if dataset is not None:
            self.load(dataset, dtype, itype)

    def __repr__(self):
        return f"ContactsData(\
            \nncontacts={self.ncontacts},\
            \ncontacts={self.contacts})"

    def load(self, dataset, dtype=np.float64, itype=np.int32):
        # Load the data from the dataset
        self.ncontacts = dataset["nc"][()].astype(itype)
        self.contacts = [ContactData(dataset[f"contacts/{c}"], dtype) for c in range(self.ncontacts)]

    def store(self, dataset, namespace: str = ""):
        # Store the data into the dataset
        dataset[namespace + "/nc"] = ctypes.c_long(self.ncontacts)
        for c, contact in enumerate(self.contacts):
            contact.store(dataset, namespace=f"{namespace}/contacts/{c}")

    def update_from(self, simulator: Simulator, wid: int = 0, dtype=np.float64, itype=np.int32):
        self.ncontacts = int(simulator.contacts.world_active_contacts.numpy()[wid])
        self.contacts = [ContactData() for _ in range(self.ncontacts)]
        # Update the contact data from the contacts object if data is available
        if self.ncontacts > 0:
            # Extract numpy arrays from the system data
            contacts_wid = simulator.contacts._data.wid.numpy().astype(itype)
            contacts_body_A = simulator.contacts._data.body_A.numpy().astype(dtype)
            contacts_body_B = simulator.contacts._data.body_B.numpy().astype(dtype)
            contacts_gapfunc = simulator.contacts._data.gapfunc.numpy().astype(dtype)
            contacts_frame = simulator.contacts._data.frame.numpy().astype(dtype)
            contacts_material = simulator.contacts._data.material.numpy().astype(dtype)
            # Update the contact data
            for k in range(self.ncontacts):
                if contacts_wid[k] == wid:
                    self.contacts[k].gid_A = int(contacts_body_A[k, 3])
                    self.contacts[k].bid_A = int(contacts_body_A[k, 3])
                    self.contacts[k].position_A[:] = contacts_body_A[k, 0:3].astype(np.float64)
                    self.contacts[k].gid_B = int(contacts_body_B[k, 3])
                    self.contacts[k].bid_B = int(contacts_body_B[k, 3])
                    self.contacts[k].position_B[:] = contacts_body_B[k, 0:3].astype(np.float64)
                    self.contacts[k].frame[:] = contacts_frame[k, :, :]
                    self.contacts[k].position[:] = contacts_body_A[k, 0:3].astype(np.float64)
                    self.contacts[k].normal[:] = contacts_gapfunc[k, 0:3].astype(np.float64)
                    self.contacts[k].penetration = contacts_gapfunc[k, 3]
                    self.contacts[k].friction = contacts_material[k, 0]
                    self.contacts[k].restitutuon = contacts_material[k, 1]


# NumPy-based container for the DualProblemInfo data loaded from HDF5
class DualProblemInfoData:
    def __init__(self, dataset=None, itype=np.int32):
        self.njdims = None
        self.nb = None
        self.nj = None
        self.nl = None
        self.nc = None
        self.nbd = None
        self.njd = None
        self.nld = None
        self.ncd = None
        self.ij = None
        self.il = None
        self.ic = None
        self.nbc = None
        self.ncts = None
        if dataset is not None:
            self.load(dataset, itype)

    def __repr__(self):
        return f"DualProblemInfo(\
            \nnjdims={self.njdims}, \
            \nnb={self.nb}, \
            \nnj={self.nj}, \
            \nnl={self.nl}, \
            \nnc={self.nc}, \
            \nnbd={self.nbd}, \
            \nnjd={self.njd}, \
            \nnld={self.nld}, \
            \nncd={self.ncd}, \
            \nij={self.ij}, \
            \nil={self.il}, \
            \nic={self.ic}, \
            \nnbc={self.nbc}, \
            \nnd={self.ncts})"

    def load(self, dataset, itype=np.int32):
        self.njdims = dataset["njdims"][:].astype(itype)
        self.nb = dataset["nb"][()].astype(itype)
        self.nj = dataset["nj"][()].astype(itype)
        self.nl = dataset["nl"][()].astype(itype)
        self.nc = dataset["nc"][()].astype(itype)
        self.nbd = dataset["nbd"][()].astype(itype)
        self.njd = dataset["njd"][()].astype(itype)
        self.nld = dataset["nld"][()].astype(itype)
        self.ncd = dataset["ncd"][()].astype(itype)
        self.ij = dataset["ij"][()].astype(itype)
        self.il = dataset["il"][()].astype(itype)
        self.ic = dataset["ic"][()].astype(itype)
        self.nbc = dataset["nbc"][()].astype(itype)
        self.ncts = dataset["ncts"][()].astype(itype)

    def store(self, dataset, namespace: str = ""):
        dataset[namespace + "/njdims"] = self.njdims[:].astype(np.int32)
        dataset[namespace + "/nb"] = ctypes.c_long(self.nb)
        dataset[namespace + "/nj"] = ctypes.c_long(self.nj)
        dataset[namespace + "/nl"] = ctypes.c_long(self.nl)
        dataset[namespace + "/nc"] = ctypes.c_long(self.nc)
        dataset[namespace + "/nbd"] = ctypes.c_long(self.nbd)
        dataset[namespace + "/njd"] = ctypes.c_long(self.njd)
        dataset[namespace + "/nld"] = ctypes.c_long(self.nld)
        dataset[namespace + "/ncd"] = ctypes.c_long(self.ncd)
        dataset[namespace + "/ij"] = ctypes.c_long(self.ij)
        dataset[namespace + "/il"] = ctypes.c_long(self.il)
        dataset[namespace + "/ic"] = ctypes.c_long(self.ic)
        dataset[namespace + "/nbc"] = ctypes.c_long(self.nbc)
        dataset[namespace + "/ncts"] = ctypes.c_long(self.ncts)

    def configure(self, simulator: Simulator, wid: int = 0, dtype=np.float64, itype=np.int32):
        # Update the problem info from the simulator model
        self.njdims = np.array(simulator.model.worlds[wid].joint_cts, dtype=itype)
        self.nb = simulator.model.worlds[wid].num_bodies
        self.nj = simulator.model.worlds[wid].num_joints
        self.nl = 0
        self.nc = 0
        self.nbd = simulator.model.worlds[wid].num_body_dofs
        self.njd = simulator.model.worlds[wid].num_joint_cts
        self.nld = 0
        self.ncd = 0
        self.nbc = 0
        self.ij = 0
        self.il = self.njd
        self.ic = self.il + self.nld
        self.ncts = self.njd

    def update_from(self, simulator: Simulator, wid: int = 0, dtype=np.float64, itype=np.int32):
        # Update the problem info from the simulator model
        self.nl = simulator.model_data.info.num_limits.numpy().astype(itype)[wid]
        self.nc = simulator.model_data.info.num_contacts.numpy().astype(itype)[wid]
        self.nld = simulator.model_data.info.num_limit_cts.numpy().astype(itype)[wid]
        self.ncd = simulator.model_data.info.num_contact_cts.numpy().astype(itype)[wid]
        self.ncts = simulator.model_data.info.num_total_cts.numpy().astype(itype)[wid]
        self.il = simulator.model_data.info.limit_cts_group_offset.numpy().astype(itype)[wid]
        self.ic = simulator.model_data.info.contact_cts_group_offset.numpy().astype(itype)[wid]
        self.nbc = 0


# NumPy-based container for the DualProblem data loaded from HDF5
class DualProblemData:
    def __init__(self, dataset=None, dtype=np.float64, itype=np.int32):
        # Problem dimensions
        self.info = None
        # Problem definition
        self.D: np.ndarray | None = None
        self.v_f: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        # System quantities
        self.dt: float | None = None
        self.M: np.ndarray | None = None
        self.invM: np.ndarray | None = None
        self.J: np.ndarray | None = None
        self.h: np.ndarray | None = None
        self.u_h: np.ndarray | None = None
        self.u_minus: np.ndarray | None = None
        self.U_minus: np.ndarray | None = None
        self.T_minus: np.ndarray | None = None
        self.E_minus: np.ndarray | None = None
        self.total_mass: float | None = None
        self.total_diagonal_inertia: float | None = None
        # Residuals
        self.r_j: np.ndarray | None = None
        self.r_l: np.ndarray | None = None
        self.r_c: np.ndarray | None = None
        # Velocity biases
        self.v_i: np.ndarray | None = None
        self.v_b: np.ndarray | None = None
        # Solution
        self.lambdas: np.ndarray | None = None
        self.v_plus: np.ndarray | None = None
        # Constraint wrenches
        self.w_j: np.ndarray | None = None
        self.w_l: np.ndarray | None = None
        self.w_c: np.ndarray | None = None
        # Load data if dataset is provided
        if dataset is not None:
            self.load(dataset, dtype, itype)

    def __repr__(self):
        return f"DualProblemData(\
            \ninfo={self.info}\
            \nD({self.D.shape})=\n{self.D}\
            \nv_f({self.v_f.shape})=\n{self.v_f}\
            \nmu({self.mu.shape})=\n{self.mu}\
            \ndt={self.dt}\
            \nM({self.M.shape})=\n{self.M}\
            \ninvM({self.invM.shape})=\n{self.invM}\
            \nJ({self.J.shape})=\n{self.J}\
            \nh({self.h.shape})=\n{self.h}\
            \nu_h({self.u_h.shape})=\n{self.u_h}\
            \nu_minus({self.u_minus.shape})=\n{self.u_minus}\
            \nU_minus={self.U_minus}\
            \nT_minus={self.T_minus}\
            \nE_minus={self.E_minus}\
            \ntotal_mass={self.total_mass}\
            \ntotal_diagonal_inertia={self.total_diagonal_inertia}\
            \nr_j=\n{self.r_j}\
            \nr_l=\n{self.r_l}\
            \nr_c=\n{self.r_c}\
            \nv_i=\n{self.v_i}\
            \nv_b=\n{self.v_b}\
            \nlambdas=\n{self.lambdas}\
            \nv_plus=\n{self.v_plus}\
            \nw_j=\n{self.w_j}\
            \nw_l=\n{self.w_l}\
            \nw_c=\n{self.w_c})"

    def load(self, dataset, dtype=np.float64, itype=np.int32):
        # Load the data from the dataset
        self.info = DualProblemInfoData(dataset=dataset["info"], itype=itype)
        self.D = dataset["problem/D"][:, :].astype(dtype)
        self.v_f = dataset["problem/v_f"][:].astype(dtype)
        self.mu = dataset["problem/mu"][:].astype(dtype)
        self.dt = dataset["system/dt"][()].astype(dtype)
        self.M = dataset["system/M"][:, :].astype(dtype)
        self.invM = dataset["system/invM"][:, :].astype(dtype)
        self.J = dataset["system/J"][:, :].astype(dtype)
        self.h = dataset["system/h"][:].astype(dtype)
        self.u_h = dataset["system/u_h"][:].astype(dtype)
        self.u_minus = dataset["system/u_minus"][:].astype(dtype)
        self.U_minus = dataset["system/U_minus"][()].astype(dtype)
        self.T_minus = dataset["system/T_minus"][()].astype(dtype)
        self.E_minus = dataset["system/E_minus"][()].astype(dtype)
        self.total_mass = dataset["system/total_mass"][()].astype(dtype)
        self.total_diagonal_inertia = dataset["system/total_diagonal_inertia"][()].astype(dtype)
        self.r_j = dataset["residuals/r_j"][:].astype(dtype)
        self.r_l = dataset["residuals/r_l"][:].astype(dtype)
        self.r_c = dataset["residuals/r_c"][:].astype(dtype)
        self.v_i = dataset["vf/v_i"][:].astype(dtype)
        self.v_b = dataset["vf/v_b"][:].astype(dtype)
        self.lambdas = dataset["solution/lambdas"][:].astype(dtype)
        self.v_plus = dataset["solution/v_plus"][:].astype(dtype)
        self.w_j = dataset["wrenches/w_j"][:].astype(dtype)
        self.w_l = dataset["wrenches/w_l"][:].astype(dtype)
        self.w_c = dataset["wrenches/w_c"][:].astype(dtype)

    def store(self, dataset, namespace: str = ""):
        # Store the data into the dataset
        self.info.store(dataset, namespace=f"{namespace}/info")
        dataset[namespace + "/problem/D"] = self.D.astype(np.float64)
        dataset[namespace + "/problem/v_f"] = self.v_f.astype(np.float64)
        dataset[namespace + "/problem/mu"] = self.mu.astype(np.float64)
        dataset[namespace + "/system/dt"] = self.dt.astype(np.float64)
        dataset[namespace + "/system/M"] = self.M.astype(np.float64)
        dataset[namespace + "/system/invM"] = self.invM.astype(np.float64)
        dataset[namespace + "/system/J"] = self.J.astype(np.float64)
        dataset[namespace + "/system/h"] = self.h.astype(np.float64)
        dataset[namespace + "/system/u_h"] = self.u_h.astype(np.float64)
        dataset[namespace + "/system/u_minus"] = self.u_minus.astype(np.float64)
        dataset[namespace + "/system/U_minus"] = ctypes.c_double(self.U_minus)
        dataset[namespace + "/system/T_minus"] = ctypes.c_double(self.T_minus)
        dataset[namespace + "/system/E_minus"] = ctypes.c_double(self.E_minus)
        dataset[namespace + "/system/total_mass"] = ctypes.c_double(self.total_mass)
        dataset[namespace + "/system/total_diagonal_inertia"] = ctypes.c_double(self.total_diagonal_inertia)
        dataset[namespace + "/residuals/r_j"] = self.r_j.astype(np.float64)
        dataset[namespace + "/residuals/r_l"] = self.r_l.astype(np.float64)
        dataset[namespace + "/residuals/r_c"] = self.r_c.astype(np.float64)
        dataset[namespace + "/vf/v_i"] = self.v_i.astype(np.float64)
        dataset[namespace + "/vf/v_b"] = self.v_b.astype(np.float64)
        dataset[namespace + "/solution/lambdas"] = self.lambdas.astype(np.float64)
        dataset[namespace + "/solution/v_plus"] = self.v_plus.astype(np.float64)
        dataset[namespace + "/wrenches/w_j"] = self.w_j.astype(np.float64)
        dataset[namespace + "/wrenches/w_l"] = self.w_l.astype(np.float64)
        dataset[namespace + "/wrenches/w_c"] = self.w_c.astype(np.float64)

    def configure(self, simulator: Simulator, wid: int = 0, dtype=np.float64, itype=np.int32):
        # Configure the problem info
        self.info = DualProblemInfoData()
        self.info.configure(simulator, wid=wid, dtype=dtype, itype=itype)
        # Initialize the problem definition
        self.D = np.zeros((self.info.ncts, self.info.ncts), dtype=dtype)
        self.v_f = np.zeros((self.info.ncts,), dtype=dtype)
        self.mu = np.zeros((self.info.nc,), dtype=dtype)
        # Initialize the system quantities
        self.dt = simulator.model.time.dt.numpy().astype(dtype)[wid]
        self.J = np.zeros((self.info.njd, self.info.nbd), dtype=dtype)
        self.M = np.zeros((self.info.nbd, self.info.nbd), dtype=dtype)
        self.invM = np.zeros((self.info.nbd, self.info.nbd), dtype=dtype)
        self.h = np.zeros((self.info.nbd,), dtype=dtype)
        self.u_h = np.zeros((self.info.nbd,), dtype=dtype)
        self.u_minus = np.zeros((self.info.nbd,), dtype=dtype)
        self.U_minus = 0.0
        self.T_minus = 0.0
        self.E_minus = 0.0
        self.total_mass = 0.0
        self.total_diagonal_inertia = 0.0
        # Initialize the residuals
        self.r_j = np.zeros((self.info.njd,), dtype=dtype)
        self.r_l = np.zeros((self.info.nld,), dtype=dtype)
        self.r_c = np.zeros((self.info.ncd,), dtype=dtype)
        # Initialize the velocity biases
        self.v_i = np.zeros((self.info.ncts,), dtype=dtype)
        self.v_b = np.zeros((self.info.ncts,), dtype=dtype)
        # Initialize the solution variables
        self.lambdas = np.zeros((self.info.ncts,), dtype=dtype)
        self.v_plus = np.zeros((self.info.ncts,), dtype=dtype)
        # Initialize the constraint wrenches
        self.w_j = np.zeros((self.info.nb, 6), dtype=dtype)
        self.w_l = np.zeros((self.info.nb, 6), dtype=dtype)
        self.w_c = np.zeros((self.info.nb, 6), dtype=dtype)

    def update_from(self, simulator: Simulator, wid: int = 0, dtype=np.float64, itype=np.int32):
        # Update the problem info
        self.info.update_from(simulator, wid=wid, dtype=dtype, itype=itype)

        # Update the problem definition
        maxdim = simulator.problem.data.maxdim.numpy().astype(itype)[wid]
        dim = simulator.problem.data.dim.numpy().astype(itype)[wid]
        self.D = simulator.problem.data.D.numpy().reshape((maxdim, maxdim))[:dim, :dim].astype(dtype)
        self.v_f = simulator.problem.data.v_f.numpy()[:dim].astype(dtype)
        self.mu = simulator.problem.data.mu.numpy()[: self.info.nc].astype(dtype)
        # self.D = simulator.problem.data.D.numpy().reshape((maxdim, maxdim)).astype(dtype)
        # self.v_f = simulator.problem.data.v_f.numpy().astype(dtype)
        # self.mu = simulator.problem.data.mu.numpy().astype(dtype)

        # Construct a list of generalized inverse mass matrices of each world
        from newton._src.solvers.kamino.tests.utils.make import (  # noqa: PLC0415
            make_generalized_mass_matrices,
            make_inverse_generalized_mass_matrices,
        )

        self.M = make_generalized_mass_matrices(simulator.model, simulator.model_data)[wid].astype(dtype)
        self.invM = make_inverse_generalized_mass_matrices(simulator.model, simulator.model_data)[wid].astype(dtype)

        self.J = simulator.jacobians.data.J_cts_data.numpy().reshape((maxdim, self.info.nbd))[:dim, :].astype(dtype)
        # self.J = simulator.jacobians.data.J_cts_data.numpy().reshape((maxdim, self.info.nbd)).astype(dtype)
        # # Update the system quantities

        self.h = simulator.problem.data.h.numpy().astype(dtype).flatten()
        # self.u_h = simulator.model_data.u_h.numpy().astype(dtype)

        self.u_minus = simulator.data.s_p.u_i.numpy().astype(dtype).flatten()
        # self.U_minus = float(simulator.model_data.U_minus.numpy())
        # self.T_minus = float(simulator.model_data.T_minus.numpy())
        # self.E_minus = float(simulator.model_data.E_minus.numpy())
        # self.total_mass = float(simulator.model_data.total_mass.numpy())
        # self.total_diagonal_inertia = float(simulator.model_data.total_diagonal_inertia.numpy())

        # # Update the residuals
        # self.r_j = simulator.residuals.r_j.numpy().astype(dtype)
        # self.r_l = simulator.residuals.r_l.numpy().astype(dtype)
        # self.r_c = simulator.residuals.r_c.numpy().astype(dtype)

        # Update the velocity biases
        self.v_i = simulator.problem.data.v_i.numpy()[:dim].astype(dtype)
        self.v_b = simulator.problem.data.v_b.numpy()[:dim].astype(dtype)

        # Update the solution variables
        self.lambdas = simulator._dual_solver.data.solution.lambdas.numpy()[:dim].astype(dtype)
        self.v_plus = simulator._dual_solver.data.solution.v_plus.numpy()[:dim].astype(dtype)

        # Update the constraint wrenches
        self.w_j = simulator.model_data.bodies.w_j_i.numpy().astype(dtype)
        self.w_l = simulator.model_data.bodies.w_l_i.numpy().astype(dtype)
        self.w_c = simulator.model_data.bodies.w_c_i.numpy().astype(dtype)


# NumPy-based container for the SystemInfo data loaded from HDF5
class SystemInfoData:
    def __init__(self, dataset=None, dtype=float, itype=int):
        # Problem properties
        self.jacobian_rank: int = 0
        self.mass_ratio: float = 0.0
        self.constraint_density: float = 0.0
        # Load data if dataset is provided
        if dataset is not None:
            self.load(dataset, dtype, itype)

    def __repr__(self):
        return f"SystemInfoData(\
            \njacobian_rank={self.jacobian_rank}\
            \nmass_ratio={self.mass_ratio}\
            \nconstraint_density={self.constraint_density})"

    def load(self, dataset, dtype=float, itype=int):
        self.jacobian_rank = dataset["rankJ"][()].astype(itype)
        self.mass_ratio = dataset["mass_ratio"][()].astype(dtype)
        self.constraint_density = dataset["constraint_density"][()].astype(dtype)

    def store(self, dataset, namespace: str = ""):
        dataset[namespace + "/rankJ"] = self.jacobian_rank.astype(float)
        dataset[namespace + "/mass_ratio"] = self.mass_ratio.astype(float)
        dataset[namespace + "/constraint_density"] = self.constraint_density.astype(float)


###
# Dataset Renderer
###


class DatasetRenderer:
    def __init__(self, sysname: str, datafile: h5py.File, dt: float = 0.001):
        self.datafile = datafile
        self.system_name = sysname
        self.namescope = "Worlds/" + sysname
        self.timestep = dt
        self.framecount = 0

        # Initialize the system meta-data entry
        self.datafile[self.namescope + "/info/timestep"] = self.timestep
        self.datafile[self.namescope + "/info/framestep"] = 1

    def reset(self):
        self.framecount = 0
        self.datafile[self.namescope + "/info/steps"] = 0
        self.datafile[self.namescope + "/info/maxtime"] = 0.0

    def add_frame(
        self, system: RigidBodySystemData, contacts: ContactsData | None = None, problem: DualProblemData | None = None
    ):
        system.store(self.datafile, namespace=self.namescope + "/frames/" + str(self.framecount) + "/RigidBodySystem")
        if contacts is not None:
            contacts.store(self.datafile, namespace=self.namescope + "/frames/" + str(self.framecount) + "/Contacts")
        if problem is not None:
            problem.store(self.datafile, namespace=self.namescope + "/frames/" + str(self.framecount) + "/DualProblem")
        self.framecount += 1

    def save(self):
        # Append the final sequence info
        self.datafile[self.namescope + "/info/steps"] = self.framecount
        self.datafile[self.namescope + "/info/maxtime"] = self.timestep * self.framecount
        # Close the file
        self.datafile.flush()
        self.datafile.close()
