# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: Various utilities related to handling joints
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import warp as wp

from ......tests import get_kamino_testing_asset
from ..._src.core.joints import JointActuationType, JointDoFType
from ..._src.core.model import ModelKamino
from ..._src.utils.io.usd import USDImporter

###
# Module interface
###

__all__ = [
    "actuator_coords_from_units",
    "actuator_dofs_from_units",
    "get_actuators_q_quaternion_first_ids",
    "run_test_single_joint_examples",
]


def run_test_single_joint_examples(
    test_fun: Callable,
    test_name: str = "test",
    unary_joints: bool = True,
    binary_joints: bool = True,
    passive_joints: bool = True,
    actuators: bool = True,
    device: wp.DeviceLike = None,
):
    """
    Runs a test function over all or a subset of the single-joint examples (e.g. to check some derivatives for all joint types)

    Parameters
    ----------
    test_fun: function
        test function to run on each example, with signature kamino.core.ModelKamino -> bool, returning a success flag
    test_name: str, optional
        a name for the test to print as part of the error message upon failure (default: "test")
    unary_joints: bool, optional
        whether to include unary joint examples (NOTE: currently unsupported)
    binary_joints: bool, optional
        whether to include binary joint examples
    passive_joints: bool, optional
        whether to include passive joint examples
    actuators: bool, optional
        whether to include actuator examples
    device: DeviceLike, optional
        device on which to allocate the test models (default: None)

    Returns
    -------
    success: bool
        whether all tests succeeded
    """

    # List file paths of examples
    file_paths = []
    if unary_joints and passive_joints:
        file_paths.extend(
            [
                get_kamino_testing_asset("joints/test_joint_cartesian_passive_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_cylindrical_passive_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_fixed_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_prismatic_passive_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_revolute_passive_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_spherical_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_universal_passive_unary.usda"),
            ]
        )
    if binary_joints and passive_joints:
        file_paths.extend(
            [
                get_kamino_testing_asset("joints/test_joint_cartesian_passive.usda"),
                get_kamino_testing_asset("joints/test_joint_cylindrical_passive.usda"),
                get_kamino_testing_asset("joints/test_joint_fixed.usda"),
                get_kamino_testing_asset("joints/test_joint_prismatic_passive.usda"),
                get_kamino_testing_asset("joints/test_joint_revolute_passive.usda"),
                get_kamino_testing_asset("joints/test_joint_spherical.usda"),
                get_kamino_testing_asset("joints/test_joint_universal_passive.usda"),
            ]
        )
    if unary_joints and actuators:
        file_paths.extend(
            [
                get_kamino_testing_asset("joints/test_joint_cartesian_actuated_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_cylindrical_actuated_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_prismatic_actuated_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_revolute_actuated_unary.usda"),
                get_kamino_testing_asset("joints/test_joint_universal_actuated_unary.usda"),
                # Note: missing actuated spherical and free
            ]
        )
    if binary_joints and actuators:
        file_paths.extend(
            [
                get_kamino_testing_asset("joints/test_joint_cartesian_actuated.usda"),
                get_kamino_testing_asset("joints/test_joint_cylindrical_actuated.usda"),
                get_kamino_testing_asset("joints/test_joint_prismatic_actuated.usda"),
                get_kamino_testing_asset("joints/test_joint_revolute_actuated.usda"),
                get_kamino_testing_asset("joints/test_joint_universal_actuated.usda"),
                # Note: missing actuated spherical and free
            ]
        )

    # Load and test all examples
    success = True
    for file_path in file_paths:
        importer = USDImporter()
        builder = importer.import_from(source=file_path)
        file_stem_split = os.path.basename(file_path).split(".")[0].split("_")
        unary_binary_str = "unary" if file_stem_split[-1] == "unary" else "binary"
        passive_actuated_str = (
            "actuated" if len(file_stem_split) > 3 and file_stem_split[3] == "actuated" else "passive"
        )
        joint_type_str = file_stem_split[2]

        # Run test
        model = builder.finalize(device=device, requires_grad=False, base_auto=False)
        single_test_success = test_fun(model)
        success &= single_test_success
        if not single_test_success:
            print(f"{test_name} failed for {unary_binary_str} {passive_actuated_str} {joint_type_str} joint")
    return success


def get_actuators_q_quaternion_first_ids(model: ModelKamino):
    """Lists the first index of every unit quaternion 4-segment in the model's actuated coordinates."""
    act_types = model.joints.act_type.numpy()
    dof_types = model.joints.dof_type.numpy()
    num_coords = model.joints.num_coords.numpy()
    coord_id = 0
    quat_ids = []
    for jt_id in range(model.size.sum_of_num_joints):
        if act_types[jt_id] == JointActuationType.PASSIVE:
            continue
        if dof_types[jt_id] == JointDoFType.SPHERICAL:
            quat_ids.append(coord_id)
        elif dof_types[jt_id] == JointDoFType.FREE:
            quat_ids.append(coord_id + 3)
        coord_id += num_coords[jt_id]
    return quat_ids


def actuator_coords_from_units(
    model: ModelKamino,
    pos_val: float = 0.1,
    angle_val: float = np.radians(20.0),
    quat_val: float = 1.0,
) -> np.ndarray:
    """Helper generating actuator coords for a model, given one value to use per type of physical unit."""
    coords = []
    joint_dof_type_np = model.joints.dof_type.numpy()
    joint_act_type_np = model.joints.act_type.numpy()
    for jid in range(model.size.sum_of_num_joints):
        if joint_act_type_np[jid] == JointActuationType.PASSIVE:
            continue
        dof_type = joint_dof_type_np[jid]
        if dof_type == JointDoFType.CARTESIAN:
            coords.extend([pos_val, pos_val, pos_val])
        elif dof_type == JointDoFType.CYLINDRICAL:
            coords.extend([pos_val, angle_val])
        elif dof_type == JointDoFType.FIXED:
            pass
        elif dof_type == JointDoFType.FREE:
            coords.extend([pos_val, pos_val, pos_val, quat_val, quat_val, quat_val, quat_val])
        elif dof_type == JointDoFType.PRISMATIC:
            coords.extend([pos_val])
        elif dof_type == JointDoFType.REVOLUTE:
            coords.extend([angle_val])
        elif dof_type == JointDoFType.SPHERICAL:
            coords.extend([quat_val, quat_val, quat_val, quat_val])
        elif dof_type == JointDoFType.UNIVERSAL:
            coords.extend([angle_val, angle_val])
    return np.asarray(coords)


def actuator_dofs_from_units(
    model: ModelKamino,
    lin_vel_val: float = 0.5,
    ang_vel_val: float = np.radians(90.0),
) -> np.ndarray:
    """Helper generating actuator dofs for a model, given one value to use per type of physical unit."""
    dofs = []
    joint_dof_type_np = model.joints.dof_type.numpy()
    joint_act_type_np = model.joints.act_type.numpy()
    for jid in range(model.size.sum_of_num_joints):
        if joint_act_type_np[jid] == JointActuationType.PASSIVE:
            continue
        dof_type = joint_dof_type_np[jid]
        if dof_type == JointDoFType.CARTESIAN:
            dofs.extend([lin_vel_val, lin_vel_val, lin_vel_val])
        elif dof_type == JointDoFType.CYLINDRICAL:
            dofs.extend([lin_vel_val, ang_vel_val])
        elif dof_type == JointDoFType.FIXED:
            pass
        elif dof_type == JointDoFType.FREE:
            dofs.extend([lin_vel_val, lin_vel_val, lin_vel_val, ang_vel_val, ang_vel_val, ang_vel_val])
        elif dof_type == JointDoFType.PRISMATIC:
            dofs.extend([lin_vel_val])
        elif dof_type == JointDoFType.REVOLUTE:
            dofs.extend([ang_vel_val])
        elif dof_type == JointDoFType.SPHERICAL:
            dofs.extend([ang_vel_val, ang_vel_val, ang_vel_val])
        elif dof_type == JointDoFType.UNIVERSAL:
            dofs.extend([ang_vel_val, ang_vel_val])
    return np.asarray(dofs)
