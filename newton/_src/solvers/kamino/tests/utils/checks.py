# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: COMPARISON UTILITIES
"""

import unittest
from typing import Any

import numpy as np

from ..._src.core.bodies import RigidBodiesModel
from ..._src.core.builder import ModelBuilderKamino
from ..._src.core.control import ControlKamino
from ..._src.core.geometry import GeometriesModel
from ..._src.core.joints import JointsModel
from ..._src.core.materials import MaterialPairsModel, MaterialsModel
from ..._src.core.model import ModelKamino, ModelKaminoInfo
from ..._src.core.size import SizeKamino
from ..._src.core.state import StateKamino
from ..._src.utils import logger as msg

###
# Module interface
###

__all__ = [
    "arrays_equal",
    "assert_builders_equal",
    "assert_control_equal",
    "assert_model_bodies_equal",
    "assert_model_equal",
    "assert_model_geoms_equal",
    "assert_model_info_equal",
    "assert_model_joints_equal",
    "assert_model_material_pairs_equal",
    "assert_model_materials_equal",
    "assert_model_size_equal",
    "assert_state_equal",
    "lists_equal",
    "matrices_equal",
    "vectors_equal",
]


###
# Array-like comparisons
###


def lists_equal(list1, list2) -> bool:
    return np.array_equal(list1, list2)


def arrays_equal(arr1, arr2, tolerance=1e-6) -> bool:
    return np.allclose(arr1, arr2, atol=tolerance)


def matrices_equal(m1, m2, tolerance=1e-6) -> bool:
    return np.allclose(m1, m2, atol=tolerance)


def vectors_equal(v1, v2, tolerance=1e-6) -> bool:
    return np.allclose(v1, v2, atol=tolerance)


###
# Utilities
###


def assert_scalar_attributes_equal(test: unittest.TestCase, obj0: Any, obj1: Any, attributes: list[str]) -> None:
    for attr in attributes:
        # Check if attribute exists in both objects
        obj_name = obj0.__class__.__name__
        has_attr0 = hasattr(obj0, attr)
        has_attr1 = hasattr(obj1, attr)
        if not has_attr0 and not has_attr1:
            msg.debug(f"Skipping attribute '{attr}' comparison for {obj_name} because it is missing in both objects.")
            continue
        elif not has_attr0 or not has_attr1:
            test.fail(
                f"Attribute '{attr}' is missing in one of the objects: "
                f" {obj_name} has_attr0={has_attr0}, has_attr1={has_attr1}"
            )
        # Retrieve attributes for logging
        attr0 = getattr(obj0, attr)
        attr1 = getattr(obj1, attr)
        # Test scalar attribute values
        msg.debug("Comparing %s.%s: actual=%s, desired=%s", obj_name, attr, attr0, attr1)
        test.assertEqual(
            first=attr0,
            second=attr1,
            msg=f"{obj0.__class__.__name__}.{attr} are not equal.",
        )


def assert_array_attributes_equal(
    test: unittest.TestCase,
    obj0: Any,
    obj1: Any,
    attributes: list[str],
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    for attr in attributes:
        # Check if attribute exists in both objects
        obj_name = obj0.__class__.__name__
        has_attr0 = hasattr(obj0, attr)
        has_attr1 = hasattr(obj1, attr)
        if not has_attr0 and not has_attr1:
            msg.debug(f"Skipping attribute '{attr}' comparison for {obj_name} because it is missing in both objects.")
            continue
        elif not has_attr0 or not has_attr1:
            test.fail(
                f"Attribute '{attr}' is missing in one of the objects: "
                f" {obj_name} has_attr0={has_attr0}, has_attr1={has_attr1}"
            )
        # Retrieve attributes for logging
        attr0 = getattr(obj0, attr)
        attr1 = getattr(obj1, attr)
        # Check if attributes are array-like
        attr0_is_array = hasattr(attr0, "shape")
        attr1_is_array = hasattr(attr1, "shape")
        if not attr0_is_array and not attr1_is_array:
            msg.debug(
                f"\nSkipping attribute '{obj_name}.{attr}' comparison: both of the objects are not array-like: "
                f"\n0: {obj_name}.{attr}: {type(attr0)}\n1: {obj_name}.{attr}: {type(attr1)}"
            )
            continue
        elif not attr0_is_array or not attr1_is_array:
            test.fail(
                f"Attribute '{attr}' is not array-like in one of the objects: "
                f" {obj_name}.{attr} has_attr0_shape={getattr(attr0, 'shape', None)}, "
                f"has_attr1_shape={getattr(attr1, 'shape', None)}"
            )
        # Test array attribute shapes
        shape0 = attr0.shape
        shape1 = attr1.shape
        test.assertEqual(shape0, shape1, f"{obj_name}.{attr} shapes are not equal.")
        # Test array attribute values
        diff = attr0 - attr1
        msg.debug("Comparing %s:\nactual:\n%s\ndesired:\n%s\ndiff:\n%s", f"{obj_name}.{attr}", attr0, attr1, diff)
        np.testing.assert_allclose(
            actual=attr0.numpy(),
            desired=attr1.numpy(),
            err_msg=f"{obj_name}.{attr} are not equal.",
            rtol=rtol.get(attr, 1e-6) if rtol else 1e-6,
            atol=atol.get(attr, 1e-6) if atol else 1e-6,
        )


###
# Container comparisons
###


def assert_builders_equal(
    test: unittest.TestCase,
    builder1: ModelBuilderKamino,
    builder2: ModelBuilderKamino,
    skip_colliders: bool = False,
    skip_materials: bool = False,
):
    """
    Compares two ModelBuilderKamino instances for equality.
    """
    test.assertEqual(builder1.num_bodies, builder2.num_bodies)
    test.assertEqual(builder1.num_joints, builder2.num_joints)
    test.assertEqual(builder1.num_geoms, builder2.num_geoms)
    test.assertEqual(builder1.num_materials, builder2.num_materials)

    for i in range(builder1.num_bodies):
        test.assertEqual(builder1.bodies[i].wid, builder2.bodies[i].wid)
        test.assertEqual(builder1.bodies[i].bid, builder2.bodies[i].bid)
        test.assertAlmostEqual(builder1.bodies[i].m_i, builder2.bodies[i].m_i)
        test.assertTrue(matrices_equal(builder1.bodies[i].i_I_i, builder2.bodies[i].i_I_i))
        test.assertTrue(vectors_equal(builder1.bodies[i].q_i_0, builder2.bodies[i].q_i_0))
        test.assertTrue(vectors_equal(builder1.bodies[i].u_i_0, builder2.bodies[i].u_i_0))

    for j in range(builder1.num_joints):
        test.assertEqual(builder1.joints[j].wid, builder2.joints[j].wid)
        test.assertEqual(builder1.joints[j].jid, builder2.joints[j].jid)
        test.assertEqual(builder1.joints[j].act_type, builder2.joints[j].act_type)
        test.assertEqual(builder1.joints[j].dof_type, builder2.joints[j].dof_type)
        test.assertEqual(builder1.joints[j].bid_B, builder2.joints[j].bid_B)
        test.assertEqual(builder1.joints[j].bid_F, builder2.joints[j].bid_F)
        test.assertTrue(
            vectors_equal(builder1.joints[j].B_r_Bj, builder2.joints[j].B_r_Bj),
            f"Joint {j} B_r_Bj:\nleft:\n{builder1.joints[j].B_r_Bj}\nright:\n{builder2.joints[j].B_r_Bj}",
        )
        test.assertTrue(
            vectors_equal(builder1.joints[j].F_r_Fj, builder2.joints[j].F_r_Fj),
            f"Joint {j} F_r_Fj:\nleft:\n{builder1.joints[j].F_r_Fj}\nright:\n{builder2.joints[j].F_r_Fj}",
        )
        test.assertTrue(
            matrices_equal(builder1.joints[j].X_j, builder2.joints[j].X_j),
            f"Joint {j} X_j:\nleft:\n{builder1.joints[j].X_j}\nright:\n{builder2.joints[j].X_j}",
        )
        test.assertTrue(
            arrays_equal(builder1.joints[j].q_j_min, builder2.joints[j].q_j_min),
            f"Joint {j} q_j_min:\nleft:\n{builder1.joints[j].q_j_min}\nright:\n{builder2.joints[j].q_j_min}",
        )
        test.assertTrue(
            arrays_equal(builder1.joints[j].q_j_max, builder2.joints[j].q_j_max),
            f"Joint {j} q_j_max:\nleft:\n{builder1.joints[j].q_j_max}\nright:\n{builder2.joints[j].q_j_max}",
        )
        test.assertTrue(
            arrays_equal(builder1.joints[j].dq_j_max, builder2.joints[j].dq_j_max),
            f"Joint {j} dq_j_max:\nleft:\n{builder1.joints[j].dq_j_max}\nright:\n{builder2.joints[j].dq_j_max}",
        )
        test.assertTrue(
            arrays_equal(builder1.joints[j].tau_j_max, builder2.joints[j].tau_j_max),
            f"Joint {j} tau_j_max:\nleft:\n{builder1.joints[j].tau_j_max}\nright:\n{builder2.joints[j].tau_j_max}",
        )
        test.assertTrue(
            arrays_equal(builder1.joints[j].a_j, builder2.joints[j].a_j),
            f"Joint {j} a_j:\nleft:\n{builder1.joints[j].a_j}\nright:\n{builder2.joints[j].a_j}",
        )
        test.assertTrue(
            arrays_equal(builder1.joints[j].b_j, builder2.joints[j].b_j),
            f"Joint {j} b_j:\nleft:\n{builder1.joints[j].b_j}\nright:\n{builder2.joints[j].b_j}",
        )
        test.assertTrue(
            arrays_equal(builder1.joints[j].k_p_j, builder2.joints[j].k_p_j),
            f"Joint {j} k_p_j:\nleft:\n{builder1.joints[j].k_p_j}\nright:\n{builder2.joints[j].k_p_j}",
        )
        test.assertTrue(
            arrays_equal(builder1.joints[j].k_d_j, builder2.joints[j].k_d_j),
            f"Joint {j} k_d_j:\nleft:\n{builder1.joints[j].k_d_j}\nright:\n{builder2.joints[j].k_d_j}",
        )
        test.assertEqual(builder1.joints[j].num_coords, builder2.joints[j].num_coords)
        test.assertEqual(builder1.joints[j].num_dofs, builder2.joints[j].num_dofs)
        test.assertEqual(builder1.joints[j].num_passive_coords, builder2.joints[j].num_passive_coords)
        test.assertEqual(builder1.joints[j].num_passive_dofs, builder2.joints[j].num_passive_dofs)
        test.assertEqual(builder1.joints[j].num_actuated_coords, builder2.joints[j].num_actuated_coords)
        test.assertEqual(builder1.joints[j].num_actuated_dofs, builder2.joints[j].num_actuated_dofs)
        test.assertEqual(builder1.joints[j].num_actuated_dofs, builder2.joints[j].num_actuated_dofs)
        test.assertEqual(builder1.joints[j].num_cts, builder2.joints[j].num_cts)
        test.assertEqual(builder1.joints[j].num_dynamic_cts, builder2.joints[j].num_dynamic_cts)
        test.assertEqual(builder1.joints[j].num_kinematic_cts, builder2.joints[j].num_kinematic_cts)
        test.assertEqual(builder1.joints[j].coords_offset, builder2.joints[j].coords_offset)
        test.assertEqual(builder1.joints[j].dofs_offset, builder2.joints[j].dofs_offset)
        test.assertEqual(builder1.joints[j].passive_coords_offset, builder2.joints[j].passive_coords_offset)
        test.assertEqual(builder1.joints[j].passive_dofs_offset, builder2.joints[j].passive_dofs_offset)
        test.assertEqual(builder1.joints[j].actuated_coords_offset, builder2.joints[j].actuated_coords_offset)
        test.assertEqual(builder1.joints[j].actuated_dofs_offset, builder2.joints[j].actuated_dofs_offset)
        test.assertEqual(builder1.joints[j].cts_offset, builder2.joints[j].cts_offset)
        test.assertEqual(builder1.joints[j].dynamic_cts_offset, builder2.joints[j].dynamic_cts_offset)
        test.assertEqual(builder1.joints[j].kinematic_cts_offset, builder2.joints[j].kinematic_cts_offset)
        test.assertEqual(builder1.joints[j].is_binary, builder2.joints[j].is_binary)
        test.assertEqual(builder1.joints[j].is_passive, builder2.joints[j].is_passive)
        test.assertEqual(builder1.joints[j].is_actuated, builder2.joints[j].is_actuated)
        test.assertEqual(builder1.joints[j].is_dynamic, builder2.joints[j].is_dynamic)
        test.assertEqual(builder1.joints[j].is_implicit_pd, builder2.joints[j].is_implicit_pd)

    for k in range(builder1.num_geoms):
        test.assertEqual(builder1.geoms[k].wid, builder2.geoms[k].wid)
        test.assertEqual(builder1.geoms[k].gid, builder2.geoms[k].gid)
        test.assertEqual(builder1.geoms[k].mid, builder2.geoms[k].mid)
        test.assertEqual(builder1.geoms[k].body, builder2.geoms[k].body)
        test.assertEqual(builder1.geoms[k].shape.type, builder2.geoms[k].shape.type)
        test.assertEqual(builder1.geoms[k].shape.num_params, builder2.geoms[k].shape.num_params)
        test.assertTrue(lists_equal(builder1.geoms[k].shape.paramsvec, builder2.geoms[k].shape.paramsvec))
        if not skip_materials:
            test.assertEqual(builder1.geoms[k].material, builder2.geoms[k].material)
        if not skip_colliders:
            test.assertEqual(builder1.geoms[k].group, builder2.geoms[k].group)
            test.assertEqual(builder1.geoms[k].collides, builder2.geoms[k].collides)
            test.assertEqual(builder1.geoms[k].max_contacts, builder2.geoms[k].max_contacts)
            test.assertEqual(builder1.geoms[k].gap, builder2.geoms[k].gap)
            test.assertEqual(builder1.geoms[k].margin, builder2.geoms[k].margin)

    if not skip_materials:
        for m in range(builder1.num_materials):
            test.assertEqual(builder1.materials[m].wid, builder2.materials[m].wid)
            test.assertEqual(builder1.materials[m].mid, builder2.materials[m].mid)
            test.assertEqual(builder1.materials[m].density, builder2.materials[m].density)
            test.assertEqual(builder1.materials[m].restitution, builder2.materials[m].restitution)
            test.assertEqual(builder1.materials[m].static_friction, builder2.materials[m].static_friction)
            test.assertEqual(builder1.materials[m].dynamic_friction, builder2.materials[m].dynamic_friction)


###
# Container comparisons
###


def assert_state_equal(
    test: unittest.TestCase, state0: StateKamino, state1: StateKamino, excluded: list[str] | None = None
) -> None:
    attributes = ["q_i", "u_i", "w_i", "q_j", "q_j_p", "dq_j", "lambda_j"]
    if excluded:
        attributes = [attr for attr in attributes if attr not in excluded]
    assert_array_attributes_equal(test, state0, state1, attributes)


def assert_control_equal(
    test: unittest.TestCase, control0: ControlKamino, control1: ControlKamino, excluded: list[str] | None = None
) -> None:
    attributes = ["tau_j", "q_j_ref", "dq_j_ref", "tau_j_ref"]
    if excluded:
        attributes = [attr for attr in attributes if attr not in excluded]
    assert_array_attributes_equal(test, control0, control1, attributes)


def assert_model_size_equal(
    test: unittest.TestCase, size0: SizeKamino, size1: SizeKamino, excluded: list[str] | None = None
) -> None:
    attributes = [
        "num_worlds",
        "sum_of_num_bodies",
        "max_of_num_bodies",
        "sum_of_num_joints",
        "max_of_num_joints",
        "sum_of_num_passive_joints",
        "max_of_num_passive_joints",
        "sum_of_num_actuated_joints",
        "max_of_num_actuated_joints",
        "sum_of_num_dynamic_joints",
        "max_of_num_dynamic_joints",
        "sum_of_num_geoms",
        "max_of_num_geoms",
        "sum_of_num_material_pairs",
        "max_of_num_material_pairs",
        "sum_of_num_body_dofs",
        "max_of_num_body_dofs",
        "sum_of_num_joint_coords",
        "max_of_num_joint_coords",
        "sum_of_num_joint_dofs",
        "max_of_num_joint_dofs",
        "sum_of_num_passive_joint_coords",
        "max_of_num_passive_joint_coords",
        "sum_of_num_passive_joint_dofs",
        "max_of_num_passive_joint_dofs",
        "sum_of_num_actuated_joint_coords",
        "max_of_num_actuated_joint_coords",
        "sum_of_num_actuated_joint_dofs",
        "max_of_num_actuated_joint_dofs",
        "sum_of_num_joint_cts",
        "max_of_num_joint_cts",
        "sum_of_num_dynamic_joint_cts",
        "max_of_num_dynamic_joint_cts",
        "sum_of_num_kinematic_joint_cts",
        "max_of_num_kinematic_joint_cts",
        "sum_of_max_limits",
        "max_of_max_limits",
        "sum_of_max_contacts",
        "max_of_max_contacts",
        "sum_of_max_unilaterals",
        "max_of_max_unilaterals",
        "sum_of_max_total_cts",
        "max_of_max_total_cts",
    ]
    if excluded:
        attributes = [attr for attr in attributes if attr not in excluded]
    assert_scalar_attributes_equal(test, size0, size1, attributes)


def assert_model_info_equal(
    test: unittest.TestCase, info0: ModelKaminoInfo, info1: ModelKaminoInfo, excluded: list[str] | None = None
) -> None:
    assert_scalar_attributes_equal(test, info0, info1, ["num_worlds"])
    array_attributes = [
        "num_bodies",
        "num_joints",
        "num_passive_joints",
        "num_actuated_joints",
        "num_dynamic_joints",
        "num_geoms",
        "num_body_dofs",
        "num_joint_coords",
        "num_joint_dofs",
        "num_passive_joint_coords",
        "num_passive_joint_dofs",
        "num_actuated_joint_coords",
        "num_actuated_joint_dofs",
        "num_joint_cts",
        "num_joint_dynamic_cts",
        "num_joint_kinematic_cts",
        "max_limit_cts",
        "max_contact_cts",
        "max_total_cts",
        "bodies_offset",
        "joints_offset",
        "geoms_offset",
        "body_dofs_offset",
        "joint_coords_offset",
        "joint_dofs_offset",
        "joint_passive_coords_offset",
        "joint_passive_dofs_offset",
        "joint_actuated_coords_offset",
        "joint_actuated_dofs_offset",
        "joint_cts_offset",
        "joint_dynamic_cts_offset",
        "joint_kinematic_cts_offset",
        "total_cts_offset",
        "joint_dynamic_cts_group_offset",
        "joint_kinematic_cts_group_offset",
        "base_body_index",
        "base_joint_index",
        "mass_min",
        "mass_max",
        "mass_total",
        "inertia_total",
    ]
    if excluded:
        array_attributes = [attr for attr in array_attributes if attr not in excluded]
    assert_array_attributes_equal(test, info0, info1, array_attributes)


def assert_model_bodies_equal(
    test: unittest.TestCase,
    bodies0: RigidBodiesModel,
    bodies1: RigidBodiesModel,
    excluded: list[str] | None = None,
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    assert_scalar_attributes_equal(test, bodies0, bodies1, ["num_bodies", "label"])
    array_attributes = [
        "wid",
        "bid",
        "i_r_com_i",
        "m_i",
        "inv_m_i",
        "i_I_i",
        "inv_i_I_i",
        "q_i_0",
        "u_i_0",
    ]
    if excluded:
        array_attributes = [attr for attr in array_attributes if attr not in excluded]
    assert_array_attributes_equal(test, bodies0, bodies1, array_attributes, rtol=rtol, atol=atol)


def assert_model_joints_equal(
    test: unittest.TestCase, joints0: JointsModel, joints1: JointsModel, excluded: list[str] | None = None
) -> None:
    assert_scalar_attributes_equal(test, joints0, joints1, ["num_joints", "label"])
    array_attributes = [
        "wid",
        "jid",
        "dof_type",
        "act_type",
        "bid_B",
        "bid_F",
        "B_r_Bj",
        "F_r_Fj",
        "X_j",
        "q_j_min",
        "q_j_max",
        "dq_j_max",
        "tau_j_max",
        "a_j",
        "b_j",
        "k_p_j",
        "k_d_j",
        "q_j_0",
        "dq_j_0",
        "num_coords",
        "num_dofs",
        "num_cts",
        "num_dynamic_cts",
        "num_kinematic_cts",
        "coords_offset",
        "dofs_offset",
        "passive_coords_offset",
        "passive_dofs_offset",
        "actuated_coords_offset",
        "actuated_dofs_offset",
        "cts_offset",
        "dynamic_cts_offset",
        "kinematic_cts_offset",
    ]
    if excluded:
        array_attributes = [attr for attr in array_attributes if attr not in excluded]
    assert_array_attributes_equal(test, joints0, joints1, array_attributes)


def assert_model_geoms_equal(
    test: unittest.TestCase,
    geoms0: GeometriesModel,
    geoms1: GeometriesModel,
    excluded: list[str] | None = None,
) -> None:
    scalar_attributes = [
        "num_geoms",
        "num_collidable",
        "num_collidable_pairs",
        "num_excluded_pairs",
        "model_minimum_contacts",
        "world_minimum_contacts",
        "label",
    ]
    array_attributes = [
        "wid",
        "gid",
        "bid",
        "type",
        "flags",
        "ptr",
        "params",
        "offset",
        "material",
        "group",
        "gap",
        "margin",
        "collidable_pairs",
        "excluded_pairs",
    ]
    if excluded:
        scalar_attributes = [attr for attr in scalar_attributes if attr not in excluded]
        array_attributes = [attr for attr in array_attributes if attr not in excluded]
    assert_scalar_attributes_equal(test, geoms0, geoms1, scalar_attributes)
    assert_array_attributes_equal(test, geoms0, geoms1, array_attributes)


def assert_model_materials_equal(
    test: unittest.TestCase, materials0: MaterialsModel, materials1: MaterialsModel, excluded: list[str] | None = None
) -> None:
    assert_scalar_attributes_equal(test, materials0, materials1, ["num_materials"])
    array_attributes = [
        # "density",
        "restitution",
        "static_friction",
        "dynamic_friction",
    ]
    if excluded:
        array_attributes = [attr for attr in array_attributes if attr not in excluded]
    assert_array_attributes_equal(test, materials0, materials1, array_attributes)


def assert_model_material_pairs_equal(
    test: unittest.TestCase,
    matpairs0: MaterialPairsModel,
    matpairs1: MaterialPairsModel,
    excluded: list[str] | None = None,
) -> None:
    assert_scalar_attributes_equal(test, matpairs0, matpairs1, ["num_material_pairs"])
    array_attributes = [
        "restitution",
        "static_friction",
        "dynamic_friction",
    ]
    if excluded:
        array_attributes = [attr for attr in array_attributes if attr not in excluded]
    assert_array_attributes_equal(test, matpairs0, matpairs1, array_attributes)


def assert_model_equal(
    test: unittest.TestCase,
    model0: ModelKamino,
    model1: ModelKamino,
    skip_geom_source_ptr: bool = False,
    skip_geom_group_and_collides: bool = False,
    skip_geom_margin_and_gap: bool = False,
    excluded: list[str] | None = None,
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    assert_model_size_equal(test, model0.size, model1.size, excluded)
    assert_model_info_equal(test, model0.info, model1.info, excluded)
    assert_model_bodies_equal(test, model0.bodies, model1.bodies, excluded, rtol=rtol, atol=atol)
    assert_model_joints_equal(test, model0.joints, model1.joints, excluded)
    geom_excluded = excluded
    if skip_geom_source_ptr or skip_geom_group_and_collides or skip_geom_margin_and_gap:
        geom_excluded = [] if excluded is None else list(excluded)
        if skip_geom_source_ptr:
            geom_excluded.append("ptr")
        if skip_geom_group_and_collides:
            geom_excluded.extend(["group", "collides"])
        if skip_geom_margin_and_gap:
            geom_excluded.extend(["margin", "gap"])
    assert_model_geoms_equal(
        test,
        model0.geoms,
        model1.geoms,
        excluded=geom_excluded,
    )
    assert_model_materials_equal(test, model0.materials, model1.materials, excluded)
    assert_model_material_pairs_equal(test, model0.material_pairs, model1.material_pairs, excluded)
