# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: COMPARISON UTILITIES
"""

import unittest
from typing import Any

import numpy as np

from newton._src.solvers.kamino._src.core.bodies import RigidBodiesModel
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.control import ControlKamino
from newton._src.solvers.kamino._src.core.geometry import GeometriesModel
from newton._src.solvers.kamino._src.core.joints import JointsModel
from newton._src.solvers.kamino._src.core.materials import MaterialPairsModel, MaterialsModel
from newton._src.solvers.kamino._src.core.model import ModelKamino, ModelKaminoInfo
from newton._src.solvers.kamino._src.core.size import SizeKamino
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.utils import logger as msg

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


def assert_scalar_attributes_equal(
    test: unittest.TestCase,
    obj0: Any,
    obj1: Any,
    attributes: list[str],
    mapping: list[int] | None = None,
) -> None:
    """Compare scalar attributes on two objects, optionally permuting list-valued fields through `mapping`."""
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
        if mapping is not None:
            test.assertIsInstance(attr0, list, f"{obj_name}.{attr} must be list-valued to use mapping.")
            test.assertIsInstance(attr1, list, f"{obj_name}.{attr} must be list-valued to use mapping.")
            test.assertEqual(len(attr0), len(attr1), f"{obj_name}.{attr} lengths are not equal.")
            for i, other_idx in enumerate(mapping):
                msg.debug(
                    "Comparing %s.%s[%d]: actual=%s, desired=%s",
                    obj_name,
                    attr,
                    i,
                    attr0[i],
                    attr1[other_idx],
                )
                test.assertEqual(
                    first=attr0[i],
                    second=attr1[other_idx],
                    msg=f"{obj_name}.{attr}[{i}] are not equal.",
                )
        else:
            # Test scalar attribute values
            msg.debug("Comparing %s.%s: actual=%s, desired=%s", obj_name, attr, attr0, attr1)
            test.assertEqual(
                first=attr0,
                second=attr1,
                msg=f"{obj_name}.{attr} are not equal.",
            )


def assert_array_attributes_equal(
    test: unittest.TestCase,
    obj0: Any,
    obj1: Any,
    attributes: list[str],
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
    mapping: list[int] | None = None,
    index_remaps: dict[str, list[int]] | None = None,
) -> None:
    """Compare array attributes, permuting rows and remapping referenced indices when requested.

    `mapping` permutes rows of `obj1` to align with `obj0` (e.g. after reordering entities by
    label). `index_remaps` separately translates *values* held by an attribute that reference
    another entity's row in `obj1`'s index space (e.g. a body index) into `obj0`'s index space,
    and applies independently of whether row permutation is active.
    """
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
        actual = attr0.numpy()
        desired = attr1.numpy()
        if mapping is not None and len(mapping) == desired.shape[0]:
            desired = desired[mapping]
        if index_remaps is not None and attr in index_remaps:
            desired = np.asarray(desired).copy()
            remap = index_remaps[attr]
            for i, value in enumerate(desired):
                if value >= 0:
                    desired[i] = remap[value]
        # Unbounded limits are stored as inf (e.g. JointsModel.tau_j_max), so this purely
        # informational diff hits inf - inf. Left unguarded it raises a RuntimeWarning, which
        # CI turns into a test error via --strict-warnings.
        with np.errstate(invalid="ignore"):
            diff = actual - desired
        msg.debug("Comparing %s:\nactual:\n%s\ndesired:\n%s\ndiff:\n%s", f"{obj_name}.{attr}", actual, desired, diff)
        np.testing.assert_allclose(
            actual=actual,
            desired=desired,
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

    for body1, body2 in zip(builder1.all_bodies, builder2.all_bodies, strict=True):
        test.assertEqual(body1.wid, body2.wid)
        test.assertEqual(body1.bid, body2.bid)
        test.assertAlmostEqual(body1.m_i, body2.m_i)
        test.assertTrue(matrices_equal(body1.i_I_i, body2.i_I_i))
        test.assertTrue(vectors_equal(body1.q_i_0, body2.q_i_0))
        test.assertTrue(vectors_equal(body1.u_i_0, body2.u_i_0))

    for j, (joint1, joint2) in enumerate(zip(builder1.all_joints, builder2.all_joints, strict=True)):
        test.assertEqual(joint1.wid, joint2.wid)
        test.assertEqual(joint1.jid, joint2.jid)
        test.assertEqual(joint1.act_type, joint2.act_type)
        test.assertEqual(joint1.dof_type, joint2.dof_type)
        test.assertEqual(joint1.bid_B, joint2.bid_B)
        test.assertEqual(joint1.bid_F, joint2.bid_F)
        test.assertTrue(
            vectors_equal(joint1.B_r_Bj, joint2.B_r_Bj),
            f"Joint {j} B_r_Bj:\nleft:\n{joint1.B_r_Bj}\nright:\n{joint2.B_r_Bj}",
        )
        test.assertTrue(
            vectors_equal(joint1.F_r_Fj, joint2.F_r_Fj),
            f"Joint {j} F_r_Fj:\nleft:\n{joint1.F_r_Fj}\nright:\n{joint2.F_r_Fj}",
        )
        test.assertTrue(
            matrices_equal(joint1.X_Bj, joint2.X_Bj),
            f"Joint {j} X_Bj:\nleft:\n{joint1.X_Bj}\nright:\n{joint2.X_Bj}",
        )
        test.assertTrue(
            matrices_equal(joint1.X_Fj, joint2.X_Fj),
            f"Joint {j} X_Fj:\nleft:\n{joint1.X_Fj}\nright:\n{joint2.X_Fj}",
        )
        test.assertTrue(
            arrays_equal(joint1.q_j_min, joint2.q_j_min),
            f"Joint {j} q_j_min:\nleft:\n{joint1.q_j_min}\nright:\n{joint2.q_j_min}",
        )
        test.assertTrue(
            arrays_equal(joint1.q_j_max, joint2.q_j_max),
            f"Joint {j} q_j_max:\nleft:\n{joint1.q_j_max}\nright:\n{joint2.q_j_max}",
        )
        test.assertTrue(
            arrays_equal(joint1.dq_j_max, joint2.dq_j_max),
            f"Joint {j} dq_j_max:\nleft:\n{joint1.dq_j_max}\nright:\n{joint2.dq_j_max}",
        )
        test.assertTrue(
            arrays_equal(joint1.tau_j_max, joint2.tau_j_max),
            f"Joint {j} tau_j_max:\nleft:\n{joint1.tau_j_max}\nright:\n{joint2.tau_j_max}",
        )
        test.assertTrue(
            arrays_equal(joint1.a_j, joint2.a_j),
            f"Joint {j} a_j:\nleft:\n{joint1.a_j}\nright:\n{joint2.a_j}",
        )
        test.assertTrue(
            arrays_equal(joint1.b_j, joint2.b_j),
            f"Joint {j} b_j:\nleft:\n{joint1.b_j}\nright:\n{joint2.b_j}",
        )
        test.assertTrue(
            arrays_equal(joint1.k_p_j, joint2.k_p_j),
            f"Joint {j} k_p_j:\nleft:\n{joint1.k_p_j}\nright:\n{joint2.k_p_j}",
        )
        test.assertTrue(
            arrays_equal(joint1.k_d_j, joint2.k_d_j),
            f"Joint {j} k_d_j:\nleft:\n{joint1.k_d_j}\nright:\n{joint2.k_d_j}",
        )
        test.assertEqual(joint1.num_coords, joint2.num_coords)
        test.assertEqual(joint1.num_dofs, joint2.num_dofs)
        test.assertEqual(joint1.num_passive_coords, joint2.num_passive_coords)
        test.assertEqual(joint1.num_passive_dofs, joint2.num_passive_dofs)
        test.assertEqual(joint1.num_actuated_coords, joint2.num_actuated_coords)
        test.assertEqual(joint1.num_actuated_dofs, joint2.num_actuated_dofs)
        test.assertEqual(joint1.num_actuated_dofs, joint2.num_actuated_dofs)
        test.assertEqual(joint1.num_cts, joint2.num_cts)
        test.assertEqual(joint1.num_dynamic_cts, joint2.num_dynamic_cts)
        test.assertEqual(joint1.num_kinematic_cts, joint2.num_kinematic_cts)
        test.assertEqual(joint1.coords_offset, joint2.coords_offset)
        test.assertEqual(joint1.dofs_offset, joint2.dofs_offset)
        test.assertEqual(joint1.passive_coords_offset, joint2.passive_coords_offset)
        test.assertEqual(joint1.passive_dofs_offset, joint2.passive_dofs_offset)
        test.assertEqual(joint1.actuated_coords_offset, joint2.actuated_coords_offset)
        test.assertEqual(joint1.actuated_dofs_offset, joint2.actuated_dofs_offset)
        test.assertEqual(joint1.cts_offset, joint2.cts_offset)
        test.assertEqual(joint1.dynamic_cts_offset, joint2.dynamic_cts_offset)
        test.assertEqual(joint1.kinematic_cts_offset, joint2.kinematic_cts_offset)
        test.assertEqual(joint1.is_binary, joint2.is_binary)
        test.assertEqual(joint1.is_passive, joint2.is_passive)
        test.assertEqual(joint1.is_actuated, joint2.is_actuated)
        test.assertEqual(joint1.is_dynamic, joint2.is_dynamic)
        test.assertEqual(joint1.is_implicit_pd, joint2.is_implicit_pd)

    for geom1, geom2 in zip(builder1.all_geoms, builder2.all_geoms, strict=True):
        test.assertEqual(geom1.wid, geom2.wid)
        test.assertEqual(geom1.gid, geom2.gid)
        test.assertEqual(geom1.mid, geom2.mid)
        test.assertEqual(geom1.body, geom2.body)
        shape1 = builder1.shapes[geom1.uid]
        shape2 = builder2.shapes[geom2.uid]
        test.assertEqual(shape1.type, shape2.type)
        test.assertTrue(lists_equal(shape1.paramsvec, shape2.paramsvec))
        test.assertTrue(vectors_equal(shape1.params, shape2.params))
        if not skip_materials:
            test.assertEqual(geom1.material, geom2.material)
        if not skip_colliders:
            test.assertEqual(geom1.group, geom2.group)
            test.assertEqual(geom1.collides, geom2.collides)
            test.assertEqual(geom1.max_contacts, geom2.max_contacts)
            test.assertEqual(geom1.gap, geom2.gap)
            test.assertEqual(geom1.margin, geom2.margin)

    if not skip_materials:
        for m in range(builder1.num_materials):
            test.assertEqual(builder1.materials[m].wid, builder2.materials[m].wid)
            test.assertEqual(builder1.materials[m].mid, builder2.materials[m].mid)
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
    test: unittest.TestCase,
    info0: ModelKaminoInfo,
    info1: ModelKaminoInfo,
    excluded: list[str] | None = None,
    body_index_remap: list[int] | None = None,
    joint_index_remap: list[int] | None = None,
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
    ]
    if excluded:
        array_attributes = [attr for attr in array_attributes if attr not in excluded]
    index_remaps = {}
    if body_index_remap is not None and "base_body_index" in array_attributes:
        index_remaps["base_body_index"] = body_index_remap
    if joint_index_remap is not None and "base_joint_index" in array_attributes:
        index_remaps["base_joint_index"] = joint_index_remap
    assert_array_attributes_equal(test, info0, info1, array_attributes, index_remaps=index_remaps or None)


def _assert_world_local_ids(test: unittest.TestCase, obj: Any, id_attr: str, wid_attr: str = "wid") -> None:
    """Assert a world-local id array (e.g. ``bid``/``jid``/``gid``) is a valid sequential position
    within each world, i.e. ``ids[row] == row - world_offset``.

    Used in place of a direct cross-model comparison of these ids when rows have been permuted:
    once reordered, there is no meaningful pairwise comparison left to make between the two
    models' own world-local ids, only the per-model invariant.
    """
    ids = getattr(obj, id_attr).numpy()
    if ids.size == 0:
        return
    wids = getattr(obj, wid_attr).numpy()
    unique_wids, first_index = np.unique(wids, return_index=True)
    offsets = first_index[np.searchsorted(unique_wids, wids)]
    expected = np.arange(ids.size) - offsets
    np.testing.assert_array_equal(
        ids,
        expected,
        err_msg=f"{obj.__class__.__name__}.{id_attr} is not a valid sequential world-local id.",
    )


def _assert_geom_pairs_equal(
    test: unittest.TestCase,
    geoms0: GeometriesModel,
    geoms1: GeometriesModel,
    attr: str,
    geom_index_remap: list[int] | None,
) -> None:
    """Compare a geometry-pair array (``collidable_pairs``/``excluded_pairs``).

    These hold geom indices that are absolute w.r.t. the model rather than per-entity rows, so
    they cannot be row-permuted or index-remapped by the generic array-attribute comparison:
    remapping the referenced geom indices can change both a pair's internal (min, max) order and
    the array's sort order, so rows are canonicalized before comparing.
    """
    obj_name = geoms0.__class__.__name__
    pairs0 = getattr(geoms0, attr).numpy()
    pairs1 = getattr(geoms1, attr).numpy()
    test.assertEqual(pairs0.shape, pairs1.shape, f"{obj_name}.{attr} shapes are not equal.")
    if pairs0.size == 0:
        return
    if geom_index_remap is not None:
        remap = np.asarray(geom_index_remap)
        pairs1 = remap[pairs1]
    canonical0 = sorted(tuple(sorted(pair)) for pair in pairs0.tolist())
    canonical1 = sorted(tuple(sorted(pair)) for pair in pairs1.tolist())
    test.assertEqual(canonical0, canonical1, f"{obj_name}.{attr} are not equal.")


def _assert_joint_packed_arrays_equal(
    test: unittest.TestCase,
    joints0: JointsModel,
    joints1: JointsModel,
    attributes: list[str],
    offset_attr: str,
    perm: list[int],
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    offsets0 = getattr(joints0, offset_attr).numpy()
    offsets1 = getattr(joints1, offset_attr).numpy()
    for attr in attributes:
        if not hasattr(joints0, attr) or not hasattr(joints1, attr):
            continue
        arr0 = getattr(joints0, attr)
        arr1 = getattr(joints1, attr)
        if arr0 is None or arr1 is None:
            continue
        values0 = arr0.numpy()
        values1 = arr1.numpy()
        for ref_idx in range(joints0.num_joints):
            other_idx = perm[ref_idx]
            start0, end0 = offsets0[ref_idx], offsets0[ref_idx + 1]
            start1, end1 = offsets1[other_idx], offsets1[other_idx + 1]
            test.assertEqual(
                end0 - start0,
                end1 - start1,
                msg=f"{joints0.__class__.__name__}.{attr} slice size mismatch for joint {ref_idx}.",
            )
            np.testing.assert_allclose(
                actual=values0[start0:end0],
                desired=values1[start1:end1],
                err_msg=f"{joints0.__class__.__name__}.{attr} are not equal for joint {ref_idx}.",
                rtol=rtol.get(attr, 1e-6) if rtol else 1e-6,
                atol=atol.get(attr, 1e-6) if atol else 1e-6,
            )


def assert_model_bodies_equal(
    test: unittest.TestCase,
    bodies0: RigidBodiesModel,
    bodies1: RigidBodiesModel,
    excluded: list[str] | None = None,
    mapping: list[int] | None = None,
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    """Compare two rigid-body models, optionally matching rows by label permutation."""
    assert_scalar_attributes_equal(test, bodies0, bodies1, ["num_bodies"])
    if excluded is None or "label" not in excluded:
        assert_scalar_attributes_equal(test, bodies0, bodies1, ["label"], mapping=mapping)
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
    if mapping is not None and "bid" in array_attributes:
        # `bid` is world-local and positional: once rows are reordered there is no cross-model
        # value left to compare, only the per-model invariant that it is still a valid ordering.
        array_attributes = [attr for attr in array_attributes if attr != "bid"]
        _assert_world_local_ids(test, bodies0, "bid")
        _assert_world_local_ids(test, bodies1, "bid")
    assert_array_attributes_equal(
        test,
        bodies0,
        bodies1,
        array_attributes,
        rtol=rtol,
        atol=atol,
        mapping=mapping,
    )


def assert_model_joints_equal(
    test: unittest.TestCase,
    joints0: JointsModel,
    joints1: JointsModel,
    excluded: list[str] | None = None,
    mapping: list[int] | None = None,
    body_index_remap: list[int] | None = None,
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    """Compare two joint models, optionally matching rows and remapping body references."""
    assert_scalar_attributes_equal(test, joints0, joints1, ["num_joints"])
    if excluded is None or "label" not in excluded:
        assert_scalar_attributes_equal(test, joints0, joints1, ["label"], mapping=mapping)
    dof_flat_attributes = [
        "q_j_min",
        "q_j_max",
        "dq_j_max",
        "tau_j_max",
        "a_j",
        "b_j",
        "k_p_j",
        "k_d_j",
        "dq_j_0",
    ]
    coord_flat_attributes = ["q_j_0"]
    per_joint_attributes = [
        "wid",
        "jid",
        "dof_type",
        "act_type",
        "bid_B",
        "bid_F",
        "B_r_Bj",
        "F_r_Fj",
        "X_Bj",
        "X_Fj",
        "num_coords",
        "num_dofs",
        "num_cts",
        "num_dynamic_cts",
        "num_kinematic_cts",
    ]
    if mapping is None:
        per_joint_attributes.extend(dof_flat_attributes)
        per_joint_attributes.extend(coord_flat_attributes)
        per_joint_attributes.extend(
            [
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
        )
    if excluded:
        per_joint_attributes = [attr for attr in per_joint_attributes if attr not in excluded]
        dof_flat_attributes = [attr for attr in dof_flat_attributes if attr not in excluded]
        coord_flat_attributes = [attr for attr in coord_flat_attributes if attr not in excluded]
    if mapping is not None and "jid" in per_joint_attributes:
        # See the matching comment on `bid` in assert_model_bodies_equal: `jid` is world-local
        # and positional, so once rows are reordered only the per-model invariant is meaningful.
        per_joint_attributes = [attr for attr in per_joint_attributes if attr != "jid"]
        _assert_world_local_ids(test, joints0, "jid")
        _assert_world_local_ids(test, joints1, "jid")
    index_remaps = None
    if body_index_remap is not None:
        index_remaps = {
            "bid_B": body_index_remap,
            "bid_F": body_index_remap,
        }
    assert_array_attributes_equal(
        test,
        joints0,
        joints1,
        per_joint_attributes,
        mapping=mapping,
        index_remaps=index_remaps,
        rtol=rtol,
        atol=atol,
    )
    if mapping is not None:
        _assert_joint_packed_arrays_equal(
            test, joints0, joints1, dof_flat_attributes, "dofs_offset", mapping, rtol=rtol, atol=atol
        )
        _assert_joint_packed_arrays_equal(
            test, joints0, joints1, coord_flat_attributes, "coords_offset", mapping, rtol=rtol, atol=atol
        )


def assert_model_geoms_equal(
    test: unittest.TestCase,
    geoms0: GeometriesModel,
    geoms1: GeometriesModel,
    excluded: list[str] | None = None,
    mapping: list[int] | None = None,
    body_index_remap: list[int] | None = None,
    geom_index_remap: list[int] | None = None,
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    """Compare two geometry models, optionally matching rows and remapping body/geometry references."""
    scalar_attributes = [
        "num_geoms",
        "num_collidable",
        "num_collidable_pairs",
        "num_excluded_pairs",
        "model_minimum_contacts",
        "world_minimum_contacts",
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
    ]
    pair_attributes = ["collidable_pairs", "excluded_pairs"]
    if excluded:
        scalar_attributes = [attr for attr in scalar_attributes if attr not in excluded]
        array_attributes = [attr for attr in array_attributes if attr not in excluded]
        pair_attributes = [attr for attr in pair_attributes if attr not in excluded]
    assert_scalar_attributes_equal(test, geoms0, geoms1, scalar_attributes)
    if excluded is None or "label" not in excluded:
        assert_scalar_attributes_equal(test, geoms0, geoms1, ["label"], mapping=mapping)
    if mapping is not None and "gid" in array_attributes:
        # See the matching comment on `bid` in assert_model_bodies_equal: `gid` is world-local
        # and positional, so once rows are reordered only the per-model invariant is meaningful.
        array_attributes = [attr for attr in array_attributes if attr != "gid"]
        _assert_world_local_ids(test, geoms0, "gid")
        _assert_world_local_ids(test, geoms1, "gid")
    index_remaps = {"bid": body_index_remap} if body_index_remap is not None else None
    assert_array_attributes_equal(
        test,
        geoms0,
        geoms1,
        array_attributes,
        rtol=rtol,
        atol=atol,
        mapping=mapping,
        index_remaps=index_remaps,
    )
    for attr in pair_attributes:
        _assert_geom_pairs_equal(test, geoms0, geoms1, attr, geom_index_remap)


def assert_model_materials_equal(
    test: unittest.TestCase, materials0: MaterialsModel, materials1: MaterialsModel, excluded: list[str] | None = None
) -> None:
    assert_scalar_attributes_equal(test, materials0, materials1, ["num_materials"])
    array_attributes = [
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
    allow_reordering: bool = False,
) -> None:
    """Compare two Kamino models, allowing for reordering of entities by label within each world.

    Args:
        skip_geom_source_ptr: If True, excludes the geometry source ``ptr`` attribute.
        skip_geom_group_and_collides: If True, excludes the geometry ``group``/``collides``
            attributes.
        skip_geom_margin_and_gap: If True, excludes the geometry ``margin``/``gap`` attributes.
        excluded: Attribute names to exclude from all comparisons.
        rtol: Per-attribute relative tolerance overrides for floating-point array comparisons.
        atol: Per-attribute absolute tolerance overrides for floating-point array comparisons.
        allow_reordering: If True, bodies/joints/geoms are matched by label within each world
            instead of by row order, so ``model0`` and ``model1`` may store entities of the same
            world in a different order. Defaults to ``False``.
    """
    assert_model_size_equal(test, model0.size, model1.size, excluded)

    body_mapping = None
    joint_mapping = None
    geom_mapping = None
    body_index_remap = None
    joint_index_remap = None
    geom_index_remap = None
    if allow_reordering:
        num_worlds = model0.info.num_worlds

        def _label_mapping(
            labels_0: list[str] | None,
            labels_1: list[str] | None,
            ranges: list[tuple[int, int]],
        ) -> list[int]:
            """Return indices into ``labels_1`` that align with ``labels_0`` within each world."""
            if labels_0 is None or labels_1 is None:
                test.fail(
                    "Cannot compare with allow_reordering=True because label is unset on one of "
                    f"the models (model0 label is None: {labels_0 is None}, "
                    f"model1 label is None: {labels_1 is None})."
                )
            test.assertEqual(len(labels_0), len(labels_1))
            mapping = list(range(len(labels_0)))
            for world_id, (range_start, range_end) in enumerate(ranges):
                labels_0_world = labels_0[range_start:range_end]
                labels_1_world = labels_1[range_start:range_end]
                test.assertEqual(
                    sorted(labels_0_world),
                    sorted(labels_1_world),
                    f"Label sets differ in world {world_id}.",
                )
                index_1 = {label: i for i, label in enumerate(labels_1_world)}
                test.assertEqual(
                    len(index_1),
                    len(labels_1_world),
                    f"Duplicate labels found in world {world_id}.",
                )
                for local_idx, label in enumerate(labels_0_world):
                    mapping[range_start + local_idx] = range_start + index_1[label]
            return mapping

        def _entity_index_remap(mapping: list[int]) -> list[int]:
            """Map global entity indices in the other model to indices in the reference model."""
            remap = [0] * len(mapping)
            for ref_idx, other_idx in enumerate(mapping):
                remap[other_idx] = ref_idx
            return remap

        def _none_if_identity(mapping: list[int]) -> list[int] | None:
            """Collapse an identity permutation back to `None`.

            Preserves the strict comparison path (which validates more, e.g. per-joint offset
            attributes,) for worlds where entities are already in matching order, and only pays
            for the permutation-aware comparison where reordering actually occurred.
            """
            return None if mapping == list(range(len(mapping))) else mapping

        # Ranges are already validated to match through the check on the model
        # info, so the same ranges can be used for both models
        bodies_offset = model0.info.bodies_offset.numpy().tolist()
        body_ranges = [(bodies_offset[w], bodies_offset[w + 1]) for w in range(num_worlds)]
        joints_offset = model0.info.joints_offset.numpy().tolist()
        num_joints = model0.info.num_joints.numpy().tolist()
        joint_ranges = [(joints_offset[w], joints_offset[w] + num_joints[w]) for w in range(num_worlds)]
        geoms_offset = model0.info.geoms_offset.numpy().tolist()
        num_geoms = model0.info.num_geoms.numpy().tolist()
        geom_ranges = [(geoms_offset[w], geoms_offset[w] + num_geoms[w]) for w in range(num_worlds)]

        body_mapping = _none_if_identity(_label_mapping(model0.bodies.label, model1.bodies.label, body_ranges))
        joint_mapping = _none_if_identity(_label_mapping(model0.joints.label, model1.joints.label, joint_ranges))
        geom_mapping = _none_if_identity(_label_mapping(model0.geoms.label, model1.geoms.label, geom_ranges))

        if body_mapping is not None:
            body_index_remap = _entity_index_remap(body_mapping)
        if joint_mapping is not None:
            joint_index_remap = _entity_index_remap(joint_mapping)
        if geom_mapping is not None:
            geom_index_remap = _entity_index_remap(geom_mapping)

    assert_model_info_equal(
        test,
        model0.info,
        model1.info,
        excluded,
        body_index_remap=body_index_remap,
        joint_index_remap=joint_index_remap,
    )
    assert_model_bodies_equal(
        test,
        model0.bodies,
        model1.bodies,
        excluded,
        mapping=body_mapping,
        rtol=rtol,
        atol=atol,
    )
    assert_model_joints_equal(
        test,
        model0.joints,
        model1.joints,
        excluded,
        mapping=joint_mapping,
        body_index_remap=body_index_remap,
        rtol=rtol,
        atol=atol,
    )
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
        mapping=geom_mapping,
        body_index_remap=body_index_remap,
        geom_index_remap=geom_index_remap,
        rtol=rtol,
        atol=atol,
    )
    assert_model_materials_equal(test, model0.materials, model1.materials, excluded)
    assert_model_material_pairs_equal(test, model0.material_pairs, model1.material_pairs, excluded)
