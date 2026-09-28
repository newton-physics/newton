# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: COMPARISON UTILITIES
"""

import unittest
from typing import Any

import numpy as np
import warp as wp

from newton import BodyFlags, Model
from newton._src.solvers.kamino._src.core.bodies import convert_body_com_to_origin
from newton._src.solvers.kamino._src.core.control import ControlKamino
from newton._src.solvers.kamino._src.core.joints import JointActuationType
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.utils import logger as msg

###
# Module interface
###

__all__ = [
    "assert_array_attributes_equal",
    "assert_control_equal",
    "assert_model_conversion_consistency",
    "assert_model_info_size_consistency",
    "assert_state_equal",
]


###
# Utilities
###


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


def _assert_array_is_alias(
    test: unittest.TestCase,
    kamino_obj: Any,
    kamino_attr: str,
    newton_model: Model,
    newton_attr: str,
) -> None:
    """Assert a Kamino container attribute is the exact same array as the Newton
    model attribute it was converted from, i.e. the conversion aliases it rather
    than copying or recomputing it.
    """
    kamino_array = getattr(kamino_obj, kamino_attr)
    newton_array = getattr(newton_model, newton_attr)
    if kamino_array is None and newton_array is None:
        return
    test.assertIs(
        kamino_array,
        newton_array,
        f"{kamino_obj.__class__.__name__}.{kamino_attr} is not an alias of Model.{newton_attr}.",
    )


###
# Model conversion consistency checks
###


def _expected_wid(model_newton: Model, newton_attr: str) -> np.ndarray:
    """Compute the expected Kamino `wid` array for a `newton_attr` in
    ``{"body_world", "joint_world", "shape_world"}``, which might be
    renumbered for single-world models.
    """
    raw = getattr(model_newton, newton_attr).numpy()
    if model_newton.world_count != 1 or not np.any(raw < 0):
        return raw
    has_dedicated_global_gravity = model_newton.gravity.shape[0] > model_newton.world_count
    if newton_attr == "body_world" and has_dedicated_global_gravity:
        return raw
    expected = raw.copy()
    expected[expected < 0] = 0
    return expected


def _assert_model_bodies_conversion_consistency(
    test: unittest.TestCase,
    model_newton: Model,
    model_kamino: ModelKamino,
    excluded: list[str] | None = None,
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    """Check `RigidBodiesModel` fields against the `newton.Model` they were converted from."""
    excluded = excluded or []
    rtol = rtol or {}
    atol = atol or {}
    bodies = model_kamino.bodies

    # Pure aliases: mass, inertia, COM, and initial velocity.
    aliases = [
        ("i_r_com_i", "body_com"),
        ("u_i_0", "body_qd"),
    ]
    for kamino_attr, newton_attr in aliases:
        if kamino_attr not in excluded:
            _assert_array_is_alias(test, bodies, kamino_attr, model_newton, newton_attr)

    if model_newton.body_count == 0:
        return

    # `wid` may be renumbered for single-world models with global (-1) entities,
    # so it is compared against the expected post-renumbering value.
    if "wid" not in excluded:
        np.testing.assert_array_equal(
            bodies.wid.numpy(),
            _expected_wid(model_newton, "body_world"),
            err_msg="RigidBodiesModel.wid does not match the expected Model.body_world.",
        )

    # Inverse transformation: mass, inertia, and their inverses are near-identical
    # copies, except for bodies flagged KINEMATIC or PROXY.
    if "m_i" not in excluded or "inv_m_i" not in excluded or "i_I_i" not in excluded or "inv_i_I_i" not in excluded:
        body_flags = model_newton.body_flags.numpy()
        is_kinematic = body_flags != BodyFlags.DYNAMIC
        if "m_i" not in excluded:
            m_i_expected = model_newton.body_mass.numpy()
            m_i_expected[is_kinematic] = 0.0
            np.testing.assert_array_equal(
                bodies.m_i.numpy(),
                m_i_expected,
                err_msg="RigidBodiesModel.m_i does not match expected masked Model.body_mass",
            )
        if "inv_m_i" not in excluded:
            inv_m_i_expected = model_newton.body_inv_mass.numpy()
            inv_m_i_expected[is_kinematic] = 0.0
            np.testing.assert_array_equal(
                bodies.inv_m_i.numpy(),
                inv_m_i_expected,
                err_msg="RigidBodiesModel.inv_m_i does not match expected masked Model.body_inv_mass",
            )
        if "i_I_i" not in excluded:
            i_I_i_expected = model_newton.body_inertia.numpy()
            i_I_i_expected[is_kinematic, :, :] = 0.0
            np.testing.assert_array_equal(
                bodies.i_I_i.numpy(),
                i_I_i_expected,
                err_msg="RigidBodiesModel.i_I_i does not match expected masked Model.body_inertia",
            )
        if "inv_i_I_i" not in excluded:
            inv_i_I_i_expected = model_newton.body_inv_inertia.numpy()
            inv_i_I_i_expected[is_kinematic, :, :] = 0.0
            np.testing.assert_array_equal(
                bodies.inv_i_I_i.numpy(),
                inv_i_I_i_expected,
                err_msg="RigidBodiesModel.m_inv_i_I_i does not match expected masked Model.body_inv_inertia",
            )

    # Inverse transformation: `q_i_0` stores COM-frame world poses; inverting it
    # back to body-origin frame must recover `Model.body_q` exactly.
    if "q_i_0" not in excluded:
        body_q_recovered = wp.empty_like(bodies.q_i_0)
        convert_body_com_to_origin(bodies.i_r_com_i, bodies.q_i_0, body_q_recovered)
        np.testing.assert_allclose(
            body_q_recovered.numpy(),
            model_newton.body_q.numpy(),
            rtol=rtol.get("q_i_0", 1e-5),
            atol=atol.get("q_i_0", 1e-6),
            err_msg="RigidBodiesModel.q_i_0 does not recover Model.body_q via convert_body_com_to_origin().",
        )


def _assert_model_joints_conversion_consistency(
    test: unittest.TestCase,
    model_newton: Model,
    model_kamino: ModelKamino,
    excluded: list[str] | None = None,
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    """Check `JointsModel` fields against the `newton.Model` they were converted from."""
    excluded = excluded or []
    rtol = rtol or {}
    atol = atol or {}
    joints = model_kamino.joints

    # Pure aliases: limits, gains, dynamics coefficients, initial coordinates,
    # and parent/child body indices.
    aliases = [
        ("q_j_min", "joint_limit_lower"),
        ("q_j_max", "joint_limit_upper"),
        ("dq_j_max", "joint_velocity_limit"),
        ("tau_j_max", "joint_effort_limit"),
        ("a_j", "joint_armature"),
        ("b_j", "joint_damping"),
        ("f_j", "joint_friction"),
        ("k_p_j", "joint_target_ke"),
        ("k_d_j", "joint_target_kd"),
        ("q_j_0", "joint_q"),
        ("dq_j_0", "joint_qd"),
        ("bid_B", "joint_parent"),
        ("bid_F", "joint_child"),
    ]
    for kamino_attr, newton_attr in aliases:
        if kamino_attr not in excluded:
            _assert_array_is_alias(test, joints, kamino_attr, model_newton, newton_attr)

    # `wid` may be renumbered for single-world models with global (-1) entities,
    # so it is compared against the expected post-renumbering value.
    if "wid" not in excluded:
        np.testing.assert_array_equal(
            joints.wid.numpy(),
            _expected_wid(model_newton, "joint_world"),
            err_msg="JointsModel.wid does not match the expected Model.joint_world.",
        )

    if model_newton.joint_count == 0:
        return

    # Inverse transformation: `B_r_Bj`/`F_r_Fj` are the joint attachment points
    # expressed relative to each body's COM; adding the COM back must recover
    # the parent/child joint transform translations Newton stores in
    # `joint_X_p`/`joint_X_c`.
    if "B_r_Bj" not in excluded or "F_r_Fj" not in excluded:
        body_com = model_newton.body_com.numpy()
        joint_parent = model_newton.joint_parent.numpy()
        joint_child = model_newton.joint_child.numpy()
        p_r_p_j = model_newton.joint_X_p.numpy()[:, :3]
        c_r_c_j = model_newton.joint_X_c.numpy()[:, :3]

        if "B_r_Bj" not in excluded:
            has_parent = joint_parent >= 0
            parent_com = np.zeros_like(p_r_p_j)
            parent_com[has_parent] = body_com[joint_parent[has_parent]]
            np.testing.assert_allclose(
                joints.B_r_Bj.numpy() + parent_com,
                p_r_p_j,
                rtol=rtol.get("B_r_Bj", 1e-5),
                atol=atol.get("B_r_Bj", 1e-6),
                err_msg="JointsModel.B_r_Bj does not recover Model.joint_X_p's translation.",
            )

        if "F_r_Fj" not in excluded:
            child_com = body_com[joint_child]
            np.testing.assert_allclose(
                joints.F_r_Fj.numpy() + child_com,
                c_r_c_j,
                rtol=rtol.get("F_r_Fj", 1e-5),
                atol=atol.get("F_r_Fj", 1e-6),
                err_msg="JointsModel.F_r_Fj does not recover Model.joint_X_c's translation.",
            )

    # Inverse transformation: `X_Bj`/`X_Fj` re-express the joint's DoF axes
    # (defined in the joint's own local frame) in the parent/child body frame,
    # by rotating them with the parent/child joint transform's rotation
    # (`Model.joint_X_p`/`joint_X_c`).
    # Columns beyond a joint's DoF count complete an orthonormal basis chosen
    # internally by the conversion and have no Newton counterpart, so they are
    # never compared.
    if "X_Bj" not in excluded or "X_Fj" not in excluded:
        joint_axis = model_newton.joint_axis.numpy()
        joint_qd_start = model_newton.joint_qd_start.numpy()
        num_dofs = joints.num_dofs.numpy()
        joint_X_p = model_newton.joint_X_p.numpy()
        joint_X_c = model_newton.joint_X_c.numpy()
        X_Bj = joints.X_Bj.numpy()
        X_Fj = joints.X_Fj.numpy()
        for j in range(model_newton.joint_count):
            dof_start = int(joint_qd_start[j])
            q_p = wp.quatf(*joint_X_p[j, 3:7])
            q_c = wp.quatf(*joint_X_c[j, 3:7])
            # X_Bj/X_Fj are always 3x3: a FREE joint has 6 DoFs (3 translational +
            # 3 rotational) but only the first 3 (translational) axes feed the
            # matrix -- Newton requires them to equal the rotational axes for a
            # free joint, so comparing against the first 3 is valid either way.
            for k in range(min(int(num_dofs[j]), 3)):
                axis = wp.vec3f(*joint_axis[dof_start + k])
                if "X_Bj" not in excluded:
                    np.testing.assert_allclose(
                        X_Bj[j][:, k],
                        np.array(wp.quat_rotate(q_p, axis)),
                        rtol=rtol.get("X_Bj", 1e-5),
                        atol=atol.get("X_Bj", 1e-6),
                        err_msg=f"JointsModel.X_Bj column {k} of joint {j} does not match Newton model.",
                    )
                if "X_Fj" not in excluded:
                    np.testing.assert_allclose(
                        X_Fj[j][:, k],
                        np.array(wp.quat_rotate(q_c, axis)),
                        rtol=rtol.get("X_Fj", 1e-5),
                        atol=atol.get("X_Fj", 1e-6),
                        err_msg=f"JointsModel.X_Fj column {k} of joint {j} does not match Newton model.",
                    )

    # Inverse transformation: `dof_act_types` is a per-DoF lookup-table
    # conversion of Newton's `joint_target_mode` alone (gains/armature/damping/
    # effort_limit do not factor into this classification).
    if "dof_act_types" not in excluded and model_newton.joint_dof_count > 0:
        expected_dof_act_types = np.array(
            [int(JointActuationType.from_newton(mode)) for mode in model_newton.joint_target_mode.numpy()]
        )
        np.testing.assert_array_equal(
            joints.dof_act_types.numpy(),
            expected_dof_act_types,
            err_msg="JointsModel.dof_act_types does not match JointActuationType.from_newton(Model.joint_target_mode).",
        )


def _assert_model_geoms_conversion_consistency(
    test: unittest.TestCase,
    model_newton: Model,
    model_kamino: ModelKamino,
    excluded: list[str] | None = None,
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    """Check `GeometriesModel` fields against the `newton.Model` they were converted from."""
    excluded = excluded or []
    rtol = rtol or {}
    atol = atol or {}
    geoms = model_kamino.geoms

    # Pure aliases: label/type/flags/pointer/scale/collision-group/gap/margin
    # and the collidable-pair list are pure aliases.
    aliases = [
        ("label", "shape_label"),
        ("type", "shape_type"),
        ("flags", "shape_flags"),
        ("ptr", "shape_source_ptr"),
        ("params", "shape_scale"),
        ("group", "shape_collision_group"),
        ("gap", "shape_gap"),
        ("margin", "shape_margin"),
        ("collidable_pairs", "shape_contact_pairs"),
        ("bid", "shape_body"),
    ]
    for kamino_attr, newton_attr in aliases:
        if kamino_attr not in excluded:
            _assert_array_is_alias(test, geoms, kamino_attr, model_newton, newton_attr)

    # `wid` may be renumbered for single-world models with global (-1) entities,
    # so it is compared against the expected post-renumbering value.
    if "wid" not in excluded:
        np.testing.assert_array_equal(
            geoms.wid.numpy(),
            _expected_wid(model_newton, "shape_world"),
            err_msg="GeometriesModel.wid does not match the expected (possibly renumbered) Model.shape_world.",
        )

    if model_newton.shape_count == 0:
        return

    # Inverse transform: `offset` stores shape transforms with the translation
    # made COM-relative; adding the body's COM back must recover
    # `Model.shape_transform`.
    if "offset" not in excluded:
        body_com = model_newton.body_com.numpy()
        shape_body = model_newton.shape_body.numpy()
        shape_transform = model_newton.shape_transform.numpy()
        offset = geoms.offset.numpy()

        expected_translation = offset[:, :3].copy()
        attached = shape_body >= 0
        expected_translation[attached] += body_com[shape_body[attached]]
        np.testing.assert_allclose(
            expected_translation,
            shape_transform[:, :3],
            rtol=rtol.get("offset", 1e-5),
            atol=atol.get("offset", 1e-6),
            err_msg="GeometriesModel.offset translation does not recover Model.shape_transform.",
        )
        np.testing.assert_allclose(
            offset[:, 3:7],
            shape_transform[:, 3:7],
            rtol=rtol.get("offset", 1e-5),
            atol=atol.get("offset", 1e-6),
            err_msg="GeometriesModel.offset rotation does not match Model.shape_transform.",
        )

    # Inverse transform: `material` indexes into `ModelKamino.materials`,
    # deduplicated by (static friction, restitution); every collidable shape's
    # material properties must match the Newton shape properties it was
    # registered from.
    if "material" not in excluded:
        geom_material = geoms.material.numpy()
        has_material = geom_material >= 0
        material_idx = geom_material[has_material]
        shape_mu = model_newton.shape_material_mu.numpy()[has_material]
        shape_restitution = model_newton.shape_material_restitution.numpy()[has_material]

        static_friction = model_kamino.materials.static_friction.numpy()
        dynamic_friction = model_kamino.materials.dynamic_friction.numpy()
        restitution = model_kamino.materials.restitution.numpy()
        np.testing.assert_allclose(
            static_friction[material_idx],
            shape_mu,
            rtol=rtol.get("material", 1e-6),
            atol=atol.get("material", 1e-6),
            err_msg="MaterialsModel.static_friction does not match Model.shape_material_mu.",
        )
        np.testing.assert_allclose(
            dynamic_friction[material_idx],
            shape_mu,
            rtol=rtol.get("material", 1e-6),
            atol=atol.get("material", 1e-6),
            err_msg="MaterialsModel.dynamic_friction does not match Model.shape_material_mu.",
        )
        np.testing.assert_allclose(
            restitution[material_idx],
            shape_restitution,
            rtol=rtol.get("material", 1e-6),
            atol=atol.get("material", 1e-6),
            err_msg="MaterialsModel.restitution does not match Model.shape_material_restitution.",
        )


def assert_model_conversion_consistency(
    test: unittest.TestCase,
    model_newton: Model,
    model_kamino: ModelKamino,
    excluded: list[str] | None = None,
    rtol: dict[str, float] | None = None,
    atol: dict[str, float] | None = None,
) -> None:
    """Check that a `ModelKamino` built via `ModelKamino.from_newton()` is a
    faithful conversion of the `newton.Model` it was built from.

    Each field is validated against the documented semantics of the conversion
    itself, either as a strict alias of a Newton array or as an invertible
    transform of one.

    See `assert_model_bookkeeping_consistent` for the structural (per-world/per-joint
    count and offset) checks this does not cover.

    Args:
        test: The test case to report failures against.
        model_newton: The source `newton.Model` the conversion was built from.
        model_kamino: The `ModelKamino` produced by `ModelKamino.from_newton(model_newton)`.
        excluded: Kamino-side field names to skip (e.g. ``"q_i_0"``, ``"X_Bj"``,
            ``"material"``). A name is matched against whichever body/joint/geom
            field of that name exists, so e.g. ``"wid"`` skips it in all three.
        rtol: Per-field relative tolerance overrides for the invertible-transform
            checks (e.g. ``{"q_i_0": 1e-4}``), keyed the same way as `excluded`.
        atol: Per-field absolute tolerance overrides, keyed the same way as `rtol`.
    """
    _assert_model_bodies_conversion_consistency(
        test, model_newton, model_kamino, excluded=excluded, rtol=rtol, atol=atol
    )
    _assert_model_joints_conversion_consistency(
        test, model_newton, model_kamino, excluded=excluded, rtol=rtol, atol=atol
    )
    _assert_model_geoms_conversion_consistency(
        test, model_newton, model_kamino, excluded=excluded, rtol=rtol, atol=atol
    )


def assert_model_info_size_consistency(
    test: unittest.TestCase,
    model_kamino: ModelKamino,
    excluded: list[str] | None = None,
) -> None:
    """Check the internal self-consistency of `ModelKamino`'s structural bookkeeping.

    The per-world/per-joint counts and start-index offsets have no single Newton
    array to compare against, so instead this verifies the prefix-sum invariant
    every such (count, offset) pair must satisfy:
    - `offset[0] == 0`
    - `offset[i] + count[i] == offset[i + 1]` for every entity `i` but the last
    - `offset[-1] + count[-1] == sum(count) == <matching SizeKamino total>`

    Two field groups are covered:
    - Per-world counts/offsets on `ModelKaminoInfo` (e.g. `num_bodies`/`bodies_offset`).
    - Per-joint, model-wide counts/offsets on `JointsModel` (e.g. `num_coords`/`coords_offset`).

    Args:
        test: The test case to report failures against.
        model_kamino: The model.
        excluded: Count-array field names to skip (matching the first element of
            the relevant table, e.g. ``"num_bodies"`` or ``"num_dofs"``).
    """
    excluded = excluded or []

    def _check(count: np.ndarray, offset: np.ndarray, total: int, label: str) -> None:
        # Some offset arrays carry a trailing (grand-total) entry beyond the one
        # start index per entity, others don't; only the first `n` entries
        # (one start index per entity) are needed here.
        n = count.shape[0]
        test.assertGreaterEqual(offset.shape[0], n, f"{label}: offset array is shorter than the count array.")
        offset_n = offset[:n]
        test.assertEqual(int(offset_n[0]), 0, f"{label}: offset does not start at 0.")
        np.testing.assert_array_equal(
            offset_n[:-1] + count[:-1],
            offset_n[1:],
            err_msg=f"{label}: offset[i] + count[i] != offset[i + 1].",
        )
        test.assertEqual(int(offset_n[-1]) + int(count[-1]), total, f"{label}: last offset + count != total.")
        test.assertEqual(int(count.sum()), total, f"{label}: sum(count) != total.")
        if offset.shape[0] > n:
            # Some arrays carry one extra trailing entry equal to the grand
            # total; check it whenever the offset array is long enough to have one.
            test.assertEqual(int(offset[n]), total, f"{label}: trailing total entry does not match total.")

    # (info_count_attr, info_offset_attr, size_sum_attr) triples: per-world
    # entity/DoF/constraint counts in `ModelKaminoInfo`, each with a matching
    # per-world start-index offset array and model-wide total on `SizeKamino`.
    model_info_bookkeeping_fields: list[tuple[str, str, str]] = [
        ("num_bodies", "bodies_offset", "sum_of_num_bodies"),
        ("num_joints", "joints_offset", "sum_of_num_joints"),
        ("num_geoms", "geoms_offset", "sum_of_num_geoms"),
        ("num_body_dofs", "body_dofs_offset", "sum_of_num_body_dofs"),
        ("num_joint_coords", "joint_coords_offset", "sum_of_num_joint_coords"),
        ("num_joint_dofs", "joint_dofs_offset", "sum_of_num_joint_dofs"),
        ("num_passive_joint_coords", "joint_passive_coords_offset", "sum_of_num_passive_joint_coords"),
        ("num_passive_joint_dofs", "joint_passive_dofs_offset", "sum_of_num_passive_joint_dofs"),
        ("num_actuated_joint_coords", "joint_actuated_coords_offset", "sum_of_num_actuated_joint_coords"),
        ("num_actuated_joint_dofs", "joint_actuated_dofs_offset", "sum_of_num_actuated_joint_dofs"),
        ("num_joint_bilateral_cts", "joint_bilateral_cts_offset", "sum_of_num_bilateral_joint_cts"),
        ("num_joint_dynamic_cts", "joint_dynamic_cts_offset", "sum_of_num_dynamic_joint_cts"),
        ("num_joint_kinematic_cts", "joint_kinematic_cts_offset", "sum_of_num_kinematic_joint_cts"),
        ("num_joint_bounded_cts", "joint_bounded_cts_offset", "sum_of_num_bounded_joint_cts"),
        ("num_joint_friction_cts", "joint_friction_cts_offset", "sum_of_num_friction_joint_cts"),
        ("num_joint_effort_cts", "joint_effort_cts_offset", "sum_of_num_effort_joint_cts"),
    ]

    info = model_kamino.info
    size = model_kamino.size
    for count_attr, offset_attr, sum_attr in model_info_bookkeeping_fields:
        if count_attr in excluded:
            continue
        count = getattr(info, count_attr).numpy()
        offset = getattr(info, offset_attr).numpy()
        total = getattr(size, sum_attr)
        _check(count, offset, total, f"ModelKaminoInfo.{count_attr}/{offset_attr}")

    # (joints_count_attr, joints_offset_attr, size_sum_attr) triples: per-joint,
    # already-globalized (model-wide, not per-world) counts on `JointsModel`,
    # each with a matching (joint_count + 1)-length prefix-sum offset array.
    model_joints_bookkeeping_fields: list[tuple[str, str, str]] = [
        ("num_coords", "coords_offset", "sum_of_num_joint_coords"),
        ("num_dofs", "dofs_offset", "sum_of_num_joint_dofs"),
        ("num_bilateral_cts", "bilateral_cts_offset", "sum_of_num_bilateral_joint_cts"),
        ("num_dynamic_cts", "dynamic_cts_offset", "sum_of_num_dynamic_joint_cts"),
        ("num_kinematic_cts", "kinematic_cts_offset", "sum_of_num_kinematic_joint_cts"),
        ("num_bounded_cts", "bounded_cts_offset", "sum_of_num_bounded_joint_cts"),
        ("num_friction_cts", "friction_cts_offset", "sum_of_num_friction_joint_cts"),
        ("num_effort_cts", "effort_cts_offset", "sum_of_num_effort_joint_cts"),
    ]

    joints = model_kamino.joints
    for count_attr, offset_attr, sum_attr in model_joints_bookkeeping_fields:
        if count_attr in excluded:
            continue
        count = getattr(joints, count_attr).numpy()
        offset = getattr(joints, offset_attr).numpy()
        total = getattr(size, sum_attr)
        _check(count, offset, total, f"JointsModel.{count_attr}/{offset_attr}")


###
# Container comparisons
###


def assert_state_equal(
    test: unittest.TestCase, state0: StateKamino, state1: StateKamino, excluded: list[str] | None = None
) -> None:
    attributes = [
        "q_i",
        "u_i",
        "w_i",
        "q_j",
        "q_j_p",
        "dq_j",
        "lambda_kin_j",
        "lambda_dyn_j",
        "lambda_f_j",
        "lambda_tau_j",
    ]
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
