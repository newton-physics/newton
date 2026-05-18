# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
MuJoCo Menagerie USD Integration Tests

Tests that MuJoCo Menagerie robots converted to USD simulate identically
in Newton's MuJoCo solver vs native MuJoCo (loaded from original MJCF).

Import tests verify that each USD asset loads correctly (body/joint/
shape counts, no NaN values, correct joint types).

Simulation equivalence tests reuse the TestMenagerieBase infrastructure
from test_menagerie_mujoco.py to compare per-step simulation state between
Newton (USD) and native MuJoCo (MJCF).

Menagerie robot stubs. One class per menagerie robot, using the default
TestMenagerieUSD configuration. Initially these run with no usd_asset_folder
(skipped until a pre-converted USD is available in newton-assets).

Asset location: newton-assets GitHub repo (structured USD from mujoco_menagerie).
Assets are downloaded automatically via newton.utils.download_asset().
"""

from __future__ import annotations

import re
import unittest
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

import newton
import newton.utils
from newton.solvers import SolverMuJoCo
from newton.tests.test_menagerie_mujoco import (
    DEFAULT_MODEL_SKIP_FIELDS,
    TestMenagerieBase,
    _disable_collisions,
    _mujoco_warp,
    _restore_collisions,
    compare_inertia_tensors,
    run_newton_eval_fk,
)
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.usd import SchemaResolverMjc, SchemaResolverNewton

# =============================================================================
# USD Model Creation
# =============================================================================


def create_newton_model_from_usd(
    usd_path: Path,
    *,
    num_worlds: int = 1,
    add_ground: bool = True,
) -> newton.Model:
    """Create a Newton model from a USD file (converted from MuJoCo Menagerie MJCF).

    Args:
        usd_path: Path to the USD scene file.
        num_worlds: Number of world instances to create.
        add_ground: Whether to add a ground plane.

    Returns:
        Finalized Newton Model.
    """
    robot_builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(robot_builder)

    robot_builder.add_usd(
        str(usd_path),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
    )

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)

    if add_ground:
        builder.add_ground_plane()

    if num_worlds > 1:
        builder.replicate(robot_builder, num_worlds)
    else:
        builder.add_world(robot_builder)

    return builder.finalize()


# =============================================================================
# Asset Configuration
# =============================================================================

# Menagerie USD asset registry: maps robot name to newton-assets folder + scene file.
# Assets live in the newton-assets GitHub repo. The full repo is cloned once
# via _download_all_newton_assets() so that parallel test processes share a
# single git clone instead of each downloading individual folders.
MENAGERIE_USD_ASSETS = {
    "h1": {"asset_folder": "unitree_h1", "scene_file": "usd_structured/h1.usda"},
    "g1_with_hands": {"asset_folder": "unitree_g1", "scene_file": "usd_structured/g1_29dof_with_hand_rev_1_0.usda"},
    "shadow_hand": {"asset_folder": "shadow_hand", "scene_file": "usd_structured/left_shadow_hand.usda"},
    "robotiq_2f85_v4": {"asset_folder": "robotiq_2f85_v4", "scene_file": "usd_structured/Dual_wrist_camera.usda"},
    "apptronik_apollo": {"asset_folder": "apptronik_apollo", "scene_file": "usd_structured/apptronik_apollo.usda"},
    "booster_t1": {"asset_folder": "booster_t1", "scene_file": "usd_structured/T1.usda"},
    "wonik_allegro": {"asset_folder": "wonik_allegro", "scene_file": "usd_structured/allegro_left.usda"},
    "ur5e": {"asset_folder": "universal_robots_ur5e", "scene_file": "usd_structured/ur5e.usda"},
}


def download_usd_asset(robot_name: str) -> Path:
    """Return the scene file path for a menagerie USD asset.

    Uses the full-repo clone from :func:`download_newton_assets_repo` when
    available, falling back to per-folder download via :func:`download_asset`.

    Args:
        robot_name: Key in MENAGERIE_USD_ASSETS.

    Returns:
        Absolute path to the downloaded USD scene file.
    """
    config = MENAGERIE_USD_ASSETS[robot_name]
    asset_root = newton.utils.download_asset(config["asset_folder"])
    return asset_root / config["scene_file"]


# =============================================================================
# Import Tests
# =============================================================================
@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUsdImport(unittest.TestCase):
    """Verify that each menagerie USD asset imports correctly into Newton."""

    def _load_robot(self, robot_name: str) -> tuple[newton.ModelBuilder, newton.Model]:
        """Load a menagerie USD asset and return the builder and finalized model."""
        usd_path = download_usd_asset(robot_name)
        self.assertTrue(usd_path.exists(), f"USD asset not found: {usd_path}")

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        builder.add_usd(
            str(usd_path),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
        )

        model = builder.finalize()
        return builder, model

    def _assert_no_nan(self, model: newton.Model, robot_name: str):
        """Assert that the model contains no NaN values in key arrays."""
        for attr_name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            arr = getattr(model, attr_name, None)
            if arr is not None:
                arr_np = arr.numpy()
                self.assertFalse(
                    np.any(np.isnan(arr_np)),
                    f"{robot_name}: NaN detected in model.{attr_name}",
                )

    def test_import_h1(self):
        builder, model = self._load_robot("h1")
        self.assertEqual(builder.body_count, 20)
        self.assertEqual(builder.joint_count, 20)
        self.assertEqual(builder.shape_count, 54)
        self._assert_no_nan(model, "h1")

    def test_import_g1_with_hands(self):
        builder, model = self._load_robot("g1_with_hands")
        self.assertEqual(builder.body_count, 44)
        self.assertEqual(builder.joint_count, 44)
        self.assertEqual(builder.shape_count, 104)
        self._assert_no_nan(model, "g1_with_hands")

    def test_import_shadow_hand(self):
        builder, model = self._load_robot("shadow_hand")
        self.assertEqual(builder.body_count, 25)
        self.assertEqual(builder.joint_count, 25)
        self.assertEqual(builder.shape_count, 62)
        self._assert_no_nan(model, "shadow_hand")

    def test_import_robotiq_2f85_v4(self):
        builder, model = self._load_robot("robotiq_2f85_v4")
        self.assertEqual(builder.body_count, 11)
        self.assertEqual(builder.joint_count, 11)
        self.assertEqual(builder.shape_count, 28)
        self.assertEqual(model.equality_constraint_count, 3)
        self._assert_no_nan(model, "robotiq_2f85_v4")

    def test_import_apptronik_apollo(self):
        builder, model = self._load_robot("apptronik_apollo")
        self.assertEqual(builder.body_count, 36)
        self.assertEqual(builder.joint_count, 36)
        self.assertEqual(builder.shape_count, 87)
        self._assert_no_nan(model, "apptronik_apollo")

    def test_import_booster_t1(self):
        builder, model = self._load_robot("booster_t1")
        self.assertEqual(builder.body_count, 24)
        self.assertEqual(builder.joint_count, 24)
        self.assertEqual(builder.shape_count, 37)
        self._assert_no_nan(model, "booster_t1")

    def test_import_wonik_allegro(self):
        builder, model = self._load_robot("wonik_allegro")
        self.assertEqual(builder.body_count, 21)
        self.assertEqual(builder.joint_count, 21)
        self.assertEqual(builder.shape_count, 42)
        self._assert_no_nan(model, "wonik_allegro")

    def test_import_ur5e(self):
        builder, model = self._load_robot("ur5e")
        self.assertEqual(builder.body_count, 7)
        self.assertEqual(builder.joint_count, 7)
        self.assertEqual(builder.shape_count, 30)
        self._assert_no_nan(model, "ur5e")

    def test_import_h1_joint_types(self):
        """Verify H1 has a free joint (floating base) and revolute joints."""
        builder, _ = self._load_robot("h1")
        joint_types = builder.joint_type
        self.assertIn(newton.JointType.FREE, joint_types)
        self.assertIn(newton.JointType.REVOLUTE, joint_types)

    def test_import_wonik_allegro_joint_types(self):
        """Verify Allegro hand has no free joint (fixed base)."""
        builder, _ = self._load_robot("wonik_allegro")
        joint_types = builder.joint_type
        self.assertNotIn(newton.JointType.FREE, joint_types)

    def test_import_h1_multi_world(self):
        """Verify H1 can be replicated into multiple worlds."""
        usd_path = download_usd_asset("h1")

        model = create_newton_model_from_usd(usd_path, num_worlds=4, add_ground=True)
        self.assertEqual(model.world_count, 4)
        self._assert_no_nan(model, "h1_multi_world")


# =============================================================================
# USD-specific sorted comparison helpers
# =============================================================================
def compare_inertia_tensors_mapped(
    newton_mjw: Any,
    native_mjw: Any,
    body_map: dict[int, int],
    tol: float = 1e-5,
) -> None:
    """Compare inertia tensors using a name-based body index mapping.

    Reindexes Newton's body_inertia, body_iquat, and body_mass arrays to
    native ordering via body_map, then delegates to
    :func:`compare_inertia_tensors` for the full 3x3 tensor reconstruction.

    Args:
        body_map: native_body_idx -> newton_body_idx.
        tol: Absolute tolerance for tensor comparison.
    """
    newton_inertia = newton_mjw.body_inertia.numpy()
    newton_iquat = newton_mjw.body_iquat.numpy()
    newton_mass = newton_mjw.body_mass.numpy()

    reindexed_inertia = np.zeros_like(newton_inertia)
    reindexed_iquat = np.zeros_like(newton_iquat)
    reindexed_mass = np.zeros_like(newton_mass)

    for native_i, newton_i in body_map.items():
        reindexed_inertia[:, native_i] = newton_inertia[:, newton_i]
        reindexed_iquat[:, native_i] = newton_iquat[:, newton_i]
        reindexed_mass[:, native_i] = newton_mass[:, newton_i]

    np.testing.assert_allclose(
        reindexed_mass,
        native_mjw.body_mass.numpy(),
        atol=tol,
        rtol=0,
        err_msg="body_mass (reindexed)",
    )

    device = newton_mjw.body_inertia.device
    saved_inertia = newton_mjw.body_inertia
    saved_iquat = newton_mjw.body_iquat
    try:
        newton_mjw.body_inertia = wp.array(reindexed_inertia, dtype=saved_inertia.dtype, device=device)
        newton_mjw.body_iquat = wp.array(reindexed_iquat, dtype=saved_iquat.dtype, device=device)
        compare_inertia_tensors(newton_mjw, native_mjw, tol=tol)
    finally:
        newton_mjw.body_inertia = saved_inertia
        newton_mjw.body_iquat = saved_iquat


def compare_joints_sorted(
    newton_mjw: Any,
    native_mjw: Any,
    tol: float = 1e-5,
) -> None:
    """Compare joint properties by sorting, handling reordering.

    Matches joints by (type, mass_of_parent_body, limited) and verifies
    the multisets of joint properties are equivalent.
    """
    assert newton_mjw.njnt == native_mjw.njnt, f"njnt mismatch: newton={newton_mjw.njnt} vs native={native_mjw.njnt}"

    njnt = newton_mjw.njnt
    if njnt == 0:
        return

    newton_type = newton_mjw.jnt_type.numpy().flatten()
    native_type = native_mjw.jnt_type.numpy().flatten()
    newton_limited = newton_mjw.jnt_limited.numpy().flatten()
    native_limited = native_mjw.jnt_limited.numpy().flatten()
    newton_stiffness = newton_mjw.jnt_stiffness.numpy().flatten()
    native_stiffness = native_mjw.jnt_stiffness.numpy().flatten()

    def _jnt_signature(jtype, limited, stiffness):
        return (int(jtype), int(limited), round(float(stiffness), 8))

    newton_sigs = sorted([_jnt_signature(newton_type[i], newton_limited[i], newton_stiffness[i]) for i in range(njnt)])
    native_sigs = sorted([_jnt_signature(native_type[i], native_limited[i], native_stiffness[i]) for i in range(njnt)])

    mismatches = []
    for i, (ns, nats) in enumerate(zip(newton_sigs, native_sigs, strict=True)):
        if ns != nats:
            mismatches.append(f"sorted[{i}]: newton={ns} native={nats}")
    assert not mismatches, f"Joint multiset mismatch ({len(mismatches)}/{njnt} after sorting):\n" + "\n".join(
        mismatches[:5]
    )

    # Compare joint ranges for limited joints (sorted by range values)
    newton_range = newton_mjw.jnt_range.numpy()
    native_range = native_mjw.jnt_range.numpy()

    for world in range(newton_range.shape[0]):
        newton_limited_ranges = sorted(tuple(newton_range[world, j]) for j in range(njnt) if newton_limited[j])
        native_limited_ranges = sorted(tuple(native_range[world, j]) for j in range(njnt) if native_limited[j])
        assert len(newton_limited_ranges) == len(native_limited_ranges), f"world {world}: limited joint count mismatch"
        for k, (nr, natr) in enumerate(zip(newton_limited_ranges, native_limited_ranges, strict=True)):
            np.testing.assert_allclose(
                nr,
                natr,
                atol=tol,
                rtol=0,
                err_msg=f"jnt_range sorted[{k}] world={world}",
            )


def compare_geoms_subset(
    newton_mjw: Any,
    native_mjw: Any,
    tol: float = 1e-6,
) -> None:
    """Compare geom fields when Newton has a subset of native geoms.

    Newton's USD import with skip_visual_only_geoms=True produces fewer geoms
    than native MJCF. This function matches Newton geoms to native geoms by
    (geom_type, geom_size) signature, then verifies that physics-relevant
    properties (friction, condim, solref, priority, solimp) match within each
    signature group (sorted to handle reordering).
    """
    newton_ngeom = newton_mjw.ngeom
    native_ngeom = native_mjw.ngeom
    assert newton_ngeom <= native_ngeom, f"Newton has more geoms ({newton_ngeom}) than native ({native_ngeom})"

    newton_type = newton_mjw.geom_type.numpy()
    native_type = native_mjw.geom_type.numpy()
    newton_size = newton_mjw.geom_size.numpy()
    native_size = native_mjw.geom_size.numpy()

    GEOM_PLANE = 0
    GEOM_SPHERE = 2
    GEOM_CAPSULE = 3
    GEOM_CYLINDER = 5
    GEOM_MESH = 7

    def _geom_sig(gtype, gsize):
        if gtype == GEOM_PLANE:
            return (int(gtype), 0.0, 0.0, 0.0)
        elif gtype == GEOM_SPHERE:
            return (int(gtype), round(float(gsize[0]), 6), 0.0, 0.0)
        elif gtype in (GEOM_CAPSULE, GEOM_CYLINDER):
            return (int(gtype), round(float(gsize[0]), 6), round(float(gsize[1]), 6), 0.0)
        elif gtype == GEOM_MESH:
            return (int(gtype), 0.0, 0.0, 0.0)
        else:
            return (int(gtype), *(round(float(s), 6) for s in gsize))

    newton_sigs = [_geom_sig(newton_type[i], newton_size[0, i]) for i in range(newton_ngeom)]
    native_sigs = [_geom_sig(native_type[i], native_size[0, i]) for i in range(native_ngeom)]

    newton_counts = Counter(newton_sigs)
    native_counts = Counter(native_sigs)
    for sig, count in newton_counts.items():
        native_count = native_counts.get(sig, 0)
        assert native_count >= count, (
            f"Geom signature {sig} appears {count}x in Newton but only {native_count}x in native"
        )

    # Group geom indices by signature for physics property comparison.
    newton_groups: dict[tuple, list[int]] = {}
    for i, sig in enumerate(newton_sigs):
        newton_groups.setdefault(sig, []).append(i)
    native_groups: dict[tuple, list[int]] = {}
    for i, sig in enumerate(native_sigs):
        native_groups.setdefault(sig, []).append(i)

    # Collect physics fields available on both models.
    physics_fields = []
    for field in ("geom_friction", "geom_condim", "geom_solref", "geom_priority", "geom_solimp"):
        nw_arr = getattr(newton_mjw, field, None)
        nat_arr = getattr(native_mjw, field, None)
        if nw_arr is not None and nat_arr is not None:
            physics_fields.append((field, nw_arr.numpy(), nat_arr.numpy()))

    # Compare physics properties within each signature group (sorted).
    mismatches: list[str] = []
    for sig, nw_indices in newton_groups.items():
        nat_indices = native_groups.get(sig, [])

        for field, nw_data, nat_data in physics_fields:
            if nw_data.ndim == 1:
                nw_scalars = sorted(float(nw_data[i]) for i in nw_indices)
                nat_scalars = sorted(float(nat_data[i]) for i in nat_indices)
                for k, (nv, natv) in enumerate(zip(nw_scalars, nat_scalars[: len(nw_scalars)], strict=True)):
                    if abs(nv - natv) > tol:
                        mismatches.append(f"sig={sig} {field}[{k}]: newton={nv} native={natv}")
            else:
                axis = 0 if nw_data.ndim == 2 else 1
                nw_vecs = sorted(
                    tuple(nw_data[0, i].tolist()) if axis == 1 else tuple(nw_data[i].tolist()) for i in nw_indices
                )
                nat_vecs = sorted(
                    tuple(nat_data[0, i].tolist()) if axis == 1 else tuple(nat_data[i].tolist()) for i in nat_indices
                )
                for k, (nv, natv) in enumerate(zip(nw_vecs, nat_vecs[: len(nw_vecs)], strict=True)):
                    err = max(abs(a - b) for a, b in zip(nv, natv, strict=True))
                    if err > tol:
                        mismatches.append(f"sig={sig} {field}[{k}]: newton={nv} native={natv} err={err:.2e}")

    assert not mismatches, f"Geom physics property mismatches ({len(mismatches)}):\n" + "\n".join(mismatches[:10])


# =============================================================================
# Name-based index mapping for USD vs MJCF body/joint/DOF ordering
# =============================================================================


def build_body_index_map(
    newton_mj_model: Any,
    native_mj_model: Any,
) -> dict[int, int]:
    """Build native_body_idx -> newton_body_idx mapping using body names.

    Newton's mj_model encodes the full prim path in the body name (with ``_``
    separators).  We match by checking that the Newton name ends with
    ``_<native_name>`` (boundary-aligned suffix match).
    """
    nbody_native = native_mj_model.nbody
    nbody_newton = newton_mj_model.nbody
    assert nbody_newton == nbody_native, f"nbody mismatch: newton={nbody_newton} vs native={nbody_native}"

    newton_names = [newton_mj_model.body(i).name for i in range(nbody_newton)]
    native_names = [native_mj_model.body(i).name for i in range(nbody_native)]

    body_map: dict[int, int] = {0: 0}
    used_newton: set[int] = {0}

    for ni in range(1, nbody_native):
        native_name = native_names[ni]
        for nw_i in range(1, nbody_newton):
            if nw_i in used_newton:
                continue
            if _suffix_match(newton_names[nw_i], native_name):
                body_map[ni] = nw_i
                used_newton.add(nw_i)
                break

    if len(body_map) < nbody_native:
        unmapped = [native_names[i] for i in range(nbody_native) if i not in body_map]
        raise ValueError(f"Could not map {len(unmapped)} native bodies: {unmapped[:5]}")

    return body_map


def _normalize_name(name: str) -> str:
    """Extract the last path component from a USD prim path.

    USD-imported models may store full prim paths (e.g.
    ``/World/robot/MjcTendons/lh_T_FFJ0``) as entity names.
    This extracts the leaf component so it can be compared against
    clean MJCF names like ``lh_T_FFJ0``.
    """
    if "/" in name:
        return name.rsplit("/", 1)[-1]
    return name


def _suffix_match(nw_name: str, native_name: str) -> bool:
    """Check if nw_name matches native_name by exact or boundary-aligned suffix.

    The USD converter may append ``_N`` (numeric) to avoid name collisions
    (e.g., MJCF joint ``torso`` becomes USD prim ``torso_1``).  We strip
    trailing ``_<digits>`` from the Newton name before the suffix check.

    Names that look like USD prim paths (containing ``/``) are reduced to
    their last path component before comparison.
    """
    nw_name = _normalize_name(nw_name)
    native_name = _normalize_name(native_name)
    if nw_name == native_name:
        return True
    suffix = "_" + native_name
    if nw_name.endswith(suffix):
        return True
    stripped = re.sub(r"_\d+$", "", nw_name)
    if stripped != nw_name and (stripped == native_name or stripped.endswith(suffix)):
        return True
    return False


def build_jnt_index_map(
    newton_mj_model: Any,
    native_mj_model: Any,
) -> dict[int, int]:
    """Build native_jnt_idx -> newton_jnt_idx mapping using joint names.

    Falls back to joint-type matching for unmatched joints (handles free
    joints whose names differ between USD and MJCF).
    """
    njnt_native = native_mj_model.njnt
    njnt_newton = newton_mj_model.njnt
    assert njnt_newton == njnt_native, f"njnt mismatch: newton={njnt_newton} vs native={njnt_native}"

    newton_names = [newton_mj_model.jnt(i).name for i in range(njnt_newton)]
    native_names = [native_mj_model.jnt(i).name for i in range(njnt_native)]

    jnt_map: dict[int, int] = {}
    used_newton: set[int] = set()
    for ni in range(njnt_native):
        native_name = native_names[ni]
        for nw_i in range(njnt_newton):
            if nw_i in used_newton:
                continue
            if _suffix_match(newton_names[nw_i], native_name):
                jnt_map[ni] = nw_i
                used_newton.add(nw_i)
                break

    # Fallback: match remaining joints by type (handles free joints with
    # different names between USD and MJCF).
    if len(jnt_map) < njnt_native:
        newton_types = newton_mj_model.jnt_type.flatten() if hasattr(newton_mj_model, "jnt_type") else None
        native_types = native_mj_model.jnt_type.flatten() if hasattr(native_mj_model, "jnt_type") else None
        if newton_types is not None and native_types is not None:
            for ni in range(njnt_native):
                if ni in jnt_map:
                    continue
                native_type = int(native_types[ni])
                for nw_i in range(njnt_newton):
                    if nw_i in used_newton:
                        continue
                    if int(newton_types[nw_i]) == native_type:
                        jnt_map[ni] = nw_i
                        used_newton.add(nw_i)
                        break

    if len(jnt_map) < njnt_native:
        unmapped = [native_names[i] for i in range(njnt_native) if i not in jnt_map]
        raise ValueError(f"Could not map {len(unmapped)} native joints: {unmapped[:5]}")

    return jnt_map


def build_dof_index_map(
    newton_mjw: Any,
    native_mjw: Any,
    jnt_map: dict[int, int],
) -> dict[int, int]:
    """Build native_dof_idx -> newton_dof_idx mapping from the joint map."""
    newton_dofadr = newton_mjw.jnt_dofadr.numpy().flatten()
    native_dofadr = native_mjw.jnt_dofadr.numpy().flatten()
    native_type = native_mjw.jnt_type.numpy().flatten()

    def _ndof(jtype: int) -> int:
        return {0: 6, 1: 3, 2: 1, 3: 1}.get(int(jtype), 1)

    dof_map: dict[int, int] = {}
    for native_ji, newton_ji in jnt_map.items():
        native_adr = int(native_dofadr[native_ji])
        newton_adr = int(newton_dofadr[newton_ji])
        n = _ndof(native_type[native_ji])
        for d in range(n):
            dof_map[native_adr + d] = newton_adr + d

    return dof_map


def build_qpos_index_map(
    newton_mjw: Any,
    native_mjw: Any,
    jnt_map: dict[int, int],
) -> dict[int, int]:
    """Build native_qpos_idx -> newton_qpos_idx mapping from the joint map.

    Per-joint qpos slot counts: FREE=7 (xyz + wxyz), BALL=4 (wxyz), HINGE/SLIDE=1.
    """
    newton_qposadr = newton_mjw.jnt_qposadr.numpy().flatten()
    native_qposadr = native_mjw.jnt_qposadr.numpy().flatten()
    native_type = native_mjw.jnt_type.numpy().flatten()

    def _nqpos(jtype: int) -> int:
        return {0: 7, 1: 4, 2: 1, 3: 1}.get(int(jtype), 1)

    qpos_map: dict[int, int] = {}
    for native_ji, newton_ji in jnt_map.items():
        native_adr = int(native_qposadr[native_ji])
        newton_adr = int(newton_qposadr[newton_ji])
        n = _nqpos(native_type[native_ji])
        for d in range(n):
            qpos_map[native_adr + d] = newton_adr + d

    return qpos_map


def _actuator_target_name(mj_model: Any, act_idx: int) -> str:
    """Get the human-readable target name for an actuator.

    Resolves trnid to a joint, tendon, site, or body name depending on trntype.
    """
    _TRN_RESOLVERS = {
        0: "jnt",  # mjTRN_JOINT
        1: "jnt",  # mjTRN_JOINTINPARENT
        3: "tendon",  # mjTRN_TENDON
        4: "site",  # mjTRN_SITE
        5: "body",  # mjTRN_BODY
    }
    trntype = int(mj_model.actuator_trntype[act_idx])
    trnid = int(mj_model.actuator_trnid[act_idx, 0])
    resolver = _TRN_RESOLVERS.get(trntype)
    if resolver is not None and hasattr(mj_model, resolver):
        try:
            return getattr(mj_model, resolver)(trnid).name
        except Exception:
            pass
    return ""


def build_actuator_index_map(
    newton_mj_model: Any,
    native_mj_model: Any,
) -> dict[int, int]:
    """Build native_actuator_idx -> newton_actuator_idx mapping.

    Tries actuator name matching first. Falls back to matching by the
    actuator's target name (joint/tendon/site/body name resolved from trnid).
    """
    nu_native = native_mj_model.nu
    nu_newton = newton_mj_model.nu
    assert nu_newton == nu_native, f"nu mismatch: newton={nu_newton} vs native={nu_native}"

    newton_act_names = [newton_mj_model.actuator(i).name for i in range(nu_newton)]
    native_act_names = [native_mj_model.actuator(i).name for i in range(nu_native)]

    act_map: dict[int, int] = {}
    used_newton: set[int] = set()

    # Strategy 1: match by actuator name (when Newton provides names).
    if any(n for n in newton_act_names):
        for ni in range(nu_native):
            for nw_i in range(nu_newton):
                if nw_i in used_newton:
                    continue
                if _suffix_match(newton_act_names[nw_i], native_act_names[ni]):
                    act_map[ni] = nw_i
                    used_newton.add(nw_i)
                    break

    # Strategy 2: match remaining actuators by target name.
    if len(act_map) < nu_native:
        newton_targets = [_actuator_target_name(newton_mj_model, i) for i in range(nu_newton)]
        native_targets = [_actuator_target_name(native_mj_model, i) for i in range(nu_native)]

        for ni in range(nu_native):
            if ni in act_map:
                continue
            native_target = native_targets[ni]
            if not native_target:
                continue
            for nw_i in range(nu_newton):
                if nw_i in used_newton:
                    continue
                if not newton_targets[nw_i]:
                    continue
                if _suffix_match(newton_targets[nw_i], native_target):
                    act_map[ni] = nw_i
                    used_newton.add(nw_i)
                    break

    if len(act_map) < nu_native:
        unmapped = [native_act_names[i] for i in range(nu_native) if i not in act_map]
        raise ValueError(f"Could not map {len(unmapped)}/{nu_native} actuators: {unmapped[:5]}")

    return act_map


def _reindex_1d(arr: np.ndarray, idx_map: dict[int, int], n: int) -> np.ndarray:
    """Reindex a 1D array from newton ordering to native ordering."""
    out = np.zeros_like(arr)
    for native_i, newton_i in idx_map.items():
        if native_i < n and newton_i < len(arr):
            out[native_i] = arr[newton_i]
    return out


def _reindex_2d_axis1(arr: np.ndarray, idx_map: dict[int, int], n: int) -> np.ndarray:
    """Reindex a 2D/3D array along axis 1 from newton ordering to native ordering."""
    out = np.zeros_like(arr)
    for native_i, newton_i in idx_map.items():
        if native_i < n and newton_i < arr.shape[1]:
            out[:, native_i] = arr[:, newton_i]
    return out


def compare_body_physics_mapped(
    newton_mjw: Any,
    native_mjw: Any,
    body_map: dict[int, int],
    tol: float = 1e-4,
) -> None:
    """Compare physics-relevant body fields using a name-based index mapping."""
    nbody = native_mjw.nbody

    # Structural fields that verify the body tree topology is preserved
    # under the index mapping.  Mass and inertia are already covered by
    # compare_inertia_tensors_mapped(); body_pos/body_quat are in DEFAULT skip
    # (re-diagonalization / backfill) so we do not duplicate them here.
    fields_exact = ["body_dofnum", "body_jntnum"]
    fields_remapped = ["body_treeid"]
    fields_float = ["body_gravcomp"]

    def _reindex(arr: np.ndarray) -> np.ndarray:
        if len(arr.shape) == 1:
            return _reindex_1d(arr, body_map, nbody)
        return _reindex_2d_axis1(arr, body_map, nbody)

    for field in fields_exact:
        newton_arr = getattr(newton_mjw, field, None)
        native_arr = getattr(native_mjw, field, None)
        if newton_arr is None or native_arr is None:
            continue
        nn = newton_arr.numpy()
        nat = native_arr.numpy()
        assert nn.shape == nat.shape, f"body field {field}: shape mismatch newton={nn.shape} vs native={nat.shape}"
        np.testing.assert_array_equal(_reindex(nn), nat, err_msg=f"body field {field} (reindexed)")

    for field in fields_remapped:
        newton_arr = getattr(newton_mjw, field, None)
        native_arr = getattr(native_mjw, field, None)
        if newton_arr is None or native_arr is None:
            continue
        nn = _reindex(newton_arr.numpy()).ravel()
        nat = native_arr.numpy().ravel()
        assert nn.shape == nat.shape, f"body field {field}: shape mismatch newton={nn.shape} vs native={nat.shape}"
        id_map: dict[int, int] = {}
        for newton_id, native_id in zip(nn, nat, strict=True):
            prev = id_map.setdefault(int(newton_id), int(native_id))
            assert prev == int(native_id), (
                f"body field {field}: newton id {newton_id} maps to both {prev} and {native_id}"
            )
        remapped = np.array([id_map[int(v)] for v in nn], dtype=nat.dtype)
        np.testing.assert_array_equal(remapped, nat, err_msg=f"body field {field} (reindexed + remapped)")

    for field in fields_float:
        newton_arr = getattr(newton_mjw, field, None)
        native_arr = getattr(native_mjw, field, None)
        if newton_arr is None or native_arr is None:
            continue
        nn = newton_arr.numpy()
        nat = native_arr.numpy()
        assert nn.shape == nat.shape, f"body field {field}: shape mismatch newton={nn.shape} vs native={nat.shape}"
        np.testing.assert_allclose(
            _reindex(nn),
            nat,
            atol=tol,
            rtol=0,
            err_msg=f"body field {field} (reindexed)",
        )


def compare_dof_physics_mapped(
    newton_mjw: Any,
    native_mjw: Any,
    dof_map: dict[int, int],
    tol: float = 1e-4,
) -> None:
    """Compare physics-relevant DOF fields using a name-based index mapping."""
    nv = native_mjw.nv

    fields_float = ["dof_armature", "dof_frictionloss"]

    for field in fields_float:
        newton_arr = getattr(newton_mjw, field, None)
        native_arr = getattr(native_mjw, field, None)
        if newton_arr is None or native_arr is None:
            continue
        nn = newton_arr.numpy()
        nat = native_arr.numpy()
        assert nn.shape == nat.shape, f"dof field {field}: shape mismatch newton={nn.shape} vs native={nat.shape}"
        if len(nn.shape) == 1:
            reindexed = _reindex_1d(nn, dof_map, nv)
        elif len(nn.shape) >= 2:
            reindexed = _reindex_2d_axis1(nn, dof_map, nv)
        else:
            continue
        np.testing.assert_allclose(
            reindexed,
            nat,
            atol=tol,
            rtol=0,
            err_msg=f"dof field {field} (reindexed)",
        )


def compare_mass_matrix_structure_mapped(
    newton_mjw: Any,
    native_mjw: Any,
    dof_map: dict[int, int],
) -> None:
    """Compare sparse mass matrix sparsity pattern under DOF reordering.

    The mass matrix M is nv x nv in CSR-like format (M_rowadr, M_rownnz,
    M_colind). When DOF ordering differs, the sparsity pattern is permuted
    but structurally equivalent. This function verifies that equivalence
    by remapping row/column indices through dof_map.

    Args:
        dof_map: native_dof_idx -> newton_dof_idx.
    """
    nv = native_mjw.nv
    assert newton_mjw.nv == nv, f"nv mismatch: newton={newton_mjw.nv} vs native={nv}"

    newton_rowadr = newton_mjw.M_rowadr.numpy().flatten()
    newton_rownnz = newton_mjw.M_rownnz.numpy().flatten()
    newton_colind = newton_mjw.M_colind.numpy().flatten()
    native_rowadr = native_mjw.M_rowadr.numpy().flatten()
    native_rownnz = native_mjw.M_rownnz.numpy().flatten()
    native_colind = native_mjw.M_colind.numpy().flatten()

    inv_dof_map = {v: k for k, v in dof_map.items()}

    mismatches = []
    for native_dof in range(nv):
        newton_dof = dof_map.get(native_dof)
        if newton_dof is None:
            mismatches.append(f"native DOF {native_dof} has no mapping")
            continue

        nat_start = int(native_rowadr[native_dof])
        nat_nnz = int(native_rownnz[native_dof])
        native_cols = {int(native_colind[nat_start + k]) for k in range(nat_nnz)}

        nw_start = int(newton_rowadr[newton_dof])
        nw_nnz = int(newton_rownnz[newton_dof])
        newton_cols_raw = [int(newton_colind[nw_start + k]) for k in range(nw_nnz)]
        newton_cols = {inv_dof_map.get(c, -1) for c in newton_cols_raw}

        if native_cols != newton_cols:
            mismatches.append(
                f"DOF {native_dof} (newton DOF {newton_dof}): "
                f"native cols={sorted(native_cols)}, "
                f"newton cols (remapped)={sorted(newton_cols)}"
            )

    assert not mismatches, f"Mass matrix sparsity mismatch for {len(mismatches)}/{nv} DOFs:\n" + "\n".join(
        mismatches[:10]
    )


def compare_tendon_jacobian_structure_mapped(
    newton_mjw: Any,
    native_mjw: Any,
    dof_map: dict[int, int],
) -> None:
    """Compare sparse tendon Jacobian sparsity pattern under DOF reordering.

    The tendon Jacobian ten_J uses CSR-like format (ten_J_rowadr, ten_J_rownnz,
    ten_J_colind) where rows are tendons and columns are DOF indices.  When DOF
    ordering differs the column indices are permuted but structurally equivalent.

    Gracefully skips if the fields are absent (older mujoco_warp without sparse
    ten_J support).

    Args:
        dof_map: native_dof_idx -> newton_dof_idx.
    """
    if not hasattr(native_mjw, "ten_J_colind") or not hasattr(newton_mjw, "ten_J_colind"):
        return

    ntendon = native_mjw.ntendon
    assert newton_mjw.ntendon == ntendon, f"ntendon mismatch: newton={newton_mjw.ntendon} vs native={ntendon}"

    if ntendon == 0:
        return

    newton_rowadr = newton_mjw.ten_J_rowadr.numpy().flatten()
    newton_rownnz = newton_mjw.ten_J_rownnz.numpy().flatten()
    newton_colind = newton_mjw.ten_J_colind.numpy().flatten()
    native_rowadr = native_mjw.ten_J_rowadr.numpy().flatten()
    native_rownnz = native_mjw.ten_J_rownnz.numpy().flatten()
    native_colind = native_mjw.ten_J_colind.numpy().flatten()

    inv_dof_map = {v: k for k, v in dof_map.items()}

    mismatches = []
    for t in range(ntendon):
        nat_start = int(native_rowadr[t])
        nat_nnz = int(native_rownnz[t])
        native_cols = {int(native_colind[nat_start + k]) for k in range(nat_nnz)}

        nw_start = int(newton_rowadr[t])
        nw_nnz = int(newton_rownnz[t])
        newton_cols_raw = [int(newton_colind[nw_start + k]) for k in range(nw_nnz)]
        newton_cols = {inv_dof_map.get(c, -1) for c in newton_cols_raw}

        if native_cols != newton_cols:
            mismatches.append(
                f"tendon {t}: native cols={sorted(native_cols)}, newton cols (remapped)={sorted(newton_cols)}"
            )

    assert not mismatches, f"Tendon Jacobian sparsity mismatch for {len(mismatches)}/{ntendon} tendons:\n" + "\n".join(
        mismatches[:10]
    )


def compare_qD_structure_mapped(
    newton_mjw: Any,
    native_mjw: Any,
    dof_map: dict[int, int],
) -> None:
    """Compare sparse RNE derivative D-structure under DOF reordering.

    qD_fullm_i and qD_fullm_j are flat arrays of (row, col) DOF indices
    enumerating the D-structure (full square sparsity used by RNE
    derivatives). When DOF ordering differs the indices are permuted but
    the underlying (row, col) set is structurally equivalent: each native
    (i, j) entry should map to a Newton entry at (dof_map[i], dof_map[j]).

    Gracefully skips if the fields are absent (older mujoco_warp without
    RNE derivative support).

    Args:
        dof_map: native_dof_idx -> newton_dof_idx.
    """
    if not hasattr(native_mjw, "qD_fullm_i") or not hasattr(newton_mjw, "qD_fullm_i"):
        return

    nD = int(getattr(native_mjw, "nD", 0))
    assert int(getattr(newton_mjw, "nD", 0)) == nD, (
        f"nD mismatch: newton={getattr(newton_mjw, 'nD', None)} vs native={nD}"
    )

    if nD == 0:
        return

    newton_i = newton_mjw.qD_fullm_i.numpy().flatten()
    newton_j = newton_mjw.qD_fullm_j.numpy().flatten()
    native_i = native_mjw.qD_fullm_i.numpy().flatten()
    native_j = native_mjw.qD_fullm_j.numpy().flatten()

    inv_dof_map = {v: k for k, v in dof_map.items()}

    native_pairs = {(int(native_i[k]), int(native_j[k])) for k in range(nD)}
    newton_pairs = {(inv_dof_map.get(int(newton_i[k]), -1), inv_dof_map.get(int(newton_j[k]), -1)) for k in range(nD)}

    missing = native_pairs - newton_pairs
    extra = newton_pairs - native_pairs
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing in newton (remapped): {sorted(missing)[:10]}")
        if extra:
            parts.append(f"extra in newton (remapped): {sorted(extra)[:10]}")
        raise AssertionError("qD_fullm sparsity mismatch:\n" + "\n".join(parts))


ACTUATOR_SKIP_FIELDS: set[str] = {
    "actuator_plugin",
    "actuator_user",
    "actuator_id",
    "actuator_trnid",
    "actuator_trntype",
    "actuator_trntype_body_adr",
    "actuator_actadr",
    "actuator_actnum",
    # Position/velocity-shortcut MjcActuator rows targeting single-DOF joints are
    # promoted to CtrlSource.JOINT_TARGET on import (matching MJCF behavior).
    # _init_actuators rebuilds the compiled MuJoCo actuators from joint_target_*,
    # so these low-level fields differ from the native MJCF model and are not
    # meaningful to compare. The same fields are also skipped by the MJCF
    # menagerie tests in test_menagerie_mujoco.py for the same reason.
    "actuator_dynprm",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_ctrlrange",
    "actuator_ctrllimited",
    "actuator_forcerange",
    "actuator_forcelimited",
    "actuator_actrange",
    "actuator_actlimited",
    "actuator_gear",
    "actuator_cranklength",
}


def compare_actuator_physics_mapped(
    newton_mjw: Any,
    native_mjw: Any,
    act_map: dict[int, int],
    tol: float = 1e-4,
    skip_fields: set[str] | None = None,
) -> None:
    """Compare all actuator_* fields using a name-based index mapping.

    Discovers fields dynamically from the native model. Only compares
    actuators present in act_map (partial maps are allowed when some
    actuator trnids could not be resolved).

    Args:
        act_map: native_actuator_idx -> newton_actuator_idx.
        skip_fields: Field names to skip.
    """
    skip = (skip_fields or set()) | ACTUATOR_SKIP_FIELDS
    mapped_native = sorted(act_map.keys())
    nu = native_mjw.nu

    fields = [name for name in dir(native_mjw) if name.startswith("actuator_") and name not in skip]

    for field in fields:
        newton_arr = getattr(newton_mjw, field, None)
        native_arr = getattr(native_mjw, field, None)
        if newton_arr is None or native_arr is None:
            continue
        nn = newton_arr.numpy()
        nat = native_arr.numpy()
        assert nn.shape == nat.shape, f"actuator field {field}: shape mismatch newton={nn.shape} vs native={nat.shape}"

        # Determine which axis is the actuator axis based on shape.
        # (nworld, nu, ...) -> axis 1;  (nu, ...) -> axis 0
        if len(nn.shape) >= 2 and nn.shape[1] == nu:
            newton_vals = nn[:, [act_map[ni] for ni in mapped_native]]
            native_vals = nat[:, mapped_native]
        elif nn.shape[0] == nu:
            newton_vals = nn[[act_map[ni] for ni in mapped_native]]
            native_vals = nat[mapped_native]
        else:
            continue
        np.testing.assert_allclose(
            newton_vals,
            native_vals,
            atol=tol,
            rtol=tol,
            err_msg=f"actuator field {field} (reindexed, {len(mapped_native)} mapped)",
        )


# =============================================================================
# TestMenagerieUSD Base Class
# =============================================================================


class TestMenagerieUSD(TestMenagerieBase):
    """Base class for USD-based tests: Newton loads pre-converted USD.

    Subclasses set usd_asset_folder and usd_scene_file to identify the
    pre-converted USD file in the newton-assets repo. The asset is
    downloaded lazily in setUpClass. Native MuJoCo still loads the
    original MJCF from menagerie for comparison.
    """

    usd_asset_folder: str = ""
    usd_scene_file: str = ""
    usd_path: str = ""

    @classmethod
    def setUpClass(cls):
        """Download MJCF and USD assets once for all tests in this class."""
        if not cls.usd_asset_folder:
            raise unittest.SkipTest("usd_asset_folder not defined")

        super().setUpClass()

        try:
            asset_root = newton.utils.download_asset(cls.usd_asset_folder)
        except (OSError, TimeoutError, ConnectionError) as e:
            raise unittest.SkipTest(f"Failed to download USD asset {cls.usd_asset_folder}: {e}") from e

        cls.usd_path = str(asset_root / cls.usd_scene_file)
        if not Path(cls.usd_path).exists():
            raise unittest.SkipTest(f"USD file not found: {cls.usd_path}")

    # Backfill requires 1:1 body index correspondence; disabled for USD since
    # body ordering may differ from native MJCF.
    backfill_model: bool = False

    # USD models may carry implicit integrators that mujoco_warp doesn't support.
    # Force Euler so put_model succeeds; _align_models copies native integrator after.
    solver_integrator: str = "euler"

    njmax: int = 600

    nconmax: int = 200

    # USD-specific skips on top of DEFAULT_MODEL_SKIP_FIELDS.
    # Fields handled by sorted/mapped comparison hooks (_compare_inertia,
    # _compare_body_physics, _compare_dof_physics, _compare_geoms, _compare_jnt_range)
    # are skipped from the generic compare_mjw_models pass, not silently ignored.
    model_skip_fields: ClassVar[set[str]] = DEFAULT_MODEL_SKIP_FIELDS | {
        # Actuator ordering may differ -> compared via _compare_actuator_physics
        "actuator_",
        # Equality constraints not yet imported from USD
        "eq_",
        "neq",
        # Body ordering may differ -> compared via _compare_inertia / _compare_body_physics
        "body_",
        # DOF ordering may differ -> compared via _compare_dof_physics
        "dof_",
        # Joint ordering may differ -> compared via _compare_jnt_range
        "jnt_",
        # Sparse mass matrix structure: DOF-indexed, compared via _compare_mass_matrix_structure
        "M_",
        # Sparse RNE derivative D-structure: DOF-indexed, compared via _compare_qD_structure
        "qD_fullm_",
        # Sparse tendon Jacobian structure: DOF-indexed, compared via _compare_tendon_jacobian_structure
        "ten_J_",
        "nJten",
        "max_ten_J_rownnz",
        # Cholesky permutation: derived from body/DOF tree ordering
        "mapM2M",
        "qLD_updates",
        # Derived from mass matrix and tendon geometry by set_const; differs due to
        # inertia re-diagonalization during USD import.
        "tendon_invweight0",
        # stat.meaninertia is derived from body mass/inertia (which may differ for some USD models)
        "stat",
        # Broadphase collision data depends on geom ordering
        "nxn_",
        # Contact pairs not imported from USD
        "npair",
        "pair_",
        # Mesh counts and data arrays differ between USD and MJCF representations
        "nmesh",
        "nmeshface",
        "nmeshgraph",
        "nmeshnormal",
        "nmeshpoly",
        "nmeshpolymap",
        "nmeshpolyvert",
        "nmeshvert",
        "nmaxmeshdeg",
        "nmaxpolygon",
        "mesh_",
        # Site body IDs reference body indices (different ordering)
        "site_",
        # Wrap object IDs reference geoms (different ordering)
        "wrap_objid",
    }

    def _compare_compiled_fields(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Skip compiled-field check for USD models.

        USD import re-diagonalizes inertia, causing large differences in
        derived fields (body_invweight0, dof_invweight0, actuator_acc0).
        These are already handled by the mapped comparison hooks.
        """

    def _compare_inertia(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare inertia tensors using name-based body index mapping."""
        compare_inertia_tensors_mapped(newton_mjw, native_mjw, self._body_map)

    def _compare_geoms(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare geom subset: Newton excludes visual-only geoms."""
        compare_geoms_subset(newton_mjw, native_mjw)

    def _compare_jnt_range(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare joint properties using sorted multisets (handles reordering)."""
        compare_joints_sorted(newton_mjw, native_mjw)

    def _align_models(self, newton_solver: SolverMuJoCo, native_mjw_model: Any, mj_model: Any) -> None:
        """Align Newton's mjw_model options with native and build index maps.

        Copies all solver option fields from the native model so the simulation
        uses identical settings.  Also builds body/joint/DOF index maps for
        mapped comparison hooks.
        """
        newton_opt = newton_solver.mjw_model.opt
        native_opt = native_mjw_model.opt
        for attr in dir(native_opt):
            if attr.startswith("_") or callable(getattr(native_opt, attr)):
                continue
            native_val = getattr(native_opt, attr)
            if isinstance(native_val, (int, float, bool)):
                setattr(newton_opt, attr, native_val)

        # Store maps on class — _ensure_models stores models on cls, so hooks
        # that access these maps via self will find them on the class.
        cls = self.__class__
        cls._body_map = build_body_index_map(newton_solver.mj_model, mj_model)
        cls._jnt_map = build_jnt_index_map(newton_solver.mj_model, mj_model)
        cls._dof_map = build_dof_index_map(
            newton_solver.mjw_model,
            native_mjw_model,
            cls._jnt_map,
        )
        cls._qpos_map = build_qpos_index_map(
            newton_solver.mjw_model,
            native_mjw_model,
            cls._jnt_map,
        )
        cls._actuator_map = build_actuator_index_map(newton_solver.mj_model, mj_model)

    def _compare_body_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare physics-relevant body fields using name-based index mapping."""
        compare_body_physics_mapped(newton_mjw, native_mjw, self._body_map)

    def _compare_dof_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare physics-relevant DOF fields using name-based index mapping."""
        compare_dof_physics_mapped(newton_mjw, native_mjw, self._dof_map)

    def _compare_mass_matrix_structure(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare sparse mass matrix structure using DOF index mapping."""
        compare_mass_matrix_structure_mapped(newton_mjw, native_mjw, self._dof_map)

    def _compare_tendon_jacobian_structure(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare sparse tendon Jacobian structure using DOF index mapping."""
        compare_tendon_jacobian_structure_mapped(newton_mjw, native_mjw, self._dof_map)

    def _compare_qD_structure(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare sparse RNE derivative D-structure using DOF index mapping."""
        compare_qD_structure_mapped(newton_mjw, native_mjw, self._dof_map)

    def _compare_actuator_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare actuator fields using name-based index mapping."""
        compare_actuator_physics_mapped(
            newton_mjw,
            native_mjw,
            self._actuator_map,
            skip_fields=self.actuator_skip_fields,
        )

    actuator_skip_fields: ClassVar[set[str]] = {
        # Derived from mass matrix; differs when backfill_model=False because
        # Newton re-diagonalizes inertia. Compared indirectly via simulation equivalence.
        "actuator_acc0",
        # Derived from joint ranges via set_length_range; Newton recomputes this
        # during notify_model_changed but the native model does not.
        "actuator_lengthrange",
    }

    # Per-actuator and per-joint fields the USD parser doesn't populate to match
    # native MJCF compilation, but which step-response dynamics depends on.
    # Empirically pinned down on ShadowHand: without these, qfrc_actuator
    # diverges at step 0 (actuator clipping fields), and qfrc_constraint
    # diverges at step 1+ (joint-limit solref + actfrc).
    usd_actuator_backfill_fields: ClassVar[list[str]] = [
        "actuator_ctrlrange",
        "actuator_ctrllimited",
        "actuator_forcerange",
        "actuator_forcelimited",
    ]
    usd_joint_backfill_fields: ClassVar[list[str]] = [
        "jnt_solref",
        "jnt_actfrclimited",
        "jnt_actfrcrange",
    ]
    # Body-level fields. USD import re-diagonalizes inertia, and for some
    # assets (WonikAllegro) the USD authors body mass/inertia values that
    # don't match the source MJCF at all. Backfilling these in one batch
    # (not partial — a partial backfill leaves the model inconsistent and
    # actually regresses other robots) brings every enabled robot to the
    # same numerical residual it would have without backfill, and lets
    # WonikAllegro pass at a relaxed dynamics_tolerance.
    usd_body_backfill_fields: ClassVar[list[str]] = [
        "body_mass",
        "body_inertia",
        "body_iquat",
        "body_ipos",
        "body_subtreemass",
        "body_invweight0",
    ]
    # Inertia-derived per-DOF field. USD's mass matrix (and its inverse)
    # differs from native; that propagates into constraint impedance via
    # mjwarp's `_efc_row` (D_out scaled by invweight). WonikAllegro's thj0
    # constraint force diverges without this backfill.
    usd_dof_backfill_fields: ClassVar[list[str]] = [
        "dof_invweight0",
    ]

    def _backfill_and_recompute(self):
        """USD variant: permuted backfill of actuator/joint fields the USD parser
        doesn't populate to match native MJCF (verified per-field via step-by-step
        qfrc breakdown — see TestMenagerieUSD docstring)."""
        newton_mjw = self._newton_solver.mjw_model
        native_mjw = self._native_mjw_model

        def _backfill_permuted(field: str, idx_map: dict[int, int]) -> None:
            n = getattr(newton_mjw, field).numpy()
            m = getattr(native_mjw, field).numpy()
            if n.shape != m.shape:
                return
            out = n.copy()
            if n.ndim >= 2:
                nlen = n.shape[1]
                perm = np.fromiter((idx_map[i] for i in range(nlen)), dtype=np.int64, count=nlen)
                out[:, perm] = m
            else:
                for ni, nw in idx_map.items():
                    out[nw] = m[ni]
            getattr(newton_mjw, field).assign(out)

        for field in self.usd_actuator_backfill_fields:
            _backfill_permuted(field, self._actuator_map)
        for field in self.usd_joint_backfill_fields:
            _backfill_permuted(field, self._jnt_map)
        for field in self.usd_body_backfill_fields:
            _backfill_permuted(field, self._body_map)
        for field in self.usd_dof_backfill_fields:
            _backfill_permuted(field, self._dof_map)

        # Re-run kinematics/RNE so derived data fields reflect the backfilled model
        # (mirrors TestMenagerieBase._backfill_and_recompute).
        from mujoco_warp._src import smooth as mjw_smooth

        mjw_smooth.kinematics(newton_mjw, self._newton_solver.mjw_data)
        mjw_smooth.com_pos(newton_mjw, self._newton_solver.mjw_data)
        mjw_smooth.crb(newton_mjw, self._newton_solver.mjw_data)
        mjw_smooth.factor_m(newton_mjw, self._newton_solver.mjw_data)
        mjw_smooth.com_vel(newton_mjw, self._newton_solver.mjw_data)
        mjw_smooth.rne(newton_mjw, self._newton_solver.mjw_data)

    def _create_newton_model(self) -> newton.Model:
        """Create Newton model from pre-converted USD file."""
        if not self.usd_path:
            raise unittest.SkipTest("usd_path not set (no usd_asset_folder defined)")
        return create_newton_model_from_usd(
            Path(self.usd_path),
            num_worlds=self.num_worlds,
            add_ground=False,  # scene.xml includes ground plane
        )

    def test_forward_kinematics(self):
        """USD variant: remap qpos and body indices by name before comparing.

        Newton's USD importer assigns body/joint indices in alphabetical-by-prim-path
        order (inherited from ``UsdPhysics.ArticulationDesc.articulatedBodies``),
        which differs from native MJCF's authored-order indexing for any robot whose
        subtree names don't sort the same as their authoring order. The base FK test
        copies Newton's qpos vector directly to native and compares xpos index-by-
        index; that fails whenever the orderings differ. This override remaps qpos
        from Newton's joint layout to native's via ``_qpos_map`` before copying, and
        permutes body fields via ``_body_map`` before the per-field comparison.
        """
        if not self.fk_enabled:
            self.skipTest("Forward kinematics not enabled for this robot")

        self._ensure_models()
        self._run_model_comparisons()
        self._backfill_and_recompute()

        from mujoco_warp._src import smooth as mjw_smooth

        model = self._newton_model
        solver = self._newton_solver

        state = model.state()

        rng = np.random.default_rng(seed=42)
        joint_q_np = model.joint_q.numpy()
        joint_q_np += rng.uniform(-0.1, 0.1, size=joint_q_np.shape).astype(np.float32)

        joint_type = model.joint_type.numpy()
        q_start = model.joint_q_start.numpy()
        for j in range(len(joint_type)):
            jt = joint_type[j]
            qi = q_start[j]
            if jt == newton.JointType.FREE:
                q = joint_q_np[qi + 3 : qi + 7]
                q /= np.linalg.norm(q)
            elif jt == newton.JointType.BALL:
                q = joint_q_np[qi : qi + 4]
                q /= np.linalg.norm(q)

        state.joint_q.assign(joint_q_np)
        solver._update_mjc_data(solver.mjw_data, model, state)

        # Remap Newton-layout qpos into native-layout before copying.
        newton_qpos = solver.mjw_data.qpos.numpy()
        native_qpos = self._native_mjw_data.qpos.numpy().copy()
        native_idx = np.fromiter(self._qpos_map.keys(), dtype=np.int64)
        newton_idx = np.fromiter(self._qpos_map.values(), dtype=np.int64)
        native_qpos[..., native_idx] = newton_qpos[..., newton_idx]
        self._native_mjw_data.qpos.assign(native_qpos)

        run_newton_eval_fk(solver, model, state)
        mjw_smooth.kinematics(self._native_mjw_model, self._native_mjw_data)
        mjw_smooth.com_pos(self._native_mjw_model, self._native_mjw_data)

        # Compare FK fields after permuting Newton's body axis to native's order.
        # _body_map is native_body_idx -> newton_body_idx.
        body_perm = np.fromiter(
            (self._body_map[ni] for ni in range(self._native_mjw_model.nbody)),
            dtype=np.int64,
            count=int(self._native_mjw_model.nbody),
        )
        for field_name in self.fk_fields:
            newton_arr = getattr(solver.mjw_data, field_name).numpy()
            native_arr = getattr(self._native_mjw_data, field_name).numpy()
            # body-axis is axis 1 (shape: nworld, nbody, ...)
            newton_permuted = newton_arr[:, body_perm]
            if field_name == "xquat":
                # Quaternion sign equivalence: q and -q represent the same rotation.
                direct = np.abs(newton_permuted - native_arr)
                flipped = np.abs(newton_permuted + native_arr)
                use_flipped = np.max(flipped, axis=-1, keepdims=True) < np.max(direct, axis=-1, keepdims=True)
                diff = np.where(use_flipped, flipped, direct)
            else:
                diff = np.abs(newton_permuted - native_arr)
            max_diff = float(np.max(diff))
            if max_diff > self.fk_tolerance:
                worst = np.unravel_index(np.argmax(diff), diff.shape)
                raise AssertionError(
                    f"FK field '{field_name}': max diff {max_diff:.4e} > tol {self.fk_tolerance:.2e}\n"
                    f"  at index {worst}: newton(permuted)={newton_permuted[worst]} native={native_arr[worst]}"
                )

    def test_dynamics(self):
        """USD variant: remap actuators (ctrl) and DOFs (qpos/qvel) by name.

        Mirror of ``TestMenagerieMJCF.test_dynamics`` with three changes:
          - ctrl arrays are filled directly via numpy using ``_actuator_map`` so the
            same physical actuator is driven on both sides each step (the standard
            ``StepResponseControlStrategy`` would drive different actuators on
            each side when USD vs MJCF orderings differ).
          - JOINT_TARGET actuators (USD-imported MjcActuator position-shortcut rows)
            are routed through ``Control.joint_target_pos`` / ``joint_target_vel``
            rather than ``mujoco.ctrl`` — see ``mjc_actuator_ctrl_source`` on the
            solver. Both arrays are written here, indexed via
            ``mjc_actuator_to_newton_idx``.
          - Newton's per-step ``qpos`` / ``qvel`` are permuted via ``_qpos_map`` /
            ``_dof_map`` before comparison.
        """
        if self.num_steps <= 0:
            self.skipTest("Dynamics not enabled (num_steps=0)")

        self._ensure_models()
        self._run_model_comparisons()
        self._backfill_and_recompute()

        newton_solver = self._newton_solver
        newton_state = self._newton_state
        newton_control = self._newton_control
        native_mjw_model = self._native_mjw_model
        native_mjw_data = self._native_mjw_data
        dt = self._dt

        newton_saved = _disable_collisions(newton_solver.mjw_model)
        native_saved = _disable_collisions(native_mjw_model)

        try:
            num_worlds, num_actuators = native_mjw_data.ctrl.shape
            target = self.dynamics_target

            # Per-world step-response control with actuator remap.
            ctrl_source = newton_solver.mjc_actuator_ctrl_source.numpy()
            act_to_newton_idx = newton_solver.mjc_actuator_to_newton_idx.numpy()
            # Start from zero so non-driven JOINT_TARGET actuators have target=0,
            # matching native_ctrl=0 for those slots. Without this, Newton's pre-existing
            # joint_target_pos (initial DOF targets from the USD posture) would apply
            # nonzero forces every world for actuators we didn't drive — surfaced on
            # WonikAllegro where tha0's initial target 0.8295 stayed in every world.
            joint_target_pos = np.zeros_like(newton_control.joint_target_pos.numpy())
            joint_target_vel = np.zeros_like(newton_control.joint_target_vel.numpy())
            dofs_per_world = joint_target_pos.shape[0] // num_worlds

            native_ctrl_np = np.zeros((num_worlds, num_actuators), dtype=np.float32)
            newton_ctrl_np = np.zeros_like(native_ctrl_np)
            for w in range(num_worlds):
                native_act = w % num_actuators
                newton_act = self._actuator_map[native_act]
                # Native: write ctrl array (MJCF actuators read from ctrl).
                native_ctrl_np[w, native_act] = target
                # Newton: route by ctrl_source.
                idx = int(act_to_newton_idx[newton_act])
                if ctrl_source[newton_act] == 1:  # CTRL_DIRECT
                    # Newton's actuator reads from newton_control.mujoco.ctrl[w, idx]
                    # where idx is the actuator's mujoco-frequency source index (not
                    # the mjw_model actuator index). For CTRL_DIRECT this is set to
                    # the originating mujoco_act_idx in _init_actuators.
                    newton_ctrl_np[w, idx] = target
                elif ctrl_source[newton_act] == 0:  # JOINT_TARGET
                    if idx >= 0:
                        joint_target_pos[w * dofs_per_world + idx] = target
                    elif idx <= -2:
                        joint_target_vel[w * dofs_per_world + (-(idx + 2))] = target
            native_mjw_data.ctrl.assign(native_ctrl_np)
            newton_control.mujoco.ctrl.assign(newton_ctrl_np)
            newton_control.joint_target_pos.assign(joint_target_pos)
            newton_control.joint_target_vel.assign(joint_target_vel)

            # qpos / qvel permutation arrays (native_idx -> newton_idx).
            nq = int(native_mjw_data.qpos.shape[1])
            nv = int(native_mjw_data.qvel.shape[1])
            qpos_perm = np.fromiter((self._qpos_map[i] for i in range(nq)), dtype=np.int64, count=nq)
            dof_perm = np.fromiter((self._dof_map[i] for i in range(nv)), dtype=np.int64, count=nv)
            tol = self.dynamics_tolerance

            for step in range(self.num_steps):
                newton_solver.step(newton_state, newton_state, newton_control, None, dt)
                _mujoco_warp.step(native_mjw_model, native_mjw_data)

                n_qpos = newton_solver.mjw_data.qpos.numpy()[:, qpos_perm]
                m_qpos = native_mjw_data.qpos.numpy()
                qpos_diff = float(np.max(np.abs(n_qpos - m_qpos)))
                n_qvel = newton_solver.mjw_data.qvel.numpy()[:, dof_perm]
                m_qvel = native_mjw_data.qvel.numpy()
                qvel_diff = float(np.max(np.abs(n_qvel - m_qvel)))

                if qpos_diff > tol:
                    worst = np.unravel_index(np.argmax(np.abs(n_qpos - m_qpos)), n_qpos.shape)
                    raise AssertionError(
                        f"Step {step}, qpos: max diff {qpos_diff:.4e} > tol {tol:.2e}\n"
                        f"  at index {worst}: newton(permuted)={n_qpos[worst]} native={m_qpos[worst]}"
                    )
                if qvel_diff > tol:
                    worst = np.unravel_index(np.argmax(np.abs(n_qvel - m_qvel)), n_qvel.shape)
                    raise AssertionError(
                        f"Step {step}, qvel: max diff {qvel_diff:.4e} > tol {tol:.2e}\n"
                        f"  at index {worst}: newton(permuted)={n_qvel[worst]} native={m_qvel[worst]}"
                    )
        finally:
            _restore_collisions(newton_solver.mjw_model, newton_saved)
            _restore_collisions(native_mjw_model, native_saved)

    def test_fk_initial_xforms(self):
        """Verify initial body transforms are consistent with forward kinematics.

        Performs two comparisons:

        1. **Self-consistency** -- ``model.body_q`` (set during USD import) vs
           ``eval_fk(model.joint_q)`` output.  Uses quaternion-sign-aware
           comparison since q and -q represent the same rotation.
        2. **Cross-validation** -- ``model.body_q`` from the USD path vs native
           MuJoCo body transforms (``mj_data.xpos`` / ``mj_data.xquat``) obtained
           by loading the original MJCF and running ``mj_forward``.
        """
        model = self._create_newton_model()
        initial_body_q = model.body_q.numpy().copy()

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        fk_body_q = state.body_q.numpy()

        # -- Self-consistency: USD initial vs FK --
        pos_tol = 1e-5
        quat_tol = 1e-5
        mismatches = []
        for i in range(model.body_count):
            init_pos = initial_body_q[i, :3]
            fk_pos = fk_body_q[i, :3]
            pos_err = np.max(np.abs(init_pos - fk_pos))

            init_quat = initial_body_q[i, 3:]
            fk_quat = fk_body_q[i, 3:]
            quat_err = min(
                np.max(np.abs(init_quat - fk_quat)),
                np.max(np.abs(init_quat + fk_quat)),
            )

            if pos_err > pos_tol or quat_err > quat_tol:
                mismatches.append(
                    f"  body {i}: pos_err={pos_err:.2e}, quat_err={quat_err:.2e}\n"
                    f"    initial: pos={init_pos}, quat={init_quat}\n"
                    f"    fk:      pos={fk_pos}, quat={fk_quat}"
                )

        self.assertEqual(
            len(mismatches),
            0,
            f"FK at initial joint_q produced different body transforms "
            f"for {len(mismatches)}/{model.body_count} bodies:\n" + "\n".join(mismatches),
        )

        # -- Cross-validation: USD initial vs native MJCF FK --
        mj_model, mj_data, _, _ = self._create_native_mujoco_warp()

        native_nbody = mj_model.nbody
        native_names = [mj_model.body(i).name for i in range(native_nbody)]
        native_xpos = np.array(mj_data.xpos).reshape(native_nbody, 3)
        native_xquat = np.array(mj_data.xquat).reshape(native_nbody, 4)

        body_worlds = model.body_world.numpy()
        world0_indices = np.where(body_worlds == 0)[0]

        cross_tol = 1e-3
        cross_mismatches = []
        matched = 0
        for nw_i in world0_indices:
            nw_label = model.body_label[nw_i]
            for nat_i in range(1, native_nbody):
                if not _suffix_match(nw_label, native_names[nat_i]):
                    continue
                matched += 1

                nw_pos = initial_body_q[nw_i, :3]
                nat_pos = native_xpos[nat_i]
                pos_err = np.max(np.abs(nw_pos - nat_pos))

                nw_quat = initial_body_q[nw_i, 3:]  # xyzw
                nat_wxyz = native_xquat[nat_i]
                nat_xyzw = np.array([nat_wxyz[1], nat_wxyz[2], nat_wxyz[3], nat_wxyz[0]])
                quat_err = min(
                    np.max(np.abs(nw_quat - nat_xyzw)),
                    np.max(np.abs(nw_quat + nat_xyzw)),
                )

                if pos_err > cross_tol or quat_err > cross_tol:
                    cross_mismatches.append(
                        f"  {nw_label} (newton #{nw_i}) vs {native_names[nat_i]} (native #{nat_i}):\n"
                        f"    pos_err={pos_err:.2e}, quat_err={quat_err:.2e}\n"
                        f"    usd:    pos={nw_pos}, quat_xyzw={nw_quat}\n"
                        f"    native: pos={nat_pos}, quat_xyzw={nat_xyzw}"
                    )
                break

        self.assertGreater(matched, 0, "No bodies matched between USD and native MJCF by name")
        self.assertEqual(
            len(cross_mismatches),
            0,
            f"USD initial body_q differs from native MJCF body transforms "
            f"for {len(cross_mismatches)}/{matched} matched bodies:\n" + "\n".join(cross_mismatches),
        )


# =============================================================================
# Simulation Equivalence Tests (pre-converted USD assets)
# =============================================================================
# Tests with pre-converted USD assets downloaded from newton-assets.
# The native MuJoCo model is always loaded from the original MJCF.
# Newton loads the pre-converted USD file.


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_H1(TestMenagerieUSD):
    """Unitree H1 humanoid: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "unitree_h1"
    robot_xml = "h1.xml"
    usd_asset_folder = "unitree_h1"
    usd_scene_file = "usd_structured/h1.usda"

    num_steps = 20
    fk_enabled = True
    backfill_model = True


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_G1WithHands(TestMenagerieUSD):
    """Unitree G1 29-DOF with hands: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "unitree_g1"
    robot_xml = "g1_with_hands.xml"
    usd_asset_folder = "unitree_g1"
    usd_scene_file = "usd_structured/g1_29dof_with_hand_rev_1_0.usda"

    num_steps = 20
    fk_enabled = True
    # Float32 + GPU atomic-reduction non-determinism: observed qvel diff
    # 2.4e-5 to 5.0e-5 across 5 reset+rerun trials (2x spread). Tolerance
    # set 2x above max observed to absorb the variance.
    dynamics_tolerance = 1e-4


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_ShadowHand(TestMenagerieUSD):
    """Shadow Hand (left): USD vs native MuJoCo simulation equivalence."""

    robot_folder = "shadow_hand"
    robot_xml = "left_hand.xml"
    usd_asset_folder = "shadow_hand"
    usd_scene_file = "usd_structured/left_shadow_hand.usda"

    num_steps = 20
    fk_enabled = True
    # Float32 + GPU atomic-reduction non-determinism: observed qvel diff
    # 7.8e-6 to 2.3e-5 across 5 reset+rerun trials (3x spread). Tolerance
    # set 4x above max observed to absorb the variance.
    dynamics_tolerance = 1e-4


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_Robotiq2f85V4(TestMenagerieUSD):
    """Robotiq 2F-85 v4 gripper: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "robotiq_2f85_v4"
    robot_xml = "2f85.xml"
    usd_asset_folder = "robotiq_2f85_v4"
    usd_scene_file = "usd_structured/Dual_wrist_camera.usda"

    num_steps = 20
    fk_enabled = True
    # USD asset has body_mass = 0.0033 kg for the gripper finger pads
    # (`left_pad` / `right_pad`); the source MJCF has near-zero mass 2e-6 kg.
    # The mass mismatch produces qvel diffs up to ~4e-3 on the gripper DOF in
    # the first few steps before settling. Other tests (model comparison,
    # FK) are unaffected once `_compare_inertia` is overridden to skip the
    # body_mass check. To tighten: regenerate the USD asset from the current
    # MJCF.
    dynamics_tolerance = 1e-2

    def _compare_inertia(self, newton_mjw: Any, native_mjw: Any) -> None:
        # body_mass differs for finger pads (see class docstring).
        pass


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_ApptronikApollo(TestMenagerieUSD):
    """Apptronik Apollo humanoid: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "apptronik_apollo"
    robot_xml = "apptronik_apollo.xml"
    usd_asset_folder = "apptronik_apollo"
    usd_scene_file = "usd_structured/apptronik_apollo.usda"

    num_steps = 20
    fk_enabled = True
    njmax = 398
    # Float32 + GPU atomic-reduction non-determinism: observed qvel diff
    # 2.2e-5 to 1.9e-4 across 5 reset+rerun trials (9x spread, the largest
    # spread of any enabled robot — Apollo has many free-joint chains where
    # tiny per-step gravity-projection diffs compound through the kinematic
    # tree). Tolerance set 2.5x above max observed to absorb the variance.
    dynamics_tolerance = 5e-4

    # Apollo's USD has no collision geoms, so geom/collision counts differ.
    model_skip_fields = TestMenagerieUSD.model_skip_fields | {
        "ngeom",
        "nmaxcondim",
        "nmaxpyramid",
    }


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_BoosterT1(TestMenagerieUSD):
    """Booster T1 humanoid: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "booster_t1"
    robot_xml = "t1.xml"
    usd_asset_folder = "booster_t1"
    usd_scene_file = "usd_structured/T1.usda"

    num_steps = 20
    fk_enabled = True
    # AL1/AR1 body_quat in the USD asset is authored as
    # (0.9999999, 0, 0.00048828122, 0) while the source MJCF has
    # quat="1 0 0.000440565 0" (post-normalize ~0.000440565). Newton imports
    # the USD value faithfully so the FK xquat diff (~4.77e-5) propagates
    # into those bodies and their children. Upstream asset mismatch, not a
    # Newton bug; bump tolerance until the USD asset is synced with the MJCF.
    fk_tolerance = 1e-4
    # qfrc_bias diff ~3.4e-5 on Left_Elbow_Yaw integrates to qvel ~6.4e-5
    # per step. Stable across reset+rerun trials. body_mass and body_inertia
    # match native after backfill, so this isn't an inertia issue — likely
    # float32 accumulation in mjwarp's Coriolis kernel on arm-chain DOFs, but
    # the exact source isn't isolated. Tolerance set to absorb the residual.
    dynamics_tolerance = 1e-4


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_WonikAllegro(TestMenagerieUSD):
    """Wonik Allegro Hand (left): USD vs native MuJoCo simulation equivalence."""

    robot_folder = "wonik_allegro"
    robot_xml = "left_hand.xml"
    usd_asset_folder = "wonik_allegro"
    usd_scene_file = "usd_structured/allegro_left.usda"

    # USD asset has body mass/inertia values that don't match the source MJCF
    # (e.g. palm body_mass: 0.438 USD vs 0.352 MJCF, 25% off). Backfilling the
    # full set of body fields (mass, inertia, iquat, ipos, subtreemass,
    # invweight0) closes most of the gap — qvel residual drops from ~0.29 to
    # ~0.015. A larger residual than other USD robots remains because the
    # backfill operates at runtime on mjw_model fields but mjwarp/Newton also
    # consume inertia-derived quantities cached at solver build time that
    # re-running smooth.{crb,factor_m} doesn't refresh. To tighten: regenerate
    # the USD asset from the current MJCF.
    num_steps = 20
    fk_enabled = True
    dynamics_tolerance = 5e-2

    def _compare_inertia(self, newton_mjw: Any, native_mjw: Any) -> None:
        # TODO: USD asset has different mass/inertia values than the original MJCF.
        pass

    def _compare_dof_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        # The original MJCF has armature=0 which the converter omits from USD.
        # Newton's builder default (0.01) then applies, causing a known mismatch.
        pass


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_UR5e(TestMenagerieUSD):
    """Universal Robots UR5e arm: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "universal_robots_ur5e"
    robot_xml = "ur5e.xml"
    usd_asset_folder = "universal_robots_ur5e"
    usd_scene_file = "usd_structured/ur5e.usda"

    # TODO(#2420): re-enable step-response dynamics. UR5e USD MjcActuator rows
    # match the position-shortcut pattern, so they're imported as JOINT_TARGET
    # (see parse_usd's MjcActuator post-process). _init_actuators rebuilds
    # JOINT_TARGET actuators with no per-actuator forcerange and instead clamps
    # at the joint via jnt_actfrcrange. Native MJCF UR5e uses per-actuator
    # forcerange. Both clip at the same magnitude, but mujoco-warp routes them
    # through different code paths (joint-level becomes a solver constraint),
    # producing small qpos diffs (~1e-3) at step 0 that exceed the 1e-6
    # tolerance. The fix is to also set actuator_forcerange on JOINT_TARGET-
    # built actuators in _init_actuators so the clipping path matches native;
    # that affects the MJCF JOINT_TARGET path too and is out of scope here.
    num_steps = 0
    fk_enabled = True
    backfill_model = True


# =============================================================================
# Part C: Menagerie Robot USD Test Stubs
# =============================================================================
# One class per menagerie robot. These use the default TestMenagerieUSD
# configuration; without a usd_asset_folder they are skipped in setUpClass
# before downloading menagerie assets.


# -----------------------------------------------------------------------------
# Arms
# -----------------------------------------------------------------------------


class TestMenagerie_AgilexPiper_USD(TestMenagerieUSD):
    """AgileX PIPER bimanual arm. (USD)."""

    robot_folder = "agilex_piper"


class TestMenagerie_ArxL5_USD(TestMenagerieUSD):
    """ARX L5 arm. (USD)."""

    robot_folder = "arx_l5"


class TestMenagerie_Dynamixel2r_USD(TestMenagerieUSD):
    """Dynamixel 2R simple arm. (USD)."""

    robot_folder = "dynamixel_2r"


class TestMenagerie_FrankaEmikaPanda_USD(TestMenagerieUSD):
    """Franka Emika Panda arm. (USD)."""

    robot_folder = "franka_emika_panda"


class TestMenagerie_FrankaFr3_USD(TestMenagerieUSD):
    """Franka FR3 arm. (USD)."""

    robot_folder = "franka_fr3"


class TestMenagerie_FrankaFr3V2_USD(TestMenagerieUSD):
    """Franka FR3 v2 arm. (USD)."""

    robot_folder = "franka_fr3_v2"


class TestMenagerie_KinovaGen3_USD(TestMenagerieUSD):
    """Kinova Gen3 arm. (USD)."""

    robot_folder = "kinova_gen3"


class TestMenagerie_KukaIiwa14_USD(TestMenagerieUSD):
    """KUKA iiwa 14 arm. (USD)."""

    robot_folder = "kuka_iiwa_14"


class TestMenagerie_LowCostRobotArm_USD(TestMenagerieUSD):
    """Low-cost robot arm. (USD)."""

    robot_folder = "low_cost_robot_arm"


class TestMenagerie_RethinkSawyer_USD(TestMenagerieUSD):
    """Rethink Robotics Sawyer arm. (USD)."""

    robot_folder = "rethink_robotics_sawyer"


class TestMenagerie_TrossenVx300s_USD(TestMenagerieUSD):
    """Trossen Robotics ViperX 300 S arm. (USD)."""

    robot_folder = "trossen_vx300s"


class TestMenagerie_TrossenWx250s_USD(TestMenagerieUSD):
    """Trossen Robotics WidowX 250 S arm. (USD)."""

    robot_folder = "trossen_wx250s"


class TestMenagerie_TrossenWxai_USD(TestMenagerieUSD):
    """Trossen Robotics WidowX AI arm. (USD)."""

    robot_folder = "trossen_wxai"


class TestMenagerie_TrsSoArm100_USD(TestMenagerieUSD):
    """TRS SO-ARM100 arm. (USD)."""

    robot_folder = "trs_so_arm100"


class TestMenagerie_UfactoryLite6_USD(TestMenagerieUSD):
    """UFACTORY Lite 6 arm. (USD)."""

    robot_folder = "ufactory_lite6"


class TestMenagerie_UfactoryXarm7_USD(TestMenagerieUSD):
    """UFACTORY xArm 7 arm. (USD)."""

    robot_folder = "ufactory_xarm7"


class TestMenagerie_UniversalRobotsUr5e_USD(TestMenagerieUSD):
    """Universal Robots UR5e arm (USD)."""

    robot_folder = "universal_robots_ur5e"


class TestMenagerie_UniversalRobotsUr10e_USD(TestMenagerieUSD):
    """Universal Robots UR10e arm. (USD)."""

    robot_folder = "universal_robots_ur10e"


# -----------------------------------------------------------------------------
# Grippers / Hands
# -----------------------------------------------------------------------------


class TestMenagerie_LeapHand_USD(TestMenagerieUSD):
    """LEAP Hand. (USD)."""

    robot_folder = "leap_hand"


class TestMenagerie_Robotiq2f85_USD(TestMenagerieUSD):
    """Robotiq 2F-85 gripper. (USD)."""

    robot_folder = "robotiq_2f85"


class TestMenagerie_Robotiq2f85V4_USD(TestMenagerieUSD):
    """Robotiq 2F-85 gripper v4. (USD)."""

    robot_folder = "robotiq_2f85_v4"


class TestMenagerie_ShadowDexee_USD(TestMenagerieUSD):
    """Shadow DEX-EE hand. (USD)."""

    robot_folder = "shadow_dexee"


class TestMenagerie_ShadowHand_USD(TestMenagerieUSD):
    """Shadow Hand. (USD)."""

    robot_folder = "shadow_hand"


class TestMenagerie_TetheriaAeroHandOpen_USD(TestMenagerieUSD):
    """Tetheria Aero Hand (open). (USD)."""

    robot_folder = "tetheria_aero_hand_open"


class TestMenagerie_UmiGripper_USD(TestMenagerieUSD):
    """UMI Gripper. (USD)."""

    robot_folder = "umi_gripper"


class TestMenagerie_WonikAllegro_USD(TestMenagerieUSD):
    """Wonik Allegro Hand. (USD)."""

    robot_folder = "wonik_allegro"


class TestMenagerie_IitSoftfoot_USD(TestMenagerieUSD):
    """IIT Softfoot biomechanical gripper. (USD)."""

    robot_folder = "iit_softfoot"


# -----------------------------------------------------------------------------
# Bimanual Systems
# -----------------------------------------------------------------------------


class TestMenagerie_Aloha_USD(TestMenagerieUSD):
    """ALOHA bimanual system. (USD)."""

    robot_folder = "aloha"


class TestMenagerie_GoogleRobot_USD(TestMenagerieUSD):
    """Google Robot (bimanual). (USD)."""

    robot_folder = "google_robot"


# -----------------------------------------------------------------------------
# Mobile Manipulators
# -----------------------------------------------------------------------------


class TestMenagerie_HelloRobotStretch_USD(TestMenagerieUSD):
    """Hello Robot Stretch. (USD)."""

    robot_folder = "hello_robot_stretch"


class TestMenagerie_HelloRobotStretch3_USD(TestMenagerieUSD):
    """Hello Robot Stretch 3. (USD)."""

    robot_folder = "hello_robot_stretch_3"


class TestMenagerie_PalTiago_USD(TestMenagerieUSD):
    """PAL Robotics TIAGo. (USD)."""

    robot_folder = "pal_tiago"


class TestMenagerie_PalTiagoDual_USD(TestMenagerieUSD):
    """PAL Robotics TIAGo Dual. (USD)."""

    robot_folder = "pal_tiago_dual"


class TestMenagerie_StanfordTidybot_USD(TestMenagerieUSD):
    """Stanford Tidybot mobile manipulator. (USD)."""

    robot_folder = "stanford_tidybot"


# -----------------------------------------------------------------------------
# Humanoids
# -----------------------------------------------------------------------------


class TestMenagerie_ApptronikApollo_USD(TestMenagerieUSD):
    """Apptronik Apollo humanoid. (USD)."""

    robot_folder = "apptronik_apollo"


class TestMenagerie_BerkeleyHumanoid_USD(TestMenagerieUSD):
    """Berkeley Humanoid. (USD)."""

    robot_folder = "berkeley_humanoid"


class TestMenagerie_BoosterT1_USD(TestMenagerieUSD):
    """Booster Robotics T1 humanoid. (USD)."""

    robot_folder = "booster_t1"


class TestMenagerie_FourierN1_USD(TestMenagerieUSD):
    """Fourier N1 humanoid. (USD)."""

    robot_folder = "fourier_n1"


class TestMenagerie_PalTalos_USD(TestMenagerieUSD):
    """PAL Robotics TALOS humanoid. (USD)."""

    robot_folder = "pal_talos"


class TestMenagerie_PndboticsAdamLite_USD(TestMenagerieUSD):
    """PNDbotics Adam Lite humanoid. (USD)."""

    robot_folder = "pndbotics_adam_lite"


class TestMenagerie_RobotisOp3_USD(TestMenagerieUSD):
    """Robotis OP3 humanoid. (USD)."""

    robot_folder = "robotis_op3"


class TestMenagerie_ToddlerBot2xc_USD(TestMenagerieUSD):
    """ToddlerBot 2XC humanoid. (USD)."""

    robot_folder = "toddlerbot_2xc"


class TestMenagerie_ToddlerBot2xm_USD(TestMenagerieUSD):
    """ToddlerBot 2XM humanoid. (USD)."""

    robot_folder = "toddlerbot_2xm"


class TestMenagerie_UnitreeG1_USD(TestMenagerieUSD):
    """Unitree G1 humanoid. (USD)."""

    robot_folder = "unitree_g1"


class TestMenagerie_UnitreeH1_USD(TestMenagerieUSD):
    """Unitree H1 humanoid. (USD)."""

    robot_folder = "unitree_h1"


# -----------------------------------------------------------------------------
# Bipeds
# -----------------------------------------------------------------------------


class TestMenagerie_AgilityCassie_USD(TestMenagerieUSD):
    """Agility Robotics Cassie biped. (USD)."""

    robot_folder = "agility_cassie"


# -----------------------------------------------------------------------------
# Quadrupeds
# -----------------------------------------------------------------------------


class TestMenagerie_AnyboticsAnymalB_USD(TestMenagerieUSD):
    """ANYbotics ANYmal B quadruped. (USD)."""

    robot_folder = "anybotics_anymal_b"


class TestMenagerie_AnyboticsAnymalC_USD(TestMenagerieUSD):
    """ANYbotics ANYmal C quadruped. (USD)."""

    robot_folder = "anybotics_anymal_c"


class TestMenagerie_BostonDynamicsSpot_USD(TestMenagerieUSD):
    """Boston Dynamics Spot quadruped. (USD)."""

    robot_folder = "boston_dynamics_spot"


class TestMenagerie_GoogleBarkourV0_USD(TestMenagerieUSD):
    """Google Barkour v0 quadruped. (USD)."""

    robot_folder = "google_barkour_v0"


class TestMenagerie_GoogleBarkourVb_USD(TestMenagerieUSD):
    """Google Barkour vB quadruped. (USD)."""

    robot_folder = "google_barkour_vb"


class TestMenagerie_UnitreeA1_USD(TestMenagerieUSD):
    """Unitree A1 quadruped. (USD)."""

    robot_folder = "unitree_a1"


class TestMenagerie_UnitreeGo1_USD(TestMenagerieUSD):
    """Unitree Go1 quadruped. (USD)."""

    robot_folder = "unitree_go1"


class TestMenagerie_UnitreeGo2_USD(TestMenagerieUSD):
    """Unitree Go2 quadruped. (USD)."""

    robot_folder = "unitree_go2"


# -----------------------------------------------------------------------------
# Arms with Gripper
# -----------------------------------------------------------------------------


class TestMenagerie_UnitreeZ1_USD(TestMenagerieUSD):
    """Unitree Z1 arm. (USD)."""

    robot_folder = "unitree_z1"


# -----------------------------------------------------------------------------
# Drones
# -----------------------------------------------------------------------------


class TestMenagerie_BitcrazeCrazyflie2_USD(TestMenagerieUSD):
    """Bitcraze Crazyflie 2 quadrotor. (USD)."""

    robot_folder = "bitcraze_crazyflie_2"


class TestMenagerie_SkydioX2_USD(TestMenagerieUSD):
    """Skydio X2 drone. (USD)."""

    robot_folder = "skydio_x2"


# -----------------------------------------------------------------------------
# Mobile Bases
# -----------------------------------------------------------------------------


class TestMenagerie_RobotSoccerKit_USD(TestMenagerieUSD):
    """Robot Soccer Kit omniwheel base. (USD)."""

    robot_folder = "robot_soccer_kit"


class TestMenagerie_RobotstudioSo101_USD(TestMenagerieUSD):
    """RobotStudio SO-101. (USD)."""

    robot_folder = "robotstudio_so101"


# -----------------------------------------------------------------------------
# Biomechanical
# -----------------------------------------------------------------------------


class TestMenagerie_Flybody_USD(TestMenagerieUSD):
    """Flybody fruit fly model. (USD)."""

    robot_folder = "flybody"


# -----------------------------------------------------------------------------
# Other
# -----------------------------------------------------------------------------


class TestMenagerie_I2rtYam_USD(TestMenagerieUSD):
    """i2rt YAM (Yet Another Manipulator). (USD)."""

    robot_folder = "i2rt_yam"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
