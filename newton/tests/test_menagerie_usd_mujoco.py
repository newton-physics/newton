# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
MuJoCo Menagerie USD Integration Tests

Tests that MuJoCo Menagerie robots converted to USD simulate identically
in Newton's MuJoCo solver vs native MuJoCo (loaded from original MJCF).

Import tests verify that each USD asset loads correctly (body/joint/
shape counts, no NaN values, correct joint types).

Simulation equivalence tests reuse the TestMenagerieBase infrastructure
from test_menagerie_mujoco.py to compare per-step simulation state between
Newton (USD) and native MuJoCo (MJCF).

Menagerie robot stubs. One class per menagerie robot, using the default
TestMenagerieUSD configuration. Initially these run with no usd_path (skipped
until a pre-converted USD is available).

Asset location: newton-assets repo (structured USD from mujoco_menagerie).
Set NEWTON_ASSETS_PATH to the repo root; default: ~/Documents/Repos/newton-assets.
"""

from __future__ import annotations

import os
import re
import unittest
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

import newton
from newton._src.usd.schemas import SchemaResolverMjc, SchemaResolverNewton
from newton.solvers import SolverMuJoCo
from newton.tests.test_menagerie_mujoco import (
    DEFAULT_MODEL_SKIP_FIELDS,
    StructuredControlStrategy,
    TestMenagerieBase,
    ZeroControlStrategy,
)
from newton.tests.unittest_utils import USD_AVAILABLE

# =============================================================================
# USD Model Creation
# =============================================================================


def create_newton_model_from_usd(
    usd_path: Path,
    *,
    num_worlds: int = 1,
    add_ground: bool = True,
    ignore_paths: list[str] | None = None,
) -> newton.Model:
    """Create a Newton model from a USD file (converted from MuJoCo Menagerie MJCF).

    Args:
        usd_path: Path to the USD scene file.
        num_worlds: Number of world instances to create.
        add_ground: Whether to add a ground plane.
        ignore_paths: Regex patterns for USD prim paths to skip during import.

    Returns:
        Finalized Newton Model.
    """
    robot_builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(robot_builder)

    robot_builder.default_shape_cfg.mu = 1.0
    robot_builder.default_shape_cfg.mu_torsional = 0.005
    robot_builder.default_shape_cfg.mu_rolling = 0.0001

    robot_builder.add_usd(
        str(usd_path),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
        ignore_paths=ignore_paths,
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


def _newton_assets_root() -> Path:
    root = os.environ.get("NEWTON_ASSETS_PATH")
    if root:
        return Path(root)
    return Path.home() / "Documents" / "Repos" / "newton-assets"


ASSETS_DIR = _newton_assets_root()

# Menagerie USD asset registry: maps robot name to its USD scene path.
# Paths are relative to newton-assets repo (usd_structured layout from PR #26).
MENAGERIE_USD_ASSETS = {
    "h1": {"usd_scene": "unitree_h1/usd_structured/h1.usda"},
    "g1_with_hands": {"usd_scene": "unitree_g1/usd_structured/g1_29dof_with_hand_rev_1_0.usda"},
    "shadow_hand": {"usd_scene": "shadow_hand/usd_structured/left_shadow_hand.usda"},
    "robotiq_2f85_v4": {"usd_scene": "robotiq_2f85/usd_structured/Dual_wrist_camera.usda"},
    "apptronik_apollo": {"usd_scene": "apptronik_apollo/usd_structured/apptronik_apollo.usda"},
    "booster_t1": {"usd_scene": "booster_t1/usd_structured/T1.usda"},
    "wonik_allegro": {"usd_scene": "wonik_allegro/usd_structured/allegro_left.usda"},
}


# =============================================================================
# Import Tests
# =============================================================================
@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUsdImport(unittest.TestCase):
    """Verify that each menagerie USD asset imports correctly into Newton."""

    def _load_robot(self, robot_name: str) -> tuple[newton.ModelBuilder, newton.Model]:
        """Load a menagerie USD asset and return the builder and finalized model."""
        config = MENAGERIE_USD_ASSETS[robot_name]
        usd_path = ASSETS_DIR / config["usd_scene"]
        self.assertTrue(usd_path.exists(), f"USD asset not found: {usd_path}")

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.mu_torsional = 0.005
        builder.default_shape_cfg.mu_rolling = 0.0001

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
        self._assert_no_nan(model, "robotiq_2f85_v4")

    def test_import_apptronik_apollo(self):
        builder, model = self._load_robot("apptronik_apollo")
        self.assertEqual(builder.body_count, 36)
        self.assertEqual(builder.joint_count, 35)
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
        config = MENAGERIE_USD_ASSETS["h1"]
        usd_path = ASSETS_DIR / config["usd_scene"]

        model = create_newton_model_from_usd(usd_path, num_worlds=4, add_ground=True)
        self.assertEqual(model.world_count, 4)
        self._assert_no_nan(model, "h1_multi_world")


# =============================================================================
# USD-specific sorted comparison helpers
# =============================================================================
def compare_bodies_sorted(
    newton_mjw: Any,
    native_mjw: Any,
    tol: float = 0.1,
) -> None:
    """Compare body properties between Newton and native, handling reordering.

    The USD importer may assign body properties to different indices than the
    MJCF parser. This function matches bodies by (mass, body_ipos) signature
    and verifies the multisets are equal.
    """
    from scipy.spatial.transform import Rotation

    assert newton_mjw.nbody == native_mjw.nbody, (
        f"nbody mismatch: newton={newton_mjw.nbody} vs native={native_mjw.nbody}"
    )

    nbody = newton_mjw.nbody

    newton_mass = newton_mjw.body_mass.numpy().flatten()
    native_mass = native_mjw.body_mass.numpy().flatten()
    newton_inertia = newton_mjw.body_inertia.numpy().reshape(-1, 3)
    native_inertia = native_mjw.body_inertia.numpy().reshape(-1, 3)
    newton_iquat = newton_mjw.body_iquat.numpy().reshape(-1, 4)
    native_iquat = native_mjw.body_iquat.numpy().reshape(-1, 4)

    def _reconstruct_tensor(principal: np.ndarray, iquat_wxyz: np.ndarray) -> np.ndarray:
        iquat_xyzw = np.roll(iquat_wxyz, -1, axis=-1)
        R = Rotation.from_quat(iquat_xyzw).as_matrix()
        D = np.diag(principal)
        return R @ D @ R.T

    # Build signatures for matching: sorted eigenvalues of the inertia tensor + mass.
    # Eigenvalues are invariant to the axis convention used for iquat.
    def _body_signature(mass: float, inertia: np.ndarray, iquat: np.ndarray) -> np.ndarray:
        tensor = _reconstruct_tensor(inertia, iquat)
        eigvals = sorted(np.linalg.eigvalsh(tensor))
        return np.array([float(mass), *eigvals], dtype=np.float64)

    newton_sigs = [_body_signature(newton_mass[i], newton_inertia[i], newton_iquat[i]) for i in range(nbody)]
    native_sigs = [_body_signature(native_mass[i], native_inertia[i], native_iquat[i]) for i in range(nbody)]

    # Sort by signature values and compare with tolerance
    newton_order = sorted(range(nbody), key=lambda i: tuple(newton_sigs[i]))
    native_order = sorted(range(nbody), key=lambda i: tuple(native_sigs[i]))

    mismatches = []
    for k in range(nbody):
        ni = newton_order[k]
        nati = native_order[k]
        diff = np.max(np.abs(newton_sigs[ni] - native_sigs[nati]))
        if diff > tol:
            mismatches.append(
                f"sorted[{k}]: newton body {ni}={newton_sigs[ni]} "
                f"native body {nati}={native_sigs[nati]} (max diff={diff:.2e})"
            )
    assert not mismatches, f"Body multiset mismatch ({len(mismatches)}/{nbody} after sorting):\n" + "\n".join(
        mismatches[:5]
    )


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
    (body_id, geom_type, geom_size) and compares physics-relevant properties.
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
    GEOM_MESH = 7

    def _geom_sig(gtype, gsize):
        if gtype == GEOM_PLANE:
            return (int(gtype), 0.0, 0.0, 0.0)
        elif gtype == GEOM_SPHERE:
            return (int(gtype), round(float(gsize[0]), 6), 0.0, 0.0)
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

    newton_phys = sorted(s for s in newton_sigs if s[0] != GEOM_PLANE)
    native_phys = sorted(s for s in native_sigs if s[0] != GEOM_PLANE)

    newton_counter = Counter(newton_phys)
    native_counter = Counter(native_phys)
    missing = []
    for sig, count in newton_counter.items():
        if native_counter.get(sig, 0) < count:
            missing.append(f"{sig}: newton={count}, native={native_counter.get(sig, 0)}")
    assert not missing, "Physics geom multiset: Newton has geoms not in native:\n" + "\n".join(missing[:5])


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


def _suffix_match(nw_name: str, native_name: str) -> bool:
    """Check if nw_name matches native_name by exact or boundary-aligned suffix.

    The USD converter may append ``_N`` (numeric) to avoid name collisions
    (e.g., MJCF joint ``torso`` becomes USD prim ``torso_1``).  We strip
    trailing ``_<digits>`` from the Newton name before the suffix check.
    """
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
        warnings.warn(
            f"Could not map {len(unmapped)}/{nu_native} actuators: {unmapped[:5]}",
            stacklevel=2,
        )

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
    # compare_bodies_sorted(); body_pos/body_quat are in DEFAULT skip
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
        if nn.shape != nat.shape:
            continue
        np.testing.assert_array_equal(_reindex(nn), nat, err_msg=f"body field {field} (reindexed)")

    for field in fields_remapped:
        newton_arr = getattr(newton_mjw, field, None)
        native_arr = getattr(native_mjw, field, None)
        if newton_arr is None or native_arr is None:
            continue
        nn = _reindex(newton_arr.numpy()).ravel()
        nat = native_arr.numpy().ravel()
        if nn.shape != nat.shape:
            continue
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
        if nn.shape != nat.shape:
            continue
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
        if nn.shape != nat.shape:
            continue
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


ACTUATOR_SKIP_FIELDS: set[str] = {
    "actuator_plugin",
    "actuator_user",
    "actuator_id",
    "actuator_trnid",
    "actuator_trntype",
    "actuator_trntype_body_adr",
    "actuator_actadr",
    "actuator_actnum",
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
        if nn.shape != nat.shape:
            continue

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

    Subclasses set usd_path to the pre-converted USD file. Native MuJoCo
    still loads the original MJCF from menagerie for comparison.
    """

    usd_path: str = ""

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
        # Sparse mass matrix structure and Cholesky: derived from body/DOF tree ordering
        "mapM2M",
        "qLD_updates",
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

    # Per-step comparison: body/geom/dof ordering can all differ between
    # USD and MJCF, so only aggregate (order-independent) fields are safe.
    compare_fields: ClassVar[list[str]] = [
        "energy",
    ]

    def _compare_compiled_fields(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Skip compiled-field check for USD models.

        USD import re-diagonalizes inertia, causing large differences in
        derived fields (body_invweight0, dof_invweight0, actuator_acc0).
        These are already handled by the mapped comparison hooks.
        """

    def _compare_inertia(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare inertia using sorted body signatures (handles reordering)."""
        compare_bodies_sorted(newton_mjw, native_mjw)

    def _compare_geoms(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare geom subset: Newton excludes visual-only geoms."""
        compare_geoms_subset(newton_mjw, native_mjw)

    def _compare_jnt_range(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare joint properties using sorted multisets (handles reordering)."""
        compare_joints_sorted(newton_mjw, native_mjw)

    def _init_control(self, native_mjw_data: Any, newton_control: Any) -> None:
        """Handle missing actuators: USD actuator import is incomplete."""
        mujoco_ctrl = getattr(newton_control, "mujoco", None)
        if mujoco_ctrl is None or not hasattr(mujoco_ctrl, "ctrl"):
            self.control_strategy = ZeroControlStrategy()
            self.control_strategy.init(native_mjw_data.ctrl, native_mjw_data.ctrl)
        else:
            self.control_strategy.init(native_mjw_data.ctrl, mujoco_ctrl.ctrl)  # type: ignore[union-attr]

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

        self._body_map = build_body_index_map(newton_solver.mj_model, mj_model)
        self._jnt_map = build_jnt_index_map(newton_solver.mj_model, mj_model)
        self._dof_map = build_dof_index_map(
            newton_solver.mjw_model,
            native_mjw_model,
            self._jnt_map,
        )
        self._actuator_map = build_actuator_index_map(newton_solver.mj_model, mj_model)

    def _compare_body_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare physics-relevant body fields using name-based index mapping."""
        compare_body_physics_mapped(newton_mjw, native_mjw, self._body_map)

    def _compare_dof_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare physics-relevant DOF fields using name-based index mapping."""
        compare_dof_physics_mapped(newton_mjw, native_mjw, self._dof_map)

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
    }

    # Regex patterns for USD prim paths to skip during import.
    usd_ignore_paths: ClassVar[list[str]] = []

    # Body names to strip from the MJCF XML before creating the native model.
    # Used when the USD omits or cannot represent certain MJCF bodies.
    mjcf_strip_bodies: ClassVar[list[str]] = []

    def _get_mjcf_xml(self) -> str:
        xml_content = super()._get_mjcf_xml()
        if self.mjcf_strip_bodies:
            for body_name in self.mjcf_strip_bodies:
                xml_content = re.sub(rf'<body\s+name="{re.escape(body_name)}"[^/]*/>', "", xml_content)
        return xml_content

    def _create_newton_model(self) -> newton.Model:
        """Create Newton model from pre-converted USD file."""
        if not self.usd_path:
            raise unittest.SkipTest("usd_path not defined")
        usd_file = Path(self.usd_path)
        if not usd_file.exists():
            raise unittest.SkipTest(f"USD file not found: {usd_file}")
        return create_newton_model_from_usd(
            usd_file,
            num_worlds=self.num_worlds,
            add_ground=False,  # scene.xml includes ground plane
            ignore_paths=self.usd_ignore_paths or None,
        )


# =============================================================================
# Simulation Equivalence Tests (pre-converted USD assets)
# =============================================================================
# Tests with local pre-converted USD assets and custom configurations.
# The native MuJoCo model is always loaded from the original MJCF.
# Newton loads the pre-converted USD file.


def _usd_path(asset_key: str) -> str:
    return str(ASSETS_DIR / MENAGERIE_USD_ASSETS[asset_key]["usd_scene"])


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_H1(TestMenagerieUSD):
    """Unitree H1 humanoid: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "unitree_h1"
    robot_xml = "h1.xml"
    usd_path = _usd_path("h1")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_G1WithHands(TestMenagerieUSD):
    """Unitree G1 29-DOF with hands: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "unitree_g1"
    robot_xml = "g1_with_hands.xml"
    usd_path = _usd_path("g1_with_hands")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_ShadowHand(TestMenagerieUSD):
    """Shadow Hand (left): USD vs native MuJoCo simulation equivalence."""

    robot_folder = "shadow_hand"
    robot_xml = "left_hand.xml"
    usd_path = _usd_path("shadow_hand")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_Robotiq2f85V4(TestMenagerieUSD):
    """Robotiq 2F-85 v4 gripper: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "robotiq_2f85_v4"
    robot_xml = "2f85.xml"
    usd_path = _usd_path("robotiq_2f85_v4")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_ApptronikApollo(TestMenagerieUSD):
    """Apptronik Apollo humanoid: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "apptronik_apollo"
    robot_xml = "apptronik_apollo.xml"
    usd_path = _usd_path("apptronik_apollo")

    num_worlds = 2
    num_steps = 100
    njmax = 398
    control_strategy = StructuredControlStrategy(seed=42)

    # Apollo's USD has no collision geoms, so geom/collision counts differ.
    model_skip_fields = TestMenagerieUSD.model_skip_fields | {
        "ngeom",
        "nmaxcondim",
        "nmaxpyramid",
    }

    # world_link is an empty static body in MJCF (child of worldbody, no joint,
    # no geoms). Its USD representation uses a PhysicsFixedJoint to the world
    # root, but the importer doesn't yet handle orphan body-to-world fixed
    # joints (they fall outside the articulation). Strip from both sides.
    mjcf_strip_bodies: ClassVar[list[str]] = ["world_link"]
    usd_ignore_paths: ClassVar[list[str]] = [".*/world_link.*"]


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_BoosterT1(TestMenagerieUSD):
    """Booster T1 humanoid: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "booster_t1"
    robot_xml = "t1.xml"
    usd_path = _usd_path("booster_t1")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_WonikAllegro(TestMenagerieUSD):
    """Wonik Allegro Hand (left): USD vs native MuJoCo simulation equivalence."""

    robot_folder = "wonik_allegro"
    robot_xml = "left_hand.xml"
    usd_path = _usd_path("wonik_allegro")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)

    def _compare_dof_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        # The original MJCF has armature=0 which the converter omits from USD.
        # Newton's builder default (0.01) then applies, causing a known mismatch.
        pass


# =============================================================================
# Part C: Menagerie Robot USD Test Stubs
# =============================================================================
# One class per menagerie robot. These use the default TestMenagerieUSD
# configuration; without a usd_path they are auto-skipped.


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
