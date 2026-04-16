# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo Menagerie integration tests.

Verifies that robots from the MuJoCo Menagerie produce equivalent simulation
results when loaded via MJCF into Newton's SolverMuJoCo vs native mujoco_warp.

Architecture::

    TestMenagerieBase           Abstract base with all test infrastructure
    ├── TestMenagerieMJCF       Load Newton model from MJCF
    │   ├── TestMenagerie_UniversalRobotsUr5e   (enabled)
    │   ├── TestMenagerie_ApptronikApollo       (enabled)
    │   └── ...                                 (61 robots total, most skipped)
    └── TestMenagerieUSD        Load Newton model from USD (all skipped)

Test tiers (each robot can enable independently):
    - ``test_model_comparison()``: Deterministic model field checks — always runs.
    - ``test_forward_kinematics()``: Compares body poses from joint positions
      (no forces/contacts). Gated by ``fk_enabled``.
    - ``test_dynamics()``: Per-DOF step response — each world commands one actuator
      to a target position. Collisions disabled, both sides run full
      ``mujoco_warp.step()`` independently. Gated by ``num_steps > 0``.

Each test:
    1. Downloads the robot from menagerie (cached).
    2. Creates a Newton model (via MJCF) and a native mujoco_warp model.
    3. Compares model fields with physics-equivalence checks for inertia, solref, etc.
    4. Optionally runs forward kinematics, comparing body poses.
    5. Optionally runs dynamics (step-response), comparing per-step qpos/qvel.

Per-robot configuration (override in subclass):
    - ``backfill_model``: Copy computed model fields from native to Newton to
      isolate simulation diffs from model compilation diffs.
    - ``dynamics_target`` / ``dynamics_tolerance``: Step-response target and tolerance.
    - ``model_skip_fields``: Fields to skip in model comparison.
"""

from __future__ import annotations

import os
import unittest
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import JointType
from newton._src.utils.download_assets import MENAGERIE_REF, MENAGERIE_URL, download_git_folder
from newton._src.utils.import_mjcf import _load_and_expand_mjcf
from newton.solvers import SolverMuJoCo

# Check for mujoco availability via SolverMuJoCo's lazy import mechanism
try:
    _mujoco, _mujoco_warp = SolverMuJoCo.import_mujoco()
    MUJOCO_AVAILABLE = True
except ImportError:
    _mujoco = None
    _mujoco_warp = None
    MUJOCO_AVAILABLE = False


# =============================================================================
# Asset Management
# =============================================================================

# If set, use this path as the root of an already-cloned mujoco_menagerie repo
# instead of downloading. Example: export NEWTON_MENAGERIE_PATH=/path/to/mujoco_menagerie
NEWTON_MENAGERIE_PATH_ENV = "NEWTON_MENAGERIE_PATH"


def download_menagerie_asset(
    robot_folder: str,
    cache_dir: str | None = None,
    force_refresh: bool = False,
) -> Path:
    """
    Download a robot folder from the MuJoCo Menagerie repository.

    If the environment variable NEWTON_MENAGERIE_PATH is set to the root of an
    already-cloned mujoco_menagerie repo, that path is used and no download occurs.

    Args:
        robot_folder: The folder name in the menagerie repo (e.g., "unitree_go2")
        cache_dir: Optional cache directory override
        force_refresh: If True, re-download even if cached

    Returns:
        Path to the downloaded robot folder
    """
    local_root = os.environ.get(NEWTON_MENAGERIE_PATH_ENV)
    if local_root and not force_refresh:
        path = Path(local_root) / robot_folder
        if path.exists():
            return path

    return download_git_folder(
        MENAGERIE_URL,
        robot_folder,
        cache_dir=cache_dir,
        ref=MENAGERIE_REF,
        force_refresh=force_refresh,
    )


# =============================================================================
# Model Source Factory
# =============================================================================


def create_newton_model_from_mjcf(
    mjcf_path: Path,
    *,
    num_worlds: int = 1,
    add_ground: bool = True,
    parse_visuals: bool = False,
) -> newton.Model:
    """
    Create a Newton model from an MJCF file.

    Args:
        mjcf_path: Path to the MJCF XML file
        num_worlds: Number of world instances to create
        add_ground: Whether to add a ground plane
        parse_visuals: Whether to parse visual-only geoms (default False for physics testing)

    Returns:
        Finalized Newton Model
    """
    # Create articulation builder for the robot
    robot_builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(robot_builder)

    # floating defaults to None, which honors the MJCF's explicit joint definitions.
    # Menagerie models define their own <freejoint> tags for floating-base robots.
    robot_builder.add_mjcf(
        str(mjcf_path),
        parse_visuals=parse_visuals,
        ctrl_direct=True,
    )

    # Create main builder and replicate
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
# Control Strategies
# =============================================================================


class ControlStrategy:
    """Base class for control generation strategies."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None):
        """Reset the RNG state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    @abstractmethod
    def init(self, native_ctrl: wp.array, newton_ctrl: wp.array):
        """Initialize with the ctrl arrays that will be filled.

        Args:
            native_ctrl: Native mujoco_warp data ctrl array (num_worlds, num_actuators)
            newton_ctrl: Newton control.mujoco.ctrl array (num_worlds * num_actuators,)
        """
        ...

    @abstractmethod
    def fill_control(self, t: float):
        """Fill control values into the initialized arrays at time t."""
        ...


@wp.kernel
def step_response_control_kernel(
    native_ctrl: wp.array[wp.float32],  # type: ignore[valid-type]
    newton_ctrl: wp.array[wp.float32],  # type: ignore[valid-type]
    target: wp.float32,
    num_actuators: int,
):
    """Set ctrl[world_i, act_i] = target when world_i % nu == act_i, else 0."""
    i = wp.tid()
    world_i = i // num_actuators
    act_i = i % num_actuators  # type: ignore[operator]
    val = float(0.0)
    if world_i % num_actuators == act_i:
        val = target
    native_ctrl[i] = val
    newton_ctrl[i] = val


class StepResponseControlStrategy(ControlStrategy):
    """Each world commands one actuator to a target position, others stay at zero."""

    def __init__(self, target: float = 0.3, seed: int = 42):
        super().__init__(seed)
        self.target = target
        self._native_ctrl: wp.array | None = None
        self._newton_ctrl: wp.array | None = None
        self._n: int = 0
        self._num_actuators: int = 0

    def init(self, native_ctrl: wp.array, newton_ctrl: wp.array):
        num_worlds, num_actuators = native_ctrl.shape
        self._native_ctrl = native_ctrl.flatten()
        self._newton_ctrl = newton_ctrl
        self._n = num_worlds * num_actuators
        self._num_actuators = num_actuators

    def fill_control(self, t: float):
        if self._native_ctrl is None:
            raise RuntimeError("Call init() first")
        wp.launch(
            step_response_control_kernel,
            dim=self._n,
            inputs=[
                self._native_ctrl,
                self._newton_ctrl,
                self.target,
                self._num_actuators,
            ],
        )


# =============================================================================
# Comparison
# =============================================================================

# Default tolerances for MjData field comparison.
# Default fields to compare in FK test
DEFAULT_FK_FIELDS: list[str] = [
    "xpos",
    "xquat",
]

# Default fields to skip in MjWarpModel comparison (internal/non-comparable)
DEFAULT_MODEL_SKIP_FIELDS: set[str] = {
    "__",
    "ptr",
    "body_conaffinity",
    "body_contype",
    "exclude_signature",
    # TileSet types: comparison function doesn't handle these
    "qM_tiles",
    "qLD_tiles",
    "qLDiagInv_tiles",
    # Visualization group: Newton defaults to 0, native may use other groups
    "geom_group",
    # Collision exclusions: Newton needs to fix parent/child filtering to match MuJoCo
    "nexclude",
    # Lights: Newton doesn't parse lights from MJCF
    "light_",
    "nlight",
    # Cameras: Newton doesn't parse cameras from MJCF
    "cam_",
    "ncam",
    # Sensors: Newton doesn't parse sensors from MJCF
    "sensor",
    "nsensor",
    # Materials: Newton doesn't pass materials to MuJoCo spec
    "mat_",
    "nmat",
    # Mocap bodies: Newton handles fixed base differently
    "mocap_",
    "nmocap",
    "body_mocapid",
    # Inertia representation: Newton re-diagonalizes, giving same physics but different
    # principal axis ordering and orientation. Compare via compare_inertia_tensors() instead.
    "body_inertia",
    # Inertia frame offset: derived from inertia diagonalization. Differs when Newton
    # produces different principal axes (e.g. for bodies with mesh-based visual geoms).
    "body_ipos",
    # Inertia frame orientation: derived from inertia diagonalization.
    "body_iquat",
    # Collision filtering: Newton uses different representation but equivalent behavior
    "geom_conaffinity",
    "geom_contype",
    # Joint actuator force limits: Newton unconditionally sets jnt_actfrclimited=True with
    # effort_limit (default 1e6), while MuJoCo defaults to False when no actuatorfrcrange
    # is specified in MJCF. When limit is never hit, this has NO numerical effect.
    "jnt_actfrclimited",
    "jnt_actfrcrange",
    # Solref fields: Newton uses direct mode (-ke, -kd), native uses standard mode (tc, dr)
    # Compare via compare_solref_physics() instead for physics equivalence
    "dof_solref",
    "eq_solref",
    "geom_solref",
    "jnt_solref",
    "pair_solref",
    "pair_solreffriction",
    "tendon_solref_fri",
    "tendon_solref_lim",
    # RGBA: Newton uses different default color for geoms without explicit rgba
    "geom_rgba",
    # Size: Compared via compare_geom_fields_unordered() which understands type-specific semantics
    "geom_size",
    # Site size: Only a subset of the 3 elements is meaningful per type (sphere=1,
    # capsule/cylinder=2, box=3). Compared via _compare_sites() instead.
    "site_size",
    # Range: Compared via compare_jnt_range() which only checks limited joints
    # (MuJoCo ignores range when jnt_limited=False, Newton stores [-1e10, 1e10])
    "jnt_range",
    # Timestep: not registered as custom attribute (conflicts with step() parameter).
    # Extracted from native model at runtime instead.
    "opt.timestep",
    # Integrator: Newton may select a different integrator than the MJCF default.
    # The solver forces the correct integrator at runtime regardless.
    "opt.integrator",
    # Geom ordering: Newton's solver may order geoms differently (e.g. colliders before
    # visuals). Content is verified by compare_geom_fields_unordered() instead.
    "body_geomadr",
    "body_geomnum",
    "geom_",
    "pair_geom",  # geom indices depend on geom ordering
    "nxn_",  # broadphase pairs depend on geom ordering
    # Compilation-dependent fields: validated at 1e-3 by compare_compiled_model_fields()
    # Derived from inertia by set_const; differs when inertia representation differs. Backfilled.
    "body_invweight0",
    # Derived from inertia by set_const; differs when inertia representation differs. Backfilled.
    # Derived from inertia and dof_armature by set_const_0. Backfilled.
    "dof_invweight0",
    # Body frame position/orientation: compilation-dependent, derived from joint and inertia
    # frames by mj_setConst. Differs due to inertia re-diagonalization. Backfilled.
    "body_pos",
    "body_quat",
    # Subtree mass: sum of masses in subtree, differs when body_mass differs (visual geom mass).
    "body_subtreemass",
    # Computed from mass matrix and actuator moment at qpos0; differs due to inertia
    # re-diagonalization. Backfilled instead.
    "actuator_acc0",
    "actuator_lengthrange",  # Derived from joint ranges, computed by set_length_range
    "stat",  # meaninertia derived from invweight0
    # Meshes: Newton / trimesh may create a different number of meshes (nmesh differs),
    # so ALL per-mesh fields have incompatible shapes. Skip everything mesh-related.
    "nmesh",
    "nmeshvert",
    "nmeshnormal",
    "nmeshpoly",
    "nmeshface",
    "nmaxmeshdeg",
    "nmaxpolygon",
    "mesh_",
}


def compare_compiled_model_fields(
    newton_mjw: Any,
    native_mjw: Any,
    fields: list[str] | None = None,
    tol: float = 1e-3,
) -> None:
    """Compare model fields that depend on compilation (mj_setConst).

    These fields (invweight0, body_pos, body_quat, etc.) may have small
    numerical differences between Newton's and MuJoCo's model compilation.
    A 1e-3 tolerance catches real parser bugs while allowing expected
    compilation differences.

    Fields already validated by compare_inertia_tensors (body_inertia,
    body_iquat) are skipped.

    Args:
        newton_mjw: Newton's MjWarpModel
        native_mjw: Native MuJoCo's MjWarpModel
        fields: Field names to compare (defaults to MODEL_BACKFILL_FIELDS)
        tol: Maximum allowed absolute difference (default 1e-3)
    """
    if fields is None:
        fields = MODEL_BACKFILL_FIELDS

    # Validated by compare_inertia_tensors() with physics-equivalence check
    skip_fields = {"body_inertia", "body_iquat"}

    for field in fields:
        if field in skip_fields:
            continue

        native_arr = getattr(native_mjw, field, None)
        newton_arr = getattr(newton_mjw, field, None)

        if native_arr is None or newton_arr is None:
            continue
        if not hasattr(native_arr, "numpy") or not hasattr(newton_arr, "numpy"):
            continue

        assert native_arr.shape == newton_arr.shape, (
            f"Compiled field '{field}' shape mismatch: {newton_arr.shape} vs {native_arr.shape}"
        )

        diff = float(np.max(np.abs(native_arr.numpy().astype(float) - newton_arr.numpy().astype(float))))
        assert diff <= tol, (
            f"Compiled field '{field}' has diff {diff:.6e} > tol {tol:.0e}. "
            f"This likely indicates a parser bug, not a compilation difference."
        )


def compare_models(
    newton_mjw: Any,
    native_mjw: Any,
    skip_fields: set[str] | None = None,
    backfill_fields: list[str] | None = None,
) -> None:
    """Run all model comparison checks between Newton and native MuJoCo models.

    Consolidates the full suite of structural, physical, and compiled-field
    comparisons into a single entry point. Checks that involve per-index body,
    geom, joint, or DOF comparison are skipped when the corresponding field
    prefix is in skip_fields (as used by USD tests with reordered indices).

    Args:
        skip_fields: Substrings to skip in field-level comparison.
        backfill_fields: Fields to validate at relaxed tolerance via
            :func:`compare_compiled_model_fields`.
    """
    if skip_fields is None:
        skip_fields = set()

    def _skipped(prefix: str) -> bool:
        return any(s in prefix for s in skip_fields)

    compare_mjw_models(newton_mjw, native_mjw, skip_fields=skip_fields)

    if not _skipped("body_inertia"):
        compare_inertia_tensors(newton_mjw, native_mjw)

    for solref_field in [
        "dof_solref",
        "eq_solref",
        "geom_solref",
        "jnt_solref",
        "pair_solref",
        "pair_solreffriction",
        "tendon_solref_fri",
        "tendon_solref_lim",
    ]:
        if any(s in solref_field for s in skip_fields):
            continue
        if hasattr(newton_mjw, solref_field) and hasattr(native_mjw, solref_field):
            newton_arr = getattr(newton_mjw, solref_field)
            native_arr = getattr(native_mjw, solref_field)
            if newton_arr is not None and native_arr is not None:
                if hasattr(newton_arr, "shape") and newton_arr.shape == native_arr.shape and newton_arr.shape[0] > 0:
                    compare_solref_physics(newton_mjw, native_mjw, solref_field)

    if not _skipped("geom_") and newton_mjw.ngeom == native_mjw.ngeom:
        compare_geom_fields_unordered(newton_mjw, native_mjw, skip_fields=skip_fields)

    if not _skipped("jnt_"):
        compare_jnt_range(newton_mjw, native_mjw)

    if not _skipped("body_invweight0"):
        compare_compiled_model_fields(newton_mjw, native_mjw, backfill_fields)

    if not _skipped("site_"):
        compare_site_sizes(newton_mjw, native_mjw)


def compare_inertia_tensors(
    newton_mjw: Any,
    native_mjw: Any,
    tol: float = 1e-5,
) -> None:
    """Compare inertia by reconstructing full 3x3 tensors from principal moments + iquat.

    MuJoCo stores inertia as principal moments + orientation quaternion. The eig3
    determinant fix ensures both produce valid quaternions, but eigenvalue ordering
    may differ. Reconstruction verifies physics equivalence: I = R @ diag(d) @ R.T
    """
    from scipy.spatial.transform import Rotation

    newton_inertia = newton_mjw.body_inertia.numpy()  # (nworld, nbody, 3)
    native_inertia = native_mjw.body_inertia.numpy()
    newton_iquat = newton_mjw.body_iquat.numpy()  # (nworld, nbody, 4) wxyz
    native_iquat = native_mjw.body_iquat.numpy()

    assert newton_inertia.shape == native_inertia.shape, (
        f"body_inertia shape mismatch: {newton_inertia.shape} vs {native_inertia.shape}"
    )

    nworld, nbody, _ = newton_inertia.shape

    # Vectorized reconstruction: I = R @ diag(d) @ R.T for all bodies at once
    def reconstruct_all(principal: np.ndarray, iquat_wxyz: np.ndarray) -> np.ndarray:
        """Reconstruct full tensors from principal moments and wxyz quaternions."""
        # scipy uses xyzw, convert from wxyz
        iquat_xyzw = np.roll(iquat_wxyz, -1, axis=-1)
        flat_quats = iquat_xyzw.reshape(-1, 4)
        R = Rotation.from_quat(flat_quats).as_matrix()  # (n, 3, 3)
        flat_principal = principal.reshape(-1, 3)
        # I = R @ diag(d) @ R.T, vectorized
        D = np.einsum("ni,nij->nij", flat_principal, np.eye(3)[None, :, :].repeat(len(flat_principal), axis=0))
        tensors = np.einsum("nij,njk,nlk->nil", R, D, R)
        return tensors.reshape(nworld, nbody, 3, 3)

    newton_tensors = reconstruct_all(newton_inertia, newton_iquat)
    native_tensors = reconstruct_all(native_inertia, native_iquat)

    np.testing.assert_allclose(
        newton_tensors,
        native_tensors,
        rtol=0,
        atol=tol,
        err_msg="Inertia tensor mismatch (reconstructed from principal + iquat)",
    )


def solref_to_ke_kd(solref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert MuJoCo solref to (ke, kd) for physics-equivalence comparison.

    Args:
        solref: Array of shape (..., 2) with [timeconst, dampratio] or [-ke, -kd]

    Returns:
        (ke, kd) arrays with same leading dimensions
    """
    timeconst = solref[..., 0]
    dampratio = solref[..., 1]

    # Direct mode: both negative -> solref = (-ke, -kd)
    direct_mode = (timeconst < 0) & (dampratio < 0)

    # Standard mode: ke = 1/(tc^2 * dr^2), kd = 2/tc
    ke_standard = 1.0 / (timeconst**2 * dampratio**2)
    kd_standard = 2.0 / timeconst

    # Direct mode: ke = -tc, kd = -dr
    ke_direct = -timeconst
    kd_direct = -dampratio

    ke = np.where(direct_mode, ke_direct, ke_standard)
    kd = np.where(direct_mode, kd_direct, kd_standard)

    return ke, kd


def compare_solref_physics(
    newton_mjw: Any,
    native_mjw: Any,
    field_name: str,
    tol: float = 1e-3,
) -> None:
    """Compare solref fields by converting to effective ke/kd values.

    MuJoCo solref can be in standard mode [timeconst, dampratio] or
    direct mode [-ke, -kd]. This compares the physics-equivalent ke/kd.
    """
    newton_solref = getattr(newton_mjw, field_name).numpy()
    native_solref = getattr(native_mjw, field_name).numpy()

    assert newton_solref.shape == native_solref.shape, (
        f"{field_name} shape mismatch: {newton_solref.shape} vs {native_solref.shape}"
    )

    # Mask out zero solrefs (e.g. pair_solreffriction defaults to [0,0] meaning "unused").
    # Both sides should have identical zeros; physics conversion would produce inf.
    nonzero = (newton_solref[..., 0] != 0) | (native_solref[..., 0] != 0)
    if not nonzero.any():
        # All zeros on both sides — nothing to compare
        np.testing.assert_array_equal(newton_solref, native_solref, err_msg=f"{field_name} zero-solref mismatch")
        return

    newton_ke, newton_kd = solref_to_ke_kd(newton_solref)
    native_ke, native_kd = solref_to_ke_kd(native_solref)

    # Only compare non-zero entries
    np.testing.assert_allclose(
        newton_ke[nonzero],
        native_ke[nonzero],
        rtol=tol,
        atol=0,
        err_msg=f"{field_name} ke mismatch (physics-equivalent)",
    )
    np.testing.assert_allclose(
        newton_kd[nonzero],
        native_kd[nonzero],
        rtol=tol,
        atol=0,
        err_msg=f"{field_name} kd mismatch (physics-equivalent)",
    )


def compare_geom_fields_unordered(
    newton_mjw: Any,
    native_mjw: Any,
    skip_fields: set[str] | None = None,
    tol: float = 1e-6,
) -> None:
    """Compare geom fields by matching geoms across models regardless of ordering.

    Matches geoms by (body_id, geom_type) pairs within each body, then compares
    physics-relevant fields for each matched pair. This handles models where
    Newton and native MuJoCo order geoms differently (e.g. colliders first vs
    MJCF order).

    Args:
        newton_mjw: Newton's mujoco_warp model.
        native_mjw: Native mujoco_warp model.
        skip_fields: Fields to skip (uses substring matching like the main comparison).
        tol: Tolerance for floating-point comparisons.
    """
    skip_fields = skip_fields or set()

    newton_bodyid = newton_mjw.geom_bodyid.numpy()  # (ngeom,)
    native_bodyid = native_mjw.geom_bodyid.numpy()
    newton_type = newton_mjw.geom_type.numpy()  # (ngeom,)
    native_type = native_mjw.geom_type.numpy()

    assert len(newton_bodyid) == len(native_bodyid), (
        f"ngeom mismatch: newton={len(newton_bodyid)} vs native={len(native_bodyid)}"
    )

    ngeom = len(newton_bodyid)

    # Build matching: for each body, collect geom indices grouped by type.
    # Then match in order within each (body, type) group.
    def _group_by_body_type(bodyid, gtype):
        groups = defaultdict(list)
        for i in range(len(bodyid)):
            groups[(int(bodyid[i]), int(gtype[i]))].append(i)
        return groups

    newton_groups = _group_by_body_type(newton_bodyid, newton_type)
    native_groups = _group_by_body_type(native_bodyid, native_type)

    # Verify same set of (body, type) keys
    assert newton_groups.keys() == native_groups.keys(), (
        f"geom (body, type) groups differ:\n"
        f"  newton-only: {newton_groups.keys() - native_groups.keys()}\n"
        f"  native-only: {native_groups.keys() - newton_groups.keys()}"
    )

    # Build index mapping: newton_idx -> native_idx
    newton_to_native = np.full(ngeom, -1, dtype=np.int32)
    for key in newton_groups:
        n_indices = newton_groups[key]
        nat_indices = native_groups[key]
        assert len(n_indices) == len(nat_indices), (
            f"geom count mismatch for (body={key[0]}, type={key[1]}): "
            f"newton={len(n_indices)} vs native={len(nat_indices)}"
        )
        for ni, nati in zip(n_indices, nat_indices, strict=True):
            newton_to_native[ni] = nati

    assert np.all(newton_to_native >= 0), "Failed to match all geoms"

    # Compare fields using the mapping
    GEOM_PLANE = 0

    geom_fields = [
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_margin",
        "geom_gap",
        "geom_solmix",
        "geom_solref",
        "geom_solimp",
    ]

    for field_name in geom_fields:
        if any(s in field_name for s in skip_fields):
            continue
        newton_arr = getattr(newton_mjw, field_name, None)
        native_arr = getattr(native_mjw, field_name, None)
        if newton_arr is None or native_arr is None:
            continue
        newton_np = newton_arr.numpy()
        native_np = native_arr.numpy()

        if newton_np.ndim >= 2 and newton_np.shape[0] == newton_mjw.nworld:
            # Batched: (nworld, ngeom, ...)
            for w in range(newton_np.shape[0]):
                reordered_native = native_np[w][newton_to_native]
                for g in range(ngeom):
                    if newton_type[g] == GEOM_PLANE and field_name in ("geom_pos", "geom_quat"):
                        continue  # plane pos/quat may differ cosmetically
                    np.testing.assert_allclose(
                        newton_np[w, g],
                        reordered_native[g],
                        atol=tol,
                        rtol=0,
                        err_msg=f"{field_name}[world={w}, geom={g}]",
                    )
        else:
            # Non-batched: (ngeom, ...)
            reordered_native = native_np[newton_to_native]
            for g in range(ngeom):
                if newton_type[g] == GEOM_PLANE and field_name in ("geom_pos", "geom_quat"):
                    continue
                np.testing.assert_allclose(
                    newton_np[g],
                    reordered_native[g],
                    atol=tol,
                    rtol=0,
                    err_msg=f"{field_name}[geom={g}]",
                )

    # Compare geom_size with type-specific semantics
    if not any("geom_size" in s for s in skip_fields):
        newton_size = newton_mjw.geom_size.numpy()
        native_size = native_mjw.geom_size.numpy()
        for w in range(newton_size.shape[0]):
            for g in range(ngeom):
                gtype = newton_type[g]
                n_sz = newton_size[w, g]
                nat_sz = native_size[w, newton_to_native[g]]
                if gtype == GEOM_PLANE:
                    continue
                elif gtype == 2:  # SPHERE
                    np.testing.assert_allclose(
                        n_sz[0],
                        nat_sz[0],
                        atol=tol,
                        rtol=0,
                        err_msg=f"geom_size[{w},{g}] (SPHERE) radius",
                    )
                elif gtype in (3, 5):  # CAPSULE, CYLINDER
                    np.testing.assert_allclose(
                        n_sz[:2],
                        nat_sz[:2],
                        atol=tol,
                        rtol=0,
                        err_msg=f"geom_size[{w},{g}] (CAPSULE/CYLINDER)",
                    )
                else:
                    np.testing.assert_allclose(
                        n_sz,
                        nat_sz,
                        atol=tol,
                        rtol=0,
                        err_msg=f"geom_size[{w},{g}] (type={gtype})",
                    )


def compare_site_sizes(
    newton_mjw: Any,
    native_mjw: Any,
    tol: float = 1e-6,
) -> None:
    """Compare site_size with type-specific semantics.

    MuJoCo stores 3 floats per site in site_size, but only a subset is
    meaningful depending on the site type:
        - Sphere (2): only radius (size[0]).
        - Capsule (3), Cylinder (5): radius and half-length (size[:2]).
        - Box (6): all 3 half-extents.
    """
    nsite = newton_mjw.nsite
    if nsite == 0:
        return
    assert nsite == native_mjw.nsite, f"nsite mismatch: newton={nsite} vs native={native_mjw.nsite}"

    newton_type = newton_mjw.site_type.numpy()
    newton_size = newton_mjw.site_size.numpy()
    native_size = native_mjw.site_size.numpy()

    # Flatten type to 1D: may be (nsite,) or (nworld, nsite)
    if newton_type.ndim == 2:
        newton_type = newton_type[0]

    # Normalize size to 3D (nworld, nsite, 3): may be (nsite, 3) or (nworld, nsite, 3)
    if newton_size.ndim == 2:
        newton_size = newton_size[np.newaxis]
        native_size = native_size[np.newaxis]

    for w in range(newton_size.shape[0]):
        for s in range(nsite):
            stype = newton_type[s]
            n_sz = newton_size[w, s]
            nat_sz = native_size[w, s]
            if stype == 2:  # SPHERE
                np.testing.assert_allclose(
                    n_sz[0],
                    nat_sz[0],
                    atol=tol,
                    rtol=0,
                    err_msg=f"site_size[{w},{s}] (SPHERE) radius",
                )
            elif stype in (3, 5):  # CAPSULE, CYLINDER
                np.testing.assert_allclose(
                    n_sz[:2],
                    nat_sz[:2],
                    atol=tol,
                    rtol=0,
                    err_msg=f"site_size[{w},{s}] (CAPSULE/CYLINDER)",
                )
            else:
                np.testing.assert_allclose(
                    n_sz,
                    nat_sz,
                    atol=tol,
                    rtol=0,
                    err_msg=f"site_size[{w},{s}] (type={stype})",
                )


def compare_jnt_range(
    newton_mjw: Any,
    native_mjw: Any,
    tol: float = 1e-6,
) -> None:
    """Compare jnt_range only for limited joints.

    MuJoCo ignores jnt_range when jnt_limited=False, so unlimited joints
    may have different range values (Newton uses [-1e10, 1e10], MuJoCo
    uses [0, 0]) without affecting physics. Only compare range values
    for joints where both sides agree the joint is limited.
    """
    newton_range = newton_mjw.jnt_range.numpy()
    native_range = native_mjw.jnt_range.numpy()
    newton_limited = newton_mjw.jnt_limited.numpy()
    native_limited = native_mjw.jnt_limited.numpy()

    assert newton_range.shape == native_range.shape, (
        f"jnt_range shape mismatch: {newton_range.shape} vs {native_range.shape}"
    )
    np.testing.assert_array_equal(newton_limited, native_limited, err_msg="jnt_limited mismatch")

    for world in range(newton_range.shape[0]):
        for jnt in range(newton_range.shape[1]):
            if native_limited[jnt]:
                np.testing.assert_allclose(
                    newton_range[world, jnt],
                    native_range[world, jnt],
                    atol=tol,
                    rtol=0,
                    err_msg=f"jnt_range[{world},{jnt}] mismatch (limited joint)",
                )


# =============================================================================
# Forward Kinematics Helpers
# =============================================================================


@wp.func
def _quat_xyzw_to_wxyz(q: wp.quat) -> wp.quat:
    return wp.quat(q[3], q[0], q[1], q[2])


@wp.kernel
def _copy_body_q_to_mjwarp_kernel(
    mjc_body_to_newton: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    # outputs
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
):
    """Copy Newton body_q transforms into mjwarp xpos/xquat arrays."""
    world, mjc_body = wp.tid()
    newton_body = mjc_body_to_newton[world, mjc_body]
    if newton_body >= 0:
        t = body_q[newton_body]
        xpos[world, mjc_body] = wp.transform_get_translation(t)
        xquat[world, mjc_body] = _quat_xyzw_to_wxyz(wp.transform_get_rotation(t))


def run_newton_eval_fk(solver: SolverMuJoCo, model: newton.Model, state: newton.State):
    """Run Newton's FK and copy results into the solver's mjwarp data."""
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    nworld = solver.mjc_body_to_newton.shape[0]
    nbody = solver.mjc_body_to_newton.shape[1]
    wp.launch(
        _copy_body_q_to_mjwarp_kernel,
        dim=(nworld, nbody),
        inputs=[solver.mjc_body_to_newton, state.body_q],
        outputs=[solver.mjw_data.xpos, solver.mjw_data.xquat],
        device=model.device,
    )


def compare_mjdata_field(
    newton_mjw_data: Any,
    native_mjw_data: Any,
    field_name: str,
    tol: float,
    step: int,
) -> None:
    """
    Compare a single MjWarpData field using numpy.

    Fails immediately with detailed info on first mismatch.
    """
    newton_arr = getattr(newton_mjw_data, field_name, None)
    native_arr = getattr(native_mjw_data, field_name, None)

    if newton_arr is None and native_arr is None:
        raise AssertionError(
            f"Step {step}, field '{field_name}': not found on either side (check for typos in compare_fields)"
        )
    if newton_arr is None:
        raise AssertionError(f"Step {step}, field '{field_name}': missing on Newton side but present on native")
    if native_arr is None:
        raise AssertionError(f"Step {step}, field '{field_name}': missing on native side but present on Newton")

    if newton_arr.size == 0:
        return

    # Sync and copy to numpy
    newton_np = newton_arr.numpy()
    native_np = native_arr.numpy()

    # Skip world body (index 0 on body axis) for cfrc_int and cacc —
    # mujoco_warp's _cfrc_backward accumulates child forces into the world
    # body without zeroing it first, causing stale values across rne calls.
    # MuJoCo C's mj_rne uses local arrays and never writes d->cfrc_int,
    # and cacc[world] is only meaningful as the gravity seed, not as output.
    if field_name in ("cfrc_int", "cacc") and newton_np.ndim >= 2:
        newton_np = newton_np[:, 1:]
        native_np = native_np[:, 1:]

    # Quaternion sign handling: q and -q represent the same rotation.
    # Pick one sign per quaternion (not per component) to avoid mixing branches.
    if newton_arr.dtype == wp.quat:
        direct = np.abs(newton_np - native_np)
        flipped = np.abs(newton_np + native_np)
        use_flipped = np.max(flipped, axis=-1, keepdims=True) < np.max(direct, axis=-1, keepdims=True)
        diff = np.where(use_flipped, flipped, direct)
    else:
        diff = np.abs(newton_np - native_np)
    max_diff = float(np.max(diff))

    if np.isnan(max_diff):
        raise AssertionError(f"Step {step}, field '{field_name}': diff contains NaN")

    if max_diff > tol:
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        newton_val = float(newton_np[max_idx])
        native_val = float(native_np[max_idx])

        raise AssertionError(
            f"Step {step}, field '{field_name}': max diff {max_diff:.6e} > tol {tol:.6e}\n"
            f"  at index {max_idx}: newton={newton_val:.6e}, native={native_val:.6e}"
        )


# Fields in MjWarpModel.opt with (nworld, ...) dimension that can be batched.
# From mujoco_warp/_src/types.py Option class: fields marked with array("*", ...)
MJWARP_OPT_BATCHED_FIELDS: list[str] = [
    "timestep",
    "tolerance",
    "ls_tolerance",
    "ccd_tolerance",
    "density",
    "viscosity",
    "gravity",
    "wind",
    "magnetic",
    "impratio_invsqrt",
]

# Fields in MjWarpModel with (nworld, ...) dimension that can be batched/randomized.
# From mujoco_warp/_src/types.py: fields marked with (*, ...) in their dimension specs.
MJWARP_MODEL_BATCHED_FIELDS: list[str] = [
    # qpos
    "qpos0",
    "qpos_spring",
    # body
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "body_mass",
    "body_subtreemass",
    "body_inertia",
    "body_invweight0",
    "body_gravcomp",
    # joint
    "jnt_solref",
    "jnt_solimp",
    "jnt_pos",
    "jnt_axis",
    "jnt_stiffness",
    "jnt_range",
    "jnt_actfrcrange",
    "jnt_margin",
    # dof
    "dof_solref",
    "dof_solimp",
    "dof_frictionloss",
    "dof_armature",
    "dof_damping",
    "dof_invweight0",
    # geom
    "geom_matid",
    "geom_solmix",
    "geom_solref",
    "geom_solimp",
    "geom_size",
    "geom_aabb",
    "geom_rbound",
    "geom_pos",
    "geom_quat",
    "geom_friction",
    "geom_margin",
    "geom_gap",
    "geom_rgba",
    # site
    "site_pos",
    "site_quat",
    # camera
    "cam_pos",
    "cam_quat",
    "cam_poscom0",
    "cam_pos0",
    "cam_mat0",
    # light
    "light_type",
    "light_castshadow",
    "light_active",
    "light_pos",
    "light_dir",
    "light_poscom0",
    "light_pos0",
    "light_dir0",
    # material
    "mat_texrepeat",
    "mat_rgba",
    # pair
    "pair_solref",
    "pair_solreffriction",
    "pair_solimp",
    "pair_margin",
    "pair_gap",
    "pair_friction",
    # equality constraint
    "eq_solref",
    "eq_solimp",
    "eq_data",
    # tendon
    "tendon_solref_lim",
    "tendon_solimp_lim",
    "tendon_solref_fri",
    "tendon_solimp_fri",
    "tendon_range",
    "tendon_actfrcrange",
    "tendon_margin",
    "tendon_stiffness",
    "tendon_damping",
    "tendon_armature",
    "tendon_frictionloss",
    "tendon_lengthspring",
    "tendon_length0",
    "tendon_invweight0",
    # actuator
    "actuator_dynprm",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "actuator_actrange",
    "actuator_gear",
    "actuator_cranklength",
    "actuator_acc0",
    "actuator_lengthrange",
]


def _expand_batched_fields(target_obj: Any, reference_obj: Any, field_names: list[str]) -> None:
    """Helper to expand batched fields in target to match reference shapes."""
    for field_name in field_names:
        ref_arr = getattr(reference_obj, field_name, None)
        tgt_arr = getattr(target_obj, field_name, None)

        if ref_arr is None or tgt_arr is None:
            continue
        if not hasattr(ref_arr, "numpy") or not hasattr(tgt_arr, "numpy"):
            continue

        ref_nworld = ref_arr.shape[0]
        tgt_nworld = tgt_arr.shape[0]

        # Only expand if reference has more worlds than target
        if ref_nworld > tgt_nworld and tgt_nworld == 1:
            # Tile to match reference: (1, ...) -> (ref_nworld, ...)
            arr_np = tgt_arr.numpy()
            tiled = np.tile(arr_np, (ref_nworld,) + (1,) * (arr_np.ndim - 1))
            new_arr = wp.array(tiled, dtype=tgt_arr.dtype, device=tgt_arr.device)
            setattr(target_obj, field_name, new_arr)


# Model fields to backfill from native MuJoCo to eliminate compilation differences:
# - body_inertia, body_iquat: Newton re-diagonalizes inertia (different eig3 ordering)
# - body_invweight0: Derived from inertia, used in make_constraint for efc_D scaling
# - body_pos, body_quat: Newton recomputes from joint transforms (~3e-8 float diff)
MODEL_BACKFILL_FIELDS: list[str] = [
    "body_inertia",
    "body_iquat",
    "body_invweight0",
    "dof_invweight0",
    "body_pos",
    "body_quat",
    "body_subtreemass",
    "actuator_acc0",
]


def expand_mjw_model_to_match(target_mjw: Any, reference_mjw: Any) -> None:
    """Expand batched fields in target MjWarpModel to match reference model's shapes.

    mujoco_warp.put_model() creates arrays with nworld=1 by default, using
    modulo indexing for batch access. This function tiles target arrays to
    match the reference model's nworld dimension where the reference has
    already been expanded.

    Args:
        target_mjw: The model to expand (typically native mujoco_warp)
        reference_mjw: The reference model (typically Newton's mjw_model)
    """
    # Expand main model fields
    _expand_batched_fields(target_mjw, reference_mjw, MJWARP_MODEL_BATCHED_FIELDS)

    # Expand opt fields (nested Option object)
    if hasattr(target_mjw, "opt") and hasattr(reference_mjw, "opt"):
        _expand_batched_fields(target_mjw.opt, reference_mjw.opt, MJWARP_OPT_BATCHED_FIELDS)


def backfill_model_from_native(
    newton_mjw: Any,
    native_mjw: Any,
    fields: list[str] | None = None,
) -> None:
    """Copy computed model fields from native MuJoCo to Newton's mjw_model.

    This eliminates numerical differences caused by Newton's model compilation
    differing from MuJoCo's mj_setConst(). Useful for isolating simulation
    differences from model compilation differences during testing.

    Validation of these fields is handled by compare_compiled_model_fields().

    Args:
        newton_mjw: Newton's MjWarpModel to update
        native_mjw: Native MuJoCo's MjWarpModel to copy from
        fields: List of field names to copy (defaults to MODEL_BACKFILL_FIELDS)
    """
    if fields is None:
        fields = MODEL_BACKFILL_FIELDS

    for field in fields:
        native_arr = getattr(native_mjw, field, None)
        newton_arr = getattr(newton_mjw, field, None)

        if native_arr is None or newton_arr is None:
            continue
        if not hasattr(native_arr, "numpy") or not hasattr(newton_arr, "numpy"):
            continue

        if native_arr.shape == newton_arr.shape:
            newton_arr.assign(native_arr)


def compare_mjw_models(
    newton_mjw: Any,
    native_mjw: Any,
    skip_fields: set[str] | None = None,
    tol: float = 1e-6,
) -> None:
    """Compare ALL fields of two MjWarpModel objects. Asserts on first mismatch."""
    if skip_fields is None:
        skip_fields = {"__", "ptr"}

    for attr in dir(native_mjw):
        if any(s in attr for s in skip_fields):
            continue

        native_val = getattr(native_mjw, attr, None)
        newton_val = getattr(newton_mjw, attr, None)

        if callable(native_val) or (native_val is None and newton_val is None):
            continue

        # Handle tuples of warp arrays (e.g., body_tree)
        if isinstance(native_val, tuple) and len(native_val) > 0 and hasattr(native_val[0], "numpy"):
            assert isinstance(newton_val, tuple), f"{attr}: type mismatch (expected tuple)"
            assert len(native_val) == len(newton_val), f"{attr}: tuple length {len(newton_val)} != {len(native_val)}"
            for i, (nv, ntv) in enumerate(zip(native_val, newton_val, strict=True)):
                native_np: np.ndarray = nv.numpy()
                newton_np: np.ndarray = ntv.numpy()
                assert native_np.shape == newton_np.shape, f"{attr}[{i}]: shape {newton_np.shape} != {native_np.shape}"
                if native_np.size > 0:
                    np.testing.assert_allclose(newton_np, native_np, rtol=tol, atol=tol, err_msg=f"{attr}[{i}]")
        # Handle warp arrays (have .numpy() method)
        elif hasattr(native_val, "numpy"):
            assert newton_val is not None and hasattr(newton_val, "numpy"), f"{attr}: type mismatch"
            native_np = native_val.numpy()  # type: ignore[union-attr]
            newton_np = newton_val.numpy()  # type: ignore[union-attr]
            assert native_np.shape == newton_np.shape, f"{attr}: shape {newton_np.shape} != {native_np.shape}"
            if native_np.size > 0:
                np.testing.assert_allclose(newton_np, native_np, rtol=tol, atol=tol, err_msg=attr)
        elif isinstance(native_val, np.ndarray):
            assert isinstance(newton_val, np.ndarray), f"{attr}: type mismatch"
            assert native_val.shape == newton_val.shape, f"{attr}: shape {newton_val.shape} != {native_val.shape}"
            if native_val.size > 0:
                np.testing.assert_allclose(newton_val, native_val, rtol=tol, atol=tol, err_msg=attr)
        elif isinstance(native_val, (int, float, np.number)):
            assert newton_val is not None, f"{attr}: newton is None"
            assert abs(float(newton_val) - float(native_val)) < tol, f"{attr}: {newton_val} != {native_val}"
        elif attr == "stat" and hasattr(native_val, "meaninertia"):
            # Special case: Statistic object - compare meaninertia with tolerance
            assert hasattr(newton_val, "meaninertia"), f"{attr}: newton missing meaninertia"
            newton_mi = newton_val.meaninertia
            native_mi = native_val.meaninertia
            # Handle both scalar and array cases
            if hasattr(newton_mi, "numpy"):
                newton_mi = newton_mi.numpy()
            if hasattr(native_mi, "numpy"):
                native_mi = native_mi.numpy()
            diff = np.max(np.abs(np.asarray(newton_mi) - np.asarray(native_mi)))
            assert diff < tol, f"{attr}.meaninertia: diff={diff:.2e} > tol={tol:.0e}"
        elif attr == "opt":
            # Special case: Option object - compare each field
            for opt_attr in dir(native_val):
                if opt_attr.startswith("_"):
                    continue
                # Check if this opt sub-field should be skipped
                opt_full_name = f"opt.{opt_attr}"
                if any(skip in opt_full_name for skip in skip_fields):
                    continue
                opt_newton = getattr(newton_val, opt_attr, None)
                opt_native = getattr(native_val, opt_attr, None)
                if opt_newton is None or opt_native is None or callable(opt_native):
                    continue
                if hasattr(opt_native, "numpy"):
                    np.testing.assert_allclose(
                        opt_newton.numpy(),
                        opt_native.numpy(),
                        rtol=tol,
                        atol=tol,
                        err_msg=f"{attr}.{opt_attr}",
                    )
                elif isinstance(opt_native, (int, float, np.number, bool)):
                    assert opt_newton == opt_native, f"{attr}.{opt_attr}: {opt_newton} != {opt_native}"
                # Skip enum comparisons (they compare fine by value)
        else:
            assert newton_val == native_val, f"{attr}: {newton_val} != {native_val}"


# =============================================================================
# Randomization (placeholder for future implementation)
# =============================================================================


def apply_randomization(
    newton_model: newton.Model,
    mj_solver: SolverMuJoCo,
    seed: int = 42,
    mass_scale: tuple[float, float] | None = None,
    friction_range: tuple[float, float] | None = None,
    damping_scale: tuple[float, float] | None = None,
    armature_scale: tuple[float, float] | None = None,
) -> None:
    """
    Apply randomized properties to both Newton model and MuJoCo solver.

    Uses the SolverMuJoCo remappings to ensure both sides get identical values.

    Args:
        newton_model: Newton model to randomize
        mj_solver: MuJoCo solver (uses its remappings)
        seed: Random seed
        mass_scale: Scale range for masses, e.g., (0.8, 1.2)
        friction_range: Range for friction coefficients, e.g., (0.3, 1.0)
        damping_scale: Scale range for damping, e.g., (0.5, 2.0)
        armature_scale: Scale range for armature, e.g., (0.5, 2.0)
    """
    # Skip if no randomization requested
    if all(x is None for x in [mass_scale, friction_range, damping_scale, armature_scale]):
        return

    raise NotImplementedError(
        "Randomization requires SolverMuJoCo remappings to ensure both sides receive identical randomized values"
    )


# =============================================================================
# Base Test Class
# =============================================================================


@unittest.skipIf(not MUJOCO_AVAILABLE, "mujoco/mujoco_warp not installed")
class TestMenagerieBase(unittest.TestCase):
    """
    Base class for MuJoCo Menagerie integration tests.

    Subclasses must define:
        - robot_folder: str - menagerie folder name
        - robot_xml: str - path to XML within folder

    Optional overrides:
        - num_worlds: int - number of parallel worlds (default: 34)
        - num_steps: int - dynamics steps to run (default: 0, dynamics disabled)
        - dynamics_target: float - step-response target position offset (default: 0.3)
        - dynamics_tolerance: float - qpos/qvel comparison tolerance (default: 1e-6)
        - skip_reason: str | None - if set, skip this test
    """

    # Must be defined by subclasses
    robot_folder: str = ""
    robot_xml: str = "scene.xml"  # Default; most menagerie robots use scene.xml

    # Configurable defaults
    num_worlds: int = 34  # One warp per GPU warp lane (32) + 2 extra to test non-power-of-2
    num_steps: int = 0  # Dynamics steps (0 = dynamics test skipped)
    dt: float = 0.002  # Fallback; actual dt extracted from native model in test

    # Dynamics test: step-response per DOF. Each world commands one actuator to
    # a target position (wrapping with modulo). Collisions disabled.
    dynamics_target: float = 0.3  # Position offset for step-response target
    dynamics_tolerance: float = 1e-6  # Tolerance for qpos/qvel comparison

    # Model comparison: fields to SKIP (substrings to match)
    # Override in subclass with: model_skip_fields = DEFAULT_MODEL_SKIP_FIELDS | {"extra", "fields"}
    model_skip_fields: ClassVar[set[str]] = DEFAULT_MODEL_SKIP_FIELDS

    # Skip reason (set to a string to skip test, leave unset or None to run)
    skip_reason: str | None = None

    # Skip visual-only geoms on the native side via compiler discardvisual="true".
    # Note: discardvisual may also strip collision geoms that have contype=conaffinity=0
    # and are not referenced in <pair>/<exclude>/sensor elements (seen with Apollo).
    # For such models, set parse_visuals=True and discard_visual=False instead.
    discard_visual: bool = True

    # Parse visual geoms in Newton. When True, Newton includes visual geoms so both
    # sides can be compared without discardvisual. Set discard_visual=False alongside.
    parse_visuals: bool = False

    # Backfill computed model fields from native to eliminate compilation diffs.
    # See MODEL_BACKFILL_FIELDS for the default set; override backfill_fields per-robot.
    backfill_model: bool = False
    backfill_fields: list[str] | None = None  # None = use MODEL_BACKFILL_FIELDS

    njmax: int | None = None  # Max constraint rows per world (None = auto from MuJoCo)
    nconmax: int | None = None  # Max contacts per world (None = auto from MuJoCo)
    # Override integrator for SolverMuJoCo
    solver_integrator: str | int | None = None
    # Forward kinematics test: compare body poses computed from joint positions.
    # No forces, no contacts, fully deterministic.
    # Set to True per robot to enable test_forward_kinematics().
    fk_enabled: bool = False
    fk_fields: ClassVar[list[str]] = DEFAULT_FK_FIELDS
    fk_tolerance: float = 2e-6

    @classmethod
    def setUpClass(cls):
        """Download assets once for all tests in this class."""
        if cls.skip_reason:
            raise unittest.SkipTest(cls.skip_reason)

        if not cls.robot_folder:
            raise unittest.SkipTest("robot_folder not defined")

        # Download the robot assets
        try:
            cls.asset_path = download_menagerie_asset(cls.robot_folder)
        except (OSError, TimeoutError, ConnectionError) as e:
            raise unittest.SkipTest(f"Failed to download {cls.robot_folder}: {e}") from e

        cls.mjcf_path = cls.asset_path / cls.robot_xml
        if not cls.mjcf_path.exists():
            raise unittest.SkipTest(f"MJCF file not found: {cls.mjcf_path}")

    @abstractmethod
    def _create_newton_model(self) -> newton.Model:
        """Create Newton model from the source (MJCF or USD).

        Subclasses must implement this to define how Newton loads the model:
        - TestMenagerieMJCF: Load directly from MJCF
        - TestMenagerieUSD: Convert MJCF to USD, then load USD

        Note: The native MuJoCo comparison always loads from MJCF (ground truth).
        See _create_native_mujoco_warp() which is shared by all subclasses.
        """
        ...

    def _align_models(self, newton_solver: SolverMuJoCo, native_mjw_model: Any, mj_model: Any) -> None:
        """Hook for subclass-specific model alignment before comparison.

        Called after both models are built and expanded but before
        compare_mjw_models. Override in subclasses to fix up known
        discrepancies between the model sources.
        """

    def _compare_inertia(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare inertia tensors between Newton and native models.

        Default: no-op (covered by compare_models for same-order pipelines).
        Override in subclasses where body ordering may differ.
        """

    def _compare_geoms(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare geom fields between Newton and native models.

        Default: no-op (covered by compare_models for same-order pipelines).
        Override in subclasses where geom ordering may differ.
        """

    def _compare_jnt_range(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare joint ranges between Newton and native models.

        Default: no-op (covered by compare_models for same-order pipelines).
        Override in subclasses where joint ordering may differ.
        """

    def _compare_body_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare physics-relevant body fields (mass, pos, quat, etc.).

        Default: no-op (covered by compare_mjw_models for same-order pipelines).
        Override in subclasses where body ordering may differ.
        """

    def _compare_dof_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare physics-relevant DOF fields (armature, damping, etc.).

        Default: no-op (covered by compare_mjw_models for same-order pipelines).
        Override in subclasses where DOF ordering may differ.
        """

    def _compare_mass_matrix_structure(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare sparse mass matrix structure (M_colind, M_rowadr, M_rownnz).

        Default: no-op (covered by compare_mjw_models for same-order pipelines).
        Override in subclasses where DOF ordering may differ.
        """

    def _compare_tendon_jacobian_structure(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare sparse tendon Jacobian structure (ten_J_colind, ten_J_rowadr, ten_J_rownnz).

        Default: no-op (covered by compare_mjw_models for same-order pipelines).
        Override in subclasses where DOF ordering may differ.
        """

    def _compare_compiled_fields(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare compilation-dependent fields at relaxed tolerance.

        Default: no-op (covered by compare_models for same-order pipelines).
        Override in subclasses to skip or adjust.
        """

    def _compare_actuator_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        """Compare actuator fields (gainprm, biasprm, acc0, gear, etc.).

        Default: no-op (covered by compare_mjw_models for same-order pipelines).
        Override in subclasses where actuator ordering may differ.
        """

    def _load_assets(self) -> dict[str, bytes]:
        """Load mesh/texture assets from the MJCF directory for from_xml_string."""
        assets: dict[str, bytes] = {}
        asset_dir = self.mjcf_path.parent

        # Common mesh and texture extensions
        mesh_extensions = (".stl", ".obj", ".msh", ".STL", ".OBJ", ".MSH")
        texture_extensions = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

        for ext in mesh_extensions + texture_extensions:
            for filepath in asset_dir.rglob(f"*{ext}"):
                # Use relative path from asset_dir as the key
                rel_path = filepath.relative_to(asset_dir)
                with open(filepath, "rb") as f:
                    assets[str(rel_path)] = f.read()

        return assets

    def _get_mjcf_xml(self) -> str:
        """Get MJCF XML content with includes expanded and optional compiler modifications.

        Uses Newton's include processor to expand <include> elements, then optionally
        inserts <compiler discardvisual="true"/> to make MuJoCo discard visual-only geoms.
        """
        import xml.etree.ElementTree as ET  # noqa: PLC0415

        # Use Newton's include processor to expand all includes
        root, _ = _load_and_expand_mjcf(str(self.mjcf_path))
        xml_content = ET.tostring(root, encoding="unicode")

        if self.discard_visual:
            compiler = root.find("compiler")
            if compiler is None:
                compiler = ET.SubElement(root, "compiler")
            compiler.set("discardvisual", "true")
            xml_content = ET.tostring(root, encoding="unicode")

        return xml_content

    def _create_native_mujoco_warp(self) -> tuple[Any, Any, Any, Any]:
        """Create native mujoco_warp model/data from the same MJCF.

        Returns:
            (mj_model, mj_data, mjw_model, mjw_data) tuple
        """
        assert _mujoco is not None
        assert _mujoco_warp is not None

        # Create base MuJoCo model/data (uses default initialization)
        if self.discard_visual:
            xml_content = self._get_mjcf_xml()
            # from_xml_string needs the assets path for meshes
            mj_model = _mujoco.MjModel.from_xml_string(xml_content, assets=self._load_assets())
        else:
            mj_model = _mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        mj_data = _mujoco.MjData(mj_model)
        _mujoco.mj_forward(mj_model, mj_data)

        # Create mujoco_warp model/data with multiple worlds
        # Note: put_model creates arrays with nworld=1, expansion happens in _ensure_models
        mjw_model = _mujoco_warp.put_model(mj_model)
        mjw_data = _mujoco_warp.put_data(
            mj_model, mj_data, nworld=self.num_worlds, njmax=self.njmax, nconmax=self.nconmax
        )

        return mj_model, mj_data, mjw_model, mjw_data

    def _ensure_models(self):
        """Create Newton and native models if not already done (lazy init).

        Stores models on the **class** so all test method instances share
        them without re-creating expensive GPU resources.
        """
        cls = self.__class__
        if hasattr(cls, "_newton_solver"):
            return

        assert _mujoco is not None
        assert _mujoco_warp is not None

        # Create models and solvers — stored on cls for reuse across test methods
        cls._newton_model = self._create_newton_model()
        cls._newton_state = cls._newton_model.state()
        cls._newton_control = cls._newton_model.control()
        solver_kwargs = {
            "skip_visual_only_geoms": not self.parse_visuals,
            "njmax": self.njmax,
            "nconmax": self.nconmax,
        }
        if self.solver_integrator is not None:
            solver_kwargs["integrator"] = self.solver_integrator

        cls._newton_solver = SolverMuJoCo(cls._newton_model, **solver_kwargs)

        cls._mj_model, cls._mj_data_native, cls._native_mjw_model, cls._native_mjw_data = (
            self._create_native_mujoco_warp()
        )

        # Expand native model's batched arrays to match Newton's shapes
        # Newton is the reference - only expand fields that Newton has expanded
        expand_mjw_model_to_match(cls._native_mjw_model, cls._newton_solver.mjw_model)

        # Extract timestep from native model (Newton doesn't parse <option timestep="..."/> yet)
        # TODO: Remove this workaround once Newton's MJCF parser supports timestep extraction
        cls._dt = float(cls._mj_model.opt.timestep)

        # Hook for subclass-specific model alignment (USD fixups, etc.)
        self._align_models(cls._newton_solver, cls._native_mjw_model, cls._mj_model)

        # Disable sensor_rne_postconstraint on native — Newton doesn't support
        # sensors, so rne_postconstraint would compute cacc/cfrc_int on native
        # but not on Newton, causing spurious diffs.
        cls._native_mjw_model.sensor_rne_postconstraint = False

    def _run_model_comparisons(self):
        """Run all model field comparisons and subclass hooks."""
        compare_models(
            self._newton_solver.mjw_model,
            self._native_mjw_model,
            skip_fields=self.model_skip_fields,
            backfill_fields=self.backfill_fields,
        )

        # Subclass hooks for pipelines with reordered bodies/DOFs/actuators (USD).
        # Default implementations are no-ops; compare_models already covers the
        # same-order case.
        self._compare_inertia(self._newton_solver.mjw_model, self._native_mjw_model)
        self._compare_geoms(self._newton_solver.mjw_model, self._native_mjw_model)
        self._compare_jnt_range(self._newton_solver.mjw_model, self._native_mjw_model)
        self._compare_body_physics(self._newton_solver.mjw_model, self._native_mjw_model)
        self._compare_dof_physics(self._newton_solver.mjw_model, self._native_mjw_model)
        self._compare_mass_matrix_structure(self._newton_solver.mjw_model, self._native_mjw_model)
        self._compare_tendon_jacobian_structure(self._newton_solver.mjw_model, self._native_mjw_model)
        self._compare_actuator_physics(self._newton_solver.mjw_model, self._native_mjw_model)
        self._compare_compiled_fields(self._newton_solver.mjw_model, self._native_mjw_model)

    def _backfill_and_recompute(self):
        """Backfill computed model fields from native and re-run kinematics/RNE."""
        if not self.backfill_model:
            return
        backfill_model_from_native(self._newton_solver.mjw_model, self._native_mjw_model, self.backfill_fields)
        # Re-run kinematics and RNE (without collision) so data fields
        # (qfrc_bias, qM, etc.) reflect backfilled model. The initial forward
        # ran before backfill and produced stale values.
        from mujoco_warp._src import smooth as mjw_smooth

        mjw_smooth.kinematics(self._newton_solver.mjw_model, self._newton_solver.mjw_data)
        mjw_smooth.com_pos(self._newton_solver.mjw_model, self._newton_solver.mjw_data)
        mjw_smooth.crb(self._newton_solver.mjw_model, self._newton_solver.mjw_data)
        mjw_smooth.factor_m(self._newton_solver.mjw_model, self._newton_solver.mjw_data)
        mjw_smooth.com_vel(self._newton_solver.mjw_model, self._newton_solver.mjw_data)
        mjw_smooth.rne(self._newton_solver.mjw_model, self._newton_solver.mjw_data)

    def test_model_comparison(self):
        """Verify Newton and native mujoco_warp models have equivalent fields.

        Deterministic — compares model parameters, inertia tensors, solref,
        geoms, joint ranges, compiled fields, and actuator physics. No simulation.
        """
        self._ensure_models()
        self._run_model_comparisons()

    def test_forward_kinematics(self):
        """Verify forward kinematics produce equivalent body poses.

        Computes body positions and orientations from joint positions on both
        sides (no forces, no contacts) and compares. Fully deterministic.
        """
        if not self.fk_enabled:
            self.skipTest("Forward kinematics not enabled for this robot")

        self._ensure_models()
        self._run_model_comparisons()
        self._backfill_and_recompute()

        from mujoco_warp._src import smooth as mjw_smooth

        model = self._newton_model
        solver = self._newton_solver

        # Use a local state so we don't mutate shared state used by other tests
        state = model.state()

        # Perturb joint positions so FK has something to compute
        rng = np.random.default_rng(seed=42)
        joint_q_np = model.joint_q.numpy()
        joint_q_np += rng.uniform(-0.1, 0.1, size=joint_q_np.shape).astype(np.float32)

        # Renormalize quaternions for free and ball joints (perturbation
        # denormalizes them, which is invalid input for eval_fk)
        joint_type = model.joint_type.numpy()
        q_start = model.joint_q_start.numpy()
        for j in range(len(joint_type)):
            jt = joint_type[j]
            qi = q_start[j]
            if jt == JointType.FREE:
                q = joint_q_np[qi + 3 : qi + 7]
                q /= np.linalg.norm(q)
            elif jt == JointType.BALL:
                q = joint_q_np[qi : qi + 4]
                q /= np.linalg.norm(q)

        state.joint_q.assign(joint_q_np)

        # Sync perturbed joints to Newton's mjwarp qpos
        solver._update_mjc_data(solver.mjw_data, model, state)

        # Copy the same qpos to native so both sides start from identical joint positions
        self._native_mjw_data.qpos.assign(solver.mjw_data.qpos.numpy())

        # Newton side: eval_fk → copy body poses to mjwarp data
        run_newton_eval_fk(solver, model, state)

        # Native side: mjwarp kinematics
        mjw_smooth.kinematics(self._native_mjw_model, self._native_mjw_data)
        mjw_smooth.com_pos(self._native_mjw_model, self._native_mjw_data)

        # Compare FK fields
        for field_name in self.fk_fields:
            compare_mjdata_field(solver.mjw_data, self._native_mjw_data, field_name, self.fk_tolerance, step=-1)

    def test_dynamics(self):
        """Verify per-DOF step response matches between Newton and native mjwarp.

        Each world commands one actuator to a target position (wrapping with
        modulo). Collisions disabled to isolate joint dynamics. Uses split
        pipeline for deterministic comparison.
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

        # Disable collisions on both sides (save/restore to avoid mutating cached model)
        newton_saved = _disable_collisions(newton_solver.mjw_model)
        native_saved = _disable_collisions(native_mjw_model)

        try:
            # Initialize step-response control
            strategy = StepResponseControlStrategy(target=self.dynamics_target)
            strategy.init(native_mjw_data.ctrl, newton_control.mujoco.ctrl)
            strategy.fill_control(0.0)

            # Step loop — both sides run full mujoco_warp.step() independently.
            # Both sides run full mujoco_warp.step() — no contacts with collisions disabled.
            for step in range(self.num_steps):
                strategy.fill_control(step * dt)
                newton_solver.step(newton_state, newton_state, newton_control, None, dt)
                _mujoco_warp.step(native_mjw_model, native_mjw_data)

                # Compare qpos and qvel
                compare_mjdata_field(newton_solver.mjw_data, native_mjw_data, "qpos", self.dynamics_tolerance, step)
                compare_mjdata_field(newton_solver.mjw_data, native_mjw_data, "qvel", self.dynamics_tolerance, step)
        finally:
            _restore_collisions(newton_solver.mjw_model, newton_saved)
            _restore_collisions(native_mjw_model, native_saved)


def _disable_collisions(mjw_model: Any) -> dict:
    """Disable contact generation and return saved state for restoration.

    Uses mjDSBL_CONTACT plus zeroing contype/conaffinity.
    Returns saved values so _restore_collisions can undo the changes.
    """
    import mujoco

    saved = {
        "disableflags": int(mjw_model.opt.disableflags),
        "contype": mjw_model.geom_contype.numpy().copy(),
        "conaffinity": mjw_model.geom_conaffinity.numpy().copy(),
    }
    mjw_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    contype = mjw_model.geom_contype.numpy()
    contype[:] = 0
    mjw_model.geom_contype.assign(contype)
    conaffinity = mjw_model.geom_conaffinity.numpy()
    conaffinity[:] = 0
    mjw_model.geom_conaffinity.assign(conaffinity)
    return saved


def _restore_collisions(mjw_model: Any, saved: dict) -> None:
    """Restore collision settings from saved state."""
    mjw_model.opt.disableflags = saved["disableflags"]
    mjw_model.geom_contype.assign(saved["contype"])
    mjw_model.geom_conaffinity.assign(saved["conaffinity"])


# =============================================================================
# Model Source Base Classes
# =============================================================================
# These intermediate classes define HOW Newton loads the model.
# The native MuJoCo comparison always loads from MJCF (ground truth).


class TestMenagerieMJCF(TestMenagerieBase):
    """Base class for MJCF-based tests: Newton loads directly from MJCF."""

    def _create_newton_model(self) -> newton.Model:
        """Create Newton model by loading MJCF directly."""
        return create_newton_model_from_mjcf(
            self.mjcf_path,
            num_worlds=self.num_worlds,
            add_ground=False,  # scene.xml includes ground plane
            parse_visuals=self.parse_visuals,
        )


# =============================================================================
# Robot Test Classes
# =============================================================================
# Each robot from the menagerie gets its own test class.
# Initially all are skipped; enable as support is verified.
# Total: 61 robots (excluding test/ folder and realsense_d435i sensor)


# -----------------------------------------------------------------------------
# Arms (14 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_AgilexPiper(TestMenagerieMJCF):
    """AgileX PIPER bimanual arm."""

    robot_folder = "agilex_piper"

    skip_reason = "Not yet implemented"


class TestMenagerie_ArxL5(TestMenagerieMJCF):
    """ARX L5 arm."""

    robot_folder = "arx_l5"

    skip_reason = "Not yet implemented"


class TestMenagerie_Dynamixel2r(TestMenagerieMJCF):
    """Dynamixel 2R simple arm."""

    robot_folder = "dynamixel_2r"

    skip_reason = "Not yet implemented"


class TestMenagerie_FrankaEmikaPanda(TestMenagerieMJCF):
    """Franka Emika Panda arm."""

    robot_folder = "franka_emika_panda"
    num_steps = 20
    dynamics_tolerance = 5e-5
    fk_enabled = True
    backfill_model = True


class TestMenagerie_FrankaFr3(TestMenagerieMJCF):
    """Franka FR3 arm."""

    robot_folder = "franka_fr3"

    skip_reason = "Not yet implemented"


class TestMenagerie_FrankaFr3V2(TestMenagerieMJCF):
    """Franka FR3 v2 arm."""

    robot_folder = "franka_fr3_v2"
    # Dynamics disabled: qpos drift 3.5e-3 from body_ipos diff (#2170)
    num_steps = 0
    fk_enabled = True
    fk_tolerance = 5e-6  # float32 precision (max diff ~1.2e-6)
    backfill_model = True


class TestMenagerie_KinovaGen3(TestMenagerieMJCF):
    """Kinova Gen3 arm."""

    robot_folder = "kinova_gen3"

    skip_reason = "Not yet implemented"


class TestMenagerie_KukaIiwa14(TestMenagerieMJCF):
    """KUKA iiwa 14 arm."""

    robot_folder = "kuka_iiwa_14"

    skip_reason = "Not yet implemented"


class TestMenagerie_LowCostRobotArm(TestMenagerieMJCF):
    """Low-cost robot arm."""

    robot_folder = "low_cost_robot_arm"

    skip_reason = "Not yet implemented"


class TestMenagerie_RethinkSawyer(TestMenagerieMJCF):
    """Rethink Robotics Sawyer arm."""

    robot_folder = "rethink_robotics_sawyer"

    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenVx300s(TestMenagerieMJCF):
    """Trossen Robotics ViperX 300 S arm."""

    robot_folder = "trossen_vx300s"

    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenWx250s(TestMenagerieMJCF):
    """Trossen Robotics WidowX 250 S arm."""

    robot_folder = "trossen_wx250s"

    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenWxai(TestMenagerieMJCF):
    """Trossen Robotics WidowX AI arm."""

    robot_folder = "trossen_wxai"

    skip_reason = "Not yet implemented"


class TestMenagerie_TrsSoArm100(TestMenagerieMJCF):
    """TRS SO-ARM100 arm."""

    robot_folder = "trs_so_arm100"

    skip_reason = "Not yet implemented"


class TestMenagerie_UfactoryLite6(TestMenagerieMJCF):
    """UFACTORY Lite 6 arm."""

    robot_folder = "ufactory_lite6"

    skip_reason = "Not yet implemented"


class TestMenagerie_UfactoryXarm7(TestMenagerieMJCF):
    """UFACTORY xArm 7 arm."""

    robot_folder = "ufactory_xarm7"

    skip_reason = "Not yet implemented"


class TestMenagerie_UniversalRobotsUr5e(TestMenagerieMJCF):
    """Universal Robots UR5e arm."""

    robot_folder = "universal_robots_ur5e"

    num_steps = 20
    backfill_model = True
    fk_enabled = True


class TestMenagerie_UniversalRobotsUr10e(TestMenagerieMJCF):
    """Universal Robots UR10e arm."""

    robot_folder = "universal_robots_ur10e"
    num_steps = 20
    fk_enabled = True
    backfill_model = True


# -----------------------------------------------------------------------------
# Grippers / Hands (9 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_LeapHand(TestMenagerieMJCF):
    """LEAP Hand."""

    robot_folder = "leap_hand"
    robot_xml = "scene_right.xml"
    num_steps = 20
    dynamics_tolerance = 5e-5
    fk_enabled = True
    backfill_model = True


class TestMenagerie_Robotiq2f85(TestMenagerieMJCF):
    """Robotiq 2F-85 gripper."""

    robot_folder = "robotiq_2f85"

    skip_reason = "Not yet implemented"


class TestMenagerie_Robotiq2f85V4(TestMenagerieMJCF):
    """Robotiq 2F-85 gripper v4."""

    robot_folder = "robotiq_2f85_v4"
    skip_reason = "mujoco_warp: implicit integrators and fluid model not implemented"


class TestMenagerie_ShadowDexee(TestMenagerieMJCF):
    """Shadow DEX-EE hand."""

    robot_folder = "shadow_dexee"

    skip_reason = "Not yet implemented"


class TestMenagerie_ShadowHand(TestMenagerieMJCF):
    """Shadow Hand."""

    robot_folder = "shadow_hand"
    robot_xml = "scene_right.xml"
    # Dynamics disabled: tendon_invweight0 diff causes qvel drift 2.3e-5 (#2170)
    num_steps = 0
    fk_enabled = True
    # tendon_invweight0 is compilation-dependent (derived from inertia)
    model_skip_fields = DEFAULT_MODEL_SKIP_FIELDS | {"tendon_invweight0"}
    backfill_model = True


class TestMenagerie_TetheriaAeroHandOpen(TestMenagerieMJCF):
    """Tetheria Aero Hand (open)."""

    robot_folder = "tetheria_aero_hand_open"

    skip_reason = "Not yet implemented"


class TestMenagerie_UmiGripper(TestMenagerieMJCF):
    """UMI Gripper."""

    robot_folder = "umi_gripper"
    skip_reason = "mujoco_warp: implicit integrators and fluid model not implemented"


class TestMenagerie_WonikAllegro(TestMenagerieMJCF):
    """Wonik Allegro Hand."""

    robot_folder = "wonik_allegro"
    robot_xml = "scene_right.xml"
    # Dynamics disabled: body_mass diff causes qpos divergence 3.4e-3 (#2170)
    num_steps = 0
    fk_enabled = True
    # TODO(#2170): body_mass differs — Newton computes different masses for visual geoms
    model_skip_fields = DEFAULT_MODEL_SKIP_FIELDS | {"body_mass"}
    # Step response disabled: body_mass diffs cause constraint mismatch


class TestMenagerie_IitSoftfoot(TestMenagerieMJCF):
    """IIT Softfoot biomechanical gripper."""

    robot_folder = "iit_softfoot"

    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Bimanual Systems (2 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_Aloha(TestMenagerieMJCF):
    """ALOHA bimanual system."""

    robot_folder = "aloha"
    # Dynamics disabled: multiple import issues — dof_damping, eq_, ngeom (#2170)
    num_steps = 0
    fk_enabled = False  # FK fails (xpos diff 0.14) due to import bugs (#2170)
    # TODO(#2170): dof_damping, jnt_range, eq_, ngeom differ
    # jnt_ is broad but needed: compare_jnt_range runs outside model_skip_fields
    model_skip_fields = DEFAULT_MODEL_SKIP_FIELDS | {"dof_damping", "eq_", "neq", "ngeom", "jnt_"}


class TestMenagerie_GoogleRobot(TestMenagerieMJCF):
    """Google Robot (bimanual)."""

    robot_folder = "google_robot"

    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Mobile Manipulators (5 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_HelloRobotStretch(TestMenagerieMJCF):
    """Hello Robot Stretch."""

    robot_folder = "hello_robot_stretch"

    skip_reason = "Not yet implemented"


class TestMenagerie_HelloRobotStretch3(TestMenagerieMJCF):
    """Hello Robot Stretch 3."""

    robot_folder = "hello_robot_stretch_3"

    skip_reason = "Not yet implemented"


class TestMenagerie_PalTiago(TestMenagerieMJCF):
    """PAL Robotics TIAGo."""

    robot_folder = "pal_tiago"

    skip_reason = "Not yet implemented"


class TestMenagerie_PalTiagoDual(TestMenagerieMJCF):
    """PAL Robotics TIAGo Dual."""

    robot_folder = "pal_tiago_dual"

    skip_reason = "Not yet implemented"


class TestMenagerie_StanfordTidybot(TestMenagerieMJCF):
    """Stanford Tidybot mobile manipulator."""

    robot_folder = "stanford_tidybot"

    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Humanoids (10 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_ApptronikApollo(TestMenagerieMJCF):
    """Apptronik Apollo humanoid.

    Apollo uses contype=conaffinity=0 on all geoms (including collision primitives)
    and relies on explicit <pair> elements for contacts. This means discardvisual
    incorrectly strips unreferenced collision geoms. We parse visual geoms on both
    sides to get matching geom counts.

    Kinematic injection is disabled because mujoco_warp's Euler integrator with
    implicit damping uses atomic_add internally (factor_solve_i), so injecting
    kinematic fields breaks the natural float32 noise correlation and increases
    divergence. Without injection, xpos stays within ~5e-5 over 100 steps.
    """

    robot_folder = "apptronik_apollo"
    backfill_model = True
    # Dynamics disabled: qvel divergence 5.8e-5 (model compilation diffs amplified by free joint)
    num_steps = 0
    fk_enabled = True
    njmax = 128  # initial 63 constraints may grow during stepping
    discard_visual = False
    parse_visuals = True


class TestMenagerie_BerkeleyHumanoid(TestMenagerieMJCF):
    """Berkeley Humanoid."""

    robot_folder = "berkeley_humanoid"

    skip_reason = "Not yet implemented"


class TestMenagerie_BoosterT1(TestMenagerieMJCF):
    """Booster Robotics T1 humanoid."""

    robot_folder = "booster_t1"
    num_steps = 20
    dynamics_tolerance = 2e-5  # float32 noise varies when running in batch
    fk_enabled = True
    backfill_model = True


class TestMenagerie_FourierN1(TestMenagerieMJCF):
    """Fourier N1 humanoid."""

    robot_folder = "fourier_n1"

    skip_reason = "Not yet implemented"


class TestMenagerie_PalTalos(TestMenagerieMJCF):
    """PAL Robotics TALOS humanoid."""

    robot_folder = "pal_talos"

    skip_reason = "Not yet implemented"


class TestMenagerie_PndboticsAdamLite(TestMenagerieMJCF):
    """PNDbotics Adam Lite humanoid."""

    robot_folder = "pndbotics_adam_lite"

    skip_reason = "Not yet implemented"


class TestMenagerie_RobotisOp3(TestMenagerieMJCF):
    """Robotis OP3 humanoid."""

    robot_folder = "robotis_op3"

    skip_reason = "Not yet implemented"


class TestMenagerie_ToddlerBot2xc(TestMenagerieMJCF):
    """ToddlerBot 2XC humanoid."""

    robot_folder = "toddlerbot_2xc"

    skip_reason = "Not yet implemented"


class TestMenagerie_ToddlerBot2xm(TestMenagerieMJCF):
    """ToddlerBot 2XM humanoid."""

    robot_folder = "toddlerbot_2xm"

    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeG1(TestMenagerieMJCF):
    """Unitree G1 humanoid."""

    robot_folder = "unitree_g1"
    # Dynamics disabled: actuator_biasprm diff causes qvel divergence 3.3e-6 (#2170)
    num_steps = 0
    fk_enabled = True
    # TODO(#2170): actuator_biasprm has tiny fp diffs (1.7e-5) — likely precision issue
    model_skip_fields = DEFAULT_MODEL_SKIP_FIELDS | {"actuator_biasprm"}


class TestMenagerie_UnitreeH1(TestMenagerieMJCF):
    """Unitree H1 humanoid."""

    robot_folder = "unitree_h1"
    num_steps = 20
    fk_enabled = True
    backfill_model = True


# -----------------------------------------------------------------------------
# Bipeds (1 robot)
# -----------------------------------------------------------------------------


class TestMenagerie_AgilityCassie(TestMenagerieMJCF):
    """Agility Robotics Cassie biped."""

    robot_folder = "agility_cassie"
    skip_reason = "Closed-loop kinematic chains cause different DOF layout"


# -----------------------------------------------------------------------------
# Quadrupeds (8 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_AnyboticsAnymalB(TestMenagerieMJCF):
    """ANYbotics ANYmal B quadruped."""

    robot_folder = "anybotics_anymal_b"

    skip_reason = "Not yet implemented"


class TestMenagerie_AnyboticsAnymalC(TestMenagerieMJCF):
    """ANYbotics ANYmal C quadruped."""

    robot_folder = "anybotics_anymal_c"
    # Dynamics disabled: qvel divergence 7.2e-5 (model compilation diffs amplified by free joint)
    num_steps = 0
    fk_enabled = True


class TestMenagerie_BostonDynamicsSpot(TestMenagerieMJCF):
    """Boston Dynamics Spot quadruped."""

    robot_folder = "boston_dynamics_spot"
    num_steps = 20
    dynamics_tolerance = 5e-6
    fk_enabled = True
    backfill_model = True


class TestMenagerie_GoogleBarkourV0(TestMenagerieMJCF):
    """Google Barkour v0 quadruped."""

    robot_folder = "google_barkour_v0"

    skip_reason = "Not yet implemented"


class TestMenagerie_GoogleBarkourVb(TestMenagerieMJCF):
    """Google Barkour vB quadruped."""

    robot_folder = "google_barkour_vb"

    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeA1(TestMenagerieMJCF):
    """Unitree A1 quadruped."""

    robot_folder = "unitree_a1"

    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeGo1(TestMenagerieMJCF):
    """Unitree Go1 quadruped."""

    robot_folder = "unitree_go1"

    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeGo2(TestMenagerieMJCF):
    """Unitree Go2 quadruped."""

    robot_folder = "unitree_go2"
    # Dynamics disabled: qvel drift 1.2e-5 (model compilation diffs amplified by free joint)
    num_steps = 0
    fk_enabled = True


# -----------------------------------------------------------------------------
# Arms with Gripper (Unitree Z1)
# -----------------------------------------------------------------------------


class TestMenagerie_UnitreeZ1(TestMenagerieMJCF):
    """Unitree Z1 arm."""

    robot_folder = "unitree_z1"

    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Drones (2 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_BitcrazeCrazyflie2(TestMenagerieMJCF):
    """Bitcraze Crazyflie 2 quadrotor."""

    robot_folder = "bitcraze_crazyflie_2"

    skip_reason = "Not yet implemented"


class TestMenagerie_SkydioX2(TestMenagerieMJCF):
    """Skydio X2 drone."""

    robot_folder = "skydio_x2"

    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Mobile Bases (2 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_RobotSoccerKit(TestMenagerieMJCF):
    """Robot Soccer Kit omniwheel base."""

    robot_folder = "robot_soccer_kit"

    skip_reason = "Not yet implemented"


class TestMenagerie_RobotstudioSo101(TestMenagerieMJCF):
    """RobotStudio SO-101."""

    robot_folder = "robotstudio_so101"
    # Dynamics disabled: body_mass diff causes qpos divergence 3.4e-5 (#2170)
    num_steps = 0
    fk_enabled = True
    # TODO(#2170): body_mass differs for some bodies
    model_skip_fields = DEFAULT_MODEL_SKIP_FIELDS | {"body_mass"}


# -----------------------------------------------------------------------------
# Biomechanical (1 robot)
# -----------------------------------------------------------------------------


class TestMenagerie_Flybody(TestMenagerieMJCF):
    """Flybody fruit fly model."""

    robot_folder = "flybody"

    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Other (1 robot)
# -----------------------------------------------------------------------------


class TestMenagerie_I2rtYam(TestMenagerieMJCF):
    """i2rt YAM (Yet Another Manipulator)."""

    robot_folder = "i2rt_yam"

    skip_reason = "Not yet implemented"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
