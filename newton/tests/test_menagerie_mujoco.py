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
MuJoCo Menagerie Integration Tests

This module tests that robots from the MuJoCo Menagerie simulate identically
when loaded via MJCF into Newton's MuJoCo solver vs native MuJoCo.

Test Architecture:
    - TestMenagerieBase: Abstract base class with all test infrastructure
    - TestMenagerieMJCF / TestMenagerieUSD: Model source variants
    - TestMenagerie_<RobotName>: One derived class per menagerie robot

Each test:
    1. Downloads the robot from menagerie (cached)
    2. Creates Newton model (via MJCF or USD factory)
    3. Creates native MuJoCo model from same source
    4. Compares model fields (with physics-equivalence checks for inertia, solref, etc.)
    5. Runs simulation with configurable control strategies
    6. Compares per-step state values within tolerance

Known limitations and workarounds:
    - Friction defaults: Newton's MJCF parser must explicitly set MuJoCo friction
      defaults [slide=1.0, torsion=0.005, roll=0.0001] on the builder. See
      create_newton_model_from_mjcf().
    - Model field skips: See DEFAULT_MODEL_SKIP_FIELDS for fields that are skipped
      in model comparison (with inline comments explaining each).
    - Model backfill: Newton's model compilation differs from MuJoCo's mj_setConst()
      (e.g. inertia re-diagonalization, body_pos/quat recomputation from joint transforms).
      Set backfill_model=True to copy computed fields from native, isolating simulation
      diffs from model compilation diffs. See MODEL_BACKFILL_FIELDS.
    - Split pipeline: mujoco_warp uses wp.atomic_add() for contact and constraint
      allocation, causing non-deterministic ordering with >8 worlds. Set
      use_split_pipeline=True to inject contacts+constraints from Newton into native,
      bypassing this. Sorted comparisons verify the rows match modulo ordering.

TODOs:
    - Parse timestep from MJCF <option timestep="..."/> into Newton model
    - Fix collision exclusion (nexclude) parent/child filtering in Newton
    - Align mocap body handling between Newton and MuJoCo
"""

from __future__ import annotations

import re
import time
import unittest
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.utils.download_assets import download_git_folder
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

MENAGERIE_GIT_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def download_menagerie_asset(
    robot_folder: str,
    cache_dir: str | None = None,
    force_refresh: bool = False,
) -> Path:
    """
    Download a robot folder from the MuJoCo Menagerie repository.

    Args:
        robot_folder: The folder name in the menagerie repo (e.g., "unitree_go2")
        cache_dir: Optional cache directory override
        force_refresh: If True, re-download even if cached

    Returns:
        Path to the downloaded robot folder
    """
    return download_git_folder(
        MENAGERIE_GIT_URL,
        robot_folder,
        cache_dir=cache_dir,
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

    # Use MuJoCo's default friction values for comparison testing
    # MuJoCo defaults: [slide=1.0, torsion=0.005, roll=0.0001]
    robot_builder.default_shape_cfg.mu = 1.0
    robot_builder.default_shape_cfg.mu_torsional = 0.005
    robot_builder.default_shape_cfg.mu_rolling = 0.0001

    # Use floating=None to honor the MJCF's explicit joint definitions.
    # Menagerie models define their own <freejoint> tags for floating-base robots.
    # Disable ensure_nonstatic_links to match MuJoCo's handling of zero-mass bodies.
    robot_builder.add_mjcf(
        str(mjcf_path),
        parse_visuals=parse_visuals,
        ensure_nonstatic_links=False,
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


def create_newton_model_from_usd(
    mjcf_path: Path,
    *,
    num_worlds: int = 1,
    add_ground: bool = True,
) -> newton.Model:
    """
    Create a Newton model by converting MJCF to USD first.

    NOTE: This is a placeholder for future USD converter integration.

    Args:
        mjcf_path: Path to the MJCF XML file
        num_worlds: Number of world instances to create
        add_ground: Whether to add a ground plane

    Returns:
        Finalized Newton Model
    """
    raise NotImplementedError(
        "USD conversion path not yet implemented. Waiting for MuJoCo USD converter to be finalized."
    )


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


class ZeroControlStrategy(ControlStrategy):
    """Always returns zero control - useful for drop tests."""

    def __init__(self):
        super().__init__(seed=0)

    def init(self, native_ctrl: wp.array, newton_ctrl: wp.array):
        """Initialize - ctrl arrays are already zeroed by MuJoCo."""
        pass

    def fill_control(self, t: float):
        """No-op - control stays at zero."""
        pass


@wp.kernel
def generate_sinusoidal_control_kernel_dual(
    frequencies: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    phases: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    native_ctrl: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    newton_ctrl: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    t: wp.float32,
    amplitude: wp.float32,
    num_actuators: int,
):
    """Generate sinusoidal control pattern and write to both arrays."""
    i = wp.tid()
    act_idx = i % num_actuators  # type: ignore[operator]
    freq = frequencies[act_idx]
    phase = phases[i]
    two_pi = 6.28318530717959
    val = amplitude * wp.sin(two_pi * freq * t + phase)  # type: ignore[operator]
    clamped = wp.clamp(val, -1.0, 1.0)  # type: ignore[arg-type]
    native_ctrl[i] = clamped
    newton_ctrl[i] = clamped


class StructuredControlStrategy(ControlStrategy):
    """
    Generate structured control patterns designed to explore joint limits.

    Uses a Warp kernel to generate sinusoidal controls directly on GPU.
    Each world gets a slightly different phase offset for variation.
    """

    def __init__(
        self,
        seed: int = 42,
        amplitude_scale: float = 0.8,
        frequency_range: tuple[float, float] = (0.5, 2.0),
    ):
        super().__init__(seed)
        self.amplitude_scale = amplitude_scale
        self.frequency_range = frequency_range
        self._frequencies: wp.array | None = None
        self._phases: wp.array | None = None
        self._native_ctrl: wp.array | None = None
        self._newton_ctrl: wp.array | None = None
        self._n: int = 0

    def init(self, native_ctrl: wp.array, newton_ctrl: wp.array):
        """Initialize with the ctrl arrays to fill."""
        num_worlds, num_actuators = native_ctrl.shape
        n = num_worlds * num_actuators

        frequencies_np = self.rng.uniform(self.frequency_range[0], self.frequency_range[1], num_actuators)
        phases_np = self.rng.uniform(0, 2 * np.pi, n)

        self._frequencies = wp.array(frequencies_np, dtype=wp.float32)
        self._phases = wp.array(phases_np, dtype=wp.float32)
        self._native_ctrl = native_ctrl.flatten()
        self._newton_ctrl = newton_ctrl
        self._n = n

    def fill_control(self, t: float):
        """Fill control values into both arrays with single kernel launch."""
        if self._frequencies is None:
            raise RuntimeError("Call init() first")
        wp.launch(
            generate_sinusoidal_control_kernel_dual,
            dim=self._n,
            inputs=[
                self._frequencies,
                self._phases,
                self._native_ctrl,
                self._newton_ctrl,
                t,
                self.amplitude_scale,
                self._n // self._frequencies.shape[0],  # num_actuators
            ],
        )


@wp.kernel
def generate_random_control_kernel_dual(
    prev_ctrl: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    native_ctrl: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    newton_ctrl: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    seed: int,
    step: int,
    noise_scale: wp.float32,
    smoothing: wp.float32,
):
    """Generate random control with smoothing, writing to both arrays."""
    i = wp.tid()
    state = wp.rand_init(seed, i + step * 1000000)  # type: ignore[arg-type, operator]
    rand_val = wp.randf(state) * 2.0 - 1.0
    noise = rand_val * noise_scale
    one_minus_smooth = 1.0 - smoothing
    val = smoothing * prev_ctrl[i] + one_minus_smooth * noise
    clamped = wp.clamp(val, -1.0, 1.0)
    native_ctrl[i] = clamped
    newton_ctrl[i] = clamped
    prev_ctrl[i] = clamped


class RandomControlStrategy(ControlStrategy):
    """
    Generate random control values with configurable noise.

    Uses a Warp kernel to generate smoothed random controls on GPU.
    Each world gets independent random controls.
    """

    def __init__(
        self,
        seed: int = 42,
        noise_scale: float = 0.5,
        smoothing: float = 0.9,
    ):
        super().__init__(seed)
        self.noise_scale = noise_scale
        self.smoothing = smoothing
        self._prev_ctrl: wp.array | None = None
        self._native_ctrl: wp.array | None = None
        self._newton_ctrl: wp.array | None = None
        self._n: int = 0
        self._step: int = 0

    def init(self, native_ctrl: wp.array, newton_ctrl: wp.array):
        """Initialize with the ctrl arrays to fill."""
        n = native_ctrl.shape[0] * native_ctrl.shape[1]
        self._prev_ctrl = wp.zeros(n, dtype=wp.float32)
        self._native_ctrl = native_ctrl.flatten()
        self._newton_ctrl = newton_ctrl
        self._n = n
        self._step = 0

    def fill_control(self, t: float):
        """Fill control values into both arrays with single kernel launch."""
        if self._prev_ctrl is None:
            raise RuntimeError("Call init() first")
        wp.launch(
            generate_random_control_kernel_dual,
            dim=self._n,
            inputs=[
                self._prev_ctrl,
                self._native_ctrl,
                self._newton_ctrl,
                self.rng.integers(0, 2**31),
                self._step,
                self.noise_scale,
                self.smoothing,
            ],
        )
        self._step += 1


# =============================================================================
# Comparison
# =============================================================================

# Default tolerances for MjData field comparison.
# Two tolerance classes: tight (1e-6) and loose (1e-4).
# With backfill_model=True, tests can verify numerical equivalence more strictly.
DEFAULT_TOLERANCES: dict[str, float] = {
    # Tight (1e-6): kinematics, positions, orientations, mass matrix
    "qpos": 1e-6,
    "xpos": 1e-6,
    "xquat": 1e-6,
    "xmat": 1e-6,
    "geom_xpos": 1e-6,
    "geom_xmat": 1e-6,
    "site_xpos": 1e-6,
    "site_xmat": 1e-6,
    "subtree_com": 1e-6,
    "actuator_length": 1e-6,
    "qfrc_passive": 1e-6,
    "energy": 1e-6,
    "qM": 1e-6,
    # Loose (1e-4): velocities, accelerations, forces
    "qvel": 1e-4,
    "cvel": 1e-4,
    "actuator_velocity": 1e-4,
    "qfrc_bias": 1e-4,
    "cacc": 1e-4,
    "qacc": 1e-4,
    "cfrc_int": 1e-4,
    "qfrc_actuator": 1e-4,
    "actuator_force": 1e-4,
}

# Default fields to compare in MjData (core physics + dynamics)
DEFAULT_COMPARE_FIELDS: list[str] = [
    # Core state
    "qpos",
    "qvel",
    "qacc",
    # Body kinematics
    "xpos",
    "xquat",
    "xmat",
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
    # Forces
    "qfrc_bias",
    "qfrc_passive",
    "qfrc_actuator",
    # Composite quantities
    "subtree_com",
    # Dynamics
    "cvel",
    "cacc",
    "cfrc_int",
    "energy",
    "qM",
    # Actuators
    "actuator_length",
    "actuator_velocity",
    "actuator_force",
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
    # Size: Compared via compare_geom_sizes() which understands type-specific semantics
    "geom_size",
    # Range: Compared via compare_jnt_range() which only checks limited joints
    # (MuJoCo ignores range when jnt_limited=False, Newton stores [-1e10, 1e10])
    "jnt_range",
}


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


def compare_geom_sizes(
    newton_mjw: Any,
    native_mjw: Any,
    tol: float = 1e-6,
) -> None:
    """Compare geom_size arrays accounting for type-specific semantics.

    MuJoCo geom size interpretation varies by type:
    - PLANE (0): [visual_x, visual_y, spacing] - purely decorative, skip
    - SPHERE (2): [radius, -, -] - only first component matters
    - CAPSULE (3): [radius, half_length, -] - first two components matter
    - ELLIPSOID (4): [x, y, z radii] - all three matter
    - CYLINDER (5): [radius, half_length, -] - first two components matter
    - BOX (6): [x, y, z half-sizes] - all three matter
    - MESH (7): [scale_x, scale_y, scale_z] - all three matter

    Newton fills zero components with first nonzero value for visualization,
    but this doesn't affect physics. This function compares only the
    physics-relevant components for each geom type.
    """
    # MuJoCo geom type constants
    GEOM_PLANE = 0
    GEOM_SPHERE = 2
    GEOM_CAPSULE = 3
    GEOM_ELLIPSOID = 4
    GEOM_CYLINDER = 5
    GEOM_BOX = 6
    GEOM_MESH = 7

    newton_size = newton_mjw.geom_size.numpy()  # (nworld, ngeom, 3)
    native_size = native_mjw.geom_size.numpy()
    newton_type = newton_mjw.geom_type.numpy()  # (ngeom,) - shared across worlds
    native_type = native_mjw.geom_type.numpy()

    assert newton_size.shape == native_size.shape, (
        f"geom_size shape mismatch: {newton_size.shape} vs {native_size.shape}"
    )
    assert newton_type.shape == native_type.shape, (
        f"geom_type shape mismatch: {newton_type.shape} vs {native_type.shape}"
    )

    # Geom types must match
    np.testing.assert_array_equal(newton_type, native_type, err_msg="geom_type mismatch")

    nworld, ngeom, _ = newton_size.shape

    for world in range(nworld):
        for geom in range(ngeom):
            gtype = newton_type[geom]  # geom_type is (ngeom,), not (nworld, ngeom)
            n_size = newton_size[world, geom]
            nat_size = native_size[world, geom]

            if gtype == GEOM_PLANE:
                # Plane size is purely visual (Newton hardcodes [5,5,5], native uses MJCF value)
                # Both are infinite for collision - skip comparison
                continue
            elif gtype == GEOM_SPHERE:
                # Only radius (first component) matters
                np.testing.assert_allclose(
                    n_size[0],
                    nat_size[0],
                    atol=tol,
                    rtol=0,
                    err_msg=f"geom_size[{world},{geom}] (SPHERE) radius mismatch",
                )
            elif gtype in (GEOM_CAPSULE, GEOM_CYLINDER):
                # Radius and half-length (first two components) matter
                np.testing.assert_allclose(
                    n_size[:2],
                    nat_size[:2],
                    atol=tol,
                    rtol=0,
                    err_msg=f"geom_size[{world},{geom}] (CAPSULE/CYLINDER) mismatch",
                )
            elif gtype in (GEOM_ELLIPSOID, GEOM_BOX, GEOM_MESH):
                # All three components matter
                np.testing.assert_allclose(
                    n_size,
                    nat_size,
                    atol=tol,
                    rtol=0,
                    err_msg=f"geom_size[{world},{geom}] (ELLIPSOID/BOX/MESH) mismatch",
                )
            else:
                # Unknown type - compare all (fail if different)
                np.testing.assert_allclose(
                    n_size,
                    nat_size,
                    atol=tol,
                    rtol=0,
                    err_msg=f"geom_size[{world},{geom}] (type={gtype}) mismatch",
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
# Contact Injection Helpers
# =============================================================================


def run_native_step1_rest(
    native_model: Any,
    native_data: Any,
) -> None:
    """Run rest of step1 after fwd_position: sensors + fwd_velocity.

    Call this after run_native_make_constraint() or run_native_transmission().
    """
    from mujoco_warp._src import forward as mjw_forward
    from mujoco_warp._src import sensor
    from mujoco_warp._src.types import EnableBit

    m, d = native_model, native_data
    energy = m.opt.enableflags & EnableBit.ENERGY

    d.sensordata.zero_()
    sensor.sensor_pos(m, d)

    if energy:
        if m.sensor_e_potential == 0:
            sensor.energy_pos(m, d)
    else:
        d.energy.zero_()

    mjw_forward.fwd_velocity(m, d)
    sensor.sensor_vel(m, d)

    if energy:
        if m.sensor_e_kinetic == 0:
            sensor.energy_vel(m, d)


def run_native_fwd_position_pre_constraint(
    native_model: Any,
    native_data: Any,
) -> None:
    """Run mujoco_warp fwd_position up to and including collision detection.

    This runs: kinematics, com_pos, camlight, flex, tendon, crb, tendon_armature,
    factor_m, and collision. Does NOT run make_constraint or transmission.
    """
    from mujoco_warp._src import (
        collision_driver,
        smooth,
    )

    m, d = native_model, native_data
    smooth.kinematics(m, d)
    smooth.com_pos(m, d)
    smooth.camlight(m, d)
    smooth.flex(m, d)
    smooth.tendon(m, d)
    smooth.crb(m, d)
    smooth.tendon_armature(m, d)
    smooth.factor_m(m, d)
    if m.opt.run_collision_detection:
        collision_driver.collision(m, d)


def run_native_make_constraint(
    native_model: Any,
    native_data: Any,
) -> None:
    """Run mujoco_warp make_constraint only."""
    from mujoco_warp._src import constraint

    constraint.make_constraint(native_model, native_data)


def run_native_transmission(
    native_model: Any,
    native_data: Any,
) -> None:
    """Run mujoco_warp transmission only (after constraint injection)."""
    from mujoco_warp._src import smooth

    smooth.transmission(native_model, native_data)


def run_native_step2(
    native_model: Any,
    native_data: Any,
) -> None:
    """Run mujoco_warp step2: fwd_actuation + fwd_acceleration + solve + integrate.

    Uses the correct integrator based on model.opt.integrator.
    Note: fwd_acceleration does NOT re-run collision detection.
    Note: Does NOT call wp.synchronize() - caller should sync if needed.
    """
    from mujoco_warp._src import forward as mjw_forward

    mjw_forward.step2(native_model, native_data)


def inject_contacts(src_data: Any, dst_data: Any) -> None:
    """Copy all contact arrays from src_data to dst_data.

    This ensures both Newton and native use identical contact data for constraint solving,
    bypassing the non-deterministic contact ordering issue in mujoco_warp.
    """
    dst_data.nacon.assign(src_data.nacon)
    dst_data.contact.worldid.assign(src_data.contact.worldid)
    dst_data.contact.geom.assign(src_data.contact.geom)
    dst_data.contact.pos.assign(src_data.contact.pos)
    dst_data.contact.dist.assign(src_data.contact.dist)
    dst_data.contact.frame.assign(src_data.contact.frame)
    dst_data.contact.includemargin.assign(src_data.contact.includemargin)
    dst_data.contact.friction.assign(src_data.contact.friction)
    dst_data.contact.solref.assign(src_data.contact.solref)
    dst_data.contact.solreffriction.assign(src_data.contact.solreffriction)
    dst_data.contact.solimp.assign(src_data.contact.solimp)
    dst_data.contact.dim.assign(src_data.contact.dim)
    dst_data.contact.efc_address.assign(src_data.contact.efc_address)
    wp.synchronize()


def compare_contacts_sorted(
    newton_data: Any,
    native_data: Any,
    tol: float = 1e-6,
    boundary_threshold: float = 1e-6,
) -> tuple[bool, str]:
    """Compare contacts between Newton and native after sorting by (worldid, geom1, geom2, pos).

    Contacts with |dist| < boundary_threshold are considered numerically unstable and are
    ignored when comparing contact sets. This handles cases where tiny position differences
    (~1e-7) cause contacts at the threshold to flip in/out.

    Args:
        newton_data: Newton's MjWarpData
        native_data: Native MuJoCo's MjWarpData
        tol: Tolerance for numeric field comparisons (pos, dist, frame)
        boundary_threshold: Contacts with |dist| below this are ignored in set comparison

    Returns:
        (matches, message) - True if contacts match within tolerance, with diagnostic message.
    """
    wp.synchronize()

    nacon_newton = int(newton_data.nacon.numpy()[0])
    nacon_native = int(native_data.nacon.numpy()[0])

    if nacon_newton == 0 and nacon_native == 0:
        return True, "No contacts"

    # Get contact data as sets of (worldid, geom1, geom2) tuples
    def get_contact_set(data: Any, nacon: int, threshold: float) -> tuple[set, dict]:
        """Returns (significant_contacts_set, all_contacts_dict)."""
        worldid = data.contact.worldid.numpy()[:nacon]
        geom = data.contact.geom.numpy()[:nacon]
        pos = data.contact.pos.numpy()[:nacon]
        dist = data.contact.dist.numpy()[:nacon]
        frame = data.contact.frame.numpy()[:nacon]

        # Create set of significant contacts (|dist| >= threshold)
        significant_set = set()
        all_contacts = {}
        for i in range(nacon):
            key = (int(worldid[i]), int(geom[i, 0]), int(geom[i, 1]))
            all_contacts[key] = {"pos": pos[i], "dist": dist[i], "frame": frame[i]}
            if abs(dist[i]) >= threshold:
                significant_set.add(key)

        return significant_set, all_contacts

    newton_sig, newton_all = get_contact_set(newton_data, nacon_newton, boundary_threshold)
    native_sig, native_all = get_contact_set(native_data, nacon_native, boundary_threshold)

    # Check if significant contacts match
    only_newton = newton_sig - native_sig
    only_native = native_sig - newton_sig

    if only_newton or only_native:
        # Check if mismatched contacts are near boundary in the other set
        real_mismatch = []
        for key in only_newton:
            if key not in native_all:
                real_mismatch.append(f"Newton-only (not in native): {key}")
        for key in only_native:
            if key not in newton_all:
                real_mismatch.append(f"Native-only (not in newton): {key}")

        if real_mismatch:
            return False, f"Significant contact mismatch: {real_mismatch[:3]}"

    # For common significant contacts, compare numeric values
    common_contacts = newton_sig & native_sig
    if common_contacts:
        max_pos_diff = 0.0
        max_dist_diff = 0.0
        max_frame_diff = 0.0
        for key in common_contacts:
            nc = newton_all[key]
            na = native_all[key]
            max_pos_diff = max(max_pos_diff, float(np.abs(nc["pos"] - na["pos"]).max()))
            max_dist_diff = max(max_dist_diff, float(abs(nc["dist"] - na["dist"])))
            max_frame_diff = max(max_frame_diff, float(np.abs(nc["frame"] - na["frame"]).max()))

        if max_pos_diff > tol:
            return False, f"pos diff: {max_pos_diff:.2e}"
        if max_dist_diff > tol:
            return False, f"dist diff: {max_dist_diff:.2e}"
        if max_frame_diff > tol:
            return False, f"frame diff: {max_frame_diff:.2e}"

    # Report boundary contacts that differ (for information, not failure)
    boundary_only_newton = len(newton_all) - len(newton_sig)
    boundary_only_native = len(native_all) - len(native_sig)

    return (
        True,
        f"OK (sig={len(common_contacts)}, boundary: newton={boundary_only_newton}, native={boundary_only_native})",
    )


def compare_constraints_sorted(
    newton_data: Any,
    native_data: Any,
    tol: float = 1e-5,
) -> tuple[bool, str]:
    """Compare constraint rows between Newton and native, allowing reordering.

    mujoco_warp's make_constraint allocates efc row indices via wp.atomic_add(),
    so identical contacts produce identical constraint rows but in non-deterministic
    order. This function verifies the rows match by sorting each world's constraints
    by a deterministic key (type, id, J-hash).

    Args:
        newton_data: Newton's MjWarpData
        native_data: Native MuJoCo's MjWarpData
        tol: Tolerance for numeric comparisons (D, aref, pos, vel)

    Returns:
        (matches, message) - True if constraint rows match modulo ordering.
    """
    wp.synchronize()

    nefc_newton = newton_data.nefc.numpy()
    nefc_native = native_data.nefc.numpy()

    # Constraint counts per world must match
    if not np.array_equal(nefc_newton, nefc_native):
        diff_worlds = np.where(nefc_newton != nefc_native)[0]
        return False, (
            f"nefc mismatch in {len(diff_worlds)} worlds, "
            f"e.g. world {diff_worlds[0]}: newton={nefc_newton[diff_worlds[0]]} native={nefc_native[diff_worlds[0]]}"
        )

    nworld = len(nefc_newton)

    # Get constraint arrays
    newton_type = newton_data.efc.type.numpy()  # (nworld, njmax)
    native_type = native_data.efc.type.numpy()
    newton_id = newton_data.efc.id.numpy()
    native_id = native_data.efc.id.numpy()
    newton_D = newton_data.efc.D.numpy()
    native_D = native_data.efc.D.numpy()
    newton_J = newton_data.efc.J.numpy()
    native_J = native_data.efc.J.numpy()
    newton_aref = newton_data.efc.aref.numpy()
    native_aref = native_data.efc.aref.numpy()

    max_D_diff = 0.0
    max_J_diff = 0.0
    max_aref_diff = 0.0

    for w in range(nworld):
        n = int(nefc_newton[w])
        if n == 0:
            continue

        # Build sort keys: (type, id) for each row — deterministic per constraint
        newton_keys = [(int(newton_type[w, i]), int(newton_id[w, i])) for i in range(n)]
        native_keys = [(int(native_type[w, i]), int(native_id[w, i])) for i in range(n)]

        newton_order = sorted(range(n), key=lambda i: newton_keys[i])
        native_order = sorted(range(n), key=lambda i: native_keys[i])

        # Verify keys match after sorting
        sorted_newton_keys = [newton_keys[i] for i in newton_order]
        sorted_native_keys = [native_keys[i] for i in native_order]
        if sorted_newton_keys != sorted_native_keys:
            # Find first mismatch
            for k in range(n):
                if sorted_newton_keys[k] != sorted_native_keys[k]:
                    return False, (
                        f"world {w}: constraint key mismatch at sorted pos {k}: "
                        f"newton={sorted_newton_keys[k]} native={sorted_native_keys[k]}"
                    )

        # Compare numeric fields in sorted order
        for k in range(n):
            ni, nai = newton_order[k], native_order[k]
            max_D_diff = max(max_D_diff, abs(float(newton_D[w, ni] - native_D[w, nai])))
            max_J_diff = max(max_J_diff, float(np.max(np.abs(newton_J[w, ni] - native_J[w, nai]))))
            max_aref_diff = max(max_aref_diff, abs(float(newton_aref[w, ni] - native_aref[w, nai])))

    if max_D_diff > tol:
        return False, f"efc.D diff: {max_D_diff:.2e} > tol {tol:.0e}"
    if max_J_diff > tol:
        return False, f"efc.J diff: {max_J_diff:.2e} > tol {tol:.0e}"
    if max_aref_diff > tol:
        return False, f"efc.aref diff: {max_aref_diff:.2e} > tol {tol:.0e}"

    total = int(nefc_newton.sum())
    return True, f"OK ({total} constraints, max D diff={max_D_diff:.2e}, J diff={max_J_diff:.2e})"


def inject_constraints(src_data: Any, dst_data: Any) -> None:
    """Copy all constraint (efc) arrays from src_data to dst_data.

    This ensures both Newton and native use identical constraint data for solving,
    bypassing the non-deterministic constraint row ordering from wp.atomic_add()
    in make_constraint.
    """
    # Constraint counts
    dst_data.ne.assign(src_data.ne)
    dst_data.nf.assign(src_data.nf)
    dst_data.nl.assign(src_data.nl)
    dst_data.nefc.assign(src_data.nefc)
    # Constraint arrays
    dst_data.efc.type.assign(src_data.efc.type)
    dst_data.efc.id.assign(src_data.efc.id)
    dst_data.efc.J.assign(src_data.efc.J)
    dst_data.efc.pos.assign(src_data.efc.pos)
    dst_data.efc.margin.assign(src_data.efc.margin)
    dst_data.efc.D.assign(src_data.efc.D)
    dst_data.efc.vel.assign(src_data.efc.vel)
    dst_data.efc.aref.assign(src_data.efc.aref)
    dst_data.efc.frictionloss.assign(src_data.efc.frictionloss)
    dst_data.efc.state.assign(src_data.efc.state)
    wp.synchronize()


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

    if newton_arr is None or native_arr is None:
        return

    if newton_arr.size == 0:
        return

    # Sync and copy to numpy
    wp.synchronize()
    newton_np = newton_arr.numpy()
    native_np = native_arr.numpy()

    # Vectorized comparison
    diff = np.abs(newton_np - native_np)
    max_diff = float(np.max(diff))

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
    "body_pos",
    "body_quat",
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
    tol: float = 1e-3,
) -> None:
    """Copy computed model fields from native MuJoCo to Newton's mjw_model.

    This eliminates numerical differences caused by Newton's model compilation
    differing from MuJoCo's mj_setConst(). Useful for isolating simulation
    differences from model compilation differences during testing.

    Before copying, each field is verified to be within ``tol`` of the native value.
    Fields that are expected to have large relative differences (body_inertia,
    body_iquat — verified separately via compare_inertia_tensors) are exempt.
    This catches real parser bugs (e.g. body_pos off by 1.0) while allowing
    expected small compilation differences (e.g. body_pos off by 3e-8).

    Args:
        newton_mjw: Newton's MjWarpModel to update
        native_mjw: Native MuJoCo's MjWarpModel to copy from
        fields: List of field names to copy (defaults to MODEL_BACKFILL_FIELDS)
        tol: Maximum allowed absolute difference before backfill (default 1e-3)
    """
    if fields is None:
        fields = MODEL_BACKFILL_FIELDS

    # Fields verified separately (inertia re-diag gives large but physics-equivalent diffs)
    skip_verification = {
        "body_inertia",
        "body_iquat",
        "body_invweight0",
        "dof_invweight0",
        "actuator_acc0",
    }

    for field in fields:
        native_arr = getattr(native_mjw, field, None)
        newton_arr = getattr(newton_mjw, field, None)

        if native_arr is None or newton_arr is None:
            continue
        if not hasattr(native_arr, "numpy") or not hasattr(newton_arr, "numpy"):
            continue

        # Only copy if shapes match exactly
        if native_arr.shape == newton_arr.shape:
            # Verify diff is within tolerance (catch parser bugs)
            if field not in skip_verification:
                diff = float(np.max(np.abs(native_arr.numpy().astype(float) - newton_arr.numpy().astype(float))))
                assert diff <= tol, (
                    f"Backfill field '{field}' has diff {diff:.6e} > tol {tol:.0e}. "
                    f"This likely indicates a parser bug, not a compilation difference."
                )
            newton_arr.assign(native_arr)

    wp.synchronize()


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


def print_mjdata_diff(
    newton_mjw_data: Any,
    native_mjw_data: Any,
    compare_fields: list[str],
    tolerances: dict[str, float],
    step_num: int,
) -> None:
    """Print max difference for each MjData field."""
    wp.synchronize()
    for field_name in compare_fields:
        newton_arr = getattr(newton_mjw_data, field_name, None)
        native_arr = getattr(native_mjw_data, field_name, None)
        if newton_arr is None or native_arr is None or newton_arr.size == 0:
            continue
        newton_np = newton_arr.numpy()
        native_np = native_arr.numpy()
        # Handle shape mismatch
        if newton_np.shape != native_np.shape:
            print(f"  {field_name}: SHAPE MISMATCH newton={newton_np.shape} vs native={native_np.shape}")
            continue
        max_diff = float(np.max(np.abs(newton_np - native_np)))
        tol = tolerances.get(field_name, 1e-6)
        status = "OK" if max_diff <= tol else "FAIL"
        print(f"  {field_name}: max_diff={max_diff:.6e} (tol={tol:.0e}) [{status}]")


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

    # TODO: Implement randomization using SolverMuJoCo remappings
    # This requires careful coordination between Newton arrays and MuJoCo arrays
    # The remappings in SolverMuJoCo (e.g., body indices, joint indices) must be used
    # to ensure both sides receive identical randomized values
    _ = newton_model, mj_solver, seed  # Suppress unused warnings for now


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
        - num_worlds: int - number of parallel worlds (default: 2)
        - num_steps: int - simulation steps to run (default: 100)
        - dt: float - timestep fallback (actual dt extracted from native model)
        - control_strategy: ControlStrategy - how to generate controls
        - compare_fields: list[str] - MjData fields to compare
        - tolerances: dict[str, float] - per-field tolerances
        - skip_reason: str | None - if set, skip this test
    """

    # Must be defined by subclasses
    robot_folder: str = ""
    robot_xml: str = "scene.xml"  # Default; most menagerie robots use scene.xml

    # Configurable defaults
    num_worlds: int = 34
    num_steps: int = 100
    dt: float = 0.002  # Fallback; actual dt extracted from native model in test

    # Control strategy (can override in subclass)
    control_strategy: ControlStrategy | None = None

    # Data comparison: explicit list of fields TO compare
    compare_fields: ClassVar[list[str]] = DEFAULT_COMPARE_FIELDS

    # Tolerances: override specific fields per-test, merged with defaults
    # Example: tolerance_overrides = {"qacc": 0.1, "qfrc_actuator": 0.01}
    tolerance_overrides: ClassVar[dict[str, float]] = {}

    @property
    def tolerances(self) -> dict[str, float]:
        """Get tolerances with per-test overrides merged in."""
        return {**DEFAULT_TOLERANCES, **self.tolerance_overrides}

    # Model comparison: fields to SKIP (substrings to match)
    # Override in subclass with: model_skip_fields = DEFAULT_MODEL_SKIP_FIELDS | {"extra", "fields"}
    model_skip_fields: ClassVar[set[str]] = DEFAULT_MODEL_SKIP_FIELDS

    # Skip reason (set to a string to skip test, leave unset or None to run)
    skip_reason: str | None = None

    # Debug mode: opens viewer for visual debugging
    debug_visual: bool = False
    debug_view_newton: bool = False  # False=Native, True=Newton

    # Skip visual-only geoms on the native side via compiler discardvisual="true".
    # Note: discardvisual may also strip some collision geoms on bodies with many visual
    # meshes (seen with Apollo). Models affected by this need per-test geom skips.
    discard_visual: bool = True

    # Backfill computed model fields from native to eliminate compilation diffs.
    # See MODEL_BACKFILL_FIELDS for the default set; override backfill_fields per-robot.
    backfill_model: bool = False
    backfill_fields: list[str] | None = None  # None = use MODEL_BACKFILL_FIELDS

    # Use split pipeline to bypass mujoco_warp non-deterministic ordering.
    # Injects contacts and constraints from Newton → native after verifying they
    # match (modulo ordering). Enables bit-identical results with 34+ worlds.
    # When False, runs both full pipelines (may need looser tolerances).
    use_split_pipeline: bool = False

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
        except Exception as e:
            raise unittest.SkipTest(f"Failed to download {cls.robot_folder}: {e}") from e

        cls.mjcf_path = cls.asset_path / cls.robot_xml
        if not cls.mjcf_path.exists():
            raise unittest.SkipTest(f"MJCF file not found: {cls.mjcf_path}")

    def setUp(self):
        """Set up test fixtures."""
        # Default control strategy
        if self.control_strategy is None:
            self.control_strategy = StructuredControlStrategy(seed=42)

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

    def _load_assets(self) -> dict[str, bytes]:
        """Load mesh/texture assets from the MJCF directory for from_xml_string."""
        assets: dict[str, bytes] = {}
        asset_dir = self.mjcf_path.parent

        # Common mesh extensions
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
            # Check if there's already a <compiler> tag
            if "<compiler" in xml_content:
                # Check if it's self-closing: <compiler ... />
                if re.search(r"<compiler[^>]*/\s*>", xml_content):
                    # Self-closing tag: insert attribute before />
                    xml_content = re.sub(
                        r"<compiler([^>]*)/\s*>",
                        r'<compiler\1 discardvisual="true"/>',
                        xml_content,
                    )
                else:
                    # Non-self-closing tag: insert attribute before >
                    xml_content = re.sub(
                        r"<compiler([^>]*)>",
                        r'<compiler\1 discardvisual="true">',
                        xml_content,
                    )
            else:
                # No compiler tag - insert new one after <mujoco...>
                xml_content = re.sub(
                    r"(<mujoco[^>]*>)",
                    r'\1\n  <compiler discardvisual="true"/>',
                    xml_content,
                )

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
        # Note: put_model creates arrays with nworld=1, expansion happens in test_simulation_equivalence
        mjw_model = _mujoco_warp.put_model(mj_model)
        mjw_data = _mujoco_warp.put_data(mj_model, mj_data, nworld=self.num_worlds)

        return mj_model, mj_data, mjw_model, mjw_data

    def test_simulation_equivalence(self):
        """
        Main test: verify Newton's SolverMuJoCo and native mujoco_warp produce equivalent results.

        Both sides use mujoco_warp with the same number of worlds.
        Each world receives different controls (varied by phase/noise).
        Uses CUDA graphs when available for performance.

        If debug_visual=True, opens MuJoCo viewer (set debug_view_newton to choose view).
        """
        assert _mujoco is not None
        assert _mujoco_warp is not None
        assert self.control_strategy is not None

        # Create models and solvers
        newton_model = self._create_newton_model()
        newton_state = newton_model.state()
        newton_control = newton_model.control()
        newton_solver = SolverMuJoCo(newton_model)

        mj_model, mj_data_native, native_mjw_model, native_mjw_data = self._create_native_mujoco_warp()

        # Expand native model's batched arrays to match Newton's shapes
        # Newton is the reference - only expand fields that Newton has expanded
        expand_mjw_model_to_match(native_mjw_model, newton_solver.mjw_model)

        # Extract timestep from native model (Newton doesn't parse <option timestep="..."/> yet)
        # TODO: Remove this workaround once Newton's MJCF parser supports timestep extraction
        dt = float(mj_model.opt.timestep)

        # Compare mjw_model structures
        compare_mjw_models(newton_solver.mjw_model, native_mjw_model, skip_fields=self.model_skip_fields)

        # Compare reconstructed inertia tensors (principal + iquat -> full 3x3)
        # The eig3 determinant fix ensures these match even if iquat orientation differs
        if not any("compare_inertia" in s for s in self.model_skip_fields):
            compare_inertia_tensors(newton_solver.mjw_model, native_mjw_model)

        # Compare solref fields by physics equivalence (direct mode vs standard mode)
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
            if any(s in solref_field for s in self.model_skip_fields):
                continue
            if hasattr(newton_solver.mjw_model, solref_field) and hasattr(native_mjw_model, solref_field):
                newton_arr = getattr(newton_solver.mjw_model, solref_field)
                native_arr = getattr(native_mjw_model, solref_field)
                # Only compare if both have data and shapes match (geom counts may differ)
                if newton_arr is not None and native_arr is not None:
                    if (
                        hasattr(newton_arr, "shape")
                        and newton_arr.shape == native_arr.shape
                        and newton_arr.shape[0] > 0
                    ):
                        compare_solref_physics(newton_solver.mjw_model, native_mjw_model, solref_field)

        # Compare geom sizes with type-specific semantics (requires matching geom counts)
        if newton_solver.mjw_model.ngeom == native_mjw_model.ngeom:
            compare_geom_sizes(newton_solver.mjw_model, native_mjw_model)

        # Compare joint ranges only for limited joints (unlimited joints may differ in representation)
        compare_jnt_range(newton_solver.mjw_model, native_mjw_model)

        # Optional: backfill computed fields from native to Newton to eliminate
        # numerical differences from model compilation (enables tighter tolerances for dynamics)
        # Must happen AFTER all model comparisons
        if self.backfill_model:
            backfill_model_from_native(newton_solver.mjw_model, native_mjw_model, self.backfill_fields)

        # Initialize control strategy with the ctrl arrays it will fill
        self.control_strategy.init(native_mjw_data.ctrl, newton_control.mujoco.ctrl)  # type: ignore[union-attr]

        # Setup viewer if in debug mode
        viewer = None
        if self.debug_visual:
            import mujoco.viewer

            view_name = "NEWTON" if self.debug_view_newton else "NATIVE"
            print(f"\n=== VISUAL DEBUG MODE ({view_name}) ===")
            print("Close viewer to exit.\n")

            if self.debug_view_newton:
                viewer = mujoco.viewer.launch_passive(newton_solver.mj_model, newton_solver.mj_data)
            else:
                viewer = mujoco.viewer.launch_passive(mj_model, mj_data_native)

        # Helper: sync mjw_data to mj_data for viewer
        def sync_to_viewer():
            assert _mujoco_warp is not None
            wp.synchronize()
            if self.debug_view_newton:
                _mujoco_warp.get_data_into(newton_solver.mj_data, newton_solver.mj_model, newton_solver.mjw_data)
            else:
                _mujoco_warp.get_data_into(mj_data_native, mj_model, native_mjw_data)

        # Helper: step both simulations (full pipeline mode)
        def step_both_full(step_num: int, newton_graph: Any = None, native_graph: Any = None):
            t = step_num * dt
            self.control_strategy.fill_control(t)  # type: ignore[union-attr]

            if newton_graph and native_graph:
                wp.capture_launch(native_graph)
                wp.capture_launch(newton_graph)
            else:
                _mujoco_warp.step(native_mjw_model, native_mjw_data)  # type: ignore[union-attr]
                newton_solver.step(newton_state, newton_state, newton_control, None, dt)

        # Helper: step with contact injection (split pipeline mode)
        def step_with_contact_injection(
            step_num: int,
            newton_graph: Any = None,
            pre_constraint_graph: Any = None,
            make_constraint_graph: Any = None,
            post_constraint_graph: Any = None,
        ):
            """
            Split pipeline that injects contacts and constraints to bypass non-determinism.

            Flow:
            1. Newton: full step (forward + integrate)
            2. Native: kinematics through collision
            3. Verify contacts match (sorted), inject Newton's contacts
            4. Native: make_constraint
            5. Verify constraints match (sorted), inject Newton's constraints
            6. Native: transmission + step1_rest + step2
            """
            t = step_num * dt
            self.control_strategy.fill_control(t)  # type: ignore[union-attr]

            # 1. Newton full step
            if newton_graph:
                wp.capture_launch(newton_graph)
            else:
                newton_solver.step(newton_state, newton_state, newton_control, None, dt)

            # 2. Native: kinematics through collision
            if pre_constraint_graph:
                wp.capture_launch(pre_constraint_graph)
            else:
                run_native_fwd_position_pre_constraint(native_mjw_model, native_mjw_data)
            wp.synchronize()

            # 3. Verify contacts match (sorted), inject
            contacts_match, contact_msg = compare_contacts_sorted(newton_solver.mjw_data, native_mjw_data, tol=1e-5)
            if not contacts_match:
                raise AssertionError(f"Step {step_num}: Contact mismatch - {contact_msg}")
            inject_contacts(newton_solver.mjw_data, native_mjw_data)

            # 4. Native: make_constraint
            if make_constraint_graph:
                wp.capture_launch(make_constraint_graph)
            else:
                run_native_make_constraint(native_mjw_model, native_mjw_data)
            wp.synchronize()

            # 5. Verify constraints match (sorted), inject
            constraints_match, constraint_msg = compare_constraints_sorted(
                newton_solver.mjw_data, native_mjw_data, tol=1e-5
            )
            if not constraints_match:
                raise AssertionError(f"Step {step_num}: Constraint mismatch - {constraint_msg}")
            inject_constraints(newton_solver.mjw_data, native_mjw_data)

            # 6. Native: transmission + step1_rest + step2
            if post_constraint_graph:
                wp.capture_launch(post_constraint_graph)
            else:
                run_native_transmission(native_mjw_model, native_mjw_data)
                run_native_step1_rest(native_mjw_model, native_mjw_data)
                run_native_step2(native_mjw_model, native_mjw_data)
            wp.synchronize()

        # Helper: compare at step (for non-visual mode)
        def compare_at_step(step_num: int):
            for field_name in self.compare_fields:
                tol = self.tolerances.get(field_name, 1e-6)
                compare_mjdata_field(newton_solver.mjw_data, native_mjw_data, field_name, tol, step_num)

        # Initial viewer sync
        if viewer:
            sync_to_viewer()
            viewer.sync()

        # Setup CUDA graphs
        use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        newton_graph = None
        native_graph = None
        # For contact injection mode:
        pre_constraint_graph = None
        make_constraint_graph = None
        post_constraint_graph = None

        # Compare/print initial state
        if self.debug_visual:
            print("Initial state:")
            print_mjdata_diff(newton_solver.mjw_data, native_mjw_data, self.compare_fields, self.tolerances, -1)
        else:
            compare_at_step(-1)

        # Main simulation loop
        max_steps = 500 if self.debug_visual else self.num_steps

        for step in range(max_steps):
            if viewer and not viewer.is_running():
                break

            if self.use_split_pipeline:
                # Contact+constraint injection: capture graphs on step 0, use thereafter
                if step == 0 and use_cuda_graph:
                    self.control_strategy.fill_control(0.0)  # type: ignore[union-attr]

                    # Capture Newton step
                    with wp.ScopedCapture() as capture:
                        newton_solver.step(newton_state, newton_state, newton_control, None, dt)
                    newton_graph = capture.graph

                    # Capture native kinematics through collision
                    with wp.ScopedCapture() as capture:
                        run_native_fwd_position_pre_constraint(native_mjw_model, native_mjw_data)
                    pre_constraint_graph = capture.graph
                    wp.synchronize()

                    # Compare and inject contacts (not in graph - data-dependent)
                    contacts_match, contact_msg = compare_contacts_sorted(
                        newton_solver.mjw_data, native_mjw_data, tol=1e-5
                    )
                    if not contacts_match:
                        raise AssertionError(f"Step {step}: Contact mismatch - {contact_msg}")
                    inject_contacts(newton_solver.mjw_data, native_mjw_data)

                    # Capture native make_constraint
                    with wp.ScopedCapture() as capture:
                        run_native_make_constraint(native_mjw_model, native_mjw_data)
                    make_constraint_graph = capture.graph
                    wp.synchronize()

                    # Compare and inject constraints (not in graph - data-dependent)
                    constraints_match, constraint_msg = compare_constraints_sorted(
                        newton_solver.mjw_data, native_mjw_data, tol=1e-5
                    )
                    if not constraints_match:
                        raise AssertionError(f"Step {step}: Constraint mismatch - {constraint_msg}")
                    inject_constraints(newton_solver.mjw_data, native_mjw_data)

                    # Capture native transmission + step1_rest + step2
                    with wp.ScopedCapture() as capture:
                        run_native_transmission(native_mjw_model, native_mjw_data)
                        run_native_step1_rest(native_mjw_model, native_mjw_data)
                        run_native_step2(native_mjw_model, native_mjw_data)
                    post_constraint_graph = capture.graph
                    wp.synchronize()
                else:
                    # Use captured graphs (or fallback if not available)
                    step_with_contact_injection(
                        step, newton_graph, pre_constraint_graph, make_constraint_graph, post_constraint_graph
                    )
            elif step == 0 and use_cuda_graph:
                # Step 0: capture CUDA graphs if available (full pipeline mode only)
                self.control_strategy.fill_control(0.0)  # type: ignore[union-attr]

                with wp.ScopedCapture() as capture:
                    newton_solver.step(newton_state, newton_state, newton_control, None, dt)
                newton_graph = capture.graph

                with wp.ScopedCapture() as capture:
                    _mujoco_warp.step(native_mjw_model, native_mjw_data)
                native_graph = capture.graph
            else:
                # Full pipeline mode
                step_both_full(step, newton_graph, native_graph)

            # Viewer sync
            if viewer:
                sync_to_viewer()
                viewer.sync()
                time.sleep(dt)

            # Compare or print diff
            if self.debug_visual:
                if (step + 1) % 50 == 0:
                    print(f"Step {step + 1}:")
                    print_mjdata_diff(
                        newton_solver.mjw_data, native_mjw_data, self.compare_fields, self.tolerances, step
                    )
            else:
                compare_at_step(step)

        # Cleanup
        if viewer:
            print(f"\nFinal ({max_steps} steps):")
            print_mjdata_diff(
                newton_solver.mjw_data, native_mjw_data, self.compare_fields, self.tolerances, max_steps - 1
            )

            while viewer.is_running():
                time.sleep(0.1)
            viewer.close()
            self.skipTest("Visual debug mode completed")


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
        )


class TestMenagerieUSD(TestMenagerieBase):
    """Base class for USD-based tests: Newton loads MJCF converted to USD.

    The MJCF file is converted to USD using mujoco_usd_converter, then
    Newton loads the USD file. Native MuJoCo still loads the original MJCF.
    """

    # All USD tests are skipped until the converter is ready
    skip_reason: str | None = "USD converter not yet implemented"

    def _create_newton_model(self) -> newton.Model:
        """Create Newton model by converting MJCF to USD first."""
        return create_newton_model_from_usd(
            self.mjcf_path,
            num_worlds=self.num_worlds,
            add_ground=False,  # scene.xml includes ground plane
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


class TestMenagerie_AgilexPiper_USD(TestMenagerieUSD):
    """AgileX PIPER bimanual arm. (USD)."""

    robot_folder = "agilex_piper"


class TestMenagerie_ArxL5(TestMenagerieMJCF):
    """ARX L5 arm."""

    robot_folder = "arx_l5"

    skip_reason = "Not yet implemented"


class TestMenagerie_ArxL5_USD(TestMenagerieUSD):
    """ARX L5 arm. (USD)."""

    robot_folder = "arx_l5"


class TestMenagerie_Dynamixel2r(TestMenagerieMJCF):
    """Dynamixel 2R simple arm."""

    robot_folder = "dynamixel_2r"

    skip_reason = "Not yet implemented"


class TestMenagerie_Dynamixel2r_USD(TestMenagerieUSD):
    """Dynamixel 2R simple arm. (USD)."""

    robot_folder = "dynamixel_2r"


class TestMenagerie_FrankaEmikaPanda(TestMenagerieMJCF):
    """Franka Emika Panda arm."""

    robot_folder = "franka_emika_panda"

    skip_reason = "Not yet implemented"


class TestMenagerie_FrankaEmikaPanda_USD(TestMenagerieUSD):
    """Franka Emika Panda arm. (USD)."""

    robot_folder = "franka_emika_panda"


class TestMenagerie_FrankaFr3(TestMenagerieMJCF):
    """Franka FR3 arm."""

    robot_folder = "franka_fr3"

    skip_reason = "Not yet implemented"


class TestMenagerie_FrankaFr3_USD(TestMenagerieUSD):
    """Franka FR3 arm. (USD)."""

    robot_folder = "franka_fr3"


class TestMenagerie_FrankaFr3V2(TestMenagerieMJCF):
    """Franka FR3 v2 arm."""

    robot_folder = "franka_fr3_v2"

    skip_reason = "Not yet implemented"


class TestMenagerie_FrankaFr3V2_USD(TestMenagerieUSD):
    """Franka FR3 v2 arm. (USD)."""

    robot_folder = "franka_fr3_v2"


class TestMenagerie_KinovaGen3(TestMenagerieMJCF):
    """Kinova Gen3 arm."""

    robot_folder = "kinova_gen3"

    skip_reason = "Not yet implemented"


class TestMenagerie_KinovaGen3_USD(TestMenagerieUSD):
    """Kinova Gen3 arm. (USD)."""

    robot_folder = "kinova_gen3"


class TestMenagerie_KukaIiwa14(TestMenagerieMJCF):
    """KUKA iiwa 14 arm."""

    robot_folder = "kuka_iiwa_14"

    skip_reason = "Not yet implemented"


class TestMenagerie_KukaIiwa14_USD(TestMenagerieUSD):
    """KUKA iiwa 14 arm. (USD)."""

    robot_folder = "kuka_iiwa_14"


class TestMenagerie_LowCostRobotArm(TestMenagerieMJCF):
    """Low-cost robot arm."""

    robot_folder = "low_cost_robot_arm"

    skip_reason = "Not yet implemented"


class TestMenagerie_LowCostRobotArm_USD(TestMenagerieUSD):
    """Low-cost robot arm. (USD)."""

    robot_folder = "low_cost_robot_arm"


class TestMenagerie_RethinkSawyer(TestMenagerieMJCF):
    """Rethink Robotics Sawyer arm."""

    robot_folder = "rethink_robotics_sawyer"

    skip_reason = "Not yet implemented"


class TestMenagerie_RethinkSawyer_USD(TestMenagerieUSD):
    """Rethink Robotics Sawyer arm. (USD)."""

    robot_folder = "rethink_robotics_sawyer"


class TestMenagerie_TrossenVx300s(TestMenagerieMJCF):
    """Trossen Robotics ViperX 300 S arm."""

    robot_folder = "trossen_vx300s"

    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenVx300s_USD(TestMenagerieUSD):
    """Trossen Robotics ViperX 300 S arm. (USD)."""

    robot_folder = "trossen_vx300s"


class TestMenagerie_TrossenWx250s(TestMenagerieMJCF):
    """Trossen Robotics WidowX 250 S arm."""

    robot_folder = "trossen_wx250s"

    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenWx250s_USD(TestMenagerieUSD):
    """Trossen Robotics WidowX 250 S arm. (USD)."""

    robot_folder = "trossen_wx250s"


class TestMenagerie_TrossenWxai(TestMenagerieMJCF):
    """Trossen Robotics WidowX AI arm."""

    robot_folder = "trossen_wxai"

    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenWxai_USD(TestMenagerieUSD):
    """Trossen Robotics WidowX AI arm. (USD)."""

    robot_folder = "trossen_wxai"


class TestMenagerie_TrsSoArm100(TestMenagerieMJCF):
    """TRS SO-ARM100 arm."""

    robot_folder = "trs_so_arm100"

    skip_reason = "Not yet implemented"


class TestMenagerie_TrsSoArm100_USD(TestMenagerieUSD):
    """TRS SO-ARM100 arm. (USD)."""

    robot_folder = "trs_so_arm100"


class TestMenagerie_UfactoryLite6(TestMenagerieMJCF):
    """UFACTORY Lite 6 arm."""

    robot_folder = "ufactory_lite6"

    skip_reason = "Not yet implemented"


class TestMenagerie_UfactoryLite6_USD(TestMenagerieUSD):
    """UFACTORY Lite 6 arm. (USD)."""

    robot_folder = "ufactory_lite6"


class TestMenagerie_UfactoryXarm7(TestMenagerieMJCF):
    """UFACTORY xArm 7 arm."""

    robot_folder = "ufactory_xarm7"

    skip_reason = "Not yet implemented"


class TestMenagerie_UfactoryXarm7_USD(TestMenagerieUSD):
    """UFACTORY xArm 7 arm. (USD)."""

    robot_folder = "ufactory_xarm7"


class TestMenagerie_UniversalRobotsUr5e(TestMenagerieMJCF):
    """Universal Robots UR5e arm."""

    robot_folder = "universal_robots_ur5e"

    control_strategy = StructuredControlStrategy(seed=42)
    num_worlds = 34
    num_steps = 500

    # Backfill eliminates model compilation differences (inertia re-diagonalization).
    # Contact injection bypasses non-deterministic contact ordering in broadphase.
    # Constraint injection bypasses non-deterministic efc row ordering in make_constraint.
    # Together these give bit-identical results, so no tolerance overrides needed.
    backfill_model = True
    use_split_pipeline = True
    model_skip_fields = DEFAULT_MODEL_SKIP_FIELDS | {
        "actuator_acc0",  # derived from mass matrix + actuator moment; differs due to inertia re-diag
    }


class TestMenagerie_UniversalRobotsUr5e_USD(TestMenagerieUSD):
    """Universal Robots UR5e arm (USD)."""

    robot_folder = "universal_robots_ur5e"


class TestMenagerie_UniversalRobotsUr10e(TestMenagerieMJCF):
    """Universal Robots UR10e arm."""

    robot_folder = "universal_robots_ur10e"

    skip_reason = "Not yet implemented"


class TestMenagerie_UniversalRobotsUr10e_USD(TestMenagerieUSD):
    """Universal Robots UR10e arm. (USD)."""

    robot_folder = "universal_robots_ur10e"


# -----------------------------------------------------------------------------
# Grippers / Hands (9 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_LeapHand(TestMenagerieMJCF):
    """LEAP Hand."""

    robot_folder = "leap_hand"

    skip_reason = "Not yet implemented"


class TestMenagerie_LeapHand_USD(TestMenagerieUSD):
    """LEAP Hand. (USD)."""

    robot_folder = "leap_hand"


class TestMenagerie_Robotiq2f85(TestMenagerieMJCF):
    """Robotiq 2F-85 gripper."""

    robot_folder = "robotiq_2f85"

    skip_reason = "Not yet implemented"


class TestMenagerie_Robotiq2f85_USD(TestMenagerieUSD):
    """Robotiq 2F-85 gripper. (USD)."""

    robot_folder = "robotiq_2f85"


class TestMenagerie_Robotiq2f85V4(TestMenagerieMJCF):
    """Robotiq 2F-85 gripper v4."""

    robot_folder = "robotiq_2f85_v4"

    skip_reason = "Not yet verified"


class TestMenagerie_Robotiq2f85V4_USD(TestMenagerieUSD):
    """Robotiq 2F-85 gripper v4. (USD)."""

    robot_folder = "robotiq_2f85_v4"


class TestMenagerie_ShadowDexee(TestMenagerieMJCF):
    """Shadow DEX-EE hand."""

    robot_folder = "shadow_dexee"

    skip_reason = "Not yet implemented"


class TestMenagerie_ShadowDexee_USD(TestMenagerieUSD):
    """Shadow DEX-EE hand. (USD)."""

    robot_folder = "shadow_dexee"


class TestMenagerie_ShadowHand(TestMenagerieMJCF):
    """Shadow Hand."""

    robot_folder = "shadow_hand"

    skip_reason = "Not yet verified"


class TestMenagerie_ShadowHand_USD(TestMenagerieUSD):
    """Shadow Hand. (USD)."""

    robot_folder = "shadow_hand"


class TestMenagerie_TetheriaAeroHandOpen(TestMenagerieMJCF):
    """Tetheria Aero Hand (open)."""

    robot_folder = "tetheria_aero_hand_open"

    skip_reason = "Not yet implemented"


class TestMenagerie_TetheriaAeroHandOpen_USD(TestMenagerieUSD):
    """Tetheria Aero Hand (open). (USD)."""

    robot_folder = "tetheria_aero_hand_open"


class TestMenagerie_UmiGripper(TestMenagerieMJCF):
    """UMI Gripper."""

    robot_folder = "umi_gripper"

    skip_reason = "Not yet implemented"


class TestMenagerie_UmiGripper_USD(TestMenagerieUSD):
    """UMI Gripper. (USD)."""

    robot_folder = "umi_gripper"


class TestMenagerie_WonikAllegro(TestMenagerieMJCF):
    """Wonik Allegro Hand."""

    robot_folder = "wonik_allegro"

    skip_reason = "Not yet verified"


class TestMenagerie_WonikAllegro_USD(TestMenagerieUSD):
    """Wonik Allegro Hand. (USD)."""

    robot_folder = "wonik_allegro"


class TestMenagerie_IitSoftfoot(TestMenagerieMJCF):
    """IIT Softfoot biomechanical gripper."""

    robot_folder = "iit_softfoot"

    skip_reason = "Not yet implemented"


class TestMenagerie_IitSoftfoot_USD(TestMenagerieUSD):
    """IIT Softfoot biomechanical gripper. (USD)."""

    robot_folder = "iit_softfoot"


# -----------------------------------------------------------------------------
# Bimanual Systems (2 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_Aloha(TestMenagerieMJCF):
    """ALOHA bimanual system."""

    robot_folder = "aloha"

    skip_reason = "Not yet implemented"


class TestMenagerie_Aloha_USD(TestMenagerieUSD):
    """ALOHA bimanual system. (USD)."""

    robot_folder = "aloha"


class TestMenagerie_GoogleRobot(TestMenagerieMJCF):
    """Google Robot (bimanual)."""

    robot_folder = "google_robot"

    skip_reason = "Not yet implemented"


class TestMenagerie_GoogleRobot_USD(TestMenagerieUSD):
    """Google Robot (bimanual). (USD)."""

    robot_folder = "google_robot"


# -----------------------------------------------------------------------------
# Mobile Manipulators (5 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_HelloRobotStretch(TestMenagerieMJCF):
    """Hello Robot Stretch."""

    robot_folder = "hello_robot_stretch"

    skip_reason = "Not yet implemented"


class TestMenagerie_HelloRobotStretch_USD(TestMenagerieUSD):
    """Hello Robot Stretch. (USD)."""

    robot_folder = "hello_robot_stretch"


class TestMenagerie_HelloRobotStretch3(TestMenagerieMJCF):
    """Hello Robot Stretch 3."""

    robot_folder = "hello_robot_stretch_3"

    skip_reason = "Not yet implemented"


class TestMenagerie_HelloRobotStretch3_USD(TestMenagerieUSD):
    """Hello Robot Stretch 3. (USD)."""

    robot_folder = "hello_robot_stretch_3"


class TestMenagerie_PalTiago(TestMenagerieMJCF):
    """PAL Robotics TIAGo."""

    robot_folder = "pal_tiago"

    skip_reason = "Not yet implemented"


class TestMenagerie_PalTiago_USD(TestMenagerieUSD):
    """PAL Robotics TIAGo. (USD)."""

    robot_folder = "pal_tiago"


class TestMenagerie_PalTiagoDual(TestMenagerieMJCF):
    """PAL Robotics TIAGo Dual."""

    robot_folder = "pal_tiago_dual"

    skip_reason = "Not yet implemented"


class TestMenagerie_PalTiagoDual_USD(TestMenagerieUSD):
    """PAL Robotics TIAGo Dual. (USD)."""

    robot_folder = "pal_tiago_dual"


class TestMenagerie_StanfordTidybot(TestMenagerieMJCF):
    """Stanford Tidybot mobile manipulator."""

    robot_folder = "stanford_tidybot"

    skip_reason = "Not yet implemented"


class TestMenagerie_StanfordTidybot_USD(TestMenagerieUSD):
    """Stanford Tidybot mobile manipulator. (USD)."""

    robot_folder = "stanford_tidybot"


# -----------------------------------------------------------------------------
# Humanoids (10 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_ApptronikApollo(TestMenagerieMJCF):
    """Apptronik Apollo humanoid."""

    robot_folder = "apptronik_apollo"
    backfill_model = True
    model_skip_fields = DEFAULT_MODEL_SKIP_FIELDS | {
        "body_geomadr",  # visual geom count differs (Newton includes visual geoms)
        "body_geomnum",
        "body_invweight0",  # differs due to inertia re-diagonalization
        "dof_invweight0",
        "ngeom",  # Newton includes visual geoms → different geom count
        "geom_",  # all geom-indexed fields have shape mismatch due to visual geoms
        "nsite",  # Newton doesn't parse sites from MJCF
        "site_",
        "mesh_",  # Newton doesn't pass meshes to MuJoCo spec
        "nmesh",
        "pair_geom",  # collision pair geom indices differ due to geom count
        "nxn_",  # broadphase pairs differ due to geom count
        "opt.iterations",  # Newton doesn't parse <option iterations/ls_iterations> from MJCF
        "opt.ls_iterations",
        "opt.timestep",  # Newton doesn't parse <option timestep> from MJCF
        "stat",  # meaninertia differs due to inertia re-diagonalization
        "nmaxpolygon",  # mesh-related (Newton doesn't pass meshes)
        "nmaxmeshdeg",
        "body_tree",  # tuple comparison; content equivalent but objects differ
        "qLD_updates",
        "compare_inertia",  # zero-mass bodies cause large inertia reconstruction diffs
        "actuator_acc0",  # derived from mass matrix + actuator moment; differs due to inertia
    }


class TestMenagerie_ApptronikApollo_USD(TestMenagerieUSD):
    """Apptronik Apollo humanoid. (USD)."""

    robot_folder = "apptronik_apollo"


class TestMenagerie_BerkeleyHumanoid(TestMenagerieMJCF):
    """Berkeley Humanoid."""

    robot_folder = "berkeley_humanoid"

    skip_reason = "Not yet implemented"


class TestMenagerie_BerkeleyHumanoid_USD(TestMenagerieUSD):
    """Berkeley Humanoid. (USD)."""

    robot_folder = "berkeley_humanoid"


class TestMenagerie_BoosterT1(TestMenagerieMJCF):
    """Booster Robotics T1 humanoid."""

    robot_folder = "booster_t1"

    skip_reason = "Not yet verified"


class TestMenagerie_BoosterT1_USD(TestMenagerieUSD):
    """Booster Robotics T1 humanoid. (USD)."""

    robot_folder = "booster_t1"


class TestMenagerie_FourierN1(TestMenagerieMJCF):
    """Fourier N1 humanoid."""

    robot_folder = "fourier_n1"

    skip_reason = "Not yet implemented"


class TestMenagerie_FourierN1_USD(TestMenagerieUSD):
    """Fourier N1 humanoid. (USD)."""

    robot_folder = "fourier_n1"


class TestMenagerie_PalTalos(TestMenagerieMJCF):
    """PAL Robotics TALOS humanoid."""

    robot_folder = "pal_talos"

    skip_reason = "Not yet implemented"


class TestMenagerie_PalTalos_USD(TestMenagerieUSD):
    """PAL Robotics TALOS humanoid. (USD)."""

    robot_folder = "pal_talos"


class TestMenagerie_PndboticsAdamLite(TestMenagerieMJCF):
    """PNDbotics Adam Lite humanoid."""

    robot_folder = "pndbotics_adam_lite"

    skip_reason = "Not yet implemented"


class TestMenagerie_PndboticsAdamLite_USD(TestMenagerieUSD):
    """PNDbotics Adam Lite humanoid. (USD)."""

    robot_folder = "pndbotics_adam_lite"


class TestMenagerie_RobotisOp3(TestMenagerieMJCF):
    """Robotis OP3 humanoid."""

    robot_folder = "robotis_op3"

    skip_reason = "Not yet implemented"


class TestMenagerie_RobotisOp3_USD(TestMenagerieUSD):
    """Robotis OP3 humanoid. (USD)."""

    robot_folder = "robotis_op3"


class TestMenagerie_ToddlerBot2xc(TestMenagerieMJCF):
    """ToddlerBot 2XC humanoid."""

    robot_folder = "toddlerbot_2xc"

    skip_reason = "Not yet implemented"


class TestMenagerie_ToddlerBot2xc_USD(TestMenagerieUSD):
    """ToddlerBot 2XC humanoid. (USD)."""

    robot_folder = "toddlerbot_2xc"


class TestMenagerie_ToddlerBot2xm(TestMenagerieMJCF):
    """ToddlerBot 2XM humanoid."""

    robot_folder = "toddlerbot_2xm"

    skip_reason = "Not yet implemented"


class TestMenagerie_ToddlerBot2xm_USD(TestMenagerieUSD):
    """ToddlerBot 2XM humanoid. (USD)."""

    robot_folder = "toddlerbot_2xm"


class TestMenagerie_UnitreeG1(TestMenagerieMJCF):
    """Unitree G1 humanoid."""

    robot_folder = "unitree_g1"

    skip_reason = "Not yet verified"


class TestMenagerie_UnitreeG1_USD(TestMenagerieUSD):
    """Unitree G1 humanoid. (USD)."""

    robot_folder = "unitree_g1"


class TestMenagerie_UnitreeH1(TestMenagerieMJCF):
    """Unitree H1 humanoid."""

    robot_folder = "unitree_h1"

    skip_reason = "Not yet verified"


class TestMenagerie_UnitreeH1_USD(TestMenagerieUSD):
    """Unitree H1 humanoid. (USD)."""

    robot_folder = "unitree_h1"


# -----------------------------------------------------------------------------
# Bipeds (1 robot)
# -----------------------------------------------------------------------------


class TestMenagerie_AgilityCassie(TestMenagerieMJCF):
    """Agility Robotics Cassie biped."""

    robot_folder = "agility_cassie"

    skip_reason = "Not yet implemented"


class TestMenagerie_AgilityCassie_USD(TestMenagerieUSD):
    """Agility Robotics Cassie biped. (USD)."""

    robot_folder = "agility_cassie"


# -----------------------------------------------------------------------------
# Quadrupeds (8 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_AnyboticsAnymalB(TestMenagerieMJCF):
    """ANYbotics ANYmal B quadruped."""

    robot_folder = "anybotics_anymal_b"

    skip_reason = "Not yet implemented"


class TestMenagerie_AnyboticsAnymalB_USD(TestMenagerieUSD):
    """ANYbotics ANYmal B quadruped. (USD)."""

    robot_folder = "anybotics_anymal_b"


class TestMenagerie_AnyboticsAnymalC(TestMenagerieMJCF):
    """ANYbotics ANYmal C quadruped."""

    robot_folder = "anybotics_anymal_c"

    skip_reason = "Not yet implemented"


class TestMenagerie_AnyboticsAnymalC_USD(TestMenagerieUSD):
    """ANYbotics ANYmal C quadruped. (USD)."""

    robot_folder = "anybotics_anymal_c"


class TestMenagerie_BostonDynamicsSpot(TestMenagerieMJCF):
    """Boston Dynamics Spot quadruped."""

    robot_folder = "boston_dynamics_spot"

    skip_reason = "Not yet implemented"


class TestMenagerie_BostonDynamicsSpot_USD(TestMenagerieUSD):
    """Boston Dynamics Spot quadruped. (USD)."""

    robot_folder = "boston_dynamics_spot"


class TestMenagerie_GoogleBarkourV0(TestMenagerieMJCF):
    """Google Barkour v0 quadruped."""

    robot_folder = "google_barkour_v0"

    skip_reason = "Not yet implemented"


class TestMenagerie_GoogleBarkourV0_USD(TestMenagerieUSD):
    """Google Barkour v0 quadruped. (USD)."""

    robot_folder = "google_barkour_v0"


class TestMenagerie_GoogleBarkourVb(TestMenagerieMJCF):
    """Google Barkour vB quadruped."""

    robot_folder = "google_barkour_vb"

    skip_reason = "Not yet implemented"


class TestMenagerie_GoogleBarkourVb_USD(TestMenagerieUSD):
    """Google Barkour vB quadruped. (USD)."""

    robot_folder = "google_barkour_vb"


class TestMenagerie_UnitreeA1(TestMenagerieMJCF):
    """Unitree A1 quadruped."""

    robot_folder = "unitree_a1"

    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeA1_USD(TestMenagerieUSD):
    """Unitree A1 quadruped. (USD)."""

    robot_folder = "unitree_a1"


class TestMenagerie_UnitreeGo1(TestMenagerieMJCF):
    """Unitree Go1 quadruped."""

    robot_folder = "unitree_go1"

    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeGo1_USD(TestMenagerieUSD):
    """Unitree Go1 quadruped. (USD)."""

    robot_folder = "unitree_go1"


class TestMenagerie_UnitreeGo2(TestMenagerieMJCF):
    """Unitree Go2 quadruped."""

    robot_folder = "unitree_go2"

    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeGo2_USD(TestMenagerieUSD):
    """Unitree Go2 quadruped. (USD)."""

    robot_folder = "unitree_go2"


# -----------------------------------------------------------------------------
# Arms with Gripper (Unitree Z1)
# -----------------------------------------------------------------------------


class TestMenagerie_UnitreeZ1(TestMenagerieMJCF):
    """Unitree Z1 arm."""

    robot_folder = "unitree_z1"

    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeZ1_USD(TestMenagerieUSD):
    """Unitree Z1 arm. (USD)."""

    robot_folder = "unitree_z1"


# -----------------------------------------------------------------------------
# Drones (2 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_BitcrazeCrazyflie2(TestMenagerieMJCF):
    """Bitcraze Crazyflie 2 quadrotor."""

    robot_folder = "bitcraze_crazyflie_2"

    skip_reason = "Not yet implemented"


class TestMenagerie_BitcrazeCrazyflie2_USD(TestMenagerieUSD):
    """Bitcraze Crazyflie 2 quadrotor. (USD)."""

    robot_folder = "bitcraze_crazyflie_2"


class TestMenagerie_SkydioX2(TestMenagerieMJCF):
    """Skydio X2 drone."""

    robot_folder = "skydio_x2"

    skip_reason = "Not yet implemented"


class TestMenagerie_SkydioX2_USD(TestMenagerieUSD):
    """Skydio X2 drone. (USD)."""

    robot_folder = "skydio_x2"


# -----------------------------------------------------------------------------
# Mobile Bases (2 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_RobotSoccerKit(TestMenagerieMJCF):
    """Robot Soccer Kit omniwheel base."""

    robot_folder = "robot_soccer_kit"

    skip_reason = "Not yet implemented"


class TestMenagerie_RobotSoccerKit_USD(TestMenagerieUSD):
    """Robot Soccer Kit omniwheel base. (USD)."""

    robot_folder = "robot_soccer_kit"


class TestMenagerie_RobotstudioSo101(TestMenagerieMJCF):
    """RobotStudio SO-101."""

    robot_folder = "robotstudio_so101"

    skip_reason = "Not yet implemented"


class TestMenagerie_RobotstudioSo101_USD(TestMenagerieUSD):
    """RobotStudio SO-101. (USD)."""

    robot_folder = "robotstudio_so101"


# -----------------------------------------------------------------------------
# Biomechanical (1 robot)
# -----------------------------------------------------------------------------


class TestMenagerie_Flybody(TestMenagerieMJCF):
    """Flybody fruit fly model."""

    robot_folder = "flybody"

    skip_reason = "Not yet implemented"


class TestMenagerie_Flybody_USD(TestMenagerieUSD):
    """Flybody fruit fly model. (USD)."""

    robot_folder = "flybody"


# -----------------------------------------------------------------------------
# Other (1 robot)
# -----------------------------------------------------------------------------


class TestMenagerie_I2rtYam(TestMenagerieMJCF):
    """i2rt YAM (Yet Another Manipulator)."""

    robot_folder = "i2rt_yam"

    skip_reason = "Not yet implemented"


class TestMenagerie_I2rtYam_USD(TestMenagerieUSD):
    """i2rt YAM (Yet Another Manipulator). (USD)."""

    robot_folder = "i2rt_yam"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
