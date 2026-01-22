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
    - TestMenagerie_<RobotName>: One derived class per menagerie robot

Each test:
    1. Downloads the robot from menagerie (cached)
    2. Creates Newton model (via MJCF or USD factory)
    3. Creates native MuJoCo model from same source
    4. Runs simulation with configurable control strategies
    5. Compares per-step state values within tolerance
"""

from __future__ import annotations

import unittest
from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

import newton
from newton._src.utils.download_assets import download_git_folder
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


class ModelSourceType(Enum):
    """Type of model source for Newton model creation."""

    MJCF = auto()  # Load directly from MJCF
    USD = auto()  # Convert MJCF to USD, then load (future)


def create_newton_model_from_mjcf(
    mjcf_path: Path,
    *,
    floating: bool = True,
    num_worlds: int = 1,
    add_ground: bool = True,
) -> newton.Model:
    """
    Create a Newton model from an MJCF file.

    Args:
        mjcf_path: Path to the MJCF XML file
        floating: Whether the robot has a floating base
        num_worlds: Number of world instances to create
        add_ground: Whether to add a ground plane

    Returns:
        Finalized Newton Model
    """
    # Create articulation builder for the robot
    robot_builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(robot_builder)

    robot_builder.add_mjcf(
        str(mjcf_path),
        floating=floating,
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
    floating: bool = True,
    num_worlds: int = 1,
    add_ground: bool = True,
) -> newton.Model:
    """
    Create a Newton model by converting MJCF to USD first.

    NOTE: This is a placeholder for future USD converter integration.

    Args:
        mjcf_path: Path to the MJCF XML file
        floating: Whether the robot has a floating base
        num_worlds: Number of world instances to create
        add_ground: Whether to add a ground plane

    Returns:
        Finalized Newton Model
    """
    raise NotImplementedError(
        "USD conversion path not yet implemented. Waiting for MuJoCo USD converter to be finalized."
    )


def create_newton_model(
    mjcf_path: Path,
    source_type: ModelSourceType = ModelSourceType.MJCF,
    **kwargs,
) -> newton.Model:
    """
    Factory function to create Newton model from various sources.

    Args:
        mjcf_path: Path to the source MJCF file
        source_type: How to create the model (MJCF direct or via USD)
        **kwargs: Passed to the specific creation function

    Returns:
        Finalized Newton Model
    """
    if source_type == ModelSourceType.MJCF:
        return create_newton_model_from_mjcf(mjcf_path, **kwargs)
    elif source_type == ModelSourceType.USD:
        return create_newton_model_from_usd(mjcf_path, **kwargs)
    else:
        raise ValueError(f"Unknown model source type: {source_type}")


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
    def get_control(
        self,
        t: float,
        step: int,
        num_worlds: int,
        num_actuators: int,
    ) -> np.ndarray:
        """
        Generate control values for the given timestep.

        Args:
            t: Current simulation time
            step: Current step number
            num_worlds: Number of parallel worlds
            num_actuators: Number of actuators per world

        Returns:
            Control array of shape (num_worlds, num_actuators)
        """
        pass


class ZeroControlStrategy(ControlStrategy):
    """Always returns zero control - useful for drop tests."""

    def get_control(self, t, step, num_worlds, num_actuators):
        return np.zeros((num_worlds, num_actuators))


class StructuredControlStrategy(ControlStrategy):
    """
    Generate structured control patterns designed to explore joint limits.

    Combines sinusoids at different frequencies with occasional step changes.
    Each world gets a slightly different phase offset for variation.
    """

    def __init__(
        self,
        seed: int = 42,
        amplitude_scale: float = 0.8,
        frequency_range: tuple[float, float] = (0.5, 2.0),
        step_probability: float = 0.05,
    ):
        super().__init__(seed)
        self.amplitude_scale = amplitude_scale
        self.frequency_range = frequency_range
        self.step_probability = step_probability
        self._frequencies: np.ndarray | None = None
        self._phases: np.ndarray | None = None  # Shape: (num_worlds, num_actuators)
        self._step_values: np.ndarray | None = None

    def _init_for_model(self, num_worlds: int, num_actuators: int):
        """Initialize frequencies and phases."""
        self._frequencies = self.rng.uniform(self.frequency_range[0], self.frequency_range[1], num_actuators)
        # Different phase per world for variation
        self._phases = self.rng.uniform(0, 2 * np.pi, (num_worlds, num_actuators))
        self._step_values = np.zeros((num_worlds, num_actuators))

    def get_control(self, t, step, num_worlds, num_actuators):
        if self._frequencies is None or self._phases is None or self._phases.shape != (num_worlds, num_actuators):
            self._init_for_model(num_worlds, num_actuators)

        assert self._frequencies is not None
        assert self._phases is not None
        assert self._step_values is not None

        # Base sinusoidal pattern: broadcast frequencies across worlds
        # Shape: (num_worlds, num_actuators)
        control = self.amplitude_scale * np.sin(2 * np.pi * self._frequencies[np.newaxis, :] * t + self._phases)

        # Occasional step changes (random world and actuator)
        if self.rng.random() < self.step_probability:
            world_idx = self.rng.integers(0, num_worlds)
            act_idx = self.rng.integers(0, num_actuators)
            self._step_values[world_idx, act_idx] = self.rng.uniform(-1, 1)

        control += 0.3 * self._step_values

        return np.clip(control, -1, 1)


class RandomControlStrategy(ControlStrategy):
    """
    Generate random control values with configurable noise.

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
        self._prev_control: np.ndarray | None = None

    def get_control(self, t, step, num_worlds, num_actuators):
        if self._prev_control is None or self._prev_control.shape != (num_worlds, num_actuators):
            self._prev_control = np.zeros((num_worlds, num_actuators))

        # Random noise with smoothing (independent per world)
        noise = self.rng.uniform(-1, 1, (num_worlds, num_actuators)) * self.noise_scale
        control = self.smoothing * self._prev_control + (1 - self.smoothing) * noise
        self._prev_control = control

        return np.clip(control, -1, 1)


# =============================================================================
# Comparison
# =============================================================================

# Default tolerances for MjData field comparison
DEFAULT_TOLERANCES: dict[str, float] = {
    "qpos": 1e-6,
    "qvel": 1e-5,
    "qacc": 1e-4,
    "xpos": 1e-6,
    "xquat": 1e-6,
    "ctrl": 1e-10,
}

# Default fields to compare
DEFAULT_COMPARE_FIELDS: list[str] = ["qpos", "qvel"]


def compare_mjdata(
    newton_mjw_data: Any,
    native_mjw_data: Any,
    step: int,
    world_idx: int = 0,
    fields: list[str] | None = None,
    tolerances: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """
    Compare MjWarpData from Newton's SolverMuJoCo against native mujoco_warp.

    Both sides have the same batched structure: arrays have shape [num_worlds, ...].

    Args:
        newton_mjw_data: MjWarpData from SolverMuJoCo
        native_mjw_data: MjWarpData from native mujoco_warp
        step: Current step number
        world_idx: Which world to compare
        fields: Fields to compare (default: ["qpos", "qvel"])
        tolerances: Per-field tolerances (default: see DEFAULT_TOLERANCES)

    Returns:
        (passed, errors) tuple where errors is a list of error messages
    """
    fields = fields or DEFAULT_COMPARE_FIELDS
    tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}

    errors: list[str] = []

    for field_name in fields:
        tol = tolerances.get(field_name, 1e-6)

        newton_arr = getattr(newton_mjw_data, field_name, None)
        native_arr = getattr(native_mjw_data, field_name, None)

        if newton_arr is None or native_arr is None:
            continue

        # Convert to numpy if needed (warp arrays)
        if hasattr(newton_arr, "numpy"):
            newton_arr = newton_arr.numpy()
        if hasattr(native_arr, "numpy"):
            native_arr = native_arr.numpy()

        newton_arr = np.asarray(newton_arr)
        native_arr = np.asarray(native_arr)

        # Select the specific world (first dimension is world index)
        newton_world = newton_arr[world_idx].flatten()
        native_world = native_arr[world_idx].flatten()

        if len(newton_world) != len(native_world):
            errors.append(f"step {step}, {field_name}: size mismatch ({len(newton_world)} vs {len(native_world)})")
            continue

        if len(newton_world) == 0:
            continue

        max_error = float(np.max(np.abs(newton_world - native_world)))
        if max_error > tol:
            errors.append(f"step {step}, {field_name}: max error {max_error:.2e} > {tol:.2e}")

    return len(errors) == 0, errors


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
        - floating: bool - whether robot has floating base

    Optional overrides:
        - num_worlds: int - number of parallel worlds (default: 2)
        - num_steps: int - simulation steps to run (default: 100)
        - dt: float - timestep (default: 0.002)
        - control_strategy: ControlStrategy - how to generate controls
        - compare_fields: list[str] - MjData fields to compare
        - tolerances: dict[str, float] - per-field tolerances
        - skip_reason: str | None - if set, skip this test
    """

    # Must be defined by subclasses
    robot_folder: str = ""
    robot_xml: str = ""
    floating: bool = True

    # Configurable defaults
    num_worlds: int = 2
    num_steps: int = 100
    dt: float = 0.002

    # Control strategy (can override in subclass)
    control_strategy: ControlStrategy | None = None

    # Comparison settings (override in subclass as needed)
    compare_fields: list[str] = DEFAULT_COMPARE_FIELDS
    tolerances: dict[str, float] = DEFAULT_TOLERANCES

    # Skip reason (set to None to enable test)
    skip_reason: str | None = "Not yet implemented"

    # Model source type (for parametrized testing)
    model_source_type: ModelSourceType = ModelSourceType.MJCF

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

    def _create_newton_model(self) -> newton.Model:
        """Create Newton model using the factory."""
        return create_newton_model(
            self.mjcf_path,
            source_type=self.model_source_type,
            floating=self.floating,
            num_worlds=self.num_worlds,
            add_ground=True,
        )

    def _create_native_mujoco_warp(self) -> tuple[Any, Any, Any, Any]:
        """Create native mujoco_warp model/data from the same MJCF.

        Returns:
            (mj_model, mj_data, mjw_model, mjw_data) tuple
        """
        assert _mujoco is not None
        assert _mujoco_warp is not None

        # Create base MuJoCo model/data (uses default initialization)
        mj_model = _mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        mj_data = _mujoco.MjData(mj_model)
        _mujoco.mj_forward(mj_model, mj_data)

        # Create mujoco_warp model/data with multiple worlds
        mjw_model = _mujoco_warp.put_model(mj_model)
        mjw_data = _mujoco_warp.put_data(mj_model, mj_data, nworld=self.num_worlds)

        return mj_model, mj_data, mjw_model, mjw_data

    def test_simulation_equivalence(self):
        """
        Main test: verify Newton's SolverMuJoCo and native mujoco_warp produce equivalent results.

        Both sides use mujoco_warp with the same number of worlds.
        Each world receives different controls (varied by phase/noise).
        """
        assert _mujoco is not None
        assert _mujoco_warp is not None
        assert self.control_strategy is not None

        # Create Newton model and solver (num_worlds is batch dimension)
        newton_model = self._create_newton_model()
        newton_state_0 = newton_model.state()
        newton_state_1 = newton_model.state()
        newton_control = newton_model.control()

        # Create Newton's MuJoCo solver (uses warp backend with batched worlds)
        newton_solver = SolverMuJoCo(newton_model)

        # Create native mujoco_warp model/data with same number of worlds
        _, _, native_mjw_model, native_mjw_data = self._create_native_mujoco_warp()

        # Get number of actuators from native model
        num_actuators = native_mjw_data.ctrl.shape[1] if native_mjw_data.ctrl.shape[1] > 0 else 0

        # Run simulation and compare
        all_errors: list[str] = []

        for step in range(self.num_steps):
            t = step * self.dt

            # Generate control: shape (num_worlds, num_actuators)
            ctrl = self.control_strategy.get_control(t, step, self.num_worlds, num_actuators)

            # Apply same controls to both sides
            newton_solver.mjw_data.ctrl.assign(ctrl)
            native_mjw_data.ctrl.assign(ctrl)

            # Step both using mujoco_warp
            _mujoco_warp.step(native_mjw_model, native_mjw_data)
            newton_solver.step(newton_state_0, newton_state_1, newton_control, None, self.dt)
            newton_state_0, newton_state_1 = newton_state_1, newton_state_0

            # Compare mjw_data arrays directly (both have same shape)
            for world_idx in range(self.num_worlds):
                passed, errors = compare_mjdata(
                    newton_solver.mjw_data,
                    native_mjw_data,
                    step,
                    world_idx,
                    fields=self.compare_fields,
                    tolerances=self.tolerances,
                )
                if not passed:
                    all_errors.extend([f"world {world_idx}: {e}" for e in errors])

        # Report failures
        if all_errors:
            msg = "\n".join(all_errors[:20])
            if len(all_errors) > 20:
                msg += f"\n... and {len(all_errors) - 20} more errors"
            self.fail(f"Simulation mismatch:\n{msg}")


# =============================================================================
# Robot Test Classes
# =============================================================================
# Each robot from the menagerie gets its own test class.
# Initially all are skipped; enable as support is verified.


# -----------------------------------------------------------------------------
# Arms
# -----------------------------------------------------------------------------


class TestMenagerie_AgilexPiper(TestMenagerieBase):
    """AgileX PIPER arm."""

    robot_folder = "agilex_piper"
    robot_xml = "piper.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_ArxL5(TestMenagerieBase):
    """ARX L5 arm."""

    robot_folder = "arx_l5"
    robot_xml = "arx_l5.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_FrankaEmikaPanda(TestMenagerieBase):
    """Franka Emika Panda arm."""

    robot_folder = "franka_emika_panda"
    robot_xml = "panda.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_GoogleRobot(TestMenagerieBase):
    """Google Robot arm."""

    robot_folder = "google_robot"
    robot_xml = "robot.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_KukaIiwa14(TestMenagerieBase):
    """KUKA iiwa 14 arm."""

    robot_folder = "kuka_iiwa_14"
    robot_xml = "iiwa14.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_RethinkSawyer(TestMenagerieBase):
    """Rethink Robotics Sawyer arm."""

    robot_folder = "rethink_robotics_sawyer"
    robot_xml = "sawyer.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_Robotiq2f85V3(TestMenagerieBase):
    """Robotiq 2F-85 gripper v3."""

    robot_folder = "robotiq_2f85_v3"
    robot_xml = "2f85.xml"
    floating = False
    skip_reason = "Robot not found in menagerie"


class TestMenagerie_Robotiq2f85V4(TestMenagerieBase):
    """Robotiq 2F-85 gripper v4."""

    robot_folder = "robotiq_2f85_v4"
    robot_xml = "2f85.xml"
    floating = False
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_UniversalRobotsUr5e(TestMenagerieBase):
    """Universal Robots UR5e arm."""

    robot_folder = "universal_robots_ur5e"
    robot_xml = "ur5e.xml"
    floating = False
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_UniversalRobotsUr10e(TestMenagerieBase):
    """Universal Robots UR10e arm."""

    robot_folder = "universal_robots_ur10e"
    robot_xml = "ur10e.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_ViperwX300(TestMenagerieBase):
    """ViperX 300 arm."""

    robot_folder = "trossen_vx300s"
    robot_xml = "vx300s.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_XArm7(TestMenagerieBase):
    """xArm 7 arm."""

    robot_folder = "ufactory_xarm7"
    robot_xml = "xarm7.xml"
    floating = False
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Grippers / Hands
# -----------------------------------------------------------------------------


class TestMenagerie_ShadowDexEe(TestMenagerieBase):
    """Shadow DEX-EE hand."""

    robot_folder = "shadow_dex_ee"
    robot_xml = "shadow_hand.xml"
    floating = False
    skip_reason = "Robot not found in menagerie"


class TestMenagerie_ShadowHand(TestMenagerieBase):
    """Shadow Hand."""

    robot_folder = "shadow_hand"
    robot_xml = "left_hand.xml"
    floating = False
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_TrossenPincherX100(TestMenagerieBase):
    """Trossen Robotics PincherX 100 gripper."""

    robot_folder = "trossen_px100"
    robot_xml = "px100.xml"
    floating = False
    skip_reason = "Robot not found in menagerie"


class TestMenagerie_TrossenWidowX250(TestMenagerieBase):
    """Trossen Robotics WidowX 250 arm."""

    robot_folder = "trossen_wx250s"
    robot_xml = "wx250s.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_WonikAllegro(TestMenagerieBase):
    """Wonik Allegro Hand."""

    robot_folder = "wonik_allegro"
    robot_xml = "left_hand.xml"
    floating = False
    skip_reason = "HIGH PRIORITY - Not yet implemented"


# -----------------------------------------------------------------------------
# Dual Arms
# -----------------------------------------------------------------------------


class TestMenagerie_Aloha2(TestMenagerieBase):
    """ALOHA 2 dual arm system."""

    robot_folder = "aloha"
    robot_xml = "aloha.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenAloha(TestMenagerieBase):
    """Trossen Robotics ALOHA system."""

    robot_folder = "trossen_aloha"
    robot_xml = "aloha.xml"
    floating = False
    skip_reason = "Robot not found in menagerie"


# -----------------------------------------------------------------------------
# Bipeds
# -----------------------------------------------------------------------------


class TestMenagerie_AgilityCassie(TestMenagerieBase):
    """Agility Robotics Cassie biped."""

    robot_folder = "agility_cassie"
    robot_xml = "cassie.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Legged Manipulators
# -----------------------------------------------------------------------------


class TestMenagerie_AnyboticsAnymalDArm(TestMenagerieBase):
    """ANYbotics ANYmal D with arm."""

    robot_folder = "anybotics_anymal_d"
    robot_xml = "anymal_d.xml"
    floating = True
    skip_reason = "Robot not found in menagerie (available in newton-assets)"


class TestMenagerie_BostonDynamicsSpotArm(TestMenagerieBase):
    """Boston Dynamics Spot with arm."""

    robot_folder = "boston_dynamics_spot"
    robot_xml = "spot_arm.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Mobile Manipulators
# -----------------------------------------------------------------------------


class TestMenagerie_GoogleMobileAloha(TestMenagerieBase):
    """Google Mobile ALOHA."""

    robot_folder = "google_mobile_aloha"
    robot_xml = "mobile_aloha.xml"
    floating = True
    skip_reason = "Robot not found in menagerie"


class TestMenagerie_HelloStretch(TestMenagerieBase):
    """Hello Robot Stretch."""

    robot_folder = "hello_robot_stretch"
    robot_xml = "stretch.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_PalTiago(TestMenagerieBase):
    """PAL Robotics TIAGo."""

    robot_folder = "pal_tiago"
    robot_xml = "tiago.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_PalTiagoDual(TestMenagerieBase):
    """PAL Robotics TIAGo Dual."""

    robot_folder = "pal_tiago_dual"
    robot_xml = "tiago_dual.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Mobile Bases
# -----------------------------------------------------------------------------


class TestMenagerie_RobotSoccerKit(TestMenagerieBase):
    """Robot Soccer Kit omniwheel base."""

    robot_folder = "robot_soccer_kit"
    robot_xml = "robot_soccer_kit.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Humanoids
# -----------------------------------------------------------------------------


class TestMenagerie_PndboticsAdamLite(TestMenagerieBase):
    """PNDbotics Adam Lite humanoid."""

    robot_folder = "pndbotics_adam_lite"
    robot_xml = "adam_lite.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_ApptronikApollo(TestMenagerieBase):
    """Apptronik Apollo humanoid."""

    robot_folder = "apptronik_apollo"
    robot_xml = "apptronik_apollo.xml"
    floating = True
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_BerkeleyHumanoid(TestMenagerieBase):
    """Berkeley Humanoid."""

    robot_folder = "berkeley_humanoid"
    robot_xml = "berkeley_humanoid.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_BoosterT1(TestMenagerieBase):
    """Booster Robotics T1 humanoid."""

    robot_folder = "booster_t1"
    robot_xml = "t1.xml"
    floating = True
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_FourierN1(TestMenagerieBase):
    """Fourier N1 humanoid."""

    robot_folder = "fourier_n1"
    robot_xml = "n1.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_RobotisOp3(TestMenagerieBase):
    """Robotis OP3 humanoid."""

    robot_folder = "robotis_op3"
    robot_xml = "op3.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_PalTalos(TestMenagerieBase):
    """PAL Robotics TALOS humanoid."""

    robot_folder = "pal_talos"
    robot_xml = "talos.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeG1(TestMenagerieBase):
    """Unitree G1 humanoid."""

    robot_folder = "unitree_g1"
    robot_xml = "g1.xml"
    floating = True
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_UnitreeH1(TestMenagerieBase):
    """Unitree H1 humanoid."""

    robot_folder = "unitree_h1"
    robot_xml = "h1.xml"
    floating = True
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_ToddlerBot2xc(TestMenagerieBase):
    """Stanford ToddlerBot 2XC humanoid."""

    robot_folder = "toddlerbot_2xc"
    robot_xml = "toddlerbot_2xc.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_ToddlerBot2xm(TestMenagerieBase):
    """Stanford ToddlerBot 2XM humanoid."""

    robot_folder = "toddlerbot_2xm"
    robot_xml = "toddlerbot_2xm.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Quadrupeds
# -----------------------------------------------------------------------------


class TestMenagerie_AnyboticsAnymalB(TestMenagerieBase):
    """ANYbotics ANYmal B quadruped."""

    robot_folder = "anybotics_anymal_b"
    robot_xml = "anymal_b.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_AnyboticsAnymalC(TestMenagerieBase):
    """ANYbotics ANYmal C quadruped."""

    robot_folder = "anybotics_anymal_c"
    robot_xml = "anymal_c.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_BostonDynamicsSpot(TestMenagerieBase):
    """Boston Dynamics Spot quadruped."""

    robot_folder = "boston_dynamics_spot"
    robot_xml = "spot.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeA1(TestMenagerieBase):
    """Unitree A1 quadruped."""

    robot_folder = "unitree_a1"
    robot_xml = "a1.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeGo1(TestMenagerieBase):
    """Unitree Go1 quadruped."""

    robot_folder = "unitree_go1"
    robot_xml = "go1.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_UnitreeGo2(TestMenagerieBase):
    """Unitree Go2 quadruped."""

    robot_folder = "unitree_go2"
    robot_xml = "go2.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_GoogleBarkourV0(TestMenagerieBase):
    """Google Barkour v0 quadruped."""

    robot_folder = "google_barkour_v0"
    robot_xml = "barkour_v0.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_GoogleBarkourVb(TestMenagerieBase):
    """Google Barkour vB quadruped."""

    robot_folder = "google_barkour_vb"
    robot_xml = "barkour_vb.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Biomechanical
# -----------------------------------------------------------------------------


class TestMenagerie_IitSoftfoot(TestMenagerieBase):
    """IIT Softfoot biomechanical model."""

    robot_folder = "iit_softfoot"
    robot_xml = "softfoot.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_Flybody(TestMenagerieBase):
    """Flybody insect model."""

    robot_folder = "flybody"
    robot_xml = "fruitfly.xml"
    floating = True
    skip_reason = "Not yet implemented"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
