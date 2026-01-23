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

import time
import unittest
from abc import abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

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
    ) -> np.ndarray | wp.array:
        """
        Generate control values for the given timestep.

        Args:
            t: Current simulation time
            step: Current step number
            num_worlds: Number of parallel worlds
            num_actuators: Number of actuators per world

        Returns:
            Control array of shape (num_worlds, num_actuators) - numpy or warp array
        """
        pass


class ZeroControlStrategy(ControlStrategy):
    """Always returns zero control - useful for drop tests."""

    def __init__(self):
        super().__init__(seed=0)
        self._ctrl: wp.array | None = None

    def get_control(self, t, step, num_worlds, num_actuators):
        if self._ctrl is None or self._ctrl.shape != (num_worlds, num_actuators):
            self._ctrl = wp.zeros((num_worlds, num_actuators), dtype=wp.float32)
        return self._ctrl


@wp.kernel
def generate_sinusoidal_control_kernel(
    frequencies: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    phases: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    ctrl_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    t: wp.float32,
    amplitude: wp.float32,
    num_actuators: int,
):
    """Generate sinusoidal control pattern on GPU."""
    i = wp.tid()
    act_idx = i % num_actuators  # type: ignore[operator]
    freq = frequencies[act_idx]
    phase = phases[i]  # phases are stored flat: phases[world_idx * num_actuators + act_idx]
    two_pi = 6.28318530717959  # 2 * pi
    val = amplitude * wp.sin(two_pi * freq * t + phase)  # type: ignore[operator]
    ctrl_out[i] = wp.clamp(val, -1.0, 1.0)  # type: ignore[arg-type]


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
        self._ctrl: wp.array | None = None

    def _init_for_model(self, num_worlds: int, num_actuators: int):
        """Initialize frequencies and phases as Warp arrays."""
        frequencies_np = self.rng.uniform(self.frequency_range[0], self.frequency_range[1], num_actuators)
        # Different phase per world for variation - flattened for kernel
        phases_np = self.rng.uniform(0, 2 * np.pi, (num_worlds, num_actuators)).flatten()

        self._frequencies = wp.array(frequencies_np, dtype=wp.float32)
        self._phases = wp.array(phases_np, dtype=wp.float32)
        self._ctrl = wp.zeros(num_worlds * num_actuators, dtype=wp.float32)

    def get_control(self, t, step, num_worlds, num_actuators):
        if self._frequencies is None or self._phases is None or self._ctrl is None:
            self._init_for_model(num_worlds, num_actuators)

        assert self._frequencies is not None
        assert self._phases is not None
        assert self._ctrl is not None

        wp.launch(
            generate_sinusoidal_control_kernel,
            dim=num_worlds * num_actuators,
            inputs=[
                self._frequencies,
                self._phases,
                self._ctrl,
                t,
                self.amplitude_scale,
                num_actuators,
            ],
        )
        return self._ctrl.reshape((num_worlds, num_actuators))


@wp.kernel
def generate_random_control_kernel(
    prev_ctrl: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    ctrl_out: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    seed: int,
    step: int,
    noise_scale: wp.float32,
    smoothing: wp.float32,
):
    """Generate random control with smoothing on GPU."""
    i = wp.tid()
    # Initialize RNG with unique seed per element and step
    state = wp.rand_init(seed, i + step * 1000000)  # type: ignore[arg-type, operator]
    # Generate uniform random in [-1, 1]
    rand_val = wp.randf(state) * 2.0 - 1.0
    noise = rand_val * noise_scale
    # Smooth with previous control
    one_minus_smooth = 1.0 - smoothing
    val = smoothing * prev_ctrl[i] + one_minus_smooth * noise
    ctrl_out[i] = wp.clamp(val, -1.0, 1.0)
    # Update prev for next step
    prev_ctrl[i] = ctrl_out[i]


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
        self._ctrl: wp.array | None = None

    def _init_for_model(self, num_worlds: int, num_actuators: int):
        """Initialize control arrays."""
        self._prev_ctrl = wp.zeros(num_worlds * num_actuators, dtype=wp.float32)
        self._ctrl = wp.zeros(num_worlds * num_actuators, dtype=wp.float32)

    def get_control(self, t, step, num_worlds, num_actuators):
        if self._prev_ctrl is None or self._ctrl is None:
            self._init_for_model(num_worlds, num_actuators)

        assert self._prev_ctrl is not None
        assert self._ctrl is not None

        wp.launch(
            generate_random_control_kernel,
            dim=num_worlds * num_actuators,
            inputs=[
                self._prev_ctrl,
                self._ctrl,
                self.rng.integers(0, 2**31),  # Random seed for this step
                step,
                self.noise_scale,
                self.smoothing,
            ],
        )
        return self._ctrl.reshape((num_worlds, num_actuators))


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

        # Handle warp arrays (have .numpy() method)
        if hasattr(native_val, "numpy"):
            assert newton_val is not None and hasattr(newton_val, "numpy"), f"{attr}: type mismatch"
            native_np: np.ndarray = native_val.numpy()  # type: ignore[union-attr]
            newton_np: np.ndarray = newton_val.numpy()  # type: ignore[union-attr]
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
    num_worlds: int = 16
    num_steps: int = 100
    dt: float = 0.002

    # Control strategy (can override in subclass)
    control_strategy: ControlStrategy | None = None

    # Data comparison: explicit list of fields TO compare
    compare_fields: ClassVar[list[str]] = DEFAULT_COMPARE_FIELDS
    tolerances: ClassVar[dict[str, float]] = DEFAULT_TOLERANCES

    # Model comparison: fields to SKIP (substrings to match)
    model_skip_fields: ClassVar[set[str]] = {"__", "ptr"}

    # Skip reason (set to a string to skip test, leave unset or None to run)
    skip_reason: str | None = None

    # Model source type (for parametrized testing)
    model_source_type: ModelSourceType = ModelSourceType.MJCF

    # Debug mode: opens viewer for visual debugging
    debug_visual: bool = False
    debug_view_newton: bool = False  # False=Native, True=Newton

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
            add_ground=False,  # Native MJCF doesn't have ground plane
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

        # Compare mjw_model structures
        compare_mjw_models(newton_solver.mjw_model, native_mjw_model, skip_fields=self.model_skip_fields)

        # Get number of actuators from native model (for control generation)
        num_actuators = native_mjw_data.ctrl.shape[1] if native_mjw_data.ctrl.shape[1] > 0 else 0

        # Setup viewer if in debug mode
        viewer = None
        if self.debug_visual:
            import mujoco.viewer  # noqa: PLC0415

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

        # Helper: step both simulations
        def step_both(step_num: int, newton_graph: Any = None, native_graph: Any = None):
            t = step_num * self.dt
            ctrl = self.control_strategy.get_control(t, step_num, self.num_worlds, num_actuators)  # type: ignore[union-attr]
            newton_solver.mjw_data.ctrl.assign(ctrl)
            native_mjw_data.ctrl.assign(ctrl)

            if newton_graph and native_graph:
                wp.capture_launch(native_graph)
                wp.capture_launch(newton_graph)
            else:
                _mujoco_warp.step(native_mjw_model, native_mjw_data)  # type: ignore[union-attr]
                newton_solver.step(newton_state, newton_state, newton_control, None, self.dt)

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

            # Step 0: capture CUDA graphs if available
            if step == 0 and use_cuda_graph:
                ctrl = self.control_strategy.get_control(0, 0, self.num_worlds, num_actuators)
                newton_solver.mjw_data.ctrl.assign(ctrl)
                native_mjw_data.ctrl.assign(ctrl)

                with wp.ScopedCapture() as capture:
                    newton_solver.step(newton_state, newton_state, newton_control, None, self.dt)
                newton_graph = capture.graph

                with wp.ScopedCapture() as capture:
                    _mujoco_warp.step(native_mjw_model, native_mjw_data)
                native_graph = capture.graph
            else:
                step_both(step, newton_graph, native_graph)

            # Viewer sync
            if viewer:
                sync_to_viewer()
                viewer.sync()
                time.sleep(self.dt)

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
# Robot Test Classes
# =============================================================================
# Each robot from the menagerie gets its own test class.
# Initially all are skipped; enable as support is verified.
# Total: 61 robots (excluding test/ folder and realsense_d435i sensor)


# -----------------------------------------------------------------------------
# Arms (14 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_AgilexPiper(TestMenagerieBase):
    """AgileX PIPER bimanual arm."""

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


class TestMenagerie_Dynamixel2r(TestMenagerieBase):
    """Dynamixel 2R simple arm."""

    robot_folder = "dynamixel_2r"
    robot_xml = "2r.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_FrankaEmikaPanda(TestMenagerieBase):
    """Franka Emika Panda arm."""

    robot_folder = "franka_emika_panda"
    robot_xml = "panda.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_FrankaFr3(TestMenagerieBase):
    """Franka FR3 arm."""

    robot_folder = "franka_fr3"
    robot_xml = "fr3.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_FrankaFr3V2(TestMenagerieBase):
    """Franka FR3 v2 arm."""

    robot_folder = "franka_fr3_v2"
    robot_xml = "fr3.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_KinovaGen3(TestMenagerieBase):
    """Kinova Gen3 arm."""

    robot_folder = "kinova_gen3"
    robot_xml = "gen3.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_KukaIiwa14(TestMenagerieBase):
    """KUKA iiwa 14 arm."""

    robot_folder = "kuka_iiwa_14"
    robot_xml = "iiwa14.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_LowCostRobotArm(TestMenagerieBase):
    """Low-cost robot arm."""

    robot_folder = "low_cost_robot_arm"
    robot_xml = "low_cost_robot_arm.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_RethinkSawyer(TestMenagerieBase):
    """Rethink Robotics Sawyer arm."""

    robot_folder = "rethink_robotics_sawyer"
    robot_xml = "sawyer.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenVx300s(TestMenagerieBase):
    """Trossen Robotics ViperX 300 S arm."""

    robot_folder = "trossen_vx300s"
    robot_xml = "vx300s.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenWx250s(TestMenagerieBase):
    """Trossen Robotics WidowX 250 S arm."""

    robot_folder = "trossen_wx250s"
    robot_xml = "wx250s.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_TrossenWxai(TestMenagerieBase):
    """Trossen Robotics WidowX AI arm."""

    robot_folder = "trossen_wxai"
    robot_xml = "wxai.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_TrsSoArm100(TestMenagerieBase):
    """TRS SO-ARM100 arm."""

    robot_folder = "trs_so_arm100"
    robot_xml = "so_arm100.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_UfactoryLite6(TestMenagerieBase):
    """UFACTORY Lite 6 arm."""

    robot_folder = "ufactory_lite6"
    robot_xml = "lite6.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_UfactoryXarm7(TestMenagerieBase):
    """UFACTORY xArm 7 arm."""

    robot_folder = "ufactory_xarm7"
    robot_xml = "xarm7.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_UniversalRobotsUr5e(TestMenagerieBase):
    """Universal Robots UR5e arm."""

    robot_folder = "universal_robots_ur5e"
    robot_xml = "ur5e.xml"
    floating = False
    control_strategy = ZeroControlStrategy()
    num_worlds = 1  # For debugging
    debug_visual = True  # Enable viewer
    debug_view_newton = False  # False=Native, True=Newton


class TestMenagerie_UniversalRobotsUr10e(TestMenagerieBase):
    """Universal Robots UR10e arm."""

    robot_folder = "universal_robots_ur10e"
    robot_xml = "ur10e.xml"
    floating = False
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Grippers / Hands (9 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_LeapHand(TestMenagerieBase):
    """LEAP Hand."""

    robot_folder = "leap_hand"
    robot_xml = "left_hand.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_Robotiq2f85(TestMenagerieBase):
    """Robotiq 2F-85 gripper."""

    robot_folder = "robotiq_2f85"
    robot_xml = "2f85.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_Robotiq2f85V4(TestMenagerieBase):
    """Robotiq 2F-85 gripper v4."""

    robot_folder = "robotiq_2f85_v4"
    robot_xml = "2f85.xml"
    floating = False
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_ShadowDexee(TestMenagerieBase):
    """Shadow DEX-EE hand."""

    robot_folder = "shadow_dexee"
    robot_xml = "shadow_dexee.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_ShadowHand(TestMenagerieBase):
    """Shadow Hand."""

    robot_folder = "shadow_hand"
    robot_xml = "left_hand.xml"
    floating = False
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_TetheriaAeroHandOpen(TestMenagerieBase):
    """Tetheria Aero Hand (open)."""

    robot_folder = "tetheria_aero_hand_open"
    robot_xml = "aero_hand_open.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_UmiGripper(TestMenagerieBase):
    """UMI Gripper."""

    robot_folder = "umi_gripper"
    robot_xml = "umi_gripper.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_WonikAllegro(TestMenagerieBase):
    """Wonik Allegro Hand."""

    robot_folder = "wonik_allegro"
    robot_xml = "left_hand.xml"
    floating = False
    skip_reason = "HIGH PRIORITY - Not yet implemented"


class TestMenagerie_IitSoftfoot(TestMenagerieBase):
    """IIT Softfoot biomechanical gripper."""

    robot_folder = "iit_softfoot"
    robot_xml = "softfoot.xml"
    floating = False
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Bimanual Systems (2 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_Aloha(TestMenagerieBase):
    """ALOHA bimanual system."""

    robot_folder = "aloha"
    robot_xml = "aloha.xml"
    floating = False
    skip_reason = "Not yet implemented"


class TestMenagerie_GoogleRobot(TestMenagerieBase):
    """Google Robot (bimanual)."""

    robot_folder = "google_robot"
    robot_xml = "robot.xml"
    floating = False
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Mobile Manipulators (5 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_HelloRobotStretch(TestMenagerieBase):
    """Hello Robot Stretch."""

    robot_folder = "hello_robot_stretch"
    robot_xml = "stretch.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_HelloRobotStretch3(TestMenagerieBase):
    """Hello Robot Stretch 3."""

    robot_folder = "hello_robot_stretch_3"
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


class TestMenagerie_StanfordTidybot(TestMenagerieBase):
    """Stanford Tidybot mobile manipulator."""

    robot_folder = "stanford_tidybot"
    robot_xml = "tidybot.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Humanoids (10 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_ApptronikApollo(TestMenagerieBase):
    """Apptronik Apollo humanoid."""

    robot_folder = "apptronik_apollo"
    robot_xml = "apollo.xml"
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


class TestMenagerie_PalTalos(TestMenagerieBase):
    """PAL Robotics TALOS humanoid."""

    robot_folder = "pal_talos"
    robot_xml = "talos.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_PndboticsAdamLite(TestMenagerieBase):
    """PNDbotics Adam Lite humanoid."""

    robot_folder = "pndbotics_adam_lite"
    robot_xml = "adam_lite.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_RobotisOp3(TestMenagerieBase):
    """Robotis OP3 humanoid."""

    robot_folder = "robotis_op3"
    robot_xml = "op3.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_ToddlerBot2xc(TestMenagerieBase):
    """ToddlerBot 2XC humanoid."""

    robot_folder = "toddlerbot_2xc"
    robot_xml = "toddlerbot_2xc.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_ToddlerBot2xm(TestMenagerieBase):
    """ToddlerBot 2XM humanoid."""

    robot_folder = "toddlerbot_2xm"
    robot_xml = "toddlerbot_2xm.xml"
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


# -----------------------------------------------------------------------------
# Bipeds (1 robot)
# -----------------------------------------------------------------------------


class TestMenagerie_AgilityCassie(TestMenagerieBase):
    """Agility Robotics Cassie biped."""

    robot_folder = "agility_cassie"
    robot_xml = "cassie.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Quadrupeds (8 robots)
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


# -----------------------------------------------------------------------------
# Arms with Gripper (Unitree Z1)
# -----------------------------------------------------------------------------


class TestMenagerie_UnitreeZ1(TestMenagerieBase):
    """Unitree Z1 arm."""

    robot_folder = "unitree_z1"
    robot_xml = "z1.xml"
    floating = False
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Drones (2 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_BitcrazeCrazyflie2(TestMenagerieBase):
    """Bitcraze Crazyflie 2 quadrotor."""

    robot_folder = "bitcraze_crazyflie_2"
    robot_xml = "cf2.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_SkydioX2(TestMenagerieBase):
    """Skydio X2 drone."""

    robot_folder = "skydio_x2"
    robot_xml = "x2.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Mobile Bases (2 robots)
# -----------------------------------------------------------------------------


class TestMenagerie_RobotSoccerKit(TestMenagerieBase):
    """Robot Soccer Kit omniwheel base."""

    robot_folder = "robot_soccer_kit"
    robot_xml = "robot_soccer_kit.xml"
    floating = True
    skip_reason = "Not yet implemented"


class TestMenagerie_RobotstudioSo101(TestMenagerieBase):
    """RobotStudio SO-101."""

    robot_folder = "robotstudio_so101"
    robot_xml = "so101.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Biomechanical (1 robot)
# -----------------------------------------------------------------------------


class TestMenagerie_Flybody(TestMenagerieBase):
    """Flybody fruit fly model."""

    robot_folder = "flybody"
    robot_xml = "fruitfly.xml"
    floating = True
    skip_reason = "Not yet implemented"


# -----------------------------------------------------------------------------
# Other (1 robot)
# -----------------------------------------------------------------------------


class TestMenagerie_I2rtYam(TestMenagerieBase):
    """i2rt YAM (Yet Another Manipulator)."""

    robot_folder = "i2rt_yam"
    robot_xml = "yam.xml"
    floating = False
    skip_reason = "Not yet implemented"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
