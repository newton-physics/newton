# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rendering benchmarks for the site-backed camera sensor.

``FastSensorCamera`` and ``FastSensorCameraPixel`` measure Isaac Lab's Franka
cabinet scene with tiled and pixel-priority rendering in CI. The other scene
benchmarks cover varying visual complexity and are intended for hill-climbing
renderer performance:

- ``franka_cabinet``: Isaac Lab's Franka cabinet scene; the arm pulls the top drawer open and pushes it shut.
- ``quadruped``: an ANYmal D quadruped trotting in place while turning.
- ``shapes_256``: 256 primitive shapes raining onto the ground and tumbling as they fall.
- ``cloth``: a 32x32-vertex cloth (1,922 triangles) dropping onto a sphere and draping over it.
- ``particles``: 1,000 particles poured layer by layer between two walls.
- ``gaussians``: a Gaussian splat object (20,000 splats) picked up, carried and set down beside another.

Like MuJoCo Warp's replayed benchmark trajectories, every scene moves: one world's motion is
recorded at setup, and each frame replays it into every world at a different phase, then refits
the shape and particle BVHs and renders. Timings therefore include the per-frame refit and
deformable sync that a moving scene pays, but no simulation.

Each scene is described once as a :class:`ScenePreset` in :data:`SCENES` and is
shared between the ASV benchmark classes and the preview generator, so
previews show exactly what is benchmarked.

Run directly to benchmark, or to write animated GIF previews of each scene (a
single-world view and a 4x4 grid of 16 worlds)::

    uv run asv/benchmarks/simulation/bench_sensor_camera.py
    uv run asv/benchmarks/simulation/bench_sensor_camera.py --preview
"""

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import math
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

import newton
import newton.examples
import newton.utils
from newton import ShapeFlags
from newton.sensors import SensorCamera

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_SEKTION_CABINET_FOLDER = "assets/urdf/sektion_cabinet_model"
ISAACGYM_ENVS_COMMIT = "aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3"

# Renderer configuration applied to every scene benchmark. These are benchmark
# settings, not library defaults: SAH builds a higher-quality shape BVH
# (slower one-time build, faster ray traversal), and rendering uses the tiled
# traversal order with 8x8 pixel tiles.
BVH_CONSTRUCTOR = "sah"
KERNEL_BLOCK_DIM = 64
RENDER_ORDER = SensorCamera.RenderOrder.TILED
RENDER_TILE_WIDTH = 8
RENDER_TILE_HEIGHT = 8

# Arm "ready" pose facing the cabinet drawer, from Isaac Lab's
# FrankaCabinetDirectEnvCfg (panda_joint* values mapped to the FR3 URDF names).
_FRANKA_CABINET_ARM_POSE = {
    "fr3_joint1": 1.157,
    "fr3_joint2": -1.066,
    "fr3_joint3": -0.155,
    "fr3_joint4": -2.239,
    "fr3_joint5": -1.841,
    "fr3_joint6": 1.003,
    "fr3_joint7": 0.469,
    "fr3_finger_joint1": 0.035,
    "fr3_finger_joint2": 0.035,
}

# Nominal standing configuration, from Isaac Lab's ANYmal locomotion configs.
_ANYMAL_STANDING_POSE = {
    "LF_HAA": 0.0,
    "LF_HFE": 0.4,
    "LF_KFE": -0.8,
    "RF_HAA": 0.0,
    "RF_HFE": 0.4,
    "RF_KFE": -0.8,
    "LH_HAA": 0.0,
    "LH_HFE": -0.4,
    "LH_KFE": 0.8,
    "RH_HAA": 0.0,
    "RH_HFE": -0.4,
    "RH_KFE": 0.8,
}

_MANY_SHAPES_COUNT = 256
_MANY_SHAPES_COLORS = (
    (0.27, 0.47, 0.67),
    (0.40, 0.80, 0.93),
    (0.13, 0.53, 0.20),
    (0.80, 0.73, 0.27),
    (0.93, 0.40, 0.47),
)


def _set_joint_positions(builder: newton.ModelBuilder, joint_positions: dict[str, float]) -> None:
    """Set initial positions of single-DOF joints by joint name.

    Importers produce hierarchical labels like ``"fr3/fr3_joint1"``, so names
    are matched against the last path component of each joint label.
    """
    remaining = dict(joint_positions)
    for joint_index, label in enumerate(builder.joint_label):
        value = remaining.pop(label.rsplit("/", 1)[-1], None)
        if value is not None:
            builder.joint_q[builder.joint_q_start[joint_index]] = value
    if remaining:
        raise ValueError(f"Joints not found in builder: {sorted(remaining)}")


def _disable_collision_handling(builder: newton.ModelBuilder) -> None:
    """Strip collision flags so finalization skips collision-pair generation.

    Rendering does not consume collision state, and pair enumeration dominates
    model build time at benchmark world counts.
    """
    collide = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    builder.shape_flags = [int(flags) & ~collide for flags in builder.shape_flags]
    builder.shape_collision_filter_pairs = []


def _build_quadruped() -> newton.ModelBuilder:
    """An ANYmal D quadruped standing on the ground."""
    builder = newton.ModelBuilder()
    asset_path = newton.utils.download_asset("anybotics_anymal_d")
    builder.add_usd(
        str(asset_path / "usd" / "anymal_d.usda"),
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    builder.joint_q[:3] = [0.0, 0.0, 0.62]
    _set_joint_positions(builder, _ANYMAL_STANDING_POSE)
    return builder


def _build_franka_cabinet() -> newton.ModelBuilder:
    """Isaac Lab's Franka cabinet (open-drawer) scene.

    Layout follows FrankaCabinetDirectEnvCfg: the Sektion cabinet sits at the
    origin raised 0.4 m, with the arm 1 m away rotated to face the drawers.
    The cabinet URDF comes from IsaacGymEnvs (Isaac Lab itself references a
    Nucleus-hosted USD of the same asset).
    """
    cabinet_folder = newton.examples.download_external_git_folder(
        ISAACGYM_ENVS_REPO_URL,
        ISAACGYM_SEKTION_CABINET_FOLDER,
        ref=ISAACGYM_ENVS_COMMIT,
    )
    builder = newton.ModelBuilder()
    builder.add_urdf(
        cabinet_folder / "urdf" / "sektion_cabinet_2.urdf",
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()),
        floating=False,
        joint_damping_mode="passive",
    )
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi)),
        floating=False,
        joint_damping_mode="passive",
    )
    _set_joint_positions(builder, _FRANKA_CABINET_ARM_POSE)
    return builder


def _build_shapes_256() -> newton.ModelBuilder:
    """A 16x16 grid of primitive shapes (spheres, boxes, capsules, cylinders) resting on the ground."""
    builder = newton.ModelBuilder()
    rng = random.Random(1234)
    grid = math.isqrt(_MANY_SHAPES_COUNT)
    spacing = 0.8
    for index in range(_MANY_SHAPES_COUNT):
        row, col = divmod(index, grid)
        position = wp.vec3(
            (col - (grid - 1) / 2.0) * spacing,
            (row - (grid - 1) / 2.0) * spacing,
            0.25,
        )
        orientation = wp.quat_rpy(
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-math.pi, math.pi),
            rng.uniform(-math.pi, math.pi),
        )
        body = builder.add_body(xform=wp.transform(position, orientation))
        color = rng.choice(_MANY_SHAPES_COLORS)
        kind = index % 4
        if kind == 0:
            builder.add_shape_sphere(body, radius=0.18, color=color)
        elif kind == 1:
            builder.add_shape_box(body, hx=0.16, hy=0.16, hz=0.16, color=color)
        elif kind == 2:
            builder.add_shape_capsule(body, radius=0.1, half_height=0.15, color=color)
        else:
            builder.add_shape_cylinder(body, radius=0.14, half_height=0.15, color=color)
    return builder


_CLOTH_VERTICES = 32
_PARTICLE_CLOUD_SIDE = 10
_GAUSSIAN_SPLATS = 20000


def _build_cloth() -> newton.ModelBuilder:
    """A 32x32-vertex cloth draped over a sphere."""
    builder = newton.ModelBuilder()
    builder.add_shape_sphere(
        -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), radius=0.45, color=(0.27, 0.47, 0.67)
    )
    cell = 1.6 / (_CLOTH_VERTICES - 1)
    builder.add_cloth_grid(
        pos=wp.vec3(-0.8, -0.8, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=_CLOTH_VERTICES - 1,
        dim_y=_CLOTH_VERTICES - 1,
        cell_x=cell,
        cell_y=cell,
        mass=0.01,
        color=(0.93, 0.40, 0.47),
    )
    # Rest on the draped profile; the motion drops the cloth onto it.
    for i, q in enumerate(builder.particle_q):
        x, y = float(q[0]), float(q[1])
        builder.particle_q[i] = (x, y, 0.55 + 0.45 * math.exp(-2.5 * (x * x + y * y)))
    return builder


def _build_particles() -> newton.ModelBuilder:
    """A 10x10x10 pile of particles between two walls."""
    builder = newton.ModelBuilder()
    for y in (0.62, -0.62):
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, y, 0.3), wp.quat_identity()),
            hx=0.6,
            hy=0.02,
            hz=0.3,
            color=(0.6, 0.6, 0.65),
        )
    rng = np.random.default_rng(1234)
    spacing = 1.0 / _PARTICLE_CLOUD_SIDE
    for iz, iy, ix in np.ndindex((_PARTICLE_CLOUD_SIDE,) * 3):
        jitter = rng.uniform(-0.2, 0.2, 3) * spacing
        position = (np.array([ix - 4.5, iy - 4.5, 0.0]) * spacing + [0.0, 0.0, 0.1 + iz * spacing] + jitter).tolist()
        builder.add_particle(wp.vec3(*position), wp.vec3(0.0), 0.01, radius=0.04)
    return builder


def _splat_shell(seed: int, radii: tuple[float, float, float], color: tuple[float, float, float]) -> newton.Gaussian:
    """Splats scattered over an ellipsoid shell, like a scanned object's surface."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(_GAUSSIAN_SPLATS, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions * np.asarray(radii) * rng.uniform(0.97, 1.03, (_GAUSSIAN_SPLATS, 1))
    rotations = rng.normal(size=(_GAUSSIAN_SPLATS, 4))
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    # Degree-0 coefficients giving ``color`` with some variation.
    dc = (np.asarray(color) + rng.uniform(-0.1, 0.1, (_GAUSSIAN_SPLATS, 3)) - 0.5) / 0.28209479177387814
    return newton.Gaussian(
        positions=positions.astype(np.float32),
        rotations=rotations.astype(np.float32),
        scales=rng.uniform(0.008, 0.025, (_GAUSSIAN_SPLATS, 3)).astype(np.float32),
        opacities=rng.uniform(0.3, 0.9, _GAUSSIAN_SPLATS).astype(np.float32),
        sh_coeffs=dc.astype(np.float32),
    )


def _build_gaussians() -> newton.ModelBuilder:
    """A Gaussian splat object on a free body beside a resting one and a box."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()))
    builder.add_shape_gaussian(body, gaussian=_splat_shell(1, (0.35, 0.35, 0.35), (0.85, 0.55, 0.3)))
    builder.add_shape_gaussian(
        -1,
        xform=wp.transform(wp.vec3(0.25, -0.35, 0.3), wp.quat_identity()),
        gaussian=_splat_shell(2, (0.3, 0.2, 0.25), (0.3, 0.6, 0.85)),
    )
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(-0.6, 0.5, 0.25), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.25)
    return builder


# A motion maps a one-world model and a loop phase in [0, 1) to that world's joint and
# particle coordinates (``None`` where the world has none), which become body and particle poses.
Motion = Callable[[newton.Model, float], tuple[np.ndarray | None, np.ndarray | None]]


def _keyframes(t: float, times: list[float], values: list[list[float]]) -> np.ndarray:
    """Interpolate each column of ``values`` linearly between keyframes at ``times``."""
    return np.array([np.interp(t, times, column) for column in np.asarray(values, dtype=np.float64).T])


def _joint_coordinates(model: newton.Model) -> dict[str, int]:
    """First coordinate index of each joint, by the last path component of its label."""
    starts = model.joint_q_start.numpy()
    return {label.rsplit("/", 1)[-1]: int(starts[j]) for j, label in enumerate(model.joint_label)}


def _quat_about_z(angle: float) -> list[float]:
    return [0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)]


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton products of ``(..., 4)`` xyzw quaternions."""
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return np.concatenate([aw * bv + bw * av + np.cross(av, bv), aw * bw - np.sum(av * bv, -1, keepdims=True)], -1)


def _fall(t: np.ndarray, start: np.ndarray, duration: float) -> np.ndarray:
    """Remaining fall height fraction in [0, 1]: 1 before ``start``, easing to 0 like a dropped object."""
    return 1.0 - np.clip((t - start) / duration, 0.0, 1.0) ** 2


# Arm joint values [rad] from inverse kinematics on the hand's TCP: hovering in front of the
# top drawer handle, then holding the handle with the drawer opened by 0, 0.1, 0.2 and 0.3 m.
_FRANKA_PREGRASP = [1.677, -1.416, -0.747, -2.199, -2.141, 2.066, 0.187]
_FRANKA_PULL = [
    [1.939, -1.331, -0.946, -1.829, -2.271, 2.207, 0.132],
    [1.728, -1.390, -0.798, -2.123, -2.177, 2.100, 0.182],
    [1.565, -1.475, -0.620, -2.352, -2.033, 2.001, 0.192],
    [1.451, -1.577, -0.405, -2.517, -1.834, 1.902, 0.174],
]


def _franka_cabinet_motion(model: newton.Model, t: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Reach the top drawer, grasp it, pull it open, push it shut, release and return."""
    ready = [_FRANKA_CABINET_ARM_POSE[f"fr3_joint{i}"] for i in range(1, 8)]
    opened, closed = 0.04, 0.012
    # Columns: drawer opening [m], seven arm joints [rad], finger opening [m].
    frames = [
        (0.00, 0.0, ready, 0.035),
        (0.14, 0.0, _FRANKA_PREGRASP, opened),
        (0.22, 0.0, _FRANKA_PULL[0], opened),
        (0.28, 0.0, _FRANKA_PULL[0], closed),
        *((0.28 + 0.08 * k, 0.1 * k, _FRANKA_PULL[k], closed) for k in (1, 2, 3)),
        *((0.60 + 0.08 * k, 0.1 * (3 - k), _FRANKA_PULL[3 - k], closed) for k in (1, 2, 3)),
        (0.90, 0.0, _FRANKA_PULL[0], opened),
        (0.95, 0.0, _FRANKA_PREGRASP, opened),
        (1.00, 0.0, ready, 0.035),
    ]
    drawer, *arm, finger = _keyframes(t, [f[0] for f in frames], [[f[1], *f[2], f[3]] for f in frames])
    coords = _joint_coordinates(model)
    q = model.joint_q.numpy()
    q[coords["drawer_top_joint"]] = drawer
    q[[coords[f"fr3_joint{i}"] for i in range(1, 8)]] = arm
    q[[coords["fr3_finger_joint1"], coords["fr3_finger_joint2"]]] = finger
    return q, None


def _quadruped_motion(model: newton.Model, t: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Trot in place, diagonal leg pairs swinging in turn, while turning once per loop."""
    coords = _joint_coordinates(model)
    q = model.joint_q.numpy()
    phase = 2.0 * math.pi * 6.0 * t
    q[2] += 0.015 * math.cos(2.0 * phase)
    q[3:7] = _quat_about_z(2.0 * math.pi * t)
    for leg, pair in (("LF", 0), ("RH", 0), ("RF", 1), ("LH", 1)):
        swing = max(0.0, math.sin(phase + math.pi * pair))
        side = 1.0 if leg[1] == "F" else -1.0
        # Flex the hip and knee to lift the foot; hind legs bend the opposite way.
        q[coords[f"{leg}_HFE"]] = _ANYMAL_STANDING_POSE[f"{leg}_HFE"] + side * 0.35 * swing
        q[coords[f"{leg}_KFE"]] = _ANYMAL_STANDING_POSE[f"{leg}_KFE"] - side * 0.7 * swing
    return q, None


def _shapes_256_motion(model: newton.Model, t: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Rain every shape onto the ground from 3 m, tumbling about its own axis until it lands."""
    rng = np.random.default_rng(7)
    start = rng.uniform(0.0, 0.6, _MANY_SHAPES_COUNT)
    axes = rng.normal(size=(_MANY_SHAPES_COUNT, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    q = model.joint_q.numpy().reshape(_MANY_SHAPES_COUNT, 7)
    fall = _fall(t, start, 0.25)
    q[:, 2] += 3.0 * fall
    angle = 6.0 * fall[:, None]
    q[:, 3:] = _quat_multiply(np.concatenate([axes * np.sin(0.5 * angle), np.cos(0.5 * angle)], 1), q[:, 3:])
    return q.reshape(-1), None


def _cloth_motion(model: newton.Model, t: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Drop the cloth flat onto the sphere, let it drape over it, then lift it off again."""
    draped = model.particle_q.numpy()
    lift = np.clip(1.6 * abs(2.0 * t - 1.0) - 0.3, 0.0, 1.0)
    draped[:, 2] = np.maximum(draped[:, 2], 0.55 + 0.6 * lift)
    return None, draped


def _particles_motion(model: newton.Model, t: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pour the pile in from above layer by layer, bottom layer first."""
    q = model.particle_q.numpy()
    layer = np.arange(len(q)) // _PARTICLE_CLOUD_SIDE**2
    q[:, 2] += 1.5 * _fall(t, 0.07 * layer, 0.2)
    return None, q


def _gaussians_motion(model: newton.Model, t: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pick the splat object up, carry it toward the camera, set it down and bring it back."""
    # Columns: x, y, z [m] and yaw [rad] of the carried object.
    x, y, z, yaw = _keyframes(
        t,
        [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0],
        [
            [0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.8, 0.0],
            [0.35, 0.45, 0.8, 1.6],
            [0.35, 0.45, 0.4, 1.6],
            [0.35, 0.45, 0.4, 1.6],
            [0.35, 0.45, 0.8, 1.6],
            [0.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.4, 0.0],
        ],
    )
    q = model.joint_q.numpy()
    q[:7] = [x, y, z, *_quat_about_z(yaw)]
    return q, None


@dataclass(frozen=True)
class ScenePreset:
    """A benchmark scene: one world's worth of content plus a camera pose."""

    build: Callable[[], newton.ModelBuilder]
    """Builds a ModelBuilder holding a single world of scene content."""

    motion: Motion
    """The scene's looping task motion."""

    camera_eye: tuple[float, float, float]
    """Camera position [m]."""

    camera_target: tuple[float, float, float]
    """Point the camera looks at [m]."""

    light_direction: tuple[float, float, float] | None = None
    """Directional-light travel direction; ``None`` uses the sensor default."""


SCENES: dict[str, ScenePreset] = {
    "quadruped": ScenePreset(
        _build_quadruped, _quadruped_motion, camera_eye=(2.2, 2.2, 1.1), camera_target=(0.0, 0.0, 0.45)
    ),
    # Light comes from the camera-facing upper-front octant so it lands on the
    # cabinet drawers (which face +x) and top, both visible to the camera.
    "franka_cabinet": ScenePreset(
        _build_franka_cabinet,
        _franka_cabinet_motion,
        camera_eye=(2.4, 1.8, 1.3),
        camera_target=(0.4, 0.0, 0.5),
        light_direction=(-1.0, -0.4, -0.45),
    ),
    "shapes_256": ScenePreset(
        _build_shapes_256, _shapes_256_motion, camera_eye=(9.0, 9.0, 6.0), camera_target=(0.0, 0.0, 0.0)
    ),
    "cloth": ScenePreset(_build_cloth, _cloth_motion, camera_eye=(2.0, 1.6, 1.6), camera_target=(0.0, 0.0, 0.5)),
    "particles": ScenePreset(
        _build_particles, _particles_motion, camera_eye=(1.8, 1.4, 1.3), camera_target=(0.0, 0.0, 0.4)
    ),
    "gaussians": ScenePreset(
        _build_gaussians, _gaussians_motion, camera_eye=(2.0, 1.6, 1.2), camera_target=(0.0, 0.0, 0.35)
    ),
}


def _look_at_transform(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> wp.transformf:
    """Camera-to-world transform at *eye* looking toward *target* (camera space: -Z forward, +Y up)."""
    eye_v = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye_v
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    rotation = np.stack([right, camera_up, -forward], axis=1)
    return wp.transformf(wp.vec3f(*eye), wp.quat_from_matrix(wp.mat33f(rotation.flatten())))


MOTION_FRAME_COUNT = 120
"""Frames in one loop of a scene's motion."""

WORLD_PHASE_STRIDE = 37
"""Frames between the motion phases of neighboring worlds, so worlds show different poses."""


@wp.kernel(enable_backward=False)
def replay_motion(frames: wp.array2d[Any], frame: wp.int32, out: wp.array[Any]):
    """Write one world's recorded ``frames`` into every world of ``out``, each at its own phase."""
    i = wp.tid()
    world, index = i // frames.shape[1], i % frames.shape[1]
    out[i] = frames[(frame + world * WORLD_PHASE_STRIDE) % frames.shape[0], index]


def record_motion(
    world: newton.ModelBuilder, motion: Motion, device: wp.DeviceLike
) -> tuple[wp.array | None, wp.array | None]:
    """Record one world's body and particle poses over a loop of ``motion``; ``None`` where it has none."""
    model = world.finalize(device=device)
    state = model.state()
    body_q, particle_q = [], []
    for frame in range(MOTION_FRAME_COUNT):
        joint_q, particles = motion(model, frame / MOTION_FRAME_COUNT)
        if model.body_count:
            newton.eval_fk(model, wp.array(joint_q, dtype=wp.float32, device=device), model.joint_qd, state)
            body_q.append(state.body_q.numpy())
        if model.particle_count:
            particle_q.append(particles)
    return tuple(
        wp.array(np.stack(frames), dtype=dtype, device=device) if frames else None
        for frames, dtype in ((body_q, wp.transformf), (particle_q, wp.vec3f))
    )


class SceneRig:
    """A moving scene replicated across worlds with a camera sensor ready to render."""

    def __init__(
        self,
        preset: ScenePreset,
        world_count: int,
        resolution: int,
        render_order: SensorCamera.RenderOrder,
        camera_fov_deg: float = 45.0,
        *,
        backend: Literal["warp", "optix"] = "warp",
    ):
        world = preset.build()
        _disable_collision_handling(world)

        camera = _look_at_transform(preset.camera_eye, preset.camera_target)

        scene = newton.ModelBuilder()
        scene.default_bvh_cfg.mesh_constructor = "cubql"
        scene.replicate(world, world_count)
        scene.add_ground_plane(color=(0.6, 0.6, 0.6))

        self.body_frames, self.particle_frames = record_motion(world, preset.motion, wp.get_device())
        self.model = scene.finalize()
        self.state = self.model.state()
        self.backend = backend

        light_direction = wp.vec3f(*preset.light_direction) if preset.light_direction is not None else None
        self.sensor = SensorCamera(self.model, backend=backend)
        self.sensor.create_default_light(enable_shadows=True, direction=light_direction)
        self.sensor.assign_checkerboard_material(shape_indices=np.arange(self.model.shape_count, dtype=np.int32))
        self.sensor.default_render_config.render_order = render_order
        self.sensor.default_render_config.tile_width = RENDER_TILE_WIDTH
        self.sensor.default_render_config.tile_height = RENDER_TILE_HEIGHT
        self.sensor.default_render_config.enable_shadows = True
        self.sensor.default_render_config.enable_textures = True
        self.sensor.default_render_config.block_dim = KERNEL_BLOCK_DIM

        # The caller owns the rays and the per-view transforms passed to update().
        self.rays = SensorCamera.compute_camera_rays_pinhole(
            resolution, resolution, camera_fov=math.radians(camera_fov_deg), device=self.model.device
        )
        # World-fixed camera: point every world's view from the same look-at pose.
        camera_row = np.array([camera[i] for i in range(7)], dtype=np.float32)
        self.camera_transforms = wp.array(
            np.tile(camera_row, (world_count, 1)), dtype=wp.transformf, device=self.model.device
        )

        self.color_image = self.sensor.allocate_image_color(world_count, resolution, resolution)
        self.depth_image = self.sensor.allocate_image_depth(world_count, resolution, resolution)

        self.pose(0)
        self.model.bvh_build_shapes(self.state, bvh_constructor=BVH_CONSTRUCTOR)
        if backend == "optix":
            self.model.bvh_build_shapes(self.state, backend="optix")
        self.model.bvh_build_particles(self.state, bvh_constructor=BVH_CONSTRUCTOR)

    def pose(self, frame: int):
        """Replay motion ``frame`` into every world's bodies and particles."""
        for frames, out in ((self.body_frames, self.state.body_q), (self.particle_frames, self.state.particle_q)):
            if frames is not None:
                wp.launch(replay_motion, dim=len(out), inputs=[frames, frame], outputs=[out], device=self.model.device)

    def step(self, frame: int):
        """Move the scene to motion ``frame`` and refit the BVHs the camera traverses."""
        self.pose(frame)
        self.model.bvh_refit_shapes(self.state, backend=self.backend)
        if self.model.particle_count:
            self.model.bvh_refit_particles(self.state)

    def render(self, color: bool = True, depth: bool = True):
        """Render the current state, syncing the deformable surfaces it moved."""
        self.sensor.update(
            self.state,
            self.camera_transforms,
            self.rays,
            color_image=self.color_image if color else None,
            depth_image=self.depth_image if depth else None,
        )


def _pr_gate_skips_output(color: bool, depth: bool, selected_modes: tuple[tuple[bool, bool], ...]) -> bool:
    return bool(os.environ.get("NEWTON_ASV_PR_GATE")) and (color, depth) not in selected_modes


class SceneBenchmark(ABC):
    """Shared ASV harness; subclasses pick a scene from :data:`SCENES` and their params.

    Abstract until a subclass names its scene, so ASV does not collect it as a benchmark.
    """

    param_names = ["resolution", "world_count", "iterations"]
    render_order = RENDER_ORDER
    backend = "warp"
    pr_gate_output_modes: tuple[tuple[bool, bool], ...] | None = None

    @property
    @abstractmethod
    def scene(self) -> str:
        """Key of the benchmarked scene in :data:`SCENES`."""

    def setup(self, resolution: int, world_count: int, iterations: int):
        self.rig = SceneRig(SCENES[self.scene], world_count, resolution, self.render_order, backend=self.backend)
        output_modes = ((True, True), (True, False), (False, True))
        if os.environ.get("NEWTON_ASV_PR_GATE") and self.pr_gate_output_modes is not None:
            output_modes = self.pr_gate_output_modes
        # Compile and warm the update and each output combination selected by this ASV run.
        self.rig.step(0)
        for color, depth in output_modes:
            self.rig.render(color=color, depth=depth)
        wp.synchronize()

    def run(self, iterations: int, color: bool, depth: bool):
        """Step through the first ``iterations`` motion frames, rendering each."""
        for frame in range(iterations):
            self.rig.step(frame)
            self.rig.render(color=color, depth=depth)
        wp.synchronize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_render_color_depth(self, resolution: int, world_count: int, iterations: int):
        self.run(iterations, color=True, depth=True)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_render_color_only(self, resolution: int, world_count: int, iterations: int):
        self.run(iterations, color=True, depth=False)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_render_depth_only(self, resolution: int, world_count: int, iterations: int):
        self.run(iterations, color=False, depth=True)


class SensorCameraQuadruped(SceneBenchmark):
    scene = "quadruped"
    params = ([64], [4096], [50])


class FastSensorCamera(SceneBenchmark):
    scene = "franka_cabinet"
    if os.environ.get("NEWTON_ASV_PR_GATE"):
        repeat = 1
        rounds = 2
    pr_gate_output_modes = ((True, True), (False, True))
    params = ([64], [4096], [50])

    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or _pr_gate_skips_output(True, False, pr_gate_output_modes))
    def time_render_color_only(self, resolution: int, world_count: int, iterations: int):
        super().time_render_color_only(resolution, world_count, iterations)


class FastSensorCameraPixel(SceneBenchmark):
    scene = "franka_cabinet"
    if os.environ.get("NEWTON_ASV_PR_GATE"):
        repeat = 1
        rounds = 2
    render_order = SensorCamera.RenderOrder.PIXEL_PRIORITY
    pr_gate_output_modes = ((True, True),)
    params = ([64], [4096], [50])

    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or _pr_gate_skips_output(True, False, pr_gate_output_modes))
    def time_render_color_only(self, resolution: int, world_count: int, iterations: int):
        super().time_render_color_only(resolution, world_count, iterations)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0 or _pr_gate_skips_output(False, True, pr_gate_output_modes))
    def time_render_depth_only(self, resolution: int, world_count: int, iterations: int):
        super().time_render_depth_only(resolution, world_count, iterations)


class SensorCameraShapes256(SceneBenchmark):
    scene = "shapes_256"
    params = ([64], [4096], [50])


class SensorCameraCloth(SceneBenchmark):
    scene = "cloth"
    params = ([64], [4096], [50])


class SensorCameraParticles(SceneBenchmark):
    scene = "particles"
    params = ([64], [4096], [50])


class SensorCameraGaussians(SceneBenchmark):
    scene = "gaussians"
    params = ([64], [1024], [50])


PREVIEW_WORLD_COUNTS = (1, 16)


def write_preview_images(
    scene_names: list[str], output_dir: Path, image_size: int = 512, backend: Literal["warp", "optix"] = "warp"
) -> list[Path]:
    """Write an animated GIF of one motion loop of each scene for every entry in :data:`PREVIEW_WORLD_COUNTS`.

    Single-world previews render at ``image_size``; multi-world previews tile a
    square grid of per-world renders into the same overall image size.
    """
    from PIL import Image  # noqa: PLC0415  # only needed for previews (part of the examples extra)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name in scene_names:
        for world_count in PREVIEW_WORLD_COUNTS:
            views_per_row = math.isqrt(world_count)
            rig = SceneRig(SCENES[name], world_count, image_size // views_per_row, RENDER_ORDER, backend=backend)
            images = []
            for frame in range(0, MOTION_FRAME_COUNT, 2):
                rig.step(frame)
                rig.render(color=True, depth=False)
                rgba = SensorCamera.Utils.flatten_color_image_to_rgba(rig.color_image, views_per_row=views_per_row)
                # Drop alpha: background pixels have alpha 0 and would turn transparent.
                images.append(Image.fromarray(rgba.numpy()[..., :3]))
            path = output_dir / f"{name}_{world_count}_world{'s' if world_count > 1 else ''}.gif"
            images[0].save(path, save_all=True, append_images=images[1:], duration=66, loop=0)
            written.append(path)
    return written


def print_fps(name: str, duration: float, resolution: int, world_count: int, iterations: int):
    title = f"{name}"
    if iterations > 1:
        title += " average"

    average = f"{duration * 1000.0 / iterations:.2f} ms"

    fps = f"({(1.0 / (duration / iterations) * world_count):,.2f} fps)"
    print(f"{title} {'.' * (50 - len(title) - len(average))} {average} {fps if iterations > 1 else ''}")


def print_fps_results(results: dict[tuple[str, tuple[int, int, int]], float]):
    print()
    print("=== Benchmark Results (FPS) ===")
    for (method_name, params), avg in results.items():
        print_fps(method_name, avg, *params)
    print()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastSensorCamera": FastSensorCamera,
        "FastSensorCameraPixel": FastSensorCameraPixel,
        "SensorCameraQuadruped": SensorCameraQuadruped,
        "SensorCameraShapes256": SensorCameraShapes256,
        "SensorCameraCloth": SensorCameraCloth,
        "SensorCameraParticles": SensorCameraParticles,
        "SensorCameraGaussians": SensorCameraGaussians,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write GIF previews of each scene (single world and 4x4 world grid) instead of benchmarking.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("sensor_camera_previews"),
        help="Directory for preview images.",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=512,
        help="Preview image width and height [px].",
    )
    parser.add_argument(
        "--backend",
        choices=["warp", "optix"],
        default="warp",
        help="Camera shape-traversal backend.",
    )
    args = parser.parse_known_args()[0]

    if args.preview:
        for path in write_preview_images(
            sorted(SCENES), args.preview_dir, image_size=args.preview_size, backend=args.backend
        ):
            print(f"Wrote {path}")
    else:
        for key in args.bench or benchmark_list.keys():
            benchmark_list[key].backend = args.backend
            result = run_benchmark(benchmark_list[key])
            print_fps_results(result)
