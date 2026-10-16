# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Squeeze an open PET bottle from three angles and retain permanent dents.

GPU VBD solves a self-contacting 0.3 mm shell with PET inputs of 3 GPa
Young's modulus and 55 MPa yield stress (assumed nu=0.38, density=1380 kg/m^3).
Thickness-integrated membrane stiffness and dual-width bending coefficients
preserve those SI material values. Optional VBD plasticity retains yielded
bending curvature; it is an ideal bending-only law, not calibrated PET.

32,000 XPBD density-constrained water particles receive one-way contact from
the shell. Newton's ParticleSurface reconstructs the liquid and clips it to
the moving bottle and table. Water pressure does not feed back onto the shell.
The Franka follows checked IK waypoints with physical squeeze plates. Its
base and the bottle bottom are fixed; the label and neck finish are visual overlays.

Physics uses cm/kg/s for conditioning, while rendering uses meters. Water
experiences Earth gravity; PET self-weight, fracture and enclosed air are
omitted. See docs/guide/franka_bottle_squeeze.rst for equations and limitations.

Command: python -m newton.examples franka_bottle_squeeze
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik
import newton.utils
from newton.examples.multiphysics._bottle import BottleFluid, BottleSurface, bind_water_material, compute_normals
from newton.solvers import SolverBase, SolverVBD

_SCALE = 100.0
_THICKNESS = 0.0003
_YOUNG_MODULUS = 3.0e9
_YIELD_STRESS = 55.0e6
_POISSON_RATIO = 0.38
_PET_DENSITY = 1380.0
_TABLE_HEIGHT = 0.76
_BOTTLE_CENTER = np.array([0.55, 0.0, _TABLE_HEIGHT + 0.0008])
_BOTTLE_HEIGHT = 0.21
_CYCLE_DURATION = 8.0
_DEFAULT_SQUEEZE_HEIGHTS = (0.112, 0.130, 0.094)
_WATER_COLOR = (0.90, 0.97, 1.0)
_PLATE_HALF_EXTENTS = np.array([0.030, 0.0015, 0.024])
_PLATE_POSITION = np.array([0.0, 0.0015, 0.04525])


@wp.func
def _joint_target(time: float, coordinate: int, times: wp.array[float], poses: wp.array2d[float]):
    segment = int(0)
    for key in range(times.shape[0] - 1):
        if time >= times[key]:
            segment = key
    alpha = wp.clamp((time - times[segment]) / (times[segment + 1] - times[segment]), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return wp.lerp(poses[segment, coordinate], poses[segment + 1, coordinate], alpha)


@wp.kernel
def _robot_motion(
    clock: wp.array[float],
    dt: float,
    times: wp.array[float],
    poses: wp.array2d[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    i = wp.tid()
    time = clock[0] + dt
    q = _joint_target(time, i, times, poses)
    q_before = _joint_target(time - dt, i, times, poses)
    joint_q[i] = q
    joint_qd[i] = (q - q_before) / dt


@wp.kernel
def _advance_clock(clock: wp.array[float], dt: float):
    clock[0] += dt


@wp.kernel
def _scale_positions(source: wp.array[wp.vec3], output: wp.array[wp.vec3]):
    i = wp.tid()
    output[i] = 0.01 * source[i]


@wp.kernel
def _update_label(
    positions: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    vertices: wp.array[int],
    output: wp.array[wp.vec3],
    output_normals: wp.array[wp.vec3],
):
    i = wp.tid()
    # Keep the visual wrapper outside the PET surface in ray-traced views.
    normal = normals[vertices[i]]
    output[i] = positions[vertices[i]] + 0.0004 * normal
    output_normals[i] = normal


@wp.kernel
def _update_neck(
    positions: wp.array[wp.vec3],
    segments: int,
    anchors: wp.array[int],
    offsets: wp.array[wp.vec4],
    output: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    anchor = anchors[i]
    ring = 1 + ((anchor - 1) // segments) * segments
    s = anchor - ring
    tangent = wp.normalize(positions[ring + (s + 1) % segments] - positions[ring + (s + segments - 1) % segments])
    axis = wp.normalize(positions[anchor] - positions[anchor - segments])
    radial = wp.normalize(wp.cross(tangent, axis))
    axis = wp.cross(radial, tangent)
    offset = offsets[i]
    output[i] = positions[anchor] + offset[0] * radial + offset[1] * axis
    normals[i] = offset[2] * radial + offset[3] * axis


@wp.kernel
def _scale_transforms(source: wp.array[wp.transform], output: wp.array[wp.transform]):
    i = wp.tid()
    x = source[i]
    output[i] = wp.transform(0.01 * wp.transform_get_translation(x), wp.transform_get_rotation(x))


def _squeeze_keyframes(grips, retreats, finger_indices, minimum_opening):
    """Keep both fingers open during withdrawal, rotation and approach."""
    times, poses = [], []
    for cycle, (grip, retreat) in enumerate(zip(grips, retreats, strict=True)):
        closed = grip.copy()
        closed[list(finger_indices)] = minimum_opening
        for time, pose in (
            (0.0, grip),
            (0.6, grip),
            (1.7, closed),
            (2.6, closed),
            (3.4, grip),
            (3.6, grip),
            (4.4, retreat),
        ):
            times.append(cycle * _CYCLE_DURATION + time)
            poses.append(pose)
        if cycle + 1 < len(grips):
            times.append(cycle * _CYCLE_DURATION + 6.8)
            poses.append(retreats[cycle + 1])
    times.append(len(grips) * _CYCLE_DURATION)
    poses.append(retreats[-1])
    return np.asarray(times, dtype=np.float32), np.asarray(poses, dtype=np.float32)


def _bottle_radius(z):
    radius = np.interp(z, [0.0, 0.014, 0.16, 0.188, 0.21], [0.030, 0.032, 0.032, 0.012, 0.012])
    body = (z >= 0.022) & (z <= 0.152)
    return radius + body * 0.0004 * np.cos(2.0 * np.pi * (z - 0.024) / 0.026)


def _bottle_mesh(segments: int, rings: int):
    """Create a closed bottom, periodic body and genuinely open neck in SI."""
    theta = np.arange(segments) * (2.0 * np.pi / segments)
    zs = np.concatenate((np.zeros(4), np.linspace(0.003, _BOTTLE_HEIGHT, rings)))
    radii = np.concatenate((np.array([0.0075, 0.015, 0.0225, 0.030]), _bottle_radius(zs[4:])))
    points = [[0.0, 0.0, 0.0]]
    for z, radius in zip(zs, radii, strict=True):
        # A small manufactured imperfection selects a repeatable buckling mode.
        r = radius * (1.0 + 0.002 * np.cos(3.0 * theta) * np.sin(np.pi * z / _BOTTLE_HEIGHT))
        points.extend(np.column_stack((r * np.cos(theta), r * np.sin(theta), np.full(segments, z))))
    triangles = []
    for s in range(segments):
        triangles.append([0, 1 + (s + 1) % segments, 1 + s])
    for ring in range(len(zs) - 1):
        for s in range(segments):
            a = 1 + ring * segments + s
            b = 1 + ring * segments + (s + 1) % segments
            c = a + segments
            d = b + segments
            # The bottom annuli face down; the body faces outwards.
            if ring < 3 or (s + ring) % 2:
                triangles.extend(([a, b, c], [b, d, c]))
            else:
                triangles.extend(([a, b, d], [a, d, c]))
    return np.asarray(points, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def _bottle_neck(vertices, segments):
    """Bind an open rolled lip, closure beads and support flange to neck rings."""
    sides = 12
    anchors, offsets, triangles = [], [], []
    heights = vertices[1::segments, 2]
    section = np.arange(sides) * (2.0 * np.pi / sides)
    # Height, radial stand-off and cross-section radius [m]. These details
    # follow the shell without altering its collision mesh or open mouth.
    for height, stand_off, tube_radius in (
        (_BOTTLE_HEIGHT, 0.0003, 0.0007),
        (0.204, 0.0003, 0.0006),
        (0.198, 0.0003, 0.0006),
        (0.191, 0.0012, 0.0007),
    ):
        row = int(np.argmin(np.abs(heights - height)))
        start = len(anchors)
        for s in range(segments):
            anchors.extend([1 + row * segments + s] * sides)
            offsets.extend(
                np.column_stack(
                    (
                        stand_off + tube_radius * np.cos(section),
                        height - heights[row] + tube_radius * np.sin(section),
                        np.cos(section),
                        np.sin(section),
                    )
                )
            )
            for j in range(sides):
                a = start + s * sides + j
                b = start + ((s + 1) % segments) * sides + j
                c = start + s * sides + (j + 1) % sides
                d = start + ((s + 1) % segments) * sides + (j + 1) % sides
                triangles.extend(((a, b, c), (b, d, c)))
    return (
        np.asarray(anchors, dtype=np.int32),
        np.asarray(offsets, dtype=np.float32),
        np.asarray(triangles, dtype=np.int32),
    )


def _pet_bending_properties(builder: newton.ModelBuilder):
    """Convert SI flexural rigidity and yield curvature to cm-scale hinges."""
    points = np.asarray(builder.particle_q, dtype=np.float64)
    edges = np.asarray(builder.edge_indices, dtype=np.int32)
    properties = np.asarray(builder.edge_bending_properties, dtype=np.float64)
    yield_angles = np.full(len(edges), np.inf, dtype=np.float32)
    rigidity = _YOUNG_MODULUS * _THICKNESS**3 / (12.0 * (1.0 - _POISSON_RATIO**2)) * _SCALE**2
    yield_curvature = 2.0 * _YIELD_STRESS / (_YOUNG_MODULUS * _THICKNESS) / _SCALE
    for edge, (i, j, k, l) in enumerate(edges):
        if i < 0 or j < 0:
            properties[edge] = (0.0, 0.0)
            continue
        length = np.linalg.norm(points[l] - points[k])
        height0 = np.linalg.norm(np.cross(points[k] - points[i], points[l] - points[i])) / length
        height1 = np.linalg.norm(np.cross(points[l] - points[j], points[k] - points[j])) / length
        dual_width = (height0 + height1) / 3.0
        # VBD's hinge energy is 0.5 * edge_ke * edge_length * delta_angle^2.
        properties[edge] = (rigidity / dual_width, 1.0e-5)
        yield_angles[edge] = yield_curvature * dual_width
    builder.edge_bending_properties = properties.tolist()
    return yield_angles


def _water_samples(count: int, *, seed: int):
    """Emit exactly count stratified particles without filling the open mouth."""

    def sample(spacing):
        xy = np.arange(-0.032, 0.032, spacing)
        z = np.arange(0.45 * spacing, 0.193, spacing)
        points = np.stack(np.meshgrid(xy, xy, z, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = np.linalg.norm(points[:, :2], axis=1) < _bottle_radius(points[:, 2]) - 0.6 * spacing
        return points[inside]

    lower = 0.0005
    upper = 0.015
    for _ in range(32):
        middle = 0.5 * (lower + upper)
        if len(sample(middle)) >= count:
            lower = middle
        else:
            upper = middle
    points = sample(lower)
    rng = np.random.default_rng(seed)
    points = points[rng.permutation(len(points))[:count]]
    points += rng.uniform(-0.03, 0.03, points.shape) * lower
    return (points + _BOTTLE_CENTER) * _SCALE, lower * _SCALE


def _add_robot(builder, asset, *, scale):
    builder.add_urdf(
        str(asset / "urdf/fr3_franka_hand.urdf"),
        xform=wp.transform((0.0, 0.0, _TABLE_HEIGHT * scale), wp.quat_identity()),
        floating=False,
        scale=scale,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        joint_damping_mode="passive",
    )
    # Both finger frames point their local +Y outwards. Keep the contact faces
    # flush with the pads; local X is vertical in the horizontal grip pose.
    plate_cfg = newton.ModelBuilder.ShapeConfig(density=2700.0 / scale**3)
    for finger in ("fr3/fr3_leftfinger", "fr3/fr3_rightfinger"):
        builder.add_shape_box(
            builder.body_label.index(finger),
            xform=wp.transform(wp.vec3(_PLATE_POSITION * scale), wp.quat_identity()),
            hx=float(_PLATE_HALF_EXTENTS[0] * scale),
            hy=float(_PLATE_HALF_EXTENTS[1] * scale),
            hz=float(_PLATE_HALF_EXTENTS[2] * scale),
            cfg=plate_cfg,
            color=(0.32, 0.35, 0.38),
            label=f"{finger}_squeeze_plate",
        )


class Example:
    def __init__(self, viewer, args):
        if not wp.get_device().is_cuda:
            raise RuntimeError("franka_bottle_squeeze requires a CUDA device for water surface reconstruction")
        for name in (
            "water_particles",
            "substeps",
            "vbd_iterations",
            "fluid_iterations",
            "surface_voxel_size",
            "surface_max_grid_cells",
        ):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f"{name.replace('_', '-')} must be finite and positive")
        if args.segments < 12 or args.rings < 12:
            raise ValueError("segments and rings must each be at least 12")
        if args.substeps % 2:
            raise ValueError("substeps must be even so captured ping-pong states return to the same buffer")
        if not 0.0 < args.minimum_opening <= 0.04:
            raise ValueError("minimum-opening must be in (0, 0.04] meters per finger")
        if not 0.0 <= args.bottle_opacity <= 1.0:
            raise ValueError("bottle-opacity must be in [0, 1]")
        if not 0.5 * _THICKNESS <= args.water_wall_margin < 0.005:
            raise ValueError("water-wall-margin must be in [0.00015, 0.005) meters from the PET mid-surface")
        if not args.squeeze_angles or not np.isfinite(args.squeeze_angles).all():
            raise ValueError("squeeze-angles must contain finite angles in degrees")
        heights = args.squeeze_heights
        if heights is None:
            heights = [
                _DEFAULT_SQUEEZE_HEIGHTS[i % len(_DEFAULT_SQUEEZE_HEIGHTS)] for i in range(len(args.squeeze_angles))
            ]
        elif len(heights) == 1:
            heights = heights * len(args.squeeze_angles)
        if len(heights) != len(args.squeeze_angles):
            raise ValueError("squeeze-heights must contain one height or one per squeeze angle")
        if not np.isfinite(heights).all() or np.any(np.asarray(heights) < 0.06) or np.any(np.asarray(heights) > 0.145):
            raise ValueError("squeeze-heights must be finite and in [0.06, 0.145] meters above the bottle base")

        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.plasticity = args.plasticity
        self.show_particles = args.show_particles
        self.bottle_opacity = args.bottle_opacity
        self._rtx_water_material_bound = False
        self.save_frames = args.save_frames
        if self.save_frames is not None:
            self.save_frames.mkdir(parents=True, exist_ok=True)
        self.frame_index = 0
        self.minimum_opening = args.minimum_opening * _SCALE
        self.squeeze_angles = tuple(args.squeeze_angles)
        self.squeeze_heights = tuple(heights)
        self.motion_duration = len(self.squeeze_angles) * _CYCLE_DURATION

        asset = newton.utils.download_asset("franka_emika_panda")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverVBD.register_custom_attributes(builder)
        _add_robot(builder, asset, scale=_SCALE)
        self.robot_body_count = builder.body_count
        self.finger_q0 = builder.joint_q_start[builder.joint_label.index("fr3/fr3_finger_joint1")]
        self.finger_q1 = builder.joint_q_start[builder.joint_label.index("fr3/fr3_finger_joint2")]
        # Only the fingers and attached plates contact shell particles.
        for shape, body in enumerate(builder.shape_body):
            if body >= 0 and "finger" not in builder.body_label[body]:
                builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)

        vertices, triangles = _bottle_mesh(args.segments, args.rings)
        mu = _YOUNG_MODULUS * _THICKNESS / (2.0 * (1.0 + _POISSON_RATIO))
        lam = _YOUNG_MODULUS * _THICKNESS * _POISSON_RATIO / (1.0 - _POISSON_RATIO**2)
        builder.add_cloth_mesh(
            pos=wp.vec3(_BOTTLE_CENTER * _SCALE),
            rot=wp.quat_identity(),
            scale=_SCALE,
            vel=wp.vec3(0.0),
            vertices=vertices.tolist(),
            indices=triangles.flatten().tolist(),
            density=_PET_DENSITY * _THICKNESS / _SCALE**2,
            tri_ke=mu,
            tri_ka=lam,
            tri_kd=1.0e-5,
            edge_ke=1.0,
            edge_kd=1.0e-5,
            particle_radius=0.5 * _THICKNESS * _SCALE,
            color=(0.6, 0.8, 0.9),
            opacity=self.bottle_opacity,
            validate_mesh=True,
        )
        self.initial_vertices = np.asarray(builder.particle_q, dtype=np.float32)
        self.pinned = np.flatnonzero(vertices[:, 2] == 0.0)
        for i in self.pinned:
            builder.particle_mass[i] = 0.0
        yield_angles = _pet_bending_properties(builder)
        builder.custom_attributes["vbd:edge_bending_yield_angle"].values = dict(enumerate(yield_angles))
        builder.add_ground_plane(height=_TABLE_HEIGHT * _SCALE)
        builder.color()
        self.model = builder.finalize()
        self.model.soft_contact_ke = 2.0e6
        self.model.soft_contact_kd = 50.0
        self.model.soft_contact_mu = 0.45
        times, poses = self._solve_robot_poses(builder)
        lower, upper = self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy()
        if np.any(poses < lower - 1.0e-4) or np.any(poses > upper + 1.0e-4):
            raise ValueError("Requested squeeze angles and heights require a path outside the Franka joint limits")
        speeds = 1.5 * np.abs(np.diff(poses, axis=0)) / np.diff(times)[:, None]
        if np.any(speeds > self.model.joint_velocity_limit.numpy() + 1.0e-4):
            raise ValueError("Requested squeeze angles and heights require excessive joint speeds; choose closer grips")
        self.motion_times = wp.array(times, dtype=float, device=self.model.device)
        self.motion_poses = wp.array(poses, dtype=float, device=self.model.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.joint_q = wp.array(poses[0], dtype=float, device=self.model.device)
        self.joint_qd = wp.zeros_like(self.joint_q)
        newton.eval_fk(self.model, self.joint_q, self.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.joint_q, self.joint_qd, self.state_1)

        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.15)
        self.contacts = self.collision_pipeline.contacts()
        self.solver = SolverVBD(
            self.model,
            iterations=args.vbd_iterations,
            particle_enable_bending_plasticity=args.plasticity,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_margin=_THICKNESS * _SCALE,
            particle_self_contact_gap=0.30,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.08,
            collision_frequency_type={
                SolverBase.CollisionSlot.SOFT_SELF_CONTACT: SolverBase.CollisionFrequencyType.PRE_INIT,
            },
        )
        positions, spacing = _water_samples(args.water_particles, seed=args.seed)
        self.water = BottleFluid(
            positions,
            spacing,
            self.state_0.particle_q,
            triangles,
            floor=_TABLE_HEIGHT * _SCALE,
            mouth_height=(_BOTTLE_CENTER[2] + _BOTTLE_HEIGHT) * _SCALE,
            iterations=args.fluid_iterations,
            device=self.model.device,
        )

        render_builder = newton.ModelBuilder()
        _add_robot(render_builder, asset, scale=1.0)
        render_builder.add_shape_box(
            -1,
            xform=wp.transform((0.25, 0.0, _TABLE_HEIGHT - 0.03), wp.quat_identity()),
            hx=0.8,
            hy=0.55,
            hz=0.03,
            color=(0.20, 0.23, 0.25),
        )
        render_builder.add_ground_plane()
        self.render_model = render_builder.finalize()
        self.render_state = self.render_model.state()
        self.viewer.set_model(self.render_model)
        self.viewer.show_particles = False
        self.viewer.set_camera(wp.vec3(1.03, 1.05, 1.30), pitch=-16.0, yaw=-120.0)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 38.0

        self.render_shell = wp.empty_like(self.state_0.particle_q)
        wp.launch(
            _scale_positions,
            self.model.particle_count,
            inputs=[self.state_0.particle_q, self.render_shell],
            device=self.model.device,
        )
        self.shell_normals = wp.zeros_like(self.render_shell)
        self.render_water = wp.empty_like(self.water.positions)
        self.shell_indices = wp.array(triangles.flatten(), dtype=int, device=self.model.device)
        heights = vertices[triangles, 2]
        label = triangles[(heights.min(axis=1) >= 0.078) & (heights.max(axis=1) <= 0.126)]
        self.label_vertices = wp.array(label.flatten(), dtype=int, device=self.model.device)
        self.render_label = wp.empty(label.size, dtype=wp.vec3, device=self.model.device)
        self.label_normals = wp.empty_like(self.render_label)
        self.label_indices = wp.array(np.arange(label.size), dtype=int, device=self.model.device)
        # Duplicate triangle corners so UVs remain continuous across the seam.
        angle = np.unwrap(((label - 1) % args.segments) * (2.0 * np.pi / args.segments), axis=1)
        height = vertices[label, 2]
        uvs = np.stack((4.0 * angle / (2.0 * np.pi) + 0.5, (height - height.min()) / np.ptp(height)), axis=-1)
        self.label_uvs = wp.array(uvs.reshape(-1, 2), dtype=wp.vec2, device=self.model.device)
        self.label_texture = newton.examples.get_asset("bottle_label.png")
        anchors, offsets, neck_triangles = _bottle_neck(vertices, args.segments)
        self.neck_anchors = wp.array(anchors, dtype=int, device=self.model.device)
        self.neck_offsets = wp.array(offsets, dtype=wp.vec4, device=self.model.device)
        self.neck_indices = wp.array(neck_triangles.flatten(), dtype=int, device=self.model.device)
        self.render_neck = wp.empty(len(anchors), dtype=wp.vec3, device=self.model.device)
        self.neck_normals = wp.empty_like(self.render_neck)
        self.segments = args.segments
        self.surface = BottleSurface(
            self.render_shell,
            triangles,
            segments=args.segments,
            spacing=spacing / _SCALE,
            voxel_size=args.surface_voxel_size,
            margin=args.water_wall_margin,
            floor=_TABLE_HEIGHT,
            max_grid_cells=args.surface_max_grid_cells,
            device=self.model.device,
        )
        self.surface_triangle_count = 0
        self.shell_graphs = []
        if args.cuda_graph:
            # Capture each shell buffer direction separately. Keep fluid grids
            # eager: Warp 1.18 shares host grid metadata with reconstruction.
            for _ in range(2):
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self._simulate_shell()
                self.shell_graphs.append(capture.graph)

    def _solve_robot_poses(self, builder):
        """Solve each horizontal grip and a raised retreat above the neck."""
        ee = builder.body_label.index("fr3/fr3_link7")
        offset = wp.vec3(0.0, 0.0, 0.2104 * _SCALE)
        rotation_offset = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.25 * wp.pi)
        target = (_BOTTLE_CENTER + np.array([0.0, 0.0, self.squeeze_heights[0]])) * _SCALE
        objective_position = newton.ik.IKObjectivePosition(
            link_index=ee,
            link_offset=offset,
            target_positions=wp.array([target], dtype=wp.vec3, device=self.model.device),
            weight=1.0 / _SCALE,
        )
        objective_rotation = newton.ik.IKObjectiveRotation(
            link_index=ee,
            link_offset_rotation=rotation_offset,
            target_rotations=wp.array([wp.vec4(0.0, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=self.model.device),
        )
        limits = newton.ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        solver = newton.ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[objective_position, objective_rotation, limits],
            lambda_initial=0.01,
            jacobian_mode=newton.ik.IKJacobianType.ANALYTIC,
        )
        q = wp.array([[0.0, -0.6, 0.0, -2.2, 0.0, 3.17, 0.8, 4.0, 4.0]], dtype=float, device=self.model.device)
        check_state = self.model.state()
        check_velocity = wp.zeros(self.model.joint_dof_count, dtype=float, device=self.model.device)
        grips, retreats = [], []
        retreat_height = max(max(self.squeeze_heights) + 0.14, _BOTTLE_HEIGHT + 0.06)
        retreat = (_BOTTLE_CENTER + np.array([0.0, 0.0, retreat_height])) * _SCALE
        for angle, height in zip(self.squeeze_angles, self.squeeze_heights, strict=True):
            target = (_BOTTLE_CENTER + np.array([0.0, 0.0, height])) * _SCALE
            yaw = float(np.deg2rad(angle))
            grips.append(np.r_[target, yaw, 4.0, 4.0])
            retreats.append(np.r_[retreat, yaw, 4.0, 4.0])
        times, targets = _squeeze_keyframes(grips, retreats, (4, 5), self.minimum_opening)
        sampled_times, poses = [], []
        previous = targets[0]
        for key, endpoint in enumerate(targets):
            # Small Cartesian increments retain one IK branch and keep the
            # raised gripper clear while changing its heading.
            distance = np.linalg.norm(endpoint[:3] - previous[:3])
            turn = abs(endpoint[3] - previous[3])
            samples = max(1, int(np.ceil(max(distance / 1.0, turn / 0.05))))
            for sample in range(1, samples + 1):
                alpha = sample / samples
                point = (1.0 - alpha) * previous + alpha * endpoint
                rotation = (
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(point[3]))
                    * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)
                    * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
                )
                objective_rotation.set_target_rotation(0, wp.vec4(*rotation))
                objective_position.set_target_position(0, wp.vec3(point[:3]))
                solver.step(q, q, iterations=128 if key == 0 else 64)
                pose = q.numpy().flatten()
                pose[[self.finger_q0, self.finger_q1]] = point[4:]
                newton.eval_fk(self.model, wp.array(pose, device=self.model.device), check_velocity, check_state)
                achieved = check_state.body_q.numpy()[ee]
                tcp = wp.transform_point(wp.transform(wp.vec3(*achieved[:3]), wp.quat(*achieved[3:])), offset)
                orientation = wp.quat(*achieved[3:]) * rotation_offset
                angle_error = 2.0 * np.arccos(np.clip(abs(np.dot(orientation, rotation)), 0.0, 1.0))
                if np.linalg.norm(np.asarray(tcp) - point[:3]) > 0.1 or angle_error > np.deg2rad(1.0):
                    raise ValueError(
                        f"IK could not reach the {np.rad2deg(point[3]):g}-degree path; choose another angle or height"
                    )
                time = times[key] if key == 0 else (1.0 - alpha) * times[key - 1] + alpha * times[key]
                sampled_times.append(time)
                poses.append(pose)
            previous = endpoint
        return np.asarray(sampled_times, dtype=np.float32), np.asarray(poses, dtype=np.float32)

    def _simulate_shell(self):
        wp.launch(
            _robot_motion,
            self.model.joint_coord_count,
            inputs=[
                self.clock,
                self.sim_dt,
                self.motion_times,
                self.motion_poses,
                self.joint_q,
                self.joint_qd,
            ],
            device=self.model.device,
        )
        newton.eval_fk(self.model, self.joint_q, self.joint_qd, self.state_1)
        self.state_0.clear_forces()
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(_advance_clock, 1, inputs=[self.clock, self.sim_dt], device=self.model.device)

    def simulate(self):
        for substep in range(self.sim_substeps):
            if self.shell_graphs:
                wp.capture_launch(self.shell_graphs[substep % 2])
                self.state_0, self.state_1 = self.state_1, self.state_0
            else:
                self._simulate_shell()
            self.water.step(self.state_0.particle_q, self.sim_dt)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        wp.launch(
            _scale_transforms,
            self.robot_body_count,
            inputs=[self.state_0.body_q, self.render_state.body_q],
            device=self.model.device,
        )
        for source, target in ((self.state_0.particle_q, self.render_shell), (self.water.positions, self.render_water)):
            wp.launch(_scale_positions, len(source), inputs=[source, target], device=self.model.device)
        compute_normals(self.render_shell, self.shell_indices, self.shell_normals)
        wp.launch(
            _update_label,
            len(self.render_label),
            inputs=[self.render_shell, self.shell_normals, self.label_vertices, self.render_label, self.label_normals],
            device=self.model.device,
        )
        wp.launch(
            _update_neck,
            len(self.render_neck),
            inputs=[
                self.render_shell,
                self.segments,
                self.neck_anchors,
                self.neck_offsets,
                self.render_neck,
                self.neck_normals,
            ],
            device=self.model.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.render_state)
        self.viewer.log_mesh(
            "/bottle/pet",
            self.render_shell,
            self.shell_indices,
            self.shell_normals,
            color=(0.94, 0.97, 0.99),
            opacity=self.bottle_opacity,
            roughness=0.16,
            backface_culling=False,
        )
        self.viewer.log_mesh(
            "/bottle/label",
            self.render_label,
            self.label_indices,
            self.label_normals,
            uvs=self.label_uvs,
            texture=self.label_texture,
            opacity=1.0,
            roughness=0.55,
        )
        self.viewer.log_mesh(
            "/bottle/neck",
            self.render_neck,
            self.neck_indices,
            self.neck_normals,
            color=(0.94, 0.97, 0.99),
            opacity=0.85,
            roughness=0.18,
        )
        verts, indices, normals = self.surface.extract(self.render_water, self.water.surface_radii)
        if verts is not None:
            self.surface_triangle_count = indices.shape[0] // 3
            self.viewer.log_mesh(
                "/water/surface",
                verts,
                indices,
                normals,
                color=_WATER_COLOR,
                opacity=0.65,
                roughness=0.08,
                metallic=0.0,
                dynamic=True,
            )
        if self.show_particles:
            self.viewer.log_points(
                "/water/particles",
                self.render_water,
                radii=self.water.radius / _SCALE,
                colors=_WATER_COLOR,
            )
        if not self._rtx_water_material_bound:
            self._rtx_water_material_bound = bind_water_material(self.viewer, _WATER_COLOR)
        self.viewer.end_frame()
        if self.save_frames is not None:
            from PIL import Image  # noqa: PLC0415

            path = self.save_frames / f"frame_{self.frame_index:05d}.png"
            Image.fromarray(self.viewer.get_frame().numpy()).save(path)
        self.frame_index += 1

    def test_post_step(self):
        if round(self.sim_time * self.fps) % 30:
            return
        positions = self.water.positions.numpy()
        if not np.isfinite(positions).all():
            raise ValueError("Water positions contain nonfinite values")
        if np.min(positions[:, 2]) < _TABLE_HEIGHT * _SCALE - 0.01:
            raise ValueError("Water penetrated the table")
        if self.sim_time < 0.6:
            if np.any(np.abs(self.state_0.vbd.edge_plastic_angle.numpy()) > 1.0e-4):
                raise ValueError("The unloaded bottle yielded before the gripper closed")
            heights = positions[:, 2] / _SCALE - _BOTTLE_CENTER[2]
            radii = np.linalg.norm(positions[:, :2] / _SCALE - _BOTTLE_CENTER[:2], axis=1)
            if np.any(radii > _bottle_radius(heights) + 0.001):
                raise ValueError("Water escaped through the resting bottle wall")

    def test_final(self):
        shell = self.state_0.particle_q.numpy()
        water = self.water.positions.numpy()
        if not np.isfinite(shell).all() or not np.isfinite(water).all():
            raise ValueError("The shell or water contains nonfinite positions")
        np.testing.assert_allclose(shell[self.pinned], self.initial_vertices[self.pinned], atol=1.0e-5)
        if np.max(np.linalg.norm(shell - self.initial_vertices, axis=1)) > 20.0:
            raise ValueError("The bottle moved more than 20 cm from its rest shape")
        if self.surface_triangle_count == 0:
            raise ValueError("Water surface reconstruction produced no triangles")
        if self.plasticity and self.sim_time >= 2.2:
            if np.count_nonzero(np.abs(self.state_0.vbd.edge_plastic_angle.numpy()) > 1.0e-3) == 0:
                raise ValueError("The squeeze did not yield any PET bending hinges")
        if self.sim_time >= self.motion_duration - self.frame_dt:
            np.testing.assert_allclose(self.joint_q.numpy()[[self.finger_q0, self.finger_q1]], 4.0, atol=1.0e-5)
        if self.plasticity and self.sim_time >= self.motion_duration - 1.0:
            body = (self.initial_vertices[:, 2] - _BOTTLE_CENTER[2] * _SCALE > 8.0) & (
                self.initial_vertices[:, 2] - _BOTTLE_CENTER[2] * _SCALE < 14.0
            )
            if np.max(np.linalg.norm(shell[body] - self.initial_vertices[body], axis=1)) < _THICKNESS * _SCALE:
                raise ValueError("The released bottle did not retain a dent larger than the PET wall thickness")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1440)
        parser.add_argument(
            "--squeeze-angles",
            type=float,
            nargs="+",
            default=[60.0, 120.0, 0.0],
            help="Grip yaw angles around the bottle [deg]",
        )
        parser.add_argument(
            "--squeeze-heights",
            type=float,
            nargs="+",
            default=None,
            help="Grip heights above the bottle base [m]; one shared value or one per angle (default: 0.112, 0.130, 0.094 repeating)",
        )
        parser.add_argument("--water-particles", type=int, default=32_000)
        parser.add_argument("--segments", type=int, default=64)
        parser.add_argument("--rings", type=int, default=56)
        parser.add_argument("--substeps", type=int, default=8)
        parser.add_argument("--vbd-iterations", type=int, default=320)
        parser.add_argument("--fluid-iterations", type=int, default=8)
        parser.add_argument(
            "--minimum-opening",
            type=float,
            default=0.018,
            help="Final per-finger joint opening [m]; the plate gap is twice this value",
        )
        parser.add_argument("--bottle-opacity", type=float, default=0.55, help="PET shell display opacity [0, 1]")
        parser.add_argument(
            "--water-wall-margin",
            type=float,
            default=0.0005,
            help="Water surface clearance from the PET mid-surface [m]",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--plasticity", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument(
            "--cuda-graph",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture robot and VBD shell substeps; fluid and surface extraction execute outside capture",
        )
        parser.add_argument("--show-particles", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument(
            "--surface-voxel-size", type=float, default=0.002, help="Water reconstruction voxel size [m]"
        )
        parser.add_argument("--surface-max-grid-cells", type=int, default=4_000_000)
        parser.add_argument(
            "--save-frames", type=Path, default=None, help="Save rendered PNGs (use --viewer gl or rtx --headless)"
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
