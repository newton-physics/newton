# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sort parcels on a two-level conveyor with Style3D.

Build a two-level conveyor with a 0.5 m wide belt. Five parcels
of different proportions travel around the lower bend, up the ramp and around
the upper bend. Each arrival is assigned a random return belt. The selected
gate pivots across the upper deck to guide the parcel onto that belt below.
The belt tread scrolls at the prescribed surface speed.
Style3D couples the rigid boxes to cloth wrappers on the flat, long, and standard cartons.
Each wrapper is cut from two equal rectangles with extra fabric at the side seam.

Success criteria:
- Boxes accelerate from rest through friction with the moving belt.
- The tread follows the direction of travel through the bends and along the return belts.
- Boxes follow both bends, climb the ramp, and return to the lower level.
- Each parcel passes the preceding gates and drops at its assigned gate.
- The side guards keep boxes on the conveyor between transfer openings.
- The cloth can fold around each parcel while the sewn panels enclose it through discharge.

Runs:
- `python -m newton.examples style3d_parcel_sorting_conveyor`
- `python -m newton.examples style3d_parcel_sorting_conveyor --belt-speed 0.4`
"""

import math
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverStyle3D, style3d

BELT_SPEED = 1.5  # centerline speed [m/s]
TREAD_REPEAT = 0.48  # distance between successive chevrons [m]
# Dimensions are in meters. The two turns have different centerline radii.
HALF_WIDTH = 0.25
DECK_THICKNESS = 0.04
GUARD_HEIGHT = 0.14
GUARD_THICKNESS = 0.025
LOW_Z = 0.144
HIGH_Z = LOW_Z + 1.521
LOW_TURN_X = 2.1245
HIGH_TURN_X = -4.159
LOW_RADIUS = 0.75
HIGH_RADIUS = 1.3625
HIGH_RADIUS_X = 1.316
UPPER_LANE_Y = LOW_RADIUS - 2.0 * HIGH_RADIUS
GATE_PIVOTS_X = (-3.265, -2.119, -1.084)
GATE_ANGLE = math.atan2(0.54, 0.746)
GATE_LENGTH = math.hypot(0.54, 0.746)
GATE_RATE = 2.4  # hinge angular speed [rad/s]
BELT_COLOR = (0.05, 0.05, 0.05)
FRAME_COLOR = (0.54, 0.59, 0.62)
WRAPPER_COLORS = {"flat": (0.22, 0.55, 0.78), "long": (0.24, 0.65, 0.48), "carton": (0.67, 0.36, 0.52)}


@wp.kernel
def _assign_routes(
    body_q: wp.array[wp.transform],
    parcels: wp.array[int],
    gate_count: int,
    seed: int,
    visits: wp.array[int],
    routes: wp.array[int],
):
    i = wp.tid()
    pos = wp.transform_get_translation(body_q[parcels[i]])
    if pos[2] < LOW_Z + 0.4:
        routes[i] = -1
    elif routes[i] < 0 and pos[2] > HIGH_Z and pos[1] < UPPER_LANE_Y + 0.4:
        rng = wp.rand_init(seed, visits[i] * parcels.shape[0] + i)
        routes[i] = wp.randi(rng, 0, gate_count)
        visits[i] += 1


@wp.kernel
def _drive_gates(
    body_q: wp.array[wp.transform],
    parcels: wp.array[int],
    routes: wp.array[int],
    hinges: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    lead: float,
    dt: float,
    angles: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    gate = wp.tid()
    hinge = hinges[gate]
    pivot_x = wp.transform_get_translation(joint_X_p[hinge])[0]
    front_x = pivot_x - GATE_LENGTH - lead
    target = float(0.0)
    # The leading parcel has right of way until it clears the arm or drops.
    for i in range(parcels.shape[0]):
        pos = wp.transform_get_translation(body_q[parcels[i]])
        if routes[i] >= gate and pos[2] > HIGH_Z - 0.22 and front_x < pos[0] and pos[0] < pivot_x + 0.25:
            front_x = pos[0]
            target = wp.where(routes[i] == gate, GATE_ANGLE, 0.0)
    delta = wp.clamp(target - angles[gate], -GATE_RATE * dt, GATE_RATE * dt)
    angles[gate] += delta
    joint_q[q_start[hinge]] = angles[gate]
    joint_qd[qd_start[hinge]] = delta / dt


def _make_belt_tread(belts):
    """Map one repeating rubber tread along the top surfaces of the kinematic belts."""
    points, triangles, uvs = [], [], []
    path_length = 0.0
    for sections, radius in belts:
        centers = sections.mean(axis=1)
        if radius is None:
            travel = centers[-1] - centers[0]
            length = np.linalg.norm(travel)
            # A diagonal return belt has an oblique cut where it joins the lower deck.
            along = (sections - centers[0]) @ (travel / length)
        else:
            length = np.pi * radius
            # Equal angle steps give both edges the same angular velocity about the pivot.
            along = np.linspace(0.0, length, len(sections))[:, None]
        uv = np.empty((len(sections), 2, 2), dtype=np.float32)
        uv[:, :, 0] = (path_length + along) / TREAD_REPEAT
        uv[:, :, 1] = (0.0, 1.0)
        path_length += length

        start = len(points)
        strip = sections.reshape(-1, 3)
        # Lift only the visual skin clear of the solid deck to avoid depth flicker.
        points.extend(strip + np.array([0.0, 0.0, 0.001]))
        uvs.extend(uv.reshape(-1, 2))
        for i in range(start, start + len(strip) - 2, 2):
            triangles.extend((i, i + 1, i + 3, i, i + 3, i + 2))

    v, u = np.mgrid[0:96, 0:96] / 96.0
    tread = np.empty((*u.shape, 3), dtype=np.uint8)
    tread[:] = (12, 12, 12)
    tread[(u * 8) % 1.0 < 0.14] = (24, 24, 24)
    chevron = np.abs(u - (0.66 - 0.42 * np.abs(v - 0.5))) < 0.045
    tread[chevron & (v > 0.13) & (v < 0.87)] = (108, 136, 138)
    return newton.Mesh(points, triangles, uvs=uvs, texture=tread, compute_inertia=False)


@wp.kernel
def _scroll_tread(rest_uv: wp.array[wp.vec2], phase: float, uv: wp.array[wp.vec2]):
    i = wp.tid()
    uv[i] = rest_uv[i] - wp.vec2(phase, 0.0)


def _deck_panel(start, end):
    """Extrude one convex panel between adjacent cross-sections, left edge first."""
    top = np.array([start[0], start[1], end[1], end[0]])
    vertices = np.vstack((top, top - (0, 0, DECK_THICKNESS)))
    faces = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    indices = [(a, b, c, a, c, d) for a, b, c, d in faces]
    return newton.Mesh(vertices, np.asarray(indices).flatten(), compute_inertia=False)


def _add_guard(builder, label, start, end, side, height=GUARD_HEIGHT):
    """Place a stationary guard just outside a belt edge, following its slope."""
    dx, dy, dz = end - start
    run = math.hypot(dx, dy)
    outward = side * GUARD_THICKNESS / (2 * run)
    half_height = (height + DECK_THICKNESS) / 2
    center = (start + end) / 2 + (-dy * outward, dx * outward, half_height - DECK_THICKNESS)
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(*center), wp.quat_rpy(0.0, -math.atan2(dz, run), math.atan2(dy, dx))),
        hx=math.hypot(run, dz) / 2,
        hy=GUARD_THICKNESS / 2,
        hz=half_height,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.05),
        color=FRAME_COLOR,
        label=label,
    )


def _add_belt(
    builder, belts, label, sections, speed, *, pivot=None, radius=None, guard_sides=(0, 1), guard_height=GUARD_HEIGHT
):
    """Prescribe a straight belt's linear velocity or a bend's angular velocity about its pivot."""
    sections = np.asarray(sections)
    origin = sections[0].mean(axis=0) if pivot is None else np.asarray(pivot)
    xform = wp.transform(wp.vec3(*origin), wp.quat_identity())
    body = builder.add_link(xform=xform, is_kinematic=True, label=label)
    belts.append((sections, radius))
    local = sections - origin
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    for start, end in pairwise(local):
        builder.add_shape_convex_hull(body, mesh=_deck_panel(start, end), color=BELT_COLOR, cfg=cfg)
    for edge in guard_sides:
        for i, (start, end) in enumerate(pairwise(sections[:, edge])):
            _add_guard(builder, f"{label}_guard_{edge}_{i}", start, end, 1 if edge == 0 else -1, guard_height)
    if pivot is None:
        direction = sections[-1].mean(axis=0) - origin
        axis = wp.vec3(*(direction / np.linalg.norm(direction)))
        joint = builder.add_joint_prismatic(-1, body, parent_xform=xform, axis=axis)
        rate = speed
    else:
        joint = builder.add_joint_revolute(-1, body, parent_xform=xform, axis=newton.Axis.Z)
        rate = speed / radius
    builder.joint_qd[builder.joint_qd_start[joint]] = rate
    builder.add_articulation([joint])


def _straight_section(x, y, z, direction):
    dy = direction * HALF_WIDTH
    return [(x, y + dy, z), (x, y - dy, z)]


def _add_turn(builder, belts, label, pivot, radius, start_angle, speed, *, x_radius=None):
    # Both OBJ bends have ten facets. The elevated bend is slightly flattened along X.
    cx, cy, z = pivot
    if x_radius is None:
        x_radius = radius
    sections = [
        [
            (cx + (x_radius + offset) * math.cos(a), cy + (radius + offset) * math.sin(a), z)
            for offset in (-HALF_WIDTH, HALF_WIDTH)
        ]
        for a in np.linspace(start_angle, start_angle + math.pi, 11)
    ]
    _add_belt(builder, belts, label, sections, speed, pivot=pivot, radius=radius)


def _add_transfer_guards(builder, label, x_start, x_end, y, z, side, openings):
    """Guard the straight edge between its three transfer openings."""
    x = x_start
    for i, (left, right) in enumerate(openings):
        if left - x > GUARD_THICKNESS:
            _add_guard(builder, f"{label}_guard_{i}", np.array((x, y, z)), np.array((left, y, z)), side)
        x = right
    if x_end - x > GUARD_THICKNESS:
        _add_guard(builder, f"{label}_guard_end", np.array((x, y, z)), np.array((x_end, y, z)), side)


def _build_bridge(builder, speed):
    belts = []
    low_in = _straight_section(LOW_TURN_X, -LOW_RADIUS, LOW_Z, 1)
    low_out = _straight_section(LOW_TURN_X, LOW_RADIUS, LOW_Z, -1)
    high_in = _straight_section(HIGH_TURN_X, LOW_RADIUS, HIGH_Z, -1)
    high_out = _straight_section(HIGH_TURN_X, UPPER_LANE_Y, HIGH_Z, 1)

    _add_belt(
        builder,
        belts,
        "lower_deck",
        [_straight_section(-2.994, -LOW_RADIUS, LOW_Z, 1), low_in],
        speed,
        guard_sides=(0,),
    )
    _add_turn(builder, belts, "lower_bend", (LOW_TURN_X, 0.0, LOW_Z), LOW_RADIUS, -math.pi / 2, speed)
    _add_belt(builder, belts, "incline", [low_out, high_in], speed)
    _add_turn(
        builder,
        belts,
        "upper_bend",
        (HIGH_TURN_X, LOW_RADIUS - HIGH_RADIUS, HIGH_Z),
        HIGH_RADIUS,
        math.pi / 2,
        speed,
        x_radius=HIGH_RADIUS_X,
    )
    _add_belt(
        builder,
        belts,
        "upper_deck",
        [high_out, _straight_section(-1.052, UPPER_LANE_Y, HIGH_Z, 1)],
        speed,
        guard_sides=(1,),
    )

    lower_openings = []
    upper_openings = []
    hinges = []
    for i, (branch_x, pivot_x) in enumerate(zip((-4.225, -3.135, -2.033), GATE_PIVOTS_X, strict=True)):
        # The lower mouths are cut parallel to the receiving belt.
        sections = [
            [(branch_x, -1.729, LOW_Z), (branch_x + 0.25, -2.162, LOW_Z)],
            [(branch_x + 1.245, -1.0, LOW_Z), (branch_x + 2.246, -1.0, LOW_Z)],
        ]
        # The taller receiving guards catch parcels tumbling off the upper deck.
        _add_belt(builder, belts, f"return_{i}", sections, speed, guard_height=0.28)
        lower_openings.append((sections[1][0][0], sections[1][1][0]))
        # The stowed arm doubles as the rail across the discharge opening.
        upper_openings.append((pivot_x - GATE_LENGTH, pivot_x))
        pivot = wp.transform(wp.vec3(pivot_x, UPPER_LANE_Y + HALF_WIDTH + 0.0125, HIGH_Z + 0.0875), wp.quat_identity())
        arm = builder.add_link(xform=pivot, is_kinematic=True, label=f"gate_{i}")
        hinge = builder.add_joint_revolute(-1, arm, axis=newton.Axis.Z, parent_xform=pivot, label=f"hinge_{i}")
        builder.add_articulation([hinge])
        hinges.append(hinge)
        builder.add_shape_box(
            arm,
            xform=wp.transform(wp.vec3(-GATE_LENGTH / 2, 0.0, 0.0), wp.quat_identity()),
            hx=GATE_LENGTH / 2,
            hy=0.019,
            hz=0.0875,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.05),
            color=(0.92, 0.53, 0.12),
            label=f"guide_{i}",
        )
        builder.add_shape_cylinder(
            arm,
            radius=0.032,
            half_height=0.105,
            color=FRAME_COLOR,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False),
            label=f"hinge_pin_{i}",
        )

    _add_transfer_guards(
        builder, "discharge", HIGH_TURN_X, -1.052, UPPER_LANE_Y + HALF_WIDTH, HIGH_Z, 1, upper_openings
    )
    _add_transfer_guards(builder, "merge", -2.994, LOW_TURN_X, -LOW_RADIUS - HALF_WIDTH, LOW_Z, -1, lower_openings)

    for i, x in enumerate((-4.048, -2.829, -1.245, 0.602, 1.923)):
        ramp_fraction = (LOW_TURN_X - x) / (LOW_TURN_X - HIGH_TURN_X)
        height = LOW_Z + ramp_fraction * (HIGH_Z - LOW_Z) - DECK_THICKNESS
        for j, y in enumerate((0.563, 0.945)):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, y, height / 2), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=height / 2,
                color=FRAME_COLOR,
                label=f"post_{i}_{j}",
            )
    return hinges, _make_belt_tread(belts)


def _make_wrapper(half_extents, *, spacing, face_ease):
    """Fit two identical rectangular stretch-cloth panels and sew them at mid-height."""
    hx, hy, hz = half_extents
    nx, ny = np.ceil(2 * half_extents[:2] / spacing).astype(int)
    u, v = np.meshgrid(np.linspace(-1, 1, nx + 1), np.linspace(-1, 1, ny + 1))
    face = np.column_stack((hx * u.ravel(), hy * v.ravel()))
    grid = np.arange(len(face)).reshape(ny + 1, nx + 1)
    rim = np.concatenate((grid[0, :-1], grid[:-1, -1], grid[-1, :0:-1], grid[:0:-1, 0]))
    a, b, c, d = (part.ravel() for part in (grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]))
    triangles = [np.column_stack((a, b, c)), np.column_stack((a, c, d))]
    vertices = [np.column_stack((face, np.full(len(face), hz)))]

    # Cut both panels with the same face ease and allowance down to the shared seam.
    inner = (1.0 + face_ease) * np.array((hx, hy))
    outer = inner + 0.018 + 0.30 * hz
    face_uv = np.column_stack((u.ravel(), v.ravel()))
    pattern = [face_uv * inner]
    edge = rim
    count = len(face)
    for t in np.linspace(0, 1, math.ceil(hz / spacing) + 1)[1:]:
        ring = np.arange(count, count + len(rim))
        vertices.append(np.column_stack((face[rim], np.full(len(rim), hz * (1 - t)))))
        pattern.append(face_uv[rim] * ((1 - t) * inner + t * outer))
        triangles.extend(
            (
                np.column_stack((edge, ring, np.roll(ring, -1))),
                np.column_stack((edge, np.roll(ring, -1), np.roll(edge, -1))),
            )
        )
        edge = ring
        count += len(rim)

    top = np.vstack(vertices)
    uv = np.vstack(pattern)
    tris = np.vstack(triangles)
    # The lower panel has independent pattern coordinates. Its outer ring uses
    # the same physical particle indices as the upper ring, closing the seam.
    seam_start = edge[0]
    lower = np.arange(count)
    lower[:seam_start] += count
    return (
        np.vstack((top, top[:seam_start] * (1, 1, -1))),
        np.vstack((tris, lower[tris[:, ::-1]])).ravel(),
        np.vstack((uv, uv * (-1, 1))),
        np.vstack((tris, count + tris[:, ::-1])).ravel(),
    )


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 2
        self.sim_time = 0.0
        self.belt_speed = getattr(args, "belt_speed", BELT_SPEED)
        builder = newton.ModelBuilder()
        SolverStyle3D.register_custom_attributes(builder)
        builder.rigid_gap = 0.02
        # Resolve the impact onto the thin return belts after the 1.52 m drop.
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 700.0
        builder.add_ground_plane()
        hinges, tread = _build_bridge(builder, self.belt_speed)

        ramp_angle = math.atan2(HIGH_Z - LOW_Z, LOW_TURN_X - HIGH_TURN_X)
        self.box_bodies = []
        triangle_colors = []
        ramp_z = LOW_Z + (LOW_TURN_X + 1.0) * math.tan(ramp_angle)
        # Full parcel dimensions (length, width, height) in meters.
        parcels = (
            ("flat", (0.30, 0.22, 0.07), -1.6, -LOW_RADIUS, LOW_Z, 0.0, 0.0),
            ("cube", (0.20, 0.20, 0.20), LOW_TURN_X + LOW_RADIUS, 0.0, LOW_Z, 0.0, math.pi / 2),
            ("long", (0.34, 0.13, 0.12), -1.0, LOW_RADIUS, ramp_z, ramp_angle, 0.0),
            (
                "tall",
                (0.18, 0.18, 0.30),
                HIGH_TURN_X - HIGH_RADIUS_X,
                LOW_RADIUS - HIGH_RADIUS,
                HIGH_Z,
                0.0,
                math.pi / 2,
            ),
            ("carton", (0.24, 0.18, 0.16), HIGH_TURN_X - 0.30, UPPER_LANE_Y + 0.035, HIGH_Z, 0.0, 0.0),
        )
        for i, (name, size, x, y, deck_z, pitch, yaw) in enumerate(parcels):
            hx, hy, hz = (dimension / 2 for dimension in size)
            rot = wp.quat_rpy(0.0, pitch, yaw)
            z = deck_z + (hz + 0.03) / math.cos(pitch)
            body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z), rot), label=f"box_{name}")
            # The opaque fabric is the visible exterior of a wrapped parcel.
            cfg = builder.default_shape_cfg.copy()
            cfg.is_visible = name not in WRAPPER_COLORS
            builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg, color=(0.81, 0.43 + 0.06 * (i % 3), 0.21))
            self.box_bodies.append(body)
            if name in WRAPPER_COLORS:
                half_extents = np.array((hx, hy, hz))
                vertices, indices, pattern, pattern_indices = _make_wrapper(
                    half_extents + 0.005, spacing=0.02, face_ease=0.10 if name in ("long", "carton") else 0.08
                )
                style3d.add_cloth_mesh(
                    builder,
                    pos=wp.vec3(x, y, z),
                    rot=rot,
                    vel=wp.vec3(),
                    vertices=vertices,
                    indices=indices,
                    panel_verts=pattern,
                    panel_indices=pattern_indices,
                    density=0.15,
                    particle_radius=0.004,
                    tri_aniso_ke=wp.vec3(250.0, 250.0, 4.0),
                    edge_aniso_ke=wp.vec3(2.0e-7),
                    label=f"wrapper_{name}",
                )
                triangle_colors.extend([WRAPPER_COLORS[name]] * (len(indices) // 3))

        self.model = builder.finalize()
        device = self.model.device
        self.tread_points = wp.array(tread.vertices, dtype=wp.vec3, device=device)
        self.tread_indices = wp.array(tread.indices, dtype=wp.int32, device=device)
        self.tread_rest_uv = wp.array(tread.uvs, dtype=wp.vec2, device=device)
        self.tread_uv = wp.empty_like(self.tread_rest_uv)
        self.tread_texture = tread.texture
        self.tread_xform = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.tread_color = wp.array([wp.vec3(1.0)], dtype=wp.vec3, device=device)
        self.tread_material = wp.array([wp.vec4(0.9, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=device)
        seed = getattr(args, "seed", None)
        self.route_seed = int(np.random.default_rng().integers(2**31)) if seed is None else seed
        self.parcels = wp.array(self.box_bodies, dtype=int, device=device)
        self.hinges = wp.array(hinges, dtype=int, device=device)
        self.gate_angles = wp.zeros(len(hinges), dtype=float, device=device)
        self.routes = wp.full(len(parcels), -1, dtype=int, device=device)
        self.visits = wp.zeros(len(parcels), dtype=int, device=device)
        self.model.tri_color = wp.array(triangle_colors, dtype=wp.vec3, device=self.model.device)
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 0.5
        self.model.soft_contact_kf = 10.0
        self.model.soft_contact_mu = 0.7
        self.solver = SolverStyle3D(
            self.model, dynamics=SolverStyle3D.Dynamics.ALL, config=SolverStyle3D.Config(linear_iterations=15)
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, include_static_kinematic_pairs=False, soft_contact_max=4 * self.model.particle_count
        )
        self.contacts = self.collision_pipeline.contacts()
        # The belt footprints stay fixed while their prescribed twists drive contact friction.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(4.92, 4.21, 7.58), -43.34, -138.19)
        self.graph = None
        if self.model.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        dt = self.frame_dt / self.sim_substeps
        if self.belt_speed > 0:
            wp.launch(
                _assign_routes,
                dim=len(self.box_bodies),
                inputs=[self.state_0.body_q, self.parcels, len(self.hinges), self.route_seed, self.visits, self.routes],
                device=self.model.device,
            )
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                _drive_gates,
                dim=len(self.hinges),
                inputs=[
                    self.state_0.body_q,
                    self.parcels,
                    self.routes,
                    self.hinges,
                    self.model.joint_X_p,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    0.24 + abs(self.belt_speed) * GATE_ANGLE / GATE_RATE,
                    dt,
                    self.gate_angles,
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                ],
                device=self.model.device,
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        phase = (self.sim_time * self.belt_speed / TREAD_REPEAT) % 1.0
        wp.launch(
            _scroll_tread,
            dim=len(self.tread_uv),
            inputs=[self.tread_rest_uv, phase],
            outputs=[self.tread_uv],
            device=self.model.device,
        )
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/geometry/belt_tread",
            self.tread_points,
            self.tread_indices,
            uvs=self.tread_uv,
            texture=self.tread_texture,
            hidden=True,
        )
        self.viewer.log_instances(
            "/belt_tread",
            "/geometry/belt_tread",
            self.tread_xform,
            None,
            self.tread_color,
            self.tread_material,
        )
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Check that the flat parcel starts moving in the belt direction."""
        if round(self.sim_time / self.frame_dt) == 8 and abs(self.belt_speed) > 0.05:
            vx = self.state_0.body_qd.numpy()[self.box_bodies[0], 0]
            assert np.sign(self.belt_speed) * vx > 0.01, "The lower belt did not set the flat parcel in motion."

    def test_final(self):
        """Check that box poses and wrapper positions remain finite."""
        assert np.isfinite(self.state_0.body_q.numpy()[self.box_bodies]).all(), "Non-finite box pose."
        assert np.isfinite(self.state_0.particle_q.numpy()).all(), "Non-finite wrapper position."


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--belt-speed", type=float, default=BELT_SPEED, help="Belt centerline speed [m/s].")
    parser.add_argument("--seed", type=int, default=None, help="Replay the same random gate assignments.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
