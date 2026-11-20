# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example IK Stewart Platform (closed loops + hard joint limits)

Inverse kinematics of a field of Stewart platforms, solved as one batch: one IK problem per
platform.
- Six legs connect the base to the platform: a universal joint at the base, a prismatic actuator,
  a ball joint at the platform.
- The articulation tree reaches the platform through the first leg; the ball joints of the other
  five legs are loop-closing joints, added outside the articulation.
- The actuators have a limited stroke and the universal joints a limited swing.
- Every platform rides a wave that rolls across the field, heaving, swaying and tilting with it;
  the waves grow until they ask for more than the legs can reach, then calm down again.
- ``IKOptimizer.CUNLS`` keeps the loops closed (equality constraints) and the joints inside their
  limits (hard bounds): a platform whose target is out of reach stops where its legs bottom or top
  out.

Overlays:
- Legs glow orange as their actuators approach the end of their stroke.
- A green dot marks each loop-closing joint.
- A translucent red platform shows a target the legs cannot reach.

Success criteria:
- A field of Stewart platforms rides a rolling wave; as the waves grow, bands of orange legs travel
  across the field and red ghost platforms appear above and below the real ones.
- Every loop stays closed (gap under 1 mm); no actuator leaves its stroke.

Requires cuNLS (``pip install newton[cunls-cu12]`` or ``newton[cunls-cu13]``).

Runs:
- `python -m newton.examples ik_stewart_platform`
- `python -m newton.examples ik_stewart_platform --world-count 16`
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

# Geometry [m]
BASE_RADIUS = 0.30
PLATFORM_RADIUS = 0.20
JOINT_SPREAD = math.radians(10.0)  # half the angle between the two joints of a pair
BASE_HEIGHT = 0.06  # top of the base plate
PLATFORM_HEIGHT = 0.46  # platform above the base plate at rest
PLATE_HALF_THICKNESS = 0.012
PLATFORM_SPACING = 1.25  # between neighboring platforms

# Limits: actuator stroke around the rest length [m], universal joint swing [rad].
STROKE = 0.07
SWING = math.radians(50.0)

# Wave: heave [m], sway [m], tilt and yaw [rad], each at calm and at the height of the storm.
WAVE_LENGTH = 7.0  # [m]
WAVE_PERIOD = 2.5  # [s]
STORM_PERIOD = 12.0  # [s], calm -> storm -> calm
HEAVE = (0.02, 0.08)
SWAY = (0.01, 0.035)
TILT = (0.04, 0.17)
YAW = (0.03, 0.09)

# Overlays
LEG_COLOR = wp.constant(wp.vec3(0.22, 0.24, 0.28))
ROD_COLOR = wp.constant(wp.vec3(0.80, 0.82, 0.86))
GLOW_COLOR = wp.constant(wp.vec3(1.0, 0.42, 0.0))
GLOW_START = wp.constant(STROKE - 0.008)  # [m] of extension where a leg starts to glow
LOOP_COLOR = (0.1, 0.95, 0.35)
GHOST_COLOR = (1.0, 0.12, 0.1)
GHOST_OPACITY = wp.constant(0.35)
GHOST_RADIUS = wp.constant(PLATFORM_RADIUS + 0.02)
GHOST_SEGMENTS = 24
GHOST_FADE = wp.constant(0.02)  # [m]: a ghost fades in from half of this miss to 1.5 times it


@wp.kernel
def color_legs(
    joint_q: wp.array2d[wp.float32],
    stroke_coords: wp.array[wp.int32],
    leg_shapes: wp.array2d[wp.int32],
    shapes_per_world: int,
    stroke: float,
    shape_color: wp.array[wp.vec3],
):
    """Color each leg by how close its actuator is to the end of its stroke."""
    world, leg = wp.tid()
    extension = wp.abs(joint_q[world, stroke_coords[leg]])
    glow = wp.clamp((extension - GLOW_START) / (stroke - GLOW_START), 0.0, 1.0)
    base = world * shapes_per_world
    shape_color[base + leg_shapes[leg, 0]] = wp.lerp(LEG_COLOR, GLOW_COLOR, glow * glow)
    shape_color[base + leg_shapes[leg, 1]] = wp.lerp(ROD_COLOR, GLOW_COLOR, glow * glow)


@wp.kernel
def mark_loops(
    body_q: wp.array[wp.transform],
    bodies_per_world: int,
    rod_bodies: wp.array[wp.int32],
    rod_anchors: wp.array[wp.vec3],
    platform_body: int,
    platform_anchors: wp.array[wp.vec3],
    world_offsets: wp.array[wp.vec3],
    points: wp.array2d[wp.vec3],
    gaps: wp.array2d[wp.float32],
):
    """Place a dot on each loop-closing joint and measure the gap between its two anchors."""
    world, loop = wp.tid()
    base = world * bodies_per_world
    rod_end = wp.transform_point(body_q[base + rod_bodies[loop]], rod_anchors[loop])
    platform_end = wp.transform_point(body_q[base + platform_body], platform_anchors[loop])
    gap = wp.length(rod_end - platform_end)
    points[world, loop] = platform_end + world_offsets[world]
    gaps[world, loop] = gap


@wp.kernel
def place_ghosts(
    body_q: wp.array[wp.transform],
    bodies_per_world: int,
    platform_body: int,
    target_positions: wp.array[wp.vec3],
    target_rotations: wp.array[wp.vec4],
    world_offsets: wp.array[wp.vec3],
    xforms: wp.array[wp.transform],
    opacities: wp.array[wp.float32],
    ring_starts: wp.array2d[wp.vec3],
    ring_ends: wp.array2d[wp.vec3],
):
    """Show the target platform pose where the platform misses it (center error plus rim error):
    a translucent disc and a ring around it, one ring segment per thread."""
    world, segment = wp.tid()
    r = target_rotations[world]
    target = wp.transform(target_positions[world] + world_offsets[world], wp.quat(r[0], r[1], r[2], r[3]))
    platform = body_q[world * bodies_per_world + platform_body]
    miss = wp.length(wp.transform_get_translation(platform) - target_positions[world])
    turn = wp.quat_inverse(wp.transform_get_rotation(platform)) * wp.transform_get_rotation(target)
    miss += PLATFORM_RADIUS * 2.0 * wp.acos(wp.min(wp.abs(turn[3]), 1.0))
    shown = wp.clamp(miss / GHOST_FADE - 0.5, 0.0, 1.0)
    segments = ring_starts.shape[1]
    a = 2.0 * wp.pi * float(segment) / float(segments)
    b = 2.0 * wp.pi * float(segment + 1) / float(segments)
    start = wp.transform_point(target, wp.vec3(wp.cos(a), wp.sin(a), 0.0) * GHOST_RADIUS)
    end = wp.transform_point(target, wp.vec3(wp.cos(b), wp.sin(b), 0.0) * GHOST_RADIUS)
    ring_starts[world, segment] = start
    ring_ends[world, segment] = wp.where(shown > 0.0, end, start)  # an empty segment when reached
    if segment == 0:
        xforms[world] = target
        opacities[world] = GHOST_OPACITY * shown


def joint_angles():
    """Angles [rad] of the base and platform joints of the six legs.

    The base joints come in three pairs, 120 degrees apart; each leg leans to a platform joint 60
    degrees around, so the legs form the zigzag of a Gough-Stewart platform.
    """
    base, platform = [], []
    for leg in range(6):
        pair, side = divmod(leg, 2)
        sign = 1.0 if side else -1.0
        base.append(2.0 * math.pi * pair / 3.0 + sign * JOINT_SPREAD)
        platform.append(2.0 * math.pi * pair / 3.0 + sign * (math.pi / 3.0 - JOINT_SPREAD))
    return base, platform


def tilt_quats(yaw, axis, angle):
    """Quaternions (x, y, z, w) of a turn by ``angle`` [rad] about the horizontal ``axis`` after
    a turn by ``yaw`` [rad] about the vertical axis, one per platform."""
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    ca, sa = np.cos(0.5 * angle), np.sin(0.5 * angle)
    # q_tilt * q_yaw with q_tilt = (axis_x sa, axis_y sa, 0, ca) and q_yaw = (0, 0, sy, cy)
    ax, ay = axis[0] * sa, axis[1] * sa
    return np.stack([ax * cy + ay * sy, ay * cy - ax * sy, ca * sy, ca * cy], axis=1).astype(np.float32)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.world_count = args.world_count

        self.viewer = viewer
        self.viewer.picking_enabled = False

        # One platform is the IK model; the scene holds a copy of it in each world.
        robot = newton.ModelBuilder()
        self.build_stewart_platform(robot)
        self.robot_model = robot.finalize()

        scene = newton.ModelBuilder()
        scene.replicate(robot, self.world_count)
        scene.add_ground_plane()
        self.model = scene.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((PLATFORM_SPACING, PLATFORM_SPACING, 0.0))
        self.world_offsets = self.viewer.world_offsets
        side = PLATFORM_SPACING * math.ceil(math.sqrt(self.world_count))
        self.viewer.set_camera(
            pos=wp.vec3(0.3 * side + 0.4, -0.25 * side - 0.3, 0.1 * side + 0.4), pitch=-17.0, yaw=128.0
        )

        # ------------------------------------------------------------------
        # IK: platform position and orientation, joint limits; one problem per platform
        # ------------------------------------------------------------------
        self.target_positions = wp.zeros(self.world_count, dtype=wp.vec3)
        self.target_rotations = wp.zeros(self.world_count, dtype=wp.vec4)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.platform,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=self.target_positions,
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.platform,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.target_rotations,
        )
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.robot_model.joint_limit_lower,
            joint_limit_upper=self.robot_model.joint_limit_upper,
        )
        # Row i of joint_q holds the coordinates of world i.
        self.joint_q = self.model.joint_q.reshape((self.world_count, self.robot_model.joint_coord_count))
        self.ik_iters = 30
        self.solver = ik.IKSolver(
            model=self.robot_model,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.limit_obj],
            optimizer=ik.IKOptimizer.CUNLS,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        # The wave rolls diagonally across the field; its phase at each platform.
        self.wave_direction = np.array([math.cos(0.5), math.sin(0.5)])
        self.wave_phase = 2.0 * math.pi * (self.world_offsets.numpy()[:, :2] @ self.wave_direction) / WAVE_LENGTH
        # The crest's axis: platforms tilt about it, toward the trough ahead.
        self.tilt_axis = np.array([-self.wave_direction[1], self.wave_direction[0]])

        self._setup_overlays()
        self.limit_hits = 0
        self.ghost_frames = 0

    def build_stewart_platform(self, builder):
        """Add the Stewart platform: one articulation tree plus five loop-closing ball joints."""
        base_angles, platform_angles = joint_angles()
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * BASE_HEIGHT), wp.quat_identity()),
            radius=BASE_RADIUS + 0.07,
            half_height=0.5 * BASE_HEIGHT,
            color=(0.16, 0.17, 0.2),
        )
        self.platform = builder.add_link(label="platform")
        builder.add_shape_cylinder(
            self.platform, radius=PLATFORM_RADIUS, half_height=PLATE_HALF_THICKNESS, color=(0.93, 0.93, 0.95)
        )
        builder.add_shape_box(  # a payload, to show the platform's heading
            self.platform,
            xform=wp.transform(wp.vec3(0.05, 0.0, PLATE_HALF_THICKNESS + 0.03), wp.quat_identity()),
            hx=0.07,
            hy=0.045,
            hz=0.03,
            color=(0.46, 0.73, 0.0),
        )

        tree_joints, self.legs = [], []
        for leg in range(6):
            a, b = base_angles[leg], platform_angles[leg]
            base_anchor = wp.vec3(BASE_RADIUS * math.cos(a), BASE_RADIUS * math.sin(a), BASE_HEIGHT)
            platform_anchor = wp.vec3(PLATFORM_RADIUS * math.cos(b), PLATFORM_RADIUS * math.sin(b), 0.0)
            span = platform_anchor + wp.vec3(0.0, 0.0, BASE_HEIGHT + PLATFORM_HEIGHT) - base_anchor
            length = float(wp.length(span))
            lean = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.normalize(span))

            # The cylinder pivots on the base; the rod slides out of it along the leg (z).
            cylinder = builder.add_link(label=f"cylinder_{leg}")
            builder.add_shape_capsule(
                cylinder,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.3 * length), wp.quat_identity()),
                radius=0.022,
                half_height=0.27 * length,
                color=tuple(LEG_COLOR),
            )
            rod = builder.add_link(label=f"rod_{leg}")
            builder.add_shape_capsule(
                rod,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.72 * length), wp.quat_identity()),
                radius=0.011,
                half_height=0.27 * length,
                color=tuple(ROD_COLOR),
            )
            builder.add_shape_sphere(  # the ball joint's housing
                rod,
                xform=wp.transform(wp.vec3(0.0, 0.0, length), wp.quat_identity()),
                radius=0.02,
                color=(0.5, 0.52, 0.56),
            )
            universal = builder.add_joint_d6(
                parent=-1,
                child=cylinder,
                angular_axes=[
                    newton.ModelBuilder.JointDofConfig(axis=axis, limit_lower=-SWING, limit_upper=SWING)
                    for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
                ],
                parent_xform=wp.transform(base_anchor, lean),
                label=f"universal_{leg}",
            )
            actuator = builder.add_joint_prismatic(
                parent=cylinder,
                child=rod,
                axis=(0.0, 0.0, 1.0),
                limit_lower=-STROKE,
                limit_upper=STROKE,
                label=f"actuator_{leg}",
            )
            tree_joints += [universal, actuator]
            # The ball joint frame on the rod is level at rest, as is the platform.
            ball = {
                "parent": rod,
                "child": self.platform,
                "parent_xform": wp.transform(wp.vec3(0.0, 0.0, length), wp.quat_inverse(lean)),
                "child_xform": wp.transform(platform_anchor, wp.quat_identity()),
                "label": f"ball_{leg}",
            }
            self.legs.append({"cylinder": cylinder, "rod": rod, "actuator": actuator, "ball": ball})

        # The tree reaches the platform through the first leg ...
        tree_joints.append(builder.add_joint_ball(**self.legs[0]["ball"]))
        builder.add_articulation(tree_joints, label="stewart_platform")
        # ... and the ball joints of the other five legs close the loops.
        for leg in self.legs[1:]:
            builder.add_joint_ball(**leg["ball"])

    def _setup_overlays(self):
        shape_body = self.robot_model.shape_body.numpy()
        q_start = self.robot_model.joint_q_start.numpy()

        def first_shape(body):
            return int(np.flatnonzero(shape_body == body)[0])

        self.stroke_coords = wp.array([q_start[leg["actuator"]] for leg in self.legs], dtype=wp.int32)
        self.leg_shapes = wp.array(
            [[first_shape(leg["cylinder"]), first_shape(leg["rod"])] for leg in self.legs], dtype=wp.int32
        )
        loops = self.legs[1:]
        self.loop_rods = wp.array([leg["rod"] for leg in loops], dtype=wp.int32)
        self.loop_rod_anchors = wp.array(
            [wp.transform_get_translation(leg["ball"]["parent_xform"]) for leg in loops], dtype=wp.vec3
        )
        self.loop_platform_anchors = wp.array(
            [wp.transform_get_translation(leg["ball"]["child_xform"]) for leg in loops], dtype=wp.vec3
        )
        shape = (self.world_count, len(loops))
        self.loop_points = wp.zeros(shape, dtype=wp.vec3)
        self.loop_gap = wp.zeros(shape, dtype=wp.float32)
        self.loop_colors = wp.full(self.loop_points.size, wp.vec3(*LOOP_COLOR), dtype=wp.vec3)
        self.ghost_xforms = wp.zeros(self.world_count, dtype=wp.transform)
        self.ghost_opacities = wp.zeros(self.world_count, dtype=wp.float32)
        self.ghost_colors = wp.full(self.world_count, wp.vec3(*GHOST_COLOR), dtype=wp.vec3)
        self.ghost_ring_starts = wp.zeros((self.world_count, GHOST_SEGMENTS), dtype=wp.vec3)
        self.ghost_ring_ends = wp.zeros((self.world_count, GHOST_SEGMENTS), dtype=wp.vec3)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def update_targets(self):
        """Move each platform target with the wave: heave, sway along the wave, tilt toward the
        trough ahead, and yaw."""
        storm = 0.5 - 0.5 * math.cos(2.0 * math.pi * self.sim_time / STORM_PERIOD)

        def amplitude(calm_and_storm):
            calm, peak = calm_and_storm
            return calm + storm * (peak - calm)

        phase = self.wave_phase - 2.0 * math.pi * self.sim_time / WAVE_PERIOD
        positions = np.zeros((self.world_count, 3), dtype=np.float32)
        positions[:, :2] = amplitude(SWAY) * np.cos(phase)[:, None] * self.wave_direction
        positions[:, 2] = BASE_HEIGHT + PLATFORM_HEIGHT + amplitude(HEAVE) * np.sin(phase)
        rotations = tilt_quats(amplitude(YAW) * np.sin(phase + 1.0), self.tilt_axis, amplitude(TILT) * np.cos(phase))
        self.target_positions.assign(positions)
        self.target_rotations.assign(rotations)

    def update_overlays(self):
        bodies, shapes = self.robot_model.body_count, self.robot_model.shape_count
        wp.launch(
            color_legs,
            dim=(self.world_count, len(self.legs)),
            inputs=[self.joint_q, self.stroke_coords, self.leg_shapes, shapes, STROKE],
            outputs=[self.model.shape_color],
        )
        wp.launch(
            mark_loops,
            dim=self.loop_points.shape,
            inputs=[
                self.state.body_q,
                bodies,
                self.loop_rods,
                self.loop_rod_anchors,
                self.platform,
                self.loop_platform_anchors,
                self.world_offsets,
            ],
            outputs=[self.loop_points, self.loop_gap],
        )
        wp.launch(
            place_ghosts,
            dim=self.ghost_ring_starts.shape,
            inputs=[
                self.state.body_q,
                bodies,
                self.platform,
                self.target_positions,
                self.target_rotations,
                self.world_offsets,
            ],
            outputs=[self.ghost_xforms, self.ghost_opacities, self.ghost_ring_starts, self.ghost_ring_ends],
        )

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        self.update_targets()
        self.solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.update_overlays()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        gap = float(self.loop_gap.numpy().max())
        if gap > 1e-3:
            raise ValueError(f"A loop-closing joint opened by {gap * 1e3:.2f} mm")
        stroke = self.joint_q.numpy()[:, self.stroke_coords.numpy()]
        if np.any(np.abs(stroke) > STROKE + 1e-5):
            raise ValueError(f"An actuator left its stroke: [{stroke.min():.4f}, {stroke.max():.4f}] m")
        self.limit_hits += int(np.sum(np.abs(stroke) > STROKE - 1e-4))
        self.ghost_frames += int(np.any(self.ghost_opacities.numpy() > 0.0))

    def test_final(self):
        if self.limit_hits == 0 or self.ghost_frames == 0:
            raise ValueError("No actuator reached its stroke limit: the waves should outgrow the legs")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_points("loop_joints", self.loop_points.flatten(), radii=0.026, colors=self.loop_colors)
        self.viewer.log_shapes(
            "targets",
            newton.GeoType.CYLINDER,
            (GHOST_RADIUS, PLATE_HALF_THICKNESS),
            self.ghost_xforms,
            colors=self.ghost_colors,
            opacities=self.ghost_opacities,
        )
        self.viewer.log_lines(
            "target_rims", self.ghost_ring_starts.flatten(), self.ghost_ring_ends.flatten(), GHOST_COLOR
        )
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=100)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
