# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Conveyor Forces
#
# A multi-belt conveyor circuit transports rigid boxes with a velocity field
# rather than belt friction: every belt is a static, frictionless surface with
# an attached constant or pivot velocity field. Each step the
# per-contact normal forces are read back from the solver and converted into
# Coulomb-limited tangential body forces, so boxes are carried around the
# loop, through the 180-degree turn, up the incline, across the differential
# pair, and back. Runs on SolverXPBD, SolverVBD, and SolverMuJoCo.
#
# Straight belts cannot be driven kinematically, which is what motivates the
# force-based drive here. See example_basic_conveyor.py for the other
# approach: a single rotating belt moved by a prescribed joint, carrying its
# load through ordinary contact friction.
#
# The drive itself lives in three companion modules: conveyor_forces_actuators
# turns a belt's velocity field into a per-contact force, conveyor_forces_kernels
# groups contacts into patches and splits the normal force between them, and
# conveyor_forces_adapter reformats Newton's contacts and body state into the
# packed buffers those kernels read.
#
# Command: uv run -m newton.examples basic_conveyor_forces
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic import conveyor_forces_kernels
from newton.examples.basic.conveyor_forces_actuators import (
    VELOCITY_FIELD_TYPE_CONSTANT_VELOCITY,
    VELOCITY_FIELD_TYPE_PIVOT,
    VelocityFieldActuator,
)
from newton.examples.basic.conveyor_forces_adapter import NewtonConveyorAdapter

# A small positive collision margin smooths the belt-to-belt seam transitions. VBD's
# rigid-contact handling needs a larger margin than XPBD to keep bodies on the belts.
SOLVER_MARGIN = {"xpbd": 0.015, "vbd": 0.05, "mujoco": 0.015}
XPBD_ITERATIONS = 4
VBD_ITERATIONS = 15
# A frictional-to-normal impedance ratio below 1 softens the friction
# constraints so boxes slip through the tight 180 degree turn instead of
# wedging against the guide walls.
MUJOCO_IMPRATIO = 0.1
# Just above mujoco.mjMINMU, below which mujoco_warp warns about NaN-prone contacts.
MUJOCO_MIN_FRICTION = 1.1e-5

CONTACT_PROCESSING_BATCH_SIZE = 5
MAX_THREAD_COUNT = 4096

# Slowest observed fleet-average transport speed is ~1.8 m/s (XPBD) and every box travels
# at least 1.3 m within a 100-frame test; undriven boxes sit at ~0.
MIN_TRANSPORT_SPEED = 0.3  # [m/s]
MIN_TRAVEL = 0.5  # [m]

STARTUP_DURATION = 1.0  # belts ramp from rest to full speed over this time [s]
BELT_HALF_THICKNESS = 0.1  # [m]
GUARD_HALF_THICKNESS = 0.2  # [m]
GUARD_WALL_THICKNESS = 0.1  # radial thickness of the curved turn guards [m]
TURN_SEGMENTS = 48

# A contact only drives a body if its normal is within ~4.4 degrees of the belt's
# surface normal, so a box pressed against a guard wall is not carried sideways.
CONTACT_PROCESSING_THRESHOLD = 0.997

# Coulomb friction between a box material (row) and a belt material (column). The
# tangential drive force on a box is limited to this times the contact normal force.
FRICTION_TABLE = ((0.5,), (0.9,))

BELT_COLOR = (0.09, 0.09, 0.09)  # dark rubber
GUARD_COLOR = (0.66, 0.69, 0.74)  # brushed metal
BOX_COLOR = (0.72, 0.55, 0.35)  # cardboard

# Straight belts, each a static slab driven along its own +Y axis.
# (label, center [m], yaw [deg], tilt about the belt's X axis [deg], half extents [m], surface speed [m/s])
STRAIGHT_BELTS = (
    ("infeed", (0.0, 0.0, 0.5), 0.0, 0.0, (0.5, 5.05, BELT_HALF_THICKNESS), -2.0),
    ("incline", (-5.1, -3.0075, 0.6757), 0.0, 5.0, (0.5, 2.05, BELT_HALF_THICKNESS), 2.0),
    ("differential_fast", (-4.84, 4.0849, 0.8514), 0.0, 0.0, (0.25, 5.05, BELT_HALF_THICKNESS), 4.0),
    ("differential_slow", (-5.36, 4.0849, 0.8514), 0.0, 0.0, (0.25, 5.05, BELT_HALF_THICKNESS), 2.0),
    ("crossover", (-3.08, 10.1349, 0.8514), 90.0, 0.0, (1.0, 2.58, BELT_HALF_THICKNESS), -2.0),
    ("decline", (0.0, 8.0925, 0.6757), 0.0, 3.305, (0.5, 3.0475, BELT_HALF_THICKNESS), -2.0),
)

# The 180 degree turn, built from two annular sectors spinning about their own pivot.
# (label, pivot [m], start angle [deg], end angle [deg], (inner, outer) radius [m], angular speed [rad/s])
TURN_BELTS = (
    ("turn_outer", (-2.5, -5.1, 0.5), 270.0, 360.0, (2.0, 3.0), -0.8),
    ("turn_inner", (-2.6, -5.1, 0.5), 180.0, 270.0, (2.0, 3.0), -0.8),
)

# Static guard walls that keep boxes on the belts through the turns and the incline.
# (label, center [m], yaw [deg], tilt [deg], half extents [m])
GUARDS = (
    ("incline_guard", (-5.6, -3.0075, 0.6757), -15.0, 5.0, (0.1, 0.5, GUARD_HALF_THICKNESS)),
    ("crossover_guard", (-3.08, 11.2349, 0.8514), 0.0, 0.0, (2.53, 0.1, GUARD_HALF_THICKNESS)),
    ("decline_guard", (0.5, 9.3094, 0.746), -5.0, 3.305, (0.1, 1.5, GUARD_HALF_THICKNESS)),
)

# Transported boxes: (label, center [m], half extents [m], mass [kg], material index)
BOXES = (
    ("box_0", (0.0, 0.0, 0.804), (0.2, 0.3, 0.2), 2.0, 0),
    ("box_1", (-0.2, -1.0, 0.704), (0.1, 0.1, 0.1), 0.5, 0),
    ("box_2", (0.1, -2.0, 0.804), (0.2, 0.3, 0.2), 2.0, 0),
    ("box_3", (0.0, 4.0, 0.704), (0.25, 0.25, 0.1), 1.0, 1),
    ("box_4", (-0.75, -6.85, 0.704), (0.1, 0.1, 0.1), 0.5, 0),
    ("box_5", (-4.35, -6.85, 0.804), (0.2, 0.3, 0.2), 2.0, 0),
    ("box_6", (-5.1, 4.0849, 1.1554), (0.2, 0.3, 0.2), 2.0, 0),
    ("box_7", (-4.84, 4.5349, 1.0554), (0.1, 0.1, 0.1), 0.5, 0),
    ("box_8", (-5.36, 3.6349, 1.0554), (0.1, 0.1, 0.1), 0.5, 0),
    ("box_9", (-5.025, 1.3849, 1.1554), (0.2, 0.3, 0.2), 2.0, 0),
    ("box_10", (-5.1, 7.5849, 1.0554), (0.25, 0.25, 0.1), 1.0, 1),
)

# A two-link parcel hinged about Z, straddling the differential pair: the speed
# difference between the two belts folds it as it travels.
HINGED_PARCEL = (
    ("parcel_a", (-4.975, -0.7151, 1.0054), (0.125, 0.05, 0.05), 1.0, 0),
    ("parcel_b", (-5.225, -0.7151, 1.0054), (0.125, 0.05, 0.05), 1.0, 0),
)
HINGE_LIMIT = math.radians(45.0)
HINGE_LIMIT_KE = 100.0  # [N·m/rad]
HINGE_LIMIT_KD = 1.0  # [N·m·s/rad]


def belt_rotation(yaw_deg: float, tilt_deg: float) -> wp.quat:
    """Return the belt frame: yaw about world Z, then tilt about the belt's own X axis."""
    yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(yaw_deg))
    tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    return yaw * tilt


def create_annular_sector_mesh(
    inner_radius: float,
    outer_radius: float,
    half_thickness: float,
    start_angle: float,
    end_angle: float,
    segments: int,
) -> newton.Mesh:
    """Create a closed annular sector prism about the Z axis, angles in degrees."""
    angles = np.radians(np.linspace(start_angle, end_angle, segments + 1, dtype=np.float32))
    cos_theta, sin_theta = np.cos(angles), np.sin(angles)
    n = segments + 1

    def ring(radius, z):
        return np.stack((radius * cos_theta, radius * sin_theta, np.full(n, z, dtype=np.float32)), axis=1)

    vertices = np.vstack(
        (
            ring(inner_radius, half_thickness),
            ring(outer_radius, half_thickness),
            ring(inner_radius, -half_thickness),
            ring(outer_radius, -half_thickness),
        )
    ).astype(np.float32)

    top_in, top_out, bot_in, bot_out = 0, n, 2 * n, 3 * n
    indices: list[int] = []
    for i in range(segments):
        j = i + 1
        # top (+Z), bottom (-Z), outer and inner walls
        indices.extend((top_in + i, top_out + i, top_out + j, top_in + i, top_out + j, top_in + j))
        indices.extend((bot_in + i, bot_in + j, bot_out + j, bot_in + i, bot_out + j, bot_out + i))
        indices.extend((bot_out + i, bot_out + j, top_out + j, bot_out + i, top_out + j, top_out + i))
        indices.extend((bot_in + i, top_in + i, top_in + j, bot_in + i, top_in + j, bot_in + j))
    # end caps
    e = segments
    indices.extend((top_in, bot_in, bot_out, top_in, bot_out, top_out))
    indices.extend((top_in + e, top_out + e, bot_out + e, top_in + e, bot_out + e, bot_in + e))

    return newton.Mesh(vertices=vertices, indices=np.asarray(indices, dtype=np.int32), compute_inertia=False)


class Example:
    def __init__(self, viewer, args=None):
        self.solver_type = getattr(args, "solver", "xpbd") if args is not None else "xpbd"

        # VBD needs a smaller step for stable explicit conveyor-force feedback.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4 if self.solver_type == "vbd" else 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.startup_duration = STARTUP_DURATION

        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        self.belts, self.bodies = self._build_scene(builder)

        if self.solver_type == "mujoco":
            # Two MuJoCo-specific adjustments, both needed for the belts to carry anything:
            #
            # 1. A contact's friction is combined from the two shapes' materials, and MuJoCo
            #    takes their maximum where Newton's own solvers average them. Frictionless
            #    belts are therefore enough for XPBD and VBD (they see mu/2), but under MuJoCo
            #    each box keeps its own coefficient and the resulting friction constraint
            #    cancels the tangential drive: over 100 frames the boxes crawl at 0.27 m/s
            #    with mu = 0.5 against 2.24 m/s with it dropped. So the boxes are made
            #    frictionless as well. The drive stays Coulomb-limited either way, since the
            #    pipeline clamps it with FRICTION_TABLE and not with the shape materials.
            # 2. mujoco_warp warns that friction below ``mujoco.mjMINMU`` may produce NaN for
            #    condim=3 contacts, so use that floor rather than exactly zero.
            transported = {b.body for b in self.bodies}
            for shape, body in enumerate(builder.shape_body):
                builder.shape_material_mu[shape] = max(builder.shape_material_mu[shape], MUJOCO_MIN_FRICTION)
                if body in transported:
                    builder.shape_material_mu[shape] = MUJOCO_MIN_FRICTION
                    builder.shape_material_mu_torsional[shape] = 0.0
                    builder.shape_material_mu_rolling[shape] = 0.0

        margin = SOLVER_MARGIN[self.solver_type]
        mesh_types = (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH, newton.GeoType.HFIELD)
        shape_type = builder.shape_type
        for i in range(len(builder.shape_margin)):
            if shape_type[i] not in mesh_types:
                builder.shape_margin[i] = max(builder.shape_margin[i], margin)

        builder.color()
        self.model = builder.finalize()

        # The force pipeline consumes the per-contact normal force reported by the solver.
        self.model.request_contact_attributes("force")

        if self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                cone="elliptic",
                impratio=MUJOCO_IMPRATIO,
                use_mujoco_contacts=False,
                njmax=2000,
                nconmax=1000,
            )
        elif self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model, iterations=VBD_ITERATIONS, rigid_body_contact_buffer_size=2048
            )
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=XPBD_ITERATIONS)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit")
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # --- Force pipeline: adapter + velocity-field actuator. ---
        self.adapter = NewtonConveyorAdapter(self.model, self.belts, self.bodies, FRICTION_TABLE, self.contacts)

        self.velocity_field_actuator = VelocityFieldActuator()
        for velocity in self.constant_velocity_fields:
            self.velocity_field_actuator.add_constant_velocity_field(velocity)
        for pivot, angular_velocity in self.pivot_velocity_fields:
            self.velocity_field_actuator.add_pivot_velocity_field(pivot, angular_velocity)
        self.velocity_field_actuator.create_buffers(self.model.device)

        self._allocate_pipeline_buffers()

        # Newton body indices of the transported bodies, for the test hooks below.
        self.tracked_bodies = self.adapter.pipeline_to_newton.numpy().tolist()
        self.tracked_start_pos = self.state_0.body_q.numpy()[self.tracked_bodies, :3].copy()
        self.max_travel = np.zeros(len(self.tracked_bodies))

        self.viewer.set_model(self.model)
        pos, pitch, yaw = _look_at(eye=(2.0, -14.0, 12.0), target=(-2.0, -4.0, 0.5))
        self.viewer.set_camera(pos, pitch, yaw)

    def _build_scene(self, builder):
        """Build the conveyor circuit and return its belt and transported-body specs.

        Also fills ``constant_velocity_fields`` / ``pivot_velocity_fields``, which the belts
        reference by index and the actuator registers in the same order.
        """
        belt_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0)  # the drive comes from the force pipeline
        guard_cfg = newton.ModelBuilder.ShapeConfig(mu=0.2)

        self.constant_velocity_fields = []
        self.pivot_velocity_fields = []
        belts = []

        for label, pos, yaw, tilt, half_extents, speed in STRAIGHT_BELTS:
            rot = belt_rotation(yaw, tilt)
            shape = builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=wp.vec3(*pos), q=rot),
                hx=half_extents[0],
                hy=half_extents[1],
                hz=half_extents[2],
                cfg=belt_cfg,
                color=BELT_COLOR,
                label=label,
            )
            # The belt carries its surface along its own +Y axis; its normal is its own +Z.
            self.constant_velocity_fields.append(wp.quat_rotate(rot, wp.vec3(0.0, speed, 0.0)))
            belts.append(
                NewtonConveyorAdapter.Belt(
                    shape=shape,
                    velocity_field_type=VELOCITY_FIELD_TYPE_CONSTANT_VELOCITY,
                    velocity_field_id=len(self.constant_velocity_fields) - 1,
                    material_index=0,
                    surface_normal=wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0)),
                    contact_processing_threshold=CONTACT_PROCESSING_THRESHOLD,
                )
            )

        for label, pivot, start_angle, end_angle, radii, angular_speed in TURN_BELTS:
            inner_radius, outer_radius = radii
            xform = wp.transform(p=wp.vec3(*pivot), q=wp.quat_identity())
            shape = builder.add_shape_mesh(
                body=-1,
                xform=xform,
                mesh=create_annular_sector_mesh(
                    inner_radius, outer_radius, BELT_HALF_THICKNESS, start_angle, end_angle, TURN_SEGMENTS
                ),
                cfg=belt_cfg,
                color=BELT_COLOR,
                label=label,
            )
            builder.add_shape_mesh(
                body=-1,
                xform=xform,
                mesh=create_annular_sector_mesh(
                    outer_radius,
                    outer_radius + GUARD_WALL_THICKNESS,
                    GUARD_HALF_THICKNESS,
                    start_angle,
                    end_angle,
                    TURN_SEGMENTS,
                ),
                cfg=guard_cfg,
                color=GUARD_COLOR,
                label=f"{label}_guard",
            )
            self.pivot_velocity_fields.append((wp.vec3(*pivot), wp.vec3(0.0, 0.0, angular_speed)))
            belts.append(
                NewtonConveyorAdapter.Belt(
                    shape=shape,
                    velocity_field_type=VELOCITY_FIELD_TYPE_PIVOT,
                    velocity_field_id=len(self.pivot_velocity_fields) - 1,
                    material_index=0,
                    surface_normal=wp.vec3(0.0, 0.0, 1.0),
                    contact_processing_threshold=CONTACT_PROCESSING_THRESHOLD,
                )
            )

        for label, pos, yaw, tilt, half_extents in GUARDS:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=wp.vec3(*pos), q=belt_rotation(yaw, tilt)),
                hx=half_extents[0],
                hy=half_extents[1],
                hz=half_extents[2],
                cfg=guard_cfg,
                color=GUARD_COLOR,
                label=label,
            )

        bodies = []
        for label, pos, half_extents, mass, material_index in BOXES:
            body = self._add_box_body(builder, label, pos, half_extents, mass, material_index)
            builder.add_articulation([builder.add_joint_free(body)], label=label)
            bodies.append(NewtonConveyorAdapter.Body(body=body, material_index=material_index))

        (label_a, pos_a, half_a, mass_a, mat_a), (label_b, pos_b, half_b, mass_b, mat_b) = HINGED_PARCEL
        link_a = self._add_box_body(builder, label_a, pos_a, half_a, mass_a, mat_a)
        link_b = self._add_box_body(builder, label_b, pos_b, half_b, mass_b, mat_b)
        free = builder.add_joint_free(link_a)
        hinge = builder.add_joint_revolute(
            parent=link_a,
            child=link_b,
            parent_xform=wp.transform(p=wp.vec3(-half_a[0], 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(half_b[0], 0.0, 0.0), q=wp.quat_identity()),
            axis=newton.Axis.Z,
            limit_lower=-HINGE_LIMIT,
            limit_upper=HINGE_LIMIT,
            limit_ke=HINGE_LIMIT_KE,
            limit_kd=HINGE_LIMIT_KD,
            label="parcel_hinge",
        )
        builder.add_articulation([free, hinge], label="hinged_parcel")
        bodies.append(NewtonConveyorAdapter.Body(body=link_a, material_index=mat_a))
        bodies.append(NewtonConveyorAdapter.Body(body=link_b, material_index=mat_b))

        return belts, bodies

    @staticmethod
    def _add_box_body(builder, label, pos, half_extents, mass, material_index):
        """Add a transported box, sizing its density so the shape carries the authored mass."""
        hx, hy, hz = half_extents
        body = builder.add_link(xform=wp.transform(p=wp.vec3(*pos), q=wp.quat_identity()), label=label)
        cfg = newton.ModelBuilder.ShapeConfig(
            mu=FRICTION_TABLE[material_index][0],
            density=mass / (8.0 * hx * hy * hz),
        )
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg, color=BOX_COLOR, label=label)
        return body

    def _allocate_pipeline_buffers(self):
        d = self.model.device
        N = self.adapter.body_count
        C = self.adapter.max_contact_count

        self.body_to_world_transform_buffer = wp.empty(N, dtype=wp.transform, device=d)
        self.body_inverse_inertia_buffer = wp.empty(N, dtype=wp.mat33, device=d)
        self.body_to_patch_buffer = wp.empty(N, dtype=wp.uint32, device=d)
        self.body_force_buffer = wp.zeros((N, 3), dtype=wp.float32, device=d)
        self.body_torque_buffer = wp.zeros((N, 3), dtype=wp.float32, device=d)

        self.point_to_indices_map = wp.empty((C, 3), dtype=wp.uint32, device=d)
        self.friction_coefficient_buffer = wp.empty(C, dtype=wp.float32, device=d)
        self.contact_patch_buffer = wp.empty(C, dtype=conveyor_forces_kernels.Patch, device=d)
        self.mass_splitting_scale_buffer = wp.empty(C, dtype=wp.float32, device=d)
        self.adjusted_contact_normal_force_buffer = wp.empty((C, 1), dtype=wp.float32, device=d)
        self.per_point_force_torque_buffer = wp.empty(C, dtype=wp.spatial_vector, device=d)

        self.total_contact_count = wp.zeros(1, dtype=wp.uint32, device=d)
        self.total_elapsed_time = wp.zeros(1, dtype=wp.float32, device=d)
        self.global_conveyor_belt_speed_scale = wp.zeros(1, dtype=wp.float32, device=d)

        self.body_q_prev = wp.zeros(N, dtype=wp.transform, device=d)

    def _step_conveyor_pipeline(self, state):
        """Run the conveyor force pipeline on Newton-derived buffers."""
        adapter = self.adapter
        N = adapter.body_count
        M = adapter.belt_count
        C = adapter.max_contact_count
        d = self.model.device

        adapter.gather_state(state)

        # Reset the accumulated contact count for this step.
        self.total_contact_count.zero_()

        parallel_conveyor_belt_processing_count = 16
        max_body_thread_count = MAX_THREAD_COUNT // parallel_conveyor_belt_processing_count
        parallel_body_processing_count = min(max_body_thread_count, N)

        wp.launch(
            kernel=conveyor_forces_kernels.prepare_buffers,
            dim=(parallel_body_processing_count, parallel_conveyor_belt_processing_count),
            inputs=[
                parallel_body_processing_count,
                parallel_conveyor_belt_processing_count,
                N,
                M,
                self.sim_dt,
                self.startup_duration,
                adapter._body_positions_ia,
                adapter.body_orientations,
                adapter._body_com_positions_ia,
                adapter.body_com_orientations,
                adapter._body_inverse_inertias_ia,
                adapter.body_material_index,
                adapter.conveyor_belt_to_indices_map,
                adapter.friction_table,
                adapter.pair_contacts_count,
                adapter.pair_contacts_start_indices,
            ],
            outputs=[
                self.body_to_world_transform_buffer,
                self.body_inverse_inertia_buffer,
                self.point_to_indices_map,
                self.friction_coefficient_buffer,
                self.total_contact_count,
                self.total_elapsed_time,
                self.global_conveyor_belt_speed_scale,
            ],
            device=d,
        )

        parallel_body_processing_count = min(MAX_THREAD_COUNT, N)
        wp.launch(
            kernel=conveyor_forces_kernels.correlate_and_filter_contact_points,
            dim=parallel_body_processing_count,
            inputs=[
                parallel_body_processing_count,
                N,
                M,
                adapter.surface_normal_buffer,
                adapter.contact_processing_threshold_buffer,
                adapter.pair_contacts_count,
                adapter.pair_contacts_start_indices,
                adapter.flat_normal,
                adapter.flat_force,
            ],
            outputs=[
                self.contact_patch_buffer,
                self.body_to_patch_buffer,
                self.mass_splitting_scale_buffer,
            ],
            device=d,
        )

        parallel_patch_processing_count = 1
        max_body_thread_count = MAX_THREAD_COUNT // parallel_patch_processing_count
        parallel_body_processing_count = min(max_body_thread_count, N)
        wp.launch(
            kernel=conveyor_forces_kernels.redistribute_contact_force,
            dim=(parallel_body_processing_count, parallel_patch_processing_count),
            inputs=[
                parallel_body_processing_count,
                parallel_patch_processing_count,
                N,
                self.body_to_patch_buffer,
                self.contact_patch_buffer,
                adapter.flat_point,
                adapter.flat_force,
                self.body_to_world_transform_buffer,
            ],
            outputs=[
                self.adjusted_contact_normal_force_buffer,
                self.mass_splitting_scale_buffer,
            ],
            device=d,
        )

        self.velocity_field_actuator.step(
            self.sim_dt,
            C,
            self.body_to_world_transform_buffer,
            adapter._body_inverse_masses_ia,
            self.body_inverse_inertia_buffer,
            adapter._body_linear_velocities_ia,
            adapter._body_angular_velocities_ia,
            adapter.flat_point,
            adapter.flat_normal,
            self.adjusted_contact_normal_force_buffer,
            self.point_to_indices_map,
            self.mass_splitting_scale_buffer,
            self.friction_coefficient_buffer,
            self.total_contact_count,
            self.global_conveyor_belt_speed_scale,
            self.per_point_force_torque_buffer,
            max_thread_count=MAX_THREAD_COUNT,
            batch_size=CONTACT_PROCESSING_BATCH_SIZE,
            device=d,
        )

        parallel_body_processing_count = min(MAX_THREAD_COUNT, N)
        wp.launch(
            kernel=conveyor_forces_kernels.sum_up_force,
            dim=parallel_body_processing_count,
            inputs=[
                parallel_body_processing_count,
                N,
                self.body_to_patch_buffer,
                self.contact_patch_buffer,
                self.per_point_force_torque_buffer,
            ],
            outputs=[
                self.body_force_buffer,
                self.body_torque_buffer,
            ],
            device=d,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Apply the wrench computed from the previous step's contacts.
            self.adapter.apply_forces(self.state_0, self.body_force_buffer, self.body_torque_buffer)

            if self.solver_type == "vbd":
                # Force reconstruction requires the same pose history used by VBD.
                wp.copy(self.body_q_prev, self.solver.body_q_prev)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.adapter.report_contact_forces(
                self.solver, self.contacts, self.state_1, self.body_q_prev, self.sim_dt, self.solver_type
            )
            # Reformat Newton's contacts into the per-(body, belt) dense layout the pipeline
            # expects (classify -> prefix-sum -> scatter). Must run after report_contact_forces
            # so the per-contact normal force is available to the classifier.
            self.adapter.build_contact_layout(self.contacts, self.state_1)
            self._step_conveyor_pipeline(self.state_1)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        # Track how far each box has ever been from its spawn, so test_final can tell a
        # box that is genuinely carried from one that never left the belt it started on.
        travel = np.linalg.norm(self.state_0.body_q.numpy()[self.tracked_bodies, :3] - self.tracked_start_pos, axis=1)
        np.maximum(self.max_travel, travel, out=self.max_travel)

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(body_q)), "non-finite body pose"
        for b in self.tracked_bodies:
            z = float(body_q[b][2])
            assert z > -0.5, f"transported body {b} fell through the floor: z={z:.4f}"

        # Every box must actually be carried: a broken force pipeline leaves them
        # resting on the belts, which the pose checks above would happily accept.
        if self.sim_time > self.startup_duration + 0.5:
            stalled = int(np.argmin(self.max_travel))
            assert self.max_travel[stalled] > MIN_TRAVEL, (
                f"transported body {self.tracked_bodies[stalled]} was never carried off its start pose: "
                f"{self.max_travel[stalled]:.3f} m"
            )
            # Individual boxes briefly slow at belt seams and in the turn, so this is a
            # fleet average rather than a per-box floor.
            speed = np.linalg.norm(self.state_0.body_qd.numpy()[self.tracked_bodies, :3], axis=1)
            assert speed.mean() > MIN_TRANSPORT_SPEED, f"boxes are not moving with the belts: {speed.mean():.3f} m/s"


def _look_at(eye, target):
    """Return (pos, pitch_deg, yaw_deg) for a Z-up camera at ``eye`` looking at ``target``."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    d /= np.linalg.norm(d)
    pitch = np.degrees(np.arcsin(d[2]))
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    return wp.vec3(*eye), float(pitch), float(yaw)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver", type=str, choices=["xpbd", "vbd", "mujoco"], default="xpbd", help="Solver backend to use."
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
