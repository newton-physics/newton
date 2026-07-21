# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Conveyor Force
#
# Static annular conveyor with a pivot velocity field. Per-contact normal
# forces determine Coulomb-limited tangential forces applied to the boxes.
# A central hub and segmented outer wall bound the conveyor path.
#
# Command: uv run -m newton.examples conveyor_force
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic.conveyor_force_model import ConveyorForceModel

BELT_TOP_Z = 0.5
BELT_HALF_Z = 0.1
HUB_RADIUS = 1.6  # central wall: boxes cannot go inside this
WALL_RADIUS = 3.2  # outer wall: boxes cannot go past this
BELT_RADIUS = 3.5  # belt disk covers the whole ring band
RING_RADIUS = 0.5 * (HUB_RADIUS + WALL_RADIUS)  # box centerline
BELT_SPEED = 2.0  # surface speed at the ring centerline [m/s]
BOX_HALF = 0.2
WALL_SEGMENTS = 24
WALL_HALF_HEIGHT = 0.35
NUM_BOXES = 10
SPEED_RAMP_DURATION = 0.6  # [s]
CONTACT_FRICTION = 2.0e-5


def _look_at(eye, target):
    """Return (pos, pitch_deg, yaw_deg) for a Z-up camera at ``eye`` looking at ``target``."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    d /= np.linalg.norm(d)
    pitch = np.degrees(np.arcsin(d[2]))
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    return wp.vec3(*eye), float(pitch), float(yaw)


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        belt_speed = float(getattr(args, "belt_speed", BELT_SPEED)) if args is not None else BELT_SPEED
        self.solver_type = getattr(args, "solver", "xpbd") if args is not None else "xpbd"

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        wall_cfg = newton.ModelBuilder.ShapeConfig(mu=CONTACT_FRICTION, ke=1.0e5, kd=0.0)

        # Static cylinder representing the annular belt surface.
        belt_cfg = newton.ModelBuilder.ShapeConfig(mu=CONTACT_FRICTION, ke=1.0e5, kd=0.0)
        self.belt_shape = builder.add_shape_cylinder(
            body=-1,
            radius=BELT_RADIUS,
            half_height=BELT_HALF_Z,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, BELT_TOP_Z - BELT_HALF_Z), q=wp.quat_identity()),
            cfg=belt_cfg,
            color=(0.09, 0.09, 0.09),
            label="conveyor_belt",
        )

        # Inner cylindrical boundary.
        builder.add_shape_cylinder(
            body=-1,
            radius=HUB_RADIUS,
            half_height=WALL_HALF_HEIGHT,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, BELT_TOP_Z + WALL_HALF_HEIGHT), q=wp.quat_identity()),
            cfg=wall_cfg,
            color=(0.55, 0.57, 0.6),
            label="hub",
        )

        # Outer polygonal boundary.
        wall_center_r = WALL_RADIUS + 0.1
        seg_tangential = math.pi * wall_center_r / WALL_SEGMENTS * 1.15
        for k in range(WALL_SEGMENTS):
            ang = 2.0 * math.pi * k / WALL_SEGMENTS
            builder.add_shape_box(
                body=-1,
                hx=0.1,
                hy=seg_tangential,
                hz=WALL_HALF_HEIGHT,
                xform=wp.transform(
                    p=wp.vec3(
                        wall_center_r * math.cos(ang), wall_center_r * math.sin(ang), BELT_TOP_Z + WALL_HALF_HEIGHT
                    ),
                    q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), ang),
                ),
                cfg=wall_cfg,
                color=(0.55, 0.57, 0.6),
                label=f"wall_{k}",
            )

        # Dynamic boxes distributed around the ring centerline.
        self.box_bodies = []
        box_colors = [(0.85, 0.3, 0.2), (0.2, 0.7, 0.3), (0.9, 0.75, 0.2)]
        spawn_z = BELT_TOP_Z + BOX_HALF + 0.01
        for i in range(NUM_BOXES):
            ang = 2.0 * math.pi * i / NUM_BOXES
            box = builder.add_link(
                xform=wp.transform(
                    p=wp.vec3(RING_RADIUS * math.cos(ang), RING_RADIUS * math.sin(ang), spawn_z),
                    q=wp.quat_identity(),
                ),
                mass=1.0,
                label=f"box_{i}",
            )
            builder.add_shape_box(
                box,
                hx=BOX_HALF,
                hy=BOX_HALF,
                hz=BOX_HALF,
                cfg=newton.ModelBuilder.ShapeConfig(mu=CONTACT_FRICTION, restitution=0.0),
                color=box_colors[i % len(box_colors)],
            )
            builder.add_articulation([builder.add_joint_free(box)], label=f"box_{i}")
            self.box_bodies.append(box)

        builder.color()
        self.model = builder.finalize()

        # Allocate the reported per-contact force attribute.
        self.model.request_contact_attributes("force")

        if self.solver_type == "mujoco":
            # MuJoCo configuration for Newton-generated contacts.
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                cone="elliptic",
                use_mujoco_contacts=False,
                njmax=1500,
                nconmax=750,
            )
        elif self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=5, rigid_body_contact_buffer_size=1024)
        else:
            self.solver = newton.solvers.SolverXPBD(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit")
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Pivot velocity field centered on the ring's +Z axis.
        self.conveyor = ConveyorForceModel(self.model, solver_type=self.solver_type)
        omega = belt_speed / RING_RADIUS
        self.conveyor.add_pivot_belt(
            self.belt_shape,
            pivot_point=wp.vec3(0.0, 0.0, BELT_TOP_Z),
            angular_velocity=wp.vec3(0.0, 0.0, omega),
            surface_normal=wp.vec3(0.0, 0.0, 1.0),
            friction=0.6,
        )
        self.conveyor.finalize(self.contacts)

        self.viewer.set_model(self.model)
        pos, pitch, yaw = _look_at(eye=(6.5, -6.5, 6.0), target=(0.0, 0.0, BELT_TOP_Z))
        self.viewer.set_camera(pos, pitch, yaw)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.conveyor.apply(self.state_0)

            self.conveyor.snapshot_prev(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.conveyor.update(self.solver, self.contacts, self.state_1, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Linear speed ramp over the initial interval.
        self.conveyor.set_speed_scale(min(1.0, self.sim_time / SPEED_RAMP_DURATION))
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        for box in self.box_bodies:
            p = body_q[box][:3]
            assert np.all(np.isfinite(p)), f"box {box} pose is non-finite"
            r = float(math.hypot(p[0], p[1]))
            # Box stays within the ring band and on the belt surface.
            assert HUB_RADIUS - 0.3 < r < WALL_RADIUS + 0.3, f"box {box} left the ring band: r={r:.3f}"
            assert abs(float(p[2]) - (BELT_TOP_Z + BOX_HALF)) < 0.15, f"box {box} left the belt surface: z={p[2]:.4f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--belt-speed", type=float, default=BELT_SPEED, help="Ring surface speed at the centerline [m/s].")
    parser.add_argument(
        "--solver", type=str, choices=["xpbd", "vbd", "mujoco"], default="xpbd", help="Solver backend to use."
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
