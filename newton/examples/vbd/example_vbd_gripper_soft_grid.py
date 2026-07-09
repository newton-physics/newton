# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD Gripper — Soft 1x1 Grid (water-tight EDGE contact)
#
# A parallel-jaw gripper with CUBE finger pads clamps a horizontal soft 1x1
# grid (a single quad split by one diagonal edge -> two triangles) at its
# centre, then lifts and holds it in the air.
#
# The grid's four corner vertices sit far out in x (outside the jaws), so the
# only mesh feature crossing the pinch centre is the interior DIAGONAL edge.
# The legacy per-particle path finds no vertex between the jaws, so the grid is
# gripped (and lifted) only with ``enable_rigid_soft_full_surface_contact=True``
# via the water-tight soft-EDGE pass. With the flag off the grid slips out and
# falls.
#
# Command: python -m newton.examples vbd_gripper_soft_grid
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


def _box_mesh(h: float) -> newton.Mesh:
    """A triangulated cube of half-extent ``h`` (outward-facing windings) as a Newton mesh."""
    v = np.array(
        [[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h], [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]],
        dtype=np.float32,
    )
    # 12 triangles (2 per face), outward-facing: -z, +z, -y, +y, +x, -x
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 6, 2),
        (3, 7, 6),
        (1, 2, 6),
        (1, 6, 5),
        (0, 4, 7),
        (0, 7, 3),
    ]
    return newton.Mesh(v, np.array(faces, dtype=np.int32).reshape(-1))


PARAMS = {
    # simulation
    "fps": 60,
    "sim_substeps": 20,
    "solver_iterations": 5,
    "gravity": -9.8,
    "num_frames": 360,
    # water-tight rigid-soft contact. True = grip the grid by its diagonal edge (this
    # feature); False = legacy per-particle contact only (== main behavior), grid slips.
    "enable_water_tight": True,
    # cube finger pads (equal half-extents -> a cube)
    "finger_half": 0.03,
    # finger shape: False = analytic box SDF; True = a triangulated cube MESH, which
    # exercises the texture (volume) SDF path instead of the closed-form box SDF.
    "finger_mesh": False,
    "finger_density": 1000.0,
    "finger_color_left": (0.8, 0.3, 0.3),
    "finger_color_right": (0.3, 0.3, 0.8),
    # gripper gap (half-gap per finger). Closed for a gentle pinch (~2 mm compression).
    "open_half_gap": 0.075,
    "closed_half_gap": 0.038,
    # grip / lift trajectory
    "grab_z": 0.32,
    "finger_z_offset": 0.09,
    "lift_height": 0.6,
    "close_duration": 0.6,
    "pinch_duration": 0.4,
    "lift_duration": 1.6,
    "hold_duration": 1.4,
    # PD drive gains
    "gantry_drive_ke": 5.0e4,
    "gantry_drive_kd": 5.0e3,
    "finger_drive_ke": 1.0e5,
    "finger_drive_kd": 1.0e2,
    "gantry_link_mass": 0.01,
    # soft 1x1 grid: wide in X (corners outside the jaws), narrow in Y (corners
    # within |y| < open_half_gap so the OPEN jaws start clear and sweep in). The
    # diagonal edge runs corner-to-corner through the pinch centre.
    "grid_half_x": 0.12,
    "grid_half_y": 0.045,
    "grid_mass": 0.1,
    "particle_radius": 0.01,
    "grid_color": (0.2, 0.8, 0.2),
    # soft-rigid contact material (the grip) -- soft enough to avoid crushing the mesh
    "soft_contact_ke": 1.0e3,
    "soft_contact_kd": 1.0e1,
    "soft_contact_mu": 1.0,
    # collision
    "rigid_body_contact_buffer_size": 512,
    "collision_broad_phase": "nxn",
    "soft_contact_margin": 0.01,
    # cloth stiffness (soft enough that the corners drape over the grip)
    "grid_tri_ke": 5.0e2,
    "grid_tri_kd": 1.0e-1,
    # camera (fixed; side view so the draping sheet reads as a sheet)
    "camera_pos": (0.55, -0.70, 0.50),
    "camera_fov": 31.0,
    "camera_pitch": -10.0,
    "camera_yaw": 128.0,
    "draw_wireframe": False,
    "initial_paused": False,
}


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.params = PARAMS
        self.sim_time = 0.0
        self.fps = self.params["fps"]
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = self.params["sim_substeps"]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame = 0
        self._current_waypoint = 0
        self._time_in_waypoint = 0.0
        self._gripper_frac = 0.0

        builder = newton.ModelBuilder(gravity=self.params["gravity"])
        self._build_gripper(builder)
        self._add_soft_mesh(builder)

        builder.color()
        self.model = builder.finalize()

        self.model.soft_contact_ke = self.params["soft_contact_ke"]
        self.model.soft_contact_kd = self.params["soft_contact_kd"]
        self.model.soft_contact_mu = self.params["soft_contact_mu"]

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.params["solver_iterations"],
            integrate_with_external_rigid_solver=False,
            rigid_body_contact_buffer_size=self.params["rigid_body_contact_buffer_size"],
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=self.params["collision_broad_phase"],
            soft_contact_margin=self.params["soft_contact_margin"],
            enable_rigid_soft_full_surface_contact=self.params["enable_water_tight"],
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.state_1.body_q, self.state_0.body_q)

        self._build_waypoints()
        self._set_targets(self._waypoints[0][0], 0.0)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = self.params["draw_wireframe"]
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = self.params["initial_paused"]
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                wp.vec3(*self.params["camera_pos"]),
                self.params["camera_pitch"],
                self.params["camera_yaw"],
            )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.params["camera_fov"]

    # ── model construction ──────────────────────────────────────────────

    def _build_gripper(self, builder):
        p = self.params
        grab_z = p["grab_z"]
        h = p["finger_half"]

        gantry_z = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, grab_z), wp.quat_identity()),
            mass=p["gantry_link_mass"],
            label="gantry_z",
        )
        gantry_x = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, grab_z), wp.quat_identity()),
            mass=p["gantry_link_mass"],
            label="gantry_x",
        )
        left_finger = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, -p["open_half_gap"], grab_z - p["finger_z_offset"]), wp.quat_identity()),
            label="left_finger",
        )
        right_finger = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, p["open_half_gap"], grab_z - p["finger_z_offset"]), wp.quat_identity()),
            label="right_finger",
        )

        finger_cfg = newton.ModelBuilder.ShapeConfig(density=p["finger_density"], mu=p["soft_contact_mu"])
        if p["enable_water_tight"]:
            # Provision the rigid finger mesh's SDF for full-surface rigid-soft contact (analytic box
            # fingers ignore this -- finalize only builds mesh/convex SDFs).
            finger_cfg.configure_sdf(force_sdf=True)
        if p["finger_mesh"]:
            cube = _box_mesh(h)  # one mesh shared by both jaws -> a single deduplicated SDF
            builder.add_shape_mesh(left_finger, mesh=cube, cfg=finger_cfg, color=p["finger_color_left"])
            builder.add_shape_mesh(right_finger, mesh=cube, cfg=finger_cfg, color=p["finger_color_right"])
        else:
            builder.add_shape_box(left_finger, hx=h, hy=h, hz=h, cfg=finger_cfg, color=p["finger_color_left"])
            builder.add_shape_box(right_finger, hx=h, hy=h, hz=h, cfg=finger_cfg, color=p["finger_color_right"])

        j_z = builder.add_joint_prismatic(
            parent=-1,
            child=gantry_z,
            axis=wp.vec3(0.0, 0.0, 1.0),
            target_ke=p["gantry_drive_ke"],
            target_kd=p["gantry_drive_kd"],
            target_pos=grab_z,
            label="gantry_z_joint",
        )
        j_x = builder.add_joint_prismatic(
            parent=gantry_z,
            child=gantry_x,
            axis=wp.vec3(1.0, 0.0, 0.0),
            target_ke=p["gantry_drive_ke"],
            target_kd=p["gantry_drive_kd"],
            target_pos=0.0,
            label="gantry_x_joint",
        )
        j_left = builder.add_joint_prismatic(
            parent=gantry_x,
            child=left_finger,
            axis=wp.vec3(0.0, -1.0, 0.0),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, p["finger_z_offset"]), wp.quat_identity()),
            target_ke=p["finger_drive_ke"],
            target_kd=p["finger_drive_kd"],
            target_pos=p["open_half_gap"],
            label="left_finger_joint",
        )
        j_right = builder.add_joint_prismatic(
            parent=gantry_x,
            child=right_finger,
            axis=wp.vec3(0.0, 1.0, 0.0),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, p["finger_z_offset"]), wp.quat_identity()),
            target_ke=p["finger_drive_ke"],
            target_kd=p["finger_drive_kd"],
            target_pos=p["open_half_gap"],
            label="right_finger_joint",
        )
        builder.add_articulation([j_z, j_x, j_left, j_right], label="gripper")

        builder.joint_q[0] = grab_z
        builder.joint_q[1] = 0.0
        builder.joint_q[2] = p["open_half_gap"]
        builder.joint_q[3] = p["open_half_gap"]
        self._dof_z, self._dof_x, self._dof_left, self._dof_right = 0, 1, 2, 3

    def _add_soft_mesh(self, builder):
        """A horizontal soft 1x1 grid (one quad -> two triangles + one diagonal edge),
        centred on the pinch point. Wide in x (corners outside the jaws), narrow in y
        (corners within the open gap). The diagonal edge crosses the centre, so the
        gripper grips it via the water-tight EDGE pass."""
        p = self.params
        z = p["grab_z"] - p["finger_z_offset"]
        hx, hy = p["grid_half_x"], p["grid_half_y"]
        p_start = len(builder.particle_q)
        builder.add_cloth_grid(
            pos=wp.vec3(-hx, -hy, z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            cell_x=2.0 * hx,
            cell_y=2.0 * hy,
            mass=p["grid_mass"],
            particle_radius=p["particle_radius"],
            tri_ke=p["grid_tri_ke"],
            tri_kd=p["grid_tri_kd"],
        )
        self._mesh_particles = list(range(p_start, len(builder.particle_q)))

    # ── trajectory: close -> pinch -> lift -> hold ──────────────────────

    def _build_waypoints(self):
        p = self.params
        grab = wp.vec3(0.0, 0.0, p["grab_z"])
        lift = wp.vec3(0.0, 0.0, p["lift_height"])
        self._waypoints = [
            (grab, p["close_duration"], 0.0),
            (grab, p["pinch_duration"], 1.0),
            (lift, p["lift_duration"], 1.0),
            (lift, p["hold_duration"], 1.0),
        ]

    def _advance_waypoint(self):
        self._time_in_waypoint += self.frame_dt
        cur = self._waypoints[self._current_waypoint]
        nxt = self._waypoints[min(self._current_waypoint + 1, len(self._waypoints) - 1)]
        t = min(self._time_in_waypoint / cur[1], 1.0)
        target_pos = cur[0] * (1.0 - t) + nxt[0] * t
        self._gripper_frac = float(cur[2]) * (1.0 - t) + float(nxt[2]) * t
        if self._time_in_waypoint >= cur[1] and self._current_waypoint < len(self._waypoints) - 1:
            self._current_waypoint += 1
            self._time_in_waypoint = 0.0
        self._set_targets(target_pos, self._gripper_frac)

    def _set_targets(self, pos: wp.vec3, gripper_frac: float):
        p = self.params
        half_gap = p["open_half_gap"] * (1.0 - gripper_frac) + p["closed_half_gap"] * gripper_frac
        targets = np.zeros(self.model.joint_dof_count)
        targets[self._dof_z] = float(pos[2])
        targets[self._dof_x] = float(pos[0])
        targets[self._dof_left] = half_gap
        targets[self._dof_right] = half_gap
        self.control.joint_target_q.assign(wp.array(targets, dtype=float, device=self.model.device))

    # ── simulation loop ─────────────────────────────────────────────────

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.frame += 1
        self._advance_waypoint()
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """The gripped grid must be held in the air, not dropped to the ground."""
        q = self.state_0.particle_q.numpy()
        mean_z = float(q[self._mesh_particles, 2].mean())
        if not mean_z > 0.3:
            raise AssertionError(f"soft grid was dropped (mean z={mean_z:.3f}); water-tight edge grip failed")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=PARAMS["num_frames"])
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
