# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact-driven straw insertion and stirring in a single reduced-elastic cap.

Run with ``uv run -m newton.examples basic_reduced_elastic_cup_straw``.
Use ``--motion insert`` for straight insertion, or ``--view closeup`` to inspect
the slit. The default sequence lasts 9.5 seconds, including withdrawal.

The cap is one elastic link with four POD coordinates and a cup-fixed frame.
Its basis is generated from geometric bent-tongue poses, not shell-FEM output;
mass is area-lumped and modal stiffness is fitted to plate bending energy.
Only the straw trajectory is prescribed. Contact drives the free modal DOFs.
The cup and rim are visual-only geometry, representing a fixed support.

This is a qualitative small-basis demonstration: no plasticity, tearing,
self-contact, or liquid. The linear basis cannot support arbitrary folding.
The validated stirring tilt is 10 degrees; a 25-degree trial overextended the
basis and snagged during withdrawal. See the report for limitations:
https://reports.mmacklin.com/cup-straw-rom/index.html
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
import newton.viewer
from newton.examples.basic._reduced_elastic_cap import build_cap_rom, cup_mesh, ring_mesh


def shape_config(cfg, collision=True):
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=cfg.contact_ke,
        kd=cfg.contact_kd,
        mu=cfg.friction,
        margin=0.0,
        gap=cfg.contact_gap,
        has_shape_collision=collision,
        has_particle_collision=False,
    )


@wp.func
def smooth_ramp(t: float, start: float, duration: float):
    s = wp.clamp((t - start) / duration, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@wp.func
def straw_pose(t: float, lid_z: float, half_height: float, offset: float, stir: bool, tilt: float, radius: float):
    withdrawal = float(2.3)
    rotation = wp.quat_identity()
    pivot = wp.vec3(offset, 0.0, lid_z)
    if stir:
        withdrawal = 7.3
        phase = 4.0 * wp.pi * smooth_ramp(t, 2.5, 4.0)
        rotation = wp.quat_from_axis_angle(wp.vec3(-wp.sin(phase), wp.cos(phase), 0.0), tilt)
        orbit = radius * smooth_ramp(t, 2.3, 0.3) * (1.0 - smooth_ramp(t, 6.5, 0.3))
        pivot = pivot + wp.vec3(orbit * wp.cos(phase), orbit * wp.sin(phase), 0.0)
    axial_tip = 0.010 - 0.035 * smooth_ramp(t, 0.3, 1.2) + 0.035 * smooth_ramp(t, withdrawal, 1.2)
    direction = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    return wp.transform(pivot + (half_height + axial_tip) * direction, rotation)


def cylinder_sdf(points, pose, radius, half_height):
    """Signed distance to the capped cylinder in its own rotating frame [m]."""
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    local = (np.asarray(points) - pose[:3]) @ rotation
    radial = np.linalg.norm(local[:, :2], axis=1) - radius
    axial = np.abs(local[:, 2]) - half_height
    return np.hypot(np.maximum(radial, 0), np.maximum(axial, 0)) + np.minimum(np.maximum(radial, axial), 0)


@wp.kernel
def drive_straw(
    clock: wp.array[float],
    dt: float,
    body: int,
    lid_z: float,
    half_height: float,
    offset: float,
    stir: bool,
    tilt: float,
    stir_radius: float,
    pose0: wp.array[wp.transform],
    vel0: wp.array[wp.spatial_vector],
    pose1: wp.array[wp.transform],
    vel1: wp.array[wp.spatial_vector],
):
    t = clock[0]
    pose = straw_pose(t, lid_z, half_height, offset, stir, tilt, stir_radius)
    nxt = straw_pose(t + dt, lid_z, half_height, offset, stir, tilt, stir_radius)
    v = (wp.transform_get_translation(nxt) - wp.transform_get_translation(pose)) / dt
    dq = wp.transform_get_rotation(nxt) * wp.quat_inverse(wp.transform_get_rotation(pose))
    axis, angle = wp.quat_to_axis_angle(dq)
    omega = axis * (angle / dt)
    vel = wp.spatial_vector(v, omega)
    pose0[body] = pose
    pose1[body] = pose
    vel0[body] = vel
    vel1[body] = vel


@wp.kernel
def advance_clock(clock: wp.array[float], dt: float):
    clock[0] = clock[0] + dt


class Example:
    @dataclass
    class Config:
        flap_length: float = 0.010
        cap_radius: float = 0.046
        thickness: float = 0.00012
        density: float = 950.0
        young_modulus: float = 0.8e9
        poisson_ratio: float = 0.35
        damping_ratio: float = 0.15
        lid_z: float = 0.125
        straw_radius: float = 0.004
        straw_half_height: float = 0.08
        offset: float = 0.0
        friction: float = 0.20
        contact_ke: float = 2e4
        contact_kd: float = 0.02
        contact_gap: float = 0.0004
        subdivisions: int = 16
        iterations: int = 12
        substeps: int = 4
        contact_enabled: bool = True
        stir: bool = True
        tilt_degrees: float = 10.0
        stir_radius: float = 0.0005

    def __init__(self, viewer, args=None, *, config: Config | None = None):
        self.cfg = cfg = config or self.Config(stir=getattr(args, "motion", "stir") == "stir")
        self.metrics_enabled = bool(getattr(args, "test", False))
        if cfg.substeps < 1 or cfg.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        if wp.get_device().is_cuda and cfg.substeps % 2:
            raise ValueError("CUDA graph replay requires an even substep count")
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_dt = self.frame_dt / cfg.substeps
        self.records = []
        (
            self.vertices,
            self.faces,
            self.basis,
            self.basis_audit,
            self.mid,
            self.mid_faces,
            self.snapshots,
            self.sector,
            self.depth,
        ) = build_cap_rom(cfg)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis="Z")
        builder.num_rigid_contacts_per_world = 4096
        mass = self.basis_audit["cap_mass_kg"]
        inertia = mass * cfg.cap_radius**2 / 4
        # Fix only the floating frame; the four modal coordinates remain dynamic.
        self.cap = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0, 0, cfg.lid_z), wp.quat_identity()),
            mass=mass,
            inertia=wp.mat33(inertia, 0, 0, 0, inertia, 0, 0, 0, 2 * inertia),
            modal_basis=self.basis,
            is_kinematic=True,
            lock_inertia=True,
            label="whole_cap_ROM",
        )
        self.cap_shape = builder.add_shape_mesh(
            self.cap,
            mesh=newton.Mesh(self.vertices, self.faces.flatten()),
            cfg=shape_config(cfg, cfg.contact_enabled),
            color=(0.80, 0.86, 0.87),
            label="single_slit_cap",
        )
        self.straw = builder.add_body(
            xform=straw_pose(
                0.0,
                cfg.lid_z,
                cfg.straw_half_height,
                cfg.offset,
                cfg.stir,
                float(np.deg2rad(cfg.tilt_degrees)),
                cfg.stir_radius,
            ),
            is_kinematic=True,
            label="prescribed_straw",
        )
        self.straw_shape = builder.add_shape_cylinder(
            self.straw,
            radius=cfg.straw_radius,
            half_height=cfg.straw_half_height,
            cfg=shape_config(cfg),
            color=(0.78, 0.055, 0.025),
            label="straw_solid_cylinder_proxy",
        )
        visual = shape_config(cfg, False)
        p, f = cup_mesh(height=cfg.lid_z)
        self.cup_shape = builder.add_shape_mesh(
            -1,
            mesh=newton.Mesh(p, f.flatten()),
            cfg=visual,
            color=(0.72, 0.79, 0.84),
            opacity=0.23,
            label="fixed_cup_display",
        )
        p, f = ring_mesh(0.0445, 0.047, cfg.lid_z + 0.001, 0.0014)
        builder.add_shape_mesh(
            -1, mesh=newton.Mesh(p, f.flatten()), cfg=visual, color=(0.76, 0.82, 0.83), label="fixed_lid_rim"
        )
        builder.color()
        self.model = builder.finalize()
        self.vertex_phi = self.model.elastic_shape_vertex_phi.numpy().reshape(-1, 4, 3)[: len(self.vertices)].copy()
        np.testing.assert_allclose(
            self.model.elastic_shape_vertex_local.numpy()[: len(self.vertices)], self.vertices, atol=1e-7
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pipeline = newton.CollisionPipeline(self.model, reduce_contacts=False)
        self.control, self.contacts = self.model.control(), self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=cfg.iterations,
            rigid_body_contact_buffer_size=1024,
            rigid_contact_k_start=cfg.contact_ke,
            rigid_compliant_alm=False,
            rigid_contact_hard=False,
            friction_epsilon=1e-4,
            elastic_body_relaxation=0.6,
            elastic_solver_metrics=False,
        )
        elastic_id = int(self.model.body_elastic_index.numpy()[self.cap])
        owner = int(self.model.elastic_joint.numpy()[elastic_id])
        self.q_start = int(self.model.joint_q_start.numpy()[owner]) + 7
        self.tip_indices = [int(np.flatnonzero((self.sector == i) & (self.depth > 0.98))[0]) for i in range(4)]
        self.edges = np.concatenate([self.mid_faces[:, [0, 1]], self.mid_faces[:, [1, 2]], self.mid_faces[:, [2, 0]]])
        self.rest_edge_lengths = np.linalg.norm(self.mid[self.edges[:, 1]] - self.mid[self.edges[:, 0]], axis=1)
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        if viewer is not None:
            self.setup_viewer(viewer, getattr(args, "view", "overview"))

    def setup_viewer(self, viewer, view="overview"):
        self.viewer = viewer
        viewer.set_model(self.model)
        viewer.show_elastic_strain = False
        viewer.show_joints = False
        viewer.show_com = False
        if isinstance(viewer, newton.viewer.ViewerGL):
            viewer.renderer.draw_sky = False
            viewer.renderer.background_color = (0.86, 0.88, 0.90)
            viewer.renderer.sky_upper = (0.86, 0.88, 0.90)
            viewer.renderer.sky_lower = (0.86, 0.88, 0.90)
            viewer.renderer.shadow_extents = 0.35
            viewer.renderer.exposure = 1.1
            viewer.renderer.ambient_sky = (0.5, 0.5, 0.52)
            viewer.renderer.ambient_ground = (0.16, 0.16, 0.18)
        if hasattr(viewer, "camera"):
            viewer.camera.near = 0.0002
            viewer.camera.far = 10.0
            viewer.camera.fov = 38.0
        if view == "closeup":
            viewer.set_camera(pos=wp.vec3(0.028, -0.040, 0.164), pitch=-41, yaw=125)
        elif view == "underside":
            viewer.set_camera(pos=wp.vec3(0.031, -0.044, 0.097), pitch=24, yaw=125)
        else:
            viewer.set_camera(pos=wp.vec3(0.17, -0.24, 0.245), pitch=-26, yaw=125)

    def simulate(self):
        c = self.cfg
        for _ in range(c.substeps):
            wp.launch(
                drive_straw,
                dim=1,
                inputs=[
                    self.clock,
                    self.sim_dt,
                    self.straw,
                    c.lid_z,
                    c.straw_half_height,
                    c.offset,
                    c.stir,
                    float(np.deg2rad(c.tilt_degrees)),
                    c.stir_radius,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_clock, dim=1, inputs=[self.clock, self.sim_dt], device=self.model.device)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        if self.metrics_enabled:
            self.update_metrics()

    def update_metrics(self):
        c = self.cfg
        poses = self.state_0.body_q.numpy()
        q = self.state_0.joint_q.numpy()
        if not (np.isfinite(poses).all() and np.isfinite(q).all()):
            raise AssertionError(f"Non-finite state at {self.sim_time:.3f} s")
        amplitudes = q[self.q_start : self.q_start + 4]
        disp = np.einsum("vmc,m->vc", self.vertex_phi, amplitudes)
        vertices = self.vertices + disp
        rot = np.asarray(wp.quat_to_matrix(wp.quat(*poses[self.cap, 3:])), dtype=float).reshape(3, 3)
        world = vertices @ rot.T + poses[self.cap, :3]
        # Check face-interior samples in addition to the actual contact vertices.
        tri = world[self.faces]
        samples = np.concatenate(
            [
                world,
                tri.mean(axis=1),
                (tri[:, 0] + tri[:, 1]) / 2,
                (tri[:, 1] + tri[:, 2]) / 2,
                (tri[:, 2] + tri[:, 0]) / 2,
            ]
        )
        center = poses[self.straw, :3]
        sdf = cylinder_sdf(samples, poses[self.straw], c.straw_radius, c.straw_half_height)
        axis = np.asarray(wp.quat_rotate(wp.quat(*poses[self.straw, 3:]), wp.vec3(0, 0, 1)))
        tip = center - c.straw_half_height * axis
        lid_intersection = center + (c.lid_z - center[2]) / axis[2] * axis
        tip_drop = -disp[self.tip_indices, 2]
        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        mids = vertices[: len(self.mid)]
        edge1 = mids[self.edges[:, 1]] - mids[self.edges[:, 0]]
        strain = np.linalg.norm(edge1, axis=1) / self.rest_edge_lengths - 1
        self.records.append(
            {
                "time": self.sim_time,
                "tip_drop_m": tip_drop.tolist(),
                "penetration_m": max(0.0, -float(sdf.min())),
                "max_deformation_m": float(np.linalg.norm(disp, axis=1).max()),
                "max_edge_length_change": float(np.abs(strain).max()),
                "max_rim_displacement_m": float(np.linalg.norm(disp[: len(self.mid)][self.sector < 0], axis=1).max()),
                "contacts": contact_count,
                "straw_tip_z": float(tip[2]),
                "straw_tip_world": tip.tolist(),
                "straw_lid_intersection": lid_intersection.tolist(),
                "straw_tilt_deg": float(np.rad2deg(np.arccos(np.clip(axis[2], -1, 1)))),
                "q": amplitudes.tolist(),
            }
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def summary(self):
        hold = [r for r in self.records if 1.8 < r["time"] < 2.25]
        release = [r for r in self.records if r["time"] > (9.1 if self.cfg.stir else 4.1)]
        hold_drop = np.array([r["tip_drop_m"] for r in hold])
        result = {
            "basis": self.basis_audit,
            "formulation": "one reduced-elastic cap, four POD modes, fixed frame; no flap hinges",
            "device": str(self.model.device),
            "graph_enabled": self.graph is not None,
            "elastic_body_count": int(self.model.elastic_body_count),
            "frames": len(self.records),
            "duration_s": self.sim_time,
            "max_deformation_m": max(r["max_deformation_m"] for r in self.records),
            "max_penetration_m": max(r["penetration_m"] for r in self.records),
            "max_edge_length_change": max(r["max_edge_length_change"] for r in self.records),
            "max_rim_displacement_m": max(r["max_rim_displacement_m"] for r in self.records),
            "hold_mean_tip_drop_m": hold_drop.mean(axis=0).tolist() if hold else [],
            "hold_tip_peak_to_peak_m": float(np.ptp(hold_drop, axis=0).max()) if hold else None,
            "hold_contact_dropouts": sum(r["contacts"] == 0 for r in hold),
            "final_tip_drop_m": self.records[-1]["tip_drop_m"],
            "release_max_deformation_m": max(r["max_deformation_m"] for r in release) if release else None,
        }
        if self.cfg.stir:
            moving = [r for r in self.records if 2.8 < r["time"] < 6.4]
            stopped = [r for r in self.records if 7.0 < r["time"] < 7.25]
            tips = np.array([r["tip_drop_m"] for r in moving])
            result.update(
                {
                    "stir_contact_dropouts": sum(r["contacts"] == 0 for r in moving),
                    "stir_tip_range_m": np.ptp(tips, axis=0).tolist(),
                    "stir_max_tip_step_m": float(np.abs(np.diff(tips, axis=0)).max()),
                    "stir_max_tip_second_difference_m": float(np.abs(np.diff(tips, n=2, axis=0)).max()),
                    "post_stir_hold_tip_peak_to_peak_m": float(
                        np.ptp([r["tip_drop_m"] for r in stopped], axis=0).max()
                    ),
                    "post_stir_hold_contact_dropouts": sum(r["contacts"] == 0 for r in stopped),
                }
            )
        return result

    def test_final(self):
        if not self.records:
            raise AssertionError("Enable --test to collect validation metrics")
        duration = 9.5 if self.cfg.stir else 4.5
        if self.sim_time < duration - 1.0e-5:
            raise AssertionError(f"Run at least {round(duration / self.frame_dt)} frames to validate release")
        s = self.summary()
        assert s["elastic_body_count"] == 1
        assert np.isfinite(self.state_0.body_q.numpy()).all()
        assert np.isfinite(self.state_0.joint_q.numpy()).all()
        assert np.isfinite(self.state_0.body_qd.numpy()).all()
        assert np.isfinite(self.state_0.joint_qd.numpy()).all()
        assert s["max_rim_displacement_m"] < 1e-6, s
        if not self.cfg.contact_enabled:
            assert s["max_deformation_m"] < 1e-5, s
            return
        assert 0.003 < s["max_deformation_m"] < 0.020, s
        assert s["max_penetration_m"] < 0.0006, s
        assert s["hold_tip_peak_to_peak_m"] < 0.0002, s
        assert s["hold_contact_dropouts"] == 0, s
        assert s["release_max_deformation_m"] < 0.0002, s
        if self.cfg.stir:
            assert s["max_edge_length_change"] < 0.15, s
            assert s["stir_contact_dropouts"] == 0, s
            assert min(s["stir_tip_range_m"]) > 0.001, s
            assert s["stir_max_tip_step_m"] < 0.0004, s
            assert s["stir_max_tip_second_difference_m"] < 0.0001, s
            assert s["post_stir_hold_tip_peak_to_peak_m"] < 0.0002, s
            assert s["post_stir_hold_contact_dropouts"] == 0, s


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=570)
    parser.add_argument("--motion", choices=("stir", "insert"), default="stir", help="Prescribed straw motion")
    parser.add_argument("--view", choices=("overview", "closeup", "underside"), default="overview")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
