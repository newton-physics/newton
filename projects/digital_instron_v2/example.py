# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Digital Instron Midsole
#
# A fully dynamic running-shoe midsole built from the calibrated column model
# in projects/digital_instron_v2. The midsole is a live bed of nonlinear
# viscoelastic Hyperfoam-Maxwell-Pasternak springs (see dynamics.py) sampled
# from the shoe mesh and coupled into Newton rigid-body physics through
# state.body_f. Three modes reuse the same foundation:
#
#   --mode instron  (default) Displacement-controlled digital Instron: a
#                   shoe-last crosshead squishes the midsole against the ground
#                   plane through the measured compression cycle and the
#                   force-displacement hysteresis loop is recorded.
#   --mode settle   A free, massive midsole rests in stable equilibrium on the
#                   foundation under gravity and resists a lateral load through
#                   Coulomb foam-shear friction.
#   --mode stride   A synthetic running stride rolls a foot heel-to-toe over the
#                   foundation, producing a ground-reaction force profile and a
#                   migrating center of pressure.
#   --mode attached A fully dynamic, foot-mounted shoe with mass and inertia. A
#                   damped bilateral "upper" keeps the midsole coupled to the foot
#                   for the whole stride, so the shoe presses the foam into the
#                   ground in stance and the whole bed lifts clear with the foot in
#                   flight, with the stance/flight ground reaction recorded.
#
# Command: uv run -m projects.digital_instron_v2.example --mode instron
#
###########################################################################

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

from .dynamics import (
    FoundationConfig,
    MidsoleFoundation,
    attach_coupling,
    attached_columns,
    build_foundation_geometry,
    column_colors,
    column_world_positions,
    cyclic_displacement,
    load_fitted_material,
    load_measured_cycle,
    synthetic_stride,
)
from .geometry import load_mesh, transform_mesh

MANIFEST = "DigitalInstron/manifest_v2.json"
INSTRON_CYCLES = 6  # warm-up cycles before the reported hysteresis loop
SETTLE_PUSH_N = 4.0  # lateral load used to probe foam-shear friction [N]
SETTLE_MASS_KG = 0.8  # midsole + representative attachment mass [kg]
STRIDE_PERIOD_S = 0.6  # synthetic running-stride period [s]
ATTACHED_MASS_KG = 0.5  # dynamic shoe mass: midsole + effective foot/attachment [kg]
ATTACHED_PERIOD_S = 0.7  # attached running-stride period [s]


def _loop_area(x: np.ndarray, y: np.ndarray) -> float:
    """Signed hysteresis-loop area (dissipated energy per cycle) via the shoelace formula [J]."""
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class Example:
    """Fully dynamic viscoelastic midsole driven as a digital Instron, a settling load, or a stride."""

    def __init__(self, viewer, args=None):
        newton.use_coord_layout_targets = True
        self.mode = getattr(args, "mode", "instron")
        manifest = getattr(args, "manifest", MANIFEST)
        self.viewer = viewer
        self.device = wp.get_preferred_device()

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.geo = build_foundation_geometry(manifest)
        self.material = load_fitted_material(manifest)
        self.column_count = len(self.geo.slack_m)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        anchor_local, z_free, config = self._build_mode(builder, manifest)

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._graph = None
        self._use_graph = self._attached and self.device.is_cuda and not getattr(args, "eager", False)
        if self._attached:
            q0 = self._target_pose(0.0)
            self.state_0.body_q.assign(q0.reshape(1, 7))
            self.state_1.body_q.assign(q0.reshape(1, 7))
            self._coupling_force = wp.zeros(1, dtype=wp.float32, device=self.device)

        self.foundation = MidsoleFoundation(
            anchor_local,
            z_free,
            self.geo.slack_m,
            np.full(self.column_count, self.geo.area_m2),
            self.geo.neighbors,
            self.geo.spacing_m,
            self.material,
            self.carrier,
            self.model.body_com,
            config,
            self.device,
        )
        self._anchor_local = wp.array(np.ascontiguousarray(anchor_local, np.float32), dtype=wp.vec3, device=self.device)
        self._z_free = wp.array(np.ascontiguousarray(z_free, np.float32), dtype=wp.float32, device=self.device)
        self._points = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        self._colors = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        foam_base = np.column_stack([self.geo.uv_m[:, 0], self.geo.uv_m[:, 1], self.geo.z_bottom_m])
        self._foam_base = wp.array(np.ascontiguousarray(foam_base, np.float32), dtype=wp.vec3, device=self.device)
        self._foam_top = wp.zeros(self.column_count, dtype=wp.vec3, device=self.device)
        self._slack = wp.array(np.ascontiguousarray(self.geo.slack_m, np.float32), dtype=wp.float32, device=self.device)
        # Compression that saturates the contact colour [m]; the attached stride penetrates
        # deeper than the bench Instron, so it needs a coarser scale.
        self._color_ref = 0.012 if self.mode == "attached" else 0.008

        # Diagnostics recorded once per frame.
        self.history: list[dict] = []

        self.viewer.set_model(self.model)
        span = float(np.ptp(self.geo.uv_m[:, 0]))
        eye = (self._cx + 0.9 * span, self._cy - 0.8 * span, 0.5 * span)
        target = (self._cx, self._cy, 0.02)
        self.viewer.set_camera(*_look_at(eye, target))

    # -- scene construction ------------------------------------------------
    def _build_mode(self, builder, manifest):
        """Configure the carrier body, foundation anchors, and driver for the active mode."""
        geo = self.geo
        self._attached = False
        self._cx = float(geo.uv_m[:, 0].mean())
        self._cy = float(geo.uv_m[:, 1].mean())
        self._cz = float(geo.surface_m.mean())

        if self.mode == "settle":
            anchor_local = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.slack_m])
            z_free = geo.slack_m.copy()
            self.carrier = builder.add_body(
                mass=SETTLE_MASS_KG,
                com=wp.vec3(0.0, 0.0, 0.0),
                inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
            )
            cfg = FoundationConfig(
                stretch_floor=0.05, normal_damping=8.0, friction_stiffness=2.0e4, friction=10.0, mu=1.0
            )
            self._add_plate_visual(builder)
            self._driven = False

        elif self.mode == "stride":
            anchor_local = np.column_stack(
                [geo.uv_m[:, 0] - self._cx, geo.uv_m[:, 1] - self._cy, geo.surface_m - self._cz]
            )
            z_free = geo.z_free_m.copy()
            self.carrier = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
            cfg = FoundationConfig(stretch_floor=0.05)
            span = float(np.ptp(geo.uv_m[:, 0]))
            self._stride = synthetic_stride(0.014, 5.0, 0.12 * span, STRIDE_PERIOD_S)
            self._add_last_visual(builder, manifest, at_com=True)
            self._driven = True

        elif self.mode == "attached":
            # Dynamic, foot-mounted shoe. The shoe body carries the foam against the
            # ground (settle physics: ground reaction in stance, zero in flight), and a
            # damped PD "upper" holds it to the prescribed foot stride so the two never
            # separate. The COM sits at the footprint centroid so the shoe pitches
            # realistically heel-to-toe.
            self._com_z = float(geo.z_free_m.mean())
            self._cz = self._com_z
            # The foam is the shoe's sole, so anchor the columns at the outsole (bottom) in the
            # body frame: the whole foam bed then lifts with the shoe in flight and compresses
            # only by penetrating the ground plane (z_free = 0) in stance.
            anchor_local = np.column_stack(
                [geo.uv_m[:, 0] - self._cx, geo.uv_m[:, 1] - self._cy, geo.z_bottom_m - self._com_z]
            )
            z_free = np.zeros(self.column_count, dtype=np.float32)
            span_x = float(np.ptp(geo.uv_m[:, 0]))
            # Effective rotational inertia of the foot + shank that drives the pitch. A bare
            # thin-slab shoe inertia is far too small for the stiff, spatially distributed
            # foam and makes the explicit pitch mode diverge.
            self.carrier = builder.add_body(
                mass=ATTACHED_MASS_KG,
                com=wp.vec3(0.0, 0.0, 0.0),
                inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 0.08),
            )
            cfg = FoundationConfig(
                stretch_floor=0.05, normal_damping=8.0, friction_stiffness=2.0e4, friction=10.0, mu=1.0
            )
            # The stiff foam + light shoe is numerically stiff, so integrate the attached
            # stride with finer substeps than the driven/settle scenarios (converged at 128).
            self.sim_substeps = 128
            self.sim_dt = self.frame_dt / self.sim_substeps
            self._stride = synthetic_stride(0.024, 7.0, 0.08 * span_x, ATTACHED_PERIOD_S)
            # Compliant, bilateral shoe upper: near-silent in flight, stiff and damped in
            # stance so the foot presses the midsole down without separating.
            self._kp_lin, self._kd_lin = 1.0e5, 450.0
            self._kp_ang, self._kd_ang = 300.0, 40.0
            self._max_force = 20000.0
            self._add_last_visual(builder, manifest, at_com=True)
            self._build_target_trajectory()
            self._driven = False
            self._attached = True

        else:  # instron
            self.mode = "instron"
            anchor_local = np.column_stack([geo.uv_m[:, 0], geo.uv_m[:, 1], geo.surface_m])
            z_free = geo.z_free_m.copy()
            self.carrier = builder.add_body(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
            cfg = FoundationConfig(stretch_floor=0.05)
            time_s, disp_m, self._measured_force = load_measured_cycle(manifest)
            self._measured_disp = disp_m
            self._depth, self._period = cyclic_displacement(time_s, disp_m)
            self._add_last_visual(builder, manifest, at_com=False)
            self._driven = True

        return anchor_local, z_free, cfg

    def _add_last_visual(self, builder, manifest, at_com):
        """Attach the shoe-last indenter mesh to the carrier body for rendering."""
        path = Path(manifest).resolve()
        cfg = json.loads(path.read_text())
        src = next(t for t in cfg["trials"] if t["fixture"] == "fullfoot_last")["indenter"]
        last = load_mesh(path.parent / src["path"], 0.001, src["rotation_deg"], src["crop_height_m"])
        transform_mesh(last, src.get("pose_rotation_deg", [0, 0, 0]), src.get("pose_translation_m", [0, 0, 0]))
        verts = np.asarray(last.vertices, np.float32).copy()
        # Match the contact-surface offset that build_foundation_geometry applied to the
        # physics so the rendered last rests on the foam top instead of inside the midsole.
        verts[:, self.geo.thickness_axis] += self.geo.indenter_shift_m
        verts[:, 2] -= self.geo.z_shift_m
        if at_com:
            verts[:, 0] -= self._cx
            verts[:, 1] -= self._cy
            verts[:, 2] -= self._cz
        mesh = newton.Mesh(verts, np.asarray(last.faces, np.int32).flatten())
        builder.add_shape_mesh(
            self.carrier,
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            color=(0.7, 0.72, 0.75),
            label="shoe_last",
        )

    def _add_plate_visual(self, builder):
        """Attach a thin load plate to the free midsole body for rendering."""
        hx = 0.5 * float(np.ptp(self.geo.uv_m[:, 0])) + 0.01
        hy = 0.5 * float(np.ptp(self.geo.uv_m[:, 1])) + 0.01
        builder.add_shape_box(
            self.carrier,
            xform=wp.transform(wp.vec3(self._cx, self._cy, float(self.geo.slack_m.mean()) + 0.006), wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=0.005,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            color=(0.55, 0.45, 0.35),
            label="load_plate",
        )

    # -- simulation --------------------------------------------------------
    def _carrier_pose(self, t):
        """Return the prescribed carrier transform (7-vector) for a driven mode at time ``t``."""
        if self.mode == "stride":
            pos, rot = self._stride(t)
            return np.array(
                [self._cx + pos[0], self._cy + pos[1], self._cz + pos[2], rot[0], rot[1], rot[2], rot[3]],
                dtype=np.float32,
            )
        depth = self._depth(t)
        return np.array([0.0, 0.0, -depth, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _target_pose(self, t):
        """Return the prescribed foot/shoe target transform (7-vector) for the attached mode."""
        pos, rot = self._stride(t)
        return np.array(
            [self._cx + pos[0], self._cy + pos[1], self._com_z + pos[2], rot[0], rot[1], rot[2], rot[3]],
            dtype=np.float32,
        )

    def _pose_velocity(self, prev, cur, dt):
        """Return the (linear, angular) velocity carrying the target from ``prev`` to ``cur``."""
        v = (cur[:3] - prev[:3]) / dt
        q_rel = _quat_mul(cur[3:7], _quat_inv(prev[3:7]))
        if q_rel[3] < 0.0:
            q_rel = -q_rel
        w = 2.0 * q_rel[:3] / dt
        return v, w

    def _build_target_trajectory(self):
        """Precompute one stride period of per-substep target poses/velocities on the GPU.

        The prescribed foot stride is periodic, so evaluating it once into device arrays lets
        the attached substep loop run with no per-substep host work, which in turn lets the
        whole frame be captured into a single CUDA graph.
        """
        period = int(round(ATTACHED_PERIOD_S / self.sim_dt))
        poses = np.stack([self._target_pose(j * self.sim_dt) for j in range(period)]).astype(np.float32)
        vels = np.zeros((period, 6), dtype=np.float32)
        for j in range(period):
            v, w = self._pose_velocity(poses[(j - 1) % period], poses[j], self.sim_dt)
            vels[j] = np.concatenate([v, w])
        self._period_substeps = period
        self._target_traj = wp.array(np.ascontiguousarray(poses), dtype=wp.transform, device=self.device)
        self._target_vel_traj = wp.array(np.ascontiguousarray(vels), dtype=wp.spatial_vector, device=self.device)
        self._substep_counter = wp.zeros(1, dtype=wp.int32, device=self.device)

    def _attached_substeps(self):
        """Advance the attached dynamic shoe by one frame using only device-side work."""
        for _ in range(self.sim_substeps):
            # The fused reset zeros the carrier wrench too, so no separate clear_forces launch.
            self.foundation.apply(self.state_0, self.sim_dt, clear_body_force=True)
            wp.launch(
                attach_coupling,
                dim=1,
                inputs=[
                    self.carrier,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self._target_traj,
                    self._target_vel_traj,
                    self._substep_counter,
                    self._period_substeps,
                    self._kp_lin,
                    self._kd_lin,
                    self._kp_ang,
                    self._kd_ang,
                    self._max_force,
                    self.state_0.body_f,
                    self._coupling_force,
                ],
                device=self.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self):
        """Advance the foundation and, for the free-body modes, the rigid-body solver by one frame."""
        if self._attached:
            # The attached loop is fully device-side, so capture the whole frame into a CUDA
            # graph once and replay it -- this removes the per-substep kernel-launch overhead
            # that dominates the cost of this launch-bound scenario.
            if self._use_graph:
                if self._graph is None:
                    with wp.ScopedCapture(device=self.device) as capture:
                        self._attached_substeps()
                    self._graph = capture.graph
                wp.capture_launch(self._graph)
            else:
                self._attached_substeps()
            self.sim_time += self.frame_dt
            return

        for _ in range(self.sim_substeps):
            if self._driven:
                self.state_0.body_q.assign(self._carrier_pose(self.sim_time).reshape(1, 7))
                self.state_0.body_qd.zero_()
                self.state_0.clear_forces()
                self.foundation.apply(self.state_0, self.sim_dt)
            else:
                self.state_0.clear_forces()
                if self._pushing():
                    self.state_0.body_f.assign(np.array([[SETTLE_PUSH_N, 0, 0, 0, 0, 0]], dtype=np.float32))
                self.foundation.apply(self.state_0, self.sim_dt)
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def _pushing(self):
        """Return True while the settle-mode lateral friction probe is active."""
        return self.mode == "settle" and 1.5 <= self.sim_time < 1.9

    def step(self):
        self.simulate()
        self._record()

    def _record(self):
        """Capture one frame of force-displacement / ground-reaction diagnostics."""
        diag = self.foundation.diagnostics()
        entry = {"t": self.sim_time, **diag}
        if self.mode == "instron":
            entry["depth"] = self._depth(self.sim_time)
        elif self.mode == "stride":
            body_q = self.state_0.body_q.numpy()[self.carrier]
            entry["foot_x"] = float(body_q[0])
        elif self.mode == "attached":
            body_q = self.state_0.body_q.numpy()[self.carrier]
            entry["com_z"] = float(body_q[2])
            entry["com_x"] = float(body_q[0])
            entry["coupling_n"] = float(self._coupling_force.numpy()[0])
            entry["target_z"] = float(self._target_pose(self.sim_time)[2])
        else:
            body_q = self.state_0.body_q.numpy()[self.carrier]
            entry["com_z"] = float(body_q[2])
            entry["com_x"] = float(body_q[0])
        self.history.append(entry)

    # -- rendering ---------------------------------------------------------
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        wp.launch(
            column_colors,
            dim=self.column_count,
            inputs=[self.foundation.compression, self._color_ref, self._colors],
            device=self.device,
        )
        if self.mode == "attached":
            # The foam is the shoe's sole: draw each column from its ground contact up to the
            # sole-mounted top so the whole bed lifts with the shoe in flight and the springs
            # compress under the foot in stance.
            wp.launch(
                attached_columns,
                dim=self.column_count,
                inputs=[
                    self.carrier,
                    self.state_0.body_q,
                    self._anchor_local,
                    self._slack,
                    self._points,
                    self._foam_top,
                ],
                device=self.device,
            )
            self.viewer.log_lines("midsole_springs", self._points, self._foam_top, self._colors, width=0.0035)
        else:
            wp.launch(
                column_world_positions,
                dim=self.column_count,
                inputs=[self.carrier, self.state_0.body_q, self._anchor_local, self._z_free, self._points],
                device=self.device,
            )
            if self.mode == "instron":
                # Vertical foam "springs" from the ground platen to the current (compressed)
                # foam top make the depressing contact patch legible.
                self.viewer.log_lines("midsole_springs", self._foam_base, self._points, self._colors, width=0.0035)
        self.viewer.log_points("midsole_columns", self._points, radii=0.0028, colors=self._colors)
        self.viewer.end_frame()

    # -- validation --------------------------------------------------------
    def test_final(self):
        """Audit the recorded response for the active mode."""
        assert np.all(np.isfinite(self.state_0.body_q.numpy())), "non-finite carrier pose"
        assert len(self.history) > 0, "no frames simulated"
        forces = np.array([h["normal_force_n"] for h in self.history])
        assert np.all(np.isfinite(forces)), "non-finite foundation force"
        getattr(self, f"_test_{self.mode}")(forces)

    def _test_instron(self, forces):
        """Verify a dissipative hysteresis loop with a calibrated peak force."""
        times = np.array([h["t"] for h in self.history])
        depth = np.array([h["depth"] for h in self.history])
        assert self.sim_time > (INSTRON_CYCLES - 1) * self._period, "run too short for a full hysteresis cycle"
        last = times >= (INSTRON_CYCLES - 1) * self._period
        loop_depth, loop_force = depth[last], forces[last]
        assert loop_depth.sum() > 0, "no compression cycle recorded"
        peak = float(loop_force.max())
        assert 1500.0 < peak < 2600.0, f"peak force {peak:.0f} N outside calibrated range"
        # A dissipative load-unload cycle (loading branch above unloading) traces the
        # (displacement, force) plane clockwise, so the dissipated energy is the negated
        # shoelace area; a positive signed area would mean unphysical energy generation.
        dissipated = -_loop_area(loop_depth * 1000.0, loop_force)  # mm * N = mJ
        assert dissipated > 300.0, f"hysteresis loop not dissipative: {dissipated:.1f} mJ"
        assert forces.max() > 10.0 * forces[forces > 0].min(), "no nonlinear stiffening observed"
        print(
            f"[instron] peak {peak:.0f} N at {loop_depth.max() * 1000:.1f} mm "
            f"(measured {self._measured_force.max():.0f} N); hysteresis {dissipated / 1000.0:.3f} J/cycle"
        )

    def _test_settle(self, forces):
        """Verify stable weight support and lateral grip from foam-shear friction."""
        com_z = np.array([h["com_z"] for h in self.history])
        com_x = np.array([h["com_x"] for h in self.history])
        settle = np.array([h["t"] for h in self.history]) < 1.5
        weight = SETTLE_MASS_KG * float(np.linalg.norm(self.model.gravity.numpy()[0]))
        pre = forces[settle]
        assert abs(pre[-1] - weight) < 0.25 * weight, f"foundation force {pre[-1]:.2f} N != weight {weight:.2f} N"
        assert com_z[settle][-1] > -0.006, f"midsole sank too far: {com_z[settle][-1] * 1000:.2f} mm"
        assert np.std(com_z[settle][-20:]) < 5.0e-4, "midsole did not reach a stable equilibrium"
        assert abs(com_x[-1]) < 0.02, (
            f"lateral grip failed: slid {com_x[-1] * 1000:.1f} mm under a {SETTLE_PUSH_N} N push"
        )
        print(
            f"[settle] {SETTLE_MASS_KG:.2f} kg held at {-com_z[settle][-1] * 1000:.2f} mm compression "
            f"(support {pre[-1]:.2f} N vs weight {weight:.2f} N); slid {abs(com_x).max() * 1000:.2f} mm "
            f"under a {SETTLE_PUSH_N:.0f} N push"
        )

    def _test_stride(self, forces):
        """Verify a rise-and-fall ground-reaction force and a heel-to-toe center of pressure."""
        times = np.array([h["t"] for h in self.history])
        cop_x = np.array([h["cop_x_m"] for h in self.history])
        foot_x = np.array([h["foot_x"] for h in self.history])
        assert forces.max() > 300.0, f"stride ground-reaction force too small: {forces.max():.0f} N"
        peak_frac = times[np.argmax(forces)] / STRIDE_PERIOD_S
        assert 0.1 < peak_frac < 0.6, f"ground-reaction force peak at {peak_frac:.2f} of stride, not mid-stance"
        assert forces.min() < 0.1 * forces.max(), "foot never lifted clear of the foundation during swing"
        # Restrict the center-of-pressure roll to the first stance so the check is not
        # confused by the following stride's heel strike.
        stance = (times < 0.62 * STRIDE_PERIOD_S) & (forces > 0.15 * forces.max())
        rel = (cop_x - foot_x)[stance]
        assert rel.size > 4, "too few loaded stance frames to measure a roll"
        assert rel[-1] - rel[0] > 0.05, (
            f"center of pressure did not roll heel-to-toe: {(rel[-1] - rel[0]) * 1000:.1f} mm"
        )
        print(
            f"[stride] peak GRF {forces.max():.0f} N at {peak_frac * 100:.0f}% of stride; "
            f"center of pressure rolled {rel[0] * 1000:.0f} -> {rel[-1] * 1000:.0f} mm heel-to-toe"
        )

    def _test_attached(self, forces):
        """Verify a dynamic foot-mounted shoe: stance ground reaction, no flight blow-up, stays attached."""
        times = np.array([h["t"] for h in self.history])
        coupling = np.array([h["coupling_n"] for h in self.history])
        com_z = np.array([h["com_z"] for h in self.history])
        target_z = np.array([h["target_z"] for h in self.history])
        assert np.all(np.isfinite(coupling)), "non-finite shoe-upper force"
        assert np.all(np.isfinite(com_z)), "non-finite shoe pose"
        warm = times > ATTACHED_PERIOD_S  # skip the first stride's settling transient
        f, c = forces[warm], coupling[warm]
        stance = target_z[warm] < self._com_z - 0.001  # foot pressing the midsole below rest
        flight = target_z[warm] > self._com_z + 0.02  # foot lifted clear of the ground
        assert stance.any() and flight.any(), "stride never separated stance from flight"
        grf_stance, grf_flight = float(f[stance].max()), float(f[flight].max())
        assert grf_stance > 300.0, f"stance ground reaction too small: {grf_stance:.0f} N"
        # The foam is off the ground during swing, so no spurious flight contact should appear.
        assert grf_flight < 0.05 * grf_stance, f"spurious flight ground reaction: {grf_flight:.0f} N"
        c_stance, c_flight = float(c[stance].max()), float(c[flight].max())
        assert c_stance > 0.5 * grf_stance, "shoe upper did not transmit the stance load"
        assert c_flight < 0.2 * c_stance, f"shoe upper not unloaded in flight: {c_flight:.0f} N"
        # The bilateral upper keeps the shoe tracking the foot within a bounded lag all stride.
        lag = float(np.abs(com_z[warm] - target_z[warm]).max())
        assert lag < 0.03, f"shoe separated from foot by {lag * 1000:.1f} mm"
        print(
            f"[attached] {ATTACHED_MASS_KG:.1f} kg shoe: stance GRF {grf_stance:.0f} N, flight GRF {grf_flight:.1f} N; "
            f"shoe-upper force {c_flight:.0f} N (flight) -> {c_stance:.0f} N (stance); tracked foot within {lag * 1000:.1f} mm"
        )


def _quat_mul(a, b):
    """Multiply two (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_inv(a):
    """Return the inverse (conjugate) of a unit (x, y, z, w) quaternion."""
    return np.array([-a[0], -a[1], -a[2], a[3]], dtype=np.float32)


def _look_at(eye, target):
    """Return (pos, pitch_deg, yaw_deg) for a Z-up camera at ``eye`` looking at ``target``."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    d /= np.linalg.norm(d)
    pitch = np.degrees(np.arcsin(d[2]))
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    return wp.vec3(*[float(v) for v in eye]), float(pitch), float(yaw)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["instron", "settle", "stride", "attached"],
        default="instron",
        help="Simulation scenario to run (instron | settle | stride | attached).",
    )
    parser.add_argument("--manifest", type=str, default=MANIFEST, help="Digital Instron manifest path.")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Disable CUDA-graph capture in the attached mode (slower; for debugging).",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
