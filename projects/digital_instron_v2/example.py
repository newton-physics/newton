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
    build_foundation_geometry,
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
            cfg = FoundationConfig(stretch_floor=0.05, normal_damping=8.0, friction=3000.0, mu=1.0)
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
            self._add_midsole_visual(builder)
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

    def _add_midsole_visual(self, builder):
        """Attach the static midsole mesh to the world for context."""
        midsole = load_mesh(self.geo.midsole_mesh_path, 0.001)
        verts = np.asarray(midsole.vertices, np.float32).copy()
        verts[:, 2] -= self.geo.z_shift_m
        mesh = newton.Mesh(verts, np.asarray(midsole.faces, np.int32).flatten())
        builder.add_shape_mesh(
            -1,
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
            color=(0.20, 0.35, 0.55),
            label="midsole",
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

    def simulate(self):
        """Advance the foundation and, for the free-body mode, the rigid-body solver by one frame."""
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
            column_world_positions,
            dim=self.column_count,
            inputs=[self.carrier, self.state_0.body_q, self._anchor_local, self._z_free, self._points],
            device=self.device,
        )
        self.viewer.log_points("midsole_columns", self._points, radii=0.0022, colors=self.foundation.compression)
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
        choices=["instron", "settle", "stride"],
        default="instron",
        help="Simulation scenario to run.",
    )
    parser.add_argument("--manifest", type=str, default=MANIFEST, help="Digital Instron manifest path.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
