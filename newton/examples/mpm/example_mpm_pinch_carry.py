# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grab, lift, carry, and release a procedural soft cube or sphere.

GIMP uses Q1 velocity, P1d strain, and pic27 contacts by default. Two kinematic
pads close, lift the object 50 mm, carry it 100 mm, then lower it to the ground
before opening. Test mode checks grip retention and release on the ground.
No external assets are required.

Run ``--viewer null --num-frames 480 --test`` for the full eight-second
sequence at the default 60 frames per second.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

# Pad half extents [m]: 18 mm wide (x), 10 mm thick (y, the squeeze axis), 18 mm tall (z).
PAD_HALF_EXTENTS = (0.009, 0.005, 0.009)

# Scripted gripper motion: (phase name, duration [s], gap [fraction of the object width], lift [m], carry [m]).
# Each phase moves linearly from the previous phase's end values to its own.
PHASES = (
    ("settle", 0.5, 1.4, 0.0, 0.0),
    ("close", 1.0, 0.7, 0.0, 0.0),
    ("hold", 0.5, 0.7, 0.0, 0.0),
    ("lift", 1.0, 0.7, 0.05, 0.0),
    ("hold high", 0.5, 0.7, 0.05, 0.0),
    ("carry", 2.0, 0.7, 0.05, 0.1),
    ("hold carried", 0.5, 0.7, 0.05, 0.1),
    ("lower", 1.0, 0.7, 0.0, 0.1),
    ("release", 0.5, 1.4, 0.0, 0.1),
    ("settle released", 0.5, 1.4, 0.0, 0.1),
)


@wp.kernel
def advance_time(time: wp.array[float], dt: float):
    time[0] = time[0] + dt


@wp.kernel
def move_pads(
    time: wp.array[float],
    phase_end: wp.array[float],
    gap: wp.array[float],
    lift: wp.array[float],
    carry: wp.array[float],
    center: wp.vec3,
    carry_axis: wp.vec3,
    half_thickness: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Pose both pads on the piecewise-linear script; pad 0 sits at -y, pad 1 at +y."""
    pad = wp.tid()
    t = time[0]
    k = int(0)
    while k < phase_end.shape[0] - 1 and t > phase_end[k]:
        k += 1
    start = float(0.0)
    if k > 0:
        start = phase_end[k - 1]
    duration = phase_end[k] - start
    u = wp.clamp((t - start) / duration, 0.0, 1.0)
    gap_k = wp.lerp(gap[k], gap[k + 1], u)
    lift_k = wp.lerp(lift[k], lift[k + 1], u)
    carry_k = wp.lerp(carry[k], carry[k + 1], u)
    rate = wp.where(t < phase_end[phase_end.shape[0] - 1], 1.0 / duration, 0.0)
    side = wp.where(pad == 0, -1.0, 1.0)
    offset = wp.vec3(0.0, side * (0.5 * gap_k + half_thickness), lift_k)
    body_q[pad] = wp.transform(center + offset + carry_k * carry_axis, wp.quat_identity())
    velocity = wp.vec3(0.0, side * 0.5 * (gap[k + 1] - gap[k]), lift[k + 1] - lift[k])
    velocity = (velocity + (carry[k + 1] - carry[k]) * carry_axis) * rate
    body_qd[pad] = wp.spatial_vector(velocity, wp.vec3(0.0))


class Example:
    """Grab, lift, carry, and release a procedural object with SolverImplicitMPM."""

    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.pad_lookahead = args.pad_lookahead
        self.project_outside = args.project_outside
        self.verbose = args.verbose
        self.frame = 0
        self.viewer = viewer

        particles, spacing = self.create_particles(args)
        volume = spacing**3
        lower, upper = particles.min(0), particles.max(0)
        self.width = float(upper[1] - lower[1])
        center = 0.5 * (lower + upper)
        self.initial_center = particles.mean(axis=0)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, args.gravity))
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_ground_plane()
        radius = 0.5 * spacing
        for p in particles:
            builder.add_particle(pos=wp.vec3(*p), vel=wp.vec3(0.0), mass=args.density * volume, radius=radius)
        pad_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.pad_friction)
        self.pads = []
        for side in ("left", "right"):
            body = builder.add_body(is_kinematic=True, label=f"pad_{side}")
            builder.add_shape_box(
                body, hx=PAD_HALF_EXTENTS[0], hy=PAD_HALF_EXTENTS[1], hz=PAD_HALF_EXTENTS[2], cfg=pad_cfg
            )
            self.pads.append(body)
        self.model = builder.finalize()

        mpm = self.model.mpm
        mpm.young_modulus.fill_(args.young_modulus)
        mpm.poisson_ratio.fill_(args.poisson_ratio)
        mpm.friction.fill_(args.material_friction)
        mpm.yield_stress.fill_(args.yield_ratio * args.young_modulus)
        mpm.yield_pressure.fill_(args.yield_pressure_ratio * args.young_modulus)
        mpm.tensile_yield_ratio.fill_(1.0)

        config = SolverImplicitMPM.Config()
        config.voxel_size = args.voxel_size if args.voxel_size > 0.0 else 2.0 * spacing
        config.grid_type = "sparse"
        config.grid_padding = 0
        config.max_active_cell_count = args.max_active_cells
        config.strain_basis = args.strain_basis
        config.transfer_scheme = args.transfer_scheme
        config.integration_scheme = args.integration_scheme
        config.max_iterations = args.max_iterations
        config.tolerance = args.tolerance
        config.velocity_basis = args.velocity_basis
        config.collider_basis = args.collider_basis
        config.collider_velocity_mode = args.collider_velocity_mode
        config.solver = args.solver
        self.solver = SolverImplicitMPM(self.model, config)
        if args.pad_adhesion > 0.0:
            # Ground plane (body -1) keeps default material; pads get adhesion [Pa].
            self.solver.setup_collider(
                collider_body_ids=[-1, *self.pads], collider_adhesion=[0.0] + [args.pad_adhesion] * len(self.pads)
            )

        # The gripper closes on the object's middle, across its y width, and carries it along --carry-axis.
        axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0)}[args.carry_axis]
        self.pad_center = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
        self.carry_axis = wp.vec3(*axis)
        ends = np.cumsum([duration for _, duration, *_ in PHASES])
        self.phase_names = [name for name, *_ in PHASES]
        self.phase_end = wp.array(ends, dtype=float)
        self.gap = wp.array([PHASES[0][2] * self.width] + [g * self.width for *_, g, _, _ in PHASES], dtype=float)
        self.lift = wp.array([0.0] + [lift for *_, lift, _ in PHASES], dtype=float)
        self.carry = wp.array([0.0] + [carry for *_, carry in PHASES], dtype=float)
        self.time = wp.zeros(1, dtype=float)
        self.duration = float(ends[-1])

        self.state = self.model.state()
        self.move_pads(self.state)
        self.solver.reset(self.state)
        self.records: list[dict] = []

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(0.05, -0.25, 0.12), pitch=-20.0, yaw=90.0)

        self.graph = None
        if wp.get_device().is_cuda and not args.no_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def move_pads(self, state):
        wp.launch(
            move_pads,
            dim=2,
            inputs=[
                self.time,
                self.phase_end,
                self.gap,
                self.lift,
                self.carry,
                self.pad_center,
                self.carry_axis,
                PAD_HALF_EXTENTS[1],
            ],
            outputs=[state.body_q, state.body_qd],
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            if self.pad_lookahead:
                wp.launch(advance_time, dim=1, inputs=[self.time, self.sim_dt])
            self.move_pads(self.state)
            self.solver.step(self.state, self.state, None, None, self.sim_dt)
            if self.project_outside:
                self.solver.project_outside(self.state, self.state, self.sim_dt)
            if not self.pad_lookahead:
                wp.launch(advance_time, dim=1, inputs=[self.time, self.sim_dt])

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.solver.check_sparse_grid_rebuild_status()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self.frame % max(1, round(0.1 * self.fps)) == 0:
            self.record()

    def phase(self, t: float) -> str:
        ends = self.phase_end.numpy()
        return self.phase_names[min(int(np.searchsorted(ends, t)), len(ends) - 1)]

    def record(self):
        """Measure the grip and print one line of the log."""
        q = self.state.particle_q.numpy()
        pads = self.state.body_q.numpy()[self.pads, :3]
        impulses, _, colliders = self.solver.collect_collider_impulses(self.state)
        body = self.solver.collider_body_index.numpy()[np.maximum(colliders.numpy(), 0)]
        on_pads = (colliders.numpy() >= 0) & np.isin(body, self.pads)
        squeeze = 0.5 * np.abs(impulses.numpy()[on_pads, 1]).sum() / self.sim_dt

        # Particles within 3 mm of either pad's inner face.
        inner = (pads[0, 1] + PAD_HALF_EXTENTS[1], pads[1, 1] - PAD_HALF_EXTENTS[1])
        near = (np.abs(q[:, 1] - inner[0]) < 0.003) | (np.abs(q[:, 1] - inner[1]) < 0.003)
        stress = self.state.mpm.particle_stress.numpy()[near, 1, 1]
        elastic = self.state.mpm.particle_elastic_strain.numpy()[near, 1, 1] - 1.0
        row = {
            "time": self.sim_time,
            "phase": self.phase(self.sim_time),
            "object_height": float(q[:, 2].mean()),
            "travel": float(np.dot(q.mean(axis=0) - self.initial_center, np.asarray(self.carry_axis))),
            "relative_height": float(q[:, 2].mean() - pads[:, 2].mean()),
            "squeeze": float(squeeze),
            "stress": float(stress.mean()) if len(stress) else 0.0,
            "compression": float(elastic.mean()) if len(elastic) else 0.0,
        }
        self.records.append(row)
        if self.verbose:
            print(
                f"t={row['time']:5.2f} s  {row['phase']:<12s}  object z={1000 * row['object_height']:6.1f} mm  "
                f"vs pads {1000 * row['relative_height']:+6.1f} mm  squeeze {row['squeeze']:7.3f} N  "
                f"sigma_yy {row['stress']:+8.1f} Pa  Fe_yy-1 {row['compression']:+.3f}",
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self):
        """Check finite particle states, grip retention, lift, carry, and release."""
        newton.examples.test_particle_state(
            self.state, "all particles have finite positions", lambda q, qd: wp.length(q) < 1.0e3
        )
        newton.examples.test_particle_state(
            self.state, "all particles have finite velocities", lambda q, qd: wp.length(qd) < 1.0e3
        )
        if self.sim_time + 0.5 * self.frame_dt < self.duration:
            return
        held = [r for r in self.records if r["phase"] in ("hold high", "carry", "hold carried", "lower")]
        if not held or min(r["relative_height"] for r in held) < -0.01:
            raise AssertionError("The object dropped before release")
        carried = [r for r in self.records if r["phase"] == "hold carried"][-1]
        if carried["object_height"] - self.initial_center[2] < 0.035:
            raise AssertionError("The object did not reach the 50 mm lift")
        if carried["travel"] < 0.08:
            raise AssertionError("The object did not follow the 100 mm carry")
        lowered = [r for r in self.records if r["phase"] == "lower"][-1]
        if lowered["object_height"] > self.initial_center[2] + 0.003:
            raise AssertionError("The object did not reach the ground before release")
        if self.records[-1]["squeeze"] > 0.01 * max(r["squeeze"] for r in held):
            raise AssertionError("The pads still grip the object after opening")

    def summarize(self):
        """Print the grip at the end of each phase and whether the object was carried."""
        if not self.verbose or not self.records:
            return
        print("\nphase          end [s]  object vs pads [mm]  squeeze [N]  sigma_yy [Pa]  Fe_yy-1")
        for name in self.phase_names:
            rows = [r for r in self.records if r["phase"] == name]
            if rows:
                r = rows[-1]
                print(
                    f"{name:<12s}  {r['time']:7.2f}  {1000 * r['relative_height']:+18.1f}  {r['squeeze']:11.3f}  "
                    f"{r['stress']:+13.1f}  {r['compression']:+.3f}"
                )
        lifted = [r for r in self.records if r["phase"] in ("hold high", "carry", "hold carried")]
        dropped = next((r for r in lifted if r["relative_height"] < -0.01), None)
        if dropped is None:
            print("\nResult: the object stayed gripped until release.")
        else:
            print(f"\nResult: the object dropped out of the grip at t={dropped['time']:.2f} s ({dropped['phase']}).")

    @staticmethod
    def create_particles(args):
        """Sample a cube or sphere on a deterministic lattice above the ground."""
        if args.size <= 0.0 or args.spacing <= 0.0:
            raise ValueError("Object size and particle spacing must be positive")
        count = max(2, round(args.size / args.spacing))
        spacing = args.size / count
        axis = (np.arange(count, dtype=np.float32) + 0.5) * spacing - 0.5 * args.size
        points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        if args.shape == "sphere":
            points = points[np.linalg.norm(points, axis=1) <= 0.5 * args.size]
        points[:, 2] += max(0.5 * args.size, PAD_HALF_EXTENTS[2]) + 0.5 * spacing
        return points.astype(np.float32), spacing

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--verbose", action="store_true", help="Print simulation diagnostics and phase summary.")
        parser.add_argument("--shape", choices=["cube", "sphere"], default="cube")
        parser.add_argument("--size", type=float, default=0.016, help="Cube side length or sphere diameter [m].")
        parser.add_argument("--spacing", type=float, default=0.001, help="Target particle spacing [m].")
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=2, help="Solver steps per frame (120 Hz by default).")
        parser.add_argument("--carry-axis", choices=["x", "y"], default="y", help="y is the squeeze axis.")
        parser.add_argument("--pad-friction", type=float, default=1.0)
        parser.add_argument(
            "--project-outside", action="store_true", help="Project particles out of the pads each step."
        )
        parser.add_argument("--velocity-basis", type=str, default="Q1")
        parser.add_argument("--no-graph", action="store_true", help="Run substeps without CUDA graph capture.")
        parser.add_argument("--gravity", type=float, default=-9.81, help="Vertical gravity [m/s^2].")
        parser.add_argument("--transfer-scheme", choices=["apic", "pic"], default="apic")
        parser.add_argument("--integration-scheme", choices=["pic", "gimp"], default="gimp")
        parser.add_argument("--pad-lookahead", action="store_true", help="Pose pads at the end of each step.")
        parser.add_argument("--collider-velocity-mode", choices=["forward", "backward"], default="forward")
        parser.add_argument("--pad-adhesion", type=float, default=0.0, help="Pad adhesion [Pa].")
        parser.add_argument("--material-friction", type=float, default=0.35)
        parser.add_argument("--density", type=float, default=900.0)
        parser.add_argument("--young-modulus", "-ym", type=float, default=12000.0)
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.4)
        parser.add_argument("--yield-ratio", type=float, default=0.5, help="Shear yield stress over Young's modulus.")
        parser.add_argument("--yield-pressure-ratio", type=float, default=5.0)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.0, help="0 picks twice the particles spacing.")
        parser.add_argument("--max-active-cells", type=int, default=4096)
        parser.add_argument("--strain-basis", "-sb", type=str, default="P1d")
        parser.add_argument("--collider-basis", type=str, default="pic27")
        parser.add_argument("--max-iterations", "-it", type=int, default=50)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
        parser.add_argument("--solver", "-s", type=str, default="gs")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.summarize()
