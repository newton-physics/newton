# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example XPBD-VBD Coupled Solver
#
# A VBD cloth patch pushes into an XPBD particle bed through SolverCoupledProxy's
# proxy-particle path. XPBD resolves particle-particle contact against the VBD
# proxy particles, and the harvested proxy impulse is applied back to the VBD
# particles on the next coupled step.
#
# Pass ``--solver xpbd`` to run a monolithic XPBD reference, or``--solver vbd``
# to run the same scene with VBD only.
# Those baselines are provided for comparison purposes, it is expected that the
# behavior will differ as they support different features and/or parameters.
#
# Command: python -m newton.examples xpbd_vbd_coupled_solver
#          python -m newton.examples xpbd_vbd_coupled_solver --solver xpbd
#          python -m newton.examples xpbd_vbd_coupled_solver --solver vbd
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import ModelView, SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverVBD, SolverXPBD


def _configure_contact_only_xpbd_view(view: ModelView) -> None:
    # XPBD should treat VBD-owned cloth particles as collision proxies only.
    # The shared model still contains the cloth's springs/edges/triangles for
    # VBD and rendering; stripping XPBD's elastic topology keeps the secondary
    # solver contact-only for the source cloth particles.
    view.spring_count = 0
    view.tri_count = 0
    view.edge_count = 0
    view.tet_count = 0


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_type = args.solver

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        self.vbd_particles = self._emit_vbd_cloth(builder, args)
        self.xpbd_particles = self._emit_xpbd_particle_bed(builder)

        builder.color()

        self.model = builder.finalize()
        self.model.particle_mu = 0.6
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 1.0e-2 * self.model.soft_contact_ke
        self.model.soft_contact_mu = 0.6

        xpbd_kwargs = {
            "iterations": args.xpbd_iterations,
            "soft_contact_relaxation": args.xpbd_contact_relaxation,
        }
        vbd_kwargs = {
            "iterations": args.vbd_iterations,
            "particle_enable_tile_solve": True,
            "particle_enable_self_contact": False,
        }

        if self.solver_type == "coupled":
            self.solver = SolverCoupledProxy(
                model=self.model,
                entries=[
                    # VBD is the primary/source solver. It owns the cloth and
                    # receives only the harvested XPBD particle response.
                    SolverCoupledProxy.Entry(
                        name="vbd",
                        solver=lambda v: SolverVBD(model=v, **vbd_kwargs),
                        particles=self.vbd_particles,
                    ),
                    # XPBD is the secondary/destination solver. It sees the
                    # VBD particles as proxies, handles particle-particle
                    # collision, and exposes the proxy momentum change.
                    SolverCoupledProxy.Entry(
                        name="xpbd",
                        solver=lambda v: SolverXPBD(model=v, **xpbd_kwargs),
                        particles=self.xpbd_particles,
                        configure_view=_configure_contact_only_xpbd_view,
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="vbd",
                            destination="xpbd",
                            particles=self.vbd_particles,
                            mass_scale=args.mass_scale,
                            mode=args.coupling_mode,
                        ),
                    ],
                    iterations=args.proxy_iterations,
                ),
            )
        elif self.solver_type == "xpbd":
            self.solver = SolverXPBD(model=self._contact_only_xpbd_view(), **xpbd_kwargs)
        elif self.solver_type == "vbd":
            self.solver = SolverVBD(model=self.model, **vbd_kwargs)
        else:
            raise ValueError(f"Unknown solver {self.solver_type!r}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all(), "Particle positions contain NaN or inf values"

        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 10.0, f"Bounding box exploded: size={bbox_size:.2f}"
        assert min_pos[2] > -0.25, f"Excessive ground penetration: z_min={min_pos[2]:.4f}"
        assert max_pos[2] < 3.0, f"Particles escaped upward: z_max={max_pos[2]:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def _contact_only_xpbd_view(self) -> ModelView:
        view = ModelView(self.model, "xpbd")
        _configure_contact_only_xpbd_view(view)
        return view

    def _emit_vbd_cloth(self, builder: newton.ModelBuilder, args) -> list[int]:
        particle_start = builder.particle_count
        cloth_damping = args.cloth_damping
        builder.add_cloth_grid(
            pos=wp.vec3(-0.35, -0.35, 0.38),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            fix_left=True,
            fix_right=True,
            dim_x=14,
            dim_y=14,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.02,
            add_springs=True,
            spring_ke=args.cloth_stiffness,
            spring_kd=cloth_damping * args.cloth_stiffness,
            tri_ke=args.cloth_stiffness,
            tri_ka=args.cloth_stiffness,
            tri_kd=cloth_damping * 0.05 * args.cloth_stiffness,
            edge_ke=args.cloth_bending,
            edge_kd=cloth_damping * 0.05 * args.cloth_bending,
            particle_radius=0.025,
        )
        return list(range(particle_start, builder.particle_count))

    def _emit_xpbd_particle_bed(self, builder: newton.ModelBuilder) -> list[int]:
        particle_start = builder.particle_count
        builder.add_particle_grid(
            pos=wp.vec3(-0.12, -0.12, 0.66),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=5,
            dim_y=5,
            dim_z=3,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            mass=0.02,
            jitter=0.002,
            radius_mean=0.025,
        )
        return list(range(particle_start, builder.particle_count))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument(
            "--solver",
            "-s",
            help="'coupled' for VBD+XPBD coupling, or a single-solver baseline",
            type=str,
            choices=["coupled", "xpbd", "vbd"],
            default="coupled",
        )
        parser.add_argument(
            "--coupling-mode",
            help="Proxy particle state transfer mode",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
        )
        parser.add_argument(
            "--mass-scale",
            "-pmr",
            help="Scale factor for VBD effective particle mass used by XPBD proxies",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--proxy-iterations",
            help="Number of proxy relaxation passes per substep",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--xpbd-contact-relaxation",
            help="XPBD particle contact relaxation (< 1 = softer contact response)",
            type=float,
            default=0.85,
        )
        parser.add_argument(
            "--cloth-stiffness",
            help="VBD cloth stretch/shear stiffness",
            type=float,
            default=150.0,
        )
        parser.add_argument(
            "--cloth-bending",
            help="VBD cloth bending stiffness",
            type=float,
            default=0.01,
        )
        parser.add_argument(
            "--cloth-damping",
            help="VBD cloth damping multiplier converted to absolute damping coefficients",
            type=float,
            default=0.005,
        )
        parser.add_argument(
            "--xpbd-iterations",
            help="XPBD solver iterations per substep",
            type=int,
            default=16,
        )
        parser.add_argument(
            "--vbd-iterations",
            help="VBD solver iterations per substep",
            type=int,
            default=8,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
