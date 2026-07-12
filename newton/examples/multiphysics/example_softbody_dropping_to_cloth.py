# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Dropping to Cloth
#
# This simulation demonstrates a volumetric soft body (tetrahedral grid)
# dropping onto a cloth sheet. The soft body uses Neo-Hookean elasticity
# and deforms on impact with the cloth.
#
# Command: python -m newton.examples.multiphysics.example_softbody_dropping_to_cloth
#
###########################################################################

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import ModelView, SolverCoupledProxy

import newton
import newton.examples
from newton.solvers import SolverVBD


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        if self.solver_type not in {"vbd", "coupled"}:
            raise ValueError("The softbody dropping to cloth example supports the vbd and coupled solvers.")

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        cloth_particle_start = builder.particle_count
        builder.add_cloth_grid(
            pos=wp.vec3(-1.0, -1.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            fix_left=True,
            fix_right=True,
            dim_x=40,
            dim_y=40,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.0005,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e0,
            edge_ke=0.01,
            edge_kd=1e-4,
            particle_radius=0.05,
        )
        self.cloth_particles = list(range(cloth_particle_start, builder.particle_count))

        # Add soft body (tetrahedral grid) at elevated position
        soft_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=6,
            dim_y=6,
            dim_z=3,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1.0e2,
            particle_radius=0.03,
        )
        self.soft_particles = list(range(soft_particle_start, builder.particle_count))

        # Color the mesh for VBD solver
        builder.color()

        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1e0
        self.model.soft_contact_mu = 1.0

        vbd_kwargs = {
            "iterations": args.vbd_iterations,
            "particle_enable_self_contact": True,
            "particle_self_contact_radius": 0.01,
            "particle_self_contact_margin": 0.02,
            "particle_enable_tile_solve": True,
        }
        if self.solver_type == "vbd":
            self.solver = SolverVBD(model=self.model, **vbd_kwargs)
        else:

            def configure_soft_view(view: ModelView) -> None:
                view.tri_count = 0
                view.edge_count = 0

            def configure_cloth_view(view: ModelView) -> None:
                view.tet_count = 0

            soft_vbd_kwargs = {**vbd_kwargs, "particle_enable_self_contact": False}

            self.solver = SolverCoupledProxy(
                model=self.model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name="soft",
                        solver=lambda v: SolverVBD(model=v, **soft_vbd_kwargs),
                        particles=self.soft_particles,
                        configure_view=configure_soft_view,
                    ),
                    SolverCoupledProxy.Entry(
                        name="cloth",
                        solver=lambda v: SolverVBD(model=v, **vbd_kwargs),
                        particles=self.cloth_particles,
                        configure_view=configure_cloth_view,
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="soft",
                            destination="cloth",
                            particles=self.soft_particles,
                            mass_scale=args.mass_scale,
                            mode=args.coupling_mode,
                        )
                    ],
                    iterations=args.proxy_iterations,
                ),
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.contacts()

        newton.examples.configure_coupled_view(self, args)

        self.capture()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Test that bounding box size is reasonable (not exploding)
        particle_q = self.state_0.particle_q.numpy()
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)

        # Check bbox size is reasonable (cloth stretches as soft body deforms it)
        assert bbox_size < 20.0, f"Bounding box exploded: size={bbox_size:.2f}"

        # Check no excessive penetration
        assert min_pos[2] > -0.5, f"Excessive penetration: z_min={min_pos[2]:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument(
            "--solver",
            help="Type of solver",
            type=str,
            choices=["vbd", "coupled"],
            default="vbd",
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
            help="Scale factor for source effective particle mass used by VBD proxies",
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
            "--vbd-iterations",
            help="VBD solver iterations per substep",
            type=int,
            default=10,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
