# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Proxy Coupling Convergence
#
# A synthetic one-dimensional virtual-inertia problem that exercises
# SolverProxyCoupled.Config.iterations in lagged mode. The source solver owns
# one interface particle and maps the previous feedback force to an end
# velocity. The destination solver sees that particle as a proxy with scaled
# virtual inertia, lets the generic lagged prepare step subtract the previously
# applied force through that proxy mass, then returns the scalar KKT response
# force with Delassus denominator ``1/Mhat + 1/Mb``.
#
# Sweeping Config.iterations gives a compact convergence plot for the
# fixed-point error described by the proxy / virtual-inertia section of the
# ADMM coupling note.
#
# Command: python -m newton.examples proxy_coupling_convergence
#
###########################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
from newton.solvers.coupled_experimental import CouplingInterface, SolverProxyCoupled

import newton
import newton.examples
from newton.solvers import SolverBase


@wp.kernel(enable_backward=False)
def _source_velocity_step(
    particle_id: int,
    drive_velocity: float,
    source_mass: float,
    dt: float,
    particle_q_in: wp.array[wp.vec3],
    particle_f_in: wp.array[wp.vec3],
    particle_q_out: wp.array[wp.vec3],
    particle_qd_out: wp.array[wp.vec3],
):
    force = particle_f_in[particle_id][0]
    velocity = drive_velocity + dt * force / source_mass
    q0 = particle_q_in[particle_id]
    particle_q_out[particle_id] = q0 + wp.vec3(dt * velocity, 0.0, 0.0)
    particle_qd_out[particle_id] = wp.vec3(velocity, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def _response_kkt_step(
    proxy_particle_id: int,
    response_particle_id: int,
    target_velocity: float,
    dt: float,
    particle_qd_in: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    proxy_forces: wp.array[wp.vec3],
):
    proxy_velocity = particle_qd_in[proxy_particle_id][0]
    inv_proxy_mass = particle_inv_mass[proxy_particle_id]
    inv_response_mass = particle_inv_mass[response_particle_id]
    force = -(proxy_velocity - target_velocity) / (dt * (inv_proxy_mass + inv_response_mass))
    proxy_forces[proxy_particle_id] = wp.vec3(force, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def _scatter_proxy_particle_forces(
    particle_local_to_proxy_global: wp.array[int],
    proxy_forces: wp.array[wp.vec3],
    out_particle_f: wp.array[wp.vec3],
):
    local_particle = wp.tid()
    global_particle = particle_local_to_proxy_global[local_particle]
    if global_particle >= 0:
        out_particle_f[global_particle] = proxy_forces[local_particle]


@wp.kernel(enable_backward=False)
def _set_display_particles(
    source_particle_id: int,
    response_particle_id: int,
    source_x: float,
    response_x: float,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    source_q = particle_q[source_particle_id]
    response_q = particle_q[response_particle_id]
    particle_q[source_particle_id] = wp.vec3(source_x, source_q[1], source_q[2])
    particle_q[response_particle_id] = wp.vec3(response_x, response_q[1], response_q[2])
    particle_qd[source_particle_id] = wp.vec3(0.0, 0.0, 0.0)
    particle_qd[response_particle_id] = wp.vec3(0.0, 0.0, 0.0)


class _SourceVelocitySolver(SolverBase, CouplingInterface):
    """One-particle source solve for a fixed feedback force estimate."""

    def __init__(self, model, particle_id: int, drive_velocity: float, source_mass: float):
        super().__init__(model)
        self.particle_id = int(particle_id)
        self.drive_velocity = float(drive_velocity)
        self.source_mass = float(source_mass)

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)
        wp.launch(
            _source_velocity_step,
            dim=1,
            inputs=[
                self.particle_id,
                self.drive_velocity,
                self.source_mass,
                dt,
                state_in.particle_q,
                state_in.particle_f,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=self.model.device,
        )


class _KktResponseSolver(SolverBase, CouplingInterface):
    """Destination solver that returns the scalar proxy KKT force."""

    def __init__(self, model, proxy_particle_id: int, response_particle_id: int, target_velocity: float):
        super().__init__(model)
        self.proxy_particle_id = int(proxy_particle_id)
        self.response_particle_id = int(response_particle_id)
        self.target_velocity = float(target_velocity)
        self.proxy_forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global,
        out_particle_f,
        *,
        state=None,
        state_out=None,
        contacts=None,
        dt=0.0,
    ):
        del state, state_out, contacts, dt
        wp.launch(
            _scatter_proxy_particle_forces,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[particle_local_to_proxy_global, self.proxy_forces],
            outputs=[out_particle_f],
            device=self.model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)
        self.proxy_forces.zero_()
        wp.launch(
            _response_kkt_step,
            dim=1,
            inputs=[
                self.proxy_particle_id,
                self.response_particle_id,
                self.target_velocity,
                dt,
                state_in.particle_qd,
                self.model.particle_inv_mass,
            ],
            outputs=[self.proxy_forces],
            device=self.model.device,
        )


@dataclass
class _SweepResult:
    mass_scale: float
    contraction: float
    fixed_velocity: float
    velocities: np.ndarray
    errors: np.ndarray


def _proxy_fixed_point_factor(mass_scale, mass_ratio):
    """Scalar proxy fixed-point factor for ``Mhat = mass_scale * Ma``."""
    return (1.0 / mass_scale - 1.0) / (1.0 / mass_scale + mass_ratio)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_index = 0
        self.plot_path = args.plot_path
        self.factor_map_path = args.factor_map_path
        self.plot_dpi = args.plot_dpi
        self.max_coupling_iterations = args.max_coupling_iterations
        self.source_mass = args.source_mass
        self.response_mass = args.response_mass
        self.drive_velocity = args.drive_velocity
        self.response_target_velocity = args.response_target_velocity

        if self.max_coupling_iterations < 1:
            raise ValueError("--max-coupling-iterations must be >= 1")
        if self.source_mass <= 0.0:
            raise ValueError("--source-mass must be > 0")
        if self.response_mass <= 0.0:
            raise ValueError("--response-mass must be > 0")
        if len(args.proxy_mass_scale) == 0:
            raise ValueError("--proxy-mass-scale must include at least one value")
        if any(value <= 0.0 for value in args.proxy_mass_scale):
            raise ValueError("--proxy-mass-scale values must be > 0")

        builder = newton.ModelBuilder(gravity=0.0)
        self.source_particle = builder.add_particle(
            pos=wp.vec3(self.drive_velocity, -0.08, 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=self.source_mass,
            radius=0.035,
        )
        self.response_particle = builder.add_particle(
            pos=wp.vec3(self.response_target_velocity, 0.08, 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=self.response_mass,
            radius=0.035,
        )
        builder.color()

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.iteration_counts = np.arange(1, self.max_coupling_iterations + 1, dtype=np.int32)
        self.results = [self._sweep_mass_scale(float(value)) for value in args.proxy_mass_scale]

        stable_results = [result for result in self.results if result.contraction < 1.0]
        self.display_result = stable_results[0] if stable_results else self.results[0]
        self._update_display_state(0)

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True
        self.viewer.set_camera(pos=wp.vec3(0.0, -1.15, 0.55), pitch=-20.0, yaw=0.0)

    def _sweep_mass_scale(self, mass_scale: float) -> _SweepResult:
        mass_ratio = self.source_mass / self.response_mass
        contraction = abs(_proxy_fixed_point_factor(mass_scale, mass_ratio))

        velocity_gap = self.drive_velocity - self.response_target_velocity
        fixed_force = -velocity_gap / (self.frame_dt * (1.0 / self.source_mass + 1.0 / self.response_mass))
        fixed_velocity = self.drive_velocity + self.frame_dt * fixed_force / self.source_mass

        velocities = []
        errors = []
        for iterations in self.iteration_counts:
            velocity = self._run_case(int(iterations), mass_scale)
            velocities.append(velocity)
            errors.append(abs(velocity - fixed_velocity))

        return _SweepResult(
            mass_scale=mass_scale,
            contraction=contraction,
            fixed_velocity=fixed_velocity,
            velocities=np.array(velocities, dtype=np.float64),
            errors=np.array(errors, dtype=np.float64),
        )

    def _run_case(self, iterations: int, mass_scale: float) -> float:
        solver = SolverProxyCoupled(
            model=self.model,
            entries=[
                SolverProxyCoupled.Entry(
                    name="source",
                    solver=lambda v: _SourceVelocitySolver(
                        model=v,
                        **{
                            "particle_id": self.source_particle,
                            "drive_velocity": self.drive_velocity,
                            "source_mass": self.source_mass,
                        },
                    ),
                    particles=[self.source_particle],
                ),
                SolverProxyCoupled.Entry(
                    name="response",
                    solver=lambda v: _KktResponseSolver(
                        model=v,
                        **{
                            "proxy_particle_id": self.source_particle,
                            "response_particle_id": self.response_particle,
                            "target_velocity": self.response_target_velocity,
                        },
                    ),
                    particles=[self.response_particle],
                ),
            ],
            coupling=SolverProxyCoupled.Config(
                proxies=[
                    SolverProxyCoupled.Proxy(
                        source="source",
                        destination="response",
                        particles=[self.source_particle],
                        mass_scale=mass_scale,
                        mode="lagged",
                    ),
                ],
                iterations=iterations,
            ),
        )

        state_in = self.model.state()
        state_out = self.model.state()
        state_in.clear_forces()
        solver.step(state_in, state_out, self.control, self.contacts, self.frame_dt)
        return float(state_out.particle_qd.numpy()[self.source_particle, 0])

    def _update_display_state(self, sample_index: int):
        sample_index = min(sample_index, len(self.display_result.velocities) - 1)
        source_x = float(self.display_result.velocities[sample_index])
        wp.launch(
            _set_display_particles,
            dim=1,
            inputs=[
                self.source_particle,
                self.response_particle,
                source_x,
                self.response_target_velocity,
                self.state_0.particle_q,
                self.state_0.particle_qd,
            ],
            device=self.model.device,
        )

    def step(self):
        sample_index = min(self.frame_index, len(self.iteration_counts) - 1)
        self._update_display_state(sample_index)

        for result in self.results:
            self.viewer.log_scalar(
                f"velocity error mass_scale={result.mass_scale:g}",
                float(result.errors[sample_index]),
            )

        self.frame_index += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        for result in self.results:
            assert np.isfinite(result.velocities).all(), "Proxy convergence velocities contain NaN or inf values"
            assert np.isfinite(result.errors).all(), "Proxy convergence errors contain NaN or inf values"
            if result.contraction < 1.0:
                assert result.errors[-1] < result.errors[0], (
                    f"Proxy mass scale {result.mass_scale:g} did not reduce interface velocity error"
                )
        self._plot()

    def _plot(self):
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
            from matplotlib.colors import TwoSlopeNorm  # noqa: PLC0415
        except ImportError:
            self._print_summary()
            return

        _fig, ax = plt.subplots(figsize=(4.0, 4.0))
        for result in self.results:
            label = f"mhat/m={result.mass_scale:g}, |r|={result.contraction:.2f}"
            ax.semilogy(
                self.iteration_counts,
                np.maximum(result.errors, 1.0e-12),
                marker="o",
                linewidth=1.6,
                label=label,
            )

        ax.set_xlabel("Proxy iterations")
        ax.set_ylabel("Interface velocity error [m/s]")
        ax.set_title("Virtual Inertia Proxy Convergence")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=self.plot_dpi)
        plt.close()
        print(f"Proxy convergence plot saved to {self.plot_path}")

        mass_scale = np.geomspace(0.02, 10.0, 360)
        mass_ratio = np.geomspace(0.02, 50.0, 360)
        scale_grid, ratio_grid = np.meshgrid(mass_scale, mass_ratio)
        factor = _proxy_fixed_point_factor(scale_grid, ratio_grid)

        _fig, ax = plt.subplots(figsize=(5.8, 4.2))
        clipped_factor = np.clip(factor, -1.5, 1.5)
        image = ax.pcolormesh(
            scale_grid,
            ratio_grid,
            clipped_factor,
            shading="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.5, vmax=1.5),
        )
        ax.contour(scale_grid, ratio_grid, np.abs(factor), levels=[1.0], colors="black", linewidths=1.4)
        ax.contour(scale_grid, ratio_grid, factor, levels=[0.0], colors="white", linewidths=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("virtual inertia scale s = Mhat / Ma")
        ax.set_ylabel("mass ratio Ma / Mb")
        ax.set_title("Scalar Proxy Fixed-Point Factor")
        cbar = plt.colorbar(image, ax=ax)
        cbar.set_label("signed r")
        ax.text(
            0.03,
            0.03,
            "black: |r| = 1\nwhite: r = 0",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        plt.tight_layout()
        plt.savefig(self.factor_map_path, dpi=self.plot_dpi)
        plt.close()
        print(f"Proxy fixed-point factor map saved to {self.factor_map_path}")

    def _print_summary(self):
        print("\nProxy convergence summary:")
        for result in self.results:
            print(f"  mhat/m={result.mass_scale:g}, |r|={result.contraction:.4f}, final_error={result.errors[-1]:.4e}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--max-coupling-iterations",
            type=int,
            default=12,
            help="Largest Config.iterations value to include in the sweep.",
        )
        parser.add_argument(
            "--proxy-mass-scale",
            type=float,
            nargs="+",
            default=[0.25, 0.5, 1.0, 2.0],
            help="Virtual inertia scale values applied to the destination proxy mass.",
        )
        parser.add_argument(
            "--source-mass",
            type=float,
            default=1.0,
            help="Source particle mass [kg].",
        )
        parser.add_argument(
            "--drive-velocity",
            type=float,
            default=1.0,
            help="Uncoupled source velocity [m/s].",
        )
        parser.add_argument(
            "--response-target-velocity",
            type=float,
            default=0.0,
            help="Velocity preferred by the response side [m/s].",
        )
        parser.add_argument(
            "--response-mass",
            type=float,
            default=1.0,
            help="Destination-side scalar mass Mb [kg].",
        )
        parser.add_argument(
            "--plot-path",
            type=str,
            default="proxy_coupling_convergence.png",
            help="Path for the saved convergence plot in test mode.",
        )
        parser.add_argument(
            "--factor-map-path",
            type=str,
            default="proxy_coupling_factor_map.png",
            help="Path for the saved scalar fixed-point factor map in test mode.",
        )
        parser.add_argument(
            "--plot-dpi",
            type=int,
            default=150,
            help="DPI for the saved convergence plot.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
