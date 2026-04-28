# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Plotting
#
# Shows how to access and plot per-step simulation diagnostics from the
# MuJoCo solver. This is useful for debugging solver performance, energy
# conservation, and contact behavior.
#
# The example loads a humanoid model that falls under gravity and collides
# with the ground, collecting per-step metrics:
#   - Solver iteration count (how hard the solver works each step)
#   - Kinetic and potential energy (conservation / dissipation)
#   - Active constraint count (contact events)
#
# These are displayed as live plots in the viewer GUI.
#
# Command: python -m newton.examples basic_plotting --world-count 4
#
###########################################################################

import warnings

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        humanoid = newton.ModelBuilder()

        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")
        humanoid.add_mjcf(
            mjcf_filename,
            ignore_names=["floor", "ground"],
            xform=wp.transform(wp.vec3(0, 0, 1.5)),
        )

        builder = newton.ModelBuilder()
        builder.replicate(humanoid, args.world_count)
        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model)

        # Enable energy computation in MuJoCo (set on whichever model backs the solver)
        try:
            import mujoco  # noqa: PLC0415

            mjm = self.solver.mjw_model if hasattr(self.solver, "mjw_model") else self.solver.mj_model
            mjm.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        except ImportError:
            pass

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.contacts()

        # Per-step diagnostics (lists grow unbounded for interactive use)
        self.log_iterations: list[float] = []
        self.log_energy_kinetic: list[float] = []
        self.log_energy_potential: list[float] = []
        self.log_nefc: list[float] = []

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            try:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
            except Exception as exc:
                self.graph = None
                warnings.warn(f"CUDA graph capture failed: {exc}", stacklevel=2)
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _read_status(self):
        d = self.solver.mjw_data if hasattr(self.solver, "mjw_data") else self.solver.mj_data

        # Solver iterations (max across constraint islands)
        niter_np = d.solver_niter.numpy() if hasattr(d.solver_niter, "numpy") else d.solver_niter
        self.log_iterations.append(float(np.max(niter_np)))

        # Energy: (world_count, 2) → sum across worlds
        energy_np = d.energy.numpy() if hasattr(d.energy, "numpy") else np.asarray(d.energy)
        self.log_energy_kinetic.append(float(energy_np[:, 0].sum()))
        self.log_energy_potential.append(float(energy_np[:, 1].sum()))

        # Active constraint count
        nefc_np = d.nefc.numpy() if hasattr(d.nefc, "numpy") else d.nefc
        self.log_nefc.append(float(np.max(nefc_np)))

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

        self._read_status()

        self.viewer.log_scalar("Solver Iterations", self.log_iterations[-1])
        self.viewer.log_scalar("Kinetic Energy", self.log_energy_kinetic[-1])
        self.viewer.log_scalar("Potential Energy", self.log_energy_potential[-1])
        self.viewer.log_scalar("Active Constraints", self.log_nefc[-1])

    def test_final(self):
        # Verify the humanoid hasn't exploded or fallen through the ground
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "bodies above ground",
            lambda q, qd: q[2] > -0.1,
        )

        self._plot()

    def _plot(self):
        """Save diagnostics plots to a PNG file."""
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            self._print_summary()
            return

        n = len(self.log_iterations)
        time = np.arange(n, dtype=np.float32) * self.frame_dt

        _fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axs[0].step(time, self.log_iterations, color="blue")
        axs[0].set_ylabel("Solver Iterations")
        axs[0].set_title("MuJoCo Simulation Diagnostics")
        axs[0].grid(True)

        axs[1].plot(time, self.log_energy_kinetic, color="red", label="kinetic")
        axs[1].plot(time, self.log_energy_potential, color="blue", label="potential")
        total = np.array(self.log_energy_kinetic) + np.array(self.log_energy_potential)
        axs[1].plot(time, total, color="black", linestyle="--", label="total")
        axs[1].set_ylabel("Energy [J]")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].step(time, self.log_nefc, color="green")
        axs[2].set_ylabel("Active Constraints")
        axs[2].set_xlabel("Time [s]")
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig("solver_convergence.png", dpi=150)
        print("Diagnostics plot saved to solver_convergence.png")
        plt.close()

    def _print_summary(self):
        """Print a text summary of diagnostics data."""
        n = len(self.log_iterations)
        if n == 0:
            print("\nSimulation diagnostics summary: no steps recorded.")
            return
        iters = np.array(self.log_iterations)
        print(f"\nSimulation diagnostics summary ({n} steps):")
        print(f"  Iterations:   mean={np.mean(iters):.1f}, max={np.max(iters):.0f}")
        print(f"  Kinetic E:    final={self.log_energy_kinetic[-1]:.4f}")
        print(f"  Potential E:  final={self.log_energy_potential[-1]:.4f}")
        print(f"  Constraints:  mean={np.mean(self.log_nefc):.1f}, max={np.max(self.log_nefc):.0f}")

    def gui(self, ui):
        n = len(self.log_iterations)
        if n == 0:
            ui.text("Waiting for simulation data...")
            return

        ui.text(f"Step: {n}")
        ui.text(f"Last iters: {int(self.log_iterations[-1])}")
        ui.text(f"Kinetic E: {self.log_energy_kinetic[-1]:.4f}")
        ui.text(f"Potential E: {self.log_energy_potential[-1]:.4f}")
        ui.text(f"Constraints: {int(self.log_nefc[-1])}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
