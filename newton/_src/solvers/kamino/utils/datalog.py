# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for simulation data logging and plotting."""

import os

import numpy as np

from ..core.builder import ModelBuilder
from ..simulation.simulator import Simulator
from ..utils import logger as msg
from ..utils.control import JointSpacePIDController

###
# Module interface
###

__all__ = [
    "SimulationLogger",
]


###
# Interfaces
###


class SimulationLogger:
    """
    TODO
    """

    plt = None
    """Class-level variable to hold the imported module"""

    @classmethod
    def initialize_plt(cls):
        """TODO"""
        if cls.plt is None:  # Only import if not already imported
            # Attempt to import matplotlib for plotting
            try:
                import matplotlib.pyplot as plt  # noqa: PLC0415

                cls.plt = plt
            except ImportError:
                return  # matplotlib is not available so we skip plotting

    def __init__(
        self,
        max_frames: int,
        sim: Simulator,
        builder: ModelBuilder,
        controller: JointSpacePIDController | None = None,
    ):
        """
        TODO
        """
        # Check if the simulation builder, and controller instances are valid
        if not isinstance(sim, Simulator):
            raise TypeError("'simulator' must be an instance of `Simulator`.")
        if not isinstance(builder, ModelBuilder):
            raise TypeError("'builder' must be an instance of `ModelBuilder`.")
        if controller is not None:
            if not isinstance(controller, JointSpacePIDController):
                raise TypeError("'controller' must be an instance of `JointSpacePIDController` or `None`.")

        # Warn if multiple worlds are present
        if sim.model.size.num_worlds > 1:
            msg.warning("SimulationLogger currently only records data from the first world.")

        # Attempt to initialize matplotlib for plotting
        self.initialize_plt()

        # Initialize internals
        self._frames: int = 0
        self._max_frames: int = max_frames
        self._sim: Simulator = sim
        self._builder: ModelBuilder = builder
        self._ctrl: JointSpacePIDController | None = controller

        # Allocate logging arrays for solver convergence info
        self.log_num_limits = np.zeros(self._max_frames, dtype=np.int32)
        self.log_num_contacts = np.zeros(self._max_frames, dtype=np.int32)
        self.log_padmm_iters = np.zeros((self._max_frames,), dtype=np.int32)
        self.log_padmm_r_p = np.zeros((self._max_frames,), dtype=np.float32)
        self.log_padmm_r_d = np.zeros((self._max_frames,), dtype=np.float32)
        self.log_padmm_r_c = np.zeros((self._max_frames,), dtype=np.float32)

        # Extract actuated DOF indices
        self._njaq: int = self._sim.model.size.sum_of_num_actuated_joint_coords
        self._njad: int = self._sim.model.size.sum_of_num_actuated_joints
        self._actuated_dofs: list[int] = []
        if self._njaq != self._njad:
            self._njaq = 0
            self._njad = 0
            msg.warning(
                f"Number of actuated joint coordinates ({self._njaq}) does not match "
                f"number of actuated joints ({self._njad}), skipping joint logging."
            )
        else:
            self._actuated_dofs = []
            dof_offset = 0
            for joint in self._builder.joints:
                if joint.is_actuated:
                    for dof in range(joint.num_dofs):
                        self._actuated_dofs.append(dof_offset + dof)
                dof_offset += joint.num_dofs

        # Allocate actuated joint logging arrays if applicable
        if self._njaq > 0 and self._njad > 0:
            self.log_q_j = np.zeros((self._max_frames, self._njaq), dtype=np.float32)
            self.log_dq_j = np.zeros((self._max_frames, self._njad), dtype=np.float32)
            self.log_tau_j = np.zeros((self._max_frames, self._njad), dtype=np.float32)
            if self._ctrl is not None:
                self.log_q_j_ref = np.zeros((self._max_frames, self._njaq), dtype=np.float32)
                self.log_dq_j_ref = np.zeros((self._max_frames, self._njad), dtype=np.float32)

        # Allocate logging arrays for solution metrics
        if self._sim.metrics is not None:
            # self.log_kinetic_energy = np.zeros((self._max_frames,), dtype=np.float32)
            # self.log_potential_energy = np.zeros((self._max_frames,), dtype=np.float32)
            # self.log_total_energy = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_eom = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_eom_argmax = np.full((self._max_frames, 2), fill_value=(-1, -1), dtype=np.int32)
            self.log_r_kinematics = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_kinematics_argmax = np.full((self._max_frames, 2), fill_value=(-1, -1), dtype=np.int32)
            self.log_r_cts_joints = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_cts_joints_argmax = np.full((self._max_frames, 2), fill_value=(-1, -1), dtype=np.int32)
            self.log_r_cts_limits = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_cts_limits_argmax = np.full((self._max_frames, 2), fill_value=(-1, -1), dtype=np.int32)
            self.log_r_cts_contacts = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_cts_contacts_argmax = np.full((self._max_frames,), fill_value=-1, dtype=np.int32)
            self.log_r_v_plus = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_v_plus_argmax = np.full((self._max_frames,), fill_value=-1, dtype=np.int32)
            self.log_r_ncp_primal = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_ncp_primal_argmax = np.full((self._max_frames,), fill_value=-1, dtype=np.int32)
            self.log_r_ncp_dual = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_ncp_dual_argmax = np.full((self._max_frames,), fill_value=-1, dtype=np.int32)
            self.log_r_ncp_compl = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_ncp_compl_argmax = np.full((self._max_frames,), fill_value=-1, dtype=np.int32)
            self.log_r_vi_natmap = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_r_vi_natmap_argmax = np.full((self._max_frames,), fill_value=-1, dtype=np.int32)
            self.log_f_ncp = np.zeros((self._max_frames,), dtype=np.float32)
            self.log_f_ccp = np.zeros((self._max_frames,), dtype=np.float32)

    def reset(self):
        """
        Resets the logging frame counter to zero.
        """
        self._frames = 0

    def log(self):
        """
        TODO
        """
        if self._frames >= self._max_frames:
            msg.warning("Maximum logging frames reached, skipping data logging.")
            return

        # Log unilateral constraints info
        if self._sim.limits.model_max_limits_host > 0:
            self.log_num_limits[self._frames] = self._sim.limits.model_active_limits.numpy()[0]
        if self._sim.contacts.model_max_contacts_host > 0:
            self.log_num_contacts[self._frames] = self._sim.contacts.data.model_active_contacts.numpy()[0]

        # Log PADMM solver info
        self.log_padmm_iters[self._frames] = self._sim.solver.data.status.numpy()[0][1]
        self.log_padmm_r_p[self._frames] = self._sim.solver.data.status.numpy()[0][2]
        self.log_padmm_r_d[self._frames] = self._sim.solver.data.status.numpy()[0][3]
        self.log_padmm_r_c[self._frames] = self._sim.solver.data.status.numpy()[0][4]

        # Log joint actuator info if available
        if self._njaq > 0 and self._njad > 0 and self._njad == self._njaq:
            self.log_q_j[self._frames, :] = self._sim.state.q_j.numpy()[self._actuated_dofs]
            self.log_dq_j[self._frames, :] = self._sim.state.dq_j.numpy()[self._actuated_dofs]
            self.log_tau_j[self._frames, :] = self._sim.control.tau_j.numpy()[self._actuated_dofs]

        # Log controller references if available
        if self._ctrl is not None:
            self.log_q_j_ref[self._frames, :] = self._ctrl.data.q_j_ref.numpy()
            self.log_dq_j_ref[self._frames, :] = self._ctrl.data.dq_j_ref.numpy()

        # Log solution metrics if available
        if self._sim.metrics is not None:
            metrics = self._sim.metrics.data
            # self.log_kinetic_energy[self._frames] = metrics.kinetic_energy.numpy()[0]
            # self.log_potential_energy[self._frames] = metrics.potential_energy.numpy()[0]
            # self.log_total_energy[self._frames] = metrics.total_energy.numpy()[0]
            self.log_r_eom[self._frames] = metrics.r_eom.numpy()[0]
            self.log_r_eom_argmax[self._frames, :] = self._unpack_key(metrics.r_eom_argmax.numpy()[0])
            self.log_r_kinematics[self._frames] = metrics.r_kinematics.numpy()[0]
            self.log_r_kinematics_argmax[self._frames, :] = self._unpack_key(metrics.r_kinematics_argmax.numpy()[0])
            self.log_r_cts_joints[self._frames] = metrics.r_cts_joints.numpy()[0]
            self.log_r_cts_joints_argmax[self._frames, :] = self._unpack_key(metrics.r_cts_joints_argmax.numpy()[0])
            self.log_r_cts_limits[self._frames] = metrics.r_cts_limits.numpy()[0]
            self.log_r_cts_limits_argmax[self._frames, :] = self._unpack_key(metrics.r_cts_limits_argmax.numpy()[0])
            self.log_r_cts_contacts[self._frames] = metrics.r_cts_contacts.numpy()[0]
            self.log_r_cts_contacts_argmax[self._frames] = metrics.r_cts_contacts_argmax.numpy()[0]
            self.log_r_v_plus[self._frames] = metrics.r_v_plus.numpy()[0]
            self.log_r_v_plus_argmax[self._frames] = metrics.r_v_plus_argmax.numpy()[0]
            self.log_r_ncp_primal[self._frames] = metrics.r_ncp_primal.numpy()[0]
            self.log_r_ncp_primal_argmax[self._frames] = metrics.r_ncp_primal_argmax.numpy()[0]
            self.log_r_ncp_dual[self._frames] = metrics.r_ncp_dual.numpy()[0]
            self.log_r_ncp_dual_argmax[self._frames] = metrics.r_ncp_dual_argmax.numpy()[0]
            self.log_r_ncp_compl[self._frames] = metrics.r_ncp_compl.numpy()[0]
            self.log_r_ncp_compl_argmax[self._frames] = metrics.r_ncp_compl_argmax.numpy()[0]
            self.log_r_vi_natmap[self._frames] = metrics.r_vi_natmap.numpy()[0]
            self.log_r_vi_natmap_argmax[self._frames] = metrics.r_vi_natmap_argmax.numpy()[0]
            self.log_f_ncp[self._frames] = metrics.f_ncp.numpy()[0]
            self.log_f_ccp[self._frames] = metrics.f_ccp.numpy()[0]

        # Progress the frame counter
        self._frames += 1

    def plot_solver_info(self, path: str | None = None, show: bool = False):
        """
        TODO
        """
        # Attempt to initialize matplotlib
        if self.plt is None:
            msg.warning("matplotlib is not available, skipping plotting.")
            return

        # Check if there are any logged frames
        if self._frames == 0:
            msg.warning("No logged frames to plot, skipping solver info plotting.")
            return

        # Create an array for time logging
        dt = self._sim.settings.dt
        time = np.arange(0, self._frames, dtype=np.float32) * dt

        # Plot the PADMM convergence information
        padmm_iters_path = os.path.join(path, "padmm_status.png") if path is not None else None
        _, axs = self.plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        # Plot the PADMM iterations
        axs[0].step(time, self.log_padmm_iters[: self._frames], label="PADMM Iterations", color="blue")
        axs[0].set_title("PADMM Solver Iterations")
        axs[0].set_ylabel("Iterations")
        axs[0].set_xlabel("Time (s)")
        axs[0].legend()
        axs[0].grid()

        # Plot the PADMM primal residuals
        eps_p = self._sim.settings.solver.primal_tolerance
        axs[1].step(time, self.log_padmm_r_p[: self._frames], label="PADMM Primal Residual", color="orange")
        axs[1].axhline(eps_p, color="black", linestyle="--", linewidth=1.0, label=f"eps_p={eps_p:.1e}")
        axs[1].set_title("PADMM Primal Residual")
        axs[1].set_ylabel("Primal Residual")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend()
        axs[1].grid()

        # Plot the PADMM dual residuals
        eps_d = self._sim.settings.solver.dual_tolerance
        axs[2].step(time, self.log_padmm_r_d[: self._frames], label="PADMM Dual Residual", color="green")
        axs[2].axhline(eps_d, color="black", linestyle="--", linewidth=1.0, label=f"eps_d={eps_d:.1e}")
        axs[2].set_title("PADMM Dual Residual")
        axs[2].set_ylabel("Dual Residual")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].grid()

        # Plot the PADMM complementarity residuals
        eps_c = self._sim.settings.solver.compl_tolerance
        axs[3].step(time, self.log_padmm_r_c[: self._frames], label="PADMM Complementarity Residual", color="red")
        axs[3].axhline(eps_c, color="black", linestyle="--", linewidth=1.0, label=f"eps_c={eps_c:.1e}")
        axs[3].set_title("PADMM Complementarity Residual")
        axs[3].set_ylabel("Complementarity Residual")
        axs[3].set_xlabel("Time (s)")
        axs[3].legend()
        axs[3].grid()

        # Adjust layout
        self.plt.tight_layout()
        # Save the figure if a path is provided
        if padmm_iters_path is not None:
            self.plt.savefig(padmm_iters_path, dpi=300)
        # Show the figure if requested
        # NOTE: This will block execution until the plot window is closed
        if show:
            self.plt.show()
        # Close the current figure to free memory
        self.plt.close()

        # Plot histogram
        padmm_iters_hist_path = os.path.join(path, "padmm_iterations_histogram.png") if path is not None else None
        self.plt.rcParams["axes.axisbelow"] = True
        self.plt.grid(True, which="major", linestyle="--", linewidth=0.5)
        self.plt.grid(True, which="minor", linestyle=":", linewidth=0.25)
        self.plt.hist(self.log_padmm_iters[: self._frames], bins=50)
        self.plt.yscale("log")  # Make Y-axis logarithmic
        self.plt.title("Histogram of PADMM Solver Iterations")
        self.plt.xlabel("Iterations")
        self.plt.ylabel("Frequency")
        # Save the figure if a path is provided
        if padmm_iters_hist_path is not None:
            self.plt.savefig(padmm_iters_hist_path, dpi=300)
        # Show the figure if requested
        # NOTE: This will block execution until the plot window is closed
        if show:
            self.plt.show()
        # Close the current figure to free memory
        self.plt.close()
        self.plt.rcParams["axes.axisbelow"] = False

    def plot_joint_tracking(self, path: str | None = None, show: bool = False):
        """
        TODO
        """
        # Attempt to initialize matplotlib
        if self.plt is None:
            msg.warning("matplotlib is not available, skipping plotting.")
            return

        # Check if there are any logged frames
        if self._frames == 0:
            msg.warning("No logged frames to plot, skipping solver info plotting.")
            return

        # Ensure that joint logging data is available
        if self._njaq == 0 or self._njad == 0:
            msg.warning("No actuated joints to plot, skipping joint-tracking plotting.")
            return

        # Create an array for time logging
        dt = self._sim.settings.dt
        time = np.arange(0, self._frames, dtype=np.float32) * dt

        # Then plot the joint tracking results
        for j in range(len(self._actuated_dofs)):
            # Set the output path for the current joint
            tracking_path = os.path.join(path, f"tracking_joint_{j}.png") if path is not None else None

            # Plot logged data after the viewer is closed
            _, axs = self.plt.subplots(3, 1, figsize=(10, 10), sharex=True)

            # Plot the measured vs reference joint positions
            axs[0].step(time, self.log_q_j[: self._frames, j], label="Measured")
            if self._ctrl:
                axs[0].step(time, self.log_q_j_ref[: self._frames, j], label="Reference", linestyle="--")
            axs[0].set_title(f"Actuator DoF {j} Position Tracking")
            axs[0].set_ylabel("Actuator Position (rad)")
            axs[0].legend()
            axs[0].grid()

            # Plot the measured vs reference joint velocities
            axs[1].step(time, self.log_dq_j[: self._frames, j], label="Measured")
            if self._ctrl:
                axs[1].step(time, self.log_dq_j_ref[: self._frames, j], label="Reference", linestyle="--")
            axs[1].set_title(f"Actuator DoF {j} Velocity Tracking")
            axs[1].set_ylabel("Actuator Velocity (rad/s)")
            axs[1].legend()
            axs[1].grid()

            # Plot the control torques
            axs[2].step(time, self.log_tau_j[: self._frames, j], label="Control Torque")
            axs[2].set_title(f"Actuator DoF {j} Control Torque")
            axs[2].set_ylabel("Torque (Nm)")
            axs[2].set_xlabel("Time (s)")
            axs[2].legend()
            axs[2].grid()

            # Adjust layout
            self.plt.tight_layout()

            # Save the figure if a path is provided
            if tracking_path is not None:
                self.plt.savefig(tracking_path, dpi=300)

            # Show the figure if requested
            # NOTE: This will block execution until the plot window is closed
            if show:
                self.plt.show()

            # Close the current figure to free memory
            self.plt.close()

    def plot_solution_metrics(self, path: str | None = None, show: bool = False):
        """
        TODO
        """
        # Attempt to initialize matplotlib
        if self.plt is None:
            msg.warning("matplotlib is not available, skipping plotting.")
            return

        # Check if there are any logged frames
        if self._frames == 0:
            msg.warning("No logged frames to plot, skipping solution metrics plotting.")
            return

        # Ensure that solution metrics data is available
        if self._sim.metrics is None:
            msg.warning("No solution metrics to plot, skipping solution metrics plotting.")
            return

        # Create an array for time logging
        dt = self._sim.settings.dt
        time = np.arange(0, self._frames, dtype=np.float32) * dt

        # Plot the solution metrics
        metrics_path = os.path.join(path, "solution_metrics.png") if path is not None else None
        _, axs = self.plt.subplots(6, 2, figsize=(15, 20), sharex=True)

        # Plot each metric
        metric_titles = [
            "EoM Residual (r_eom)",
            "Kinematics Residual (r_kinematics)",
            "Joint Constraints Residual (r_cts_joints)",
            "Limit Constraints Residual (r_cts_limits)",
            "Contact Constraints Residual (r_cts_contacts)",
            "Post-Event Constraint Velocity Residual (r_v_plus)",
            "NCP Primal Residual (r_ncp_primal)",
            "NCP Dual Residual (r_ncp_dual)",
            "NCP Complementarity Residual (r_ncp_compl)",
            "VI Natural-Map Residual (r_vi_natmap)",
            "NCP Objective (f_ncp)",
            "CCP Objective (f_ccp)",
        ]
        metric_names = [
            "r_eom",
            "r_kinematics",
            "r_cts_joints",
            "r_cts_limits",
            "r_cts_contacts",
            "r_v_plus",
            "r_ncp_primal",
            "r_ncp_dual",
            "r_ncp_compl",
            "r_vi_natmap",
            "f_ncp",
            "f_ccp",
        ]
        log_attrs = [
            self.log_r_eom,
            self.log_r_kinematics,
            self.log_r_cts_joints,
            self.log_r_cts_limits,
            self.log_r_cts_contacts,
            self.log_r_v_plus,
            self.log_r_ncp_primal,
            self.log_r_ncp_dual,
            self.log_r_ncp_compl,
            self.log_r_vi_natmap,
            self.log_f_ncp,
            self.log_f_ccp,
        ]
        for i, (title, name, log_attr) in enumerate(zip(metric_titles, metric_names, log_attrs, strict=False)):
            ax = axs[i // 2, i % 2]
            ax.step(time, log_attr[: self._frames], label=name)
            ax.set_title(title)
            ax.set_ylabel(name)
            ax.set_xlabel("Time (s)")
            ax.legend()
            ax.grid()

        # Adjust layout
        self.plt.tight_layout()

        # Save the figure if a path is provided
        if metrics_path is not None:
            self.plt.savefig(metrics_path, dpi=300)

        # Show the figure if requested
        # NOTE: This will block execution until the plot window is closed
        if show:
            self.plt.show()

        # Close the current figure to free memory
        self.plt.close()

    @staticmethod
    def _unpack_key(key: np.uint64) -> tuple[int, int]:
        """
        TODO
        """
        index1 = (key >> 32) & 0x7FFFFFFF
        index2 = key & 0x7FFFFFFF
        return int(index1), int(index2)
