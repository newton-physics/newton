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

###########################################################################
# Example Elastic Rods Bend
#
# Demonstrates elastic rod bending behavior with different stiffness values.
# Shows 5 rods side-by-side with increasing bend stiffness (from soft to stiff),
# all fixed at one end and bending under gravity.
# Uses the XPBD rod solver with a direct block-tridiagonal solve.
# A tab panel allows per-rod parameter tuning at runtime.
#
# Command: uv run -m newton.examples elastic_rods_bend
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
from newton.solvers import xpbd_rod

from newton.examples.elastic_rod.rod_mesher import BatchedRodMesher


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        num_points = 128
        spacing = 0.05
        y_separation = 0.3

        # Initial per-rod parameters (bend stiffness sweep)
        self.num_rods = args.num_rods if args is not None and hasattr(args, "num_rods") else 5
        bend_stiffness_values = np.logspace(-3, 1, self.num_rods).tolist()

        # Per-rod UI state
        self.bend_stiffness = list(bend_stiffness_values)
        self.twist_stiffness = [0.1] * self.num_rods
        self.young_modulus = [1.0e6] * self.num_rods
        self.torsion_modulus = [1.0e6] * self.num_rods
        self.rest_length = [0.05] * self.num_rods
        self.rest_bend_d1 = [0.0] * self.num_rods
        self.rest_bend_d2 = [0.0] * self.num_rods
        self.rest_twist = [0.0] * self.num_rods
        self.lock_root = [True] * self.num_rods
        self.lock_root_rotation = [True] * self.num_rods

        builder = newton.ModelBuilder()

        # Register rod solver attributes before adding rods
        newton.solvers.SolverXPBDRod.register_custom_attributes(builder)

        self.rod_particle_indices: list[list[int]] = []

        for i, bend_stiffness in enumerate(bend_stiffness_values):
            y_pos = (i - (self.num_rods - 1) / 2.0) * y_separation

            positions = np.zeros((num_points, 3), dtype=np.float32)
            for j in range(num_points):
                positions[j, 0] = j * spacing
                positions[j, 1] = y_pos
                positions[j, 2] = 1.0

            particle_indices = xpbd_rod.add_elastic_rod(
                builder,
                positions=positions,
                radius=0.005,
                particle_mass=0.05,
                bend_stiffness=bend_stiffness,
                twist_stiffness=self.twist_stiffness[i],
                young_modulus=self.young_modulus[i],
                torsion_modulus=self.torsion_modulus[i],
                lock_root=True,
                lock_root_rotation=True,
            )

            self.rod_particle_indices.append(particle_indices)

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBDRod(
            model=self.model,
            linear_damping=0.001,
            angular_damping=0.001,
            solver_backend="block_thomas",
            floor_z=0.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Batched rod mesher — single kernel launch for all rods
        self._mesher = BatchedRodMesher(
            num_rods=self.num_rods,
            num_points=num_points,
            radius=0.01,
            resolution=8,
            smoothing=3,
            particle_offsets=self.solver._rod_particle_starts,
            device=self.model.device,
        )

        # Cache CPU-side root state to avoid GPU downloads each frame
        self._free_inv_masses = []
        self._free_quat_inv_masses = []
        for ws in self.solver._rods:
            inv_m = ws.inv_masses_wp.numpy()
            self._free_inv_masses.append(float(inv_m[1]) if ws.num_points > 1 else 1.0)
            qim = ws.quat_inv_masses_wp.numpy()
            self._free_quat_inv_masses.append(float(qim[1]) if ws.num_points > 1 else 1.0)

        # Capture CUDA graph for the substep loop
        self.graph = None
        device = self.model.device
        if device.is_cuda and wp.is_mempool_enabled(device):
            with wp.ScopedCapture(device=device) as capture:
                self._simulate_substeps()
            self.graph = capture.graph
            print("CUDA graph captured for simulation substeps")
        else:
            print("CUDA graph not available, using standard kernel launches")

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui, position="side")

    def _simulate_substeps(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        self._mesher.update(self.state_0.particle_q)
        for idx in range(self.num_rods):
            self.viewer.log_mesh(
                f"/rod_mesh_{idx}",
                self._mesher.rod_vertices(idx),
                self._mesher.rod_indices(),
                self._mesher.rod_normals(idx),
                self._mesher.rod_uvs(),
            )

        self.viewer.end_frame()

    def _update_rod_stiffness(self, rod_idx):
        import warp as wp

        ws = self.solver._rods[rod_idx]
        ws.young_modulus = self.young_modulus[rod_idx]
        ws.torsion_modulus = self.torsion_modulus[rod_idx]
        bs = np.full(
            (ws.num_edges, 3),
            [self.bend_stiffness[rod_idx], self.bend_stiffness[rod_idx], self.twist_stiffness[rod_idx]],
            dtype=np.float32,
        )
        ws.bend_stiffness_wp.assign(wp.array(bs, dtype=wp.vec3, device=ws.bend_stiffness_wp.device))

    def _update_root_lock(self, rod_idx):
        import warp as wp

        ws = self.solver._rods[rod_idx]
        free_inv_mass = self._free_inv_masses[rod_idx]
        root_inv_mass = np.array([0.0 if self.lock_root[rod_idx] else free_inv_mass], dtype=np.float32)
        wp.copy(dest=ws.inv_masses_wp, src=wp.array(root_inv_mass, dtype=wp.float32, device=ws.device), count=1)

        free_quat_inv_mass = self._free_quat_inv_masses[rod_idx]
        root_quat_inv_mass = np.array(
            [0.0 if (self.lock_root[rod_idx] or self.lock_root_rotation[rod_idx]) else free_quat_inv_mass],
            dtype=np.float32,
        )
        wp.copy(dest=ws.quat_inv_masses_wp, src=wp.array(root_quat_inv_mass, dtype=wp.float32, device=ws.device), count=1)

    def _update_rest_lengths(self, rod_idx):
        import warp as wp

        ws = self.solver._rods[rod_idx]
        rl = np.full(ws.num_edges, self.rest_length[rod_idx], dtype=np.float32)
        ws.rest_lengths_wp.assign(wp.array(rl, dtype=wp.float32, device=ws.rest_lengths_wp.device))

    def _update_rest_darboux(self, rod_idx):
        import warp as wp

        ws = self.solver._rods[rod_idx]
        rd = np.full(
            (ws.num_edges, 3),
            [self.rest_bend_d1[rod_idx], self.rest_bend_d2[rod_idx], self.rest_twist[rod_idx]],
            dtype=np.float32,
        )
        ws.rest_darboux_wp.assign(wp.array(rd, dtype=wp.vec3, device=ws.rest_darboux_wp.device))

    def _render_rod_tab(self, imgui, rod_idx):
        changed_lock, self.lock_root[rod_idx] = imgui.checkbox(
            f"Lock Root Position##{rod_idx}", self.lock_root[rod_idx]
        )
        changed_lock_rot, self.lock_root_rotation[rod_idx] = imgui.checkbox(
            f"Lock Root Rotation##{rod_idx}", self.lock_root_rotation[rod_idx]
        )
        if changed_lock or changed_lock_rot:
            self._update_root_lock(rod_idx)

        imgui.separator()
        changed_bend, self.bend_stiffness[rod_idx] = imgui.slider_float(
            f"Bend Stiffness##{rod_idx}", self.bend_stiffness[rod_idx], 0.0, 10.0
        )
        changed_twist, self.twist_stiffness[rod_idx] = imgui.slider_float(
            f"Twist Stiffness##{rod_idx}", self.twist_stiffness[rod_idx], 0.0, 10.0
        )
        if changed_bend or changed_twist:
            self._update_rod_stiffness(rod_idx)

        imgui.separator()
        changed_E, self.young_modulus[rod_idx] = imgui.input_float(
            f"Young Modulus [Pa]##{rod_idx}", self.young_modulus[rod_idx], format="%.1f"
        )
        changed_G, self.torsion_modulus[rod_idx] = imgui.input_float(
            f"Torsion Modulus [Pa]##{rod_idx}", self.torsion_modulus[rod_idx], format="%.1f"
        )
        if changed_E or changed_G:
            self._update_rod_stiffness(rod_idx)

        imgui.separator()
        changed_rl, self.rest_length[rod_idx] = imgui.slider_float(
            f"Rest Length##{rod_idx}", self.rest_length[rod_idx], 0.01, 0.1
        )
        if changed_rl:
            self._update_rest_lengths(rod_idx)

        imgui.separator()
        changed_d1, self.rest_bend_d1[rod_idx] = imgui.slider_float(
            f"Rest Bend d1##{rod_idx}", self.rest_bend_d1[rod_idx], -0.1, 0.1
        )
        changed_d2, self.rest_bend_d2[rod_idx] = imgui.slider_float(
            f"Rest Bend d2##{rod_idx}", self.rest_bend_d2[rod_idx], -0.1, 0.1
        )
        changed_tw, self.rest_twist[rod_idx] = imgui.slider_float(
            f"Rest Twist##{rod_idx}", self.rest_twist[rod_idx], -0.1, 0.1
        )
        if changed_d1 or changed_d2 or changed_tw:
            self._update_rest_darboux(rod_idx)

    def _render_ui(self, imgui):
        if imgui.begin_tab_bar("rods_tab_bar"):
            for i in range(self.num_rods):
                tab_open, _ = imgui.begin_tab_item(f"Rod {i}")
                if tab_open:
                    self._render_rod_tab(imgui, i)
                    imgui.end_tab_item()
            imgui.end_tab_bar()

    def test_final(self):
        """Test elastic rod bending with different stiffness values."""
        particle_q = self.state_0.particle_q.numpy()

        # Check for NaN
        assert not np.any(np.isnan(particle_q)), "Particle positions contain NaN"
        assert np.all(np.isfinite(particle_q)), "Non-finite values in particle positions"

        # All roots should remain fixed near z=1.0
        for i, indices in enumerate(self.rod_particle_indices):
            root_z = particle_q[indices[0], 2]
            assert abs(root_z - 1.0) < 0.01, f"Rod {i} root should be fixed at z=1.0, but z={root_z:.3f}"

        # All tips should descend below initial height
        for i, indices in enumerate(self.rod_particle_indices):
            tip_z = particle_q[indices[-1], 2]
            assert tip_z < 1.0, f"Rod {i} tip should bend under gravity, but tip z={tip_z:.3f}"

        # Stiffer rods should have higher tip z (less droop)
        tip_heights = []
        for indices in self.rod_particle_indices:
            tip_heights.append(particle_q[indices[-1], 2])

        for i in range(len(tip_heights) - 1):
            assert tip_heights[i] <= tip_heights[i + 1] + 0.05, (
                f"Rod {i} tip (z={tip_heights[i]:.3f}) should be lower than or equal to "
                f"rod {i + 1} tip (z={tip_heights[i + 1]:.3f}) since it has lower bend stiffness"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-rods", type=int, default=5, help="Number of rods to simulate.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
