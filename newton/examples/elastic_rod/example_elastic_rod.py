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
# Example Elastic Rod
#
# A horizontal Cosserat elastic rod fixed at one end, bending under gravity.
# Uses the XPBD rod solver with a direct block-tridiagonal solve.
#
# Command: uv run -m newton.examples elastic_rod
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
from newton.solvers import xpbd_rod

from newton.examples.elastic_rod.rod_mesher import RodMesher


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.bend_stiffness = 0.5
        self.twist_stiffness = 0.5
        self.young_modulus = 1.0e6
        self.torsion_modulus = 1.0e6
        self.rest_bend_d1 = 0.0
        self.rest_bend_d2 = 0.0
        self.rest_twist = 0.0
        self.rest_length = 0.05
        self.lock_root = True
        self.lock_root_rotation = True

        num_points = 128
        spacing = 0.05

        builder = newton.ModelBuilder()

        # Register rod solver attributes before adding rods
        newton.solvers.SolverXPBDRod.register_custom_attributes(builder)

        # Create rod positions along X-axis at height z=1.0
        positions = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            positions[i, 0] = i * spacing
            positions[i, 2] = 1.0

        # Add the elastic rod with moderate stiffness for visible bending
        xpbd_rod.add_elastic_rod(
            builder,
            positions=positions,
            radius=0.005,
            particle_mass=0.05,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            young_modulus=self.young_modulus,
            torsion_modulus=self.torsion_modulus,
            lock_root=True,
            lock_root_rotation=True,
        )

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBDRod(
            model=self.model,
            linear_damping=0.001,
            angular_damping=0.001,
            solver_backend="block_thomas",
            floor_z=-0.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()



        # Director visualization state
        self.show_directors = False
        self.director_scale = 0.03

        # Pre-allocate GPU arrays for director line visualization
        self._director_starts = []
        self._director_ends = []
        self._director_colors = []
        for ws in self.solver._rods:
            n = ws.num_edges * 3
            device = self.model.device
            self._director_starts.append(wp.zeros(n, dtype=wp.vec3, device=device))
            self._director_ends.append(wp.zeros(n, dtype=wp.vec3, device=device))
            self._director_colors.append(wp.zeros(n, dtype=wp.vec3, device=device))

        # Create rod meshers for tube visualization
        self._meshers = []
        for ws in self.solver._rods:
            mesher = RodMesher(
                num_points=ws.num_points,
                radius=0.01,
                resolution=8,
                smoothing=3,
                device=self.model.device,
            )
            self._meshers.append(mesher)

        # Cache CPU-side root state to avoid GPU downloads each frame
        self._root_qs = []
        self._free_inv_masses = []
        self._free_quat_inv_masses = []
        for ws in self.solver._rods:
            q0 = ws.orientations_wp.numpy()[0]
            self._root_qs.append(q0.copy())
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

    def _rotate_root(self):
        import warp as wp

        rotate_speed = 1.5  # rad/s
        dx = 0.0
        dz = 0.0
        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("i"):
                dz -= rotate_speed * self.frame_dt
            if self.viewer.is_key_down("k"):
                dz += rotate_speed * self.frame_dt
            if self.viewer.is_key_down("j"):
                dx -= rotate_speed * self.frame_dt
            if self.viewer.is_key_down("l"):
                dx += rotate_speed * self.frame_dt

        if dx == 0.0 and dz == 0.0:
            return

        def qmul(a, b):
            return np.array([
                a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
                a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
                a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
                a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
            ], dtype=np.float32)

        qx = np.array([np.sin(dx / 2), 0.0, 0.0, np.cos(dx / 2)], dtype=np.float32)
        qz = np.array([0.0, 0.0, np.sin(dz / 2), np.cos(dz / 2)], dtype=np.float32)

        for rod_idx in range(len(self.solver._rods)):
            q = self._root_qs[rod_idx]
            q = qmul(qz, qmul(qx, q))
            q /= np.linalg.norm(q)
            self._root_qs[rod_idx] = q
            self.solver.set_root_orientation(rod_idx, wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    def _simulate_substeps(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self):
        if self.lock_root:
            self._rotate_root()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def _build_director_lines(self, idx, ws):
        ps = self.solver._rod_particle_starts[idx]
        wp.launch(
            xpbd_rod.compute_director_lines_kernel,
            dim=ws.num_edges * 3,
            inputs=[
                self.state_0.particle_q[ps : ps + ws.num_points],
                ws.orientations_wp,
                ws.num_edges,
                self.director_scale,
            ],
            outputs=[
                self._director_starts[idx],
                self._director_ends[idx],
                self._director_colors[idx],
            ],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        for idx, (ws, mesher) in enumerate(zip(self.solver._rods, self._meshers)):
            ps = self.solver._rod_particle_starts[idx]
            mesher.update(self.state_0.particle_q[ps : ps + ws.num_points])
            self.viewer.log_mesh(
                f"/rod_mesh_{idx}",
                mesher.vertices,
                mesher.indices,
                mesher.normals,
                mesher.uvs,
            )

        for idx, ws in enumerate(self.solver._rods):
            name = f"/rod_directors_{idx}"
            if self.show_directors:
                self._build_director_lines(idx, ws)
                self.viewer.log_lines(
                    name,
                    self._director_starts[idx],
                    self._director_ends[idx],
                    self._director_colors[idx],
                    width=0.005,
                )
            else:
                self.viewer.log_lines(name, None, None, None, hidden=True)

        self.viewer.end_frame()

    def _update_rod_stiffness(self):
        import warp as wp

        for ws in self.solver._rods:
            ws.young_modulus = self.young_modulus
            ws.torsion_modulus = self.torsion_modulus
            bs = np.full((ws.num_edges, 3), [self.bend_stiffness, self.bend_stiffness, self.twist_stiffness], dtype=np.float32)
            ws.bend_stiffness_wp.assign(wp.array(bs, dtype=wp.vec3, device=ws.bend_stiffness_wp.device))

    def _update_root_lock(self):
        import warp as wp

        for rod_idx, ws in enumerate(self.solver._rods):
            free_inv_mass = self._free_inv_masses[rod_idx]
            root_inv_mass = np.array([0.0 if self.lock_root else free_inv_mass], dtype=np.float32)
            wp.copy(dest=ws.inv_masses_wp, src=wp.array(root_inv_mass, dtype=wp.float32, device=ws.device), count=1)

            free_quat_inv_mass = self._free_quat_inv_masses[rod_idx]
            root_quat_inv_mass = np.array([0.0 if (self.lock_root or self.lock_root_rotation) else free_quat_inv_mass], dtype=np.float32)
            wp.copy(dest=ws.quat_inv_masses_wp, src=wp.array(root_quat_inv_mass, dtype=wp.float32, device=ws.device), count=1)

    def _update_rest_lengths(self):
        import warp as wp

        for ws in self.solver._rods:
            rl = np.full(ws.num_edges, self.rest_length, dtype=np.float32)
            ws.rest_lengths_wp.assign(wp.array(rl, dtype=wp.float32, device=ws.rest_lengths_wp.device))

    def _update_rest_darboux(self):
        import warp as wp

        for ws in self.solver._rods:
            rd = np.full((ws.num_edges, 3), [self.rest_bend_d1, self.rest_bend_d2, self.rest_twist], dtype=np.float32)
            ws.rest_darboux_wp.assign(wp.array(rd, dtype=wp.vec3, device=ws.rest_darboux_wp.device))

    def _render_ui(self, imgui):
        changed_lock, self.lock_root = imgui.checkbox("Lock Root Position", self.lock_root)
        changed_lock_rot, self.lock_root_rotation = imgui.checkbox("Lock Root Rotation", self.lock_root_rotation)
        if changed_lock or changed_lock_rot:
            self._update_root_lock()

        imgui.separator()
        changed_bend, self.bend_stiffness = imgui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist, self.twist_stiffness = imgui.slider_float("Twist Stiffness", self.twist_stiffness, 0.0, 1.0)
        if changed_bend or changed_twist:
            self._update_rod_stiffness()

        imgui.separator()
        changed_E, self.young_modulus = imgui.input_float("Young Modulus [Pa]", self.young_modulus, format="%.1f")
        changed_G, self.torsion_modulus = imgui.input_float("Torsion Modulus [Pa]", self.torsion_modulus, format="%.1f")
        if changed_E or changed_G:
            self._update_rod_stiffness()

        imgui.separator()
        changed_rl, self.rest_length = imgui.slider_float("Rest Length", self.rest_length, 0.01, 0.1)
        if changed_rl:
            self._update_rest_lengths()

        imgui.separator()
        changed_d1, self.rest_bend_d1 = imgui.slider_float("Rest Bend d1", self.rest_bend_d1, -0.1, 0.1)
        changed_d2, self.rest_bend_d2 = imgui.slider_float("Rest Bend d2", self.rest_bend_d2, -0.1, 0.1)
        changed_tw, self.rest_twist = imgui.slider_float("Rest Twist", self.rest_twist, -0.1, 0.1)
        if changed_d1 or changed_d2 or changed_tw:
            self._update_rest_darboux()

        imgui.separator()
        _, self.show_directors = imgui.checkbox("Show Material Frames", self.show_directors)
        _, self.director_scale = imgui.slider_float("Director Scale", self.director_scale, 0.01, 0.1)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
