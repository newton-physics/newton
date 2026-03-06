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
# Example Rod Insertion
#
# A Cosserat elastic rod being inserted into a vascular (aorta) mesh.
# Uses the XPBD rod solver with BVH-based mesh collision.
#
# Controls:
#   I / K  - Insert / Retract rod
#   J / L  - Rotate root left / right
#   U / O  - Move root up / down
#
# Command: uv run python newton/examples/elastic_rod/example_rod_insertion.py
#
###########################################################################

import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
import newton.usd
from newton.solvers import xpbd_rod

from newton.examples.elastic_rod.rod_mesher import RodMesher
from newton.examples.cosserat2.kernels.collision import (
    collide_particles_vs_triangles_bvh_kernel,
    compute_static_tri_aabbs_kernel,
)


def _resolve_vessel_usd() -> str:
    """Find the AortaWithVesselsStatic.usdc asset."""
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "..", "cosserat", "models", "AortaWithVesselsStatic.usdc"),
        os.path.join(base, "..", "gpu_warp", "models", "AortaWithVesselsStatic.usdc"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Cannot find AortaWithVesselsStatic.usdc. "
        f"Searched: {[os.path.abspath(c) for c in candidates]}"
    )


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.collision_iterations = 2

        # Rod parameters
        num_points = 128
        spacing = 0.05
        self.insertion_speed = 0.5  # m/s

        # Vessel mesh transform (matches cosserat_codex)
        self.mesh_scale = 0.01
        self.mesh_xform = wp.transform(
            wp.vec3(0.0, 0.0, 1.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0),
        )

        # Build rod positions: straight line along X-axis
        # The vessel entry is approximately at (-3.28, -0.5, 1.68) in world space
        # (negative-X edge of the aorta). The rod root starts outside the vessel
        # and extends inward (toward positive X).
        entry_point = np.array([-3.283, -0.5, 1.683], dtype=np.float32)
        rod_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # into the vessel

        positions = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            # Root (i=0) is just outside the vessel; tip extends inward
            positions[i] = entry_point + rod_direction * (i * spacing)

        builder = newton.ModelBuilder()
        newton.solvers.SolverXPBDRod.register_custom_attributes(builder)

        xpbd_rod.add_elastic_rod(
            builder,
            positions=positions,
            radius=0.005,
            particle_mass=0.05,
            bend_stiffness=0.5,
            twist_stiffness=0.5,
            young_modulus=1.0e6,
            torsion_modulus=1.0e6,
            lock_root=True,
            lock_root_rotation=True,
        )

        # Load vessel mesh
        from pxr import Usd  # noqa: PLC0415

        usd_path = _resolve_vessel_usd()
        usd_stage = Usd.Stage.Open(usd_path)
        mesh_prim = usd_stage.GetPrimAtPath("/root/Mesh/Mesh_004")
        vessel_mesh = newton.usd.get_mesh(mesh_prim)

        vessel_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e4,
            kd=1.0e2,
            mu=0.1,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        builder.add_shape_mesh(
            body=-1,
            mesh=vessel_mesh,
            scale=(self.mesh_scale, self.mesh_scale, self.mesh_scale),
            xform=self.mesh_xform,
            cfg=vessel_cfg,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()
        self.model.set_gravity([0.0, 0.0, 0.0])

        self.solver = newton.solvers.SolverXPBDRod(
            model=self.model,
            linear_damping=0.001,
            angular_damping=0.001,
            solver_backend="block_thomas",
            floor_z=-10.0,  # Disable floor collision (vessel provides confinement)
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Build vessel BVH for collision
        device = self.model.device
        vessel_vertices_np = np.array(vessel_mesh.vertices, dtype=np.float32)
        vessel_indices_np = np.array(vessel_mesh.indices, dtype=np.int32).reshape(-1, 3)
        self.num_vessel_triangles = vessel_indices_np.shape[0]

        # Apply mesh transform to vertices
        scaled = vessel_vertices_np * self.mesh_scale
        transformed = np.zeros_like(scaled)
        for i in range(len(scaled)):
            v = wp.vec3(scaled[i, 0], scaled[i, 1], scaled[i, 2])
            vt = wp.transform_point(self.mesh_xform, v)
            transformed[i] = [vt[0], vt[1], vt[2]]

        self.vessel_vertices = wp.array(transformed, dtype=wp.vec3f, device=device)
        self.vessel_indices = wp.array(vessel_indices_np, dtype=wp.int32, device=device)
        self.tri_lower = wp.zeros(self.num_vessel_triangles, dtype=wp.vec3f, device=device)
        self.tri_upper = wp.zeros(self.num_vessel_triangles, dtype=wp.vec3f, device=device)
        wp.launch(
            compute_static_tri_aabbs_kernel,
            dim=self.num_vessel_triangles,
            inputs=[self.vessel_vertices, self.vessel_indices],
            outputs=[self.tri_lower, self.tri_upper],
            device=device,
        )
        self.vessel_bvh = wp.Bvh(self.tri_lower, self.tri_upper)

        # Collision arrays
        n_particles = self.model.particle_count
        self._collision_radii = wp.array(
            np.full(n_particles, 0.005, dtype=np.float32), dtype=wp.float32, device=device,
        )
        self._collision_inv_masses = wp.zeros(n_particles, dtype=wp.float32, device=device)
        self._update_collision_inv_masses()

        # Rod mesher for tube visualization
        ws = self.solver._rods[0]
        self._mesher = RodMesher(
            num_points=ws.num_points,
            radius=0.008,
            resolution=8,
            smoothing=3,
            device=device,
        )

        # Root control state
        self._root_pos = positions[0].copy()

        # CUDA graph (disabled when collision is on — collisions use BVH queries)
        self.graph = None

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui, position="side")

    def _update_collision_inv_masses(self):
        ws = self.solver._rods[0]
        inv_masses = ws.inv_masses_wp.numpy()
        n_rod = ws.num_points
        buf = np.zeros(self.model.particle_count, dtype=np.float32)
        buf[:n_rod] = inv_masses
        self._collision_inv_masses.assign(
            wp.array(buf, dtype=wp.float32, device=self.model.device)
        )

    def _apply_mesh_collisions(self):
        """Run BVH particle-vs-triangle collision on rod positions.

        The XPBD rod solver works on its internal ``ws.positions_wp``, so we
        must: (1) copy rod positions into the staging particle array,
        (2) run the collision kernel, (3) copy corrected positions back.
        """
        ws = self.solver._rods[0]
        ps = self.solver._rod_particle_starts[0]
        device = self.model.device

        # 1. Sync rod positions → state particle array
        wp.copy(
            dest=self.state_0.particle_q, src=ws.positions_wp,
            dest_offset=ps, src_offset=0, count=ws.num_points,
        )

        # 2. Collide
        wp.launch(
            collide_particles_vs_triangles_bvh_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,
                self._collision_radii,
                self._collision_inv_masses,
                self.vessel_vertices,
                self.vessel_indices,
                self.vessel_bvh.id,
                True,   # use_gauss_seidel
                True,   # use_two_sided
            ],
            outputs=[self.state_0.particle_q],
            device=device,
        )

        # 3. Copy corrected positions back into both rod arrays
        wp.copy(
            dest=ws.positions_wp, src=self.state_0.particle_q,
            dest_offset=0, src_offset=ps, count=ws.num_points,
        )
        wp.copy(
            dest=ws.predicted_positions_wp, src=self.state_0.particle_q,
            dest_offset=0, src_offset=ps, count=ws.num_points,
        )

    def _handle_input(self):
        if not hasattr(self.viewer, "is_key_down"):
            return

        ws = self.solver._rods[0]
        ps = self.solver._rod_particle_starts[0]

        # Accumulate root position deltas from all keys
        moved = False

        # Insertion / retraction along X-axis
        if self.viewer.is_key_down("i"):
            self._root_pos[0] += self.insertion_speed * self.frame_dt
            moved = True
        if self.viewer.is_key_down("k"):
            self._root_pos[0] -= self.insertion_speed * self.frame_dt
            moved = True

        # Lateral movement (Y-axis)
        if self.viewer.is_key_down("j"):
            self._root_pos[1] -= 0.3 * self.frame_dt
            moved = True
        if self.viewer.is_key_down("l"):
            self._root_pos[1] += 0.3 * self.frame_dt
            moved = True

        # Vertical movement (Z-axis)
        if self.viewer.is_key_down("u"):
            self._root_pos[2] += 0.3 * self.frame_dt
            moved = True
        if self.viewer.is_key_down("o"):
            self._root_pos[2] -= 0.3 * self.frame_dt
            moved = True

        if moved:
            # Update root in state particle array
            q = self.state_0.particle_q.numpy()
            q[ps] = self._root_pos
            self.state_0.particle_q.assign(wp.array(q, dtype=wp.vec3, device=self.model.device))
            # Update root in rod internal positions
            p = ws.positions_wp.numpy()
            p[0] = self._root_pos
            ws.positions_wp.assign(wp.array(p, dtype=wp.vec3, device=self.model.device))

    def _simulate_substeps(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            # Apply vessel mesh collision after each substep
            for _ in range(self.collision_iterations):
                self._apply_mesh_collisions()

    def simulate(self):
        self._handle_input()
        self._simulate_substeps()

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        ws = self.solver._rods[0]
        ps = self.solver._rod_particle_starts[0]
        self._mesher.update(self.state_0.particle_q[ps : ps + ws.num_points])
        self.viewer.log_mesh(
            "/rod_mesh",
            self._mesher.vertices,
            self._mesher.indices,
            self._mesher.normals,
            self._mesher.uvs,
        )

        self.viewer.end_frame()

    def _render_ui(self, imgui):
        imgui.text("Rod Insertion Controls")
        imgui.separator()
        imgui.text("I/K - Insert / Retract")
        imgui.text("J/L - Move root left / right")
        imgui.text("U/O - Move root up / down")
        imgui.separator()

        changed, self.insertion_speed = imgui.slider_float(
            "Insertion Speed", self.insertion_speed, 0.1, 2.0,
        )
        changed, self.collision_iterations = imgui.slider_int(
            "Collision Iterations", self.collision_iterations, 0, 5,
        )

        imgui.separator()
        pos = self._root_pos
        imgui.text(f"Root: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    def test_final(self):
        """Verify the rod is still valid after simulation."""
        ws = self.solver._rods[0]
        positions = ws.positions_wp.numpy()
        assert not np.any(np.isnan(positions)), "Rod positions contain NaN"
        assert not np.any(np.isinf(positions)), "Rod positions contain Inf"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
