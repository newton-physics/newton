# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
# Example Cable Y-Junction
#
# This example shows how to simulate a Y-junction using `builder.add_rod_graph(...)`
# with a shared junction node.
#
###########################################################################

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples


def _y_dirs_xy() -> list[wp.vec3]:
    # Symmetric 3-way junction in the XY plane.
    return [
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(-0.5, math.sqrt(3.0) * 0.5, 0.0),
        wp.vec3(-0.5, -math.sqrt(3.0) * 0.5, 0.0),
    ]


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable parameters.
        cable_radius = 0.01
        num_segments_per_branch = 20
        segment_length = 0.03

        bend_stiffness = 1.0e0
        bend_damping = 1.0e-1
        stretch_stiffness = 1.0e9
        stretch_damping = 0.0

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e-1
        builder.default_shape_cfg.mu = 1.0

        cable_cfg = builder.default_shape_cfg.copy()

        z0 = 1.25
        junction = wp.vec3(0.0, 0.0, z0)
        dirs = _y_dirs_xy()

        # ------------------------------------------------------------
        # Build Y-junction graph.
        # ------------------------------------------------------------
        node_positions: list[Any] = [junction]
        edges: list[tuple[int, int]] = []
        for d in dirs:
            prev = 0
            for i in range(1, num_segments_per_branch + 1):
                p = junction + d * (float(i) * float(segment_length))
                node_positions.append(p)
                cur = len(node_positions) - 1
                edges.append((prev, cur))
                prev = cur

        self.graph_bodies, self.graph_joints = builder.add_rod_graph(
            node_positions=node_positions,
            edges=edges,
            radius=cable_radius,
            cfg=cable_cfg,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=stretch_damping,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            key="y_graph",
            wrap_in_articulation=True,
        )

        # Pin one tip capsule (end of the first branch).
        # Edges are created in contiguous blocks per branch in the construction loop above.
        tip_edge_idx = num_segments_per_branch - 1
        pinned_body = int(self.graph_bodies[tip_edge_idx])
        self.pinned_body = pinned_body
        builder.body_mass[pinned_body] = 0.0
        builder.body_inv_mass[pinned_body] = 0.0
        builder.body_inertia[pinned_body] = wp.mat33(0.0)
        builder.body_inv_inertia[pinned_body] = wp.mat33(0.0)

        if getattr(args, "ground", True):
            builder.add_ground_plane()

        builder.color(balance_colors=False)
        sim_device = wp.get_device(args.device) if args is not None and getattr(args, "device", None) else None
        self.model = builder.finalize(device=sim_device)
        self.model.set_gravity((0.0, 0.0, float(getattr(args, "gravity_z", -9.81))))

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            friction_epsilon=float(getattr(args, "friction_epsilon", 0.1)),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        if self.state_0.body_q is None:
            raise RuntimeError("Body state is not available.")
        self.pinned_body_q0 = self.state_0.body_q.numpy()[self.pinned_body].copy()

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.state_0.body_q is None or self.model.joint_parent is None or self.model.joint_child is None:
            raise RuntimeError("Body/joint state is not available.")

        rod_bodies = [int(b) for b in self.graph_bodies]

        # ---------------------------
        # Connectivity check
        # ---------------------------
        # `add_rod_graph(wrap_in_articulation=True)` builds a joint forest over the edge bodies.
        # For this Y-junction (one connected component), all rod bodies should be connected via
        # the joints returned by `add_rod_graph`.
        joint_parent = self.model.joint_parent.numpy()
        joint_child = self.model.joint_child.numpy()

        adjacency: dict[int, set[int]] = {b: set() for b in rod_bodies}
        for j in self.graph_joints:
            p = int(joint_parent[j])
            c = int(joint_child[j])
            if p in adjacency and c in adjacency:
                adjacency[p].add(c)
                adjacency[c].add(p)

        visited: set[int] = set()
        stack = [rod_bodies[0]]
        while stack:
            b = stack.pop()
            if b in visited:
                continue
            visited.add(b)
            stack.extend(adjacency[b] - visited)

        if len(visited) != len(rod_bodies):
            raise ValueError(f"Rod bodies are not fully connected (visited {len(visited)} / {len(rod_bodies)}).")

        # ---------------------------
        # Simple pose sanity checks
        # ---------------------------
        body_q_np = self.state_0.body_q.numpy()[rod_bodies]
        if not np.all(np.isfinite(body_q_np)):
            raise ValueError("NaN/Inf in cable body transforms.")

        pos = body_q_np[:, 0:3]
        if np.max(np.abs(pos[:, 0])) > 2.0 or np.max(np.abs(pos[:, 1])) > 2.0:
            raise ValueError("Cable bodies drifted too far in X/Y.")
        if np.min(pos[:, 2]) < -0.2 or np.max(pos[:, 2]) > 3.0:
            raise ValueError("Cable bodies out of Z bounds.")

        # Pinned body should not drift.
        q_now = self.state_0.body_q.numpy()[self.pinned_body]
        if np.max(np.abs(q_now[0:3] - self.pinned_body_q0[0:3])) > 1.0e-4:
            raise ValueError("Pinned tip body moved unexpectedly.")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
