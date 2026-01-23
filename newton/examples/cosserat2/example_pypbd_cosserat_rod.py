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
# Example PyPBD Cosserat Rod
#
# Demonstrates Position and Orientation Based Cosserat Rod simulation using
# the pypbd library (Python bindings for Position Based Dynamics).
#
# This example creates a helix-shaped Cosserat rod using pypbd's:
# - StretchShear constraints: maintain edge lengths and shear resistance
# - BendTwist constraints: maintain bending and twisting stiffness
#
# The rod uses quaternions to track local material frame orientations,
# enabling accurate simulation of bending, twisting, and shearing.
#
# Command: uv run -m newton.examples pypbd_cosserat_rod
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples

try:    import pypbd as pbd
except ImportError:
    raise ImportError(
        "pypbd is required for this example. "
        "Install it with: pip install pypbd"
    )


class Example:
    def __init__(self, viewer, args=None):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8  # Number of pypbd steps per frame

        self.viewer = viewer
        self.args = args

        # Rod parameters
        self.num_particles = 50
        self.num_quaternions = self.num_particles - 1
        self.num_edges = self.num_particles - 1

        # Helix parameters
        self.helix_radius = 0.5
        self.helix_height = 5.0
        self.helix_total_angle = 10.0 * math.pi

        # Stiffness parameters (0 to 1)
        self.stretching_stiffness = 1.0
        self.shearing_stiffness_x = 0.5
        self.shearing_stiffness_y = 0.5
        self.twisting_stiffness = 0.5
        self.bending_stiffness_x = 0.5
        self.bending_stiffness_y = 0.5

        # Initialize pypbd
        pbd.Logger.addConsoleSink(pbd.LogLevel.INFO)

        self.sim = pbd.Simulation.getCurrent()
        self.sim.initDefault()
        self.model = self.sim.getModel()

        # Configure time step controller
        ts = self.sim.getTimeStep()
        ts.setValueUInt(pbd.TimeStepController.NUM_SUB_STEPS, 3)
        ts.setValueUInt(pbd.TimeStepController.MAX_ITERATIONS, 5)

        # Track index offsets for particles and quaternions
        pd = self.model.getParticles()
        self.particle_offset = pd.getNumberOfParticles()

        od = self.model.getOrientations()
        self.quaternion_offset = od.getNumberOfQuaternions()

        # Build the Cosserat rod
        self._build_rod()

        # Build Newton model for visualization
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add particles to Newton model for visualization
        for i in range(self.num_particles):
            pos = pd.getPosition(i + self.particle_offset)
            mass = 0.0 if i == 0 else 1.0
            builder.add_particle(
                pos=(pos[0], pos[1], pos[2]),
                vel=(0.0, 0.0, 0.0),
                mass=mass,
                radius=0.05,
            )

        self.newton_model = builder.finalize()
        self.state = self.newton_model.state()

        # Initialize viewer
        self.viewer.set_model(self.newton_model)
        self.viewer.show_particles = True

        # Director visualization buffers
        device = self.newton_model.device
        num_director_lines = self.num_quaternions * 3
        self.director_line_starts = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_ends = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.director_line_colors = wp.zeros(num_director_lines, dtype=wp.vec3, device=device)
        self.show_directors = True
        self.director_scale = 0.15

    def _build_rod(self):
        """Build the Cosserat rod as a helix."""
        # Generate helix particle positions
        points = []
        for i in range(self.num_particles):
            t = i / self.num_particles
            angle = self.helix_total_angle * t
            x = self.helix_radius * math.cos(angle)
            y = self.helix_radius * math.sin(angle)
            z = self.helix_height * t
            points.append([x, y, z])

        # Generate quaternions for each edge
        # Each quaternion represents the local material frame orientation
        quaternions = []
        from_vec = np.array([0.0, 0.0, 1.0])  # Default tangent direction

        for i in range(self.num_quaternions):
            # Compute edge direction (tangent)
            p0 = np.array(points[i])
            p1 = np.array(points[i + 1])
            to_vec = p1 - p0
            to_vec = to_vec / np.linalg.norm(to_vec)

            # Compute quaternion that rotates from_vec to to_vec
            q = self._quaternion_from_two_vectors(from_vec, to_vec)

            if i == 0:
                quaternions.append(q)
            else:
                # Compose with previous quaternion
                q_prev = quaternions[i - 1]
                q_composed = self._quaternion_multiply(q, q_prev)
                quaternions.append(q_composed)

            from_vec = to_vec

        # Create indices for particle connectivity
        # Format: [v0, v1, v1, v2, v2, v3, ...] - but pypbd uses [v0, v1] for each edge
        indices = []
        for i in range(self.num_particles - 1):
            indices.append(i)
            indices.append(i + 1)

        # Create indices for quaternions (one per edge)
        indices_quaternions = list(range(self.num_quaternions))

        # Add line model to pypbd
        self.model.addLineModel(
            self.num_particles,
            self.num_quaternions,
            points,
            quaternions,
            indices,
            indices_quaternions,
        )

        # Set particle masses
        pd = self.model.getParticles()
        for i in range(self.num_particles):
            idx = i + self.particle_offset
            if i == 0:
                pd.setMass(idx, 0.0)  # First particle is static
            else:
                pd.setMass(idx, 1.0)

        # Set quaternion masses
        od = self.model.getOrientations()
        for i in range(self.num_quaternions):
            idx = i + self.quaternion_offset
            if i == 0:
                od.setMass(idx, 0.0)  # First quaternion is static
            else:
                od.setMass(idx, 1.0)

        # Add StretchShear constraints for each edge
        # Each edge connects particles (i, i+1) and uses quaternion i
        for i in range(self.num_edges):
            v1 = i + self.particle_offset
            v2 = i + 1 + self.particle_offset
            q1 = i + self.quaternion_offset
            self.model.addStretchShearConstraint(
                v1, v2, q1,
                self.stretching_stiffness,
                self.shearing_stiffness_x,
                self.shearing_stiffness_y,
            )

        # Add BendTwist constraints between consecutive quaternions
        for i in range(self.num_edges - 1):
            q1 = i + self.quaternion_offset
            q2 = i + 1 + self.quaternion_offset
            self.model.addBendTwistConstraint(
                q1, q2,
                self.twisting_stiffness,
                self.bending_stiffness_x,
                self.bending_stiffness_y,
            )

    def _quaternion_from_two_vectors(self, v1, v2):
        """Compute quaternion that rotates v1 to v2."""
        v1 = np.array(v1, dtype=np.float64)
        v2 = np.array(v2, dtype=np.float64)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        dot = np.dot(v1, v2)

        if dot > 0.999999:
            # Vectors are nearly parallel
            return [0.0, 0.0, 0.0, 1.0]
        elif dot < -0.999999:
            # Vectors are nearly opposite
            # Find an orthogonal vector
            if abs(v1[0]) < 0.9:
                ortho = np.cross(v1, [1, 0, 0])
            else:
                ortho = np.cross(v1, [0, 1, 0])
            ortho = ortho / np.linalg.norm(ortho)
            return [ortho[0], ortho[1], ortho[2], 0.0]

        cross = np.cross(v1, v2)
        w = 1.0 + dot
        q = [cross[0], cross[1], cross[2], w]
        # Normalize
        norm = math.sqrt(sum(x * x for x in q))
        return [x / norm for x in q]

    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions (x, y, z, w format)."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]

    def simulate(self):
        """Run pypbd simulation steps."""
        for _ in range(self.sim_substeps):
            self.sim.getTimeStep().step(self.model)

    def _sync_state(self):
        """Sync pypbd particle positions to Newton state."""
        pd = self.model.getParticles()
        positions = np.zeros((self.num_particles, 3), dtype=np.float32)

        for i in range(self.num_particles):
            pos = pd.getPosition(i + self.particle_offset)
            positions[i] = [pos[0], pos[1], pos[2]]

        self.state.particle_q = wp.array(positions, dtype=wp.vec3, device=self.newton_model.device)

    def _update_director_visualization(self):
        """Update director lines for visualization."""
        od = self.model.getOrientations()
        pd = self.model.getParticles()

        starts = []
        ends = []
        colors = []

        for i in range(self.num_quaternions):
            # Get edge midpoint
            p0 = pd.getPosition(i + self.particle_offset)
            p1 = pd.getPosition(i + 1 + self.particle_offset)
            midpoint = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2]

            # Get quaternion
            q = od.getQuaternion(i + self.quaternion_offset)

            # Compute directors (d1, d2, d3) from quaternion
            # d1 = q * (1,0,0) * q^-1
            # d2 = q * (0,1,0) * q^-1
            # d3 = q * (0,0,1) * q^-1
            d1 = self._rotate_vector_by_quaternion([1, 0, 0], q)
            d2 = self._rotate_vector_by_quaternion([0, 1, 0], q)
            d3 = self._rotate_vector_by_quaternion([0, 0, 1], q)

            scale = self.director_scale

            # d1 - red
            starts.append(midpoint)
            ends.append([midpoint[0] + d1[0] * scale, midpoint[1] + d1[1] * scale, midpoint[2] + d1[2] * scale])
            colors.append([1.0, 0.0, 0.0])

            # d2 - green
            starts.append(midpoint)
            ends.append([midpoint[0] + d2[0] * scale, midpoint[1] + d2[1] * scale, midpoint[2] + d2[2] * scale])
            colors.append([0.0, 1.0, 0.0])

            # d3 - blue
            starts.append(midpoint)
            ends.append([midpoint[0] + d3[0] * scale, midpoint[1] + d3[1] * scale, midpoint[2] + d3[2] * scale])
            colors.append([0.0, 0.0, 1.0])

        self.director_line_starts = wp.array(starts, dtype=wp.vec3, device=self.newton_model.device)
        self.director_line_ends = wp.array(ends, dtype=wp.vec3, device=self.newton_model.device)
        self.director_line_colors = wp.array(colors, dtype=wp.vec3, device=self.newton_model.device)

    def _rotate_vector_by_quaternion(self, v, q):
        """Rotate vector v by quaternion q (x, y, z, w format)."""
        x, y, z, w = q[0], q[1], q[2], q[3]
        vx, vy, vz = v[0], v[1], v[2]

        # q * v * q^-1 (optimized formula)
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)

        return [
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        ]

    def step(self):
        """Advance simulation by one frame."""
        self.simulate()
        self._sync_state()
        self.sim_time += self.frame_dt

    def render(self):
        """Render current state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)

        # Visualize material frames (directors)
        if self.show_directors:
            self._update_director_visualization()
            self.viewer.log_lines(
                "/directors",
                self.director_line_starts,
                self.director_line_ends,
                self.director_line_colors,
            )
        else:
            self.viewer.log_lines("/directors", None, None, None)

        self.viewer.end_frame()

    def gui(self, ui):
        """GUI controls."""
        ui.text("PyPBD Cosserat Rod")
        ui.text(f"Particles: {self.num_particles}")
        ui.text(f"Simulation time: {self.sim_time:.2f}s")
        ui.text(f"PyPBD time: {pbd.TimeManager.getCurrent().getTime():.2f}s")

        ui.separator()
        ui.text("Visualization")
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.05, 0.5)

    def test_final(self):
        """Validation method run after simulation completes."""
        # Verify that the first particle (anchor) hasn't moved significantly
        pd = self.model.getParticles()

        anchor_pos = pd.getPosition(self.particle_offset)
        initial_pos = [
            self.helix_radius * math.cos(0),
            self.helix_radius * math.sin(0),
            0.0,
        ]

        dist = math.sqrt(
            (anchor_pos[0] - initial_pos[0]) ** 2
            + (anchor_pos[1] - initial_pos[1]) ** 2
            + (anchor_pos[2] - initial_pos[2]) ** 2
        )
        assert dist < 0.01, f"Anchor particle moved: distance = {dist}"

        # Verify all quaternions are normalized
        od = self.model.getOrientations()

        for i in range(self.num_quaternions):
            q = od.getQuaternion(i + self.quaternion_offset)
            norm = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
            assert abs(norm - 1.0) < 0.1, f"Quaternion {i} not normalized: norm = {norm}"


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Enable particle visualization if using GL viewer
    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    # Create and run example
    example = Example(viewer, args)

    newton.examples.run(example, args)
