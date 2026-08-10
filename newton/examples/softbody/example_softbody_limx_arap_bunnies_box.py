# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody LIMX ARAP Bunnies Box
#
# Eight volumetric ARAP bunnies fall into an open box. A shared VF/EE operator
# uses local geometry caps for one-ring pairs and uniform 3 mm thickness for
# nonlocal pairs. Every 0.01 s frame uses one Newton increment and 50 PCG steps.
#
# Command: uv run -m newton.examples softbody_limx_arap_bunnies_box
#
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

_BUNNY_CONFIGURATIONS = (
    ((-0.11, -0.17, 0.25), (0.0, 0.0, 0.0), (0.08, 0.04, -0.10)),
    ((0.11, -0.17, 0.25), (4.0, -3.0, 6.0), (-0.06, 0.03, -0.12)),
    ((-0.11, 0.17, 0.25), (-5.0, 4.0, -7.0), (0.05, -0.04, -0.08)),
    ((0.11, 0.17, 0.25), (3.0, 5.0, 4.0), (-0.07, -0.03, -0.11)),
    ((-0.11, -0.17, 0.52), (-4.0, -5.0, 8.0), (0.06, 0.02, -0.20)),
    ((0.11, -0.17, 0.52), (5.0, 3.0, -5.0), (-0.05, 0.04, -0.18)),
    ((-0.11, 0.17, 0.52), (2.0, -6.0, 6.0), (0.04, -0.03, -0.22)),
    ((0.11, 0.17, 0.52), (-3.0, 4.0, -8.0), (-0.06, -0.02, -0.19)),
)


@wp.kernel
def _mark_cross_bunny_contacts(
    contact_ids: wp.array2d[int],
    contact_count: wp.array[int],
    contact_capacity: int,
    arity: int,
    feature_split: int,
    particles_per_bunny: int,
    saw_cross_contact: wp.array[int],
):
    contact = wp.tid()
    active_count = contact_count[0]
    if active_count > contact_capacity:
        active_count = contact_capacity
    if contact >= active_count:
        return

    bunny_0 = contact_ids[contact, 0] // particles_per_bunny
    for local_index in range(feature_split, arity):
        if contact_ids[contact, local_index] // particles_per_bunny != bunny_0:
            wp.atomic_max(saw_cross_contact, 0, 1)


class Example:
    """Drop eight mutually colliding volumetric bunnies into an open box."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 0.01
        self.sim_time = 0.0
        self.box_min = np.asarray([-0.36, -0.40], dtype=np.float32)
        self.box_max = np.asarray([0.36, 0.40], dtype=np.float32)
        self.box_floor = 0.0
        self.box_wall_top = 0.75
        self.bunny_count = len(_BUNNY_CONFIGURATIONS)
        self.maximum_contact_counts = np.zeros(2, dtype=np.int64)
        self.minimum_determinant = np.inf
        self.maximum_box_penetration = 0.0
        self.maximum_speed = 0.0

        mesh_path = Path(__file__).resolve().parents[1] / "assets" / "bunny_tet.npz"
        mesh = newton.TetMesh.create_from_file(str(mesh_path))
        self.particles_per_bunny = mesh.vertex_count

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        source_to_world = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        for center, rpy_degrees, velocity in _BUNNY_CONFIGURATIONS:
            perturbation = wp.quat_rpy(*(math.radians(value) for value in rpy_degrees))
            rotation = wp.normalize(perturbation * source_to_world)
            builder.add_soft_mesh(
                pos=wp.vec3(*center),
                rot=rotation,
                scale=0.15,
                vel=wp.vec3(*velocity),
                mesh=mesh,
                density=1000.0,
                k_mu=0.0,
                k_lambda=0.0,
                k_damp=0.0,
                add_surface_mesh_edges=False,
            )
        self._add_box_shapes(builder)
        self.model = builder.finalize()

        self.tetrahedra = self.model.tet_indices.numpy()
        self.surface_vertex_indices = np.unique(self.model.tri_indices.numpy()).astype(np.int32)
        inverse_rest_matrices = self.model.tet_poses.numpy()
        self.arap_constraint = newton.solvers.ConstraintTetrahedronARAP(
            self.tetrahedra.tolist(),
            [wp.mat33(*matrix.reshape(-1)) for matrix in inverse_rest_matrices],
            [3.0e5] * self.model.tet_count,
            self.model.particle_count,
            self.model.device,
        )
        self.self_collision = newton.solvers.ConstraintSelfCollision(
            self.model,
            thickness=0.003,
            stiffness=None,
            max_contacts=262144,
            stiffness_factors=(0.5, 0.3, 1.5),
            geometry_radius_scale=0.25,
            geometry_radius_topology_local_only=True,
            friction=0.0,
            friction_epsilon=1.0e-2,
            enable_edge_face=False,
            use_outward_normals=True,
        )
        plane_parameters = (
            ((0.0, 0.0, 1.0), self.box_floor),
            ((1.0, 0.0, 0.0), float(self.box_min[0])),
            ((-1.0, 0.0, 0.0), -float(self.box_max[0])),
            ((0.0, 1.0, 0.0), float(self.box_min[1])),
            ((0.0, -1.0, 0.0), -float(self.box_max[1])),
        )
        self.box_contacts = [
            newton.solvers.ConstraintStaticPlaneContact(
                normal=normal,
                offset=offset,
                thickness=0.003,
                stiffness=2.0e4,
                normal_damping=0.0,
                friction=0.05,
                friction_epsilon=1.0e-4,
                particle_count=self.model.particle_count,
                device=self.model.device,
                particle_indices=self.surface_vertex_indices,
            )
            for normal, offset in plane_parameters
        ]
        dynamic_constraints = newton.solvers.ConstraintGroupDynamic([self.self_collision, *self.box_contacts])
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            [self.arap_constraint],
            nonlinear_iterations=1,
            linear_iterations=50,
            velocity_damping=1.0,
            dynamic_operator=dynamic_constraints,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.saw_cross_bunny_contact = wp.zeros(1, dtype=wp.int32, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.25, -1.55, 1.08), -18.0, 140.0)

    def _add_box_shapes(self, builder: newton.ModelBuilder) -> None:
        floor_half_thickness = 0.05
        floor_margin = 0.05
        box_color = wp.vec3(0.34, 0.22, 0.12)

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, self.box_floor - floor_half_thickness), wp.quat_identity()),
            hx=float(self.box_max[0]) + floor_margin,
            hy=float(self.box_max[1]) + floor_margin,
            hz=floor_half_thickness,
            color=box_color,
        )

    def step(self):
        """Advance one undamped 0.01-second Newton step."""
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        """Render all bunnies and the floor."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Keep all bunnies finite, positive-volume, and contained by the box."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("LIMX ARAP bunnies state must remain finite")

        edges = np.stack(
            (
                positions[self.tetrahedra[:, 1]] - positions[self.tetrahedra[:, 0]],
                positions[self.tetrahedra[:, 2]] - positions[self.tetrahedra[:, 0]],
                positions[self.tetrahedra[:, 3]] - positions[self.tetrahedra[:, 0]],
            ),
            axis=2,
        )
        minimum_determinant = float(np.linalg.det(edges).min())
        self.minimum_determinant = min(self.minimum_determinant, minimum_determinant)
        if minimum_determinant <= 0.0:
            raise AssertionError("LIMX ARAP bunnies tetrahedra must remain positive-volume")

        floor_penetration = self.box_floor - float(positions[:, 2].min())
        below_wall_top = positions[:, 2] <= self.box_wall_top
        contained = positions[below_wall_top]
        wall_penetration = 0.0
        if len(contained) > 0:
            wall_penetration = max(
                float(self.box_min[0]) - float(contained[:, 0].min()),
                float(contained[:, 0].max()) - float(self.box_max[0]),
                float(self.box_min[1]) - float(contained[:, 1].min()),
                float(contained[:, 1].max()) - float(self.box_max[1]),
                0.0,
            )
        self.maximum_box_penetration = max(self.maximum_box_penetration, floor_penetration, wall_penetration)
        if floor_penetration > 0.04:
            raise AssertionError("A bunny penetrated catastrophically below the box")
        if wall_penetration > 0.04:
            raise AssertionError("A bunny escaped catastrophically through a box wall")

        buffers = (
            self.self_collision.vertex_face_contacts,
            self.self_collision.edge_edge_contacts,
        )
        contact_counts = []
        for buffer in buffers:
            overflow_count = int(buffer.overflow_count.numpy()[0])
            if overflow_count != 0:
                raise AssertionError("LIMX ARAP bunny contact capacity overflowed")
            contact_counts.append(min(int(buffer.count.numpy()[0]), buffer.capacity))
        self.maximum_contact_counts = np.maximum(self.maximum_contact_counts, contact_counts)
        self.maximum_speed = max(self.maximum_speed, float(np.linalg.norm(velocities, axis=1).max()))

        wp.launch(
            _mark_cross_bunny_contacts,
            dim=self.self_collision.max_contacts,
            inputs=[
                self.self_collision.vertex_face_contacts.ids,
                self.self_collision.vertex_face_contacts.count,
                self.self_collision.max_contacts,
                4,
                1,
                self.particles_per_bunny,
            ],
            outputs=[self.saw_cross_bunny_contact],
            device=self.model.device,
        )
        wp.launch(
            _mark_cross_bunny_contacts,
            dim=self.self_collision.max_contacts,
            inputs=[
                self.self_collision.edge_edge_contacts.ids,
                self.self_collision.edge_edge_contacts.count,
                self.self_collision.max_contacts,
                4,
                2,
                self.particles_per_bunny,
            ],
            outputs=[self.saw_cross_bunny_contact],
            device=self.model.device,
        )

    def test_final(self):
        """Record at least one cross-bunny VF or EE contact without instability."""
        self.test_post_step()
        if self.sim_time <= 0.0:
            raise AssertionError("LIMX ARAP bunnies scene did not advance")
        if self.sim_time >= 0.5 and int(self.saw_cross_bunny_contact.numpy()[0]) == 0:
            raise AssertionError("LIMX ARAP bunnies never generated a cross-bunny VF or EE contact")

    @staticmethod
    def create_parser():
        """Create the standard Newton example parser."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
