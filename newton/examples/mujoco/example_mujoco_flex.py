# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example MuJoCo Flex

Simulates Newton deformables as MuJoCo flexes. A cloth (2D flex), a cable
(1D flex held by an edge equality constraint), and a soft block (3D flex)
fall inside ``SolverMuJoCo``: the cloth drapes over a rigid box, and the
cable lands across the soft block. The cloth and cable are authored with
:func:`newton.solvers.mujoco.add_flex`; the block's tetrahedra are converted
automatically.

Success criteria:
- The cloth drapes over the box, and the cable drapes across the soft block next to it.
- No deformable passes through the box or the ground, and the cable stays near its rest length.

Runs:
- `python -m newton.examples mujoco_flex`
- `python -m newton.examples mujoco_flex --world-count 16`
"""

import itertools

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCo, mujoco

BOX_HALF = 0.15
CLOTH_CELLS, CLOTH_SIZE = 16, 0.6
CABLE_SEGMENTS, CABLE_LENGTH = 30, 0.9
PARTICLE_RADIUS = 0.01
# A cloth flex radius of about half the cloth spacing keeps its triangles outside the box edges.
CLOTH_RADIUS = 0.02
CLOTH_COLOR = (0.4, 0.8, 0.93)
BLOCK_COLOR = (0.13, 0.53, 0.2)
CABLE_COLOR = (0.85, 0.3, 0.2)


@wp.kernel(enable_backward=False)
def cable_render_points(points: wp.array[wp.vec3], starts: wp.array[wp.int32], out: wp.array2d[wp.vec3]):
    """Sample each world's cable at its particles and segment midpoints so the spheres overlap into a rope."""
    world, j = wp.tid()
    a = starts[world] + j // 2
    out[world, j] = 0.5 * (points[a] + points[a + j % 2])


def build_world() -> newton.ModelBuilder:
    """Build one world: a cloth above a rigid box, and a cable above a soft block beside it."""
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    # MuJoCo Warp 4.0.1 and earlier sink flexes into shapes by the shape's gap, so keep gaps at zero.
    builder.default_shape_cfg.gap = 0.0

    box = builder.add_body(xform=wp.transform((0.0, 0.0, BOX_HALF), wp.quat_identity()), label="box")
    builder.add_shape_box(box, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)

    start = builder.tri_count
    builder.add_cloth_grid(
        pos=wp.vec3(-0.5 * CLOTH_SIZE, -0.5 * CLOTH_SIZE, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=CLOTH_CELLS,
        dim_y=CLOTH_CELLS,
        cell_x=CLOTH_SIZE / CLOTH_CELLS,
        cell_y=CLOTH_SIZE / CLOTH_CELLS,
        mass=0.002,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        particle_radius=PARTICLE_RADIUS,
        color=CLOTH_COLOR,
    )
    mujoco.add_flex(
        builder, triangles=range(start, builder.tri_count), elastic2d="both", thickness=1.0e-3, radius=CLOTH_RADIUS
    )

    segment = CABLE_LENGTH / CABLE_SEGMENTS
    cable = [
        builder.add_particle(
            (0.45, -0.5 * CABLE_LENGTH + i * segment, 0.5), (0.0, 0.0, 0.0), 0.005, radius=PARTICLE_RADIUS
        )
        for i in range(CABLE_SEGMENTS + 1)
    ]
    start = builder.spring_count
    for a, b in itertools.pairwise(cable):
        builder.add_spring(a, b, 0.0, 0.0, 0.0)
    mujoco.add_flex(builder, springs=range(start, builder.spring_count), edgeequality=True)

    builder.add_soft_grid(
        pos=wp.vec3(0.35, -0.1, 0.1),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        dim_z=4,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        density=200.0,
        k_mu=2.0e4,
        k_lambda=2.0e4,
        k_damp=0.0,
        particle_radius=PARTICLE_RADIUS,
        color=BLOCK_COLOR,
    )
    return builder


class Example:
    def __init__(self, viewer, args):
        """Build the replicated flex scene and the MuJoCo solver."""
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.world_count = args.world_count

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.gap = 0.0
        builder.replicate(build_world(), self.world_count, spacing=(2.0, 2.0, 0.0))
        builder.add_ground_plane()
        self.model = builder.finalize()

        # CG avoids factorizing the dense Hessian that flex DOFs make large.
        self.solver = SolverMuJoCo(self.model, solver="cg", iterations=200, ls_iterations=20, njmax=3000, nconmax=1500)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # The viewer draws the cloth and block triangles; draw the cable as overlapping spheres.
        per_world = self.model.particle_count // self.world_count
        starts = (CLOTH_CELLS + 1) ** 2 + per_world * np.arange(self.world_count)
        self.cable_starts = wp.array(starts, dtype=wp.int32)
        self.cable_points = wp.empty((self.world_count, 2 * CABLE_SEGMENTS + 1), dtype=wp.vec3)
        count = self.cable_points.size
        self.cable_radii = wp.full(count, 1.2 * PARTICLE_RADIUS, dtype=wp.float32)
        self.cable_colors = wp.full(count, wp.vec3(CABLE_COLOR), dtype=wp.vec3)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.0, -1.0, 0.7), pitch=-25.0, yaw=135.0)

        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Run the substeps of one rendered frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the scene by one rendered frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the box and the deformables."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        wp.launch(
            cable_render_points,
            self.cable_points.shape,
            inputs=[self.state_0.particle_q, self.cable_starts],
            outputs=[self.cable_points],
        )
        self.viewer.log_points("cable", self.cable_points.flatten(), self.cable_radii, self.cable_colors)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the deformables rest on the box and ground in every world."""
        q = self.state_0.particle_q.numpy().reshape(self.world_count, -1, 3)
        assert np.isfinite(q).all()
        # No particle sinks into the ground beyond its radius.
        assert q[..., 2].min() > -PARTICLE_RADIUS
        # The cloth center rests on top of the box instead of passing through it.
        cloth = CLOTH_CELLS + 1
        center = (cloth // 2) * cloth + cloth // 2
        assert np.all(q[:, center, 2] > 2.0 * BOX_HALF), q[:, center, 2]
        # The cable's edge constraint keeps its segments near their rest length.
        cable = slice(cloth * cloth, cloth * cloth + CABLE_SEGMENTS + 1)
        segment = np.linalg.norm(np.diff(q[:, cable], axis=1), axis=-1)
        np.testing.assert_allclose(segment, CABLE_LENGTH / CABLE_SEGMENTS, rtol=0.1)

    @staticmethod
    def create_parser():
        """Create command-line arguments for the number of worlds."""
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
