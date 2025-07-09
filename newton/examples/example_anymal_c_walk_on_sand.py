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
# Example Anymal C walk Coupled with Sand
#
# Shows Anymal C with a pretrained policy coupled with implicit mpm sand.
#
# Example usage:
# uv run --extra cu12 newton/examples/example_anymal_c_walk_on_sand.py
###########################################################################

import sys

import numpy as np
import warp as wp

import newton
import newton.solvers.euler.kernels  # For graph capture on CUDA <12.3
import newton.solvers.euler.particles  # For graph capture on CUDA <12.3
import newton.solvers.solver  # For graph capture on CUDA <12.3
import newton.utils
from newton.examples.example_anymal_c_walk import AnymalController
from newton.solvers.implicit_mpm import ImplicitMPMSolver


@wp.kernel
def update_collider_mesh(
    src_points: wp.array(dtype=wp.vec3),
    src_shape: wp.array(dtype=int),
    res_mesh: wp.uint64,
    shape_transforms: wp.array(dtype=wp.transform),
    shape_body_id: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    dt: float,
):
    v = wp.tid()
    res = wp.mesh_get(res_mesh)

    shape_id = src_shape[v]
    p = wp.transform_point(shape_transforms[shape_id], src_points[v])

    X_wb = body_q[shape_body_id[shape_id]]

    cur_p = res.points[v] + dt * res.velocities[v]
    next_p = wp.transform_point(X_wb, p)
    res.velocities[v] = (next_p - cur_p) / dt
    res.points[v] = cur_p


class Example:
    def __init__(
        self,
        stage_path="example_anymal_c_walk_on_sand.usd",
        voxel_size=0.05,
        particles_per_cell=3,
        tolerance=1.0e-5,
        headless=False,
        sand_friction=0.48,
        dynamic_grid=True,
    ):
        self.device = wp.get_device()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anymal_c_simple_description")
        newton.utils.parse_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            builder,
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 60
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [
            0.0,
            0.7,
            0.0,
        ]

        builder.joint_q[7:] = [
            0.03,  # LF_HAA
            0.4,  # LF_HFE
            -0.8,  # LF_KFE
            -0.03,  # RF_HAA
            0.4,  # RF_HFE
            -0.8,  # RF_KFE
            0.03,  # LH_HAA
            -0.4,  # LH_HFE
            0.8,  # LH_KFE
            -0.03,  # RH_HAA
            -0.4,  # RH_HFE
            0.8,  # RH_KFE
        ]

        # add sand particles

        max_fraction = 1.0
        particle_lo = np.array([-0.5, 0.0, -0.5])
        particle_hi = np.array([2.5, 0.15, 0.5])
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        _spawn_particles(builder, particle_res, particle_lo, particle_hi, max_fraction)

        # finalize model
        self.model = builder.finalize()

        # the policy was trained with the following inertia tensors
        # fmt: off
        self.model.body_inertia = wp.array(
            [
                [[1.30548920,  0.00067627, 0.05068519], [ 0.000676270, 2.74363500,  0.00123380], [0.05068519,  0.00123380, 2.82926230]],
                [[0.01809368,  0.00826303, 0.00475366], [ 0.008263030, 0.01629626, -0.00638789], [0.00475366, -0.00638789, 0.02370901]],
                [[0.18137439, -0.00109795, 0.05645556], [-0.001097950, 0.20255709, -0.00183889], [0.05645556, -0.00183889, 0.02763401]],
                [[0.03070243,  0.00022458, 0.00102368], [ 0.000224580, 0.02828139, -0.00652076], [0.00102368, -0.00652076, 0.00269065]],
                [[0.01809368, -0.00825236, 0.00474725], [-0.008252360, 0.01629626,  0.00638789], [0.00474725,  0.00638789, 0.02370901]],
                [[0.18137439,  0.00111040, 0.05645556], [ 0.001110400, 0.20255709,  0.00183910], [0.05645556,  0.00183910, 0.02763401]],
                [[0.03070243, -0.00022458, 0.00102368], [-0.000224580, 0.02828139,  0.00652076], [0.00102368,  0.00652076, 0.00269065]],
                [[0.01809368, -0.00825236, 0.00474726], [-0.008252360, 0.01629626,  0.00638789], [0.00474726,  0.00638789, 0.02370901]],
                [[0.18137439,  0.00111041, 0.05645556], [ 0.001110410, 0.20255709,  0.00183909], [0.05645556,  0.00183909, 0.02763401]],
                [[0.03070243, -0.00022458, 0.00102368], [-0.000224580, 0.02828139,  0.00652076], [0.00102368,  0.00652076, 0.00269065]],
                [[0.01809368,  0.00826303, 0.00475366], [ 0.008263030, 0.01629626, -0.00638789], [0.00475366, -0.00638789, 0.02370901]],
                [[0.18137439, -0.00109796, 0.05645556], [-0.001097960, 0.20255709, -0.00183888], [0.05645556, -0.00183888, 0.02763401]],
                [[0.03070243,  0.00022458, 0.00102368], [ 0.000224580, 0.02828139, -0.00652076], [0.00102368, -0.00652076, 0.00269065]]
            ],
            dtype=wp.mat33f,
        )
        self.model.body_mass = wp.array([27.99286, 2.51203, 3.27327, 0.55505, 2.51203, 3.27327, 0.55505, 2.51203, 3.27327, 0.55505, 2.51203, 3.27327, 0.55505], dtype=wp.float32,)
        # fmt: on

        self.model.particle_mu = sand_friction

        ## Grab meshes for collisions
        collider_body_idx = [idx for idx, key in enumerate(builder.body_key) if "SHANK" in key]
        collider_shape_ids = np.concatenate(
            [[m for m in self.model.body_shapes[b] if self.model.shape_geo_src[m]] for b in collider_body_idx]
        )

        collider_points, collider_indices, collider_v_shape_ids = _merge_meshes(
            [self.model.shape_geo_src[m].vertices for m in collider_shape_ids],
            [self.model.shape_geo_src[m].indices for m in collider_shape_ids],
            [self.model.shape_geo.scale.numpy()[m] for m in collider_shape_ids],
            collider_shape_ids,
        )

        self.collider_mesh = wp.Mesh(wp.clone(collider_points), collider_indices, wp.zeros_like(collider_points))
        self.collider_rest_points = collider_points
        self.collider_shape_ids = wp.array(collider_v_shape_ids, dtype=int)

        self.solver = newton.solvers.FeatherstoneSolver(self.model)

        options = ImplicitMPMSolver.Options()
        options.voxel_size = voxel_size
        options.max_fraction = max_fraction
        options.tolerance = tolerance
        options.unilateral = False
        options.max_iterations = 50
        # options.gauss_seidel = False
        options.dynamic_grid = dynamic_grid
        if not dynamic_grid:
            options.grid_padding = 5

        self.mpm_solver = ImplicitMPMSolver(self.model, options)
        self.mpm_solver.setup_collider(self.model, [self.collider_mesh])

        self.renderer = None if headless else newton.utils.SimRendererOpenGL(self.model, stage_path)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.mpm_solver.enrich_state(self.state_0)
        self.mpm_solver.enrich_state(self.state_1)

        newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._update_collider_mesh(self.state_0)

        self.control = self.model.control()
        self.controller = AnymalController(self.model, self.device)
        self.controller.get_control(self.state_0)

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            # Initial graph launch, load modules (necessary for drivers prior to CUDA 12.3)
            wp.load_module(newton.solvers.euler.kernels, device=wp.get_device())
            wp.load_module(newton.solvers.euler.particles, device=wp.get_device())
            wp.load_module(newton.solvers.solver, device=wp.get_device())

            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.robot_graph = capture.graph
        else:
            self.robot_graph = None

    def simulate_robot(self):
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.controller.assign_control(self.control, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        self._update_collider_mesh(self.state_0)
        # solve in-place, avoids having to resync robot sim state
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def step(self):
        with wp.ScopedTimer("step", synchronize=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.robot_graph)
            else:
                self.simulate_robot()

            self.simulate_sand()
            self.controller.get_control(self.state_0)

        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", synchronize=True):
            self.renderer.begin_frame(self.sim_time)

            self.renderer.render(self.state_0)

            self.renderer.end_frame()

    def _update_collider_mesh(self, state):
        wp.launch(
            update_collider_mesh,
            dim=self.collider_rest_points.shape[0],
            inputs=[
                self.collider_rest_points,
                self.collider_shape_ids,
                self.collider_mesh.id,
                self.model.shape_transform,
                self.model.shape_body,
                state.body_q,
                self.frame_dt,
            ],
        )
        self.collider_mesh.refit()


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, packing_fraction):
    Nx = res[0]
    Ny = res[1]
    Nz = res[2]

    px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

    points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)

    radius = np.max(cell_size) * 0.5
    volume = np.prod(cell_volume) * packing_fraction

    rng = np.random.default_rng()
    points += 2.0 * radius * (rng.random(points.shape) - 0.5)
    vel = np.zeros_like(points)

    builder.particle_q = points
    builder.particle_qd = vel
    builder.particle_mass = np.full(points.shape[0], volume)
    builder.particle_radius = np.full(points.shape[0], radius)
    builder.particle_flags = np.zeros(points.shape[0], dtype=int)

    print("Particle count: ", points.shape[0])


def _merge_meshes(
    points: list[np.array],
    indices: list[np.array],
    scales: list[np.array],
    shape_ids: list[int],
):
    pt_count = np.array([len(pts) for pts in points])
    offsets = np.cumsum(pt_count) - pt_count

    mesh_id = np.repeat(np.arange(len(points), dtype=int), repeats=pt_count)

    merged_points = np.vstack([pts * scale for pts, scale in zip(points, scales)])

    merged_indices = np.concatenate([idx + offsets[k] for k, idx in enumerate(indices)])

    return (
        wp.array(merged_points, dtype=wp.vec3),
        wp.array(merged_indices, dtype=int),
        wp.array(np.array(shape_ids)[mesh_id], dtype=int),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_anymal_c_walk_on_sand.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=10000, help="Total number of frames.")
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)
    parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
    parser.add_argument("--sand-friction", "-mu", type=float, default=0.48)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_known_args()[0]

    if wp.get_device(args.device).is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    with wp.ScopedDevice(args.device):
        example = Example(
            voxel_size=args.voxel_size,
            particles_per_cell=args.particles_per_cell,
            tolerance=args.tolerance,
            headless=args.headless,
            sand_friction=args.sand_friction,
            dynamic_grid=args.dynamic_grid,
        )

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
