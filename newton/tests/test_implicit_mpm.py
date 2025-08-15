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

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_sand_cube_on_plane(test, device):
    # Emits a cube of particles on the ground

    N = 4
    particles_per_cell = 3
    voxel_size = 0.5
    particle_spacing = voxel_size / particles_per_cell
    friction = 0.6
    dt = 0.04

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    builder.add_particle_grid(
        pos=wp.vec3(0.5 * particle_spacing),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=N * particles_per_cell,
        dim_y=N * particles_per_cell,
        dim_z=N * particles_per_cell,
        cell_x=particle_spacing,
        cell_y=particle_spacing,
        cell_z=particle_spacing,
        mass=1.0,
        jitter=0.0,
    )
    builder.add_ground_plane()

    model: newton.Model = builder.finalize(device=device)
    model.particle_mu = friction

    state_0: newton.State = model.state()
    state_1: newton.State = model.state()

    options = SolverImplicitMPM.Options()
    options.dynamic_grid = False  # use static grid as dynamic grid is GPU-only
    options.grid_padding = 3
    options.voxel_size = voxel_size

    solver = SolverImplicitMPM(model, options)

    solver.enrich_state(state_0)
    solver.enrich_state(state_1)

    # Run a few steps
    for _k in range(25):
        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    # Checks the final bounding box corresponds to the expected collapse
    end_pos = state_0.particle_q.numpy()
    bb_min, bb_max = np.min(end_pos, axis=0), np.max(end_pos, axis=0)
    assert bb_min[model.up_axis] > -voxel_size
    assert voxel_size < bb_max[model.up_axis] < N * voxel_size

    assert np.all(bb_min > -N * voxel_size)
    assert np.all(bb_min < 0.0)
    assert np.all(bb_max < 2 * N * voxel_size)


devices = get_test_devices(mode="basic")


class TestImplicitMPM(unittest.TestCase):
    pass


add_function_test(
    TestImplicitMPM, "test_sand_cube_on_plane", test_sand_cube_on_plane, devices=devices, check_output=False
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
