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
from functools import partial

import numpy as np
import warp as wp

import newton
from newton import ParticleFlags
from newton.tests.unittest_utils import add_function_test, get_test_devices

# fmt: off
CLOTH_POINTS = [
    (-50.0000000, 0.0000000, -50.0000000),
    (-38.8888893, 11.1111107, -50.0000000),
    (-27.7777786, 22.2222214, -50.0000000),
    (-16.6666679, 33.3333321, -50.0000000),
    (-5.5555558, 44.4444427, -50.0000000),
    (5.5555558, 55.5555573, -50.0000000),
    (16.6666679, 66.6666641, -50.0000000),
    (27.7777786, 77.7777786, -50.0000000),
    (38.8888893, 88.8888855, -50.0000000),
    (50.0000000, 100.0000000, -50.0000000),
    (-50.0000000, 0.0000000, -38.8888893),
    (-38.8888893, 11.1111107, -38.8888893),
    (-27.7777786, 22.2222214, -38.8888893),
    (-16.6666679, 33.3333321, -38.8888893),
    (-5.5555558, 44.4444427, -38.8888893),
    (5.5555558, 55.5555573, -38.8888893),
    (16.6666679, 66.6666641, -38.8888893),
    (27.7777786, 77.7777786, -38.8888893),
    (38.8888893, 88.8888855, -38.8888893),
    (50.0000000, 100.0000000, -38.8888893),
    (-50.0000000, 0.0000000, -27.7777786),
    (-38.8888893, 11.1111107, -27.7777786),
    (-27.7777786, 22.2222214, -27.7777786),
    (-16.6666679, 33.3333321, -27.7777786),
    (-5.5555558, 44.4444427, -27.7777786),
    (5.5555558, 55.5555573, -27.7777786),
    (16.6666679, 66.6666641, -27.7777786),
    (27.7777786, 77.7777786, -27.7777786),
    (38.8888893, 88.8888855, -27.7777786),
    (50.0000000, 100.0000000, -27.7777786),
    (-50.0000000, 0.0000000, -16.6666679),
    (-38.8888893, 11.1111107, -16.6666679),
    (-27.7777786, 22.2222214, -16.6666679),
    (-16.6666679, 33.3333321, -16.6666679),
    (-5.5555558, 44.4444427, -16.6666679),
    (5.5555558, 55.5555573, -16.6666679),
    (16.6666679, 66.6666641, -16.6666679),
    (27.7777786, 77.7777786, -16.6666679),
    (38.8888893, 88.8888855, -16.6666679),
    (50.0000000, 100.0000000, -16.6666679),
    (-50.0000000, 0.0000000, -5.5555558),
    (-38.8888893, 11.1111107, -5.5555558),
    (-27.7777786, 22.2222214, -5.5555558),
    (-16.6666679, 33.3333321, -5.5555558),
    (-5.5555558, 44.4444427, -5.5555558),
    (5.5555558, 55.5555573, -5.5555558),
    (16.6666679, 66.6666641, -5.5555558),
    (27.7777786, 77.7777786, -5.5555558),
    (38.8888893, 88.8888855, -5.5555558),
    (50.0000000, 100.0000000, -5.5555558),
    (-50.0000000, 0.0000000, 5.5555558),
    (-38.8888893, 11.1111107, 5.5555558),
    (-27.7777786, 22.2222214, 5.5555558),
    (-16.6666679, 33.3333321, 5.5555558),
    (-5.5555558, 44.4444427, 5.5555558),
    (5.5555558, 55.5555573, 5.5555558),
    (16.6666679, 66.6666641, 5.5555558),
    (27.7777786, 77.7777786, 5.5555558),
    (38.8888893, 88.8888855, 5.5555558),
    (50.0000000, 100.0000000, 5.5555558),
    (-50.0000000, 0.0000000, 16.6666679),
    (-38.8888893, 11.1111107, 16.6666679),
    (-27.7777786, 22.2222214, 16.6666679),
    (-16.6666679, 33.3333321, 16.6666679),
    (-5.5555558, 44.4444427, 16.6666679),
    (5.5555558, 55.5555573, 16.6666679),
    (16.6666679, 66.6666641, 16.6666679),
    (27.7777786, 77.7777786, 16.6666679),
    (38.8888893, 88.8888855, 16.6666679),
    (50.0000000, 100.0000000, 16.6666679),
    (-50.0000000, 0.0000000, 27.7777786),
    (-38.8888893, 11.1111107, 27.7777786),
    (-27.7777786, 22.2222214, 27.7777786),
    (-16.6666679, 33.3333321, 27.7777786),
    (-5.5555558, 44.4444427, 27.7777786),
    (5.5555558, 55.5555573, 27.7777786),
    (16.6666679, 66.6666641, 27.7777786),
    (27.7777786, 77.7777786, 27.7777786),
    (38.8888893, 88.8888855, 27.7777786),
    (50.0000000, 100.0000000, 27.7777786),
    (-50.0000000, 0.0000000, 38.8888893),
    (-38.8888893, 11.1111107, 38.8888893),
    (-27.7777786, 22.2222214, 38.8888893),
    (-16.6666679, 33.3333321, 38.8888893),
    (-5.5555558, 44.4444427, 38.8888893),
    (5.5555558, 55.5555573, 38.8888893),
    (16.6666679, 66.6666641, 38.8888893),
    (27.7777786, 77.7777786, 38.8888893),
    (38.8888893, 88.8888855, 38.8888893),
    (50.0000000, 100.0000000, 38.8888893),
    (-50.0000000, 0.0000000, 50.0000000),
    (-38.8888893, 11.1111107, 50.0000000),
    (-27.7777786, 22.2222214, 50.0000000),
    (-16.6666679, 33.3333321, 50.0000000),
    (-5.5555558, 44.4444427, 50.0000000),
    (5.5555558, 55.5555573, 50.0000000),
    (16.6666679, 66.6666641, 50.0000000),
    (27.7777786, 77.7777786, 50.0000000),
    (38.8888893, 88.8888855, 50.0000000),
    (50.0000000, 100.0000000, 50.0000000),
]

CLOTH_FACES = [
    1, 12, 2,
    1, 11, 12,
    2, 12, 3,
    12, 13, 3,
    3, 14, 4,
    3, 13, 14,
    4, 14, 5,
    14, 15, 5,
    5, 16, 6,
    5, 15, 16,
    6, 16, 7,
    16, 17, 7,
    7, 18, 8,
    7, 17, 18,
    8, 18, 9,
    18, 19, 9,
    9, 20, 10,
    9, 19, 20,
    11, 21, 12,
    21, 22, 12,
    12, 23, 13,
    12, 22, 23,
    13, 23, 14,
    23, 24, 14,
    14, 25, 15,
    14, 24, 25,
    15, 25, 16,
    25, 26, 16,
    16, 27, 17,
    16, 26, 27,
    17, 27, 18,
    27, 28, 18,
    18, 29, 19,
    18, 28, 29,
    19, 29, 20,
    29, 30, 20,
    21, 32, 22,
    21, 31, 32,
    22, 32, 23,
    32, 33, 23,
    23, 34, 24,
    23, 33, 34,
    24, 34, 25,
    34, 35, 25,
    25, 36, 26,
    25, 35, 36,
    26, 36, 27,
    36, 37, 27,
    27, 38, 28,
    27, 37, 38,
    28, 38, 29,
    38, 39, 29,
    29, 40, 30,
    29, 39, 40,
    31, 41, 32,
    41, 42, 32,
    32, 43, 33,
    32, 42, 43,
    33, 43, 34,
    43, 44, 34,
    34, 45, 35,
    34, 44, 45,
    35, 45, 36,
    45, 46, 36,
    36, 47, 37,
    36, 46, 47,
    37, 47, 38,
    47, 48, 38,
    38, 49, 39,
    38, 48, 49,
    39, 49, 40,
    49, 50, 40,
    41, 52, 42,
    41, 51, 52,
    42, 52, 43,
    52, 53, 43,
    43, 54, 44,
    43, 53, 54,
    44, 54, 45,
    54, 55, 45,
    45, 56, 46,
    45, 55, 56,
    46, 56, 47,
    56, 57, 47,
    47, 58, 48,
    47, 57, 58,
    48, 58, 49,
    58, 59, 49,
    49, 60, 50,
    49, 59, 60,
    51, 61, 52,
    61, 62, 52,
    52, 63, 53,
    52, 62, 63,
    53, 63, 54,
    63, 64, 54,
    54, 65, 55,
    54, 64, 65,
    55, 65, 56,
    65, 66, 56,
    56, 67, 57,
    56, 66, 67,
    57, 67, 58,
    67, 68, 58,
    58, 69, 59,
    58, 68, 69,
    59, 69, 60,
    69, 70, 60,
    61, 72, 62,
    61, 71, 72,
    62, 72, 63,
    72, 73, 63,
    63, 74, 64,
    63, 73, 74,
    64, 74, 65,
    74, 75, 65,
    65, 76, 66,
    65, 75, 76,
    66, 76, 67,
    76, 77, 67,
    67, 78, 68,
    67, 77, 78,
    68, 78, 69,
    78, 79, 69,
    69, 80, 70,
    69, 79, 80,
    71, 81, 72,
    81, 82, 72,
    72, 83, 73,
    72, 82, 83,
    73, 83, 74,
    83, 84, 74,
    74, 85, 75,
    74, 84, 85,
    75, 85, 76,
    85, 86, 76,
    76, 87, 77,
    76, 86, 87,
    77, 87, 78,
    87, 88, 78,
    78, 89, 79,
    78, 88, 89,
    79, 89, 80,
    89, 90, 80,
    81, 92, 82,
    81, 91, 92,
    82, 92, 83,
    92, 93, 83,
    83, 94, 84,
    83, 93, 94,
    84, 94, 85,
    94, 95, 85,
    85, 96, 86,
    85, 95, 96,
    86, 96, 87,
    96, 97, 87,
    87, 98, 88,
    87, 97, 98,
    88, 98, 89,
    98, 99, 89,
    89, 100, 90,
    89, 99, 100
]

# fmt: on
class ClothSim:
    def __init__(self, device, solver, use_cuda_graph=False, do_rendering=False):
        self.frame_dt = 1 / 60
        self.num_test_frames = 50
        self.iterations = 5
        self.device = device
        self.use_cuda_graph = self.device.is_cuda and use_cuda_graph
        self.builder = newton.ModelBuilder(up_axis="Y")
        self.solver_name = solver
        self.do_rendering = do_rendering
        self.fixed_particles = []
        self.renderer_scale_factor = 0.01
        # controls particle-shape contact
        self.soft_contact_margin = 1.0
        # controls self-contact of trimesh
        self.self_contact_radius = 0.1
        self.self_contact_margin = 0.1

        if solver != "semi_implicit":
            self.num_substeps = 10
        else:
            self.num_substeps = 32
        self.dt = self.frame_dt / self.num_substeps

    def set_up_sagging_experiment(self):
        self.input_scale_factor = 1.0
        self.renderer_scale_factor = 0.01
        vertices = [wp.vec3(v) * self.input_scale_factor for v in CLOTH_POINTS]
        faces_flatten = [fv - 1 for fv in CLOTH_FACES]

        kd = 1.0e-7

        if self.solver_name != "semi_implicit":
            stretching_stiffness = 1e4
            spring_ke = 1e3
            bending_ke = 10
        else:
            stretching_stiffness = 1e5
            spring_ke = 1e2
            bending_ke = 0.0

        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vertices,
            indices=faces_flatten,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.1,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=kd,
            edge_ke=bending_ke,
            add_springs=self.solver_name == "xpbd",
            spring_ke=spring_ke,
            spring_kd=0.0,
        )

        self.fixed_particles = [0, 9]

        self.finalize(ground=False)

        self.state1.particle_q.fill_(0.0)

    def set_up_bending_experiment(self):
        stretching_stiffness = 1e4
        stretching_damping = 1e-6
        bending_damping = 1e-3
        # fmt: off
        vs = [[-6.0, 0.0, -6.0], [-3.6, 0.0, -6.0], [-1.2, 0.0, -6.0], [1.2, 0.0, -6.0], [3.6, 0.0, -6.0], [6.0, 0.0, -6.0],
         [-6.0, 0.0, -3.6], [-3.6, 0.0, -3.6], [-1.2, 0.0, -3.6], [1.2, 0.0, -3.6], [3.6, 0.0, -3.6], [6.0, 0.0, -3.6],
         [-6.0, 0.0, -1.2], [-3.6, 0.0, -1.2], [-1.2, 0.0, -1.2], [1.2, 0.0, -1.2], [3.6, 0.0, -1.2], [6.0, 0.0, -1.2],
         [-6.0, 0.0, 1.2], [-3.6, 0.0, 1.2], [-1.2, 0.0, 1.2], [1.2, 0.0, 1.2], [3.6, 0.0, 1.2], [6.0, 0.0, 1.2],
         [-6.0, 0.0, 3.6], [-3.6, 0.0, 3.6], [-1.2, 0.0, 3.6], [1.2, 0.0, 3.6], [3.6, 0.0, 3.6], [6.0, 0.0, 3.6],
         [-6.0, 0.0, 6.0], [-3.6, 0.0, 6.0], [-1.2, 0.0, 6.0], [1.2, 0.0, 6.0], [3.6, 0.0, 6.0], [6.0, 0.0, 6.0]]

        fs = [0, 7, 1, 0, 6, 7, 1, 7, 2, 7, 8, 2, 2, 9, 3, 2, 8, 9, 3, 9, 4, 9, 10, 4, 4, 11, 5, 4, 10, 11, 6, 12, 7, 12, 13,
         7, 7, 14, 8, 7, 13, 14, 8, 14, 9, 14, 15, 9, 9, 16, 10, 9, 15, 16, 10, 16, 11, 16, 17, 11, 12, 19, 13, 12, 18,
         19, 13, 19, 14, 19, 20, 14, 14, 21, 15, 14, 20, 21, 15, 21, 16, 21, 22, 16, 16, 23, 17, 16, 22, 23, 18, 24, 19,
         24, 25, 19, 19, 26, 20, 19, 25, 26, 20, 26, 21, 26, 27, 21, 21, 28, 22, 21, 27, 28, 22, 28, 23, 28, 29, 23, 24,
         31, 25, 24, 30, 31, 25, 31, 26, 31, 32, 26, 26, 33, 27, 26, 32, 33, 27, 33, 28, 33, 34, 28, 28, 35, 29, 28, 34,
         35]
        # fmt: on

        vs = [wp.vec3(v) for v in vs]

        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 10.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=stretching_damping,
            edge_ke=10,
            edge_kd=bending_damping,
            add_springs=self.solver_name == "xpbd",
            spring_ke=1.0e3,
            spring_kd=0.0,
        )

        self.builder.add_cloth_mesh(
            pos=wp.vec3(15.0, 10.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=stretching_damping,
            edge_ke=100,
            edge_kd=bending_damping,
            add_springs=self.solver_name == "xpbd",
            spring_ke=1.0e3,
            spring_kd=0.0,
        )

        self.builder.add_cloth_mesh(
            pos=wp.vec3(30.0, 10.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=stretching_damping,
            edge_ke=1000,
            edge_kd=bending_damping,
            add_springs=self.solver_name == "xpbd",
            spring_ke=1.0e3,
            spring_kd=0.0,
        )

        self.fixed_particles = [0, 29, 36, 65, 72, 101]

        self.finalize()

    def set_collision_experiment(self):
        elasticity_ke = 1e4
        elasticity_kd = 1e-6

        vs1 = [wp.vec3(v) for v in [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]]
        fs1 = [0, 1, 2, 0, 2, 3]

        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs1,
            indices=fs1,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=elasticity_ke,
            tri_ka=elasticity_ke,
            tri_kd=elasticity_kd,
            add_springs=self.solver_name == "xpbd",
            spring_ke=1.0e3,
            spring_kd=0.0,
        )

        vs2 = [wp.vec3(v) for v in [[0.3, 0, 0.7], [0.3, 0, 0.2], [0.8, 0, 0.4]]]
        fs2 = [0, 1, 2]

        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.5, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs2,
            indices=fs2,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=elasticity_ke,
            tri_ka=elasticity_ke,
            tri_kd=elasticity_kd,
            add_springs=self.solver_name == "xpbd",
            spring_ke=1.0e3,
            spring_kd=0.0,
        )

        self.fixed_particles = range(0, 4)

        self.finalize(handle_self_contact=True, ground=False)
        self.model.soft_contact_ke = 1e4
        self.model.soft_contact_kd = 1e-3
        self.model.soft_contact_mu = 0.2
        self.model.set_gravity((0.0, -1000.0, 0.0))
        self.num_test_frames = 30

    def set_up_non_zero_rest_angle_bending_experiment(self):
        # fmt: off
        vs =[
            [ 0.     ,  1.     , -1.     ],
            [ 0.     ,  1.     ,  1.     ],
            [ 0.70711,  0.70711, -1.     ],
            [ 0.70711,  0.70711,  1.     ],
            [ 1.     ,  0.     , -1.     ],
            [ 1.     , -0.     ,  1.     ],
            [ 0.70711, -0.70711, -1.     ],
            [ 0.70711, -0.70711,  1.     ],
            [ 0.     , -1.     , -1.     ],
            [ 0.     , -1.     ,  1.     ],
            [-0.70711, -0.70711, -1.     ],
            [-0.70711, -0.70711,  1.     ],
            [-1.     ,  0.     , -1.     ],
            [-1.     , -0.     ,  1.     ],
            [-0.70711,  0.70711, -1.     ],
            [-0.70711,  0.70711,  1.     ],
        ]
        fs = [
             1,  2,  0,
             3,  4,  2,
             5,  6,  4,
             7,  8,  6,
             9, 10,  8,
            11, 12, 10,
             3,  5,  4,
            13, 14, 12,
            15,  0, 14,
             1,  3,  2,
             5,  7,  6,
             7,  9,  8,
             9, 11, 10,
            11, 13, 12,
            13, 15, 14,
            15,  1,  0,
        ]
        # fmt: on

        stretching_stiffness = 1e5
        stretching_damping = 1e-5
        edge_ke = 100
        bending_damping = 1e-4
        vs = [wp.vec3(v) for v in vs]

        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=stretching_damping,
            edge_ke=edge_ke,
            edge_kd=bending_damping,
            add_springs=self.solver_name == "xpbd",
            spring_ke=1.0e3,
            spring_kd=0.0,
        )
        self.fixed_particles = [0, 1]

        self.finalize(handle_self_contact=False, ground=False)

    def set_up_complex_rest_angle_bending_experiment(
        self, tri_ke=1e4, tri_kd=1e-6, edge_ke=1e3, edge_kd=0.0, fixed_particles=None, use_gravity=True
    ):
        # fmt: off
        vs =[
            [ 0.000000, -1.000000, -0.446347],
            [ 0.000000,  1.000000, -0.446347],
            [ 0.707107, -1.000000, -0.707107],
            [ 0.707107,  1.000000, -0.707107],
            [ 1.000000, -1.000000, -1.164154],
            [ 1.000000,  1.000000, -1.164155],
            [ 0.000000, -1.000000,  1.000000],
            [ 0.000000,  1.000000,  1.000000],
            [-0.707107, -1.000000,  0.707107],
            [-0.555208,  1.000000,  0.707107],
            [-0.848101, -1.000000,  0.000000],
            [-0.793222,  1.000000, -0.000000],
            [-0.500329, -1.000000, -0.707107],
            [-0.707107,  1.000000, -0.707107]
        ]
        fs = [
            1, 2, 0,
            3, 4, 2,
            7, 8, 6,
            9, 10, 8,
            3, 5, 4,
            11, 12, 10,
            13, 0, 12,
            1, 3, 2,
            7, 9, 8,
            9, 11, 10,
            11, 13, 12,
            13, 1, 0
        ]
        # fmt: on

        vs = [wp.vec3(v) for v in vs]
        self.builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 2.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=tri_ke,
            tri_ka=tri_ke,
            tri_kd=tri_kd,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
            add_springs=self.solver_name == "xpbd",
            spring_ke=1.0e3,
            spring_kd=0.0,
        )

        self.fixed_particles = fixed_particles if fixed_particles is not None else []
        self.renderer_scale_factor = 1

        self.finalize(handle_self_contact=False, ground=False, use_gravity=use_gravity)

    def set_free_falling_experiment(self):
        self.input_scale_factor = 1.0
        self.renderer_scale_factor = 0.01
        vertices = [wp.vec3(v) * self.input_scale_factor for v in CLOTH_POINTS]
        faces_flatten = [fv - 1 for fv in CLOTH_FACES]
        if self.solver_name != "semi_implicit":
            stretching_stiffness = 1e4
            spring_ke = 1e3
            bending_ke = 10
        else:
            stretching_stiffness = 1e2
            spring_ke = 1e2
            bending_ke = 10

        self.builder.add_cloth_mesh(
            vertices=vertices,
            indices=faces_flatten,
            scale=0.1,
            density=2,
            pos=wp.vec3(0.0, 4.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            edge_ke=bending_ke,
            edge_kd=0.0,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=0.0,
            add_springs=self.solver_name == "xpbd",
            spring_ke=spring_ke,
            spring_kd=0.0,
        )
        self.fixed_particles = []
        self.num_test_frames = 30
        self.finalize(ground=False)

    def set_up_body_cloth_contact_experiment(self):
        # fmt: off
        vs = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
        fs = [
            0, 1, 2,
            2, 3, 0
        ]
        # fmt: on

        if self.solver_name != "semi_implicit":
            stretching_stiffness = 1e4
            spring_ke = 1e3
            bending_ke = 10
        else:
            stretching_stiffness = 1e2
            spring_ke = 1e2
            bending_ke = 10
        particle_radius = 0.2

        vs = [wp.vec3(v) for v in vs]
        self.builder.add_cloth_mesh(
            vertices=vs,
            indices=fs,
            scale=1,
            density=2,
            pos=wp.vec3(0.0, 0.1, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            edge_ke=bending_ke,
            edge_kd=0.0,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=0.0,
            add_springs=self.solver_name == "xpbd",
            spring_ke=spring_ke,
            spring_kd=0.0,
            particle_radius=particle_radius,
        )

        self.builder.add_shape_box(-1, wp.transform(wp.vec3(0, -2, 0), wp.quat_identity()), hx=2, hy=2, hz=2)

        self.renderer_scale_factor = 0.1

        self.finalize(handle_self_contact=False, ground=False, use_gravity=True)
        self.soft_contact_margin = particle_radius * 1.1
        self.model.soft_contact_ke = stretching_stiffness

    def set_up_stitching_experiment(self):
        self.num_test_frames = 200
        vs = [
            # triangle 1
            [0.0, 0.1, 0.0],
            [0.0, 0.1, 1.0],
            [1.0, 0.1, 1.0],
            # triangle 2
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        fs = [
            0,
            1,
            2,
            3,
            4,
            5,
        ]

        if self.solver_name != "semi_implicit":
            stretching_stiffness = 1e4
            spring_ke = 1e3
            stitching_spring_ke = 1e4
            bending_ke = 10
        else:
            stretching_stiffness = 1e2
            spring_ke = 1e2
            stitching_spring_ke = 1e3
            bending_ke = 10
        particle_radius = 0.2

        vs = [wp.vec3(v) for v in vs]
        self.builder.add_cloth_mesh(
            vertices=vs,
            indices=fs,
            scale=1,
            density=2,
            pos=wp.vec3(0.0, 3, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            edge_ke=bending_ke,
            edge_kd=0.0,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=0.0,
            add_springs=self.solver_name == "xpbd",
            spring_ke=spring_ke,
            spring_kd=0.0,
            particle_radius=particle_radius,
        )

        self.springs = [
            [2, 3],
            [0, 5],
        ]

        for spring_idx in range(len(self.springs)):
            self.builder.add_spring(*self.springs[spring_idx], stitching_spring_ke, 0, 0)
        self.renderer_scale_factor = 1
        self.fixed_particles = [1]

        self.self_contact_radius = 0.1
        self.self_contact_margin = 0.1

        self.finalize(handle_self_contact=True, ground=False, use_gravity=True)

    def set_up_enable_tri_contact_experiment(self):
        # fmt: off
        vs = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
        fs = [
            0, 1, 2,
        ]
        # fmt: on

        stretching_stiffness = 1e2
        spring_ke = 1e2
        bending_ke = 10

        particle_radius = 0.2

        vs = [wp.vec3(v) for v in vs]
        self.builder.add_cloth_mesh(
            vertices=vs,
            indices=fs,
            scale=1,
            density=2,
            pos=wp.vec3(0.0, 0.1, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            edge_ke=bending_ke,
            edge_kd=0.0,
            tri_ke=stretching_stiffness,
            tri_ka=stretching_stiffness,
            tri_kd=0.0,
            add_springs=self.solver_name == "xpbd",
            spring_ke=spring_ke,
            spring_kd=0.0,
            particle_radius=particle_radius,
        )

        self.builder.add_particles(
            pos=[wp.vec3(0.35, 4.0 * particle_radius, 0.35)],
            vel=[wp.vec3(0.0, 0.0, 0.0)],
            mass=[0.1],
            radius=[0.5 * particle_radius],
        )

        self.fixed_particles = np.arange(0, len(vs))

        self.renderer_scale_factor = 0.1

        self.finalize(ground=False, use_gravity=True)
        self.soft_contact_margin = particle_radius * 1.1
        self.model.soft_contact_ke = 1e5

    def finalize(self, handle_self_contact=False, ground=True, use_gravity=True):
        builder = newton.ModelBuilder(up_axis="Y")
        builder.add_world(self.builder)
        if ground:
            builder.add_ground_plane()
        builder.color(include_bending=True)

        self.model = builder.finalize(device=self.device)
        self.model.set_gravity((0.0, -1000.0 if use_gravity else 0.0, 0.0))
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e-2

        self.set_points_fixed(self.model, self.fixed_particles)

        if self.solver_name == "vbd":
            self.solver = newton.solvers.SolverVBD(
                model=self.model,
                iterations=self.iterations,
                handle_self_contact=handle_self_contact,
                self_contact_radius=self.self_contact_radius,
                self_contact_margin=self.self_contact_margin,
            )
        elif self.solver_name == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                model=self.model,
                iterations=self.iterations,
            )
        elif self.solver_name == "semi_implicit":
            self.solver = newton.solvers.SolverSemiImplicit(self.model)
        else:
            raise ValueError("Unsupported solver type: " + self.solver_name)

        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.model.collide(self.state0)

        self.init_pos = np.array(self.state0.particle_q.numpy(), copy=True)

        self.graph = None
        if self.use_cuda_graph:
            with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _step in range(self.num_substeps):
            self.state0.clear_forces()
            contacts = self.model.collide(self.state0, soft_contact_margin=self.soft_contact_margin)
            control = self.model.control()
            self.solver.step(self.state0, self.state1, control, contacts, self.dt)
            (self.state0, self.state1) = (self.state1, self.state0)

    def run(self):
        self.sim_time = 0.0

        if self.do_rendering:
            self.viewer = newton.viewer.ViewerGL()
            self.viewer.set_model(self.model)
        else:
            self.viewer = None

        for _frame in range(self.num_test_frames):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

            if self.viewer is not None:
                self.viewer.begin_frame(self.sim_time)
                self.viewer.log_state(self.state0)
                self.viewer.end_frame()
            self.sim_time = self.sim_time + self.frame_dt

    def set_points_fixed(self, model, fixed_particles):
        if len(fixed_particles):
            flags = model.particle_flags.numpy()
            for fixed_v_id in fixed_particles:
                flags[fixed_v_id] = flags[fixed_v_id] & ~ParticleFlags.ACTIVE

            model.particle_flags = wp.array(flags, device=model.device)


def compute_current_angles(model, state):
    """Compute current angles consistent with both add_edges() in model.py and bending angle computation in integrators (XPBD, VBD, SemiImplicit)"""
    angles = []
    for i, j, k, l in model.edge_indices.numpy():
        x3 = state.particle_q.numpy()[k]  # edge start
        x4 = state.particle_q.numpy()[l]  # edge end

        if i != -1 and j != -1:
            x1 = state.particle_q.numpy()[i]  # opposite 0
            x2 = state.particle_q.numpy()[j]  # opposite 1

            n1 = np.cross(x3 - x1, x4 - x1)
            n2 = np.cross(x4 - x2, x3 - x2)
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)

            e = x4 - x3
            e = e / np.linalg.norm(e)

            cos_val = np.clip(np.dot(n1, n2), -1.0, 1.0)
            sin_val = np.dot(np.cross(n1, n2), e)
            angle = np.arctan2(sin_val, cos_val)
        else:
            angle = 0.0
        angles.append(angle)
    return np.array(angles)


def test_cloth_sagging(test, device, solver):
    example = ClothSim(device, solver, use_cuda_graph=True)
    example.set_up_sagging_experiment()

    initial_pos = example.state0.particle_q.numpy().copy()

    example.run()

    fixed_points = np.where(np.logical_not(example.model.particle_flags.numpy()))
    # examine that the simulation does not explode
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((initial_pos[fixed_points, :] == example.state0.particle_q.numpy()[fixed_points, :]).all())
    test.assertTrue((initial_pos[fixed_points, :] == example.state1.particle_q.numpy()[fixed_points, :]).all())
    test.assertTrue((final_pos < 1e5).all())
    # examine that the simulation has moved
    test.assertTrue((example.init_pos != final_pos).any())


def test_cloth_bending(test, device, solver):
    example = ClothSim(device, solver, use_cuda_graph=True)
    example.set_up_bending_experiment()

    example.run()

    # examine that the simulation does not explode
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((final_pos < 1e5).all())
    # examine that the simulation has moved
    test.assertTrue((example.init_pos != final_pos).any())


def test_cloth_bending_non_zero_rest_angle_bending(test, device, solver):
    example = ClothSim(device, solver, use_cuda_graph=True)
    example.set_up_non_zero_rest_angle_bending_experiment()

    example.run()

    # examine that the simulation does not explode
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((np.abs(final_pos) < 1e5).all())
    # examine that the simulation has moved
    test.assertTrue((example.init_pos != final_pos).any())


def test_cloth_bending_consistent_angle_computation(test, device, solver):
    example = ClothSim(device, solver, use_cuda_graph=True)
    example.set_up_complex_rest_angle_bending_experiment(
        tri_ke=1e2, tri_kd=0.0, edge_ke=1e-1, edge_kd=0.0, fixed_particles=[1], use_gravity=False
    )

    # Store rest angles
    rest_angles = example.model.edge_rest_angle.numpy()

    example.run()

    # Compute final angles
    final_angles = compute_current_angles(example.model, example.state0)

    # Verify stability and consistency between rest angle and current angle computations:
    # Without gravity (use_gravity=False), current angles should converge to rest angles
    # if the simulation is stable and angle computations are consistent
    test.assertTrue(
        np.abs(final_angles - rest_angles).max() <= 0.01,
        f"Maximum angle difference {np.abs(final_angles - rest_angles).max():.6f} rad exceeds 0.01 rad",
    )


def test_cloth_bending_with_complex_rest_angles(test, device, solver):
    example = ClothSim(device, solver, use_cuda_graph=True)
    example.set_up_complex_rest_angle_bending_experiment(
        tri_ke=1e3, tri_kd=1e-2, edge_ke=1e3, edge_kd=0.0, fixed_particles=[1], use_gravity=True
    )

    # Store rest angles for comparison
    rest_angles = example.model.edge_rest_angle.numpy()

    example.run()

    # Verify basic stability (same check as test_cloth_bending_non_zero_rest_angle_bending)
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((np.abs(final_pos) < 1e5).all())
    test.assertTrue((example.init_pos != final_pos).any())

    # Verify bending angles stay within tolerance of rest angles
    final_angles = compute_current_angles(example.model, example.state0)
    max_difference = np.abs(final_angles - rest_angles).max()
    test.assertTrue(max_difference <= 0.1, f"Maximum angle difference {max_difference:.3f} rad exceeds 0.1 rad")


# Internal forces and damping should not affect free-fall behavior.
def test_cloth_free_fall_with_internal_forces_and_damping(test, device, solver):
    example = ClothSim(device, solver, use_cuda_graph=True)
    example.set_up_complex_rest_angle_bending_experiment(
        tri_ke=5e1, tri_kd=1e-1, edge_ke=1e1, edge_kd=1e-1, fixed_particles=None, use_gravity=True
    )

    # Store initial vertex positions and rest angles for comparison
    initial_pos = example.state0.particle_q.numpy().copy()
    rest_angles = example.model.edge_rest_angle.numpy()

    example.run()

    # Get final positions
    final_pos = example.state0.particle_q.numpy()

    # Verify basic stability
    test.assertTrue((np.abs(final_pos) < 1e5).all())
    test.assertTrue((initial_pos != final_pos).any())

    # Check for non-gravitational position changes per vertex
    # Calculate position differences for each vertex
    position_diff = final_pos - initial_pos

    # Get gravity direction (normalized)
    gravity_vector = example.model.gravity.numpy()[0]  # Extract first element from warp array
    gravity_direction = gravity_vector / np.linalg.norm(gravity_vector)

    # For each vertex, project its displacement onto gravity direction
    gravity_displacement_per_vertex = np.dot(position_diff, gravity_direction)

    # Calculate non-gravitational component for each vertex
    gravity_component_per_vertex = gravity_displacement_per_vertex[:, np.newaxis] * gravity_direction[np.newaxis, :]
    non_gravity_displacement = position_diff - gravity_component_per_vertex

    # Calculate magnitude of non-gravitational displacement per vertex
    non_gravity_magnitude = np.linalg.norm(non_gravity_displacement, axis=1)

    # Find vertices with significant non-gravitational movement
    max_non_gravity_displacement = np.max(non_gravity_magnitude)
    problematic_vertices = np.where(non_gravity_magnitude > 0.01)[0]

    # Verify that non-gravitational displacement is minimal for all vertices
    test.assertTrue(
        max_non_gravity_displacement < 0.02,
        f"Non-gravitational displacement detected: max {max_non_gravity_displacement:.4f} at vertex indices {problematic_vertices}",
    )

    # Verify bending angles stay within tolerance of rest angles
    final_angles = compute_current_angles(example.model, example.state0)
    max_difference = np.abs(final_angles - rest_angles).max()
    test.assertTrue(max_difference <= 0.1, f"Maximum angle difference {max_difference:.3f} rad exceeds 0.1 rad")


def test_cloth_collision(test, device, solver):
    example = ClothSim(device, solver, use_cuda_graph=True)
    example.set_collision_experiment()

    example.run()

    # examine that the velocity has died out
    final_vel = example.state0.particle_qd.numpy()
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((np.linalg.norm(final_vel, axis=0) < 1.0).all())
    # examine that the simulation has moved
    test.assertTrue((example.init_pos != final_pos).any())


def test_cloth_free_fall(test, device, solver):
    example = ClothSim(device, solver)
    example.set_free_falling_experiment()

    initial_pos = example.state0.particle_q.numpy().copy()

    example.run()

    # examine that the simulation does not explode
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((final_pos < 1e5).all())
    # examine that the simulation has moved
    test.assertTrue((example.init_pos != final_pos).any())

    gravity = example.model.gravity.numpy()[0]
    diff = final_pos - initial_pos
    vertical_translation_norm = diff @ gravity[..., None] / (np.linalg.norm(gravity) ** 2)
    # ensure it's free-falling
    test.assertTrue((np.abs(vertical_translation_norm - 0.5 * np.linalg.norm(gravity) * (example.dt**2)) < 2e-1).all())
    horizontal_move = diff - (vertical_translation_norm * gravity)
    # ensure its horizontal translation is minimal
    test.assertTrue((np.abs(horizontal_move) < 1e-1).all())


def test_cloth_body_collision(test, device, solver):
    example = ClothSim(device, solver)
    example.set_up_body_cloth_contact_experiment()

    example.run()

    # examine that the velocity has died out
    final_vel = example.state0.particle_qd.numpy()
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue((np.linalg.norm(final_vel, axis=0) < 1.0).all())
    # examine that the simulation has moved
    test.assertTrue((np.abs(final_pos[:, 1] - 0.0) < 0.5).all())


def test_cloth_stitching(test, device, solver):
    example = ClothSim(device, solver)
    example.set_up_stitching_experiment()

    example.run()

    # examine that the velocity has died out
    final_pos = example.state0.particle_q.numpy()

    for spring_idx in range(len(example.springs)):
        test.assertTrue(
            (
                np.linalg.norm(final_pos[example.springs[spring_idx][0]] - final_pos[example.springs[spring_idx][1]])
                < 1.0
            ).all()
        )


def test_cloth_enable_tri_contact(test, device, solver):
    # Set enable_tri_contact to True
    example = ClothSim(device, solver)
    example.set_up_enable_tri_contact_experiment()
    example.solver.enable_tri_contact = True

    example.run()

    # examine that the vertical coordinate of the last particle is positive
    final_pos = example.state0.particle_q.numpy()
    test.assertTrue(final_pos[-1, 1] > 0.0)

    # Set enable_tri_contact to False
    example = ClothSim(device, solver)
    example.set_up_enable_tri_contact_experiment()
    example.solver.enable_tri_contact = False

    example.run()

    # examine that the vertical coordinate of the last particle is negative
    final_pos = example.state0.particle_q.numpy()
    print("final_pos", final_pos[-1, 1])
    test.assertTrue(final_pos[-1, 1] < 0.0)


devices = get_test_devices(mode="basic")


class TestCloth(unittest.TestCase):
    pass


tests_to_run = {
    "xpbd": [
        test_cloth_free_fall,
        test_cloth_sagging,
        test_cloth_bending,
        test_cloth_bending_consistent_angle_computation,
        test_cloth_bending_non_zero_rest_angle_bending,
        test_cloth_bending_with_complex_rest_angles,
        test_cloth_free_fall_with_internal_forces_and_damping,
        test_cloth_body_collision,
        test_cloth_stitching,
    ],
    "semi_implicit": [
        test_cloth_free_fall,
        test_cloth_sagging,
        test_cloth_bending,
        test_cloth_bending_consistent_angle_computation,
        test_cloth_bending_non_zero_rest_angle_bending,
        test_cloth_bending_with_complex_rest_angles,
        test_cloth_free_fall_with_internal_forces_and_damping,
        test_cloth_body_collision,
        test_cloth_enable_tri_contact,
    ],
    "vbd": [
        test_cloth_free_fall,
        test_cloth_sagging,
        test_cloth_bending,
        test_cloth_collision,
        test_cloth_bending_consistent_angle_computation,
        test_cloth_bending_non_zero_rest_angle_bending,
        test_cloth_bending_with_complex_rest_angles,
        test_cloth_free_fall_with_internal_forces_and_damping,
        test_cloth_body_collision,
        test_cloth_stitching,
    ],
}

for solver, tests in tests_to_run.items():
    for test in tests:
        add_function_test(
            TestCloth, f"{test.__name__}_{solver}", partial(test, solver=solver), devices=devices, check_output=False
        )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
