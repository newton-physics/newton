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

import warp as wp

from ...sim import Control, Model, State


@wp.func
def muscle_force(
    i: int,
    body_X_s: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    muscle_links: wp.array(dtype=int),
    muscle_points: wp.array(dtype=wp.vec3),
    muscle_activation: float,
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    link_0 = muscle_links[i]
    link_1 = muscle_links[i + 1]

    if link_0 == link_1:
        return 0

    r_0 = muscle_points[i]
    r_1 = muscle_points[i + 1]

    xform_0 = body_X_s[link_0]
    xform_1 = body_X_s[link_1]

    pos_0 = wp.transform_point(xform_0, r_0 - body_com[link_0])
    pos_1 = wp.transform_point(xform_1, r_1 - body_com[link_1])

    n = wp.normalize(pos_1 - pos_0)

    # todo: add passive elastic and viscosity terms
    f = n * muscle_activation

    wp.atomic_sub(body_f_s, link_0, wp.spatial_vector(f, wp.cross(pos_0, f)))
    wp.atomic_add(body_f_s, link_1, wp.spatial_vector(f, wp.cross(pos_1, f)))


@wp.kernel
def eval_muscle(
    body_X_s: wp.array(dtype=wp.transform),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    muscle_start: wp.array(dtype=int),
    muscle_params: wp.array(dtype=float),
    muscle_links: wp.array(dtype=int),
    muscle_points: wp.array(dtype=wp.vec3),
    muscle_activation: wp.array(dtype=float),
    # output
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    m_start = muscle_start[tid]
    m_end = muscle_start[tid + 1] - 1

    activation = muscle_activation[tid]

    for i in range(m_start, m_end):
        muscle_force(i, body_X_s, body_v_s, body_com, muscle_links, muscle_points, activation, body_f_s)


def eval_muscle_forces(model: Model, state: State, control: Control, body_f: wp.array):
    if model.muscle_count:
        wp.launch(
            kernel=eval_muscle,
            dim=model.muscle_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.muscle_start,
                model.muscle_params,
                model.muscle_bodies,
                model.muscle_points,
                control.muscle_activations,
            ],
            outputs=[body_f],
            device=model.device,
        )
