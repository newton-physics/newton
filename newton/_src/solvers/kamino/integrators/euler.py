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

"""
KAMINO: Euler Integrators
"""

from __future__ import annotations

import warp as wp

from ..core.math import quat_box_plus, screw, screw_angular, screw_linear
from ..core.model import Model, ModelData
from ..core.types import float32, int32, mat33f, transformf, vec3f, vec4f, vec6f

###
# Module interface
###

__all__ = [
    "integrate_semi_implicit_euler",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def semi_implicit_euler_with_logmap(
    dt: float32,
    g: vec3f,
    inv_m_i: float32,
    I_i: mat33f,
    inv_I_i: mat33f,
    p_i: transformf,
    u_i: vec6f,
    w_i: vec6f,
):
    # Extract linear and angular parts
    r_i = wp.transform_get_translation(p_i)
    q_i = wp.transform_get_rotation(p_i)
    v_i = screw_linear(u_i)
    omega_i = screw_angular(u_i)
    S_i = wp.skew(omega_i)
    f_i = screw_linear(w_i)
    tau_i = screw_angular(w_i)

    # Compute velocity update equations
    v_i_n = v_i + dt * (g + inv_m_i * f_i)
    omega_i_n = omega_i + dt * inv_I_i @ (-S_i @ I_i @ omega_i + tau_i)

    # Compute configuration update equations
    r_i_n = r_i + dt * v_i_n
    q_i_n = quat_box_plus(q_i, dt * omega_i_n)

    # Compute the new pose and twist
    p_i_n = wp.transformation(r_i_n, q_i_n)
    u_i_n = screw(v_i_n, omega_i_n)

    # Return the new pose and twist
    return p_i_n, u_i_n


###
# Kernels
###


@wp.kernel
def _integrate_semi_implicit_euler_inplace(
    # Inputs:
    model_dt: wp.array(dtype=float32),
    model_gravity: wp.array(dtype=vec4f),
    model_bodies_wid: wp.array(dtype=int32),
    model_bodies_inv_m: wp.array(dtype=float32),
    model_bodies_I: wp.array(dtype=mat33f),
    model_bodies_inv_I: wp.array(dtype=mat33f),
    state_bodies_w: wp.array(dtype=vec6f),
    # Outputs:
    state_bodies_q: wp.array(dtype=transformf),
    state_bodies_u: wp.array(dtype=vec6f),
):
    # Retrieve the thread index
    tid = wp.tid()

    # Retrieve the world index
    wid = model_bodies_wid[tid]

    # Retrieve the time step and gravity vector
    dt = model_dt[wid]
    gv = model_gravity[wid]
    g = gv.w * vec3f(gv.x, gv.y, gv.z)

    # Retrieve the model data
    inv_m_i = model_bodies_inv_m[tid]
    I_i = model_bodies_I[tid]
    inv_I_i = model_bodies_inv_I[tid]

    # Retrieve the current state of the body
    q_i = state_bodies_q[tid]
    u_i = state_bodies_u[tid]
    w_i = state_bodies_w[tid]

    # Compute the next pose and twist
    q_i_n, u_i_n = semi_implicit_euler_with_logmap(
        dt,
        g,
        inv_m_i,
        I_i,
        inv_I_i,
        q_i,
        u_i,
        w_i,
    )

    # Store the computed next pose and twist
    state_bodies_q[tid] = q_i_n
    state_bodies_u[tid] = u_i_n


###
# Launchers
###


def integrate_semi_implicit_euler(model: Model, data: ModelData):
    wp.launch(
        _integrate_semi_implicit_euler_inplace,
        dim=model.size.sum_of_num_bodies,
        inputs=[
            # Inputs:
            model.time.dt,
            model.gravity.vector,
            model.bodies.wid,
            model.bodies.inv_m_i,
            data.bodies.I_i,
            data.bodies.inv_I_i,
            data.bodies.w_i,
            # Outputs:
            data.bodies.q_i,
            data.bodies.u_i,
        ],
    )
