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

"""Frame Transform Sensor - measures transforms relative to sites."""

import warp as wp

from ..sim.model import Model
from ..sim.state import State


@wp.kernel
def compute_imu_sensor_kernel(
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    reference_sites: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qdd: wp.array(dtype=wp.spatial_vector),
    # output
    imu_sensor: wp.array(dtype=wp.spatial_vector),
):
    sensor_idx = wp.tid()

    if sensor_idx >= len(imu_sensor):
        return

    site_idx = reference_sites[sensor_idx]
    body_idx = shape_body[site_idx]
    body_acc = body_qdd[body_idx]
    site_transform = shape_transform[site_idx]

    body_quat = body_q[body_idx].q
    r = wp.quat_rotate(body_quat, site_transform.p - body_com[body_idx])

    vel_ang = wp.spatial_bottom(body_qd[body_idx])

    acc_lin = wp.spatial_top(body_acc) + wp.cross(wp.spatial_bottom(body_acc), r) + wp.cross(vel_ang, wp.cross(vel_ang, r))

    q = body_quat * site_transform.q
    imu_sensor[sensor_idx] = wp.spatial_vector(wp.quat_rotate_inv(q, acc_lin), wp.quat_rotate_inv(q, vel_ang))


class IMUSensor:
    """Sensor that measures IMU data at a site."""

    def __init__(self, model: Model, reference_sites: list[int], verbose: bool | None = None):
        self.model = model
        self.verbose = verbose if verbose is not None else wp.config.verbose

        self.model.require_state_fields("body_qdd")

        self.reference_sites_arr = wp.array(reference_sites, dtype=int, device=model.device)
        self.n_sensors = len(reference_sites)
        self.sensor_qdd = wp.zeros(self.n_sensors, dtype=wp.spatial_vector, device=model.device)

    def update(self, state: State):
        """Update the IMU sensor.

        Args:
            state: The state to update the sensor at.
        """

        wp.launch(
            compute_imu_sensor_kernel,
            dim=self.n_sensors,
            inputs=[
                self.model.body_com,
                self.model.shape_body,
                self.model.shape_transform,
                self.reference_sites_arr,
                state.body_q,
                state.body_qd,
                state.body_qdd,
            ],
            outputs=[self.sensor_qdd],
            device=self.model.device,
        )
