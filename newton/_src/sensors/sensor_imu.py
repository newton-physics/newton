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

"""IMU Sensor - measures accelerations and angular velocities at sensor sites."""

import warp as wp

from ..geometry.flags import ShapeFlags
from ..sim.model import Model
from ..sim.state import State
from ..utils.selection import match_labels


@wp.kernel
def compute_sensor_imu_kernel(
    gravity: wp.array(dtype=wp.vec3),
    body_world: wp.array(dtype=wp.int32),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    sensor_sites: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_qdd: wp.array(dtype=wp.spatial_vector),
    # output
    accelerometer: wp.array(dtype=wp.vec3),
    gyroscope: wp.array(dtype=wp.vec3),
):
    """Compute accelerations and angular velocities at sensor sites."""
    sensor_idx = wp.tid()

    if sensor_idx >= len(sensor_sites):
        return

    site_idx = sensor_sites[sensor_idx]
    body_idx = shape_body[site_idx]

    site_transform = shape_transform[site_idx]

    if body_idx < 0:
        accelerometer[sensor_idx] = wp.quat_rotate_inv(site_transform.q, -gravity[0])
        gyroscope[sensor_idx] = wp.vec3(0.0)
        return

    world_idx = body_world[body_idx]
    world_g = gravity[wp.max(world_idx, 0)]

    body_acc = body_qdd[body_idx]

    body_quat = body_q[body_idx].q
    r = wp.quat_rotate(body_quat, site_transform.p - body_com[body_idx])

    vel_ang = wp.spatial_bottom(body_qd[body_idx])

    acc_lin = (
        wp.spatial_top(body_acc)
        - world_g
        + wp.cross(wp.spatial_bottom(body_acc), r)
        + wp.cross(vel_ang, wp.cross(vel_ang, r))
    )

    q = body_quat * site_transform.q
    accelerometer[sensor_idx] = wp.quat_rotate_inv(q, acc_lin)
    gyroscope[sensor_idx] = wp.quat_rotate_inv(q, vel_ang)


class SensorIMU:
    """Inertial Measurement Unit Sensor.

    This sensor measures the acceleration and angular velocity at the sites given.

    Body Accelerations Attribute:
    This sensor requires the extended state attribute ``body_qdd`` to be computed by the solver.  This requires
    a solver that supports computing ``body_qdd``, and requesting ``body_qdd`` from the model before calling
    ``model.state()``. Instantiating the SensorIMU will automatically request ``body_qdd`` from the model by default.

    The ``sites`` parameter accepts label patterns — see :ref:`label-matching`.

    Example:
        Create a SensorIMU for a model with a list of site indices::

            # Obtain shape indices (e.g. via selection or direct indexing)
            sensor_sites = [0, 1, 2]  # indices of sites to attach IMU sensors

            model = Model()
            sensor = SensorIMU(model, sensor_sites)
            state = model.state()

            # Update after step()
            sensor.update(state)
    """

    accelerometer: wp.array(dtype=wp.vec3)
    """Linear acceleration readings in sensor frame, shape (n_sensors,)."""

    gyroscope: wp.array(dtype=wp.vec3)
    """Angular velocity readings in sensor frame, shape (n_sensors,)."""

    def __init__(
        self,
        model: Model,
        sites: str | list[str] | list[int],
        *,
        verbose: bool | None = None,
        request_state_attributes: bool = True,
    ):
        """Initialize SensorIMU.

        Transparently requests the extended state attribute ``body_qdd`` from the model, which is required for acceleration
        data.

        Args:
            model: The model to use.
            sites: List of shape indices, single pattern to match against shape
                labels, or list of patterns where any one matches.
            verbose: If True, print details. If None, uses ``wp.config.verbose``.
            request_state_attributes: If True (default), transparently request the extended state attribute ``body_qdd`` from the model.
                If False, ``model`` is not modified and the attribute must be requested elsewhere before calling ``model.state()``.
        Raises:
            ValueError: If no labels match or invalid sites are passed.
        """

        self.model = model
        self.verbose = verbose if verbose is not None else wp.config.verbose

        original_sites = sites
        sites = match_labels(model.shape_key, sites)  # FIXME: rename to label
        if not sites:
            if isinstance(original_sites, list) and len(original_sites) == 0:
                raise ValueError("'sites' must not be empty")
            raise ValueError(f"No sites matched the given pattern {original_sites!r}")

        # request acceleration state attribute
        if request_state_attributes:
            self.model.request_state_attributes("body_qdd")

        self._validate_sensor_sites(sites)

        self.sensor_sites_arr = wp.array(sites, dtype=int, device=model.device)
        self.n_sensors: int = len(sites)
        self.accelerometer = wp.zeros(self.n_sensors, dtype=wp.vec3, device=model.device)
        self.gyroscope = wp.zeros(self.n_sensors, dtype=wp.vec3, device=model.device)

        if self.verbose:
            print("SensorIMU initialized:")
            print(f"  Sites: {len(set(sites))}")
            # TODO: body per site

    def _validate_sensor_sites(self, sensor_sites: list[int]):
        """Validate the sensor sites."""
        shape_flags = self.model.shape_flags.numpy()
        for site_idx in sensor_sites:
            if site_idx < 0 or site_idx >= self.model.shape_count:
                raise ValueError(f"sensor site index {site_idx} is out of range")
            if not (shape_flags[site_idx] & ShapeFlags.SITE):
                raise ValueError(f"sensor site index {site_idx} is not a site")

    def update(self, state: State):
        """Update the IMU sensor.

        Args:
            state: The state to update the sensor from.
        """
        if state.body_qdd is None:
            raise ValueError("SensorIMU requires a State with body_qdd allocated. Create SensorIMU before State.")

        wp.launch(
            compute_sensor_imu_kernel,
            dim=self.n_sensors,
            inputs=[
                self.model.gravity,
                self.model.body_world,
                self.model.body_com,
                self.model.shape_body,
                self.model.shape_transform,
                self.sensor_sites_arr,
                state.body_q,
                state.body_qd,
                state.body_qdd,
            ],
            outputs=[self.accelerometer, self.gyroscope],
            device=self.model.device,
        )
