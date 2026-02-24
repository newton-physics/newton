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
Wind system for Newton viewer.
Provides GPU-accelerated wind for particle with CUDA graph support.
"""

import warp as wp

import newton


@wp.struct
class WindParams:
    """Wind parameters struct, stored in GPU memory for graph capture."""

    time: float
    period: float
    amplitude: float
    frequency: float
    direction: wp.vec3
    dt: float


@wp.kernel
def apply_wind_force_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    wind_params: wp.array(dtype=WindParams),
):
    """Apply sinusoidal wind impulses to particles using struct parameters."""
    tid = wp.tid()

    # Check if particle is active
    if (particle_flags[tid] & newton.ParticleFlags.ACTIVE) == 0:
        return

    # Get wind parameters from device array
    params = wind_params[0]

    # Skip if amplitude is zero
    if params.amplitude <= 0.0:
        return

    # Calculate sinusoidal wind intensity
    # Use both time-based and position-based variation for natural wind
    pos = particle_q[tid]

    # Time-based oscillation
    time_phase = 2.0 * wp.pi * params.time / params.period
    time_factor = wp.sin(time_phase * params.frequency) * 0.25 + 1.0

    # Add spatial variation based on position for more realistic patterns
    spatial_phase = 10.0 * (pos[0] + pos[1] + pos[2])
    spatial_factor = wp.sin(spatial_phase + time_phase) * 0.25 + 1.0

    # Combine factors for wind intensity
    wind_intensity = time_factor * spatial_factor

    # Apply wind force
    wind_force = params.direction * (params.amplitude * wind_intensity) * params.dt

    # Add to existing particle forces
    wp.atomic_add(particle_qd, tid, wind_force)


class Wind:
    """Wind force system for particle simulations."""

    def __init__(self, model):
        """Initialize wind system.

        Args:
            model: Newton model object
        """
        self.model = model

        # Initialize wind parameters
        self._wind_params_host = WindParams()
        self._wind_params_host.time = 0.0
        self._wind_params_host.period = 1.0
        self._wind_params_host.amplitude = 0.0
        self._wind_params_host.frequency = 1.0
        self._wind_params_host.direction = wp.vec3(1.0, 0.0, 0.0)
        self._wind_params_host.dt = 0.01

        if self.model:
            self.wind_data = wp.array([self._wind_params_host], dtype=WindParams, device=self.model.device)

    def update(self, dt):
        """Update wind time."""

        if not self.is_active():
            return

        # Update wind time
        self._wind_params_host.time += dt
        self._wind_params_host.dt = dt

        # Update device parameters array
        self.wind_data.assign([self._wind_params_host])

    def _apply_wind_force(self, state):
        """Apply wind forces to particle state.

        Args:
            state: Newton state object with particle arrays
            model: Newton model object (optional, will try to get from state)
        """
        if not self.is_active():
            return

        # Launch wind kernel
        wp.launch(
            apply_wind_force_kernel,
            dim=len(state.particle_q),
            inputs=[state.particle_q, state.particle_qd, self.model.particle_flags, self.wind_data],
        )

    def is_active(self):
        """Check if wind is active."""
        if not self.model:
            return False

        return self.model.particle_count > 0

    @property
    def time(self):
        """Get current wind time."""
        return self._wind_params_host.time

    @time.setter
    def time(self, value):
        """Set wind time."""
        self._wind_params_host.time = float(value)

    @property
    def period(self):
        """Get wind period."""
        return self._wind_params_host.period

    @period.setter
    def period(self, value):
        """Set wind period."""
        self._wind_params_host.period = float(value)

    @property
    def amplitude(self):
        """Get wind amplitude."""
        return self._wind_params_host.amplitude

    @amplitude.setter
    def amplitude(self, value):
        """Set wind amplitude."""
        self._wind_params_host.amplitude = float(value)

    @property
    def frequency(self):
        """Get wind frequency."""
        return self._wind_params_host.frequency

    @frequency.setter
    def frequency(self, value):
        """Set wind frequency."""
        self._wind_params_host.frequency = float(value)

    @property
    def direction(self):
        """Get wind direction."""
        return self._wind_params_host.direction

    @direction.setter
    def direction(self, value):
        """Set wind direction (will be normalized)."""
        if isinstance(value, list | tuple) and len(value) == 3:
            # Normalize the direction vector
            import math  # noqa: PLC0415

            length = math.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)
            if length > 0:
                self._wind_params_host.direction = wp.vec3(value[0] / length, value[1] / length, value[2] / length)
            else:
                self._wind_params_host.direction = wp.vec3(1.0, 0.0, 0.0)
        elif hasattr(value, "__len__") and len(value) == 3:
            # Handle wp.vec3 or similar
            self._wind_params_host.direction = wp.vec3(float(value[0]), float(value[1]), float(value[2]))
        else:
            raise ValueError("Direction must be a 3-element vector")
