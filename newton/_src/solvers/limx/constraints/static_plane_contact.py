# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free particle contact against a static plane."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp


@wp.kernel
def _prepare_static_plane_contact(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    normal: wp.vec3,
    offset: float,
    thickness: float,
    stiffness: float,
    normal_damping: float,
    friction: float,
    friction_epsilon: float,
    dt: float,
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
):
    particle = wp.tid()
    distance = wp.dot(normal, positions[particle]) - offset
    if distance >= thickness:
        forces[particle] = wp.vec3(0.0)
        hessians[particle] = wp.mat33(0.0)
        return

    depth = thickness - distance
    normal_outer = wp.outer(normal, normal)
    force = stiffness * depth * normal
    hessian = stiffness * normal_outer

    velocity = velocities[particle]
    normal_velocity = wp.dot(velocity, normal)
    if normal_velocity < 0.0 and normal_damping > 0.0:
        force -= normal_damping * normal_velocity * normal
        hessian += normal_damping / dt * normal_outer

    tangent = wp.identity(3, float) - normal_outer
    tangent_displacement = dt * (velocity - normal_velocity * normal)
    tangent_length = wp.length(tangent_displacement)
    friction_over_length = float(0.0)
    if tangent_length > friction_epsilon:
        friction_over_length = 1.0 / tangent_length
    else:
        friction_over_length = (-tangent_length / friction_epsilon + 2.0) / friction_epsilon
    alpha = friction * stiffness * depth * friction_over_length
    force -= alpha * tangent_displacement
    hessian += alpha * tangent

    forces[particle] = force
    hessians[particle] = hessian


@wp.kernel
def _accumulate_static_plane_force(forces: wp.array[wp.vec3], output: wp.array[wp.vec3]):
    particle = wp.tid()
    output[particle] += forces[particle]


@wp.kernel
def _static_plane_hessian_multiply(
    hessians: wp.array[wp.mat33],
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    particle = wp.tid()
    output[particle] += hessians[particle] * vector[particle]


@wp.kernel
def _accumulate_static_plane_diagonal(hessians: wp.array[wp.mat33], output: wp.array[wp.mat33]):
    particle = wp.tid()
    output[particle] += hessians[particle]


class ConstraintStaticPlaneContact:
    """One-particle matrix-free contact against a fixed one-sided plane."""

    def __init__(
        self,
        normal: Sequence[float],
        offset: float,
        thickness: float,
        stiffness: float,
        normal_damping: float,
        friction: float,
        friction_epsilon: float,
        particle_count: int,
        device: Any,
    ):
        """Create a static-plane contact operator.

        The normalized plane equation is ``dot(normal, position) - offset``.
        Contact activates below ``thickness`` on the normal side.

        Args:
            normal: Outward plane normal.
            offset: Plane offset from the origin [m].
            thickness: One-sided contact activation distance [m].
            stiffness: Normal penalty stiffness [N/m].
            normal_damping: Approaching normal damping coefficient [N·s/m].
            friction: Coulomb friction coefficient.
            friction_epsilon: Tangential displacement regularization [m].
            particle_count: Number of particles in the associated model.
            device: Warp device storing runtime arrays.
        """
        normal_array = np.asarray(normal, dtype=np.float64)
        if normal_array.shape != (3,) or not np.isfinite(normal_array).all():
            raise ValueError("normal must contain three finite components")
        normal_length = float(np.linalg.norm(normal_array))
        if normal_length <= 0.0:
            raise ValueError("normal must be nonzero")
        if not np.isfinite(offset):
            raise ValueError("offset must be finite")
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError("thickness must be finite and positive")
        if not np.isfinite(stiffness) or stiffness <= 0.0:
            raise ValueError("stiffness must be finite and positive")
        if not np.isfinite(normal_damping) or normal_damping < 0.0:
            raise ValueError("normal_damping must be finite and nonnegative")
        if not np.isfinite(friction) or friction < 0.0:
            raise ValueError("friction must be finite and nonnegative")
        if not np.isfinite(friction_epsilon) or friction_epsilon <= 0.0:
            raise ValueError("friction_epsilon must be finite and positive")
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")

        normalized = normal_array / normal_length
        self.normal = wp.vec3(float(normalized[0]), float(normalized[1]), float(normalized[2]))
        self.offset = float(offset)
        self.thickness = float(thickness)
        self.stiffness = float(stiffness)
        self.normal_damping = float(normal_damping)
        self.friction = float(friction)
        self.friction_epsilon = float(friction_epsilon)
        self.particle_count = int(particle_count)
        self.device = wp.get_device(device)

        self.forces = wp.empty(self.particle_count, dtype=wp.vec3, device=self.device)
        self.hessians = wp.empty(self.particle_count, dtype=wp.mat33, device=self.device)
        self._velocities: wp.array[wp.vec3] | None = None
        self._dt = 0.0

    def begin_step(
        self,
        positions: wp.array[wp.vec3],
        velocities: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Cache step-start velocity for lagged damping and friction.

        Args:
            positions: Step-start particle positions [m], shape ``[particle_count, 3]``.
            velocities: Step-start particle velocities [m/s], shape ``[particle_count, 3]``.
            dt: Simulation time step [s].
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self._validate_vectors((positions, "positions"), (velocities, "velocities"))
        self._velocities = velocities
        self._dt = float(dt)

    def prepare(self, positions: wp.array[wp.vec3]) -> None:
        """Cache force and PSD Hessian at the current Newton iterate.

        Args:
            positions: Linearization positions [m], shape ``[particle_count, 3]``.
        """
        if self._velocities is None:
            raise RuntimeError("begin_step() must be called before prepare()")
        self._validate_vectors((positions, "positions"))
        wp.launch(
            _prepare_static_plane_contact,
            dim=self.particle_count,
            inputs=[
                positions,
                self._velocities,
                self.normal,
                self.offset,
                self.thickness,
                self.stiffness,
                self.normal_damping,
                self.friction,
                self.friction_epsilon,
                self._dt,
            ],
            outputs=[self.forces, self.hessians],
            device=self.device,
        )

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add cached contact forces to ``output``.

        Args:
            positions: Linearization positions [m], shape ``[particle_count, 3]``.
            output: Force accumulation buffer [N], shape ``[particle_count, 3]``.
        """
        self._validate_vectors((positions, "positions"), (output, "output"))
        wp.launch(
            _accumulate_static_plane_force,
            dim=self.particle_count,
            inputs=[self.forces],
            outputs=[output],
            device=self.device,
        )

    def hessian_multiply(
        self,
        positions: wp.array[wp.vec3],
        vector: wp.array[wp.vec3],
        output: wp.array[wp.vec3],
    ) -> None:
        """Add cached contact Hessian-vector products to ``output``.

        Args:
            positions: Linearization positions [m], shape ``[particle_count, 3]``.
            vector: Particle-space input vector, shape ``[particle_count, 3]``.
            output: Particle-space accumulation buffer, shape ``[particle_count, 3]``.
        """
        self._validate_vectors((positions, "positions"), (vector, "vector"), (output, "output"))
        wp.launch(
            _static_plane_hessian_multiply,
            dim=self.particle_count,
            inputs=[self.hessians, vector],
            outputs=[output],
            device=self.device,
        )

    def accumulate_diagonal(self, positions: wp.array[wp.vec3], output: wp.array[wp.mat33]) -> None:
        """Add exact cached diagonal contact Hessian blocks.

        Args:
            positions: Linearization positions [m], shape ``[particle_count, 3]``.
            output: Hessian block accumulation buffer [N/m], shape ``[particle_count, 3, 3]``.
        """
        self._validate_vectors((positions, "positions"))
        if len(output) != self.particle_count:
            raise ValueError(f"output must contain {self.particle_count} blocks")
        if output.device != self.device:
            raise ValueError(f"output must use device {self.device}")
        if output.dtype != wp.mat33:
            raise TypeError("output must have dtype wp.mat33")
        wp.launch(
            _accumulate_static_plane_diagonal,
            dim=self.particle_count,
            inputs=[self.hessians],
            outputs=[output],
            device=self.device,
        )

    def _validate_vectors(self, *arrays: tuple[wp.array[wp.vec3], str]) -> None:
        for array, name in arrays:
            if len(array) != self.particle_count:
                raise ValueError(f"{name} must contain {self.particle_count} vectors")
            if array.device != self.device:
                raise ValueError(f"{name} must use device {self.device}")
            if array.dtype != wp.vec3:
                raise TypeError(f"{name} must have dtype wp.vec3")
