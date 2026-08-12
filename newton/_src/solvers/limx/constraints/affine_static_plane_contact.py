# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free affine-body contact against a static plane."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from ..affine_body import AffineBodyModel
from ..affine_types import mat1212, vec12


@wp.func
def _affine_point(state: vec12, rest_position: wp.vec3) -> wp.vec3:
    return wp.vec3(
        state[0] + state[3] * rest_position[0] + state[4] * rest_position[1] + state[5] * rest_position[2],
        state[1] + state[6] * rest_position[0] + state[7] * rest_position[1] + state[8] * rest_position[2],
        state[2] + state[9] * rest_position[0] + state[10] * rest_position[1] + state[11] * rest_position[2],
    )


@wp.func
def _lift_affine_vector(world_vector: wp.vec3, rest_position: wp.vec3) -> vec12:
    result = vec12(0.0)
    for axis in range(3):
        value = world_vector[axis]
        result[axis] = value
        for coordinate in range(3):
            result[3 + 3 * axis + coordinate] = value * rest_position[coordinate]
    return result


@wp.func
def _jacobian_axis(index: int) -> int:
    if index < 3:
        return index
    return (index - 3) // 3


@wp.func
def _jacobian_weight(index: int, rest_position: wp.vec3) -> float:
    if index < 3:
        return 1.0
    return rest_position[(index - 3) % 3]


@wp.func
def _lift_affine_matrix(world_matrix: wp.mat33, rest_position: wp.vec3) -> mat1212:
    result = mat1212(0.0)
    for row in range(12):
        world_row = _jacobian_axis(row)
        row_weight = _jacobian_weight(row, rest_position)
        for column in range(12):
            world_column = _jacobian_axis(column)
            column_weight = _jacobian_weight(column, rest_position)
            result[row, column] = row_weight * world_matrix[world_row, world_column] * column_weight
    return result


@wp.kernel
def _prepare_affine_static_plane_contact(
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    states: wp.array[vec12],
    velocities: wp.array[vec12],
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
    surface_vertex = wp.tid()
    body = surface_ownership[surface_vertex]
    rest_position = rest_positions[surface_vertex]
    position = _affine_point(states[body], rest_position)
    distance = wp.dot(normal, position) - offset
    if distance >= thickness:
        forces[surface_vertex] = wp.vec3(0.0)
        hessians[surface_vertex] = wp.mat33(0.0)
        return

    depth = thickness - distance
    normal_outer = wp.outer(normal, normal)
    force = stiffness * depth * normal
    hessian = stiffness * normal_outer

    velocity = _affine_point(velocities[body], rest_position)
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

    forces[surface_vertex] = force
    hessians[surface_vertex] = hessian


@wp.kernel
def _accumulate_affine_static_plane_force(
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    forces: wp.array[wp.vec3],
    output: wp.array[vec12],
):
    surface_vertex = wp.tid()
    body = surface_ownership[surface_vertex]
    lifted_force = _lift_affine_vector(forces[surface_vertex], rest_positions[surface_vertex])
    wp.atomic_add(output, body, lifted_force)


@wp.kernel
def _affine_static_plane_hessian_multiply(
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    hessians: wp.array[wp.mat33],
    vector: wp.array[vec12],
    output: wp.array[vec12],
):
    surface_vertex = wp.tid()
    body = surface_ownership[surface_vertex]
    rest_position = rest_positions[surface_vertex]
    world_vector = _affine_point(vector[body], rest_position)
    lifted_product = _lift_affine_vector(hessians[surface_vertex] * world_vector, rest_position)
    wp.atomic_add(output, body, lifted_product)


@wp.kernel
def _accumulate_affine_static_plane_diagonal(
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    hessians: wp.array[wp.mat33],
    output: wp.array[mat1212],
):
    surface_vertex = wp.tid()
    body = surface_ownership[surface_vertex]
    lifted_hessian = _lift_affine_matrix(hessians[surface_vertex], rest_positions[surface_vertex])
    wp.atomic_add(output, body, lifted_hessian)


class ConstraintAffineStaticPlaneContact:
    """Matrix-free material-point contact for affine bodies and a fixed plane."""

    def __init__(
        self,
        body_model: AffineBodyModel,
        normal: Sequence[float],
        offset: float,
        thickness: float,
        stiffness: float,
        normal_damping: float,
        friction: float,
        friction_epsilon: float,
    ):
        """Create affine-body contact against a one-sided static plane.

        Args:
            body_model: Affine body model supplying centered surface samples.
            normal: Outward plane normal.
            offset: Plane offset from the origin [m].
            thickness: One-sided contact activation distance [m].
            stiffness: Normal penalty stiffness per surface sample [N/m].
            normal_damping: Approaching normal damping coefficient per surface sample [N·s/m].
            friction: Coulomb friction coefficient.
            friction_epsilon: Tangential displacement regularization [m].
        """
        if not isinstance(body_model, AffineBodyModel):
            raise TypeError("body_model must be an AffineBodyModel")
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

        normalized = normal_array / normal_length
        self.body_model = body_model
        self.body_count = body_model.body_count
        self.surface_vertex_count = body_model.surface_vertex_count
        self.device = body_model.device
        self.normal = wp.vec3(float(normalized[0]), float(normalized[1]), float(normalized[2]))
        self.offset = float(offset)
        self.thickness = float(thickness)
        self.stiffness = float(stiffness)
        self.normal_damping = float(normal_damping)
        self.friction = float(friction)
        self.friction_epsilon = float(friction_epsilon)
        self.forces = wp.empty(self.surface_vertex_count, dtype=wp.vec3, device=self.device)
        self.hessians = wp.empty(self.surface_vertex_count, dtype=wp.mat33, device=self.device)
        self._velocities: wp.array[vec12] | None = None
        self._dt = 0.0
        self._prepared = False

    def begin_step(self, q: wp.array[vec12], qd: wp.array[vec12], dt: float) -> None:
        """Cache step-start generalized velocity for damping and friction.

        Args:
            q: Step-start affine generalized states.
            qd: Step-start affine generalized velocities [m/s, 1/s].
            dt: Simulation time step [s].
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self._validate_affine_vectors((q, "q"), (qd, "qd"))
        self._velocities = qd
        self._dt = float(dt)
        self._prepared = False

    def prepare(self, q: wp.array[vec12]) -> None:
        """Cache world-space force and PSD Hessian at the current iterate.

        Args:
            q: Current affine generalized states.
        """
        if self._velocities is None:
            raise RuntimeError("begin_step() must be called before prepare()")
        self._validate_affine_vectors((q, "q"))
        wp.launch(
            _prepare_affine_static_plane_contact,
            dim=self.surface_vertex_count,
            inputs=[
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
                q,
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
        self._prepared = True

    def accumulate_force(self, q: wp.array[vec12], output: wp.array[vec12]) -> None:
        """Add cached world contact forces lifted by each point Jacobian.

        Args:
            q: Current affine generalized states.
            output: Affine generalized force accumulation buffer [N, N·m].
        """
        self._require_prepared()
        self._validate_affine_vectors((q, "q"), (output, "output"))
        wp.launch(
            _accumulate_affine_static_plane_force,
            dim=self.surface_vertex_count,
            inputs=[self.body_model.rest_surface_vertices, self.body_model.surface_ownership, self.forces],
            outputs=[output],
            device=self.device,
        )

    def multiply(
        self,
        particle_input: wp.array[wp.vec3],
        affine_input: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        """Add matrix-free ``J.T @ H @ J`` products to affine output."""
        self._require_prepared()
        self._validate_empty_particle_vector(particle_input, "particle_input")
        self._validate_empty_particle_vector(particle_output, "particle_output")
        self._validate_affine_vectors((affine_input, "affine_input"), (affine_output, "affine_output"))
        wp.launch(
            _affine_static_plane_hessian_multiply,
            dim=self.surface_vertex_count,
            inputs=[
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
                self.hessians,
                affine_input,
            ],
            outputs=[affine_output],
            device=self.device,
        )

    def accumulate_diagonal(
        self,
        particle_diagonal: wp.array[wp.mat33],
        affine_diagonal: wp.array[mat1212],
    ) -> None:
        """Add exact lifted affine blocks to the block-Jacobi diagonal."""
        self._require_prepared()
        self._validate_empty_particle_diagonal(particle_diagonal)
        if len(affine_diagonal) != self.body_count:
            raise ValueError(f"affine_diagonal must contain {self.body_count} blocks")
        if affine_diagonal.device != self.device:
            raise ValueError(f"affine_diagonal must use device {self.device}")
        if affine_diagonal.dtype != mat1212:
            raise TypeError("affine_diagonal must have dtype mat1212")
        wp.launch(
            _accumulate_affine_static_plane_diagonal,
            dim=self.surface_vertex_count,
            inputs=[self.body_model.rest_surface_vertices, self.body_model.surface_ownership, self.hessians],
            outputs=[affine_diagonal],
            device=self.device,
        )

    def _require_prepared(self) -> None:
        if not self._prepared:
            raise RuntimeError("prepare() must be called before using affine contact contributions")

    def _validate_affine_vectors(self, *arrays: tuple[wp.array[vec12], str]) -> None:
        for array, name in arrays:
            if len(array) != self.body_count:
                raise ValueError(f"{name} must contain {self.body_count} vectors")
            if array.device != self.device:
                raise ValueError(f"{name} must use device {self.device}")
            if array.dtype != vec12:
                raise TypeError(f"{name} must have dtype vec12")

    def _validate_empty_particle_vector(self, array: wp.array[wp.vec3], name: str) -> None:
        if len(array) != 0:
            raise ValueError(f"{name} must be empty for affine-only contact")
        if array.device != self.device:
            raise ValueError(f"{name} must use device {self.device}")
        if array.dtype != wp.vec3:
            raise TypeError(f"{name} must have dtype wp.vec3")

    def _validate_empty_particle_diagonal(self, array: wp.array[wp.mat33]) -> None:
        if len(array) != 0:
            raise ValueError("particle_diagonal must be empty for affine-only contact")
        if array.device != self.device:
            raise ValueError(f"particle_diagonal must use device {self.device}")
        if array.dtype != wp.mat33:
            raise TypeError("particle_diagonal must have dtype wp.mat33")
