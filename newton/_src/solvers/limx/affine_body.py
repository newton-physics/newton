# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Affine body mass data and surface reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .affine_types import mat1212, vec12

_DEFAULT_GRAVITY = np.asarray([0.0, 0.0, -9.81], dtype=np.float64)


@wp.kernel
def _update_affine_surface_positions(
    rest_surface_vertices: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    q: wp.array[vec12],
    output: wp.array[wp.vec3],
):
    vertex = wp.tid()
    rest_position = rest_surface_vertices[vertex]
    state = q[surface_ownership[vertex]]
    output[vertex] = wp.vec3(
        state[0] + state[3] * rest_position[0] + state[4] * rest_position[1] + state[5] * rest_position[2],
        state[1] + state[6] * rest_position[0] + state[7] * rest_position[1] + state[8] * rest_position[2],
        state[2] + state[9] * rest_position[0] + state[10] * rest_position[1] + state[11] * rest_position[2],
    )


def _validate_vertices(rest_vertices: Any) -> np.ndarray:
    vertices = np.asarray(rest_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
        raise ValueError("rest_vertices must have shape (vertex_count, 3) with at least four vertices")
    if not np.isfinite(vertices).all():
        raise ValueError("rest_vertices must be finite")
    return vertices


def _validate_indices(indices: Any, width: int, vertex_count: int, name: str) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 2 or values.shape[1:] != (width,) or len(values) == 0:
        raise ValueError(f"{name} must have non-empty shape (element_count, {width})")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices")
    if np.any(values < 0) or np.any(values >= vertex_count):
        raise ValueError(f"{name} contains an out-of-range vertex index")
    return values.astype(np.int32, copy=False)


def _validate_scalar(value: float, name: str, *, allow_zero: bool) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def _initial_affine_state(initial_transform: Any, rest_centroid: np.ndarray) -> np.ndarray:
    transform_values = np.asarray(initial_transform, dtype=np.float64)
    if transform_values.shape != (7,) or not np.isfinite(transform_values).all():
        raise ValueError("initial_transform must be a finite Warp transform")
    quaternion_norm = np.linalg.norm(transform_values[3:])
    if not np.isclose(quaternion_norm, 1.0, rtol=1.0e-5, atol=1.0e-7):
        raise ValueError("initial_transform rotation must be a unit quaternion")

    transform = wp.transform(*transform_values.tolist())
    translation = np.asarray(wp.transform_get_translation(transform), dtype=np.float64)
    rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform)), dtype=np.float64).reshape(3, 3)
    centered_translation = translation + rotation @ rest_centroid
    return np.concatenate((centered_translation, rotation.reshape(-1)))


def _integrate_mass_moments(vertices: np.ndarray, tetrahedra: np.ndarray, density: float):
    reference_position = vertices[tetrahedra[0, 0]]
    tetrahedron_volumes = np.empty(len(tetrahedra), dtype=np.float64)
    volume = 0.0
    relative_first_volume_moment = np.zeros(3, dtype=np.float64)

    for tetrahedron_index, indices in enumerate(tetrahedra):
        tetrahedron = vertices[indices]
        edge_matrix = np.column_stack(
            (
                tetrahedron[1] - tetrahedron[0],
                tetrahedron[2] - tetrahedron[0],
                tetrahedron[3] - tetrahedron[0],
            )
        )
        tetrahedron_volume = np.linalg.det(edge_matrix) / 6.0
        if not np.isfinite(tetrahedron_volume) or tetrahedron_volume <= 0.0:
            raise ValueError(f"tetrahedron {tetrahedron_index} must have finite positive volume")

        tetrahedron_volumes[tetrahedron_index] = tetrahedron_volume
        volume += tetrahedron_volume
        relative_vertex_sum = np.sum(tetrahedron - reference_position, axis=0)
        relative_first_volume_moment += tetrahedron_volume * relative_vertex_sum / 4.0

    rest_centroid = reference_position + relative_first_volume_moment / volume
    centered_vertices = vertices - rest_centroid
    mass = density * volume
    # The centroid-relative basis has an analytically zero first mass moment.
    first_moment = np.zeros(3, dtype=np.float64)
    second_moment = np.zeros((3, 3), dtype=np.float64)

    for tetrahedron_volume, indices in zip(tetrahedron_volumes, tetrahedra, strict=True):
        tetrahedron = centered_vertices[indices]
        vertex_sum = np.sum(tetrahedron, axis=0)
        vertex_dyad_sum = tetrahedron.T @ tetrahedron
        second_moment += density * tetrahedron_volume * (vertex_dyad_sum + np.outer(vertex_sum, vertex_sum)) / 20.0

    return centered_vertices, rest_centroid, volume, mass, first_moment, second_moment


def _build_mass_matrix(mass: float, first_moment: np.ndarray, second_moment: np.ndarray) -> np.ndarray:
    matrix = np.zeros((12, 12), dtype=np.float64)
    for spatial_axis in range(3):
        translation_index = spatial_axis
        affine_indices = np.arange(3 + 3 * spatial_axis, 6 + 3 * spatial_axis)
        matrix[translation_index, translation_index] = mass
        matrix[translation_index, affine_indices] = first_moment
        matrix[affine_indices, translation_index] = first_moment
        matrix[np.ix_(affine_indices, affine_indices)] = second_moment
    return matrix


def _lift_gravity(
    mass_matrix: np.ndarray,
    mass: float,
    first_moment: np.ndarray,
    gravity: np.ndarray,
) -> np.ndarray:
    force = np.zeros(12, dtype=np.float64)
    force[:3] = mass * gravity
    for spatial_axis in range(3):
        start = 3 + 3 * spatial_axis
        force[start : start + 3] = gravity[spatial_axis] * first_moment
    acceleration = np.linalg.solve(mass_matrix, force)
    residual = mass_matrix @ acceleration - force
    residual_scale = np.linalg.norm(mass_matrix, ord=np.inf) * np.linalg.norm(
        acceleration, ord=np.inf
    ) + np.linalg.norm(force, ord=np.inf)
    residual_tolerance = 64.0 * np.finfo(np.float64).eps * max(residual_scale, np.finfo(np.float64).tiny)
    if not np.isfinite(acceleration).all() or np.linalg.norm(residual, ord=np.inf) > residual_tolerance:
        raise ValueError("integrated gravity solve must have a finite scaled residual")

    return acceleration


class AffineBodyModel:
    """Store one tetrahedral affine body's constant physical data."""

    def __init__(
        self,
        rest_vertices: Any,
        tetrahedron_indices: Any,
        surface_triangle_indices: Any,
        density: float,
        rigidity: float,
        initial_transform: Any,
        device: Any,
    ):
        """Create one affine body from an oriented tetrahedral mesh.

        Args:
            rest_vertices: Rest-space vertex positions [m], shape ``(vertex_count, 3)``.
            tetrahedron_indices: Positively oriented tetrahedra, shape ``(tetrahedron_count, 4)``.
            surface_triangle_indices: Surface triangles into ``rest_vertices``, shape
                ``(triangle_count, 3)``.
            density: Uniform mass density [kg/m^3].
            rigidity: ARAP rigidity coefficient [Pa].
            initial_transform: Initial rigid transform applied to the rest body.
            device: Warp device that owns the model arrays.
        """
        vertices = _validate_vertices(rest_vertices)
        tetrahedra = _validate_indices(tetrahedron_indices, 4, len(vertices), "tetrahedron_indices")
        surface_triangles = _validate_indices(
            surface_triangle_indices,
            3,
            len(vertices),
            "surface_triangle_indices",
        )
        density = _validate_scalar(density, "density", allow_zero=False)
        rigidity = _validate_scalar(rigidity, "rigidity", allow_zero=True)
        device = wp.get_device(device)

        centered_vertices, rest_centroid, volume, mass, first_moment, second_moment = _integrate_mass_moments(
            vertices, tetrahedra, density
        )
        initial_state = _initial_affine_state(initial_transform, rest_centroid)
        integrated_mass_matrix = _build_mass_matrix(mass, first_moment, second_moment)
        mass_matrix = integrated_mass_matrix.astype(np.float32)
        if not np.isfinite(mass_matrix).all():
            raise ValueError("affine mass matrix must be finite")
        try:
            np.linalg.cholesky(mass_matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("affine mass matrix must be positive definite") from error
        gravity = _lift_gravity(integrated_mass_matrix, mass, first_moment, _DEFAULT_GRAVITY)

        surface_vertex_indices = np.unique(surface_triangles.reshape(-1))
        surface_vertex_remap = np.full(len(vertices), -1, dtype=np.int32)
        surface_vertex_remap[surface_vertex_indices] = np.arange(len(surface_vertex_indices), dtype=np.int32)
        compact_surface_triangles = surface_vertex_remap[surface_triangles]
        rest_surface_vertices = centered_vertices[surface_vertex_indices]

        self.device = device
        self.body_count = 1
        self.surface_vertex_count = len(rest_surface_vertices)
        self.surface_triangle_count = len(compact_surface_triangles)
        self.rest_vertices = wp.array(centered_vertices, dtype=wp.vec3, device=device)
        self.tetrahedron_indices = wp.array(tetrahedra, dtype=int, device=device)
        self.rest_surface_vertices = wp.array(rest_surface_vertices, dtype=wp.vec3, device=device)
        self.surface_vertex_indices = wp.array(surface_vertex_indices, dtype=int, device=device)
        self.surface_triangle_indices = wp.array(compact_surface_triangles, dtype=int, device=device)
        self.surface_ownership = wp.zeros(self.surface_vertex_count, dtype=int, device=device)
        self.volumes = wp.array([volume], dtype=float, device=device)
        self.mass_matrices = wp.array([mass_matrix], dtype=mat1212, device=device)
        self.rigidities = wp.array([rigidity], dtype=float, device=device)
        self.gravity = wp.array([gravity], dtype=vec12, device=device)
        self.q = wp.array([initial_state], dtype=vec12, device=device)
        self.qd = wp.zeros(1, dtype=vec12, device=device)

    def update_surface_positions(self, q: wp.array[vec12], output: wp.array[wp.vec3]) -> None:
        """Map rest surface vertices through affine generalized states.

        Args:
            q: Affine generalized states with layout ``[t, row(A,0), row(A,1), row(A,2)]``.
            output: World-space surface positions [m], shape ``(surface_vertex_count, 3)``.
        """
        if q.dtype != vec12 or q.ndim != 1 or len(q) != self.body_count:
            raise ValueError(f"q must contain {self.body_count} vec12 state")
        if output.dtype != wp.vec3 or output.ndim != 1 or len(output) != self.surface_vertex_count:
            raise ValueError(f"output must contain {self.surface_vertex_count} vec3 positions")
        if q.device != self.device or output.device != self.device:
            raise ValueError("q and output must use the affine body model device")
        wp.launch(
            _update_affine_surface_positions,
            dim=self.surface_vertex_count,
            inputs=[self.rest_surface_vertices, self.surface_ownership, q],
            outputs=[output],
            device=self.device,
        )
