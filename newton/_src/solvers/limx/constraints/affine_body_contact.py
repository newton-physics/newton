# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free surface contact between LIMX affine bodies."""

from __future__ import annotations

import numpy as np
import warp as wp

from ....geometry.kernels import triangle_closest_point_barycentric
from ....utils.mesh import MeshAdjacency
from ..affine_body import AffineBodyModel
from ..affine_types import vec12

_MIN_CONTACT_DISTANCE = 1.0e-7
_EE_MOLLIFIER_THRESHOLD_SCALE = 1.0e-3


@wp.kernel
def _update_affine_triangle_bounds(
    positions: wp.array[wp.vec3],
    triangle_indices: wp.array2d[int],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    triangle = wp.tid()
    position_0 = positions[triangle_indices[triangle, 0]]
    position_1 = positions[triangle_indices[triangle, 1]]
    position_2 = positions[triangle_indices[triangle, 2]]
    lower_bounds[triangle] = wp.min(wp.min(position_0, position_1), position_2)
    upper_bounds[triangle] = wp.max(wp.max(position_0, position_1), position_2)


@wp.kernel
def _update_affine_edge_bounds(
    positions: wp.array[wp.vec3],
    edge_indices: wp.array2d[int],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    edge = wp.tid()
    position_0 = positions[edge_indices[edge, 2]]
    position_1 = positions[edge_indices[edge, 3]]
    lower_bounds[edge] = wp.min(position_0, position_1)
    upper_bounds[edge] = wp.max(position_0, position_1)


@wp.kernel
def _detect_affine_vertex_face_contacts(
    triangle_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    triangle_indices: wp.array2d[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    contact_count: wp.array[int],
    overflow_count: wp.array[int],
):
    vertex = wp.tid()
    vertex_position = positions[vertex]
    query = wp.bvh_query_aabb(
        triangle_bvh_id,
        vertex_position - wp.vec3(thickness),
        vertex_position + wp.vec3(thickness),
    )
    triangle = wp.int32(-1)
    while wp.bvh_query_next(query, triangle):
        index_0 = triangle_indices[triangle, 0]
        index_1 = triangle_indices[triangle, 1]
        index_2 = triangle_indices[triangle, 2]
        if surface_ownership[vertex] == surface_ownership[index_0]:
            continue

        position_0 = positions[index_0]
        position_1 = positions[index_1]
        position_2 = positions[index_2]
        barycentric = triangle_closest_point_barycentric(
            position_0,
            position_1,
            position_2,
            vertex_position,
        )
        closest_point = barycentric[0] * position_0 + barycentric[1] * position_1 + barycentric[2] * position_2
        separation = vertex_position - closest_point
        distance = wp.length(separation)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= thickness:
            continue

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue

        contact_ids[contact, 0] = vertex
        contact_ids[contact, 1] = index_0
        contact_ids[contact, 2] = index_1
        contact_ids[contact, 3] = index_2
        contact_weights[contact, 0] = 1.0
        contact_weights[contact, 1] = -barycentric[0]
        contact_weights[contact, 2] = -barycentric[1]
        contact_weights[contact, 3] = -barycentric[2]
        contact_directions[contact] = separation / distance
        contact_depths[contact] = thickness - distance


@wp.kernel
def _detect_affine_edge_edge_contacts(
    edge_bvh_id: wp.uint64,
    thickness: float,
    capacity: int,
    positions: wp.array[wp.vec3],
    rest_positions: wp.array[wp.vec3],
    surface_ownership: wp.array[int],
    edge_indices: wp.array2d[int],
    contact_ids: wp.array2d[int],
    contact_weights: wp.array2d[float],
    contact_directions: wp.array[wp.vec3],
    contact_depths: wp.array[float],
    mollifier_thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    contact_count: wp.array[int],
    overflow_count: wp.array[int],
):
    edge = wp.tid()
    index_0 = edge_indices[edge, 2]
    index_1 = edge_indices[edge, 3]
    position_0 = positions[index_0]
    position_1 = positions[index_1]
    query = wp.bvh_query_aabb(
        edge_bvh_id,
        wp.min(position_0, position_1) - wp.vec3(thickness),
        wp.max(position_0, position_1) + wp.vec3(thickness),
    )
    other_edge = wp.int32(-1)
    while wp.bvh_query_next(query, other_edge):
        if other_edge <= edge:
            continue
        index_2 = edge_indices[other_edge, 2]
        index_3 = edge_indices[other_edge, 3]
        if surface_ownership[index_0] == surface_ownership[index_2]:
            continue

        position_2 = positions[index_2]
        position_3 = positions[index_3]
        parameters = wp.closest_point_edge_edge(position_0, position_1, position_2, position_3, 1.0e-5)
        parameter_0 = parameters[0]
        parameter_1 = parameters[1]
        if parameter_0 <= 0.0 or parameter_0 >= 1.0:
            continue
        if parameter_1 <= 0.0 or parameter_1 >= 1.0:
            continue

        closest_0 = wp.lerp(position_0, position_1, parameter_0)
        closest_1 = wp.lerp(position_2, position_3, parameter_1)
        separation = closest_0 - closest_1
        distance = wp.length(separation)
        if distance <= _MIN_CONTACT_DISTANCE or distance >= thickness:
            continue

        contact = wp.atomic_add(contact_count, 0, 1)
        if contact >= capacity:
            wp.atomic_add(overflow_count, 0, 1)
            continue

        contact_ids[contact, 0] = index_0
        contact_ids[contact, 1] = index_1
        contact_ids[contact, 2] = index_2
        contact_ids[contact, 3] = index_3
        contact_weights[contact, 0] = 1.0 - parameter_0
        contact_weights[contact, 1] = parameter_0
        contact_weights[contact, 2] = -(1.0 - parameter_1)
        contact_weights[contact, 3] = -parameter_1
        contact_directions[contact] = separation / distance
        contact_depths[contact] = thickness - distance
        rest_edge_0 = rest_positions[index_1] - rest_positions[index_0]
        rest_edge_1 = rest_positions[index_3] - rest_positions[index_2]
        mollifier_thresholds[contact] = (
            _EE_MOLLIFIER_THRESHOLD_SCALE * wp.dot(rest_edge_0, rest_edge_0) * wp.dot(rest_edge_1, rest_edge_1)
        )
        mollifier_active[contact] = 0


class _AffineContactBuffer:
    def __init__(self, capacity: int, device: wp.context.Device):
        self.capacity = capacity
        self.device = device
        self.ids = wp.empty((capacity, 4), dtype=wp.int32, device=device)
        self.weights = wp.empty((capacity, 4), dtype=wp.float32, device=device)
        self.directions = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.depths = wp.empty(capacity, dtype=wp.float32, device=device)
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.overflow_count = wp.zeros(1, dtype=wp.int32, device=device)

    def clear(self) -> None:
        self.count.zero_()
        self.overflow_count.zero_()


class _AffineEdgeEdgeContactBuffer(_AffineContactBuffer):
    def __init__(self, capacity: int, device: wp.context.Device):
        super().__init__(capacity, device)
        self.mollifier_thresholds = wp.empty(capacity, dtype=wp.float32, device=device)
        self.mollifier_active = wp.zeros(capacity, dtype=wp.int32, device=device)


class ConstraintAffineBodyContact:
    """Detect frozen VF and strict-interior EE contact between affine bodies."""

    def __init__(
        self,
        body_model: AffineBodyModel,
        thickness: float,
        stiffness: float,
        normal_damping: float,
        friction: float,
        friction_epsilon: float,
        max_contacts: int = 262144,
    ):
        """Create affine-body surface contact.

        Args:
            body_model: Multi-body affine model supplying compact surface geometry.
            thickness: Two-surface contact activation distance [m].
            stiffness: Normal penalty stiffness per retained contact [N/m].
            normal_damping: Approaching normal damping coefficient [N·s/m].
            friction: Coulomb friction coefficient.
            friction_epsilon: Tangential displacement regularization [m].
            max_contacts: Maximum stored contacts for each contact type.
        """
        if not isinstance(body_model, AffineBodyModel):
            raise TypeError("body_model must be an AffineBodyModel")
        if body_model.body_count < 2:
            raise ValueError("body_model must contain at least two affine bodies")
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
        if max_contacts <= 0:
            raise ValueError("max_contacts must be positive")

        self.body_model = body_model
        self.body_count = body_model.body_count
        self.device = body_model.device
        self.thickness = float(thickness)
        self.stiffness = float(stiffness)
        self.normal_damping = float(normal_damping)
        self.friction = float(friction)
        self.friction_epsilon = float(friction_epsilon)
        self.max_contacts = int(max_contacts)

        triangles = np.asarray(body_model.surface_triangle_indices.numpy(), dtype=np.int32)
        ownership = np.asarray(body_model.surface_ownership.numpy(), dtype=np.int32)
        triangle_ownership = ownership[triangles[:, 0]]
        if not np.all(ownership[triangles] == triangle_ownership[:, None]):
            raise ValueError("Every surface triangle must belong to one affine body")
        adjacency = MeshAdjacency(triangles)
        edges = adjacency.edge_indices
        edge_ownership = ownership[edges[:, 2]]
        if not np.all(ownership[edges[:, 2:4]] == edge_ownership[:, None]):
            raise ValueError("Every surface edge must belong to one affine body")

        self.triangle_indices = wp.array(triangles, dtype=wp.int32, device=self.device)
        self.edge_indices = wp.array(edges, dtype=wp.int32, device=self.device)
        self.triangle_count = len(triangles)
        self.edge_count = len(edges)
        self.surface_vertex_count = body_model.surface_vertex_count
        self.positions = wp.empty(self.surface_vertex_count, dtype=wp.vec3, device=self.device)
        self.triangle_lower_bounds = wp.empty(self.triangle_count, dtype=wp.vec3, device=self.device)
        self.triangle_upper_bounds = wp.empty_like(self.triangle_lower_bounds)
        self.edge_lower_bounds = wp.empty(self.edge_count, dtype=wp.vec3, device=self.device)
        self.edge_upper_bounds = wp.empty_like(self.edge_lower_bounds)
        body_model.update_surface_positions(body_model.q, self.positions)
        self._update_bounds()
        self.triangle_bvh = wp.Bvh(self.triangle_lower_bounds, self.triangle_upper_bounds)
        self.edge_bvh = wp.Bvh(self.edge_lower_bounds, self.edge_upper_bounds)
        self.vertex_face_contacts = _AffineContactBuffer(self.max_contacts, self.device)
        self.edge_edge_contacts = _AffineEdgeEdgeContactBuffer(self.max_contacts, self.device)
        self._velocities: wp.array[vec12] | None = None
        self._dt = 0.0
        self._prepared = False

    def begin_step(self, q: wp.array[vec12], qd: wp.array[vec12], dt: float) -> None:
        """Cache step-start affine velocity for later contact response.

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
        """Reconstruct surfaces and freeze cross-body contacts.

        Args:
            q: Current affine generalized states.
        """
        if self._velocities is None:
            raise RuntimeError("begin_step() must be called before prepare()")
        self._validate_affine_vectors((q, "q"))
        self.body_model.update_surface_positions(q, self.positions)
        self._update_bounds()
        self.triangle_bvh.refit()
        self.edge_bvh.refit()
        self.vertex_face_contacts.clear()
        self.edge_edge_contacts.clear()
        wp.launch(
            _detect_affine_vertex_face_contacts,
            dim=self.surface_vertex_count,
            inputs=[
                self.triangle_bvh.id,
                self.thickness,
                self.max_contacts,
                self.positions,
                self.body_model.surface_ownership,
                self.triangle_indices,
            ],
            outputs=[
                self.vertex_face_contacts.ids,
                self.vertex_face_contacts.weights,
                self.vertex_face_contacts.directions,
                self.vertex_face_contacts.depths,
                self.vertex_face_contacts.count,
                self.vertex_face_contacts.overflow_count,
            ],
            device=self.device,
        )
        wp.launch(
            _detect_affine_edge_edge_contacts,
            dim=self.edge_count,
            inputs=[
                self.edge_bvh.id,
                self.thickness,
                self.max_contacts,
                self.positions,
                self.body_model.rest_surface_vertices,
                self.body_model.surface_ownership,
                self.edge_indices,
            ],
            outputs=[
                self.edge_edge_contacts.ids,
                self.edge_edge_contacts.weights,
                self.edge_edge_contacts.directions,
                self.edge_edge_contacts.depths,
                self.edge_edge_contacts.mollifier_thresholds,
                self.edge_edge_contacts.mollifier_active,
                self.edge_edge_contacts.count,
                self.edge_edge_contacts.overflow_count,
            ],
            device=self.device,
        )
        self._prepared = True

    def _update_bounds(self) -> None:
        wp.launch(
            _update_affine_triangle_bounds,
            dim=self.triangle_count,
            inputs=[self.positions, self.triangle_indices],
            outputs=[self.triangle_lower_bounds, self.triangle_upper_bounds],
            device=self.device,
        )
        wp.launch(
            _update_affine_edge_bounds,
            dim=self.edge_count,
            inputs=[self.positions, self.edge_indices],
            outputs=[self.edge_lower_bounds, self.edge_upper_bounds],
            device=self.device,
        )

    def _validate_affine_vectors(self, *arrays: tuple[wp.array[vec12], str]) -> None:
        for array, name in arrays:
            if len(array) != self.body_count:
                raise ValueError(f"{name} must contain {self.body_count} vectors")
            if array.device != self.device:
                raise ValueError(f"{name} must use device {self.device}")
            if array.dtype != vec12:
                raise TypeError(f"{name} must have dtype vec12")
