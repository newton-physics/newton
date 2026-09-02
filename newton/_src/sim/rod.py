# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Discrete rod input data."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import warp as wp

from ..core.types import Quat, Vec3, axis_to_vec3
from ..math import quat_between_vectors_robust

# Follow AOUSD's solid-circular-section fallback: shear rigidity = kGA,
# where k ~= 0.9. Newton uses k = 0.9.
_CIRCULAR_SECTION_TRANSVERSE_SHEAR_CORRECTION = 0.9


def _validate_nonnegative_float(name: str, value: float) -> float:
    """Validate and normalize a finite, nonnegative float."""
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and >= 0") from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return normalized


def _resolve_shear_modulus(
    youngs_modulus: float,
    *,
    poissons_ratio: float | None,
    shear_modulus: float | None,
) -> float:
    """Resolve shear modulus from Poisson's ratio or an explicit value."""
    if (poissons_ratio is None) == (shear_modulus is None):
        raise ValueError("exactly one of poissons_ratio and shear_modulus must be supplied")

    if shear_modulus is not None:
        return _validate_nonnegative_float("shear_modulus", shear_modulus)

    poisson = float(poissons_ratio)
    if not math.isfinite(poisson):
        raise ValueError("poissons_ratio must be finite")
    if poisson <= -1.0 or poisson >= 0.5:
        raise ValueError("poissons_ratio must satisfy -1 < nu < 0.5")
    return youngs_modulus / (2.0 * (1.0 + poisson))


def _generate_straight_points(
    start: Vec3,
    direction: Vec3,
    length: float,
    segment_count: int,
) -> list[wp.vec3]:
    """Generate uniformly spaced points along a straight centerline."""
    if segment_count < 1:
        raise ValueError("segment_count must be >= 1")
    length_m = float(length)
    if not math.isfinite(length_m):
        raise ValueError("length must be finite")
    if length_m <= 0.0:
        raise ValueError("length must be > 0")

    start_vec = axis_to_vec3(start)
    direction_vec = axis_to_vec3(direction)
    direction_length = float(wp.length(direction_vec))
    if not math.isfinite(direction_length) or direction_length <= 0.0:
        raise ValueError("direction must be finite and non-zero")
    direction_unit = direction_vec / direction_length
    spacing = length_m / segment_count
    return [start_vec + direction_unit * (spacing * i) for i in range(segment_count + 1)]


def _compute_parallel_transport_quaternions(
    points: Sequence[Vec3],
    *,
    twist_total: float = 0.0,
) -> list[wp.quat]:
    """Compute parallel-transport frames for one ordered centerline."""
    if len(points) < 2:
        raise ValueError("points must have length >= 2")

    twist = float(twist_total)
    if not math.isfinite(twist):
        raise ValueError("twist_total must be finite")

    point_vectors = [axis_to_vec3(point) for point in points]
    previous_direction = wp.vec3(0.0, 0.0, 1.0)
    segment_count = len(point_vectors) - 1
    twist_step = twist / segment_count
    frames: list[wp.quat] = []
    for segment_index in range(segment_count):
        segment = point_vectors[segment_index + 1] - point_vectors[segment_index]
        segment_length = float(wp.length(segment))
        if not math.isfinite(segment_length) or segment_length <= 0.0:
            raise ValueError("points must be finite and must not contain duplicate consecutive points")
        direction = segment / segment_length
        rotation = quat_between_vectors_robust(previous_direction, direction, 1.0e-8)
        frame = rotation if segment_index == 0 else wp.mul(rotation, frames[-1])
        if twist != 0.0:
            frame = wp.mul(wp.quat_from_axis_angle(direction, twist_step), frame)
        frames.append(frame)
        previous_direction = direction
    return frames


class Rod:
    """Represents discrete rod input for model construction.

    A rod stores prepared centerline points, segment topology, and one material
    frame per segment. It may additionally store a capsule/cross-section radius
    and either uniform isotropic material properties or a complete set of
    uniform section rigidities. Geometry and constitutive data remain writable.
    Use :meth:`newton.ModelBuilder.add_rod` to create the corresponding bodies
    and rod joints.

    Args:
        points: Centerline node positions in world space [m], shape ``(N, 3)``.
        edges: Optional simple-graph segment endpoint indices, shape ``(E, 2)``.
            Self-edges and duplicate undirected edges are rejected. If omitted,
            consecutive points form an ordered chain.
        quaternions: Optional per-segment material frames in world space, shape
            ``(E, 4)``. If omitted, an ordered chain uses parallel transport;
            other explicit topology aligns each frame's local +Z axis
            independently to its edge.
        closed: Whether an implicit ordered chain closes its last segment body
            back to its first with a rod joint. Valid only when ``edges`` is
            omitted. A closed rod requires at least three segments and
            coincident first and last points.
        radius: Optional capsule radius [m]. If omitted, assembly uses 0.1 m.
            With elastic material inputs, this also defines the circular
            cross-section used to derive section rigidities and is required.
            Directly authored rigidities already include the cross-section and
            are not rescaled by this value.
        youngs_modulus: Optional Young's modulus ``E`` [Pa]. Supplying material
            requires ``radius`` and exactly one of ``poissons_ratio`` and
            ``shear_modulus``.
        poissons_ratio: Optional Poisson's ratio ``nu``.
        shear_modulus: Optional shear modulus ``G`` [Pa].
        stretch_rigidity: Optional axial rigidity ``EA`` [N]. If supplied, all
            four section rigidities are required. Mutually exclusive with
            elastic material inputs.
        shear_rigidity: Optional effective transverse shear rigidity ``kGA``
            [N]. Mutually exclusive with elastic material inputs.
        bend_rigidity: Optional bending rigidity ``EI`` [N·m²]. Mutually
            exclusive with elastic material inputs.
        twist_rigidity: Optional torsional rigidity ``GJ`` [N·m²]. Mutually
            exclusive with elastic material inputs.

    Raises:
        ValueError: If the geometry, frames, or constitutive inputs are invalid
            or mutually inconsistent.

    Note:
        Material-derived transverse shear follows the AOUSD deformable-body
        treatment for a solid circular section, using ``kGA`` with
        ``k = 0.9``. See :meth:`newton.ModelBuilder.add_rod` for rigidity
        discretization, direct stiffness and damping controls, and topology
        limitations.

        Assigning a complete ``points`` or ``edges`` array validates its shape
        and normalizes its dtype and memory layout; in-place array edits bypass
        these property setters. When ``edges`` were omitted, changing the
        number of points regenerates the consecutive chain edges. Assigning
        ``edges`` explicitly disables that automatic update. Neither operation
        recomputes ``quaternions``. After changing points or edges, call
        :meth:`compute_frames` when the material frames should follow the
        updated geometry.
    """

    def __init__(
        self,
        points: Sequence[Vec3] | np.ndarray,
        *,
        edges: Sequence[tuple[int, int]] | np.ndarray | None = None,
        quaternions: Sequence[Quat] | np.ndarray | None = None,
        closed: bool = False,
        radius: float | None = None,
        youngs_modulus: float | None = None,
        poissons_ratio: float | None = None,
        shear_modulus: float | None = None,
        stretch_rigidity: float | None = None,
        shear_rigidity: float | None = None,
        bend_rigidity: float | None = None,
        twist_rigidity: float | None = None,
    ):
        self._edges_are_implicit = edges is None
        self.points = points
        if edges is None:
            self._edges = self._generate_ordered_chain_edges(len(self.points))
            self.closed = bool(closed)
        else:
            if closed:
                raise ValueError("closed is only valid when edges is omitted")
            self.edges = edges
            self.closed = False

        self.radius = None if radius is None else float(radius)
        self.youngs_modulus = None if youngs_modulus is None else float(youngs_modulus)
        self.poissons_ratio = None if poissons_ratio is None else float(poissons_ratio)
        self.shear_modulus = None if shear_modulus is None else float(shear_modulus)
        self.stretch_rigidity = (
            None if stretch_rigidity is None else _validate_nonnegative_float("stretch_rigidity", stretch_rigidity)
        )
        self.shear_rigidity = (
            None if shear_rigidity is None else _validate_nonnegative_float("shear_rigidity", shear_rigidity)
        )
        self.bend_rigidity = (
            None if bend_rigidity is None else _validate_nonnegative_float("bend_rigidity", bend_rigidity)
        )
        self.twist_rigidity = (
            None if twist_rigidity is None else _validate_nonnegative_float("twist_rigidity", twist_rigidity)
        )
        self._resolve_elastic_material()

        if quaternions is None:
            self.compute_frames()
        else:
            self.quaternions = quaternions
            self._points, self._edges, self._quaternions = self._normalize_and_validate_geometry()

    @staticmethod
    def _normalize_points(points: Sequence[Vec3] | np.ndarray) -> np.ndarray:
        try:
            source = np.asarray(points)
            if source.ndim != 2 or source.shape[1] != 3 or np.iscomplexobj(source):
                raise ValueError
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = np.array(source, dtype=np.float32, order="C", copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("points must have shape (N, 3)") from exc
        if len(normalized) < 2:
            raise ValueError("points must contain at least 2 points")
        if not np.isfinite(normalized).all():
            raise ValueError("points must be finite")
        return normalized

    @staticmethod
    def _normalize_edges(edges: Sequence[tuple[int, int]] | np.ndarray, point_count: int) -> np.ndarray:
        try:
            source = np.asarray(edges)
            if source.ndim != 2 or source.shape[1] != 2 or source.dtype.kind not in "iu":
                raise ValueError
            normalized = np.asarray(source, dtype=np.int64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("edges must have shape (E, 2) and contain integer indices") from exc
        if len(normalized) < 1:
            raise ValueError("edges must contain at least 1 edge")
        if normalized.min() < 0 or normalized.max() >= point_count:
            raise ValueError(f"edges must contain indices in [0, {point_count})")
        if np.any(normalized[:, 0] == normalized[:, 1]):
            raise ValueError("edges must not connect a point to itself")
        canonical_edges = {tuple(sorted(edge)) for edge in normalized.tolist()}
        if len(canonical_edges) != len(normalized):
            raise ValueError("edges must not contain duplicates")
        return np.ascontiguousarray(normalized, dtype=np.int32)

    @staticmethod
    def _normalize_quaternions(quaternions: Sequence[Quat] | np.ndarray, segment_count: int) -> np.ndarray:
        try:
            source = np.asarray(quaternions)
            if source.ndim != 2 or source.shape[1] != 4 or np.iscomplexobj(source):
                raise ValueError
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = np.array(source, dtype=np.float32, order="C", copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("quaternions must have shape (E, 4)") from exc
        if len(normalized) != segment_count:
            raise ValueError(f"quaternions must contain {segment_count} frames, got {len(normalized)}")
        if not np.isfinite(normalized).all():
            raise ValueError("quaternions must be finite")
        norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("quaternions must be non-zero")
        return np.ascontiguousarray(normalized / norms[:, None], dtype=np.float32)

    def _resolve_radius(self) -> float | None:
        """Return the normalized capsule/cross-section radius, if specified."""
        if self.radius is None:
            return None
        try:
            radius = float(self.radius)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("radius must be finite and > 0") from exc
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and > 0")
        return radius

    def _resolve_elastic_material(self) -> tuple[float, float, float] | None:
        """Resolve normalized Young's modulus, radius, and shear modulus."""
        radius = self._resolve_radius()

        material_values = (self.youngs_modulus, self.poissons_ratio, self.shear_modulus)
        rigidity_values = (
            self.stretch_rigidity,
            self.shear_rigidity,
            self.bend_rigidity,
            self.twist_rigidity,
        )
        has_material = any(value is not None for value in material_values)
        has_rigidity = any(value is not None for value in rigidity_values)
        if has_material and has_rigidity:
            raise ValueError("elastic material inputs and section rigidity inputs are mutually exclusive")
        if has_rigidity and not all(value is not None for value in rigidity_values):
            raise ValueError(
                "stretch_rigidity, shear_rigidity, bend_rigidity, and twist_rigidity must be supplied together"
            )
        if not has_material:
            return None
        if radius is None:
            raise ValueError("radius is required with elastic material inputs")
        if self.youngs_modulus is None:
            raise ValueError("youngs_modulus is required with elastic material inputs")
        youngs = _validate_nonnegative_float("youngs_modulus", self.youngs_modulus)
        shear_modulus = _resolve_shear_modulus(
            youngs,
            poissons_ratio=self.poissons_ratio,
            shear_modulus=self.shear_modulus,
        )
        return youngs, radius, shear_modulus

    @staticmethod
    def _generate_ordered_chain_edges(point_count: int) -> np.ndarray:
        return np.column_stack((np.arange(point_count - 1, dtype=np.int32), np.arange(1, point_count, dtype=np.int32)))

    @staticmethod
    def _is_ordered_chain_topology(point_count: int, edges: np.ndarray) -> bool:
        return np.array_equal(edges, Rod._generate_ordered_chain_edges(point_count))

    def _validated_centerline(self) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized points and edges after validating centerline geometry."""
        points = self._normalize_points(self.points)
        edges = self._normalize_edges(self.edges, len(points))
        with np.errstate(over="ignore", invalid="ignore"):
            vectors = points[edges[:, 1]] - points[edges[:, 0]]
            lengths = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(lengths).all():
            raise ValueError("all rod segments must have finite length")
        if np.any(lengths <= 1.0e-9):
            raise ValueError("all rod segments must have length > 1e-9 m")
        if self.closed and not self._is_ordered_chain_topology(len(points), edges):
            raise ValueError("closed rods require ordered-chain edges")
        if self.closed and len(edges) < 3:
            raise ValueError("closed rods require at least 3 segments")
        if self.closed and not np.allclose(points[0], points[-1], rtol=0.0, atol=1.0e-6):
            raise ValueError("closed rods require the first and last points to coincide")
        return points, edges

    def _normalize_and_validate_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return normalized geometry after validating the complete rod pose."""
        points, edges = self._validated_centerline()
        quaternions = self._normalize_quaternions(self.quaternions, len(edges))
        local_z = wp.vec3(0.0, 0.0, 1.0)
        for segment_index, ((start_index, end_index), frame) in enumerate(zip(edges, quaternions, strict=True)):
            segment = axis_to_vec3(points[end_index] - points[start_index])
            direction = segment / float(wp.length(segment))
            quaternion = wp.quat(*(float(value) for value in frame))
            alignment = float(wp.dot(direction, wp.quat_rotate(quaternion, local_z)))
            if not math.isfinite(alignment) or alignment < 0.999:
                raise ValueError(
                    f"quaternion at segment index {segment_index} must align local +Z with the segment direction"
                )
        return points, edges, quaternions

    @property
    def points(self) -> np.ndarray:
        """Writable centerline node positions in world space [m], shape ``(N, 3)``, float32."""
        return self._points

    @points.setter
    def points(self, value: Sequence[Vec3] | np.ndarray) -> None:
        points = self._normalize_points(value)
        if hasattr(self, "_edges"):
            if self._edges_are_implicit and len(points) != len(self._points):
                self._edges = self._generate_ordered_chain_edges(len(points))
            else:
                self._normalize_edges(self._edges, len(points))
        self._points = points

    @property
    def edges(self) -> np.ndarray:
        """Writable segment endpoint indices, shape ``(E, 2)``, int32."""
        return self._edges

    @edges.setter
    def edges(self, value: Sequence[tuple[int, int]] | np.ndarray) -> None:
        edges = self._normalize_edges(value, self.point_count)
        self._edges = edges
        self._edges_are_implicit = False

    @property
    def quaternions(self) -> np.ndarray:
        """Writable normalized segment frames in world space, shape ``(E, 4)``, float32."""
        return self._quaternions

    @quaternions.setter
    def quaternions(self, value: Sequence[Quat] | np.ndarray) -> None:
        self._quaternions = self._normalize_quaternions(value, self.segment_count)

    @property
    def point_count(self) -> int:
        """Number of centerline points."""
        return len(self.points)

    @property
    def segment_count(self) -> int:
        """Number of rod segments."""
        return len(self.edges)

    @property
    def segment_lengths(self) -> np.ndarray:
        """Current length of every rod segment [m], shape ``(E,)``."""
        vectors = self.points[self.edges[:, 1]] - self.points[self.edges[:, 0]]
        return np.linalg.norm(vectors, axis=1)

    def _resolve_section_rigidities(self) -> tuple[float, float, float, float] | None:
        """Resolve effective section rigidities EA, kGA, EI, and GJ."""
        material = self._resolve_elastic_material()
        direct_rigidities = (
            self.stretch_rigidity,
            self.shear_rigidity,
            self.bend_rigidity,
            self.twist_rigidity,
        )
        if any(value is not None for value in direct_rigidities):
            stretch_rigidity, shear_rigidity, bend_rigidity, twist_rigidity = direct_rigidities
            assert stretch_rigidity is not None
            assert shear_rigidity is not None
            assert bend_rigidity is not None
            assert twist_rigidity is not None
            return (
                _validate_nonnegative_float("stretch_rigidity", stretch_rigidity),
                _validate_nonnegative_float("shear_rigidity", shear_rigidity),
                _validate_nonnegative_float("bend_rigidity", bend_rigidity),
                _validate_nonnegative_float("twist_rigidity", twist_rigidity),
            )
        if material is None:
            return None
        youngs_modulus, radius, shear_modulus = material
        try:
            area = math.pi * radius**2
            area_moment = 0.25 * math.pi * radius**4
            polar_moment = 0.5 * math.pi * radius**4
            rigidities = (
                youngs_modulus * area,
                _CIRCULAR_SECTION_TRANSVERSE_SHEAR_CORRECTION * shear_modulus * area,
                youngs_modulus * area_moment,
                shear_modulus * polar_moment,
            )
        except OverflowError as exc:
            raise ValueError("elastic material inputs must produce finite section rigidities") from exc
        if not all(math.isfinite(rigidity) for rigidity in rigidities):
            raise ValueError("elastic material inputs must produce finite section rigidities")
        return rigidities

    @staticmethod
    def create_straight(
        start: Vec3,
        direction: Vec3,
        length: float,
        *,
        segment_count: int,
        twist_total: float = 0.0,
        radius: float | None = None,
        youngs_modulus: float | None = None,
        poissons_ratio: float | None = None,
        shear_modulus: float | None = None,
        stretch_rigidity: float | None = None,
        shear_rigidity: float | None = None,
        bend_rigidity: float | None = None,
        twist_rigidity: float | None = None,
    ) -> Rod:
        """Create a uniformly discretized straight rod.

        Args:
            start: First centerline point in world space [m].
            direction: World-space direction, which need not be normalized.
            length: Total centerline length [m].
            segment_count: Number of rod segments.
            twist_total: Twist of the final segment frame relative to untwisted
                parallel transport [rad]. The value is distributed in
                ``segment_count`` equal increments, including the first frame,
                and is not retained after construction.
            radius: Optional circular-section and capsule radius [m].
            youngs_modulus: Optional Young's modulus ``E`` [Pa].
            poissons_ratio: Optional Poisson's ratio ``nu``.
            shear_modulus: Optional shear modulus ``G`` [Pa].
            stretch_rigidity: Optional axial rigidity ``EA`` [N]. If supplied,
                all four section rigidities are required. Mutually exclusive
                with elastic material inputs.
            shear_rigidity: Optional effective transverse shear rigidity
                ``kGA`` [N]. Mutually exclusive with elastic material inputs.
            bend_rigidity: Optional bending rigidity ``EI`` [N·m²]. Mutually
                exclusive with elastic material inputs.
            twist_rigidity: Optional torsional rigidity ``GJ`` [N·m²].
                Mutually exclusive with elastic material inputs.

        Returns:
            A straight rod with parallel-transported material frames.

        Raises:
            ValueError: If the geometry, twist, material, or rigidity inputs are
                invalid.

        """
        points = _generate_straight_points(start, direction, length, segment_count)
        quaternions = _compute_parallel_transport_quaternions(points, twist_total=twist_total)
        return Rod(
            points,
            quaternions=quaternions,
            radius=radius,
            youngs_modulus=youngs_modulus,
            poissons_ratio=poissons_ratio,
            shear_modulus=shear_modulus,
            stretch_rigidity=stretch_rigidity,
            shear_rigidity=shear_rigidity,
            bend_rigidity=bend_rigidity,
            twist_rigidity=twist_rigidity,
        )

    def compute_frames(self, *, twist_total: float = 0.0) -> None:
        """Recompute segment frames from the current centerline geometry.

        Ordered chains use parallel transport and divide ``twist_total`` into
        equal per-segment increments. Other explicit topology uses independent
        edge-aligned frames and does not define graph-wide total twist.

        Args:
            twist_total: Twist of the final ordered-chain frame relative to
                untwisted parallel transport [rad]. Must be zero unless the
                topology is an ordered chain. The value is not retained.

        Returns:
            ``None``.

        Raises:
            ValueError: If the current geometry or ``twist_total`` is invalid,
                or if nonzero total twist is requested for non-chain topology.
        """
        twist = float(twist_total)
        if not math.isfinite(twist):
            raise ValueError("twist_total must be finite")

        points, edges = self._validated_centerline()
        is_ordered_chain = self._is_ordered_chain_topology(len(points), edges)
        if is_ordered_chain:
            frames = _compute_parallel_transport_quaternions(points, twist_total=twist)
        else:
            if twist != 0.0:
                raise ValueError("twist_total is only defined for an ordered chain")
            frames = []
            for start_index, end_index in edges:
                direction = axis_to_vec3(points[end_index] - points[start_index])
                length = float(wp.length(direction))
                frames.append(quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), direction / length, 1.0e-8))
        quaternions = self._normalize_quaternions(frames, len(edges))
        self._quaternions = quaternions

    def copy(self) -> Rod:
        """Return an independent copy of this rod."""
        points, edges, quaternions = self._normalize_and_validate_geometry()
        copied = Rod(
            points,
            edges=edges,
            quaternions=quaternions,
            radius=self.radius,
            youngs_modulus=self.youngs_modulus,
            poissons_ratio=self.poissons_ratio,
            shear_modulus=self.shear_modulus,
            stretch_rigidity=self.stretch_rigidity,
            shear_rigidity=self.shear_rigidity,
            bend_rigidity=self.bend_rigidity,
            twist_rigidity=self.twist_rigidity,
        )
        copied.closed = self.closed
        copied._edges_are_implicit = self._edges_are_implicit
        return copied
