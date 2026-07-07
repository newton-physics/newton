# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Lidar Sensor - measures ray-hit distances along a spherical scan pattern around sensor sites."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..geometry.flags import ShapeFlags
from ..geometry.raycast import ray_intersect_geom
from ..geometry.types import GeoType
from ..sim.model import Model
from ..sim.state import State
from ..utils.selection import match_labels


@wp.kernel
def compute_sensor_lidar_kernel(
    # sensor attachment
    sensor_sites: wp.array[int],
    sensor_worlds: wp.array[wp.int32],
    ray_directions: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    # scene shapes (shape BVH built by Model.bvh_build_shapes)
    bvh_id: wp.uint64,
    bvh_shapes_group_roots: wp.array[wp.int32],
    bvh_shape_enabled: wp.array[wp.uint32],
    shape_transform_world: wp.array[wp.transform],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    # range limits
    min_range: float,
    max_range: float,
    # output
    distances: wp.array2d[float],
):
    """Cast one lidar ray per (sensor, ray) thread against the shapes of the sensor's world.

    Rays are expressed in the sensor site frame and transformed to world space
    using the site's parent body transform from ``body_q``. Hits are found by
    querying the model shape BVH and dispatching to the internal
    ``ray_intersect_geom()`` helper per candidate shape, mirroring
    :func:`newton.intersect_ray`.

    Args:
        sensor_sites: Site (shape) index per sensor.
        sensor_worlds: World index per sensor (``-1`` for the global world).
        ray_directions: Unit ray directions in the sensor site frame, shape ``(n_rays,)``.
        shape_body: Maps shape index to parent body index (``-1`` for static shapes).
        shape_transform: Local shape transforms relative to the parent body.
        body_q: Body world transforms from the current state.
        bvh_id: Warp BVH id of the model shape BVH.
        bvh_shapes_group_roots: BVH subtree root per world group; the last entry is the global world.
        bvh_shape_enabled: Maps BVH leaf index to shape index.
        shape_transform_world: World transforms of all shapes as refit into the BVH.
        shape_type: Geometry type per shape.
        shape_scale: Geometry scale per shape.
        shape_source_ptr: Warp mesh ids for MESH, CONVEX_MESH, and HFIELD shapes.
        min_range: Hits closer than this distance [m] are ignored.
        max_range: Maximum hit distance [m]; the BVH query is pruned beyond it.
        distances: Output hit distances [m], shape ``(n_sensors, n_rays)``; ``-1.0`` on miss.
    """
    sensor_idx, ray_idx = wp.tid()

    site_idx = sensor_sites[sensor_idx]
    body_idx = shape_body[site_idx]

    # Sensor pose is always taken from the current state, independent of BVH refits.
    X_ws = shape_transform[site_idx]
    if body_idx >= 0:
        X_ws = wp.transform_multiply(body_q[body_idx], X_ws)

    origin = X_ws.p
    direction = wp.quat_rotate(X_ws.q, ray_directions[ray_idx])

    min_dist = float(max_range)
    min_shape_id = wp.int32(-1)

    # Pass 0 queries the sensor's own world; pass 1 the global world shared by all.
    for i in range(2):
        groupid = bvh_shapes_group_roots.shape[0] - 1
        if i == 0:
            groupid = sensor_worlds[sensor_idx]

        bvh_root = bvh_shapes_group_roots[groupid]
        if bvh_root < 0:
            continue

        query = wp.bvh_query_ray(bvh_id, origin, direction, bvh_root)
        bvh_shape_id = wp.int32(0)

        while wp.bvh_query_next(query, bvh_shape_id, min_dist):
            shape_id = wp.int32(bvh_shape_enabled[bvh_shape_id])

            # Never report the sensor's own site shape (present in the BVH when visible).
            if shape_id != site_idx:
                geom_type = shape_type[shape_id]

                mesh_id = wp.uint64(0)
                if geom_type == GeoType.MESH or geom_type == GeoType.CONVEX_MESH or geom_type == GeoType.HFIELD:
                    mesh_id = shape_source_ptr[shape_id]

                hit_dist, _hit_normal = ray_intersect_geom(
                    shape_transform_world[shape_id],
                    shape_scale[shape_id],
                    geom_type,
                    origin,
                    direction,
                    mesh_id,
                )
                # min_dist starts at max_range, so the upper bound is exclusive: a hit at
                # exactly max_range is a miss (the BVH query also prunes at this bound).
                if hit_dist >= min_range and hit_dist < min_dist:
                    min_dist = hit_dist
                    min_shape_id = shape_id

    distances[sensor_idx, ray_idx] = wp.where(min_shape_id < 0, -1.0, min_dist)


class SensorLidar:
    """Lidar sensor measuring ray-hit distances along a configurable spherical scan pattern.

    Each site defines a lidar frame from which a fan of
    ``azimuth_count * elevation_count`` rays is cast against the shapes of the
    site's world (and the global world, whose shapes are visible from every
    world). Rays are laid out in the site frame with ``+X`` forward
    (azimuth 0), azimuth rotating about ``+Z`` toward ``+Y``, and elevation
    tilting toward ``+Z``:

    ``direction = (cos(el) * cos(az), cos(el) * sin(az), sin(el))``

    Azimuth samples are endpoint-exclusive
    (``az = azimuth_min + i * (azimuth_max - azimuth_min) / azimuth_count``),
    so the default full ``[-pi, pi)`` sweep has no duplicate ray. Elevation
    rings are endpoint-inclusive; a single ring uses the midpoint of the
    elevation limits. Ray ``r`` of ring ``e`` is stored at index
    ``e * azimuth_count + r``.

    Hits at or farther than ``max_range`` (an exclusive bound) are reported
    as misses (``-1.0``, following the convention of
    :func:`newton.intersect_ray` and the tiled camera depth channel). Hits
    closer than ``min_range`` are ignored and the ray continues to farther
    geometry, which can be used to look past shapes mounted near the sensor
    origin.

    Raycasting uses the model shape BVH built by
    :meth:`~newton.ModelBuilder.finalize` for the initial state. Before
    updates on states where shapes have moved, refit it via
    :meth:`~newton.Model.bvh_refit_shapes`. The sensor pose itself is always
    evaluated from ``state.body_q``, so a moving sensor in an otherwise
    static scene needs no refit. Shapes without the ``VISIBLE`` flag are
    excluded, matching :class:`~newton.sensors.SensorTiledCamera`.

    The ``sites`` parameter accepts label patterns -- see :ref:`label-matching`.

    Example:

        .. testcode::

            import warp as wp
            import newton
            from newton.sensors import SensorLidar

            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform((0, 0, 1), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.1)
            builder.add_site(body, label="lidar_0")
            model = builder.finalize()

            lidar = SensorLidar(model, sites="lidar_*", max_range=50.0)
            state = model.state()

            # Refit the shape BVH whenever shapes have moved since finalize().
            model.bvh_refit_shapes(state)
            lidar.update(state)
            distances = lidar.distances.numpy()  # shape: (n_sensors, 144)
    """

    distances: wp.array2d[float]
    """Hit distances [m] in the range ``[min_range, max_range)``, shape ``(n_sensors, n_rays)``; ``-1.0`` on miss."""

    ray_directions: wp.array[wp.vec3]
    """Unit ray directions in the sensor site frame, shape ``(n_rays,)``."""

    def __init__(
        self,
        model: Model,
        sites: str | list[str] | list[int],
        *,
        azimuth_count: int = 36,
        elevation_count: int = 4,
        azimuth_min: float = -np.pi,
        azimuth_max: float = np.pi,
        elevation_min: float = -np.pi / 12.0,
        elevation_max: float = np.pi / 12.0,
        min_range: float = 0.0,
        max_range: float = 100.0,
        verbose: bool | None = None,
    ):
        """Initialize SensorLidar.

        Args:
            model: The model to use.
            sites: List of site indices, single pattern to match against site
                labels, or list of patterns where any one matches.
            azimuth_count: Number of rays per elevation ring.
            elevation_count: Number of elevation rings.
            azimuth_min: Start of the azimuth sweep [rad] about the site ``+Z`` axis.
            azimuth_max: End of the azimuth sweep [rad]; endpoint-exclusive.
            elevation_min: Elevation [rad] of the lowest ring; endpoint-inclusive.
            elevation_max: Elevation [rad] of the highest ring; endpoint-inclusive.
            min_range: Hits closer than this distance [m] are ignored.
            max_range: Hits at or farther than this distance [m] (exclusive bound) are reported as misses.
            verbose: If True, print details. If False, suppress details. If None, print details when
                ``wp.config.log_level`` is configured for debug logging.

        Raises:
            ValueError: If no labels match, invalid sites are passed, or the scan
                pattern or range parameters are invalid.
        """
        self.model = model
        self.verbose = verbose if verbose is not None else wp.config.log_level <= wp.LOG_DEBUG

        original_sites = sites
        sites = match_labels(model.shape_label, sites)
        if not sites:
            if isinstance(original_sites, list) and len(original_sites) == 0:
                raise ValueError("'sites' must not be empty")
            raise ValueError(f"No sites matched the given pattern {original_sites!r}")

        self._validate_sensor_sites(sites)

        if azimuth_count < 1:
            raise ValueError(f"'azimuth_count' must be at least 1, got {azimuth_count}")
        if elevation_count < 1:
            raise ValueError(f"'elevation_count' must be at least 1, got {elevation_count}")
        for name, value in (
            ("azimuth_min", azimuth_min),
            ("azimuth_max", azimuth_max),
            ("elevation_min", elevation_min),
            ("elevation_max", elevation_max),
            ("min_range", min_range),
            ("max_range", max_range),
        ):
            if not math.isfinite(value):
                raise ValueError(f"'{name}' must be finite, got {value}")
        if azimuth_max < azimuth_min:
            raise ValueError(f"'azimuth_max' ({azimuth_max}) must not be less than 'azimuth_min' ({azimuth_min})")
        if elevation_max < elevation_min:
            raise ValueError(
                f"'elevation_max' ({elevation_max}) must not be less than 'elevation_min' ({elevation_min})"
            )
        if min_range < 0.0:
            raise ValueError(f"'min_range' must be non-negative, got {min_range}")
        if max_range <= min_range:
            raise ValueError(f"'max_range' ({max_range}) must be greater than 'min_range' ({min_range})")

        self.azimuth_count = azimuth_count
        self.elevation_count = elevation_count
        self.min_range = min_range
        self.max_range = max_range

        self.n_sensors: int = len(sites)
        self.n_rays: int = azimuth_count * elevation_count

        self.sensor_sites_arr = wp.array(sites, dtype=int, device=model.device)

        # Each sensor casts rays in the world of its site (-1 selects the global world only).
        shape_world = model.shape_world.numpy()
        self.sensor_worlds_arr = wp.array(
            [int(shape_world[site_idx]) for site_idx in sites], dtype=wp.int32, device=model.device
        )

        self.ray_directions = wp.array(
            self._compute_ray_directions(
                azimuth_count, elevation_count, azimuth_min, azimuth_max, elevation_min, elevation_max
            ),
            dtype=wp.vec3,
            device=model.device,
        )

        self.distances = wp.full((self.n_sensors, self.n_rays), -1.0, dtype=float, device=model.device)

        if self.verbose:
            print("SensorLidar initialized:")
            print(f"  Sites: {len(set(sites))}")
            print(f"  Rays per sensor: {self.n_rays} ({azimuth_count} azimuth x {elevation_count} elevation)")
            print(f"  Range: [{min_range}, {max_range}] m")

    @staticmethod
    def _compute_ray_directions(
        azimuth_count: int,
        elevation_count: int,
        azimuth_min: float,
        azimuth_max: float,
        elevation_min: float,
        elevation_max: float,
    ) -> np.ndarray:
        """Compute unit ray directions in the sensor site frame, shape ``(n_rays, 3)``."""
        azimuths = azimuth_min + (azimuth_max - azimuth_min) * np.arange(azimuth_count) / azimuth_count
        if elevation_count == 1:
            elevations = np.array([0.5 * (elevation_min + elevation_max)])
        else:
            elevations = np.linspace(elevation_min, elevation_max, elevation_count)

        el, az = np.meshgrid(elevations, azimuths, indexing="ij")
        directions = np.stack(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
            axis=-1,
        )
        return directions.reshape(-1, 3).astype(np.float32)

    def _validate_sensor_sites(self, sensor_sites: list[int]):
        """Validate the sensor sites."""
        shape_flags = self.model.shape_flags.numpy()
        for site_idx in sensor_sites:
            if site_idx < 0 or site_idx >= self.model.shape_count:
                raise ValueError(f"sensor site index {site_idx} is out of range")
            if not (shape_flags[site_idx] & ShapeFlags.SITE):
                raise ValueError(f"sensor site index {site_idx} is not a site")

    def update(self, state: State):
        """Update the lidar sensor.

        Casts all rays against the model shape BVH and writes hit distances to
        :attr:`distances`. If shapes have moved in *state* since the BVH was
        last built or refit, call :meth:`~newton.Model.bvh_refit_shapes` first.

        Args:
            state: The state to update the sensor from. Reads ``body_q``.
        """
        if self.model.bvh_shapes is None:
            raise RuntimeError(
                "SensorLidar requires a shape BVH built for the queried state. "
                "ModelBuilder.finalize() builds one for the initial state; call "
                "model.bvh_build_shapes(state) for manually populated models and "
                "model.bvh_refit_shapes(state) after state changes."
            )

        wp.launch(
            compute_sensor_lidar_kernel,
            dim=(self.n_sensors, self.n_rays),
            inputs=[
                self.sensor_sites_arr,
                self.sensor_worlds_arr,
                self.ray_directions,
                self.model.shape_body,
                self.model.shape_transform,
                state.body_q,
                self.model.bvh_shapes.id,
                self.model.bvh_shapes_group_roots,
                self.model.bvh_shape_enabled,
                self.model.bvh_shape_world_transforms,
                self.model.shape_type,
                self.model.shape_scale,
                self.model.shape_source_ptr,
                self.min_range,
                self.max_range,
            ],
            outputs=[self.distances],
            device=self.model.device,
        )
