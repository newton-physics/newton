# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...geometry import GeoType, ShapeFlags
from ...sim import Model, State
from .shape_boundary_kernels import (
    SPH_COLLIDER_VELOCITY_BACKWARD,
    SPH_COLLIDER_VELOCITY_FORWARD,
)
from .shape_boundary_kernels import (
    collide_particle_shapes as _collide_particle_shapes,
)

_SUPPORTED_MODEL_COLLIDER_TYPES = frozenset(
    int(shape_type)
    for shape_type in (
        GeoType.PLANE,
        GeoType.SPHERE,
        GeoType.BOX,
        GeoType.CAPSULE,
        GeoType.CYLINDER,
        GeoType.ELLIPSOID,
        GeoType.CONE,
        GeoType.MESH,
        GeoType.CONVEX_MESH,
    )
)


def _finite_collider_value(value: object, name: str, *, nonnegative: bool) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} values must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} values must be non-negative")
    return result


def _validated_optional_collider_values(
    values: Sequence[float | None] | None,
    count: int,
    name: str,
    *,
    nonnegative: bool,
) -> tuple[float | None, ...] | None:
    if values is None:
        return None
    if len(values) != count:
        raise ValueError(f"{name} must match collider_body_ids length")
    return tuple(
        None if value is None else _finite_collider_value(value, name, nonnegative=nonnegative) for value in values
    )


def _validate_supported_model_colliders(model: Model) -> None:
    if not model.shape_count:
        return
    shape_flags = np.asarray(model.shape_flags.numpy(), dtype=np.int32)
    shape_types = np.asarray(model.shape_type.numpy(), dtype=np.int32)
    particle_colliders = (shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0
    unsupported = sorted({int(value) for value in shape_types[particle_colliders]} - _SUPPORTED_MODEL_COLLIDER_TYPES)
    if not unsupported:
        return

    names = []
    for value in unsupported:
        try:
            names.append(GeoType(value).name)
        except ValueError:
            names.append(str(value))
    raise NotImplementedError(f"SolverWCSPH does not support particle collision with: {', '.join(names)}")


@dataclass
class SPHBoundaryHandler:
    """Boundary backend for Newton collision shapes and standalone meshes."""

    model: Model
    enable_shape_boundaries: bool = True
    collider_model: Model | None = None
    explicit_collider_meshes: tuple[object, ...] = ()
    explicit_collider_margins: tuple[float, ...] = ()
    explicit_collider_friction: tuple[float, ...] = ()
    explicit_collider_projection_threshold: tuple[float, ...] = ()
    explicit_collider_body_ids: tuple[int, ...] = ()
    explicit_collider_mesh_ids_wp: wp.array[wp.uint64] | None = None
    explicit_collider_margins_wp: wp.array[float] | None = None
    explicit_collider_friction_wp: wp.array[float] | None = None
    explicit_collider_projection_threshold_wp: wp.array[float] | None = None
    explicit_collider_body_ids_wp: wp.array[wp.int32] | None = None
    model_collider_shape_margin_wp: wp.array[float] | None = None
    model_collider_shape_friction_wp: wp.array[float] | None = None
    model_collider_shape_projection_threshold_wp: wp.array[float] | None = None
    previous_collider_body_q_wp: wp.array[wp.transform] | None = None
    analytic_body_impulse_wp: wp.array[wp.vec3] | None = None
    analytic_body_angular_impulse_wp: wp.array[wp.vec3] | None = None
    _model_collider_body_ids: tuple[int, ...] | None = None
    _model_collider_margins: tuple[float | None, ...] | None = None
    _model_collider_friction: tuple[float | None, ...] | None = None
    _model_collider_projection_threshold: tuple[float | None, ...] | None = None

    def __post_init__(self) -> None:
        self._refresh_explicit_collider_mesh_arrays()
        self.set_model_collider_material_overrides(
            None,
            margins=None,
            friction=None,
            projection_threshold=None,
        )
        self._reset_analytic_body_impulses()

    def refresh_model(self) -> None:
        """Refresh caches sized from mutable Newton model topology/properties."""
        _validate_supported_model_colliders(self._collider_model())
        self.set_model_collider_material_overrides(
            self._model_collider_body_ids,
            margins=self._model_collider_margins,
            friction=self._model_collider_friction,
            projection_threshold=self._model_collider_projection_threshold,
        )
        self._refresh_explicit_collider_mesh_arrays()
        self._reset_analytic_body_impulses()

    def set_collider_model(self, model: Model | None) -> None:
        """Use ``model`` as the source of analytic collider shapes."""
        self.collider_model = self.model if model is None else model
        _validate_supported_model_colliders(self.collider_model)
        self.set_model_collider_material_overrides(
            None,
            margins=None,
            friction=None,
            projection_threshold=None,
        )
        self.save_collider_current_position(self._collider_model().body_q)
        self._reset_analytic_body_impulses()

    def require_collider_previous_position(self, collider_body_q: wp.array[wp.transform] | None) -> None:
        if collider_body_q is None:
            self.previous_collider_body_q_wp = None
        elif (
            self.previous_collider_body_q_wp is None or self.previous_collider_body_q_wp.shape != collider_body_q.shape
        ):
            self.previous_collider_body_q_wp = wp.clone(collider_body_q)

    def save_collider_current_position(self, collider_body_q: wp.array[wp.transform] | None) -> None:
        self.require_collider_previous_position(collider_body_q)
        if collider_body_q is not None and self.previous_collider_body_q_wp is not None:
            self.previous_collider_body_q_wp.assign(collider_body_q)

    def set_model_collider_material_overrides(
        self,
        collider_body_ids: Sequence[int] | None,
        *,
        margins: Sequence[float | None] | None,
        friction: Sequence[float | None] | None,
        projection_threshold: Sequence[float | None] | None,
    ) -> None:
        """Build per-shape material arrays for model-owned collider shapes."""
        body_ids_tuple = None if collider_body_ids is None else tuple(int(body) for body in collider_body_ids)
        body_ids = () if body_ids_tuple is None else body_ids_tuple
        margins_tuple = _validated_optional_collider_values(
            margins,
            len(body_ids),
            "SPH collider_margins",
            nonnegative=False,
        )
        friction_tuple = _validated_optional_collider_values(
            friction,
            len(body_ids),
            "SPH collider_friction",
            nonnegative=True,
        )
        projection_threshold_tuple = _validated_optional_collider_values(
            projection_threshold,
            len(body_ids),
            "SPH collider_projection_threshold",
            nonnegative=True,
        )

        self._model_collider_body_ids = body_ids_tuple
        self._model_collider_margins = margins_tuple
        self._model_collider_friction = friction_tuple
        self._model_collider_projection_threshold = projection_threshold_tuple

        collider_model = self._collider_model()
        shape_count = int(collider_model.shape_count)
        if shape_count == 0:
            empty = np.zeros(0, dtype=np.float32)
            self.model_collider_shape_margin_wp = wp.array(empty, dtype=wp.float32, device=self.model.device)
            self.model_collider_shape_friction_wp = wp.array(empty, dtype=wp.float32, device=self.model.device)
            self.model_collider_shape_projection_threshold_wp = wp.array(
                empty,
                dtype=wp.float32,
                device=self.model.device,
            )
            return

        shape_margin = np.asarray(collider_model.shape_margin.numpy(), dtype=np.float32).copy()
        shape_friction = np.asarray(collider_model.shape_material_mu.numpy(), dtype=np.float32).copy()
        shape_projection_threshold = np.zeros(shape_count, dtype=np.float32)
        shape_body = np.asarray(collider_model.shape_body.numpy(), dtype=np.int32)
        shape_flags = np.asarray(collider_model.shape_flags.numpy(), dtype=np.int32)

        if margins_tuple is not None:
            for body, margin in zip(body_ids, margins_tuple, strict=True):
                if margin is None:
                    continue
                mask = (shape_body == body) & ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
                shape_margin[mask] = margin

        if friction_tuple is not None:
            for body, mu in zip(body_ids, friction_tuple, strict=True):
                if mu is None:
                    continue
                mask = (shape_body == body) & ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
                shape_friction[mask] = mu

        if projection_threshold_tuple is not None:
            for body, value in zip(body_ids, projection_threshold_tuple, strict=True):
                if value is None:
                    continue
                mask = (shape_body == body) & ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
                shape_projection_threshold[mask] = value

        self.model_collider_shape_margin_wp = wp.array(shape_margin, dtype=wp.float32, device=self.model.device)
        self.model_collider_shape_friction_wp = wp.array(shape_friction, dtype=wp.float32, device=self.model.device)
        self.model_collider_shape_projection_threshold_wp = wp.array(
            shape_projection_threshold,
            dtype=wp.float32,
            device=self.model.device,
        )

    def set_explicit_collider_meshes(
        self,
        meshes: Sequence[object] | None,
        *,
        body_ids: Sequence[int] | None = None,
        margins: Sequence[float] | None = None,
        friction: Sequence[float] | None = None,
        projection_threshold: Sequence[float] | None = None,
    ) -> None:
        """Use standalone triangle meshes as SPH colliders."""
        self.explicit_collider_meshes = () if meshes is None else tuple(meshes)
        mesh_count = len(self.explicit_collider_meshes)
        self.explicit_collider_body_ids = (
            tuple(-1 for _ in range(mesh_count)) if body_ids is None else tuple(int(body) for body in body_ids)
        )
        self.explicit_collider_margins = tuple(0.0 for _ in range(mesh_count)) if margins is None else tuple(margins)
        self.explicit_collider_friction = tuple(0.0 for _ in range(mesh_count)) if friction is None else tuple(friction)
        self.explicit_collider_projection_threshold = (
            tuple(0.0 for _ in range(mesh_count)) if projection_threshold is None else tuple(projection_threshold)
        )
        if len(self.explicit_collider_body_ids) != mesh_count:
            raise ValueError("SPH explicit collider body ids must match collider_meshes length")
        if len(self.explicit_collider_margins) != mesh_count:
            raise ValueError("SPH explicit collider mesh margins must match collider_meshes length")
        if len(self.explicit_collider_friction) != mesh_count:
            raise ValueError("SPH explicit collider mesh friction values must match collider_meshes length")
        if len(self.explicit_collider_projection_threshold) != mesh_count:
            raise ValueError("SPH explicit collider mesh projection thresholds must match collider_meshes length")
        self.explicit_collider_margins = tuple(
            _finite_collider_value(value, "SPH explicit collider mesh margins", nonnegative=False)
            for value in self.explicit_collider_margins
        )
        self.explicit_collider_friction = tuple(
            _finite_collider_value(value, "SPH explicit collider mesh friction", nonnegative=True)
            for value in self.explicit_collider_friction
        )
        self.explicit_collider_projection_threshold = tuple(
            _finite_collider_value(value, "SPH explicit collider mesh projection thresholds", nonnegative=True)
            for value in self.explicit_collider_projection_threshold
        )
        body_count = int(self._collider_model().body_count)
        for body in self.explicit_collider_body_ids:
            if body < -1 or body >= body_count:
                raise ValueError("SPH explicit collider body id is out of range")
        self._refresh_explicit_collider_mesh_arrays()

    def _refresh_explicit_collider_mesh_arrays(self) -> None:
        mesh_ids: list[int] = []
        for mesh in self.explicit_collider_meshes:
            mesh_id = getattr(mesh, "id", None)
            if mesh_id is None and getattr(mesh, "mesh", None) is not None:
                mesh_id = getattr(mesh.mesh, "id", None)
            if mesh_id is None and hasattr(mesh, "finalize"):
                mesh_id = mesh.finalize(device=self.model.device)
            if mesh_id is None:
                raise TypeError("SPH explicit collider meshes must be Newton Mesh or Warp Mesh objects")
            mesh_ids.append(int(mesh_id))

        self.explicit_collider_mesh_ids_wp = wp.array(
            np.asarray(mesh_ids, dtype=np.uint64),
            dtype=wp.uint64,
            device=self.model.device,
        )
        self.explicit_collider_margins_wp = wp.array(
            np.asarray(self.explicit_collider_margins, dtype=np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.explicit_collider_friction_wp = wp.array(
            np.asarray(self.explicit_collider_friction, dtype=np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.explicit_collider_projection_threshold_wp = wp.array(
            np.asarray(self.explicit_collider_projection_threshold, dtype=np.float32),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.explicit_collider_body_ids_wp = wp.array(
            np.asarray(self.explicit_collider_body_ids, dtype=np.int32),
            dtype=wp.int32,
            device=self.model.device,
        )

    def _collider_model(self) -> Model:
        return self.model if self.collider_model is None else self.collider_model

    def explicit_collider_mesh_count(self) -> int:
        if not self.enable_shape_boundaries:
            return 0
        return len(self.explicit_collider_meshes)

    def _reset_analytic_body_impulses(self) -> None:
        body_count = int(self._collider_model().body_count)
        if self.analytic_body_impulse_wp is None or int(self.analytic_body_impulse_wp.shape[0]) != body_count:
            self.analytic_body_impulse_wp = wp.zeros(body_count, dtype=wp.vec3, device=self.model.device)
            self.analytic_body_angular_impulse_wp = wp.zeros(body_count, dtype=wp.vec3, device=self.model.device)
        else:
            self.analytic_body_impulse_wp.zero_()
            self.analytic_body_angular_impulse_wp.zero_()

    def collide_analytic_shapes(
        self,
        state: State,
        *,
        boundary_margin: float,
        boundary_friction: float,
        mesh_query_max_distance: float,
        body_com: wp.array[wp.vec3],
        body_mass: wp.array[float],
        body_inv_inertia: wp.array[wp.mat33],
        collider_velocity_mode: int = SPH_COLLIDER_VELOCITY_FORWARD,
        dt: float = 0.0,
    ) -> None:
        """Project fluid particles against supported analytic Newton shapes."""
        model = self.model
        collider_model = self._collider_model()
        if collider_velocity_mode == SPH_COLLIDER_VELOCITY_BACKWARD:
            self.require_collider_previous_position(state.body_q)
        self._reset_analytic_body_impulses()
        if (
            not self.enable_shape_boundaries
            or not model.particle_count
            or (not collider_model.shape_count and not self.explicit_collider_mesh_count())
        ):
            self.save_collider_current_position(state.body_q)
            return

        if collider_model.body_count:
            for name in ("body_q", "body_qd"):
                value = getattr(state, name, None)
                if (
                    value is None
                    or getattr(value, "device", None) != model.device
                    or int(value.shape[0]) != int(collider_model.body_count)
                ):
                    raise ValueError(f"SPH state.{name} must match the collider model on the solver device")

        body_q_prev = (
            self.previous_collider_body_q_wp
            if collider_velocity_mode == SPH_COLLIDER_VELOCITY_BACKWARD
            else state.body_q
        )

        wp.launch(
            _collide_particle_shapes,
            dim=model.particle_count,
            inputs=[
                collider_model.shape_count,
                collider_model.body_count,
                state.particle_q,
                state.particle_qd,
                model.particle_mass,
                model.particle_radius,
                model.particle_flags,
                model.particle_world,
                collider_model.shape_type,
                collider_model.shape_flags,
                collider_model.shape_world,
                collider_model.shape_body,
                collider_model.shape_transform,
                collider_model.shape_scale,
                collider_model.shape_source_ptr,
                self.model_collider_shape_margin_wp,
                self.model_collider_shape_friction_wp,
                self.model_collider_shape_projection_threshold_wp,
                self.explicit_collider_mesh_count(),
                self.explicit_collider_mesh_ids_wp,
                self.explicit_collider_margins_wp,
                self.explicit_collider_friction_wp,
                self.explicit_collider_projection_threshold_wp,
                self.explicit_collider_body_ids_wp,
                state.body_q,
                state.body_qd,
                body_q_prev,
                collider_model.body_flags,
                body_com,
                body_mass,
                body_inv_inertia,
                self.analytic_body_impulse_wp,
                self.analytic_body_angular_impulse_wp,
                boundary_margin,
                boundary_friction,
                mesh_query_max_distance,
                collider_velocity_mode,
                dt,
                state.sph.boundary_impulse,
            ],
            device=model.device,
        )
        self.save_collider_current_position(state.body_q)


def _resolve_collider_body_ids(
    collider_model: Model,
    collider_body_ids: list[int] | None,
    collider_meshes: list[wp.Mesh] | None,
) -> np.ndarray:
    if collider_meshes is not None:
        mesh_count = len(collider_meshes)
        if collider_body_ids is None:
            return np.full(mesh_count, -1, dtype=np.int32)
        body_ids = [int(body) if body is not None else -1 for body in collider_body_ids]
        if len(body_ids) != mesh_count:
            raise ValueError("SPH collider_body_ids must match collider_meshes length")
        for body in body_ids:
            if body < -1 or body >= int(collider_model.body_count):
                raise ValueError("SPH collider body id is out of range")
        return np.asarray(body_ids, dtype=np.int32)

    if collider_body_ids is None:
        body_ids = [
            body
            for body in range(-1, int(collider_model.body_count))
            if _collider_body_shape_count(collider_model, body) > 0
        ]
    else:
        body_ids = [int(body) for body in collider_body_ids]
        for body in body_ids:
            if body < -1 or body >= int(collider_model.body_count):
                raise ValueError("SPH collider body id is out of range")
            if _collider_body_shape_count(collider_model, body) == 0:
                raise ValueError(f"SPH collider body {body} has no particle-colliding shapes")
    return np.asarray(body_ids, dtype=np.int32)


def _collider_body_shape_count(collider_model: Model, body: int) -> int:
    if int(collider_model.shape_count) == 0:
        return 0
    shape_body = np.asarray(collider_model.shape_body.numpy(), dtype=np.int32)
    shape_flags = np.asarray(collider_model.shape_flags.numpy(), dtype=np.int32)
    mask = (shape_body == int(body)) & ((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
    return int(np.count_nonzero(mask))
