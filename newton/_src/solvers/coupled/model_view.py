# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ModelView: a lightweight proxy over a Model with attribute overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from ...geometry import ParticleFlags
from ...sim import BodyFlags, Model

if TYPE_CHECKING:
    from ...sim import State


_CORE_START_COUNT_ATTR_BY_NAME = {
    "joint_q_start": "joint_count",
    "joint_qd_start": "joint_count",
    "articulation_start": "articulation_count",
}


_CORE_WORLD_START_TOTAL_ATTR_BY_NAME = {
    "particle_world_start": "particle_count",
    "shape_world_start": "shape_count",
    "body_world_start": "body_count",
    "joint_world_start": "joint_count",
    "joint_dof_world_start": "joint_dof_count",
    "joint_coord_world_start": "joint_coord_count",
    "joint_constraint_world_start": "joint_constraint_count",
    "articulation_world_start": "articulation_count",
    "equality_constraint_world_start": "equality_constraint_count",
}


_UNREGISTERED_CORE_COUNT_ATTR_BY_NAME = {
    "particle_q": "particle_count",
    "particle_qd": "particle_count",
    "particle_mass": "particle_count",
    "particle_inv_mass": "particle_count",
    "particle_radius": "particle_count",
    "particle_flags": "particle_count",
    "particle_colors": "particle_count",
    "particle_world": "particle_count",
    "body_colors": "body_count",
    "shape_world": "shape_count",
    "body_world": "body_count",
    "joint_world": "joint_count",
    "articulation_world": "articulation_count",
    "equality_constraint_world": "equality_constraint_count",
    "constraint_mimic_world": "constraint_mimic_count",
}


_COLOR_GROUP_COUNT_ATTR_BY_NAME = {
    "particle_color_groups": "particle_count",
    "body_color_groups": "body_count",
}


_COUNT_ATTR_BY_FREQUENCY = {
    Model.AttributeFrequency.JOINT: "joint_count",
    Model.AttributeFrequency.JOINT_DOF: "joint_dof_count",
    Model.AttributeFrequency.JOINT_COORD: "joint_coord_count",
    Model.AttributeFrequency.JOINT_CONSTRAINT: "joint_constraint_count",
    Model.AttributeFrequency.BODY: "body_count",
    Model.AttributeFrequency.SHAPE: "shape_count",
    Model.AttributeFrequency.ARTICULATION: "articulation_count",
    Model.AttributeFrequency.EQUALITY_CONSTRAINT: "equality_constraint_count",
    Model.AttributeFrequency.PARTICLE: "particle_count",
    Model.AttributeFrequency.EDGE: "edge_count",
    Model.AttributeFrequency.TRIANGLE: "tri_count",
    Model.AttributeFrequency.TETRAHEDRON: "tet_count",
    Model.AttributeFrequency.SPRING: "spring_count",
    Model.AttributeFrequency.CONSTRAINT_MIMIC: "constraint_mimic_count",
    Model.AttributeFrequency.WORLD: "world_count",
}


def _types_compatible(current, value) -> bool:
    """Return True iff *value* is type-compatible with *current* for an override."""
    if isinstance(current, wp.array):
        return (
            isinstance(value, wp.array)
            and value.dtype == current.dtype
            and value.ndim == current.ndim
            and value.device == current.device
        )
    if isinstance(current, np.ndarray):
        return isinstance(value, np.ndarray) and value.dtype == current.dtype and value.ndim == current.ndim
    if isinstance(current, float) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    return isinstance(value, type(current))


def _type_summary(value) -> str:
    if isinstance(value, wp.array):
        return f"wp.array[dtype={value.dtype}, ndim={value.ndim}, device={value.device}]"
    if isinstance(value, np.ndarray):
        return f"numpy.ndarray[dtype={value.dtype}, ndim={value.ndim}]"
    return type(value).__name__


class ModelView:
    """A read-through view over a :class:`~newton.Model` that overrides a subset of attributes.

    Attribute access falls back to the parent Model for anything not explicitly
    overridden on this view.  This allows coupled solvers to present a per-solver
    "model" (e.g. with zeroed masses for non-owned bodies) without duplicating
    the full Model.

    A ``ModelView`` is intended to duck-type as a :class:`~newton.Model` for the
    purpose of constructing solvers (``SolverFoo(model=view)``).

    Example::

        view = ModelView(model, "vbd")
        view.body_inv_mass = zeroed_inv_mass  # override
        view.body_count  # delegates to model.body_count
        solver = SolverVBD(model=view)
    """

    def __init__(self, parent: Model, name: str) -> None:
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_overrides", {})
        object.__setattr__(self, "_cache", {})

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return self._count_limited_attribute(name, overrides[name])
        return self._count_limited_attribute(name, getattr(object.__getattribute__(self, "_parent"), name))

    def __setattr__(self, name: str, value) -> None:
        parent = object.__getattribute__(self, "_parent")
        if not hasattr(parent, name):
            raise AttributeError(
                f"ModelView {self.name!r} cannot override {name!r}: {type(parent).__name__} has no such attribute"
            )
        current = getattr(parent, name)
        if current is not None and value is not None and not _types_compatible(current, value):
            raise TypeError(
                f"ModelView {self.name!r} override for {name!r}: expected "
                f"{_type_summary(current)}, got {_type_summary(value)}"
            )
        if name.endswith("_count"):
            object.__getattribute__(self, "_cache").clear()
        object.__getattribute__(self, "_overrides")[name] = value

    def __delattr__(self, name: str) -> None:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            del overrides[name]
        else:
            raise AttributeError(f"ModelView '{self._name}' has no override '{name}'")

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Name of this view (e.g. ``"vbd"``, ``"mjc"``)."""
        return object.__getattribute__(self, "_name")

    @property
    def parent(self) -> Model:
        """The underlying :class:`~newton.Model`."""
        return object.__getattribute__(self, "_parent")

    @property
    def overrides(self) -> dict[str, object]:
        """Dictionary of attribute names that are overridden on this view."""
        return dict(object.__getattribute__(self, "_overrides"))

    def __repr__(self) -> str:
        overrides = object.__getattribute__(self, "_overrides")
        attrs = list(overrides.keys())
        return f"ModelView('{self.name}', overrides={attrs})"

    def _count_limited_attribute(self, name: str, value):
        """Return *value* sliced to the view-local frequency count when needed."""
        color_group_count_attr = _COLOR_GROUP_COUNT_ATTR_BY_NAME.get(name)
        if color_group_count_attr is not None:
            return self._count_limited_color_groups(
                name, value, color_group_count_attr, int(getattr(self, color_group_count_attr))
            )

        if not isinstance(value, (wp.array, np.ndarray)):
            return value

        start_count_attr = _CORE_START_COUNT_ATTR_BY_NAME.get(name)
        if start_count_attr is not None:
            count = int(getattr(self, start_count_attr)) + 1
            if value.shape[0] == count:
                return value
            if value.shape[0] < count:
                raise ValueError(
                    f"ModelView '{self.name}' has {name} with length {value.shape[0]}, below count {count}"
                )
            return value[:count]

        world_start_total_attr = _CORE_WORLD_START_TOTAL_ATTR_BY_NAME.get(name)
        if world_start_total_attr is not None:
            return self._count_limited_world_start(name, value, int(getattr(self, world_start_total_attr)))

        count = self._frequency_count_for_attribute(name)
        if count is None or value.shape[0] == count:
            return value
        if value.shape[0] < count:
            raise ValueError(f"ModelView '{self.name}' has {name} with length {value.shape[0]}, below count {count}")
        return value[:count]

    def _frequency_count_for_attribute(self, name: str) -> int | None:
        """Return the view-local count associated with a model attribute."""
        parent = object.__getattribute__(self, "_parent")
        frequency = parent.attribute_frequency.get(name)
        if frequency is None:
            count_attr = _UNREGISTERED_CORE_COUNT_ATTR_BY_NAME.get(name)
            return None if count_attr is None else int(getattr(self, count_attr))
        if isinstance(frequency, str):
            return self.custom_frequency_counts.get(frequency)

        count_attr = _COUNT_ATTR_BY_FREQUENCY.get(frequency)
        return None if count_attr is None else int(getattr(self, count_attr))

    def _count_limited_world_start(self, name: str, value, count: int):
        """Return a world-start array whose offsets do not exceed *count*."""
        if value.shape[0] == 0:
            return value
        host = value.numpy() if isinstance(value, wp.array) else value
        clipped = np.minimum(host, count).astype(host.dtype, copy=False)
        if np.array_equal(clipped, host):
            return value

        cache_name = f"__count_limited_{name}"
        cache = object.__getattribute__(self, "_cache")
        cached = cache.get(cache_name)
        if cached is not None:
            return cached
        if isinstance(value, wp.array):
            cached = wp.array(clipped, dtype=value.dtype, device=value.device)
        else:
            cached = np.array(clipped, dtype=value.dtype)
        cache[cache_name] = cached
        return cached

    def _count_limited_color_groups(self, name: str, value, count_attr: str, count: int):
        """Return color groups filtered to ids visible in a prefix-limited view."""
        if not isinstance(value, list):
            return value
        parent = object.__getattribute__(self, "_parent")
        if count >= int(getattr(parent, count_attr)):
            return value

        cache_name = f"__count_limited_{name}_{count}"
        cache = object.__getattribute__(self, "_cache")
        cached = cache.get(cache_name)
        if cached is not None:
            return cached

        filtered_groups = []
        changed = False
        for group in value:
            if not isinstance(group, wp.array):
                filtered_groups.append(group)
                continue
            host = group.numpy()
            filtered = host[host < count]
            changed = changed or filtered.shape[0] != host.shape[0]
            if filtered.shape[0] == host.shape[0]:
                filtered_groups.append(group)
            else:
                filtered_groups.append(wp.array(filtered, dtype=group.dtype, device=parent.device))

        if not changed:
            return value
        cache[cache_name] = filtered_groups
        return filtered_groups

    # ------------------------------------------------------------------
    # State creation - reuses Model.state() through this view
    # ------------------------------------------------------------------

    def state(self, requires_grad: bool | None = None) -> State:
        """Create a :class:`~newton.State` using view-local state overrides.

        This calls the normal :meth:`~newton.Model.state` implementation with
        this view as the model object, so allocation follows the attributes and
        counts visible through the view.
        """
        from ...sim import Model  # noqa: PLC0415

        with self._temporary_state_array_overrides():
            return Model.state(self, requires_grad=requires_grad)

    def get_requested_state_attributes(self) -> list[str]:
        """Return requested state attributes using view-local counts."""
        from ...sim import Model  # noqa: PLC0415

        return Model.get_requested_state_attributes(self)

    def _temporary_state_array_overrides(self):
        """Temporarily slice state source arrays to match view-local counts."""
        return _TemporaryStateArrayOverrides(self)

    def _state_source_arrays(self):
        """Yield core model arrays consumed by :meth:`newton.Model.state`."""
        yield "particle_q", self.particle_count
        yield "particle_qd", self.particle_count
        yield "body_q", self.body_count
        yield "body_qd", self.body_count
        yield "joint_q", self.joint_coord_count
        yield "joint_qd", self.joint_dof_count

    def _add_custom_attributes(
        self,
        destination: object,
        assignment,
        requires_grad: bool = False,
        clone_arrays: bool = True,
    ) -> None:
        """Delegate custom attribute creation with this view as the model."""
        from ...sim import Model  # noqa: PLC0415

        Model._add_custom_attributes(self, destination, assignment, requires_grad, clone_arrays)

    # ------------------------------------------------------------------
    # Helpers for common overrides
    # ------------------------------------------------------------------

    def _cow_array(self, name: str) -> wp.array[Any]:
        """Return a view-local mutable copy of a parent model array.

        ``ModelView`` uses copy-on-write overlay semantics for explicit
        mutator methods. Reads fall through to the parent model; the first
        mutator for an array clones the parent array into this view's
        overrides. Direct writes through a returned Warp array are not
        intercepted.
        """
        parent = object.__getattribute__(self, "_parent")
        overrides = object.__getattribute__(self, "_overrides")
        array = overrides.get(name)
        if array is None:
            array = wp.clone(getattr(parent, name))
            overrides[name] = array
        return array

    def disable_body_dynamics(self, body_indices: wp.array[int]) -> None:
        """Disable dynamics for the given body indices in this view.

        Creates overridden copies of ``body_inv_mass`` and
        ``body_inv_inertia``. Inverse inertial properties are zeroed, while
        forward mass, inertia, and body flags are left intact as metadata for
        solver conversion.

        Args:
            body_indices: 1-D int array of body indices to immobilize.
        """
        parent = object.__getattribute__(self, "_parent")
        if body_indices.shape[0] == 0:
            return

        inv_mass = self._cow_array("body_inv_mass")
        inv_inertia = self._cow_array("body_inv_inertia")

        wp.launch(
            _zero_body_inverse_dynamics_kernel,
            dim=body_indices.shape[0],
            inputs=[
                body_indices,
                inv_mass,
                inv_inertia,
            ],
            device=parent.device,
        )

    def scale_body_mass(self, body_indices: wp.array[int], factor: float) -> None:
        """Scale mass and inertia for the given body indices.

        Multiplying mass by *factor* means dividing ``body_inv_mass`` and
        ``body_inv_inertia`` by *factor*, and multiplying ``body_mass`` and
        ``body_inertia`` by *factor*.

        If overrides for these arrays already exist on this view they are
        modified in-place; otherwise fresh clones are created.

        Args:
            body_indices: 1-D int array of body indices to scale.
            factor: Multiplicative scale applied to mass / inertia.
                Values < 1 make proxy bodies lighter (softer coupling);
                values > 1 make them heavier.
        """
        if factor <= 0.0:
            raise ValueError(f"Body mass scale factor must be > 0, got {factor}")

        parent = object.__getattribute__(self, "_parent")
        if body_indices.shape[0] == 0:
            return

        inv_mass = self._cow_array("body_inv_mass")
        inv_inertia = self._cow_array("body_inv_inertia")
        mass = self._cow_array("body_mass")
        inertia = self._cow_array("body_inertia")

        wp.launch(
            _scale_body_mass_kernel,
            dim=body_indices.shape[0],
            inputs=[body_indices, float(factor), inv_mass, inv_inertia, mass, inertia],
            device=parent.device,
        )

    def set_body_mass(self, body_indices: wp.array[int], body_mass: wp.array[float]) -> None:
        """Set mass for the given body indices and scale inertia consistently.

        Existing body inertia tensors are scaled by the ratio between the new
        and current mass. This preserves the inertia shape while allowing
        couplers to install scalar effective masses. Bodies with non-positive
        current mass cannot be assigned a positive scalar mass this way because
        there is no finite inertia tensor to scale; use
        :meth:`set_body_inertial_properties` for that transition.

        Args:
            body_indices: Body ids whose mass should be replaced.
            body_mass: Replacement body masses [kg], indexed like
                ``body_indices``.
        """
        parent = object.__getattribute__(self, "_parent")
        if body_indices.shape[0] == 0:
            return

        inv_mass = self._cow_array("body_inv_mass")
        inv_inertia = self._cow_array("body_inv_inertia")
        mass = self._cow_array("body_mass")
        inertia = self._cow_array("body_inertia")

        invalid = wp.zeros(1, dtype=wp.int32, device=parent.device)
        wp.launch(
            _check_body_mass_update_kernel,
            dim=body_indices.shape[0],
            inputs=[body_indices, body_mass, mass, invalid],
            device=parent.device,
        )
        if int(invalid.numpy()[0]) != 0:
            raise ValueError(
                "Cannot assign a positive scalar body mass to a body with non-positive current mass; "
                "use set_body_inertial_properties() to provide a finite inertia tensor."
            )

        wp.launch(
            _set_body_mass_kernel,
            dim=body_indices.shape[0],
            inputs=[body_indices, body_mass, inv_mass, inv_inertia, mass, inertia],
            device=parent.device,
        )

    def set_body_inertial_properties(
        self,
        body_indices: wp.array[int],
        body_mass: wp.array[float],
        body_inertia: wp.array[wp.mat33],
    ) -> None:
        """Set mass and full inertia tensors for the given body indices.

        This replaces both the scalar mass and local body-frame inertia tensor
        for each selected body, updating inverse mass and inverse inertia
        consistently.

        Args:
            body_indices: Body ids whose inertial properties should be
                replaced.
            body_mass: Replacement body masses [kg], indexed like
                ``body_indices``.
            body_inertia: Replacement body inertia tensors [kg*m^2], indexed
                like ``body_indices``.
        """
        parent = object.__getattribute__(self, "_parent")
        if body_indices.shape[0] == 0:
            return

        inv_mass = self._cow_array("body_inv_mass")
        inv_inertia = self._cow_array("body_inv_inertia")
        mass = self._cow_array("body_mass")
        inertia = self._cow_array("body_inertia")

        wp.launch(
            _set_body_inertial_properties_kernel,
            dim=body_indices.shape[0],
            inputs=[body_indices, body_mass, body_inertia, inv_mass, inv_inertia, mass, inertia],
            device=parent.device,
        )

    def mark_proxy_bodies(self, body_indices: wp.array[int]) -> None:
        """Mark the given body indices as proxy bodies in this view.

        Creates a view-local copy of ``body_flags`` on first write and ORs the
        :attr:`~newton.BodyFlags.PROXY` bit into the selected bodies. The parent
        model is never mutated.

        Args:
            body_indices: 1-D int array of body indices to mark as proxies.
        """
        parent = object.__getattribute__(self, "_parent")
        if body_indices.shape[0] == 0:
            return

        body_flags = self._cow_array("body_flags")
        wp.launch(
            _mark_body_flag_kernel,
            dim=body_indices.shape[0],
            inputs=[body_indices, int(BodyFlags.PROXY), body_flags],
            device=parent.device,
        )

    def mark_proxy_particles(self, particle_indices: wp.array[int]) -> None:
        """Mark the given particle indices as proxy particles in this view.

        Creates a view-local copy of ``particle_flags`` on first write and ORs
        the :attr:`~newton.ParticleFlags.PROXY` bit into the selected particles.
        The parent model is never mutated.

        Args:
            particle_indices: 1-D int array of particle indices to mark.
        """
        parent = object.__getattribute__(self, "_parent")
        if particle_indices.shape[0] == 0:
            return

        particle_flags = self._cow_array("particle_flags")
        wp.launch(
            _mark_particle_flag_kernel,
            dim=particle_indices.shape[0],
            inputs=[particle_indices, int(ParticleFlags.PROXY), particle_flags],
            device=parent.device,
        )

    def deactivate_particles(self, particle_indices: wp.array[int]) -> None:
        """Clear the active flag for the given particle indices in this view.

        Creates a view-local copy of ``particle_flags`` on first write. The
        parent model is never mutated.

        Args:
            particle_indices: 1-D int array of particle indices to deactivate.
        """
        parent = object.__getattribute__(self, "_parent")
        if parent.particle_count == 0 or particle_indices.shape[0] == 0:
            return

        particle_flags = self._cow_array("particle_flags")
        wp.launch(
            _clear_particle_flag_kernel,
            dim=particle_indices.shape[0],
            inputs=[particle_indices, int(ParticleFlags.ACTIVE), particle_flags],
            device=parent.device,
        )

    def disable_joints(self, joint_indices: wp.array[int]) -> None:
        """Disable the given joints in this view.

        Creates a view-local copy of ``joint_enabled`` on first write. The
        parent model is never mutated.

        Args:
            joint_indices: 1-D int array of joint indices to disable.
        """
        parent = object.__getattribute__(self, "_parent")
        if parent.joint_count == 0 or joint_indices.shape[0] == 0:
            return

        joint_enabled = self._cow_array("joint_enabled")
        wp.launch(
            _disable_joints_kernel,
            dim=joint_indices.shape[0],
            inputs=[joint_indices, joint_enabled],
            device=parent.device,
        )

    def zero_particle_mass(self, particle_indices: wp.array[int]) -> None:
        """Zero mass and inverse mass for the given particle indices.

        Creates view-local copies of ``particle_mass`` and
        ``particle_inv_mass`` on first write and sets the selected particles to
        zero mass. The parent model is never mutated.

        Args:
            particle_indices: 1-D int array of particle indices to zero.
        """
        parent = object.__getattribute__(self, "_parent")
        if parent.particle_count == 0 or particle_indices.shape[0] == 0:
            return

        inv_mass = self._cow_array("particle_inv_mass")
        mass = self._cow_array("particle_mass")
        wp.launch(
            _zero_particle_mass_kernel,
            dim=particle_indices.shape[0],
            inputs=[particle_indices, inv_mass, mass],
            device=parent.device,
        )

    def scale_particle_mass(self, factor: float, particle_indices: wp.array[int] | None = None) -> None:
        """Scale mass for particles on this view by ``factor``.

        Multiplying mass by ``factor`` means dividing ``particle_inv_mass`` by
        ``factor`` and multiplying ``particle_mass`` by ``factor``. Used by
        the ADMM coupler to inject the proximal term as a mass rescaling.

        If overrides for these arrays already exist on this view they are
        modified in-place; otherwise fresh clones are created. The parent
        model is never mutated.

        Args:
            factor: Multiplicative scale applied to particle masses.
            particle_indices: Optional 1-D int array of particle indices to
                scale. When omitted, all particles are scaled.
        """
        if factor <= 0.0:
            raise ValueError(f"Particle mass scale factor must be > 0, got {factor}")

        parent = object.__getattribute__(self, "_parent")

        if parent.particle_count == 0:
            return

        inv_mass = self._cow_array("particle_inv_mass")
        mass = self._cow_array("particle_mass")

        if particle_indices is None:
            wp.launch(
                _scale_particle_mass_kernel,
                dim=parent.particle_count,
                inputs=[float(factor), inv_mass, mass],
                device=parent.device,
            )
        elif particle_indices.shape[0] > 0:
            wp.launch(
                _scale_particle_mass_indices_kernel,
                dim=particle_indices.shape[0],
                inputs=[particle_indices, float(factor), inv_mass, mass],
                device=parent.device,
            )

    def set_particle_mass(self, particle_indices: wp.array[int], particle_mass: wp.array[float]) -> None:
        """Set mass for the given particle indices.

        Args:
            particle_indices: Particle ids whose mass should be replaced.
            particle_mass: Replacement particle masses [kg], indexed like
                ``particle_indices``.
        """
        parent = object.__getattribute__(self, "_parent")
        if parent.particle_count == 0 or particle_indices.shape[0] == 0:
            return

        inv_mass = self._cow_array("particle_inv_mass")
        mass = self._cow_array("particle_mass")
        wp.launch(
            _set_particle_mass_kernel,
            dim=particle_indices.shape[0],
            inputs=[particle_indices, particle_mass, inv_mass, mass],
            device=parent.device,
        )


class _TemporaryStateArrayOverrides:
    """Context manager that exposes view-sized arrays during State creation."""

    _missing = object()

    def __init__(self, view: ModelView) -> None:
        self.view = view
        self.saved: dict[str, object] = {}

    def __enter__(self):
        overrides = object.__getattribute__(self.view, "_overrides")
        for name, count in self.view._state_source_arrays():
            source = getattr(self.view, name, None)
            if source is None:
                continue
            if source.shape[0] < count:
                raise ValueError(
                    f"ModelView '{self.view.name}' has {name} with length {source.shape[0]}, below count {count}"
                )
            if source.shape[0] != count:
                self.saved[name] = overrides.get(name, self._missing)
                overrides[name] = source[:count]
        return self.view

    def __exit__(self, exc_type, exc, tb):
        overrides = object.__getattribute__(self.view, "_overrides")
        for name, value in self.saved.items():
            if value is self._missing:
                del overrides[name]
            else:
                overrides[name] = value
        return False


@wp.kernel(enable_backward=False)
def _zero_body_inverse_dynamics_kernel(
    indices: wp.array[int],
    inv_mass: wp.array[float],
    inv_inertia: wp.array[wp.mat33],
):
    i = wp.tid()
    idx = indices[i]
    inv_mass[idx] = 0.0
    inv_inertia[idx] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def _zero_particle_mass_kernel(
    indices: wp.array[int],
    inv_mass: wp.array[float],
    mass: wp.array[float],
):
    i = wp.tid()
    idx = indices[i]
    inv_mass[idx] = 0.0
    mass[idx] = 0.0


@wp.kernel(enable_backward=False)
def _mark_body_flag_kernel(
    indices: wp.array[int],
    flag: int,
    body_flags: wp.array[wp.int32],
):
    i = wp.tid()
    idx = indices[i]
    body_flags[idx] = body_flags[idx] | flag


@wp.kernel(enable_backward=False)
def _mark_particle_flag_kernel(
    indices: wp.array[int],
    flag: int,
    particle_flags: wp.array[wp.int32],
):
    i = wp.tid()
    idx = indices[i]
    particle_flags[idx] = particle_flags[idx] | flag


@wp.kernel(enable_backward=False)
def _clear_particle_flag_kernel(
    indices: wp.array[int],
    flag: int,
    particle_flags: wp.array[wp.int32],
):
    i = wp.tid()
    idx = indices[i]
    particle_flags[idx] = particle_flags[idx] & (~flag)


@wp.kernel(enable_backward=False)
def _disable_joints_kernel(
    indices: wp.array[int],
    joint_enabled: wp.array[bool],
):
    i = wp.tid()
    idx = indices[i]
    joint_enabled[idx] = False


@wp.kernel(enable_backward=False)
def _scale_body_mass_kernel(
    indices: wp.array[int],
    factor: float,
    inv_mass: wp.array[float],
    inv_inertia: wp.array[wp.mat33],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
):
    i = wp.tid()
    idx = indices[i]
    inv_factor = 1.0 / factor
    inv_mass[idx] = inv_mass[idx] * inv_factor
    inv_inertia[idx] = inv_inertia[idx] * inv_factor
    mass[idx] = mass[idx] * factor
    inertia[idx] = inertia[idx] * factor


@wp.kernel(enable_backward=False)
def _check_body_mass_update_kernel(
    indices: wp.array[int],
    target_mass: wp.array[float],
    mass: wp.array[float],
    invalid: wp.array[wp.int32],
):
    i = wp.tid()
    idx = indices[i]
    if target_mass[i] > 0.0 and mass[idx] <= 0.0:
        invalid[0] = 1


@wp.kernel(enable_backward=False)
def _set_body_mass_kernel(
    indices: wp.array[int],
    target_mass: wp.array[float],
    inv_mass: wp.array[float],
    inv_inertia: wp.array[wp.mat33],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
):
    i = wp.tid()
    idx = indices[i]
    new_mass = target_mass[i]
    old_mass = mass[idx]

    if new_mass > 0.0:
        inv_mass[idx] = 1.0 / new_mass
        if old_mass > 0.0:
            factor = new_mass / old_mass
            inertia[idx] = inertia[idx] * factor
            inv_inertia[idx] = inv_inertia[idx] * (1.0 / factor)
    else:
        inv_mass[idx] = 0.0
        inv_inertia[idx] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mass[idx] = new_mass


@wp.kernel(enable_backward=False)
def _set_body_inertial_properties_kernel(
    indices: wp.array[int],
    target_mass: wp.array[float],
    target_inertia: wp.array[wp.mat33],
    inv_mass: wp.array[float],
    inv_inertia: wp.array[wp.mat33],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
):
    i = wp.tid()
    idx = indices[i]
    new_mass = target_mass[i]
    new_inertia = target_inertia[i]

    if new_mass > 0.0:
        inv_mass[idx] = 1.0 / new_mass
        inv_inertia[idx] = wp.inverse(new_inertia)
    else:
        inv_mass[idx] = 0.0
        inv_inertia[idx] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mass[idx] = new_mass
    inertia[idx] = new_inertia


@wp.kernel(enable_backward=False)
def _scale_particle_mass_kernel(
    factor: float,
    inv_mass: wp.array[float],
    mass: wp.array[float],
):
    i = wp.tid()
    inv_factor = 1.0 / factor
    inv_mass[i] = inv_mass[i] * inv_factor
    mass[i] = mass[i] * factor


@wp.kernel(enable_backward=False)
def _scale_particle_mass_indices_kernel(
    indices: wp.array[int],
    factor: float,
    inv_mass: wp.array[float],
    mass: wp.array[float],
):
    i = wp.tid()
    idx = indices[i]
    inv_factor = 1.0 / factor
    inv_mass[idx] = inv_mass[idx] * inv_factor
    mass[idx] = mass[idx] * factor


@wp.kernel(enable_backward=False)
def _set_particle_mass_kernel(
    indices: wp.array[int],
    target_mass: wp.array[float],
    inv_mass: wp.array[float],
    mass: wp.array[float],
):
    i = wp.tid()
    idx = indices[i]
    new_mass = target_mass[i]
    mass[idx] = new_mass
    if new_mass > 0.0:
        inv_mass[idx] = 1.0 / new_mass
    else:
        inv_mass[idx] = 0.0
