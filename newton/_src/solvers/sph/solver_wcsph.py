# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoothed Particle Hydrodynamics solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, ModelFlags, State
from ..solver import SolverBase
from .basic_kernels import (
    compute_acceleration_basic,
    integrate_sph_particles,
)
from .basic_kernels import (
    compute_density_pressure as _compute_density_pressure,
)
from .boundaries import (
    SPH_COLLIDER_VELOCITY_BACKWARD,
    SPH_COLLIDER_VELOCITY_FORWARD,
)
from .kernels import sph_kernel_names
from .sph_model import (
    SPHModel,
    register_sph_custom_attributes,
    validate_sph_custom_attributes,
)

__all__ = ["SolverWCSPH"]

_SPH_INTEGRATION_CARRY_FIELDS = (
    "density",
    "pressure",
    "volume",
)


def _config_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@wp.kernel
def _gather_collider_impulses(
    collider_body_index: wp.array[wp.int32],
    collider_ids: wp.array[wp.int32],
    body_impulses: wp.array[wp.vec3],
    body_angular_impulses: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    impulses: wp.array[wp.vec3],
    impulse_positions: wp.array[wp.vec3],
    impulse_ids: wp.array[wp.int32],
):
    collider = wp.tid()
    body = collider_body_index[collider]
    linear_impulse = body_impulses[body]
    angular_impulse = body_angular_impulses[body]
    center = wp.transform_point(body_q[body], body_com[body])

    moment_arm = wp.vec3(0.0)
    couple_impulse = wp.vec3(0.0)
    angular_norm = wp.length(angular_impulse)
    if angular_norm > 1.0e-12:
        reference = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(angular_impulse[0]) > wp.abs(angular_impulse[1]):
            reference = wp.vec3(0.0, 1.0, 0.0)
        moment_arm = wp.normalize(wp.cross(angular_impulse, reference))
        linear_norm = wp.length(linear_impulse)
        lever_length = wp.clamp(angular_norm / wp.max(linear_norm, 1.0e-12), 1.0e-3, 1.0)
        moment_arm *= lever_length
        couple_impulse = wp.cross(angular_impulse, moment_arm) / (2.0 * lever_length * lever_length)

    output = 2 * collider
    impulses[output] = 0.5 * linear_impulse + couple_impulse
    impulses[output + 1] = 0.5 * linear_impulse - couple_impulse
    impulse_positions[output] = center + moment_arm
    impulse_positions[output + 1] = center - moment_arm
    impulse_ids[output] = collider_ids[collider]
    impulse_ids[output + 1] = collider_ids[collider]


class SolverWCSPH(SolverBase):
    """Weakly compressible Smoothed Particle Hydrodynamics solver.

    Implements explicit WCSPH [1] with density summation, a Tait-style
    equation of state, symmetric pressure forces, viscosity, optional XSPH
    velocity smoothing, and Newton shape collision. The solver advances every
    active dynamic particle in its model; use a separate model when coupling
    SPH fluid to another particle solver.

    The time step is supplied by the caller and is not adapted automatically.
    It must satisfy the acoustic and force-based stability limits of the chosen
    particle spacing, support radius, and sound speed. A conservative acoustic
    starting point is ``dt <= 0.25 * h / c``, where ``h`` is the smoothing
    length and ``c`` is the artificial sound speed.

    Solid boundaries are enforced by post-integration shape projection. Reaction
    impulses for explicit partitioned coupling to a rigid-body solver are exposed
    through :meth:`collect_collider_impulses`.

    Call :meth:`register_custom_attributes` on your :class:`~newton.ModelBuilder`
    before building the model to enable SPH-specific per-particle material and
    state fields (for example ``sph:rest_density`` and ``sph:smoothing_length``).

    [1] https://doi.org/10.2312/SCA/SCA07/209-218

    Args:
        model: The Newton model to simulate.
        config: Optional solver configuration. Defaults to :class:`SolverWCSPH.Config`.
    """

    @dataclass
    class Config:
        """Configuration for :class:`SolverWCSPH`.

        Per-particle properties can be configured using custom attributes on the model.
        See :meth:`SolverWCSPH.register_custom_attributes` for details.
        """

        # kernel / equation of state
        kernel: Literal["poly6", "cubic", "wendland", "spiky"] = "poly6"
        """Smoothing kernel family."""
        smoothing_length: float | None = None
        """Default support radius [m] when the per-particle value is zero."""
        rest_density: float | None = None
        """Global rest-density override [kg/m^3]. ``None`` uses per-particle values."""
        sound_speed: float | None = None
        """Global artificial sound-speed override [m/s]. ``None`` uses per-particle values."""
        stiffness: float | None = None
        """Global pressure-stiffness override. ``None`` uses per-particle values."""
        pressure_exponent: float | None = None
        """Global Tait exponent override. ``None`` uses per-particle values."""

        # viscosity
        viscosity: float | None = None
        """Global dynamic-viscosity override [Pa s]. ``None`` uses per-particle values."""
        xsph: float = 0.0
        """XSPH velocity smoothing coefficient."""

        # boundaries / coupling
        enable_shape_boundaries: bool = True
        """Project fluid particles out of Newton collision shapes."""
        boundary_margin: float = 0.0
        """Additional particle-boundary separation distance [m]."""
        boundary_friction: float = 0.0
        """Boundary Coulomb friction coefficient."""
        collider_velocity_mode: Literal["forward", "backward"] = "forward"
        """Collider velocity mode. ``'forward'`` uses ``State.body_qd``;
        ``'backward'`` uses the previous body transform.
        """

        def validate(self) -> None:
            """Validate configuration values."""
            if self.kernel not in sph_kernel_names():
                available = ", ".join(sph_kernel_names())
                raise ValueError(f"Unsupported SPH kernel '{self.kernel}'. Available kernels: {available}")
            if not isinstance(self.enable_shape_boundaries, bool):
                raise ValueError("enable_shape_boundaries must be a boolean")

            for name in ("smoothing_length", "rest_density", "pressure_exponent"):
                value = getattr(self, name)
                if value is not None and _config_scalar(value, name) <= 0.0:
                    raise ValueError(f"{name} must be positive")
            for name in ("sound_speed", "stiffness", "viscosity", "xsph", "boundary_margin", "boundary_friction"):
                value = getattr(self, name)
                if value is not None and _config_scalar(value, name) < 0.0:
                    raise ValueError(f"{name} must be non-negative")
            if self.collider_velocity_mode not in ("forward", "backward"):
                raise ValueError(f"Invalid collider velocity mode: {self.collider_velocity_mode}")

    @classmethod
    @override
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register WCSPH-specific custom attributes in the ``sph`` namespace.

        Attributes registered on Model (per-particle):
            - ``sph:rest_density``: Reference density [kg/m^3]
            - ``sph:sound_speed``: Artificial sound speed [m/s]
            - ``sph:stiffness``: Alternative pressure stiffness [Pa]
            - ``sph:pressure_exponent``: Tait equation exponent
            - ``sph:pressure_min``: Minimum pressure clamp [Pa]
            - ``sph:pressure_max``: Maximum pressure clamp [Pa]
            - ``sph:viscosity``: Dynamic viscosity coefficient
            - ``sph:smoothing_length``: Kernel support radius [m]

        Attributes registered on State (per-particle):
            - ``sph:density``: Current density [kg/m^3]
            - ``sph:pressure``: Current pressure [Pa]
            - ``sph:volume``: Current particle volume [m^3]
            - ``sph:boundary_impulse``: Collider impulse applied to the particle [N s]
        """
        register_sph_custom_attributes(builder)

    def __init__(
        self,
        model: Model,
        config: Config | None = None,
    ):
        super().__init__(model)

        config = type(self).Config() if config is None else config
        config.validate()
        self.config = config
        self.collider_velocity_mode = config.collider_velocity_mode
        self.collider_velocity_mode_id = (
            SPH_COLLIDER_VELOCITY_BACKWARD
            if self.collider_velocity_mode == "backward"
            else SPH_COLLIDER_VELOCITY_FORWARD
        )
        validate_sph_custom_attributes(model)
        self._sph_model = SPHModel(model, config)

    def setup_collider(
        self,
        collider_meshes: list[wp.Mesh] | None = None,
        collider_body_ids: list[int] | None = None,
        collider_margins: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        model: Model | None = None,
        body_com: wp.array[wp.vec3] | None = None,
        body_mass: wp.array[float] | None = None,
        body_inv_inertia: wp.array[wp.mat33] | None = None,
        body_q: wp.array[wp.transform] | None = None,
    ) -> None:
        """Configure collider geometry and material properties.

        By default, collisions are set up against all shapes in the model with
        ``ShapeFlags.COLLIDE_PARTICLES``. Use this method to customize collider
        sources and materials, or to read colliders from a different model. When
        using a different model, the states passed to :meth:`step` must expose
        that model's ``body_q`` and ``body_qd`` arrays, as shown by the SPH
        two-way coupling example. Without ``collider_meshes``,
        ``collider_body_ids`` selects model bodies and all of their
        particle-colliding shapes. With ``collider_meshes``, model shapes are
        disabled and each body id instead supplies the transform for the mesh at
        the same index; use ``-1`` for a static world-space mesh.

        Args:
            collider_meshes: Warp triangular meshes used as colliders.
            collider_body_ids: Selected model body ids, or per-mesh body ids
                when ``collider_meshes`` is provided.
            collider_margins: Per-collider signed distance offsets [m].
            collider_friction: Per-collider Coulomb friction coefficients.
            collider_projection_threshold: Per-collider projection thresholds [m].
            model: The model to read collider properties from. Defaults to the solver model.
            body_com: For dynamic colliders, per-body center of mass on the solver device.
            body_mass: For dynamic colliders, per-body mass on the solver device. Pass zeros for kinematic bodies.
            body_inv_inertia: For dynamic colliders, per-body inverse inertia on the solver device.
            body_q: For dynamic colliders, per-body initial transform on the solver device.
        """
        self._sph_model.setup_collider(
            collider_meshes=collider_meshes,
            collider_body_ids=collider_body_ids,
            collider_margins=collider_margins,
            collider_friction=collider_friction,
            collider_projection_threshold=collider_projection_threshold,
            model=model,
            body_com=body_com,
            body_mass=body_mass,
            body_inv_inertia=body_inv_inertia,
            body_q=body_q,
        )

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the SPH simulation by one time step.

        Computes density, pressure, and explicit forces, integrates particle
        positions and velocities, then projects particles out of colliders.

        Args:
            state_in: Input state at the start of the step.
            state_out: Output state written with updated particle data.
                May be the same object as ``state_in`` for in-place stepping.
            control: Control input (unused; SPH properties come from the model).
            contacts: Contact information (unused; SPH boundary handling is internal).
            dt: Time step duration [s].
        """
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")

        model = self.model
        if not model.particle_count:
            return

        self._propagate_body_coupling_state(state_in, state_out)
        self._step_impl(state_in, state_out, dt, control)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh SPH runtime caches after mutable Newton model updates."""
        refresh_flags = (
            ModelFlags.MODEL_PROPERTIES
            | ModelFlags.BODY_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
        )
        if flags & refresh_flags:
            self._refresh_model_dependent_state()

    def collect_collider_impulses(
        self, state: State
    ) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.int32]]:
        """Collect boundary impulses applied by SPH fluid to colliders.

        Returns a tuple of 3 arrays:
            - Impulse values in world units.
            - Collider positions in world units.
            - Collider id, that can be mapped back to the model's body ids using
              the ``collider_body_index`` property.
        """
        sph_model = self._sph_model
        handler = sph_model.boundary_handler
        body_q = state.body_q if state.body_q is not None else sph_model.collider_body_q
        if sph_model.collider_impulse_body_index.shape[0] > 0:
            if (
                body_q is None
                or sph_model.collider_body_com is None
                or handler.analytic_body_impulse_wp is None
                or handler.analytic_body_angular_impulse_wp is None
            ):
                raise RuntimeError("SPH collider impulse buffers are not initialized")
            wp.launch(
                _gather_collider_impulses,
                dim=sph_model.collider_impulse_body_index.shape[0],
                inputs=[
                    sph_model.collider_impulse_body_index,
                    sph_model._dynamic_collider_ids,
                    handler.analytic_body_impulse_wp,
                    handler.analytic_body_angular_impulse_wp,
                    sph_model.collider_body_com,
                    body_q,
                ],
                outputs=[
                    sph_model.collider_impulse,
                    sph_model.collider_impulse_position,
                    sph_model.collider_impulse_id,
                ],
                device=self.model.device,
            )
        return sph_model.collider_impulse, sph_model.collider_impulse_position, sph_model.collider_impulse_id

    @property
    def collider_body_index(self) -> wp.array[wp.int32]:
        """Array mapping SPH collider ids to Newton body ids.

        Value ``-1`` denotes static analytic colliders, matching the convention
        used by :class:`SolverImplicitMPM`.
        """
        return self._sph_model.collider_body_index

    def project_outside(self, state_in: State, state_out: State, dt: float, gap: float | None = None) -> None:
        """Project particles outside of SPH colliders and adjust velocities.

        This mirrors :meth:`SolverImplicitMPM.project_outside` for callers that
        need an explicit collider projection pass.

        Args:
            state_in: The input state.
            state_out: The output state. Only particle_q, particle_qd, body_q,
                body_qd, and SPH boundary impulse buffers are written.
            dt: The time step, for estimating collider motion.
            gap: Maximum distance [m] for triangle-mesh closest-point queries.
                ``None`` uses the maximum SPH support radius.
        """
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")
        if gap is not None and (gap <= 0.0 or not np.isfinite(gap)):
            raise ValueError("gap must be finite and positive")
        if state_in.particle_q.device != self.model.device or state_out.particle_q.device != self.model.device:
            raise ValueError("SPH projection states must be allocated on the solver device.")

        state_out.particle_q.assign(state_in.particle_q)
        state_out.particle_qd.assign(state_in.particle_qd)
        if state_in.body_q is not None and state_out.body_q is not None:
            state_out.body_q.assign(state_in.body_q)
        if state_in.body_qd is not None and state_out.body_qd is not None:
            state_out.body_qd.assign(state_in.body_qd)
        if hasattr(state_out, "sph"):
            state_out.sph.boundary_impulse.zero_()

        self._collide_shape_boundaries(state_out, dt, gap)

    def _compute_density_pressure(self, state: State) -> None:
        """Compute SPH density and pressure fields for ``state``."""
        model = self.model
        if not model.particle_count:
            return

        self._build_neighbor_grid(state)

        wp.launch(
            _compute_density_pressure,
            dim=model.particle_count,
            inputs=[
                self._sph_model.neighbor_search.grid_id,
                state.particle_q,
                model.particle_mass,
                model.particle_flags,
                model.particle_world,
                model.sph.rest_density,
                model.sph.sound_speed,
                model.sph.stiffness,
                model.sph.pressure_exponent,
                model.sph.pressure_min,
                model.sph.pressure_max,
                model.sph.smoothing_length,
                -1.0 if self.config.rest_density is None else self.config.rest_density,
                -1.0 if self.config.sound_speed is None else self.config.sound_speed,
                -1.0 if self.config.stiffness is None else self.config.stiffness,
                -1.0 if self.config.pressure_exponent is None else self.config.pressure_exponent,
                self._sph_model.default_support_radius,
                self._sph_model.max_support_radius,
                self._sph_model.kernel_id,
            ],
            outputs=[
                state.sph.density,
                state.sph.pressure,
                state.sph.volume,
            ],
            device=model.device,
        )

    def _stage_integrate(self, state_in: State, state_out: State, dt: float) -> None:
        model = self.model
        scratch = self._sph_model.scratch

        max_velocity = model.particle_max_velocity

        wp.launch(
            integrate_sph_particles,
            dim=model.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                model.particle_inv_mass,
                model.particle_flags,
                scratch.acceleration,
                scratch.velocity_delta,
                self.config.xsph,
                dt,
                max_velocity,
            ],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=model.device,
        )

        state_out.sph.boundary_impulse.zero_()
        self._collide_shape_boundaries(state_out, dt)

        for name in _SPH_INTEGRATION_CARRY_FIELDS:
            getattr(state_out.sph, name).assign(getattr(state_in.sph, name))

    def _compute_explicit_acceleration_fields(self, state: State) -> None:
        model = self.model
        scratch = self._sph_model.scratch
        wp.launch(
            compute_acceleration_basic,
            dim=model.particle_count,
            inputs=[
                self._sph_model.neighbor_search.grid_id,
                state.particle_q,
                state.particle_qd,
                state.particle_f,
                model.particle_mass,
                model.particle_inv_mass,
                model.particle_flags,
                model.particle_world,
                model.gravity,
                model.sph.viscosity,
                model.sph.smoothing_length,
                state.sph.density,
                state.sph.pressure,
                -1.0 if self.config.viscosity is None else self.config.viscosity,
                self._sph_model.default_support_radius,
                self._sph_model.max_support_radius,
                self._sph_model.kernel_id,
                self.config.xsph > 0.0,
            ],
            outputs=[scratch.acceleration, scratch.velocity_delta],
            device=model.device,
        )

    def _stage_explicit_forces(self, state: State) -> None:
        self._compute_explicit_acceleration_fields(state)

    def _propagate_body_coupling_state(self, state_in: State, state_out: State) -> None:
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
        if state_in.body_qd is not None and state_out.body_qd is not None:
            wp.copy(state_out.body_qd, state_in.body_qd)

    def _refresh_model_dependent_state(self) -> None:
        """Refresh SPH caches derived from mutable model or config arrays."""
        self._sph_model.refresh_model()

    def _build_neighbor_grid(self, state: State) -> None:
        self._sph_model.build_neighbor_grid(state)

    def _step_impl(self, state_in: State, state_out: State, dt: float, control: Control | None) -> None:
        """Run one WCSPH step."""
        del control

        self._compute_density_pressure(state_in)
        self._stage_explicit_forces(state_in)
        self._stage_integrate(state_in, state_out, dt)

    def _collide_shape_boundaries(self, state: State, dt: float, gap: float | None = None) -> None:
        self._sph_model.collide_shape_boundaries(
            state,
            collider_velocity_mode=self.collider_velocity_mode_id,
            dt=dt,
            mesh_query_max_distance=gap,
        )
