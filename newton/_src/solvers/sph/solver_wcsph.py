# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoothed Particle Hydrodynamics solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import ParticleFlags
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
from .cpu import (
    _compute_acceleration_cpu,
    _compute_density_pressure_cpu,
    _compute_surface_fields_cpu,
    _compute_xsph_velocity_delta_cpu,
    _integrate_particles_cpu,
)
from .force_kernels import (
    compute_acceleration,
    compute_xsph_velocity_delta,
)
from .force_kernels import (
    compute_surface_fields as _compute_surface_fields,
)
from .sph_model import (
    SPHConfig,
    SPHModel,
    SPHRole,
    register_sph_custom_attributes,
    validate_sph_custom_attributes,
)

__all__ = ["SolverWCSPH"]

_SPH_CONFIG_FIELDS = fields(SPHConfig)
_SPH_CONFIG_FIELD_NAMES = frozenset(field.name for field in _SPH_CONFIG_FIELDS)
_SPH_INTEGRATION_CARRY_FIELDS = (
    "density",
    "pressure",
    "volume",
    "acceleration",
    "surface_acceleration",
    "adhesion_acceleration",
    "wetting_acceleration",
    "normal",
    "boundary_normal",
    "color_field",
    "velocity_delta",
)


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
    impulse = body_impulses[body]
    position = wp.transform_point(body_q[body], body_com[body])
    impulse_norm_sq = wp.dot(impulse, impulse)
    if impulse_norm_sq > 1.0e-16:
        position += wp.cross(impulse, body_angular_impulses[body]) / impulse_norm_sq

    impulses[collider] = impulse
    impulse_positions[collider] = position
    impulse_ids[collider] = collider_ids[collider]


def _resolve_sph_solver_config(
    config_type: type[SPHConfig],
    config: SPHConfig | None,
    overrides: Mapping[str, Any],
) -> SPHConfig:
    unknown = sorted(name for name in overrides if name not in _SPH_CONFIG_FIELD_NAMES)
    if unknown:
        joined = ", ".join(unknown)
        raise TypeError(f"Unknown SolverWCSPH config option(s): {joined}")

    resolved = config_type() if config is None else replace(config)
    for field in _SPH_CONFIG_FIELDS:
        if field.name in overrides:
            setattr(resolved, field.name, overrides[field.name])

    resolved.validate()
    return resolved


class SolverWCSPH(SolverBase):
    """Weakly compressible Smoothed Particle Hydrodynamics solver.

    Implements an explicit WCSPH particle solver with density summation, an
    equation-of-state pressure solve, viscosity, XSPH smoothing, and sampled or
    analytic boundary handling.

    Call :meth:`register_custom_attributes` on your :class:`~newton.ModelBuilder`
    before building the model to enable SPH-specific per-particle material and
    state fields (for example ``sph:role``, ``sph:rest_density``, and
    ``sph:smoothing_length``).

    Args:
        model: The Newton model to simulate.
        config: Optional solver configuration. Keyword overrides may also set
            any :class:`SolverWCSPH.Config` field.
    """

    @dataclass
    class Config(SPHConfig):
        """Configuration for :class:`SolverWCSPH`.

        Per-particle properties can be configured using custom attributes on the model.
        See :meth:`SolverWCSPH.register_custom_attributes` for details.
        """

    @classmethod
    @override
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register SPH-specific custom attributes in the ``sph`` namespace."""
        register_sph_custom_attributes(builder)

    def __init__(
        self,
        model: Model,
        config: SPHConfig | None = None,
        **config_overrides: object,
    ):
        super().__init__(model)

        config = _resolve_sph_solver_config(type(self).Config, config, config_overrides)
        self.config = config
        self.collider_velocity_mode = config.collider_velocity_mode
        self.collider_velocity_mode_id = (
            SPH_COLLIDER_VELOCITY_BACKWARD
            if self.collider_velocity_mode == "backward"
            else SPH_COLLIDER_VELOCITY_FORWARD
        )
        validate_sph_custom_attributes(model)
        self._sph_model = SPHModel(model, config)
        self._last_acceleration_fused_xsph = False

    def setup_collider(
        self,
        collider_meshes: list[wp.Mesh] | None = None,
        collider_body_ids: list[int] | None = None,
        collider_margins: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_adhesion: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        model: Model | None = None,
        body_com: wp.array[wp.vec3] | None = None,
        body_mass: wp.array[float] | None = None,
        body_inv_inertia: wp.array[wp.mat33] | None = None,
        body_q: wp.array[wp.transform] | None = None,
    ) -> None:
        """Configure Newton shapes used as SPH analytic colliders.

        This mirrors the implicit MPM collider entry point for Newton-body
        coupling. The current WCSPH backend supports Newton collision shapes
        with ``ShapeFlags.COLLIDE_PARTICLES`` and standalone ``collider_meshes``.
        CPU and GPU shape projection include primitive, model-owned triangle
        mesh, static standalone mesh, and body-driven standalone mesh colliders.

        Args:
            collider_meshes: Warp triangular meshes used as colliders.
            collider_body_ids: For dynamic colliders, per-mesh body ids.
            collider_margins: Per-mesh signed distance offsets (m).
            collider_friction: Per-mesh Coulomb friction coefficients.
            collider_adhesion: Per-mesh adhesion scale.
            collider_projection_threshold: Per-mesh projection threshold (m).
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
            collider_adhesion=collider_adhesion,
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

        Computes density, pressure, explicit forces, optional viscosity and
        boundary coupling terms, then integrates particle positions and
        velocities.

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
        """Collect analytic-boundary impulses applied by SPH fluid to colliders.

        Returns a tuple of 3 arrays:
            - Impulse values in world units.
            - Collider positions in world units.
            - Collider id, that can be mapped back to the model's body ids using
              the ``collider_body_index`` property.
        """
        sph_model = self._sph_model
        handler = sph_model.boundary_handler
        body_q = state.body_q if state.body_q is not None else sph_model.collider_body_q
        if sph_model.collider_impulse.shape[0] > 0:
            if (
                body_q is None
                or sph_model.collider_body_com is None
                or handler.analytic_body_impulse_wp is None
                or handler.analytic_body_angular_impulse_wp is None
            ):
                raise RuntimeError("SPH collider impulse buffers are not initialized")
            wp.launch(
                _gather_collider_impulses,
                dim=sph_model.collider_impulse.shape[0],
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
        need an explicit collider projection pass. ``gap`` is accepted for API
        alignment; WCSPH projection distances are configured through
        ``setup_collider`` margins and projection thresholds.

        Args:
            state_in: The input state.
            state_out: The output state. Only particle_q, particle_qd, body_q,
                body_qd, and SPH boundary impulse buffers are written.
            dt: The time step, for estimating collider motion.
            gap: Accepted for implicit MPM API compatibility.
        """
        del gap
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and positive")
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

        self._collide_shape_boundaries(state_out, dt)

    def _active_role_mask(self, role: SPHRole | int, *, dynamic: bool = False) -> np.ndarray:
        flags = self.model.particle_flags.numpy()
        roles = self.model.sph.role.numpy()
        mask = ((flags & int(ParticleFlags.ACTIVE)) != 0) & (roles == int(role))
        if dynamic:
            mask &= self.model.particle_inv_mass.numpy() > 0.0
        return mask

    def _compute_density_pressure(self, state: State) -> None:
        """Compute SPH density and pressure fields for ``state``."""
        model = self.model
        if not model.particle_count:
            return

        if getattr(model.device, "is_cpu", False):
            density, pressure, volume = _compute_density_pressure_cpu(self, state)
            state.sph.density.assign(density)
            state.sph.pressure.assign(pressure)
            state.sph.volume.assign(volume)
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
                model.sph.role,
                model.sph.rest_density,
                model.sph.sound_speed,
                model.sph.stiffness,
                model.sph.pressure_exponent,
                model.sph.pressure_min,
                model.sph.pressure_max,
                model.sph.smoothing_length,
                self.config.rest_density,
                self.config.sound_speed,
                self.config.stiffness,
                self.config.pressure_exponent,
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

        max_velocity = model.particle_max_velocity

        if getattr(model.device, "is_cpu", False):
            q_out, qd_out = _integrate_particles_cpu(self, state_in, dt, float(max_velocity))
            state_out.particle_q.assign(q_out)
            state_out.particle_qd.assign(qd_out)
        else:
            wp.launch(
                integrate_sph_particles,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_inv_mass,
                    model.particle_flags,
                    model.sph.role,
                    state_in.sph.acceleration,
                    state_in.sph.velocity_delta,
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

    def _stage_xsph_filter(self, state: State) -> None:
        model = self.model
        if self._last_acceleration_fused_xsph:
            return
        if self.config.xsph > 0.0:
            if getattr(model.device, "is_cpu", False):
                state.sph.velocity_delta.assign(_compute_xsph_velocity_delta_cpu(self, state))
            else:
                wp.launch(
                    compute_xsph_velocity_delta,
                    dim=model.particle_count,
                    inputs=[
                        self._sph_model.neighbor_search.grid_id,
                        state.particle_q,
                        state.particle_qd,
                        model.particle_mass,
                        model.particle_flags,
                        model.particle_world,
                        model.particle_inv_mass,
                        model.sph.role,
                        model.sph.smoothing_length,
                        state.sph.density,
                        self._sph_model.max_support_radius,
                        self._sph_model.kernel_id,
                    ],
                    outputs=[state.sph.velocity_delta],
                    device=model.device,
                )
        else:
            state.sph.velocity_delta.zero_()

    def _compute_explicit_acceleration_fields(self, state: State) -> None:
        model = self.model
        self._last_acceleration_fused_xsph = False
        enable_surface_tension = self.config.enable_surface_tension
        enable_boundary_adhesion = self.config.enable_boundary_adhesion
        enable_boundary_wetting = self.config.enable_boundary_wetting
        use_basic_gpu_acceleration = not (enable_surface_tension or enable_boundary_adhesion or enable_boundary_wetting)

        if getattr(model.device, "is_cpu", False):
            acceleration, surface_acceleration, adhesion_acceleration, wetting_acceleration = _compute_acceleration_cpu(
                self, state
            )
            state.sph.acceleration.assign(acceleration)
            state.sph.surface_acceleration.assign(surface_acceleration)
            state.sph.adhesion_acceleration.assign(adhesion_acceleration)
            state.sph.wetting_acceleration.assign(wetting_acceleration)
            return

        if use_basic_gpu_acceleration:
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
                    model.sph.role,
                    model.sph.viscosity,
                    model.sph.smoothing_length,
                    state.sph.density,
                    state.sph.pressure,
                    self.config.viscosity,
                    self._sph_model.max_support_radius,
                    self._sph_model.kernel_id,
                    self.config.xsph > 0.0,
                ],
                outputs=[
                    state.sph.acceleration,
                    state.sph.surface_acceleration,
                    state.sph.adhesion_acceleration,
                    state.sph.wetting_acceleration,
                    state.sph.velocity_delta,
                ],
                device=model.device,
            )
            self._last_acceleration_fused_xsph = self.config.xsph > 0.0
            return

        wp.launch(
            compute_acceleration,
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
                model.sph.role,
                model.sph.viscosity,
                model.sph.surface_tension,
                model.sph.adhesion,
                model.sph.wetting,
                model.sph.contact_angle,
                model.sph.smoothing_length,
                state.sph.density,
                state.sph.pressure,
                state.sph.volume,
                state.sph.normal,
                state.sph.boundary_normal,
                self.config.viscosity,
                self._sph_model.max_support_radius,
                self._sph_model.kernel_id,
                enable_surface_tension,
                self.config.surface_tension_normal_threshold,
                enable_boundary_adhesion,
                enable_boundary_wetting,
            ],
            outputs=[
                state.sph.acceleration,
                state.sph.surface_acceleration,
                state.sph.adhesion_acceleration,
                state.sph.wetting_acceleration,
            ],
            device=model.device,
        )

    def _stage_explicit_forces(self, state: State) -> None:
        needs_surface_fields = (
            self.config.enable_surface_tension
            or self.config.enable_boundary_adhesion
            or self.config.enable_boundary_wetting
        )
        if needs_surface_fields:
            self._compute_surface_fields(state)

        self._compute_explicit_acceleration_fields(state)
        self._stage_xsph_filter(state)

    def _compute_surface_fields(self, state: State) -> None:
        """Compute color-field and normals for free-surface workflows."""
        model = self.model
        if not model.particle_count:
            return

        if getattr(model.device, "is_cpu", False):
            color_field, normal = _compute_surface_fields_cpu(self, state)
            state.sph.color_field.assign(color_field)
            state.sph.normal.assign(normal)
            return

        self._build_neighbor_grid(state)

        wp.launch(
            _compute_surface_fields,
            dim=model.particle_count,
            inputs=[
                self._sph_model.neighbor_search.grid_id,
                state.particle_q,
                model.particle_flags,
                model.particle_world,
                model.particle_inv_mass,
                model.sph.role,
                model.sph.smoothing_length,
                state.sph.volume,
                self._sph_model.max_support_radius,
                self._sph_model.kernel_id,
            ],
            outputs=[state.sph.color_field, state.sph.normal],
            device=model.device,
        )

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

    def _support_radius_np(self, support: np.ndarray | None = None) -> np.ndarray:
        return self._sph_model.support_radius_np(support)

    def _collide_shape_boundaries(self, state: State, dt: float) -> None:
        self._sph_model.collide_shape_boundaries(state, collider_velocity_mode=self.collider_velocity_mode_id, dt=dt)
