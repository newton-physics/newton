# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in tangential adapter around the unchanged foundation normal solve."""

import numpy as np
import warp as wp

from .contact import contact_kinematics
from .friction_solver import FrictionParams, FrictionSolver
from .runtime import FoundationParams


@wp.kernel
def _prepare(
    count: int,
    plane: int,
    height: float,
    carrier: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    local: wp.array[wp.vec3],
    ground_force: wp.array[wp.vec3],
    pressed: wp.array[float],
    foundation_params: wp.array[FoundationParams],
    smoothing_speed: float,
    yield_width: float,
    use_free_velocity: int,
    points: wp.array[wp.vec3],
    normal: wp.array[float],
    com: wp.array[wp.vec3],
    velocity: wp.array[wp.spatial_vector],
    settings: wp.array[FrictionParams],
):
    i = wp.tid()
    world = i // count
    column = i % count
    body = carrier[world]
    point, center, _v, _gap = contact_kinematics(
        body_q[body], body_qd[body], body_com[body], local[column], height, plane
    )
    points[i] = point
    capacity = pressed[i]
    if plane != 0:
        capacity = ground_force[i][2]
    normal[i] = capacity
    if column == 0:
        com[world] = center
        if use_free_velocity == 0:
            velocity[world] = body_qd[body]
        p = foundation_params[world]
        cfg = FrictionParams()
        cfg.mu = p.mu
        cfg.viscous_ratio = p.friction_viscous_ratio
        cfg.release_dwell = p.friction_release_dwell_s
        cfg.smoothing_speed = smoothing_speed
        cfg.yield_width = yield_width
        settings[world] = cfg


@wp.kernel
def _publish(
    plane: int,
    force: wp.array[wp.vec2],
    anchor: wp.array[wp.vec2],
    stuck: wp.array[int],
    dwell: wp.array[float],
    column_force: wp.array[wp.vec3],
    ground_force: wp.array[wp.vec3],
    tangent_anchor: wp.array[wp.vec2],
    tangent_stuck: wp.array[int],
    tangent_dwell: wp.array[float],
):
    i = wp.tid()
    f = force[i]
    # Preserve the stored normal components, including signed internal transfer.
    column_force[i] = wp.vec3(f[0], f[1], column_force[i][2])
    if plane != 0:
        ground_force[i] = wp.vec3(f[0], f[1], ground_force[i][2])
    tangent_anchor[i] = anchor[i]
    tangent_stuck[i] = stuck[i]
    tangent_dwell[i] = dwell[i]


class FrictionAdapter:
    """Attach an optional friction solver without changing foundation normal mechanics.

    The original normal/compression kernels run once with a disabled *tangential*
    stiffness and disposable bristle histories. This adapter then replaces only
    force XY and the real bristle histories before the existing wrench reduction.
    There are no normal-law, normal-force, contact-point or material overrides.

    The adapter is a mutable forward runtime, like :class:`MidsoleFoundation`.
    For a tape rollout use :class:`FrictionSolver` directly with separate input
    histories and preallocated evaluation slots. The existing differentiable
    foundation retains its original default friction model.

    Args:
        foundation: Foundation to attach to; its default path remains unchanged until attachment.
        mobility: World-space COM inverse spatial inertia, one matrix per world.
            Supply the actual mobility, including constraints, and update it as
            orientation changes. A visual carrier's placeholder mass is not a
            valid articulated-body mobility. Zero mobility means prescribed motion.
        mode: ``bristle``, ``implicit_bristle``, ``regularized``, ``deflection``,
            or ``implicit_deflection``.
        iterations: Fixed nonlinear iterations in coupled modes.
        smoothing_speed: Regularized Coulomb transition speed [m/s].
        yield_width: Must be zero. A nonzero shoulder causes timestep-dependent static creep.
        free_velocity: Optional per-world friction-free velocity [m/s, rad/s].
            Otherwise the current carrier velocity defines a friction-only split.
            The consumer still integrates the returned wrench; do not also apply
            the solver's predicted velocity as a second impulse.
    """

    def __init__(
        self,
        foundation,
        mobility,
        *,
        mode="implicit_bristle",
        iterations=8,
        smoothing_speed=0.01,
        yield_width=0.0,
        free_velocity=None,
    ):
        current_solver = getattr(foundation, "friction_solver", None)
        if current_solver is not None:
            if not getattr(current_solver, "is_default", False):
                raise ValueError("Foundation already has a friction adapter")
        self.is_default = False
        if not np.isfinite(smoothing_speed) or smoothing_speed <= 0:
            raise ValueError("smoothing_speed must be finite and positive")
        if not np.isfinite(yield_width) or yield_width != 0.0:
            raise ValueError(
                "yield_width != 0.0 is non-idempotent at zero slip velocity (induces spurious "
                "numerical creep relaxation) and is unsupported for qualified production friction. "
                "yield_width must be 0.0."
            )
        device = foundation.compression.device
        w = foundation.world_count
        if mobility.shape != (w,) or mobility.dtype != wp.spatial_matrix or mobility.device != device:
            raise ValueError("mobility must be a world-sized spatial-matrix array on the foundation device")
        matrices = mobility.numpy()
        if not np.isfinite(matrices).all() or not np.allclose(matrices, matrices.transpose(0, 2, 1), atol=1e-7):
            raise ValueError("mobility must be finite and symmetric")
        if np.min(np.linalg.eigvalsh(matrices)) < -1e-7:
            raise ValueError("mobility must be positive semidefinite")
        if free_velocity is not None and (
            free_velocity.shape != (w,) or free_velocity.dtype != wp.spatial_vector or free_velocity.device != device
        ):
            raise ValueError("free_velocity must be a world-sized spatial-vector array on the foundation device")
        self.foundation = foundation
        self.mobility = mobility
        self.smoothing_speed = float(smoothing_speed)
        self.yield_width = float(yield_width)
        self.use_free_velocity = free_velocity is not None
        self.solver = FrictionSolver(foundation.column_count, w, mode=mode, iterations=iterations, device=device)
        n = foundation.column_count * w
        self.zero_stiffness = wp.zeros(foundation.column_count, dtype=float, device=device)
        self.scratch_anchor = wp.zeros(n, dtype=wp.vec2, device=device)
        self.scratch_stuck = wp.zeros(n, dtype=int, device=device)
        self.scratch_dwell = wp.zeros(n, dtype=float, device=device)

        if hasattr(foundation, "tangent_deflection") and foundation.tangent_deflection is not None:
            self.deflection = foundation.tangent_deflection
        else:
            self.deflection = wp.zeros(n, dtype=wp.vec2, device=device)

        if hasattr(foundation, "tangent_deflection"):
            foundation.tangent_deflection = self.deflection
        if hasattr(foundation, "tangent_maxwell_force"):
            foundation.tangent_maxwell_force.zero_()
        self.points = wp.zeros(n, dtype=wp.vec3, device=device)
        self.normal = wp.zeros(n, dtype=float, device=device)
        self.com = wp.zeros(w, dtype=wp.vec3, device=device)
        self.velocity = (
            free_velocity if free_velocity is not None else wp.zeros(w, dtype=wp.spatial_vector, device=device)
        )
        self.params = wp.zeros(w, dtype=FrictionParams, device=device)
        self.result = None

        # Detach default adapter only after all validation and allocations succeed
        if current_solver is not None and getattr(current_solver, "is_default", False):
            current_solver.detach(restore_default=False)

        foundation.friction_solver = self

    def reset(self) -> None:
        """Clear tangential deflection state."""
        self.deflection.zero_()

    def apply(self, state, dt: float) -> None:
        """Replace tangential tractions after the unchanged normal solve."""
        f = self.foundation
        device = f.compression.device
        plane = int(f.ground_height_m is not None)
        wp.launch(
            _prepare,
            dim=f.column_count * f.world_count,
            inputs=[
                f.column_count,
                plane,
                float(f.ground_height_m or 0.0),
                f.carrier,
                state.body_q,
                state.body_qd,
                f.body_com,
                f.anchor_local,
                f.ground_force,
                f.column_pressed,
                f.world_params,
                self.smoothing_speed,
                self.yield_width,
                int(self.use_free_velocity),
                self.points,
                self.normal,
                self.com,
                self.velocity,
                self.params,
            ],
            device=device,
        )
        self.result = self.solver.solve(
            self.points,
            self.normal,
            self.com,
            self.velocity,
            self.mobility,
            f.friction_kt,
            f.friction_kv,
            self.params,
            f.tangent_anchor,
            f.tangent_stuck,
            f.tangent_dwell,
            dt,
            deflection=self.deflection,
        )
        wp.copy(self.deflection, self.result.deflection)
        wp.launch(
            _publish,
            dim=f.column_count * f.world_count,
            inputs=[
                plane,
                self.result.force,
                self.result.anchor,
                self.result.stuck,
                self.result.dwell,
                f.column_force,
                f.ground_force,
                f.tangent_anchor,
                f.tangent_stuck,
                f.tangent_dwell,
            ],
            device=device,
        )

    def detach(self, restore_default: bool = True) -> None:
        """Restore the foundation's original friction path, retaining its current history."""
        if getattr(self.foundation, "friction_solver", None) is self:
            self.foundation.friction_solver = None
            if restore_default and getattr(self.foundation, "config", None) is not None:
                if getattr(self.foundation.config, "friction_model", "legacy") in (
                    "elastic_coulomb",
                    "maxwell",
                    "column_maxwell",
                ):
                    self.foundation._install_default_friction_adapter()
