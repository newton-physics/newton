# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Data containers for the Kamino DVI solver."""

from __future__ import annotations

import warp as wp

from ....config import DVISolverConfig
from ...core.size import SizeKamino
from ...linalg import DenseLinearOperatorData
from ..common import DualSolution

wp.set_module_options({"enable_backward": False})

float32 = wp.float32
int32 = wp.int32
uint64 = wp.uint64
vec2f = wp.vec2f
vec2i = wp.vec2i

_DVI_CONTACT_SOLVER_PGS = 0
_DVI_CONTACT_SOLVER_APGD = 1


@wp.struct
class DVIConfigStruct:
    """On-device DVI solver configuration."""

    tolerance: float32
    """Tolerance for the terminal full-system DVI residuals."""

    regularization: float32
    """Diagonal regularization used by projected Gauss-Seidel updates."""

    omega: float32
    """Projected Gauss-Seidel update relaxation."""

    contact_solver: int32
    """Device-side selector for PGS or APGD contact phases."""

    coupling_iterations: int32
    """Number of complete ``L -> B -> C`` coupling sweeps."""

    limit_pgs_sweeps: int32
    """Projected sweeps used by each bounded-joint and joint-limit ``L`` phase."""

    contact_pgs_sweeps: int32
    """Projected sweeps used by each PGS contact ``C`` phase."""

    tangential_warmstart_scale: float32
    """Scale applied to cached tangential reactions before each solve."""

    post_stabilization_bilateral: int32
    """Whether the bilateral block is refreshed after the final contact phase."""


@wp.struct
class DVIStatus:
    """Per-world DVI convergence status."""

    converged: int32
    """Whether all terminal feasibility, equality, and complementarity residuals satisfy tolerance."""
    iterations: int32
    """Top-level coupling passes executed; one when at most one family is active."""
    limit_iterations: int32
    """Actual projected sweeps applied to bounded-joint and joint-limit rows."""
    contact_iterations: int32
    """Actual PGS or APGD iterations applied to contact rows across all contact phases."""
    contact_backtracks: int32
    """Actual APGD Lipschitz-doubling passes across all contact phases."""
    contact_restarts: int32
    """Actual APGD momentum restarts across all contact phases."""
    contact_solver_residual: float32
    """Last APGD phase's running-minimum Res4 norm; zero for PGS."""
    invalid_contact_preconditioner: int32
    """Whether APGD rejected invalid or unequal contact-triplet scaling.

    A rejected world's exported impulses and velocities are zero sentinels and
    must not be interpreted as a physical solution.
    """
    r_natural: float32
    """Terminal infinity norm of the law-specific projected natural map."""
    r_p: float32
    """Maximum primal box- and cone-feasibility residual."""
    r_d: float32
    """Maximum dual cone-feasibility and bilateral velocity residual."""
    r_c: float32
    """Maximum absolute impulse-velocity product, directional on box rows."""
    r_b: float32
    """Bilateral constraint-space velocity residual."""


class DVIInfo:
    """Optional terminal convergence diagnostics for each simulated world."""

    def __init__(self, size: SizeKamino | None = None):
        self.status: wp.array[DVIStatus] | None = None
        """Terminal DVI status, shape ``(num_worlds,)``."""
        if size is not None:
            self.finalize(size)

    def finalize(self, size: SizeKamino) -> None:
        """Allocate diagnostic arrays for a model size."""
        self.status = wp.zeros(shape=(size.num_worlds,), dtype=DVIStatus)

    def zero(self) -> None:
        """Reset diagnostics to zero."""
        self.status.zero_()


class DVIState:
    """Scratch arrays used by the DVI solver."""

    def __init__(self, size: SizeKamino | None = None):
        self.sigma: wp.array[vec2f] | None = None
        """Zero proximal terms used when evaluating shared solution metrics."""
        self.v_aug: wp.array[float32] | None = None
        self.s: wp.array[float32] | None = None
        self.scratch: wp.array[float32] | None = None
        self.bilateral_rhs: wp.array[float32] | None = None
        self.bilateral_solution: wp.array[float32] | None = None
        self.bilateral_preconditioner: wp.array[float32] | None = None
        self.bilateral_active_dim: wp.array[int32] | None = None
        self.contact_active_mask: wp.array[wp.bool] | None = None
        self.limit_indices: wp.array[int32] | None = None
        self.contact_indices: wp.array[int32] | None = None
        self.inequality_bodies: wp.array[vec2i] | None = None
        self.inequality_body_color_masks: wp.array[uint64] | None = None
        self.inequality_colors: wp.array[int32] | None = None
        self.inequality_num_colors: wp.array[int32] | None = None
        self.inequality_ids_by_color: wp.array[int32] | None = None
        self.inequality_color_starts: wp.array[int32] | None = None
        if size is not None:
            self.finalize(size)

    def finalize(self, size: SizeKamino):
        """Allocate scratch arrays for the supplied model size."""
        self.sigma = wp.zeros(size.num_worlds, dtype=vec2f)
        self.v_aug = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.s = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.scratch = wp.zeros(size.sum_of_max_total_cts, dtype=float32)
        self.bilateral_rhs = wp.zeros(size.sum_of_num_bilateral_joint_cts, dtype=float32)
        self.bilateral_solution = wp.zeros(size.sum_of_num_bilateral_joint_cts, dtype=float32)
        self.bilateral_preconditioner = wp.zeros(size.sum_of_num_bilateral_joint_cts, dtype=float32)
        self.bilateral_active_dim = wp.zeros(size.num_worlds, dtype=int32)
        self.contact_active_mask = wp.zeros(size.num_worlds, dtype=wp.bool)
        self.limit_indices = wp.full(max(1, size.sum_of_max_limits), -1, dtype=int32)
        self.contact_indices = wp.full(max(1, size.sum_of_max_contacts), -1, dtype=int32)
        self.inequality_bodies = wp.full(max(1, size.sum_of_max_inequalities), vec2i(-1, -1), dtype=vec2i)
        self.inequality_body_color_masks = wp.zeros(max(1, size.sum_of_num_bodies), dtype=uint64)
        self.inequality_colors = wp.full(max(1, size.sum_of_max_inequalities), -1, dtype=int32)
        self.inequality_num_colors = wp.zeros(max(1, size.num_worlds), dtype=int32)
        self.inequality_ids_by_color = wp.full(max(1, size.sum_of_max_inequalities), -1, dtype=int32)
        self.inequality_color_starts = wp.zeros(max(1, size.sum_of_max_inequalities + size.num_worlds), dtype=int32)

    def reset(self):
        """Reset scratch arrays to zero."""
        self.sigma.zero_()
        self.v_aug.zero_()
        self.s.zero_()
        self.scratch.zero_()
        self.bilateral_rhs.zero_()
        self.bilateral_solution.zero_()
        self.bilateral_preconditioner.zero_()
        self.bilateral_active_dim.zero_()
        self.contact_active_mask.zero_()
        self.limit_indices.fill_(-1)
        self.contact_indices.fill_(-1)
        self.inequality_bodies.fill_(vec2i(-1, -1))
        self.inequality_body_color_masks.zero_()
        self.inequality_colors.fill_(-1)
        self.inequality_num_colors.zero_()
        self.inequality_ids_by_color.fill_(-1)
        self.inequality_color_starts.zero_()


class DVIData:
    """High-level DVI solver data."""

    def __init__(
        self,
        size: SizeKamino | None = None,
        collect_info: bool = False,
        device: wp.DeviceLike = None,
    ):
        self.config: wp.array[DVIConfigStruct] | None = None
        self.status: wp.array[DVIStatus] | None = None
        self.state: DVIState | None = None
        self.solution: DualSolution | None = None
        self.info: DVIInfo | None = None
        self.bilateral_operator: DenseLinearOperatorData | None = None
        if size is not None:
            self.finalize(size=size, collect_info=collect_info, device=device)

    def finalize(self, size: SizeKamino, collect_info: bool = False, device: wp.DeviceLike = None):
        """Allocate DVI data arrays."""
        with wp.ScopedDevice(device):
            self.config = wp.zeros(shape=(size.num_worlds,), dtype=DVIConfigStruct)
            self.status = wp.zeros(shape=(size.num_worlds,), dtype=DVIStatus)
            self.state = DVIState(size)
            self.solution = DualSolution(size)
            self.info = DVIInfo(size) if collect_info else None
            self.bilateral_operator = None


def convert_config_to_struct(config: DVISolverConfig) -> DVIConfigStruct:
    """Convert a host-side DVI config to an on-device struct."""
    config_struct = DVIConfigStruct()
    config_struct.tolerance = config.tolerance
    config_struct.regularization = config.regularization
    config_struct.omega = config.omega
    config_struct.contact_solver = (
        _DVI_CONTACT_SOLVER_APGD if config.contact_solver == "apgd" else _DVI_CONTACT_SOLVER_PGS
    )
    config_struct.coupling_iterations = config.coupling_iterations
    config_struct.limit_pgs_sweeps = config.limit_pgs_sweeps
    config_struct.contact_pgs_sweeps = config.contact_pgs_sweeps
    config_struct.tangential_warmstart_scale = config.tangential_warmstart_scale
    config_struct.post_stabilization_bilateral = int(config.post_stabilization_bilateral)
    return config_struct
