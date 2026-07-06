# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Defines data types and containers used by the PADMM solver.

High-level Settings:
- :class:`PADMMPenaltyUpdate`:
    Defines the ALM penalty update methods supported by the PADMM solver.
- :class:`PADMMWarmStartMode`:
    Defines the warmstart modes supported by the PADMM solver.

Warp Structs:
- :class:`PADMMConfigStruct`:
    Warp struct for on-device PADMM configurations.
- :class:`PADMMStatus`:
    Warp struct for on-device PADMM solver status.
- :class:`PADMMPenalty`:
    Warp struct for on-device PADMM penalty state.

Data Containers:
- :class:`PADMMState`:
    A data container managing the internal PADMM solver state arrays.
- :class:`PADMMResiduals`:
    A data container managing the PADMM solver residuals arrays.
- :class:`PADMMSolution`:
    A data container managing the PADMM solver solution arrays.
- :class:`PADMMInfo`:
    A data container managing arrays PADMM solver convergence info and performance metrics.
- :class:`PADMMData`:
    The highest-level PADMM data container, bundling all other PADMM-related data into a single object.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
import warp as wp

from ......core.types import override
from ....config import PADMMSolverConfig
from ...core.size import SizeKamino
from ...core.types import to_warp_int32_array

###
# Module interface
###

__all__ = [
    "PADMMConfigStruct",
    "PADMMData",
    "PADMMInfo",
    "PADMMPenalty",
    "PADMMPenaltyUpdate",
    "PADMMResiduals",
    "PADMMSolution",
    "PADMMState",
    "PADMMStatus",
    "PADMMWarmStartMode",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


class PADMMPenaltyUpdate(IntEnum):
    """
    An enumeration of the penalty update methods used in PADMM.
    """

    FIXED = 0
    """
    Fixed penalty:
        `rho` is initialized to `config.rho_0`, remaining constant over the solve.
    """

    # TODO: Implement adaptive penalty updates
    # LINEAR = 1
    # """
    # Linear penalty update:
    # `rho` is increased by a fixed factor.
    # """
    # BALANCED = 1
    # """
    # Balanced-residuals penalty update:
    # `rho` is increased in order for the ratio of primal/dual residuals to be close to unity.
    # """
    # SPECTRAL = 2
    # """
    # Spectral penalty update:
    # `rho` is increased by the spectral radius of the Delassus matrix.
    # """
    BALANCED = 1
    """
    Balanced-residuals penalty update:
    `rho` is increased in order for the ratio of primal/dual residuals to be close to unity.
    """

    @classmethod
    def from_string(cls, s: str) -> PADMMPenaltyUpdate:
        """Converts a string to a PADMMPenaltyUpdate enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid PADMMPenaltyUpdate: {s}. Valid options are: {[e.name for e in cls]}") from e

    @override
    def __str__(self):
        """Returns a string representation of the PADMMPenaltyUpdate."""
        return f"PADMMPenaltyUpdate.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the PADMMPenaltyUpdate."""
        return self.__str__()


class PADMMWarmStartMode(IntEnum):
    """
    An enumeration of the warmstart modes used in PADMM.
    """

    NONE = -1
    """
    No warmstart:
        The solver does not use any warmstart information and starts from
        scratch, i.e. performs a cold-start regardless of any cached state.
    """

    INTERNAL = 0
    """
    From internally cached solution:
        The solver uses its values currently in the solution
        container as warmstart information for the current solve.
    """

    CONTAINERS = 1
    """
    From externally cached solution containers:
        The solver uses values from externally provided solution
        containers as warmstart information for the current solve.
    """

    @classmethod
    def from_string(cls, s: str) -> PADMMWarmStartMode:
        """Converts a string to a PADMMWarmStartMode enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid PADMMWarmStartMode: {s}. Valid options are: {[e.name for e in cls]}") from e

    @override
    def __str__(self):
        """Returns a string representation of the PADMMWarmStartMode."""
        return f"PADMMWarmStartMode.{self.name} ({self.value})"

    @override
    def __repr__(self):
        """Returns a string representation of the PADMMWarmStartMode."""
        return self.__str__()

    @staticmethod
    def parse_usd_attribute(value: str, context: dict[str, Any] | None = None) -> str:
        """Parse warmstart option imported from USD, following the KaminoSceneAPI schema."""
        if not isinstance(value, str):
            raise TypeError("Parser expects input of type 'str'.")
        mapping = {"none": "none", "internal": "internal", "containers": "containers"}
        lower_value = value.lower().strip()
        if lower_value not in mapping:
            raise ValueError(f"Warmstart parameter '{value}' is not a valid option.")
        return mapping[lower_value]


@wp.struct
class PADMMConfigStruct:
    """
    A warp struct to hold PADMM per-world solver configurations on the target device.

    Intended to be used as ``dtype`` for warp arrays.

    Attributes:
        primal_tolerance: The target tolerance on the total primal residual `r_primal`.
            Must be greater than zero. Defaults to `1e-6`.
        dual_tolerance: The target tolerance on the total dual residual `r_dual`.
            Must be greater than zero. Defaults to `1e-6`.
        compl_tolerance: The target tolerance on the total complementarity residual `r_compl`.
            Must be greater than zero. Defaults to `1e-6`.
        restart_tolerance: The tolerance on the total combined primal-dual
            residual `r_comb`, for determining when gradient acceleration should be restarted.
            Must be greater than zero. Defaults to `0.999`.
        eta: The proximal regularization parameter.
            Must be greater than zero. Defaults to `1e-5`.
        rho_0: The initial value of the penalty parameter.
            Must be greater than zero. Defaults to `1.0`.
        a_0: The initial value of the acceleration parameter.
            Must be greater than zero. Defaults to `1.0`.
        alpha: The threshold on primal-dual residual ratios,
            used to determine when penalty updates should occur.
            Must be greater than `1.0`. Defaults to `10.0`.
        tau: The factor by which the penalty is increased/decreased
            when the primal-dual residual ratios exceed the threshold `alpha`.
            Must be greater than `1.0`. Defaults to `1.5`.
        max_iterations: The maximum number of solver iterations.
            Must be greater than zero. Defaults to `200`.
        penalty_update_freq: The permitted frequency of penalty updates.
            If zero, no updates are performed. Otherwise, updates are performed every
            `penalty_update_freq` iterations. Defaults to `10`.
        penalty_update_method: The penalty update method used to adapt the penalty parameter.
            Defaults to `PADMMPenaltyUpdate.FIXED`.
            See :class:`PADMMPenaltyUpdate` for details.
    """

    primal_tolerance: wp.float32
    """
    The target tolerance on the total primal residual `r_primal`.
    Must be greater than zero. Defaults to `1e-6`.
    """

    dual_tolerance: wp.float32
    """
    The target tolerance on the total dual residual `r_dual`.
    Must be greater than zero. Defaults to `1e-6`.
    """

    compl_tolerance: wp.float32
    """
    The target tolerance on the total complementarity residual `r_compl`.
    Must be greater than zero. Defaults to `1e-6`.
    """

    restart_tolerance: wp.float32
    """
    The tolerance applied on the total combined primal-dual residual `r_comb`,
    for determining when gradient acceleration should be restarted.
    Must be greater than zero. Defaults to `0.999`.
    """

    eta: wp.float32
    """
    The proximal regularization parameter.
    Must be greater than zero. Defaults to `1e-5`.
    """

    rho_0: wp.float32
    """
    The initial value of the ALM penalty parameter.
    Must be greater than zero. Defaults to `1.0`.
    """

    rho_min: wp.float32
    """
    The lower-bound applied to the ALM penalty parameter.
    Must be greater than zero. Defaults to `1e-5`.
    """

    a_0: wp.float32
    """
    The initial value of the acceleration parameter.
    Must be greater than zero. Defaults to `1.0`.
    """

    alpha: wp.float32
    """
    The threshold on primal-dual residual ratios,
    used to determine when penalty updates should occur.
    Must be greater than `1.0`. Defaults to `10.0`.
    """

    tau: wp.float32
    """
    The factor by which the penalty is increased/decreased
    when the primal-dual residual ratios exceed the threshold `alpha`.
    Must be greater than `1.0`. Defaults to `1.5`.
    """

    max_iterations: wp.int32
    """
    The maximum number of solver iterations.
    Must be greater than zero. Defaults to `200`.
    """

    penalty_update_freq: wp.int32
    """
    The permitted frequency of penalty updates.
    If zero, no updates are performed. Otherwise, updates are performed every
    `penalty_update_freq` iterations. Defaults to `10`.
    """

    penalty_update_method: wp.int32
    """
    The penalty update method used to adapt the penalty parameter.
    Defaults to `PADMMPenaltyUpdate.FIXED`.
    See :class:`PADMMPenaltyUpdate` for details.
    """

    linear_solver_tolerance: wp.float32
    """
    The default absolute tolerance for the iterative linear solver.
    When positive, the iterative solver's atol is initialized to this value
    at the start of each ADMM solve.
    When zero, the iterative solver's own tolerance is left unchanged.
    Must be non-negative. Defaults to `0.0`.
    """

    linear_solver_tolerance_ratio: wp.float32
    """
    The ratio used to adapt the iterative linear solver tolerance from the ADMM primal residual.
    When positive, the linear solver absolute tolerance is set to
    `ratio * ||r_primal||_2` at each ADMM iteration.
    When zero, the linear solver tolerance is not adapted (fixed tolerance).
    Must be non-negative. Defaults to `0.0`.
    """


@wp.struct
class PADMMStatus:
    """
    A warp struct to hold the PADMM per-world solver status on the target device.

    Intended to be used as ``dtype`` for warp arrays.

    Attributes:
        converged: A flag indicating whether the solver has converged (`1`) or not (`0`).
            Used internally to keep track of per-world convergence status,
            with `1` being set only when all total residuals have satisfied their
            respective tolerances. If by the end of the solve the flag is still `0`,
            it indicates that the solve reached the maximum number of iterations.
        iterations: The number of iterations performed by the solver.
            Used internally to keep track of per-world iteration counts.
        r_p: The total primal residual.
            Computed using the L-inf norm as `r_primal := || x - y ||_inf`.
        r_d: The total dual residual.
            Computed using the L-inf norm as `r_dual := || eta * (x - x_p) + rho * (y - y_p) ||_inf`.
        r_c: The total complementarity residual.
            Computed using the L-inf norm as `r_compl := || [x_k.T @ z_k] ||_inf`,
            with `k` indexing each unilateral constraint set, i.e. 1D limits and 3D contacts.
        r_dx: The total primal iterate residual.
            Computed as the L2-norm `r_dx := || x - x_p ||_2`.
        r_dy: The total slack iterate residual.
            Computed as the L2-norm `r_dy := || y - y_p ||_2`.
        r_dz: The total dual iterate residual.
            Computed as the L2-norm `r_dz := || z - z_p ||_2`.
        r_a: The total combined primal-dual residual used for acceleration restart checks.
            Computed as `r_a := rho * r_dy + (1.0 / rho) * r_dz`.
        r_a_p: The previous total combined primal-dual residual.
        r_a_pp: An auxiliary cache of the previous total combined primal-dual residual.
        restart: A flag indicating whether gradient acceleration requires a restart (`1`) or not (`0`).
            Used internally to keep track of per-world acceleration restarts.
        num_restarts: The number of acceleration restarts performed during the solve.
    """

    converged: wp.int32
    """
    A flag indicating whether the solver has converged (`1`) or not (`0`).
    Used internally to keep track of per-world convergence status,
    with `1` being set only when all total residuals have satisfied their
    respective tolerances. If by the end of the solve the flag is still `0`,
    it indicates that the solve reached the maximum number of iterations.
    """

    iterations: wp.int32
    """
    The number of iterations performed by the solver.
    Used internally to keep track of per-world iteration counts.
    """

    r_p: wp.float32
    """
    The total primal residual.
    Computed using the L-inf norm as `r_primal := || x - y ||_inf`.
    """

    r_d: wp.float32
    """
    The total dual residual.
    Computed using the L-inf norm as `r_dual := || eta * (x - x_p) + rho * (y - y_p) ||_inf`.
    """

    r_c: wp.float32
    """
    The total complementarity residual.
    Computed using the L-inf norm as `r_compl := || [x_k.T @ z_k] ||_inf`,
    with `k` indexing each unilateral constraint set, i.e. 1D limits and 3D contacts.
    """

    r_dx: wp.float32
    """
    The total primal iterate residual.
    Computed as the L2-norm `r_dx := || x - x_p ||_2`.
    """

    r_dy: wp.float32
    """
    The total slack iterate residual.
    Computed as the L2-norm `r_dy := || y - y_p ||_2`.
    """

    r_dz: wp.float32
    """
    The total dual iterate residual.
    Computed as the L2-norm `r_dz := || z - z_p ||_2`.
    """

    r_a: wp.float32
    """
    The total combined primal-dual residual used for acceleration restart checks.
    Computed as `r_a := rho * r_dy + (1.0 / rho) * r_dz`.
    """

    r_a_p: wp.float32
    """The previous total combined primal-dual residual."""

    r_a_pp: wp.float32
    """An auxiliary cache of the previous total combined primal-dual residual."""

    restart: wp.int32
    """
    A flag indicating whether gradient acceleration requires a restart (`1`) or not (`0`).
    Used internally to keep track of per-world acceleration restarts.
    """

    num_restarts: wp.int32
    """The number of acceleration restarts performed during the solve."""


@wp.struct
class PADMMPenalty:
    """
    A warp struct to hold the on-device PADMM solver penalty state.

    Intended to be used as ``dtype`` for warp arrays.

    Attributes:
        num_updates: The number of penalty updates performed during a solve.
            If a direct linear-system solver is used, this also
            equals the number of matrix factorizations performed.
        rho: The current value of the ALM penalty parameter.
            If adaptive penalty scheme is used, this value may change during
            solve operations, while being lower-bounded by `config.rho_min`
            to ensure numerical stability.
        rho_p: The previous value of the ALM penalty parameter.
            As diagonal regularization of the lhs matrix (e.g. Delassus
            operator) is performed in-place, we must keep track of the
            previous penalty value  to remove the previous regularization
            before applying the current penalty value.
    """

    num_updates: wp.int32
    """
    The number of penalty updates performed during a solve.
    If a direct linear-system solver is used, this also
    equals the number of matrix factorizations performed.
    """

    rho: wp.float32
    """
    The current value of the ALM penalty parameter.
    If adaptive penalty scheme is used, this value may change during
    solve operations, while being lower-bounded by `config.rho_min`
    to ensure numerical stability.
    """

    rho_p: wp.float32
    """
    The previous value of the ALM penalty parameter.
    As diagonal regularization of the lhs matrix (e.g. Delassus
    operator) is performed in-place, we must keep track of the
    previous penalty value  to remove the previous regularization
    before applying the current penalty value.
    """


###
# Containers
###


class PADMMState:
    """
    A data container to bundle the internal PADMM state arrays.

    Attributes:
        done: A single-element array containing the global
            flag that indicates whether the solver should terminate.
            Its value is initialized to ``num_worlds`` at the beginning of each
            solve, and decremented by one for each world that has converged.
            Shape of ``(1,)``.
        s: The De Saxce correction velocities `s = Gamma(v_plus)`.
            Shape of ``(sum_of_max_total_cts,)``.
        v: The total bias velocity vector serving as the right-hand-side of the PADMM linear system.
            Shape of ``(sum_of_max_total_cts,)``.
        x: The current PADMM primal variables.
            Shape of ``(sum_of_max_total_cts,)``.
        x_p: The previous PADMM primal variables.
            Shape of ``(sum_of_max_total_cts,)``.
        y: The current PADMM slack variables.
            Shape of ``(sum_of_max_total_cts,)``.
        y_p: The previous PADMM slack variables.
            Shape of ``(sum_of_max_total_cts,)``.
        z: The current PADMM dual variables.
            Shape of ``(sum_of_max_total_cts,)``.
        z_p: The previous PADMM dual variables.
            Shape of ``(sum_of_max_total_cts,)``.
        y_hat: The auxiliary PADMM slack variables used with gradient acceleration.
            Shape of ``(sum_of_max_total_cts,)``.
        z_hat: The auxiliary PADMM dual variables used with gradient acceleration.
            Shape of ``(sum_of_max_total_cts,)``.
        a: The per-world current Nesterov acceleration variables.
            Shape of ``(sum_of_max_total_cts,)``.
        a_p: The per-world previous Nesterov acceleration variables.
            Shape of ``(sum_of_max_total_cts,)``.
        a_factor: The per-world Nesterov factor computed from the previous and current acceleration
            variables. Shape of ``(num_worlds,)``.
    """

    def __init__(self, size: SizeKamino | None = None, use_acceleration: bool = False):
        """
        Initializes the PADMM solver state container.

        If a model size is provided, allocates the state arrays accordingly.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
        """

        self.done: wp.array[wp.int32] | None = None
        """
        A single-element array containing the global flag that indicates whether the solver should terminate.
        Its value is initialized to ``num_worlds`` at the beginning of each
        solve, and decremented by one for each world that has converged.
        Shape of ``(1,)``.
        """

        self.sigma: wp.array[wp.vec2f] | None = None
        """
        The scalar diagonal regularization applied uniformly across constraint dimensions.

        This is computed as `sigma = eta + rho`, where `eta` is the
        additional proximal parameter and `rho` is the ALM penalty.

        It is stored as a 2-element vector representing `(sigma, sigma_p)`, where
        `sigma` is the current and `sigma_p` is the previous value, used to undo
        the prior regularization when the ALM penalty parameter `rho` is updated.

        Shape of ``(num_worlds,)``.
        """

        self.s: wp.array[wp.float32] | None = None
        """
        The De Saxce correction velocities `s = Gamma(v_plus)`.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.v: wp.array[wp.float32] | None = None
        """
        The total bias velocity vector serving as the right-hand-side of the PADMM linear system.
        It is computed from the PADMM state and proximal parameters `eta` and `rho`.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.x: wp.array[wp.float32] | None = None
        """
        The current PADMM primal variables.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.x_p: wp.array[wp.float32] | None = None
        """
        The previous PADMM primal variables.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.y: wp.array[wp.float32] | None = None
        """
        The current PADMM slack variables.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.y_p: wp.array[wp.float32] | None = None
        """
        The previous PADMM slack variables.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.z: wp.array[wp.float32] | None = None
        """
        The current PADMM dual variables.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.z_p: wp.array[wp.float32] | None = None
        """
        The previous PADMM dual variables.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.y_hat: wp.array[wp.float32] | None = None
        """
        The auxiliary PADMM slack variables used with gradient acceleration.
        Only allocated if acceleration is enabled.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.z_hat: wp.array[wp.float32] | None = None
        """
        The auxiliary PADMM dual variables used with gradient acceleration.
        Only allocated if acceleration is enabled.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.a: wp.array[wp.float32] | None = None
        """
        The per-world current Nesterov acceleration variables.
        Only allocated if acceleration is enabled.
        Shape of ``(num_worlds,)``.
        """

        self.a_p: wp.array[wp.float32] | None = None
        """
        The per-world previous Nesterov acceleration variables.
        Only allocated if acceleration is enabled.
        Shape of ``(num_worlds,)``.
        """

        self.a_factor: wp.array[wp.float32] | None = None
        """
        The per-world current Nesterov acceleration factor.
        Only allocated if acceleration is enabled.
        Shape of ``(num_worlds,)``.
        """

        # Perform memory allocations if model size is specified
        if size is not None:
            self.finalize(size, use_acceleration)

    def finalize(self, size: SizeKamino, use_acceleration: bool = False):
        """
        Allocates the PADMM solver state arrays based on the model size.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
        """
        # Allocate per-world solver done flags
        self.done = wp.zeros(1, dtype=wp.int32)

        # Allocate primary state variables
        self.sigma = wp.zeros(size.num_worlds, dtype=wp.vec2f)
        self.s = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.v = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.x = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.x_p = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.y = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.y_p = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.z = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.z_p = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)

        # Allocate auxiliary state variables used with acceleration
        if use_acceleration:
            self.y_hat = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
            self.z_hat = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
            self.a = wp.zeros(size.num_worlds, dtype=wp.float32)
            self.a_p = wp.zeros(size.num_worlds, dtype=wp.float32)
            self.a_factor = wp.zeros(size.num_worlds, dtype=wp.float32)

    def reset(self, use_acceleration: bool = False):
        """
        Resets all PADMM state arrays.

        Specifically:
        - PADMM state arrays (primal, slack, dual, rhs, De Saxce correction) are set to zeros.
        - If acceleration is enabled, the momentum arrays `a_p` and `a` are set to ones.

        Args:
            use_acceleration: Whether to reset the acceleration state variables.
                If `True`, auxiliary state variables and acceleration scales are reset as well.
                Defaults to `False`.
        """
        # Reset primary state variables
        self.done.zero_()
        self.sigma.zero_()
        self.s.zero_()
        self.v.zero_()
        self.x.zero_()
        self.x_p.zero_()
        self.y.zero_()
        self.y_p.zero_()
        self.z.zero_()
        self.z_p.zero_()

        # Optionally reset acceleration state
        if use_acceleration:
            # Reset auxiliary state variables
            self.y_hat.zero_()
            self.z_hat.zero_()
            # Reset acceleration scale variables
            self.a.fill_(1.0)
            self.a_p.fill_(1.0)
            self.a_factor.zero_()


class PADMMResiduals:
    """
    A data container to bundle the internal PADMM residual arrays.

    Attributes:
        r_primal: The PADMM primal residual vector, computed as `r_primal := x - y`.
            Shape of ``(sum_of_max_total_cts,)``.
        r_dual: The PADMM dual residual vector, computed as `r_dual := eta * (x - x_p) + rho * (y - y_p)`.
            Shape of ``(sum_of_max_total_cts,)``.
        r_compl: The PADMM complementarity residual vector, computed as `r_compl := [x_j.dot(z_j)]`,
            where `j` indexes each unilateral constraint set (i.e. 1D limits and 3D contacts).
            Shape of ``(sum_of_num_unilateral_cts,)``.
        r_dx: The PADMM primal iterate residual vector, computed as `r_dx := x - x_p`.
            Shape of ``(sum_of_max_total_cts,)``.
        r_dy: The PADMM slack iterate residual vector, computed as `r_dy := y - y_p`.
            Shape of ``(sum_of_max_total_cts,)``.
        r_dz: The PADMM dual iterate residual vector, computed as `r_dz := z - z_p`.
            Shape of ``(sum_of_max_total_cts,)``.
    """

    def __init__(self, size: SizeKamino | None = None, use_acceleration: bool = False):
        """
        Initializes the PADMM residuals container.

        If a model size is provided, allocates the residuals arrays accordingly.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
        """

        self.r_primal: wp.array[wp.float32] | None = None
        """
        The PADMM primal residual vector, computed as `r_primal := x - y`.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.r_dual: wp.array[wp.float32] | None = None
        """
        The PADMM dual residual vector, computed as `r_dual := eta * (x - x_p) + rho * (y - y_p)`.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.r_compl: wp.array[wp.float32] | None = None
        """
        The PADMM complementarity residual vector, computed as `r_compl := [x_j.dot(z_j)]`,
        where `j` indexes each unilateral constraint set (i.e. 1D limits and 3D contacts).
        Shape of ``(sum_of_num_unilateral_cts,)``.
        """

        self.r_dx: wp.array[wp.float32] | None = None
        """
        The PADMM primal iterate residual vector, computed as `r_dx := x - x_p`.
        Only allocated if acceleration is enabled.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.r_dy: wp.array[wp.float32] | None = None
        """
        The PADMM slack iterate residual vector, computed as `r_dy := y - y_p`.
        Only allocated if acceleration is enabled.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.r_dz: wp.array[wp.float32] | None = None
        """
        The PADMM dual iterate residual vector, computed as `r_dz := z - z_p`.
        Only allocated if acceleration is enabled.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        # Perform memory allocations if model size is specified
        if size is not None:
            self.finalize(size, use_acceleration)

    def finalize(self, size: SizeKamino, use_acceleration: bool = False):
        """
        Allocates the residuals arrays based on the model size.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
            use_acceleration: Flag indicating whether to allocate arrays used with acceleration.
        """
        # Allocate the main residuals arrays
        self.r_primal = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.r_dual = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.r_compl = wp.zeros(size.sum_of_max_unilaterals, dtype=wp.float32)

        # Optionally allocate iterate residuals used when acceleration is enabled
        if use_acceleration:
            self.r_dx = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
            self.r_dy = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
            self.r_dz = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)

    def zero(self, use_acceleration: bool = False):
        """
        Resets all PADMM residual arrays to zeros.
        """
        self.r_primal.zero_()
        self.r_dual.zero_()
        self.r_compl.zero_()
        if use_acceleration:
            self.r_dx.zero_()
            self.r_dy.zero_()
            self.r_dz.zero_()


class PADMMSolution:
    """
    An interface container to the PADMM solver solution arrays.

    Attributes:
        lambdas: The constraint reactions (i.e. impulses) solution array.
            Shape of ``(sum_of_max_total_cts,)``.
        v_plus: The post-event constraint-space velocities solution array.
            Shape of ``(sum_of_max_total_cts,)``.
    """

    def __init__(self, size: SizeKamino | None = None):
        """
        Initializes the PADMM solution container.

        If a model size is provided, allocates the solution arrays accordingly.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
        """

        self.lambdas: wp.array[wp.float32] | None = None
        """
        The constraint reactions (i.e. impulses) solution array.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.v_plus: wp.array[wp.float32] | None = None
        """
        The post-event constraint-space velocities solution array.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        # Perform memory allocations if model size is specified
        if size is not None:
            self.finalize(size)

    def finalize(self, size: SizeKamino):
        """
        Allocates the PADMM solution arrays based on the model size.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
        """
        self.lambdas = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.v_plus = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)

    def zero(self):
        """
        Resets all PADMM solution arrays to zeros.
        """
        self.lambdas.zero_()
        self.v_plus.zero_()


class PADMMInfo:
    """
    An interface container to hold the PADMM solver convergence info arrays.

    Attributes:
        lambdas: The constraint reactions (i.e. impulses) of each world.
            Shape of ``(sum_of_max_total_cts,)``.
        v_plus: The post-event constraint-space velocities of each world.
            This is computed using the current solution as: `v_plus := v_f + D @ lambdas`.
            Shape of ``(sum_of_max_total_cts,)``.
        v_aug: The post-event augmented constraint-space velocities of each world.
            This is computed using the current solution as: `v_aug := v_plus + s`.
            Shape of ``(sum_of_max_total_cts,)``.
        s: The De Saxce correction velocities of each world.
            This is computed using the current solution as: `s := Gamma(v_plus)`.
            Shape of ``(sum_of_max_total_cts,)``.
        offsets: The residuals index offset of each world.
            Shape of ``(num_worlds,)``.
        num_restarts: History of the number of acceleration restarts performed for each world.
            Shape of ``(num_worlds * max_iters,)``.
        num_rho_updates: History of the number of penalty updates performed for each world.
            Shape of ``(num_worlds * max_iters,)``.
        a: History of PADMM acceleration variables.
            Shape of ``(num_worlds * max_iters,)``.
        norm_s: History of the L2 norm of De Saxce correction velocities.
            Shape of ``(num_worlds * max_iters,)``.
        norm_x: History of the L2 norm of primal variables.
            Shape of ``(num_worlds * max_iters,)``.
        norm_y: History of the L2 norm of slack variables.
            Shape of ``(num_worlds * max_iters,)``.
        norm_z: History of the L2 norm of dual variables.
            Shape of ``(num_worlds * max_iters,)``.
        f_ccp: History of CCP optimization objectives.
            Shape of ``(num_worlds * max_iters,)``.
        f_ncp: History of the NCP optimization objectives.
            Shape of ``(num_worlds * max_iters,)``.
        r_dx: History of the total primal iterate residual.
            Shape of ``(num_worlds * max_iters,)``.
        r_dy: History of the total slack iterate residual.
            Shape of ``(num_worlds * max_iters,)``.
        r_dz: History of the total dual iterate residual.
            Shape of ``(num_worlds * max_iters,)``.
        r_primal: History of the total primal residual.
            Shape of ``(num_worlds * max_iters,)``.
        r_dual: History of the total dual residual.
            Shape of ``(num_worlds * max_iters,)``.
        r_compl: History of the total complementarity residual.
            Shape of ``(num_worlds * max_iters,)``.
        r_comb: History of the total combined primal-dual residual.
            Shape of ``(num_worlds * max_iters,)``.
        r_comb_ratio: History of the combined primal-dual residual ratio.
            Shape of ``(num_worlds * max_iters,)``.
        r_ncp_primal: History of NCP primal residuals.
            Shape of ``(num_worlds * max_iters,)``.
        r_ncp_dual: History of NCP dual residuals.
            Shape of ``(num_worlds * max_iters,)``.
        r_ncp_compl: History of NCP complementarity residuals.
            Shape of ``(num_worlds * max_iters,)``.
        r_ncp_natmap: History of NCP natural-map residuals.
            Shape of ``(num_worlds * max_iters,)``.

    Notes:
    - The length of the arrays is determined by the maximum number of iterations
    and is filled up to the number of iterations performed by the solver on each
    solve. This allows for post-solve analysis of the convergence behavior.
    - This has a significant impact on solver performance and memory usage, so it
    is recommended to only enable this for testing and debugging purposes.
    """

    def __init__(
        self,
        size: SizeKamino | None = None,
        max_iters: int | None = None,
        use_acceleration: bool = False,
    ):
        """
        Initializes the PADMM solver info container.

        If a model size is provided, allocates the solution arrays accordingly.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
            max_iters: The maximum number of iterations for which to allocate convergence data.
        """

        self.lambdas: wp.array[wp.float32] | None = None
        """
        The constraint reactions (i.e. impulses) of each world.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.v_plus: wp.array[wp.float32] | None = None
        """
        The post-event constraint-space velocities of each world.
        This is computed using the current solution as: `v_plus := v_f + D @ lambdas`.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.v_aug: wp.array[wp.float32] | None = None
        """
        The post-event augmented constraint-space velocities of each world.
        This is computed using the current solution as: `v_aug := v_plus + s`.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.s: wp.array[wp.float32] | None = None
        """
        The De Saxce correction velocities of each world.
        This is computed using the current solution as: `s := Gamma(v_plus)`.
        Shape of ``(sum_of_max_total_cts,)``.
        """

        self.offsets: wp.array[wp.int32] | None = None
        """
        The residuals index offset of each world.
        Shape of ``(num_worlds,)``.
        """

        self.num_restarts: wp.array[wp.int32] | None = None
        """
        History of the number of acceleration restarts performed for each world.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.num_rho_updates: wp.array[wp.int32] | None = None
        """
        History of the number of penalty updates performed for each world.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.a: wp.array[wp.float32] | None = None
        """
        History of PADMM acceleration variables.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.norm_s: wp.array[wp.float32] | None = None
        """
        History of the L2 norm of De Saxce correction velocities.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.norm_x: wp.array[wp.float32] | None = None
        """
        History of the L2 norm of primal variables.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.norm_y: wp.array[wp.float32] | None = None
        """
        History of the L2 norm of slack variables.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.norm_z: wp.array[wp.float32] | None = None
        """
        History of the L2 norm of dual variables.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.f_ccp: wp.array[wp.float32] | None = None
        """
        History of CCP optimization objectives.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.f_ncp: wp.array[wp.float32] | None = None
        """
        History of the NCP optimization objectives.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_dx: wp.array[wp.float32] | None = None
        """
        History of the total primal iterate residual.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_dy: wp.array[wp.float32] | None = None
        """
        History of the total slack iterate residual.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_dz: wp.array[wp.float32] | None = None
        """
        History of the total dual iterate residual.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_primal: wp.array[wp.float32] | None = None
        """
        History of PADMM primal residuals.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_dual: wp.array[wp.float32] | None = None
        """
        History of PADMM dual residuals.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_compl: wp.array[wp.float32] | None = None
        """
        History of PADMM complementarity residuals.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_pd: wp.array[wp.float32] | None = None
        """
        History of PADMM primal-dual residual ratio.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_dp: wp.array[wp.float32] | None = None
        """
        History of PADMM dual-primal residual ratio.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_comb: wp.array[wp.float32] | None = None
        """
        History of PADMM combined residuals.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_comb_ratio: wp.array[wp.float32] | None = None
        """
        History of PADMM combined residuals ratio.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_ncp_primal: wp.array[wp.float32] | None = None
        """
        History of NCP primal residuals.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_ncp_dual: wp.array[wp.float32] | None = None
        """
        History of NCP dual residuals.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_ncp_compl: wp.array[wp.float32] | None = None
        """
        History of NCP complementarity residuals.
        Shape of ``(num_worlds * max_iters,)``.
        """

        self.r_ncp_natmap: wp.array[wp.float32] | None = None
        """
        History of NCP natural-map residuals.
        Shape of ``(num_worlds * max_iters,)``.
        """

        # Perform memory allocations if model size is specified
        if size is not None:
            self.finalize(size=size, max_iters=max_iters, use_acceleration=use_acceleration)

    def finalize(self, size: SizeKamino, max_iters: int, use_acceleration: bool = False):
        """
        Allocates the PADMM solver info arrays based on the model size and maximum number of iterations.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
            max_iters: The maximum number of iterations for which to allocate convergence data.

        Raises:
            ValueError: If either ``size.num_worlds`` or ``max_iters`` are not positive integers.
        """

        # Ensure num_worlds is valid
        if not isinstance(size.num_worlds, int) or size.num_worlds <= 0:
            raise ValueError("num_worlds must be a positive integer specifying the number of worlds.")

        # Ensure max_iters is valid
        if not isinstance(max_iters, int) or max_iters <= 0:
            raise ValueError("max_iters must be a positive integer specifying the maximum number of iterations.")

        # Allocate intermediate arrays
        self.lambdas = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.v_plus = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.v_aug = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)
        self.s = wp.zeros(size.sum_of_max_total_cts, dtype=wp.float32)

        # Compute the index offsets for the info of each world
        maxsize = max_iters * size.num_worlds
        offsets = [max_iters * i for i in range(size.num_worlds)]

        # Allocate the on-device solver info data arrays
        self.offsets = to_warp_int32_array(offsets)
        self.num_rho_updates = wp.zeros(maxsize, dtype=wp.int32)
        self.norm_s = wp.zeros(maxsize, dtype=wp.float32)
        self.norm_x = wp.zeros(maxsize, dtype=wp.float32)
        self.norm_y = wp.zeros(maxsize, dtype=wp.float32)
        self.norm_z = wp.zeros(maxsize, dtype=wp.float32)
        self.r_dx = wp.zeros(maxsize, dtype=wp.float32)
        self.r_dy = wp.zeros(maxsize, dtype=wp.float32)
        self.r_dz = wp.zeros(maxsize, dtype=wp.float32)
        self.f_ccp = wp.zeros(maxsize, dtype=wp.float32)
        self.f_ncp = wp.zeros(maxsize, dtype=wp.float32)
        self.r_primal = wp.zeros(maxsize, dtype=wp.float32)
        self.r_dual = wp.zeros(maxsize, dtype=wp.float32)
        self.r_compl = wp.zeros(maxsize, dtype=wp.float32)
        self.r_pd = wp.zeros(maxsize, dtype=wp.float32)
        self.r_dp = wp.zeros(maxsize, dtype=wp.float32)
        self.r_ncp_primal = wp.zeros(maxsize, dtype=wp.float32)
        self.r_ncp_dual = wp.zeros(maxsize, dtype=wp.float32)
        self.r_ncp_compl = wp.zeros(maxsize, dtype=wp.float32)
        self.r_ncp_natmap = wp.zeros(maxsize, dtype=wp.float32)
        if use_acceleration:
            self.num_restarts = wp.zeros(maxsize, dtype=wp.int32)
            self.a = wp.zeros(maxsize, dtype=wp.float32)
            self.r_comb = wp.zeros(maxsize, dtype=wp.float32)
            self.r_comb_ratio = wp.zeros(maxsize, dtype=wp.float32)

    def zero(self, use_acceleration: bool = False):
        """
        Resets all PADMM solver info arrays to zeros.
        """
        self.lambdas.zero_()
        self.v_plus.zero_()
        self.v_aug.zero_()
        self.s.zero_()
        self.num_rho_updates.zero_()
        self.norm_s.zero_()
        self.norm_x.zero_()
        self.norm_y.zero_()
        self.norm_z.zero_()
        self.f_ccp.zero_()
        self.f_ncp.zero_()
        self.r_dx.zero_()
        self.r_dy.zero_()
        self.r_dz.zero_()
        self.r_primal.zero_()
        self.r_dual.zero_()
        self.r_compl.zero_()
        self.r_pd.zero_()
        self.r_dp.zero_()
        self.r_ncp_primal.zero_()
        self.r_ncp_dual.zero_()
        self.r_ncp_compl.zero_()
        self.r_ncp_natmap.zero_()
        if use_acceleration:
            self.num_restarts.zero_()
            self.a.zero_()
            self.r_comb.zero_()
            self.r_comb_ratio.zero_()


class PADMMData:
    """
    A high-level container to manage all internal PADMM solver data.

    Attributes:
        config: Array of per-world solver configurations,
            of type :class:`PADMMConfigStruct` and shape ``(num_worlds,)``.
            Each element is the on-device version of :class:`PADMMConfig`.
        status: Array of per-world solver status,
            of type :class:`PADMMStatus` and shape ``(num_worlds,)``.
            Each element holds the status of the solver on
            solving the dynamics of the corresponding world.
        penalty: Array of per-world ALM penalty states,
            of type :class:`PADMMPenalty` and shape ``(num_worlds,)``.
            Each element holds the current and previous ALM penalty `rho`,
            as well as additional meta-data regarding it's adaptation.
        state: A container holding the PADMM state variable arrays.
        residuals: A container holding the PADMM residuals arrays.
        solution: A container holding the PADMM solution arrays.
        info: A container holding the PADMM solver convergence info arrays.
    """

    def __init__(
        self,
        size: SizeKamino | None = None,
        max_iters: int = 0,
        use_acceleration: bool = False,
        collect_info: bool = False,
        device: wp.DeviceLike = None,
    ):
        """
        Initializes a PADMM solver data container.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
            max_iters: The maximum number of iterations for which to allocate convergence data.
            collect_info: Set to `True` to allocate data for reporting solver convergence info.
            device: The target Warp device on which all data will be allocated.

        Raises:
            ValueError: If either ``size.num_worlds`` or ``max_iters`` are not positive integers.
        """

        self.config: wp.array[PADMMConfigStruct] | None = None
        """
        Array of on-device PADMM solver configs.
        Shape is (num_worlds,).
        """

        self.status: wp.array[PADMMStatus] | None = None
        """
        Array of PADMM solver status.
        Shape is (num_worlds,).
        """

        self.penalty: wp.array[PADMMPenalty] | None = None
        """
        Array of PADMM solver penalty parameters.
        Shape is (num_worlds,).
        """

        self.state: PADMMState | None = None
        """The PADMM internal solver state container."""

        self.residuals: PADMMResiduals | None = None
        """The PADMM residuals container."""

        self.solution: PADMMSolution | None = None
        """The PADMM solution container."""

        self.info: PADMMInfo | None = None
        """The (optional) PADMM solver info container."""

        self.linear_solver_atol: wp.array[wp.float32] | None = None
        """
        Per-world absolute tolerance array for the iterative linear solver.
        Shape is (num_worlds,).
        """

        # Perform memory allocations if model size is specified
        if size is not None:
            self.finalize(
                size=size,
                max_iters=max_iters,
                use_acceleration=use_acceleration,
                collect_info=collect_info,
                device=device,
            )

    def finalize(
        self,
        size: SizeKamino,
        max_iters: int = 0,
        use_acceleration: bool = False,
        collect_info: bool = False,
        device: wp.DeviceLike = None,
    ):
        """
        Allocates the PADMM solver data based on the model size and maximum number of iterations.

        Args:
            size: The model-size utility container holding the dimensionality of the model.
            max_iters: The maximum number of iterations for which to allocate convergence data.
            collect_info: Set to `True` to allocate data for reporting solver convergence info.
            device: The target Warp device on which all data will be allocated.

        Raises:
            ValueError: If either ``size.num_worlds`` or ``max_iters`` are not positive integers.
        """
        with wp.ScopedDevice(device):
            self.config = wp.zeros(shape=(size.num_worlds,), dtype=PADMMConfigStruct)
            self.status = wp.zeros(shape=(size.num_worlds,), dtype=PADMMStatus)
            self.penalty = wp.zeros(shape=(size.num_worlds,), dtype=PADMMPenalty)
            self.state = PADMMState(size, use_acceleration)
            self.residuals = PADMMResiduals(size, use_acceleration)
            self.solution = PADMMSolution(size)
            self.linear_solver_atol = wp.full(
                shape=(size.num_worlds,), value=np.finfo(np.float32).eps, dtype=wp.float32
            )
            if collect_info and max_iters > 0:
                self.info = PADMMInfo(size, max_iters, use_acceleration)


###
# Utilities
###


def convert_config_to_struct(config: PADMMSolverConfig) -> PADMMConfigStruct:
    """
    Converts the host-side config to the corresponding device-side object.

    Returns:
        The solver config as a warp struct.
    """
    config_struct = PADMMConfigStruct()
    config_struct.primal_tolerance = config.primal_tolerance
    config_struct.dual_tolerance = config.dual_tolerance
    config_struct.compl_tolerance = config.compl_tolerance
    config_struct.restart_tolerance = config.restart_tolerance
    config_struct.eta = config.eta
    config_struct.rho_0 = config.rho_0
    config_struct.rho_min = config.rho_min
    config_struct.a_0 = config.a_0
    config_struct.alpha = config.alpha
    config_struct.tau = config.tau
    config_struct.max_iterations = config.max_iterations
    config_struct.penalty_update_freq = config.penalty_update_freq
    config_struct.penalty_update_method = PADMMPenaltyUpdate.from_string(config.penalty_update_method)
    config_struct.linear_solver_tolerance = config.linear_solver_tolerance
    config_struct.linear_solver_tolerance_ratio = config.linear_solver_tolerance_ratio
    return config_struct
