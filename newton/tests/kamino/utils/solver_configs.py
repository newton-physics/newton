# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Common SolverKamino configuration presets for integration tests."""

from typing import Any, Literal

import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino.solver_kamino import SolverKamino

__all__ = [
    "DECOUPLED_CONFIGS",
    "DECOUPLED_CONFIG_NAMES",
    "KAMINO_CONFIGS",
    "PADMM_CONFIGS",
    "PADMM_CONFIG_NAMES",
    "make_padmm_dense_config",
    "make_padmm_sparse_config",
    "make_padmm_sparse_fused_config",
    "make_scc_dense_config",
    "make_scc_sparse_config",
    "make_single_iteration_config",
    "solve_variants",
]


def make_single_iteration_config(
    config_factory,
    *,
    warmstart_mode: Literal["none", "containers"],
    use_acceleration: bool,
) -> SolverKamino.Config:
    """Create a configuration that exposes the warm-start benefit after one iteration."""
    config = config_factory(compute_solution_metrics=True)
    if config.dynamics_solver == "padmm":
        config.padmm = kamino_config.PADMMSolverConfig(
            max_iterations=1,
            primal_tolerance=0.0,
            dual_tolerance=0.0,
            compl_tolerance=0.0,
            use_acceleration=use_acceleration,
            warmstart_mode=warmstart_mode,
        )
    else:
        config.scc = kamino_config.SCCSolverConfig(
            max_alternating_iterations=1,
            inequality_sweeps_per_iteration=1,
            tolerance=0.0,
            warmstart_mode=warmstart_mode,
        )
    return config


def make_padmm_dense_config(**kwargs: Any) -> SolverKamino.Config:
    """Create a dense PADMM configuration without collision detection."""
    return SolverKamino.Config(
        dynamics_solver="padmm",
        use_fk_solver=False,
        sparse_jacobian=False,
        sparse_dynamics=False,
        use_collision_detector=False,
        **kwargs,
    )


def make_padmm_sparse_config(**kwargs: Any) -> SolverKamino.Config:
    """Create a sparse PADMM configuration without collision detection."""
    return SolverKamino.Config(
        dynamics_solver="padmm",
        use_fk_solver=False,
        sparse_jacobian=True,
        sparse_dynamics=True,
        use_collision_detector=False,
        dynamics=kamino_config.ConstrainedDynamicsConfig(linear_solver_type="CR"),
        **kwargs,
    )


def make_padmm_sparse_fused_config(**kwargs: Any) -> SolverKamino.Config:
    """Create a sparse PADMM configuration with the fused conjugate residual solver (CUDA only)."""
    return SolverKamino.Config(
        dynamics_solver="padmm",
        use_fk_solver=False,
        sparse_jacobian=True,
        sparse_dynamics=True,
        use_collision_detector=False,
        dynamics=kamino_config.ConstrainedDynamicsConfig(linear_solver_type="CRF"),
        **kwargs,
    )


def make_scc_dense_config(**kwargs: Any) -> SolverKamino.Config:
    """Create a dense SCC configuration without collision detection."""
    return SolverKamino.Config(
        dynamics_solver="scc",
        use_fk_solver=False,
        sparse_jacobian=False,
        sparse_dynamics=False,
        use_collision_detector=False,
        **kwargs,
    )


def make_scc_sparse_config(**kwargs: Any) -> SolverKamino.Config:
    """Create a sparse SCC configuration without collision detection."""
    return SolverKamino.Config(
        dynamics_solver="scc",
        use_fk_solver=False,
        sparse_jacobian=True,
        sparse_dynamics=True,
        use_collision_detector=False,
        **kwargs,
    )


KAMINO_CONFIGS = (
    ("dense", make_padmm_dense_config),
    ("sparse", make_padmm_sparse_config),
    ("scc_dense", make_scc_dense_config),
    ("scc_sparse", make_scc_sparse_config),
)

# SCC has no ``use_acceleration`` knob, so only PADMM configs vary over it.
PADMM_CONFIG_NAMES = ("dense", "sparse")

# The PADMM entries of KAMINO_CONFIGS
PADMM_CONFIGS = tuple(entry for entry in KAMINO_CONFIGS if entry[0] in PADMM_CONFIG_NAMES)

# The entries of KAMINO_CONFIGS that decouple joint friction and effort rows from the joint
# dynamics rows of the same DoFs (see ``DynamicJointDecoupling``). Sparse SCC does not.
DECOUPLED_CONFIG_NAMES = (*PADMM_CONFIG_NAMES, "scc_dense")
DECOUPLED_CONFIGS = tuple(entry for entry in KAMINO_CONFIGS if entry[0] in DECOUPLED_CONFIG_NAMES)


def solve_variants(config_factory) -> list[tuple[dict[str, Any], SolverKamino.Config]]:
    """Return a configuration for each way its dynamics solver iterates, with a label for sub-tests.

    PADMM runs with and without acceleration. SCC runs with and without the Schur complement of
    the bilateral rows, i.e. its two ways of coupling the joint rows with the bounded rows.
    """
    variants = []
    for option in (True, False):
        config = config_factory()
        if config.dynamics_solver == "padmm":
            config.padmm.use_acceleration = option
            variants.append(({"use_acceleration": option}, config))
        else:
            config.scc.use_schur_complement = option
            variants.append(({"use_schur_complement": option}, config))
    return variants
