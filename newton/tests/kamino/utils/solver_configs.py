# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Common SolverKamino configuration presets for integration tests."""

from typing import Any

import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino.solver_kamino import SolverKamino

__all__ = [
    "KAMINO_CONFIGS",
    "PADMM_CONFIG_NAMES",
    "make_dvi_dense_config",
    "make_dvi_sparse_config",
    "make_padmm_dense_config",
    "make_padmm_sparse_config",
]


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


def make_dvi_dense_config(**kwargs: Any) -> SolverKamino.Config:
    """Create a dense DVI configuration without collision detection."""
    return SolverKamino.Config(
        dynamics_solver="dvi",
        use_fk_solver=False,
        sparse_jacobian=False,
        sparse_dynamics=False,
        use_collision_detector=False,
        **kwargs,
    )


def make_dvi_sparse_config(**kwargs: Any) -> SolverKamino.Config:
    """Create a sparse DVI configuration without collision detection."""
    return SolverKamino.Config(
        dynamics_solver="dvi",
        use_fk_solver=False,
        sparse_jacobian=True,
        sparse_dynamics=True,
        use_collision_detector=False,
        **kwargs,
    )


KAMINO_CONFIGS = (
    ("dense", make_padmm_dense_config),
    ("sparse", make_padmm_sparse_config),
    ("dvi_dense", make_dvi_dense_config),
    ("dvi_sparse", make_dvi_sparse_config),
)

# DVI has no ``use_acceleration`` knob, so only PADMM configs vary over it.
PADMM_CONFIG_NAMES = ("dense", "sparse")
