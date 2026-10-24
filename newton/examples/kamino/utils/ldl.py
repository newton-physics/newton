# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared LDL options for the Kamino examples."""

import argparse

from newton.solvers import SolverKamino


def add_ldl_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the LDL backend and its block ordering in example CLIs."""
    parser.add_argument(
        "--ldl",
        action="store_true",
        help="Use LDL for the DVI bilateral solve or the dense PADMM linear solve.",
    )
    parser.add_argument(
        "--ldl-ordering",
        choices=("meca", "rcm"),
        default="meca",
        help="LDL block ordering: minimum edge creation (default) or Reverse Cuthill-McKee.",
    )


def configure_ldl(config: SolverKamino.Config, args: argparse.Namespace | None) -> None:
    """Apply LDL options before constructing the solver and capturing its kernels."""
    requested = getattr(args, "ldl", False)
    ordering = getattr(args, "ldl_ordering", "meca")
    if config.dynamics_solver == "dvi":
        if requested:
            config.dvi.bilateral_solver_type = "LDL"
            config.dvi.bilateral_solver_kwargs = {}
        if config.dvi.bilateral_solver_type == "LDL":
            config.dvi.bilateral_solver_kwargs["ordering"] = ordering
    elif config.dynamics_solver == "padmm" and not config.sparse_dynamics:
        if requested:
            config.dynamics.linear_solver_type = "LDL"
            config.dynamics.linear_solver_kwargs = {}
        if config.dynamics.linear_solver_type == "LDL":
            config.dynamics.linear_solver_kwargs["ordering"] = ordering
    elif requested:
        raise ValueError("The LDL example option requires DVI or dense PADMM dynamics.")
