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
        default=None,
        help="Select LDL with minimum edge creation (meca) or Reverse Cuthill-McKee (rcm) ordering.",
    )


def configure_ldl(config: SolverKamino.Config | None, args: argparse.Namespace | None) -> None:
    """Apply explicit LDL options, rejecting conflicting or unsupported backends.

    Pass ``None`` for examples using another physics solver to reject Kamino-only
    options before loading assets. Unspecified options preserve the configuration,
    including an existing LDL ordering and numerical settings.
    """
    ordering = getattr(args, "ldl_ordering", None)
    requested = getattr(args, "ldl", False) or ordering is not None
    if config is None:
        if requested or getattr(args, "bilateral_solver_type", None) is not None:
            raise ValueError("LDL and bilateral solver options require the Kamino solver (--solver kamino).")
        return

    if config.dynamics_solver == "dvi":
        subconfig, prefix = config.dvi, "bilateral"
    elif config.dynamics_solver == "padmm" and not config.sparse_dynamics:
        subconfig, prefix = config.dynamics, "linear"
    else:
        if requested or config.dynamics.linear_solver_type == "LDL":
            raise ValueError("The LDL example option requires DVI or dense PADMM dynamics.")
        return

    explicit_backend = getattr(args, f"{prefix}_solver_type", None)
    if requested and explicit_backend not in (None, "LDL"):
        raise ValueError(f"LDL options conflict with --{prefix}-solver-type {explicit_backend}.")
    if requested and getattr(subconfig, f"{prefix}_solver_type") != "LDL":
        setattr(subconfig, f"{prefix}_solver_type", "LDL")
        setattr(subconfig, f"{prefix}_solver_kwargs", {})
    if getattr(subconfig, f"{prefix}_solver_type") == "LDL":
        kwargs = getattr(subconfig, f"{prefix}_solver_kwargs")
        if ordering is not None:
            kwargs["ordering"] = ordering
        else:
            kwargs.setdefault("ordering", "meca")
