# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...linalg.linear import LinearSolverTypeToName
from ...solver_kamino_impl import SolverKaminoImpl
from .problems import ProblemDimensions

###
# Module interface
###

__all__ = [
    "render_problem_dimensions_table",
    "render_solver_configs_table",
    "render_subcolumn_metrics_table",
    "render_subcolumn_table",
    "render_table",
]


###
# Internals
###


def _add_table_column_group(
    table,
    group_name: str,
    subcol_headers: list[str],
    justify: str = "left",
    color: str | None = None,
) -> None:
    # Attempt to import rich first, and warn user
    # if the necessary package is not installed
    try:
        from rich.text import Text  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The `rich` package is required for rendering tables. Install it with: pip install rich"
        ) from e

    for i, sub in enumerate(subcol_headers):
        header = Text(justify="left")
        if i == 0:
            header.append(group_name, style=f"bold {color}" if color else "bold")
        header.append("\n")
        header.append(sub, style=f"dim {color}" if color else "dim")
        col_justify = "center" if justify == "center" else justify
        table.add_column(header=header, justify=col_justify, no_wrap=True)


def _render_table_to_console_and_file(
    table,
    path: str | None = None,
    to_console: bool = False,
    max_width: int | None = None,
):
    # Attempt to import rich first, and warn user
    # if the necessary package is not installed
    try:
        from rich.console import Console  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The `rich` package is required for rendering tables. Install it with: pip install rich"
        ) from e

    if path is not None:
        path_dir = os.path.dirname(path)
        if path_dir and not os.path.exists(path_dir):
            raise ValueError(
                f"Directory for path '{path}' does not exist. Please create the directory before exporting the table."
            )
        with open(path, "w", encoding="utf-8") as f:
            console = Console(file=f, width=9999999)
            console.print(table, crop=False)
    if to_console:
        console = Console(width=max_width)
        console.rule()
        console.print(table, crop=False)
        console.rule()


def _format_cell(x, fmt):
    if callable(fmt):
        return fmt(x)
    if isinstance(fmt, str):
        try:
            return format(x, fmt)
        except Exception:
            return str(x)
    if isinstance(x, (bool, np.bool_)):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.4g}"
    return str(x)


@dataclass
class ColumnGroup:
    header: str
    subheaders: list[str]
    subfmt: list[str | Callable | None] | None = None
    justify: str = "left"
    color: str | None = None


###
# Renderers
###


def render_table(
    title: str,
    col_headers: list[str],
    col_colors: list[str],
    col_fmt: list[str | Callable | None] | None,
    data: list[Any] | np.ndarray,
    transposed: bool = False,
    max_width: int | None = None,
    path: str | None = None,
    to_console: bool = False,
):
    # Attempt to import rich first, and warn user
    # if the necessary package is not installed
    try:
        from rich import box  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The `rich` package is required for rendering tables. Install it with: pip install rich"
        ) from e

    # Initialize the table with appropriate columns and styling
    table = Table(
        title=title,
        show_header=True,
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        pad_edge=True,
    )

    # Add groups of columns based on the specified groups to include in the table
    for header, color in zip(col_headers, col_colors, strict=True):
        cheader = Text(justify="left")
        cheader.append(header, style=f"bold {color}" if color else "bold")
        table.add_column(header=cheader, justify="left", no_wrap=True)

    # Add data rows
    # If the data is transposed we need to extract from per-column format
    if transposed:
        ncols = len(data)
        nrows = len(data[0])
        for r in range(nrows):
            rowdata = []
            for c in range(ncols):
                rowdata.append(_format_cell(data[c][r], col_fmt[c] if col_fmt else None))
            table.add_row(*rowdata)
    else:
        for row in data:
            rowdata = []
            for rc in range(len(col_headers)):
                rowdata.append(_format_cell(row[rc], col_fmt[rc] if col_fmt else None))
            table.add_row(*rowdata)

    # Render the table to the console and/or save to file
    _render_table_to_console_and_file(table, path=path, to_console=to_console, max_width=max_width)


def render_subcolumn_table(
    title: str,
    cols: list[ColumnGroup],
    rows: list[list[Any]] | np.ndarray,
    max_width: int | None = None,
    path: str | None = None,
    to_console: bool = False,
):
    # Attempt to import rich first, and warn user
    # if the necessary package is not installed
    try:
        from rich import box  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The `rich` package is required for rendering tables. Install it with: pip install rich"
        ) from e

    # Initialize the table with appropriate columns and styling
    table = Table(
        title=title,
        show_header=True,
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        pad_edge=True,
    )

    # Add groups of columns based on the specified groups to include in the table
    for cgroup in cols:
        _add_table_column_group(table, cgroup.header, cgroup.subheaders, color=cgroup.color, justify=cgroup.justify)

    # Add data rows
    for row in rows:
        rowdata = []
        for rc in range(len(cols)):
            for rsc in range(len(cols[rc].subheaders)):
                rowdata.append(_format_cell(row[rc][rsc], cols[rc].subfmt[rsc] if cols[rc].subfmt else None))
        table.add_row(*rowdata)

    # Render the table to the console and/or save to file
    _render_table_to_console_and_file(table, path=path, to_console=to_console, max_width=max_width)


def render_subcolumn_metrics_table(
    title: str,
    row_header: str,
    col_titles: list[str],
    row_titles: list[str],
    subcol_titles: list[str],
    subcol_data: list[np.ndarray],
    subcol_formats: list[str | Callable] | None = None,
    max_width: int | None = None,
    path: str | None = None,
    to_console: bool = False,
):
    # Attempt to import rich first, and warn user
    # if the necessary package is not installed
    try:
        from rich import box  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The `rich` package is required for rendering tables. Install it with: pip install rich"
        ) from e

    n_metrics = len(subcol_data)
    n_problems = len(col_titles)
    n_solvers = len(row_titles)

    if len(subcol_titles) != n_metrics:
        raise ValueError("subcol_titles length must match number of metric arrays")

    for i, arr in enumerate(subcol_data):
        if arr.shape != (n_problems, n_solvers):
            raise ValueError(f"subcol_data[{i}] has shape {arr.shape}, expected {(n_problems, n_solvers)}")

    if subcol_formats is None:
        subcol_formats = [None] * n_metrics
    if len(subcol_formats) != n_metrics:
        raise ValueError("subcol_formats length must match number of metrics")

    def format_value(x, fmt):
        if callable(fmt):
            return fmt(x)
        if isinstance(fmt, str):
            try:
                return format(x, fmt)
            except Exception:
                return str(x)
        if isinstance(x, (int, np.integer)):
            return f"{int(x)}"
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.4g}"
        return str(x)

    table = Table(
        title=title,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        pad_edge=True,
    )

    # Solver column
    table.add_column(row_header, style="bold", no_wrap=True, justify="left")

    # Metric columns: problem shown only on first subcolumn in each block
    # Header is a Text object with justify="left" (works on rich 14.0.0)
    for p_name in col_titles:
        for m_idx, m_name in enumerate(subcol_titles):
            top = p_name if m_idx == 0 else ""

            header = Text(justify="left")
            if top:
                header.append(top, style="bold")
            header.append("\n")
            header.append(m_name, style="dim")

            table.add_column(
                header=header,
                justify="right",  # numeric cells
                no_wrap=True,
            )

    # Data rows
    for s_idx, solver in enumerate(row_titles):
        row = [solver]
        for p_idx in range(n_problems):
            for m_idx in range(n_metrics):
                value = subcol_data[m_idx][p_idx, s_idx]
                row.append(format_value(value, subcol_formats[m_idx]))
        table.add_row(*row)

    # Render the table to the console and/or save to file
    _render_table_to_console_and_file(table, path=path, to_console=to_console, max_width=max_width)


###
# Solver Configs
###


def render_solver_configs_table(
    configs: dict[str, SolverKaminoImpl.Config],
    path: str | None = None,
    groups: list[str] | None = None,
    to_console: bool = False,
):
    """
    Renders a rich table summarizing the solver configurations.

    Args:
        configs (dict[str, SolverKaminoImpl.Config]):
            A dictionary mapping configuration names to SolverKaminoImpl.Config objects.
        path (str, optional):
            The file path to save the rendered table as a text file. If None, the table is not saved to a file.
        groups (list[str], optional):
            A list of groups to include in the table. If None, "sparse", "linear" and "padmm" are used.\n
            Supported groups include:
            - "cts": Constraint parameters (alpha, beta, gamma, delta, preconditioning)
            - "sparse": Sparse representation settings (sparse, sparse_jacobian)
            - "linear": Linear solver settings (type, kwargs)
            - "padmm": PADMM settings (max_iterations, primal_tol, dual_tol, etc)
            - "warmstart": Warmstarting settings (mode, contact_method)
        to_console (bool, optional):
            If True, also prints the table to the console.

    Raises:
        ValueError:
            If the configs dictionary is empty or if any of the configuration objects are missing required attributes.
        IOError:
            If there is an error writing the table to the specified file path.
    """
    # Attempt to import rich first, and warn user
    # if the necessary package is not installed
    try:
        from rich import box  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The `rich` package is required for rendering tables. Install it with: pip install rich"
        ) from e

    # Initialize the table with appropriate columns and styling
    table = Table(
        title="Solver Configurations Summary",
        show_header=True,
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        pad_edge=True,
    )

    # If no groups are specified, default to showing sparsity, linear solver and PADMM settings
    if groups is None:
        groups = ["sparse", "linear", "padmm"]

    # Add the first column for configuration names
    _add_table_column_group(table, "Solver Configuration", ["Name"], color="white", justify="left")

    # Add groups of columns based on the specified groups to include in the table
    if "cts" in groups:
        _add_table_column_group(table, "Constraints", ["alpha", "beta", "gamma", "delta", "precond"], color="green")
    if "sparse" in groups:
        _add_table_column_group(table, "Representation", ["sparse_jacobian", "sparse_dynamics"], color="yellow")
    if "linear" in groups:
        _add_table_column_group(table, "Linear Solver", ["type", "kwargs"], color="magenta")
    if "padmm" in groups:
        _add_table_column_group(
            table,
            "PADMM",
            [
                "max_iterations",
                "primal_tol",
                "dual_tol",
                "compl_tol",
                "restart_tol",
                "eta",
                "rho_0",
                "rho_min",
                "penalty_update",
                "penalty_freq",
                "accel",
            ],
            color="cyan",
        )
    if "warmstart" in groups:
        _add_table_column_group(table, "Warmstarting", ["mode", "contact_method"], color="blue")

    # Add rows for each configuration
    for name, cfg in configs.items():
        cfg_row = []
        if "cts" in groups:
            cfg_row.extend(
                [
                    f"{cfg.constraints.alpha}",
                    f"{cfg.constraints.beta}",
                    f"{cfg.constraints.gamma}",
                    f"{cfg.constraints.delta}",
                    str(cfg.dynamics.preconditioning),
                ]
            )
        if "sparse" in groups:
            cfg_row.extend([str(cfg.sparse_jacobian), str(cfg.sparse_dynamics)])
        if "linear" in groups:
            cfg_row.extend(
                [
                    str(LinearSolverTypeToName[cfg.dynamics.linear_solver_type]),
                    str(cfg.dynamics.linear_solver_kwargs),
                ]
            )
        if "padmm" in groups:
            cfg_row.extend(
                [
                    str(cfg.padmm.max_iterations),
                    f"{cfg.padmm.primal_tolerance:.0e}",
                    f"{cfg.padmm.dual_tolerance:.0e}",
                    f"{cfg.padmm.compl_tolerance:.0e}",
                    f"{cfg.padmm.restart_tolerance:.0e}",
                    f"{cfg.padmm.eta:.0e}",
                    f"{cfg.padmm.rho_0}",
                    f"{cfg.padmm.rho_min}",
                    cfg.padmm.penalty_update_method,
                    str(cfg.padmm.penalty_update_freq),
                    str(cfg.padmm.use_acceleration),
                ]
            )
        if "warmstart" in groups:
            cfg_row.extend([cfg.padmm.warmstart_mode, cfg.padmm.contact_warmstart_method])
        table.add_row(name, *cfg_row)

    # Render the table to the console and/or save to file
    _render_table_to_console_and_file(table, path=path, to_console=to_console, max_width=None)


###
# Problem Dimensions
###


def render_problem_dimensions_table(
    problem_dims: dict[str, ProblemDimensions],
    path: str | None = None,
    to_console: bool = False,
):
    """
    Renders a rich table summarizing the problem dimensions.

    Args:
        configs (dict[str, SolverKaminoImpl.Config]):
            A dictionary mapping configuration names to problem dimensions.
        path (str, optional):
            The file path to save the rendered table as a text file. If None, the table is not saved to a file.
        to_console (bool, optional):
            If True, also prints the table to the console.

    Raises:
        ValueError:
            If the configs dictionary is empty or if any of the configuration objects are missing required attributes.
        IOError:
            If there is an error writing the table to the specified file path.
    """
    # Attempt to import rich first, and warn user
    # if the necessary package is not installed
    try:
        from rich import box  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The `rich` package is required for rendering tables. Install it with: pip install rich"
        ) from e

    # Initialize the table with appropriate columns and styling
    table = Table(
        title="Problem Dimensions Summary",
        show_header=True,
        box=box.SIMPLE_HEAVY,
        show_lines=True,
        pad_edge=True,
    )

    # Table header
    _add_table_column_group(
        table,
        "",
        ["Problem", "# Body Dofs", "# Joint Dofs", "Min Delassus Dim", "Max Delassus Dim"],
        color="white",
        justify="left",
    )

    # Add row for each problem
    for name, dims in problem_dims.items():
        row = [
            str(dims.num_body_dofs),
            str(dims.num_joint_dofs),
            str(dims.min_delassus_dim),
            str(dims.max_delassus_dim),
        ]
        table.add_row(name, *row)

    # Render the table to the console and/or save to file
    _render_table_to_console_and_file(table, path=path, to_console=to_console, max_width=None)
