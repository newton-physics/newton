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

"""
This module provides utilities for computing and plotting performance profiles [1,2,3],
which are useful for benchmarking and comparing the performance of different
solvers across a set of problems.

References
----------
- Dolan, E. D., & Moré, J. J. (2002).
  Benchmarking Optimization Software with Performance Profiles.
  Mathematical Programming, 91, 201-213.
  https://doi.org/10.1007/s101070100263
- Dingle, N. J., & Higham, N. J. (2013).
  Reducing the Influence of Tiny Normwise Relative Errors on Performance Profiles.
  ACM Transactions on Mathematical Software, 39(4), 24:1-24:11.
  https://doi.org/10.1145/2491491.2491494
- J. J. Moré, and S. M Wild,
  Benchmarking derivative-free optimization algorithms.
  SIAM Journal on Optimization, 20(1), 172-191, 2009
  https://doi.org/10.1137/080724083

Example
-------
.. code-block:: python
    import numpy as np
    from newton._src.solvers.kamino.utils.profiles import PerformanceProfile

    data = np.array([[1.0, 2.0, 4.0], [1.5, 1.5, 8.0]])
    pp = PerformanceProfile(data)
    pp.plot(names=["S1", "S2"])
"""

import numpy as np

from . import logger as msg

###
# Internals
###

# Local numeric constants (analogs of the C++ constants)
EPSILON: float = float(np.finfo(float).eps)
EPSILON_2: float = float(0.5 * EPSILON)


class _nullcontext:
    """Simple nullcontext for optional style management (avoid importing contextlib at module import time)"""

    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


###
# Types
###


class PerformanceProfile:
    """
    Compute performance profiles for a set of num_solvers solvers across a set of np problems.

    Given performance measurements t_{p,s} for each solver s in S and problem p in P,
    this class computes performance ratios r_{p,s} and the performance profile
    rho_s(τ), defined as:
        rho_s(τ) = (1 / |P|) * |{ p ∈ P : r_{p,s} ≤ τ }|.

    On construction, the input is optionally refined and the profile data are computed
    and cached for later use.

    Args:
        data (np.ndarray): Array of shape (num_solvers, num_problems) with measurements for num_solvers
            solvers across num_problems problems. Each entry typically represents a cost such as time,
            iterations, or error.
        success (np.ndarray | None): Optional boolean or integer array of shape
            (num_solvers, num_problems) indicating successful runs (nonzero/True means success). Failed
            runs are excluded from the profile. If None, all runs are treated as
            successful.
        taumax (float): Maximum performance ratio τ for which the profile is
            generated. If set to float('inf'), τ_max is chosen as 2x (max observed ratio).
        ppfixmax (float): Upper threshold used by the Dingle-Higham refinement when
            ppfix is True (useful when the data represent relative errors).
        ppfixmin (float): Lower floor used by the Dingle-Higham refinement when
            ppfix is True to prevent zero or near-zero values.
        ppfix (bool): If True, apply Dingle-Higham style refinement to the input
            data before computing ratios.

    Attributes:
        _t_p_min (np.ndarray | None): Per-problem best (minimal) measurement across
            solvers, used to form performance ratios.
        _r_ps (np.ndarray | None): Array of performance ratios r_{p,s}.
        _rho_s (list[np.ndarray]): Empirical distribution functions rho_s(τ) for each
            solver over the profile domain.
        _r_min (float): Minimum observed performance ratio.
        _r_max (float): Maximum observed performance ratio.
        _tau_max (float): Effective maximum τ used for the profile domain.
        _valid (bool): True if the profile was successfully computed given the input
            data and success mask; False otherwise.

    Notes:
        - If taumax is float('inf'), the profile domain upper bound is set to twice
          the maximum observed ratio to capture the right tail of the distributions.
        - The Dingle-Higham refinement (controlled by ppfix, ppfixmin, ppfixmax) is
          intended for stabilizing relative-error data by flooring very small values
          and capping extreme ratios before computing performance profiles.
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        success: np.ndarray | None = None,
        taumax: float = np.inf,
        ppfixmax: float = EPSILON_2,
        ppfixmin: float = 1e-18,
        ppfix: bool = False,
    ) -> None:
        self._t_p_min: np.ndarray | None = None
        """
        The per-problem best performance amongst all solvers\n
        For every p in P : t_p_min = min{s in S : t_ps}\n
        Dimensions are: |P|
        """

        self._r_ps: np.ndarray | None = None
        """
        The per-sample performance ratios\n
        For every s in S and p in P : r_ps = t_ps / min{s in S : t_ps}\n
        Dimensions are: |P| x |S|
        """

        self._rho_s: list[np.ndarray] = []
        """
        The per-solver cumulative distributions\n
        For every s in S : rho_s(tau) = (1/np) * size{p in P : r_ps <= tau}\n
        Dimensions are: |S| x |Rho|
        """

        self._r_min: float = np.inf
        """
        The smallest performance ratio present in the data\n
        r in [r_min, r_max]
        """

        self._r_max: float = np.inf
        """
        The largest performance ratio present in the data\n
        r in [r_min, r_max]
        """

        self._tau_max: float = np.inf
        """
        The largest performance ratio considered for generating the performance profile\n
        tau in [1, tau_max]
        """

        self._valid: bool = False
        """
        Flag to indicate if last call to 'compute' was valid\n
        This is also useful to check construction-time generation was successful.
        """

        # Compute the performance profile if data is provided
        if data is not None:
            self._valid = self.compute(
                data=data,
                success=success,
                taumax=taumax,
                ppfixmax=ppfixmax,
                ppfixmin=ppfixmin,
                ppfix=ppfix,
            )

    ###
    # Properties
    ###

    @property
    def is_valid(self) -> bool:
        return self._valid

    @property
    def performance_minima(self) -> np.ndarray:
        if self._t_p_min is None:
            raise RuntimeError("Performance profile not computed yet.")
        return self._t_p_min

    @property
    def performance_ratios(self) -> np.ndarray:
        if self._r_ps is None:
            raise RuntimeError("Performance profile not computed yet.")
        return self._r_ps

    @property
    def largest_performance_ratio(self) -> float:
        return self._r_max

    @property
    def cumulative_distributions(self) -> list[np.ndarray]:
        return self._rho_s

    ###
    # Operations
    ###

    def compute(
        self,
        data: np.ndarray,
        success: np.ndarray | None = None,
        taumax: float = np.inf,
        ppfixmax: float = EPSILON_2,
        ppfixmin: float = 1e-18,
        ppfix: bool = False,
    ) -> bool:
        # Validate input
        valid = True
        if data is None or data.size == 0:
            msg.error("`data` argument is empty!")
            valid = False
        if data.ndim != 2:
            msg.error("`data` must be 2D (num_solvers x np)!")
            valid = False
        else:
            num_solvers, num_problems = data.shape
            if num_solvers < 2:
                msg.error("`data` must contain at least two solver-wise entries (rows >= 2)!")
                valid = False
            if num_problems < 1:
                msg.error("`data` must contain at least one problem-wise entry (cols >= 1)!")
                valid = False
        if success is not None and success.size > 0:
            if success.shape != data.shape:
                msg.error("`success` flags do not match the `data` dimensions!")
                valid = False
        if not valid:
            self._valid = False
            return False

        # Dimensions
        num_solvers, num_problems = data.shape

        # Success flags default to ones
        success_ps = (
            success.astype(float) if success is not None and success.size > 0 else np.ones_like(data, dtype=float)
        )

        # Work on a copy of data
        samples = data.astype(float).copy()

        # Optional Dingle & Higham refinement (useful for relative errors)
        if ppfix:
            ppfixscaling = (ppfixmax - ppfixmin) / ppfixmax
            mask = samples < ppfixmax
            samples[mask] = ppfixmin + samples[mask] * ppfixscaling

        # Per-problem minima across solvers
        self._t_p_min = samples.min(axis=0)

        # Performance ratios r_ps = t_ps / min_s t_ps when successful, else inf
        with np.errstate(divide="ignore", invalid="ignore"):
            r_ps = np.where(success_ps > 0.0, samples / self._t_p_min[None, :], np.inf)

        # Minimal ratio (before filtering non-finite)
        self._r_min = float(np.nanmin(r_ps))

        # Replace non-finite values with zeros; negatives are left as-is and pruned later
        r_ps = np.where(np.isfinite(r_ps), r_ps, 0.0)

        # Maximal ratio observed
        self._r_max = float(np.nanmax(r_ps))

        # tau max for plotting/domain
        self._tau_max = float(taumax) if np.isfinite(taumax) else 2.0 * self._r_max
        if self._tau_max == 1.0:
            self._tau_max += EPSILON

        # Build cumulative distributions per solver
        rho_s: list[np.ndarray] = []
        for s in range(num_solvers):
            tauvec = r_ps[s, :].astype(float)
            # unique sorted values
            utaus = np.unique(tauvec)

            # prune < 1.0 (these include marker zeros)
            utaus = utaus[utaus >= 1.0]

            # prune > tau_max
            utaus = utaus[utaus <= self._tau_max]

            # counts for each unique tau value
            counts: list[int] = []
            for tau in utaus:
                counts.append(int(np.count_nonzero(tauvec == tau)))

            # cumulative distribution (normalized by number of problems)
            rhos = np.cumsum(np.array(counts, dtype=float)) / float(num_problems)

            # ensure starting at tau = 1.0 with rho = 0.0
            if utaus.size == 0 or (utaus.size > 0 and utaus[0] >= 1.0 + EPSILON):
                utaus = np.insert(utaus, 0, 1.0)
                rhos = np.insert(rhos, 0, 0.0)

            # ensure ending at tau = tau_max with flat tail
            if utaus.size <= 1 or (utaus[-1] < self._tau_max - EPSILON):
                utaus = np.append(utaus, self._tau_max)
                tail = rhos[-1] if rhos.size > 0 else 0.0
                rhos = np.append(rhos, tail)

            # store as 2xK array: first row taus, second row rhos
            rho = np.vstack([utaus, rhos])
            rho_s.append(rho)

        self._r_ps = r_ps
        self._rho_s = rho_s
        self._valid = True
        return True

    def rankings(self) -> dict[str, tuple[np.ndarray, np.ndarray, bool]]:
        """
        Compute solver rankings at tau=1.0 and at rho=1.0.

        Returns:
            tuple[np.ndarray, np.ndarray]: A pair of 1D integer arrays with length equal to the number of solvers.
                - First array: rankings based on rho at tau=1.0 (higher is better).
                - Second array: rankings based on the smallest tau at which rho=1.0 (lower is better).

        Notes:
            Rankings are dense: solvers with the same value share the same rank, and
            the next rank is incremented accordingly. Solvers that do not achieve the
            target value are ranked as -1.
        """
        # Ensure a valid profile has been computed before summarizing
        if not self._valid:
            return "Data is invalid: cannot generate rankings."

        # Names default to indices
        num_solvers = len(self._rho_s)

        # Extract the indices of the best and worst solvers at tau = 1.0 and tau = tau_max
        rankings_tau_1 = np.full((num_solvers,), -1, dtype=int)  # Best solver at tau = 1.0
        rankings_rho_1 = np.full((num_solvers,), -1, dtype=int)  # First solver to reach rho = 1.0

        # rho_s is stored as a 2xK array: [taus; rhos]
        # Collect rho(1.0) for each solver
        # (fraction of problems where it performed best)
        rhos_at_tau_1 = np.empty((num_solvers,), dtype=float)
        for s in range(num_solvers):
            taus = self._rho_s[s][0, :]
            rhos = self._rho_s[s][1, :]
            rhos_at_tau_1[s] = float(rhos[0])

        # Dense ranking: higher is better; rank 1 = best
        unique_vals = np.unique(rhos_at_tau_1)[::-1]
        for rank, val in enumerate(unique_vals, start=1):
            rankings_tau_1[rhos_at_tau_1 == val] = rank

        # Collect tau where rho first reaches 1.0 for each solver
        # (the factor by which it is close to the best solver on all problems)
        taus_at_rho_1 = np.empty((num_solvers,), dtype=float)
        for s in range(num_solvers):
            taus = self._rho_s[s][0, :]
            rhos = self._rho_s[s][1, :]
            idx = np.where(np.isclose(rhos, 1.0, atol=EPSILON, rtol=0.0))[0]
            taus_at_rho_1[s] = float(taus[idx[0]]) if idx.size else np.inf

        # Dense ranking: lower is better; rank 1 = best
        unique_vals = np.unique(taus_at_rho_1)
        for rank, val in enumerate(unique_vals, start=1):
            rankings_rho_1[taus_at_rho_1 == val] = rank

        # Return the computed rankings together with the associated values
        return {"rho@tau=1": (rankings_tau_1, rhos_at_tau_1, False), "tau@rho=1": (rankings_rho_1, taus_at_rho_1, True)}

    def plot(
        self,
        names: list[str] | None = None,
        title: str = "Performance Profile",
        xtitle: str = "Performance Ratio",
        ytitle: str = "Proportion of Problems",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        xscale: str = "log2",
        yscale: str = "linear",
        style: str | None = None,
        show: bool = True,
        path: str | None = None,
    ) -> None:
        """
        Generates a 2D plot to visualize the performance profile using Matplotlib.

        Args:
            names (str, optional): A vector of solver names used to generate plot labels.
            title (str, optional): The title of the plot.
            xtitle (str, optional): The label for the x-axis.
            ytitle (str, optional): The label for the y-axis.
            xlim (tuple, optional): The limits for the x-axis as (xmin, xmax). If None, defaults to (1.0, tau_max).
            ylim (tuple, optional): The limits for the y-axis as (ymin, ymax). If None, defaults to (0.0, 1.1).
            xscale (str, optional): The scale for the x-axis. Options are 'linear', 'log2', or 'log10'.
            yscale (str, optional): The scale for the y-axis. Options are 'linear', 'log2', or 'log10'.
            legend_loc (str, optional): The location of the legend on the plot.
            style (str, optional): An optional Matplotlib style to apply to the plot.
        """
        # Ensure a valid profile has been computed before plotting
        if not self._valid:
            msg.error("Data is invalid: aborting plot operation")
            return

        # Attempt to import matplotlib
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
            from cycler import cycler  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - optional dependency
            msg.error(f"`matplotlib` is required to plot profiles: {exc}")
            return

        # Names default to indices
        num_solvers = len(self._rho_s)
        solver_names = list(names) if names is not None else [str(i) for i in range(num_solvers)]
        if len(solver_names) < num_solvers:
            solver_names += [str(i) for i in range(len(solver_names), num_solvers)]

        # Apply an optional style
        context_mgr = plt.style.context(style) if style else _nullcontext()
        with context_mgr:
            # 60 styles = 10 colors x 6 dash styles
            colors = list(plt.cm.tab10.colors)  # 10 high-contrast colors
            dash_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 5))]  # 6 clear linestyles
            prop = cycler(linestyle=dash_styles) * cycler(color=colors)
            plt.rcParams["lines.linewidth"] = 1.8  # consistent, readable width

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

            # Set up a prop cycle with distinct colors and linestyles
            ax.set_prop_cycle(prop)

            # Plot step profiles
            lines = []
            for s in range(num_solvers):
                x = self._rho_s[s][0, :]
                y = self._rho_s[s][1, :]
                (line,) = ax.step(x, y, where="post", label=solver_names[s])
                lines.append(line)

            # Axes scales and limits
            if xscale == "log2":
                ax.set_xscale("log", base=2)
            elif xscale == "log10":
                ax.set_xscale("log", base=10)
            else:
                ax.set_xscale("linear")
            if yscale == "log2":
                ax.set_yscale("log", base=2)
            elif yscale == "log10":
                ax.set_yscale("log", base=10)
            else:
                ax.set_yscale("linear")

            xlim = (1.0, self._tau_max) if xlim is None else xlim
            ylim = (0.0, 1.1) if ylim is None else ylim
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel(xtitle)
            ax.set_ylabel(ytitle)
            ax.set_title(title)
            ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.25)
            legend = ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", fancybox=True, shadow=True)

            map_legend_to_ax = {}  # Will map legend lines to original lines.
            pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.
            for legend_line, ax_line in zip(legend.get_lines(), lines, strict=False):
                legend_line.set_picker(pickradius)  # Enable picking on the legend line.
                map_legend_to_ax[legend_line] = ax_line

            def on_pick(event):
                # On the pick event, find the original line corresponding to the legend
                # proxy line, and toggle its visibility.
                legend_line = event.artist
                # Do nothing if the source of the event is not a legend line.
                if legend_line not in map_legend_to_ax:
                    return
                ax_line = map_legend_to_ax[legend_line]
                visible = not ax_line.get_visible()
                ax_line.set_visible(visible)
                # Change the alpha on the line in the legend, so we can see what lines
                # have been toggled.
                legend_line.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()

            # Works even if the legend is draggable. This is independent from picking legend lines.
            fig.canvas.mpl_connect("pick_event", on_pick)
            legend.set_draggable(True)

            if path is not None:
                plt.savefig(path, bbox_inches="tight")
                msg.debug(f"Profile plot saved to: {path}")

            if show:
                plt.show()

            plt.close(fig)
