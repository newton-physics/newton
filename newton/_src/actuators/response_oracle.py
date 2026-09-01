# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Effective inverse-mass response for articulated systems.

:class:`ResponseOracle` owns the full inverse joint-space mass block for each
articulation. There are two ways to update it:

- :meth:`ResponseOracle.refresh` assembles the mass matrix itself.
- :meth:`ResponseOracle.refresh_from_solve` reuses a solver's own inertia
  without materializing it, so factorized solvers work too.

Both use preallocated buffers and device kernels, so they can be captured in a
CUDA graph.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp

from ..sim.articulation import eval_fk, eval_jacobian, eval_mass_matrix

__all__ = ["ResponseOracle"]

_FLOAT32_EPS = wp.constant(wp.float32(np.finfo(np.float32).eps))


@wp.kernel(enable_backward=False)
def _inverse_block_from_mass_matrix_kernel(
    H: wp.array3d[float],
    art_dof_count: wp.array[wp.int32],
    L: wp.array3d[float],
    inv_block: wp.array3d[float],
):
    """Write the full inverse block ``inv_block[a] = H_a^{-1}`` per articulation.

    Cholesky ``H = L L^T``, then for each column c solve ``H x = e_c`` (forward
    then backward substitution) and store x as column c of the inverse.
    """
    a = wp.tid()
    n = art_dof_count[a]

    for j in range(n):
        s = H[a, j, j]
        for k in range(j):
            s -= L[a, j, k] * L[a, j, k]
        # Keep the pivot above float32 cancellation noise and bound singular inverses.
        s = wp.max(s, _FLOAT32_EPS * wp.max(wp.abs(H[a, j, j]), 1.0))
        d = wp.sqrt(s)
        L[a, j, j] = d
        for i in range(j + 1, n):
            t = H[a, i, j]
            for k in range(j):
                t -= L[a, i, k] * L[a, j, k]
            L[a, i, j] = t / d

    for c in range(n):
        # forward: L y = e_c  (y accumulated into inv_block[:, c])
        for i in range(n):
            t = float(0.0)
            if i == c:
                t = 1.0
            for k in range(i):
                t -= L[a, i, k] * inv_block[a, k, c]
            inv_block[a, i, c] = t / L[a, i, i]
        # backward: L^T x = y  (overwrite in place)
        for ii in range(n):
            i = n - 1 - ii
            t = inv_block[a, i, c]
            for k in range(i + 1, n):
                t -= L[a, k, i] * inv_block[a, k, c]
            inv_block[a, i, c] = t / L[a, i, i]


@wp.kernel(enable_backward=False)
def _add_armature_kernel(
    armature: wp.array[float],
    art_dof_start: wp.array[wp.int32],
    art_dof_count: wp.array[wp.int32],
    H: wp.array3d[float],
):
    """Add joint armature to the mass-matrix diagonal.

    Solvers carry armature as extra rotor inertia on the diagonal (MuJoCo's
    ``dof_armature``, Featherstone's ``joint_armature``), but
    :func:`~newton.eval_mass_matrix` builds ``J^T M J`` without it. Omitting it
    here would overstate the response and make the solve under-drive the joint.
    """
    a, j = wp.tid()
    if j < art_dof_count[a]:
        H[a, j, j] = H[a, j, j] + armature[art_dof_start[a] + j]


@wp.kernel(enable_backward=False)
def _unit_rhs_kernel(column: int, rhs: wp.array2d[float]):
    """Right-hand side ``e_column`` for every world."""
    w, i = wp.tid()
    rhs[w, i] = wp.where(i == column, 1.0, 0.0)


@wp.kernel(enable_backward=False)
def _scatter_inverse_column_kernel(
    solution: wp.array2d[float],
    column: int,
    dof_map: wp.array2d[wp.int32],
    dofs_per_world: int,
    dof_articulation: wp.array[wp.int32],
    dof_local_index: wp.array[wp.int32],
    inv_block: wp.array3d[float],
):
    """Scatter one solved column of ``M^-1`` into the per-articulation blocks.

    ``solution[w, i]`` is entry ``i`` of ``M^-1 e_column`` in world ``w``. Entries
    coupling two articulations are dropped: the block layout has nowhere to put
    them, and a joint-space inertia is block diagonal across separate trees.
    """
    w, i = wp.tid()

    ni = w * dofs_per_world + i
    nj = w * dofs_per_world + column
    if dof_map:
        ni = dof_map[w, i]
        nj = dof_map[w, column]
    dof_count = dof_articulation.shape[0]
    if ni < 0 or nj < 0 or ni >= dof_count or nj >= dof_count:
        return

    a = dof_articulation[ni]
    if a < 0 or dof_articulation[nj] != a:
        return
    inv_block[a, dof_local_index[ni], dof_local_index[nj]] = solution[w, i]


class ResponseOracle:
    """Effective inverse-mass response for each articulation.

    :attr:`inverse_blocks` holds ``H_a^{-1}`` for each articulation
    [1/kg or 1/(kg·m²)], indexed by articulation-local DOF. Articulations with
    no entry have a zero response.

    :meth:`refresh` computes it from a mass matrix it assembles itself.
    :meth:`refresh_from_solve` reuses the solver's own inertia, which is more
    faithful to the dynamics the effort is fed into. Both run entirely in device
    kernels.
    """

    def __init__(self, model):
        """Initialize the oracle and its scratch buffers for a model.

        Args:
            model: A finalized :class:`~newton.Model` with articulations.
        """
        if model.articulation_count == 0:
            raise ValueError("ResponseOracle requires a model with articulations")
        self.model = model

        device = model.device
        art_count = model.articulation_count
        max_links = model.max_joints_per_articulation
        max_dofs = model.max_dofs_per_articulation

        joint_qd_start = model.joint_qd_start.numpy()
        articulation_start = model.articulation_start.numpy()
        articulation_end = model.articulation_end.numpy()
        starts = []
        counts = []
        for a in range(art_count):
            base = int(joint_qd_start[int(articulation_start[a])])
            end = int(joint_qd_start[int(articulation_end[a])])
            starts.append(base)
            counts.append(end - base)
        self._art_dof_starts_host = starts
        self._art_dof_counts_host = counts
        self._art_dof_start = wp.array(starts, dtype=wp.int32, device=device)
        self._art_dof_count = wp.array(counts, dtype=wp.int32, device=device)

        # Reverse lookup used to scatter a solver's mass matrix into the blocks.
        dof_articulation = [-1] * model.joint_dof_count
        dof_local_index = [0] * model.joint_dof_count
        for a in range(art_count):
            for k in range(counts[a]):
                dof_articulation[starts[a] + k] = a
                dof_local_index[starts[a] + k] = k
        self._dof_articulation = wp.array(dof_articulation, dtype=wp.int32, device=device)
        self._dof_local_index = wp.array(dof_local_index, dtype=wp.int32, device=device)

        self._H = wp.zeros((art_count, max_dofs, max_dofs), dtype=float, device=device)
        self._J = wp.zeros((art_count, max_links * 6, max_dofs), dtype=float, device=device)
        self._body_I_s = wp.zeros(model.body_count, dtype=wp.spatial_matrix, device=device)
        self._joint_S_s = wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device)
        self._L = wp.zeros_like(self._H)
        self._inv_block = wp.zeros_like(self._H)

        # Scratch so refresh() never writes to the caller's state.
        self._fk_state = model.state()

        self._uniform_dofs_per_world = None
        if model.world_count == 1:
            self._uniform_dofs_per_world = model.joint_dof_count
        elif model.joint_dof_world_start is not None:
            bounds = model.joint_dof_world_start.numpy()
            counts = {int(bounds[w + 1] - bounds[w]) for w in range(model.world_count)}
            n_global = int(bounds[-1] - bounds[-2])
            if len(counts) == 1 and n_global == 0 and sum(counts) * model.world_count == model.joint_dof_count:
                self._uniform_dofs_per_world = counts.pop()

        # Sized for the model's own layout so refresh_from_solve() can be captured
        # straight away; a dof_map of a different width resizes it on first use.
        width = self._uniform_dofs_per_world or model.joint_dof_count
        self._rhs = wp.zeros((model.world_count, width), dtype=float, device=device)
        self._sol = wp.zeros_like(self._rhs)

    @property
    def inverse_blocks(self) -> wp.array3d[float]:
        """Read-only per-articulation inverse mass blocks, shape [art_count, max_dofs, max_dofs].

        ``inverse_blocks[a, i, j]`` is the ``(i, j)`` entry of articulation
        ``a``'s inverse mass matrix ``H_a^{-1}`` (indices local to the
        articulation, 0-padded beyond its DOF count). The implicit effort mode
        uses the submatrix indexed by the actuator group's DOFs.

        Update it through :meth:`refresh` or :meth:`refresh_from_solve`. Writing into the array directly is not
        supported: the padding beyond each articulation's DOF count is assumed
        zero by the solve, and a partial write leaves no way to tell a stale
        response from a fresh one.
        """
        return self._inv_block

    def refresh(self, state) -> None:
        """Recompute :attr:`inverse_blocks` for *state*.

        Reads *state* without modifying it. Includes ``joint_armature``. Joint
        damping, joint limits, friction, contacts, constraint regularization and
        kinematic loop closures are absent, so the response comes out larger than
        anticipated and the solve yields a smaller effort than it otherwise would.
        Use :meth:`refresh_from_solve` for a solver-faithful response; loop
        closures are missing from that path too, since a solver enforces them as
        constraints rather than folding them into its inertia.

        Args:
            state: Simulation state providing ``joint_q`` / ``joint_qd``.
        """
        model = self.model
        # eval_fk overwrites body_q/body_qd, so keep it off the caller's state. It
        # only reads joint_q/joint_qd, and eval_jacobian reads joint_q from the
        # state it is handed, so the scratch state can alias rather than copy.
        fk_state = self._fk_state
        fk_state.joint_q = state.joint_q
        fk_state.joint_qd = state.joint_qd
        eval_fk(model, state.joint_q, state.joint_qd, fk_state)
        eval_jacobian(model, fk_state, J=self._J, joint_S_s=self._joint_S_s)
        eval_mass_matrix(model, fk_state, H=self._H, J=self._J, body_I_s=self._body_I_s)
        if model.joint_armature is not None:
            wp.launch(
                _add_armature_kernel,
                dim=(model.articulation_count, self._H.shape[1]),
                inputs=[model.joint_armature, self._art_dof_start, self._art_dof_count],
                outputs=[self._H],
                device=model.device,
            )
        self._invert_blocks()

    def refresh_from_solve(
        self,
        solve_inverse: Callable[[wp.array2d[float], wp.array2d[float]], None],
        dof_map: wp.array2d[wp.int32] | None = None,
    ) -> None:
        """Recompute :attr:`inverse_blocks` from a solver's own joint-space inertia.

        Prefer this over :meth:`refresh` when the solver can apply its inertia:
        the response then carries what the solver folds in (armature, tendon
        armature). The inertia never has to be materialized, so factorized
        solvers work too -- the inverse is recovered one column at a time by
        back-substituting unit vectors.

        That is one solve per DOF, all on device, so this is CUDA-graph
        capturable if *solve_inverse* is. Call it once outside capture when
        *dof_map* has a different width than the model's DOF layout; that first
        call resizes the scratch buffers.

        With :class:`~newton.solvers.SolverMuJoCo`, which factorizes its inertia
        each step::

            def solve_inverse(x, y):
                mujoco_warp.solve_m(solver.mjw_model, solver.mjw_data, x, y)


            # Simulation loop
            oracle.refresh_from_solve(solve_inverse, dof_map=solver.mjc_dof_to_newton_dof)

        Args:
            solve_inverse: Callable ``(x, y)`` writing ``x = M^-1 y``, both shaped
                ``[world_count, dof_count]`` in the solver's own DOF order.
            dof_map: Mapping from solver ``[world, dof]`` to Newton DOF index,
                negative where a solver DOF has no Newton counterpart. If
                ``None``, the solver is assumed to use Newton DOF order with the
                same DOF count in every world.
        """
        model = self.model
        if dof_map is None:
            if self._uniform_dofs_per_world is None:
                raise ValueError(
                    "dof_map is required when worlds do not all have the same DOF count, "
                    "or when the model has global joints"
                )
            dofs_per_world = self._uniform_dofs_per_world
            world_count, dof_count = model.world_count, dofs_per_world
        else:
            dofs_per_world = 0
            world_count, dof_count = dof_map.shape

        if self._rhs is None or self._rhs.shape != (world_count, dof_count):
            self._rhs = wp.zeros((world_count, dof_count), dtype=float, device=model.device)
            self._sol = wp.zeros_like(self._rhs)

        self._inv_block.zero_()
        for column in range(dof_count):
            wp.launch(
                _unit_rhs_kernel,
                dim=self._rhs.shape,
                inputs=[column],
                outputs=[self._rhs],
                device=model.device,
            )
            solve_inverse(self._sol, self._rhs)
            wp.launch(
                _scatter_inverse_column_kernel,
                dim=(world_count, dof_count),
                inputs=[
                    self._sol,
                    column,
                    dof_map,
                    dofs_per_world,
                    self._dof_articulation,
                    self._dof_local_index,
                ],
                outputs=[self._inv_block],
                device=model.device,
            )

    def _invert_blocks(self) -> None:
        """Invert the per-articulation blocks of ``self._H`` into :attr:`inverse_blocks`."""
        self._inv_block.zero_()
        wp.launch(
            _inverse_block_from_mass_matrix_kernel,
            dim=self.model.articulation_count,
            inputs=[self._H, self._art_dof_count],
            outputs=[self._L, self._inv_block],
            device=self.model.device,
        )
