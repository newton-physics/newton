# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Effective inverse-mass response for articulated systems.

:class:`ResponseOracle` owns the full inverse joint-space mass block for each
articulation. Update it with
:meth:`ResponseOracle.refresh`, which uses preallocated buffers and device
kernels and so can be captured in a CUDA graph. Both buffers are writable, so a
solver-supplied response can be assigned directly instead.
"""

from __future__ import annotations

import warp as wp

from ..sim.articulation import eval_fk, eval_jacobian, eval_mass_matrix

__all__ = ["ResponseOracle"]


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
        s = wp.max(s, 1.0e-9 * wp.max(H[a, j, j], 1.0e-9))
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


class ResponseOracle:
    """Effective inverse-mass response for each articulation.

    :attr:`inverse_blocks` holds ``H_a^{-1}`` for each articulation
    [1/kg or 1/(kg·m²)], indexed by articulation-local DOF. Articulations with
    no entry have a zero response.

    :meth:`refresh` computes it from the current mass matrix using device
    kernels. Alternatively write a solver-supplied response straight into
    :attr:`inverse_blocks` -- for example by inverting the solver's own mass
    matrix, which is both faithful and CUDA-graph capturable.
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

        self._H = wp.zeros((art_count, max_dofs, max_dofs), dtype=float, device=device)
        self._J = wp.zeros((art_count, max_links * 6, max_dofs), dtype=float, device=device)
        self._body_I_s = wp.zeros(model.body_count, dtype=wp.spatial_matrix, device=device)
        self._joint_S_s = wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device)
        self._L = wp.zeros_like(self._H)
        self._inv_block = wp.zeros_like(self._H)

        # Scratch so refresh() never writes to the caller's state.
        self._fk_state = None

    @property
    def inverse_blocks(self) -> wp.array3d[float]:
        """Per-articulation inverse mass blocks, shape [art_count, max_dofs, max_dofs].

        ``inverse_blocks[a, i, j]`` is the ``(i, j)`` entry of articulation
        ``a``'s inverse mass matrix ``H_a^{-1}`` (indices local to the
        articulation, 0-padded beyond its DOF count). The implicit effort mode
        uses the submatrix indexed by the actuator group's DOFs.
        """
        return self._inv_block

    def refresh(self, state) -> None:
        """Recompute :attr:`inverse_blocks` for *state*.

        Reads *state* without modifying it. Includes ``joint_armature`` but not
        joint damping, contacts, or constraint regularization; the response is
        therefore an upper bound, which under-drives rather than destabilizes the
        solve. For a solver-faithful response, invert the solver's own mass
        matrix into :attr:`inverse_blocks` instead.

        Args:
            state: Simulation state providing ``joint_q`` / ``joint_qd``.
        """
        model = self.model
        # eval_fk overwrites body_q/body_qd, so keep it off the caller's state.
        if self._fk_state is None:
            self._fk_state = model.state()
        fk_state = self._fk_state
        wp.copy(fk_state.joint_q, state.joint_q)
        wp.copy(fk_state.joint_qd, state.joint_qd)
        eval_fk(model, fk_state.joint_q, fk_state.joint_qd, fk_state)
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
        self._inv_block.zero_()
        wp.launch(
            _inverse_block_from_mass_matrix_kernel,
            dim=model.articulation_count,
            inputs=[self._H, self._art_dof_count, self._L],
            outputs=[self._inv_block],
            device=model.device,
        )
