# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Effective inverse-mass response for articulated systems.

:class:`ResponseOracle` owns the full inverse joint-space mass block for each
articulation and its diagonal in :attr:`alpha`. Update the response for each
state with :meth:`ResponseOracle.refresh` or
:meth:`ResponseOracle.refresh_from_forward_dynamics`. The former uses
preallocated buffers and device kernels, so it can be captured in a CUDA graph.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..sim.articulation import eval_fk, eval_jacobian, eval_mass_matrix

__all__ = ["ResponseOracle"]


@wp.kernel(enable_backward=False)
def _alpha_from_mass_matrix_kernel(
    H: wp.array3d[float],
    art_dof_start: wp.array[wp.int32],
    art_dof_count: wp.array[wp.int32],
    L: wp.array3d[float],
    y: wp.array2d[float],
    alpha: wp.array[float],
):
    """Write ``alpha[dof] = (H^{-1})_{dof,dof}`` for one articulation per thread.

    Factorizes the articulation's H block as ``L L^T`` (Cholesky), then uses
    ``(H^{-1})_{jj} = ||L^{-1} e_j||^2`` via one forward substitution per
    column. Articulation blocks are small, so a single thread per
    articulation is enough and needs no cross-thread coordination.
    """
    a = wp.tid()
    n = art_dof_count[a]
    base = art_dof_start[a]

    # Cholesky H = L L^T (lower triangle). A relative floor on the pivot
    # keeps near-singular blocks finite instead of branching to a fallback.
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

    # Forward substitution L y = e_j; alpha = sum of squares of y.
    for j in range(n):
        s = float(0.0)
        for i in range(j, n):
            t = float(0.0)
            if i == j:
                t = 1.0
            for k in range(j, i):
                t -= L[a, i, k] * y[a, k]
            t = t / L[a, i, i]
            y[a, i] = t
            s += t * t
        alpha[base + j] = s


@wp.kernel(enable_backward=False)
def _inverse_block_from_mass_matrix_kernel(
    H: wp.array3d[float],
    art_dof_start: wp.array[wp.int32],
    art_dof_count: wp.array[wp.int32],
    L: wp.array3d[float],
    inv_block: wp.array3d[float],
    alpha: wp.array[float],
):
    """Write the full inverse block ``inv_block[a] = H_a^{-1}`` per articulation.

    Cholesky ``H = L L^T``, then for each column c solve ``H x = e_c`` (forward
    then backward substitution) and store x as column c of the inverse. Store
    the diagonal in ``alpha``.
    """
    a = wp.tid()
    n = art_dof_count[a]
    base = art_dof_start[a]

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

    for j in range(n):
        alpha[base + j] = inv_block[a, j, j]


class ResponseOracle:
    """Effective inverse-mass response for each articulation.

    :attr:`inverse_blocks` stores the inverse joint-space mass matrix for each
    articulation. :attr:`alpha` stores the matrix diagonal, with
    ``alpha[dof] = (H^{-1})_{dof,dof}`` [1/kg or 1/(kg·m²)]. DOFs outside any
    articulation have a zero response.

    :meth:`refresh` computes the response from the current mass matrix using
    device kernels. :meth:`refresh_from_forward_dynamics` derives the response
    from unit-impulse probes of a solver's articulated-body dynamics.
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

        self._alpha = wp.zeros(model.joint_dof_count, dtype=float, device=device)

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

        # Lazily allocated scratch for refresh_from_forward_dynamics().
        self._probe_state_in = None
        self._probe_state_out = None
        self._probe_control = None

    @property
    def alpha(self) -> wp.array[float]:
        """Effective inverse mass per DOF [1/kg or 1/(kg·m²)], shape [joint_dof_count].

        :meth:`refresh` overwrites this persistent buffer in place. Values may
        also be written directly when only a known per-DOF response is needed.
        """
        return self._alpha

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
        """Recompute :attr:`alpha` and :attr:`inverse_blocks` for *state*.

        Args:
            state: Simulation state providing ``joint_q`` / ``joint_qd``.
        """
        model = self.model
        eval_fk(model, state.joint_q, state.joint_qd, state)
        eval_jacobian(model, state, J=self._J, joint_S_s=self._joint_S_s)
        eval_mass_matrix(model, state, H=self._H, J=self._J, body_I_s=self._body_I_s)
        self._inv_block.zero_()
        wp.launch(
            _inverse_block_from_mass_matrix_kernel,
            dim=model.articulation_count,
            inputs=[self._H, self._art_dof_start, self._art_dof_count, self._L, self._inv_block],
            outputs=[self._alpha],
            device=model.device,
        )

    def refresh_from_forward_dynamics(self, solver, state, probe_dt: float = 1.0e-4) -> None:
        """Recompute the coupled response by probing *solver*'s forward dynamics.

        Each DOF is probed with a unit generalized force. The complete
        resulting acceleration vector forms one column of the inverse mass
        matrix, including solver-specific inertia terms such as armature and
        regularization. A zero-force baseline is subtracted so gravity and
        other pose-only forces cancel.

        Costs ``joint_dof_count + 1`` forward evaluations and reads results
        back to the host, so it is not CUDA-graph capturable. Use
        :meth:`refresh` inside captured loops.

        Args:
            solver: A solver exposing ``step(state_in, state_out, control,
                contacts, dt)``.
            state: Simulation state providing the pose ``joint_q``.
            probe_dt: Small timestep for the probe; the response is read as
                ``q̇ / probe_dt``.
        """
        model = self.model
        n = model.joint_dof_count
        if self._probe_state_in is None:
            self._probe_state_in = model.state()
            self._probe_state_out = model.state()
            self._probe_control = model.control()

        si, so, control = self._probe_state_in, self._probe_state_out, self._probe_control
        # Zero velocity so Coriolis vanishes; si stays read-only across probes.
        wp.copy(si.joint_q, state.joint_q)
        si.joint_qd.zero_()
        eval_fk(model, si.joint_q, si.joint_qd, si)

        force = np.zeros(n, dtype=np.float32)
        control.joint_f.assign(force)
        solver.step(si, so, control, None, probe_dt)
        qd_baseline = so.joint_qd.numpy().copy()  # snapshot: CPU .numpy() aliases the live buffer

        response = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            force[i] = 1.0
            control.joint_f.assign(force)
            force[i] = 0.0
            solver.step(si, so, control, None, probe_dt)
            response[:, i] = (so.joint_qd.numpy() - qd_baseline) / probe_dt

        blocks = np.zeros(self._inv_block.shape, dtype=np.float32)
        for a, (base, count) in enumerate(zip(self._art_dof_starts_host, self._art_dof_counts_host, strict=True)):
            blocks[a, :count, :count] = response[base : base + count, base : base + count]
        self._inv_block.assign(blocks)
        self._alpha.assign(np.diag(response))
