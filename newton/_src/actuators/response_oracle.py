# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-DOF impulse response (effective inverse mass), refreshed on device.

:class:`ResponseOracle` owns one persistent buffer with ``alpha[dof] =
(H^{-1})_{dof,dof}``, the diagonal of the inverse joint-space mass matrix —
how much a DOF's velocity changes per unit of impulse. This quantity is the
DOF's *impulse response* (equivalently, its effective inverse mass ``1/I``).
Consumers (e.g. the implicit actuation strategy) hold the buffer and read it;
keeping it current is a separate, explicit step, done one of two ways:

1. ``oracle.refresh(state)`` — once per step, before consumers read
   :attr:`ResponseOracle.alpha`; recomputes from the model's dense mass matrix.
2. Write into :attr:`ResponseOracle.alpha` directly — e.g. from solver-computed
   data — instead of ever calling ``refresh``.

``refresh`` launches only device kernels into preallocated buffers, so it can
be captured inside a CUDA graph.
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


class ResponseOracle:
    """Per-DOF effective inverse mass, shared across consumers.

    Owns the persistent device buffer :attr:`alpha` of shape
    ``[joint_dof_count]`` with ``alpha[dof] = (H^{-1})_{dof,dof}``
    [1/kg or 1/(kg·m²)], where ``H`` is the joint-space mass matrix of the
    DOF's articulation. DOFs outside any articulation stay ``0.0``.

    Three ways to keep :attr:`alpha` current, before consumers read it:

    1. :meth:`refresh` — recompute from a dense mass matrix for the current
       pose. Launches only device kernels, so it is CUDA-graph capturable.
    2. :meth:`refresh_from_forward_dynamics` — probe a solver's
       articulated-body dynamics with unit impulses; inherits the solver's
       inertia model but is host-side (not graphable).
    3. Write into :attr:`alpha` in place from your own source (e.g. a
       solver's cached mass matrix).
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
        self._art_dof_start = wp.array(starts, dtype=wp.int32, device=device)
        self._art_dof_count = wp.array(counts, dtype=wp.int32, device=device)

        self._H = wp.zeros((art_count, max_dofs, max_dofs), dtype=float, device=device)
        self._J = wp.zeros((art_count, max_links * 6, max_dofs), dtype=float, device=device)
        self._body_I_s = wp.zeros(model.body_count, dtype=wp.spatial_matrix, device=device)
        self._joint_S_s = wp.zeros(model.joint_dof_count, dtype=wp.spatial_vector, device=device)
        self._L = wp.zeros_like(self._H)
        self._y = wp.zeros((art_count, max_dofs), dtype=float, device=device)

        # Lazily allocated scratch for refresh_from_forward_dynamics().
        self._probe_state_in = None
        self._probe_state_out = None
        self._probe_control = None

    @property
    def alpha(self) -> wp.array[float]:
        """Effective inverse mass per DOF [1/kg or 1/(kg·m²)], shape [joint_dof_count].

        Consumers hold and read this buffer; :meth:`refresh` overwrites it in
        place. To supply your own values (e.g. solver-computed inverse
        masses), write into the buffer directly instead of calling
        :meth:`refresh`. The buffer identity never changes, so values written
        here are always visible to every consumer.
        """
        return self._alpha

    def refresh(self, state) -> None:
        """Recompute :attr:`alpha` for the pose in *state*.

        Args:
            state: Simulation state providing ``joint_q`` / ``joint_qd``.
        """
        model = self.model
        eval_fk(model, state.joint_q, state.joint_qd, state)
        eval_jacobian(model, state, J=self._J, joint_S_s=self._joint_S_s)
        eval_mass_matrix(model, state, H=self._H, J=self._J, body_I_s=self._body_I_s)
        wp.launch(
            _alpha_from_mass_matrix_kernel,
            dim=model.articulation_count,
            inputs=[self._H, self._art_dof_start, self._art_dof_count, self._L, self._y],
            outputs=[self._alpha],
            device=model.device,
        )

    def refresh_from_forward_dynamics(self, solver, state, probe_dt: float = 1.0e-4) -> None:
        """Recompute :attr:`alpha` by probing *solver*'s forward dynamics.

        Uses the identity ``(M^{-1})_{ii} = q̈_i`` for a unit generalized force
        at DOF ``i``: each DOF is probed through the solver's articulated-body
        dynamics, so :attr:`alpha` inherits the solver's inertia model
        (armature, regularization) rather than a separate dense matrix. A
        zero-force baseline is subtracted so gravity and other pose-only forces
        cancel.

        Costs ``joint_dof_count + 1`` forward evaluations and reads results
        back to the host, so it is not CUDA-graph capturable — prefer
        :meth:`refresh` inside captured loops. The values match
        :meth:`refresh` when the solver adds no extra inertia.

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

        alpha = np.zeros(n, dtype=np.float32)
        for i in range(n):
            force[i] = 1.0
            control.joint_f.assign(force)
            force[i] = 0.0
            solver.step(si, so, control, None, probe_dt)
            alpha[i] = (so.joint_qd.numpy()[i] - qd_baseline[i]) / probe_dt

        self._alpha.assign(alpha)
