# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""cuNLS optimizer backend for inverse kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..enums import JointType
from ..model import Model
from .ik_common import IKJacobianType, compute_costs
from .ik_objectives import IKObjective, IKObjectiveJointLimit, IKObjectivePosition, IKObjectiveRotation

_TREE_JOINT_TYPES = (
    JointType.PRISMATIC,
    JointType.REVOLUTE,
    JointType.BALL,
    JointType.FIXED,
    JointType.FREE,
    JointType.D6,
)
_LOOP_JOINT_TYPES = (JointType.PRISMATIC, JointType.REVOLUTE, JointType.BALL, JointType.FIXED)


def _import_pycunls():
    """Import the optional cuNLS package."""
    try:
        import pycunls  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "IKOptimizer.CUNLS requires cuNLS. Install it with `pip install newton[cunls-cu12]` "
            "(CUDA 12) or `pip install newton[cunls-cu13]` (CUDA 13)."
        ) from e
    return pycunls


@wp.kernel
def _gather_rows(src: wp.array[Any], rows: wp.array[wp.int32], dst: wp.array[Any]):
    i = wp.tid()
    dst[i] = src[rows[i]]


@wp.kernel
def _gather_coords(joint_q: wp.array2d[wp.float32], coord_map: wp.array[wp.int32], q: wp.array2d[wp.float32]):
    row, c = wp.tid()
    q[row, c] = joint_q[row, coord_map[c]]


@wp.kernel
def _scatter_coords(q: wp.array2d[wp.float32], coord_map: wp.array[wp.int32], joint_q: wp.array2d[wp.float32]):
    row, c = wp.tid()
    joint_q[row, coord_map[c]] = q[row, c]


@dataclass
class _Kinematics:
    """The model's articulation trees as one cuNLS topology whose link ``k`` is body ``k``."""

    topology: Any
    """The ``pycunls.ArticulationTopology`` of all tree joints."""
    coord_map: np.ndarray
    """Model coordinate index of each topology coordinate, shape [topology coordinate count]."""
    dof_map: np.ndarray
    """Model DOF index of each topology DOF, shape [topology DOF count]."""
    loop_joints: np.ndarray
    """Model indices of the loop-closing joints."""


def _build_kinematics(pycunls, model: Model, limit: IKObjectiveJointLimit | None) -> _Kinematics:
    joint_type = model.joint_type.numpy()
    parent = model.joint_parent.numpy()
    child = model.joint_child.numpy()
    articulation = model.joint_articulation.numpy()
    articulation_end = model.articulation_end.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()

    # Tree joints precede their articulation's end; every other joint closes a loop.
    joint_ids = np.arange(model.joint_count)
    in_articulation = articulation >= 0
    tree = in_articulation & (child >= 0)
    tree[in_articulation] &= joint_ids[in_articulation] < articulation_end[articulation[in_articulation]]

    tree_joint_of_body = np.full(model.body_count, -1, dtype=np.int64)
    for j in np.flatnonzero(tree):
        if tree_joint_of_body[child[j]] >= 0:
            raise ValueError(f"IKOptimizer.CUNLS: body {child[j]} has several incoming articulation joints.")
        tree_joint_of_body[child[j]] = j
    if np.any(tree_joint_of_body < 0):
        body = int(np.flatnonzero(tree_joint_of_body < 0)[0])
        raise ValueError(f"IKOptimizer.CUNLS: body {body} is not the child of an articulation joint.")
    order = tree_joint_of_body
    for j in order:
        if joint_type[j] not in _TREE_JOINT_TYPES:
            raise ValueError(f"IKOptimizer.CUNLS does not support {JointType(joint_type[j]).name} joints.")

    coord_map = np.concatenate([np.arange(q_start[j], q_start[j + 1]) for j in order]).astype(np.int32)
    dof_map = np.concatenate([np.arange(qd_start[j], qd_start[j + 1]) for j in order]).astype(np.int32)
    lower = upper = None
    if limit is not None:
        lower = limit.joint_limit_lower.numpy()[dof_map]
        upper = limit.joint_limit_upper.numpy()[dof_map]

    topology = pycunls.ArticulationTopology(
        joint_type=joint_type[order],
        joint_parent=parent[order],  # link k is body k
        joint_child=np.arange(model.body_count),
        joint_X_p=model.joint_X_p.numpy()[order],
        joint_X_c=model.joint_X_c.numpy()[order],
        joint_q_start=np.concatenate([[0], np.cumsum(q_start[order + 1] - q_start[order])]),
        joint_qd_start=np.concatenate([[0], np.cumsum(qd_start[order + 1] - qd_start[order])]),
        joint_axis=model.joint_axis.numpy()[dof_map],
        joint_dof_dim=model.joint_dof_dim.numpy()[order],
        limit_lower=lower,
        limit_upper=upper,
    )
    loop_joints = np.flatnonzero(~tree & (child >= 0))
    for j in loop_joints:
        if joint_type[j] not in _LOOP_JOINT_TYPES:
            raise ValueError(
                f"IKOptimizer.CUNLS does not support {JointType(joint_type[j]).name} loop-closing joints "
                f"(joint {j}); supported are PRISMATIC, REVOLUTE, BALL and FIXED."
            )
    return _Kinematics(topology, coord_map, dof_map, loop_joints)


class IKOptimizerCuNLS:
    """Constrained Levenberg-Marquardt optimizer for batched inverse kinematics, backed by cuNLS.

    Each batch row is an instance of the model's articulation trees solved as an independent
    subproblem by `cuNLS <https://github.com/nvidia-isaac/cuNLS>`__ with analytic Jacobians.
    Unlike :class:`~newton.ik.IKOptimizerLM`, it keeps closed loops closed and joint limits
    hard:

    - Loop-closing joints (joints outside the articulation trees) are equality constraints.
    - Mimic joints (:attr:`Model.joint_mimic_joint`) are equality constraints.
    - :class:`~newton.ik.IKObjectiveJointLimit` makes its limits hard bounds: the coordinates
      never leave them, and its ``weight`` is unused.

    Supported objectives are :class:`~newton.ik.IKObjectivePosition`,
    :class:`~newton.ik.IKObjectiveRotation` and :class:`~newton.ik.IKObjectiveJointLimit`.
    Requires the optional ``pycunls`` package (``newton[cunls-cu12]`` or ``newton[cunls-cu13]``).

    Args:
        model: Shared articulation model.
        n_batch: Number of evaluation rows solved in parallel. This is
            typically ``n_problems * n_seeds`` after any sampling expansion.
        objectives: Ordered IK objectives applied to every batch row.
        problem_idx: Optional mapping from batch rows to base problem indices
            for per-problem objective data.
        constraint_tolerance: Largest accepted loop-closure or mimic violation [m or rad].

    Raises:
        ImportError: If ``pycunls`` is not installed.
        ValueError: If an objective, a joint type, or the articulation layout is not supported.

    Note:
        The objectives' targets are read at every :meth:`step`; other objective data (links,
        offsets, weights, joint limits) and the model are read at construction.
    """

    def __init__(
        self,
        model: Model,
        n_batch: int,
        objectives: Sequence[IKObjective],
        *,
        problem_idx: wp.array[wp.int32] | None = None,
        constraint_tolerance: float = 1.0e-5,
    ) -> None:
        pycunls = _import_pycunls()
        if not model.device.is_cuda:
            raise ValueError(f"IKOptimizer.CUNLS runs on CUDA devices only (got model device {model.device}).")
        self.model = model
        self.device = model.device
        self.n_batch = n_batch
        self.n_coords = model.joint_coord_count
        self.objectives = objectives
        self.n_residuals = sum(o.residual_dim() for o in objectives)
        self.problem_idx = (
            problem_idx
            if problem_idx is not None
            else wp.array(np.arange(n_batch, dtype=np.int32), dtype=wp.int32, device=self.device)
        )
        self.constraint_tolerance = constraint_tolerance
        self.costs = wp.zeros(n_batch, dtype=wp.float32, device=self.device)

        limits = [o for o in objectives if isinstance(o, IKObjectiveJointLimit)]
        if len(limits) > 1:
            raise ValueError("IKOptimizer.CUNLS accepts at most one IKObjectiveJointLimit.")
        with wp.ScopedDevice(self.device):  # cuNLS allocates on the current CUDA device
            self._build_problem(pycunls, model, objectives, limits[0] if limits else None)

        # The objectives evaluate the costs of the solved configurations.
        offset = 0
        for obj in objectives:
            obj.set_batch_layout(self.n_residuals, offset, n_batch)
            obj.bind_device(self.device)
            obj.init_buffers(model=model, jacobian_mode=IKJacobianType.ANALYTIC)
            offset += obj.residual_dim()

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------
    def _build_problem(self, pycunls, model: Model, objectives, limit: IKObjectiveJointLimit | None) -> None:
        n_batch = self.n_batch
        kinematics = _build_kinematics(pycunls, model, limit)
        self._pycunls = pycunls
        self._topology = kinematics.topology
        self._coord_map = wp.array(kinematics.coord_map, dtype=wp.int32, device=self.device)
        self._q = wp.zeros((n_batch, len(kinematics.coord_map)), dtype=wp.float32, device=self.device)
        self._body_q = wp.zeros((n_batch, model.body_count), dtype=wp.transform, device=self.device)
        self._residuals = wp.zeros((n_batch, max(self.n_residuals, 1)), dtype=wp.float32, device=self.device)
        self._stream = pycunls.CudaStream()
        self._wp_stream = wp.Stream(self.device, cuda_stream=self._stream.get_stream())

        self._states = pycunls.ArticulationStateBatch(self._topology, n_batch, apply_joint_limits=limit is not None)
        self._states.set_num_active_states(n_batch)
        self._problem = pycunls.Problem()
        self._problem.add_state_batch(self._states)
        self._factors = []  # (factor batch, batch row of each factor)
        self._target_gathers = []  # (objective targets, expanded targets) refreshed at each step
        self._add_objective_factors(objectives)
        self._add_loop_closure_factors(model, kinematics)
        self._add_mimic_factors(model, kinematics)
        for factor, rows in self._factors:
            self._problem.add_factor_batch(factor, [self._states], rows)
        # One subproblem per row: each converges and stops on its own.
        self._rows = wp.array(np.arange(n_batch, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._problem.set_problem_partition(n_batch, [self._rows])
        self._solvers = {}  # iterations -> minimizer

    def _per_row(self, values, count: int = 1) -> wp.array:
        """Device float array repeating ``values`` for each of ``count`` factors per batch row."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        return wp.array(np.tile(values, self.n_batch * count), dtype=wp.float32, device=self.device)

    def _add_factor(self, factor, per_row: int = 1, *, constraint: bool = False) -> None:
        """Add ``per_row`` factors per batch row, consecutive factors on the same row."""
        factor.set_num_active_factors(self.n_batch * per_row)
        if constraint:
            factor = self._pycunls.ConstraintFactorBatch(factor, self._pycunls.ConstraintKind.Equality)
        instances = np.repeat(np.arange(self.n_batch, dtype=np.int32), per_row)
        self._factors.append((factor, wp.array(instances, dtype=wp.int32, device=self.device)))

    def _expanded_targets(self, targets: wp.array) -> wp.array:
        expanded = wp.zeros(self.n_batch, dtype=targets.dtype, device=self.device)
        self._target_gathers.append((targets, expanded))
        return expanded

    def _add_objective_factors(self, objectives: Sequence[IKObjective]) -> None:
        pycunls, n = self._pycunls, self.n_batch
        for obj in objectives:
            if isinstance(obj, IKObjectivePosition):
                factor = pycunls.LinkPositionFactorBatch(
                    self._topology,
                    wp.full(n, obj.link_index, dtype=wp.int32, device=self.device),
                    self._per_row(obj.link_offset),
                    self._expanded_targets(obj.target_positions),
                    self._per_row([obj.weight] * 3),
                    n,
                )
            elif isinstance(obj, IKObjectiveRotation):
                factor = pycunls.LinkOrientationFactorBatch(
                    self._topology,
                    wp.full(n, obj.link_index, dtype=wp.int32, device=self.device),
                    self._per_row(obj.link_offset_rotation),
                    self._expanded_targets(obj.target_rotations),
                    self._per_row([obj.weight] * 3),
                    n,
                )
            elif isinstance(obj, IKObjectiveJointLimit):
                continue  # hard bounds of the states
            else:
                raise ValueError(
                    f"IKOptimizer.CUNLS does not support {type(obj).__name__}; supported objectives are "
                    "IKObjectivePosition, IKObjectiveRotation and IKObjectiveJointLimit."
                )
            self._add_factor(factor)

    def _add_loop_closure_factors(self, model: Model, kinematics: _Kinematics) -> None:
        joint_type = model.joint_type.numpy()
        parent = model.joint_parent.numpy()
        child = model.joint_child.numpy()
        frames = np.concatenate([model.joint_X_p.numpy(), model.joint_X_c.numpy()], axis=1)
        axis = model.joint_axis.numpy()
        qd_start = model.joint_qd_start.numpy()
        for kind in _LOOP_JOINT_TYPES:
            joints = kinematics.loop_joints[joint_type[kinematics.loop_joints] == kind]
            if len(joints) == 0:
                continue
            has_axis = kind in (JointType.PRISMATIC, JointType.REVOLUTE)
            m = len(joints)
            factor = self._pycunls.LoopClosureFactorBatch(
                self._topology,
                self._pycunls.ArticulationJointType(int(kind)),
                wp.array(np.tile(parent[joints], self.n_batch), dtype=wp.int32, device=self.device),
                wp.array(np.tile(child[joints], self.n_batch), dtype=wp.int32, device=self.device),
                self._per_row(frames[joints], m),
                self._per_row(axis[qd_start[joints]], m) if has_axis else None,
                self.n_batch * m,
            )
            self._add_factor(factor, m, constraint=True)

    def _add_mimic_factors(self, model: Model, kinematics: _Kinematics) -> None:
        if model.joint_mimic_joint is None:
            return
        reference = model.joint_mimic_joint.numpy()
        coeffs = model.joint_mimic_coeffs.numpy()
        qd_start = model.joint_qd_start.numpy()
        topology_dof = np.full(model.joint_dof_count, -1, dtype=np.int32)
        topology_dof[kinematics.dof_map] = np.arange(len(kinematics.dof_map), dtype=np.int32)
        pairs = []  # (mimic DOF, reference DOF, offset, multiplier) in topology DOFs
        for j in np.flatnonzero(reference >= 0):
            r = reference[j]
            for k in range(qd_start[j + 1] - qd_start[j]):
                mimic, ref = topology_dof[qd_start[j] + k], topology_dof[qd_start[r] + k]
                if mimic < 0 or ref < 0:
                    raise ValueError(
                        f"IKOptimizer.CUNLS: mimic joint {j} or its reference {r} is a loop-closing joint."
                    )
                pairs.append((mimic, ref, coeffs[j][0], coeffs[j][1]))
        if not pairs:
            return
        pairs = np.asarray(pairs)
        m = len(pairs)
        factor = self._pycunls.MimicJointFactorBatch(
            self._topology,
            wp.array(np.tile(pairs[:, 0].astype(np.int32), self.n_batch), dtype=wp.int32, device=self.device),
            wp.array(np.tile(pairs[:, 1].astype(np.int32), self.n_batch), dtype=wp.int32, device=self.device),
            self._per_row(pairs[:, 2:4], m),
            self.n_batch * m,
        )
        self._add_factor(factor, m, constraint=True)

    def _solver(self, iterations: int):
        solver = self._solvers.get(iterations)
        if solver is None:
            pycunls = self._pycunls
            lm = pycunls.LevenbergMarquardtMinimizerOptions()
            base = lm.base_options
            base.max_num_iterations = iterations
            # Absolute: residuals are meters and radians, so this stops at about 10 um. Much tighter
            # sits at the float32 rounding of forward kinematics a few meters from the origin.
            base.cost_tolerance = 1.0e-10
            base.state_tolerance = 1.0e-10
            base.reuse_structure = True
            lm.base_options = base
            options = pycunls.AugmentedLagrangianMinimizerOptions()
            options.constraint_tolerance = self.constraint_tolerance
            solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer(lm), options)
            self._solvers[iterations] = solver
        return solver

    # ------------------------------------------------------------------
    # Optimizer interface
    # ------------------------------------------------------------------
    def _set_states(self, joint_q: wp.array2d[wp.float32]) -> None:
        """Copy ``joint_q`` into the cuNLS states (forward kinematics included)."""
        wp.launch(
            _gather_coords, dim=self._q.shape, inputs=[joint_q, self._coord_map], outputs=[self._q], device=self.device
        )
        self._wp_stream.wait_stream(wp.get_stream(self.device))
        self._states.set_joint_q(self._q, self.n_batch, self._stream)

    def step(
        self,
        joint_q_in: wp.array2d[wp.float32],
        joint_q_out: wp.array2d[wp.float32],
        iterations: int = 10,
    ) -> None:
        """Solve every batch row from ``joint_q_in`` and write the solutions to ``joint_q_out``.

        Args:
            joint_q_in: Input joint coordinates, shape [n_batch, joint_coord_count].
            joint_q_out: Output buffer for the optimized coordinates, shape
                [n_batch, joint_coord_count]. It may alias ``joint_q_in``.
                Coordinates of loop-closing joints are copied from ``joint_q_in``.
            iterations: Iteration limit of each inner Levenberg-Marquardt solve. Rows stop
                earlier once converged.
        """
        if joint_q_in.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")

        for targets, expanded in self._target_gathers:
            wp.launch(
                _gather_rows,
                dim=self.n_batch,
                inputs=[targets, self.problem_idx],
                outputs=[expanded],
                device=self.device,
            )
        self._set_states(joint_q_in)
        with wp.ScopedDevice(self.device):
            self._solver(iterations).minimize(self._stream, self._problem)
        self._states.get_joint_q(self._q, self.n_batch, self._stream)
        wp.get_stream(self.device).wait_stream(self._wp_stream)

        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q_out, joint_q_in)
        wp.launch(
            _scatter_coords,
            dim=self._q.shape,
            inputs=[self._q, self._coord_map],
            outputs=[joint_q_out],
            device=self.device,
        )

    def reset(self) -> None:
        """Reset optimizer state before a new solve; every :meth:`step` starts afresh, so this is a no-op."""

    def compute_costs(self, joint_q: wp.array2d[wp.float32]) -> wp.array[wp.float32]:
        """Evaluate squared residual costs for a batch of joint configurations.

        Args:
            joint_q: Joint coordinates to evaluate, shape [n_batch, joint_coord_count].

        Returns:
            Costs for each batch row, shape [n_batch].
        """
        self._set_states(joint_q)
        self._states.get_link_transforms(self._body_q, self.n_batch, self._stream)
        wp.get_stream(self.device).wait_stream(self._wp_stream)
        offset = 0
        for obj in self.objectives:
            obj.compute_residuals(self._body_q, joint_q, self.model, self._residuals, offset, self.problem_idx)
            offset += obj.residual_dim()
        wp.launch(
            compute_costs,
            dim=self.n_batch,
            inputs=[self._residuals, self.n_residuals],
            outputs=[self.costs],
            device=self.device,
        )
        return self.costs
