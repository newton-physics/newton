# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""cuNLS optimizer backend for inverse kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ..enums import JointType
from ..model import Model
from .ik_common import IKJacobianType
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
def _gather_pose_targets(
    positions: wp.array[wp.vec3],
    rotations: wp.array[wp.vec4],
    rows: wp.array[wp.int32],
    poses: wp.array[wp.transform],
):
    # The target pose of each batch row: its problem's position and rotation targets (x, y, z, w).
    i = wp.tid()
    r = rotations[rows[i]]
    poses[i] = wp.transform(positions[rows[i]], wp.quat(r[0], r[1], r[2], r[3]))


@wp.kernel
def _gather_coords(joint_q: wp.array2d[wp.float32], coord_map: wp.array[wp.int32], q: wp.array2d[wp.float32]):
    row, c = wp.tid()
    q[row, c] = joint_q[row, coord_map[c]]


@wp.kernel
def _scatter_coords(q: wp.array2d[wp.float32], coord_map: wp.array[wp.int32], joint_q: wp.array2d[wp.float32]):
    row, c = wp.tid()
    joint_q[row, coord_map[c]] = q[row, c]


@wp.kernel
def _add_squared_residuals(residuals: wp.array2d[wp.float32], costs: wp.array[wp.float32]):
    # costs[row] += |r_row|^2 for one objective batch (one factor per batch row).
    row = wp.tid()
    total = float(0.0)
    for k in range(residuals.shape[1]):
        total += residuals[row, k] * residuals[row, k]
    costs[row] += total


def _build_kinematics(model: Model, limit: IKObjectiveJointLimit | None):
    """The model's articulation trees as one cuNLS topology whose link ``k`` is body ``k``.

    Returns:
        The ``pycunls.articulation.SceneArticulation`` (topology and scene/topology index maps)
        and the model indices of the loop-closing joints.
    """
    from pycunls.articulation import topology_from_joint_arrays  # noqa: PLC0415

    joint_type = model.joint_type.numpy()
    child = model.joint_child.numpy()
    articulation = model.joint_articulation.numpy()
    articulation_end = model.articulation_end.numpy()

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
    for j in tree_joint_of_body:
        if joint_type[j] not in _TREE_JOINT_TYPES:
            raise ValueError(f"IKOptimizer.CUNLS does not support {JointType(joint_type[j]).name} joints.")

    # Ordered by child body, so that topology link k is body k (the objectives' link indices).
    scene = topology_from_joint_arrays(
        model.joint_type,
        model.joint_parent,
        model.joint_child,
        model.joint_X_p,
        model.joint_X_c,
        model.joint_q_start,
        model.joint_qd_start,
        model.joint_axis,
        joint_dof_dim=model.joint_dof_dim,
        limit_lower=None if limit is None else limit.joint_limit_lower,
        limit_upper=None if limit is None else limit.joint_limit_upper,
        joints=tree_joint_of_body,
    )
    loop_joints = np.flatnonzero(~tree & (child >= 0))
    for j in loop_joints:
        if joint_type[j] not in _LOOP_JOINT_TYPES:
            raise ValueError(
                f"IKOptimizer.CUNLS does not support {JointType(joint_type[j]).name} loop-closing joints "
                f"(joint {j}); supported are PRISMATIC, REVOLUTE, BALL and FIXED."
            )
    return scene, loop_joints


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

    Without loop or mimic constraints, :meth:`step` runs exactly ``iterations``
    Levenberg-Marquardt iterations as :class:`~newton.ik.IKOptimizerLM` does: no convergence
    tests and nothing read back to the host, so it can be captured in a CUDA graph. With them,
    an augmented Lagrangian closes the constraints to ``constraint_tolerance``: each row stops
    once converged, and the solve reads its progress back to the host (no graph capture).

    Supported objectives are :class:`~newton.ik.IKObjectivePosition`,
    :class:`~newton.ik.IKObjectiveRotation` and :class:`~newton.ik.IKObjectiveJointLimit`.
    A position and a rotation objective on the same link are solved as one pose target.
    :meth:`compute_costs` returns the squared residuals cuNLS minimizes (orientation errors as
    rotation angles, independent of the quaternion sign; hard limits add nothing).
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
        offsets, weights, joint limits) and the model are read at construction. The solve runs
        on Warp's current stream of the model's device.
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

        # Newton's objective contract: the optimizer gives each objective its batch layout (also
        # needed by set_target_positions / set_target_rotations).
        n_residuals = sum(o.residual_dim() for o in objectives)
        offset = 0
        for obj in objectives:
            obj.set_batch_layout(n_residuals, offset, n_batch)
            obj.bind_device(self.device)
            obj.init_buffers(model=model, jacobian_mode=IKJacobianType.ANALYTIC)
            offset += obj.residual_dim()

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------
    def _build_problem(self, pycunls, model: Model, objectives, limit: IKObjectiveJointLimit | None) -> None:
        n_batch = self.n_batch
        scene, loop_joints = _build_kinematics(model, limit)
        self._pycunls = pycunls
        self._topology = scene.topology
        # The tree's coordinates are usually a contiguous block of joint_q: then the states read
        # and write it in place (strided); otherwise through a gather / scatter.
        self._coord_start = scene.coord_start
        self._coord_map = wp.array(scene.coord_index, dtype=wp.int32, device=self.device)
        self._dof_index = scene.dof_index
        self._q = wp.zeros((n_batch, len(scene.coord_index)), dtype=wp.float32, device=self.device)

        self._states = pycunls.ArticulationStateBatch(self._topology, n_batch, apply_joint_limits=limit is not None)
        self._states.set_num_active_states(n_batch)
        self._problem = pycunls.Problem()
        self._problem.add_state_batch(self._states)
        self._factors = []  # (factor batch, batch row of each factor)
        self._objective_factors = []  # (factor batch, residuals buffer): one factor per row
        self._target_gathers = []  # (kernel, inputs, output): per-row targets refreshed at each step
        self._has_constraints = False
        self._add_objective_factors(objectives)
        self._add_loop_closure_factors(model, scene, loop_joints)
        self._add_mimic_factors(model)
        for factor, rows in self._factors:
            self._problem.add_factor_batch(factor, [self._states], rows)
        # One subproblem per row: each converges and stops on its own. Its system is one dense
        # block; cuNLS's sparse Cholesky factors a batch of such blocks one warp each.
        self._row_ids = self._rows(1)
        self._problem.set_problem_partition(n_batch, [self._row_ids])
        # Row t's state, for evaluating the objectives in compute_costs.
        ambient_bytes = 4 * self._topology.ambient_size
        base = self._states.state_device_ptr(0)
        self._state_table = wp.array(
            base + ambient_bytes * np.arange(n_batch, dtype=np.uint64), dtype=wp.uint64, device=self.device
        )
        self._solvers = {}  # iterations -> minimizer

    def _per_row(self, values, dtype=wp.float32) -> wp.array:
        """Device array repeating ``values``, the data of one row's factors, for every batch row."""
        values = np.asarray(values, dtype=wp.dtype_to_numpy(dtype)).reshape(-1)
        return wp.array(np.tile(values, self.n_batch), dtype=dtype, device=self.device)

    def _rows(self, per_row: int) -> wp.array:
        """The batch row of each factor when every row has ``per_row`` consecutive factors."""
        rows = np.repeat(np.arange(self.n_batch, dtype=np.int32), per_row)
        return wp.array(rows, dtype=wp.int32, device=self.device)

    def _add_factor(self, factor, per_row: int = 1, *, constraint: bool = False) -> None:
        """Add ``per_row`` factors per batch row, consecutive factors on the same row."""
        factor.set_num_active_factors(self.n_batch * per_row)
        if constraint:
            factor = self._pycunls.ConstraintFactorBatch(factor, self._pycunls.ConstraintKind.Equality)
            self._has_constraints = True
        self._factors.append((factor, self._rows(per_row)))

    def _add_objective(self, factor, residual_size: int) -> None:
        self._add_factor(factor)
        residuals = wp.zeros((self.n_batch, residual_size), dtype=wp.float32, device=self.device)
        self._objective_factors.append((factor, residuals))

    def _gathered(self, targets: wp.array) -> wp.array:
        """The per-row copy of an objective's per-problem targets, refreshed at each step."""
        expanded = wp.zeros(self.n_batch, dtype=targets.dtype, device=self.device)
        self._target_gathers.append((_gather_rows, [targets, self.problem_idx], expanded))
        return expanded

    def _add_objective_factors(self, objectives: Sequence[IKObjective]) -> None:
        pycunls, n = self._pycunls, self.n_batch
        positions = [o for o in objectives if isinstance(o, IKObjectivePosition)]
        rotations = [o for o in objectives if isinstance(o, IKObjectiveRotation)]
        for obj in objectives:
            if not isinstance(obj, (IKObjectivePosition, IKObjectiveRotation, IKObjectiveJointLimit)):
                raise ValueError(
                    f"IKOptimizer.CUNLS does not support {type(obj).__name__}; supported objectives are "
                    "IKObjectivePosition, IKObjectiveRotation and IKObjectiveJointLimit."
                )
        paired = set()
        for p in positions:
            r = next((r for r in rotations if r.link_index == p.link_index and id(r) not in paired), None)
            link = wp.full(n, p.link_index, dtype=wp.int32, device=self.device)
            if r is None:
                factor = pycunls.LinkPositionFactorBatch(
                    self._topology,
                    link,
                    self._per_row(p.link_offset),
                    self._gathered(p.target_positions),
                    self._per_row([p.weight] * 3),
                    n,
                )
                self._add_objective(factor, 3)
                continue
            # One pose target for both: one factor batch, one pass over the link's cache.
            paired.add(id(r))
            poses = wp.zeros(n, dtype=wp.transform, device=self.device)
            self._target_gathers.append(
                (_gather_pose_targets, [p.target_positions, r.target_rotations, self.problem_idx], poses)
            )
            offsets = list(p.link_offset) + list(r.link_offset_rotation)
            factor = pycunls.LinkPoseFactorBatch(
                self._topology,
                link,
                self._per_row(offsets),
                poses,
                self._per_row([p.weight] * 3 + [r.weight] * 3),
                n,
            )
            self._add_objective(factor, 6)
        for r in rotations:
            if id(r) in paired:
                continue
            factor = pycunls.LinkOrientationFactorBatch(
                self._topology,
                wp.full(n, r.link_index, dtype=wp.int32, device=self.device),
                self._per_row(r.link_offset_rotation),
                self._gathered(r.target_rotations),
                self._per_row([r.weight] * 3),
                n,
            )
            self._add_objective(factor, 3)

    def _add_loop_closure_factors(self, model: Model, scene, loop_joints: np.ndarray) -> None:
        from pycunls.articulation import loop_closure  # noqa: PLC0415

        joint_type = model.joint_type.numpy()
        parent = model.joint_parent.numpy()
        child = model.joint_child.numpy()
        # joint_X_p then joint_X_c per joint: the frame offsets of loop_closure's two slots.
        frames = np.concatenate([model.joint_X_p.numpy(), model.joint_X_c.numpy()], axis=1)
        axis = model.joint_axis.numpy()
        qd_start = model.joint_qd_start.numpy()
        link_of_body = scene.link_of_body
        self._loops = []  # keeps the constraint rows' batches alive
        for kind in _LOOP_JOINT_TYPES:
            joints = loop_joints[joint_type[loop_joints] == kind]
            if len(joints) == 0:
                continue
            m = len(joints)
            links = np.stack(
                [np.where(parent[joints] < 0, -1, link_of_body[parent[joints]]), link_of_body[child[joints]]], 1
            )
            has_axis = kind in (JointType.PRISMATIC, JointType.REVOLUTE)
            loop = loop_closure(
                self._topology,
                self._pycunls.ArticulationJointType(int(kind)),
                self._per_row(links, dtype=wp.int32),
                self._per_row(frames[joints]),
                self._per_row(axis[qd_start[joints]]) if has_axis else None,
                capacity=self.n_batch * m,
            )
            loop.set_num_active_factors(self.n_batch * m)
            rows = self._rows(m)
            self._factors += [(constraint, rows) for constraint in loop.constraints]
            self._has_constraints = True
            self._loops.append(loop)

    def _add_mimic_factors(self, model: Model) -> None:
        if model.joint_mimic_joint is None:
            return
        reference = model.joint_mimic_joint.numpy()
        coeffs = model.joint_mimic_coeffs.numpy()
        qd_start = model.joint_qd_start.numpy()
        topology_dof = np.full(model.joint_dof_count, -1, dtype=np.int32)
        topology_dof[self._dof_index] = np.arange(len(self._dof_index), dtype=np.int32)
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
            self._per_row(pairs[:, 0], dtype=wp.int32),
            self._per_row(pairs[:, 1], dtype=wp.int32),
            self._per_row(pairs[:, 2:4]),
            self.n_batch * m,
        )
        self._add_factor(factor, m, constraint=True)

    def _solver(self, iterations: int):
        solver = self._solvers.get(iterations)
        if solver is None:
            pycunls = self._pycunls
            lm = pycunls.LevenbergMarquardtMinimizerOptions()
            lm.base_options.max_num_iterations = iterations
            lm.base_options.sparse_linear_solver_type = pycunls.SparseLinearSolverType.BlockSparseCholesky
            # Absolute: residuals are meters and radians, so this stops at about 10 um. Much tighter
            # sits at the float32 rounding of forward kinematics a few meters from the origin.
            lm.base_options.cost_tolerance = 1.0e-10
            lm.base_options.state_tolerance = 1.0e-10
            options = pycunls.AugmentedLagrangianMinimizerOptions()
            options.constraint_tolerance = self.constraint_tolerance
            options.reuse_structure = True
            if not self._has_constraints:
                # `iterations` LM steps, nothing read back: the IKOptimizerLM contract. Constraints
                # need the converging augmented Lagrangian: a fixed budget can leave them open.
                options.real_time = True
                options.max_outer_iterations = 1
                options.inner_iterations = iterations
                options.final_inner_iterations = iterations
                options.inner_line_search_steps = 0
                options.max_bound_refinements = 0
            solver = pycunls.AugmentedLagrangianMinimizer(pycunls.LevenbergMarquardtMinimizer(lm), options)
            self._solvers[iterations] = solver
        return solver

    # ------------------------------------------------------------------
    # Optimizer interface
    # ------------------------------------------------------------------
    def _set_states(self, joint_q: wp.array2d[wp.float32], stream) -> None:
        """Copy ``joint_q`` into the cuNLS states (forward kinematics included)."""
        if self._coord_start is not None:
            self._states.set_joint_q(joint_q.flatten()[self._coord_start :], self.n_batch, stream, stride=self.n_coords)
            return
        wp.launch(
            _gather_coords, dim=self._q.shape, inputs=[joint_q, self._coord_map], outputs=[self._q], device=self.device
        )
        self._states.set_joint_q(self._q, self.n_batch, stream)

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
            iterations: Levenberg-Marquardt iterations of each row. With loop or mimic
                constraints, the limit of each augmented-Lagrangian inner solve; rows stop
                earlier once converged.
        """
        if joint_q_in.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")

        stream = wp.get_stream(self.device)  # Warp work and the solve stay in order on one stream
        for kernel, inputs, output in self._target_gathers:
            wp.launch(kernel, dim=self.n_batch, inputs=inputs, outputs=[output], device=self.device)
        self._set_states(joint_q_in, stream)
        with wp.ScopedDevice(self.device):
            self._solver(iterations).minimize(stream, self._problem)
        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q_out, joint_q_in)
        if self._coord_start is not None:
            self._states.get_joint_q(
                joint_q_out.flatten()[self._coord_start :], self.n_batch, stream, stride=self.n_coords
            )
            return
        self._states.get_joint_q(self._q, self.n_batch, stream)
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
        """Evaluate the objective costs of a batch of joint configurations.

        The cost of a row is the sum of its squared objective residuals as cuNLS evaluates
        them (weighted; orientation errors as rotation angles, so the sign of a quaternion does
        not matter). Joint limits are hard and loops and mimic joints are constraints, so they add
        nothing.

        Args:
            joint_q: Joint coordinates to evaluate, shape [n_batch, joint_coord_count].

        Returns:
            Costs for each batch row, shape [n_batch].
        """
        stream = wp.get_stream(self.device)
        self._set_states(joint_q, stream)
        self.costs.zero_()
        for factor, residuals in self._objective_factors:
            factor.evaluate_batch(residuals, None, self._state_table, stream)
            wp.launch(
                _add_squared_residuals, dim=self.n_batch, inputs=[residuals], outputs=[self.costs], device=self.device
            )
        return self.costs
