# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Projected DVI solver for Kamino dual forward-dynamics problems."""

from __future__ import annotations

import warp as wp

from ....config import DVISolverConfig
from ...core.data import DataKamino
from ...core.model import ModelKamino
from ...core.size import SizeKamino
from ...core.types import to_warp_int32_array
from ...dynamics.dual import DualProblem
from ...geometry.contacts import ContactsKamino
from ...kinematics.jacobians import SparseSystemJacobians
from ...kinematics.limits import LimitsKamino
from ...linalg import DenseLinearOperatorData, DenseSquareMultiLinearInfo, LLTBlockedRCMSolver, LLTBlockedSolver
from ..common import (
    WarmStartMode,
    apply_dual_preconditioner_to_solution,
    warmstart_contact_constraints,
    warmstart_joint_constraints,
    warmstart_limit_constraints,
)
from .kernels import (
    _apply_dvi_contact_jacobi_delta,
    _build_bilateral_rhs,
    _color_dvi_contacts,
    _compute_dvi_contact_block_inverse,
    _compute_dvi_contact_jacobi_delta,
    _compute_dvi_contact_velocities,
    _compute_dvi_desaxce_corrections,
    _compute_dvi_solution_vectors,
    _compute_dvi_status_residuals,
    _copy_bilateral_block,
    _initialize_dvi_status,
    _reset_dvi_solver_data,
    _reset_dvi_status,
    _scatter_bilateral_solution,
    _set_dvi_bilateral_active_dim,
    _set_dvi_direct_status_iterations,
    _solve_dvi_contacts_colored_gs,
    _solve_dvi_limits_pgs,
    _solve_dvi_pgs,
    _unprecondition_dvi_solution,
)
from .sparse import SparseDVIPath
from .types import DVIConfigStruct, DVIData, convert_config_to_struct

wp.set_module_options({"enable_backward": False})

float32 = wp.float32


class DVISolver:
    """Solve Kamino dual problems with projected DVI iterations.

    For Kamino's dual system ``v_plus = D * lambda + v_f``, bilateral rows
    enforce zero velocity, limit rows enforce nonnegative complementarity,
    and contact rows enforce Coulomb-cone complementarity after the De Saxce
    velocity correction.

    Bilateral constraints are solved as a direct block when available, while
    limits and frictional contacts use projected Gauss-Seidel, Jacobi, or
    graph-colored updates. Dense and matrix-free sparse problems share the
    same solution, warm-start, status, and diagnostics contract.
    """

    Config = DVISolverConfig

    def __init__(
        self,
        model: ModelKamino | None = None,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SparseSystemJacobians | None = None,
        problem: DualProblem | None = None,
        config: list[DVISolver.Config] | DVISolver.Config | None = None,
        warmstart: WarmStartMode = WarmStartMode.NONE,
        collect_info: bool = False,
    ):
        """Initialize a DVI solver and optionally allocate it for a model.

        Args:
            model: Model that determines solver allocation sizes.
            data: Model data used by sparse DVI operator products.
            limits: Limit topology used by sparse DVI updates.
            contacts: Contact topology used for graph-colored contact updates.
            jacobians: Sparse constraint Jacobians used by sparse DVI updates.
            problem: Optional sparse problem used to precompute topology outside
                the first simulation step.
            config: One DVI config or one config per world.
            warmstart: Source used to initialize constraint impulses.
            collect_info: Whether to retain terminal per-world diagnostics.
        """
        self._config: list[DVISolver.Config] = []
        self._warmstart: WarmStartMode = WarmStartMode.NONE
        self._collect_info: bool = False
        self._size: SizeKamino | None = None
        self._data: DVIData | None = None
        self._bilateral_solver: LLTBlockedSolver | LLTBlockedRCMSolver | None = None
        self._max_block_iterations: int = 1
        self._max_contact_iterations: int = 1
        self._max_iterations: int = 1
        self._bilateral_solve_after_block: tuple[bool, ...] = ()
        self._has_contact_block_preconditioner: bool = False
        self._has_unilateral_constraints: bool = False
        self._contact_bid_AB: wp.array[wp.vec2i] | None = None
        self._sparse_path: SparseDVIPath | None = None
        self._all_worlds_mask: wp.array[wp.bool] | None = None
        self._device: wp.DeviceLike = None

        if model is not None:
            self.finalize(
                model=model,
                data=data,
                limits=limits,
                contacts=contacts,
                jacobians=jacobians,
                problem=problem,
                config=config,
                warmstart=warmstart,
                collect_info=collect_info,
            )

    @property
    def config(self) -> list[DVISolver.Config]:
        """Host-side per-world DVI configs."""
        return self._config

    @property
    def size(self) -> SizeKamino:
        """Model size cache."""
        return self._size

    @property
    def data(self) -> DVIData:
        """Solver data arrays."""
        if self._data is None:
            raise RuntimeError("Solver data has not been allocated yet. Call `finalize()` first.")
        return self._data

    @property
    def device(self) -> wp.DeviceLike:
        """Device on which solver data is allocated."""
        return self._device

    @property
    def all_worlds_mask(self) -> wp.array[wp.bool]:
        """Boolean mask selecting every world for sparse operator products."""
        return self._all_worlds_mask

    def finalize(
        self,
        model: ModelKamino,
        data: DataKamino | None = None,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        jacobians: SparseSystemJacobians | None = None,
        problem: DualProblem | None = None,
        config: list[DVISolver.Config] | DVISolver.Config | None = None,
        warmstart: WarmStartMode = WarmStartMode.NONE,
        collect_info: bool = False,
    ):
        """Allocate solver data and precompute model-dependent topology.

        Args:
            model: Model that determines solver allocation sizes.
            data: Model data used by sparse DVI operator products.
            limits: Limit topology used by sparse DVI updates.
            contacts: Contact topology used for graph-colored contact updates.
            jacobians: Sparse constraint Jacobians used by sparse DVI updates.
            problem: Optional sparse problem used to precompute topology.
            config: One DVI config or one config per world.
            warmstart: Source used to initialize constraint impulses.
            collect_info: Whether to retain terminal per-world diagnostics.
        """
        if model is None or not isinstance(model, ModelKamino):
            raise ValueError("A model of type `ModelKamino` must be provided.")

        self._size = model.size
        self._device = model.device
        self._config = self._check_config(model, config)
        self._warmstart = warmstart
        self._collect_info = collect_info
        self._max_iterations = max(c.max_iterations for c in self._config)
        self._max_block_iterations = max(c.block_iterations for c in self._config)
        self._max_contact_iterations = max(c.contact_iterations for c in self._config)
        self._bilateral_solve_after_block = self._make_bilateral_solve_schedule(self._config)
        self._has_contact_block_preconditioner = any(c.contact_block_preconditioner for c in self._config)
        self._has_unilateral_constraints = self._size.max_of_max_limits > 0 or self._size.max_of_max_contacts > 0
        self._data = DVIData(size=self._size, collect_info=self._collect_info, device=self._device)
        self._all_worlds_mask = wp.ones(shape=(self._size.num_worlds,), dtype=wp.bool, device=self._device)
        self._allocate_bilateral_solver(model)
        self._sparse_path = SparseDVIPath(
            device=self._device,
            size=self._size,
            data=self._data,
            model=model,
            model_data=data,
            limits=limits,
            contacts=contacts,
            jacobians=jacobians,
            bilateral_solver=self._bilateral_solver,
            max_iterations=self._max_iterations,
            max_block_iterations=self._max_block_iterations,
            max_contact_iterations=self._max_contact_iterations,
            has_contact_block_preconditioner=self._has_contact_block_preconditioner,
            has_unilateral_constraints=self._has_unilateral_constraints,
            all_worlds_mask=self._all_worlds_mask,
            should_solve_bilateral_after_block=self._should_solve_bilateral_after_block,
        )
        self.set_contacts(contacts)
        if problem is not None and problem.sparse:
            self._sparse_path.prepare(problem)

        configs = [convert_config_to_struct(c) for c in self._config]
        with wp.ScopedDevice(self._device):
            self._data.config = wp.array(configs, dtype=DVIConfigStruct)

    def _make_bilateral_solve_schedule(self, configs: list[DVISolver.Config]) -> tuple[bool, ...]:
        """Return host-side repeated bilateral solve points for direct-block DVI."""
        return tuple(
            any(next_block < c.block_iterations and next_block % c.bilateral_solve_period == 0 for c in configs)
            for next_block in range(1, self._max_block_iterations)
        )

    def _should_solve_bilateral_after_block(self, block_iteration: int) -> bool:
        """Whether the direct bilateral block should be re-solved after this block."""
        if block_iteration < 0 or block_iteration >= len(self._bilateral_solve_after_block):
            return False
        return self._bilateral_solve_after_block[block_iteration]

    def _allocate_bilateral_solver(self, model: ModelKamino):
        """Allocate the reduced dense operator used for bilateral DVI solves."""
        self._bilateral_solver = None
        self._data.bilateral_operator = None
        if model.size.sum_of_num_joint_cts == 0:
            return

        joint_cts_per_world = model.info.num_joint_cts.numpy().astype(int).tolist()
        if any(njc <= 0 for njc in joint_cts_per_world):
            return

        mat_sizes = [njc * njc for njc in joint_cts_per_world]
        mat_offsets = [0]
        for size in mat_sizes[:-1]:
            mat_offsets.append(mat_offsets[-1] + size)

        operator = DenseLinearOperatorData()
        operator.info = DenseSquareMultiLinearInfo()
        operator.info.assign(
            maxdim=model.info.num_joint_cts,
            dim=model.info.num_joint_cts,
            mio=to_warp_int32_array(mat_offsets, device=self._device),
            vio=model.info.joint_cts_offset,
            dtype=float32,
            device=self._device,
        )
        operator.mat = wp.zeros(shape=(operator.info.total_mat_size,), dtype=float32, device=self._device)
        self._data.bilateral_operator = operator
        first_config = self._config[0]
        if any(
            config.bilateral_solver_type != first_config.bilateral_solver_type
            or config.bilateral_solver_kwargs != first_config.bilateral_solver_kwargs
            for config in self._config[1:]
        ):
            raise ValueError("All worlds must use the same DVI bilateral solver configuration.")

        solver_type = first_config.bilateral_solver_type
        kwargs = dict(first_config.bilateral_solver_kwargs)
        if solver_type == "LLTB":
            # A larger factorization tile reduces panel count, while the
            # single-RHS solve benefits from more threads on its smaller tile.
            kwargs.setdefault("factorize_block_size", 64)
            kwargs.setdefault("solve_block_dim", 256)
            solver_class = LLTBlockedSolver
        else:
            solver_class = LLTBlockedRCMSolver
        self._bilateral_solver = solver_class(operator=operator, device=self._device, **kwargs)

    @staticmethod
    def _check_config(
        model: ModelKamino | None = None, config: list[DVISolver.Config] | DVISolver.Config | None = None
    ) -> list[DVISolver.Config]:
        if config is None:
            config = [DVISolver.Config()] * (model.info.num_worlds if model else 1)
        elif isinstance(config, DVISolver.Config):
            config = [config] * (model.info.num_worlds if model else 1)
        elif isinstance(config, list):
            if model is not None and len(config) != model.info.num_worlds:
                raise ValueError(f"Expected {model.info.num_worlds} configs, got {len(config)}")
            if not all(isinstance(c, DVISolver.Config) for c in config):
                raise TypeError("All configs must be instances of DVISolver.Config")
        else:
            raise TypeError(f"Expected a single object or list of `DVISolver.Config`, got {type(config)}")
        return config

    def set_contacts(self, contacts: ContactsKamino | None):
        """Cache contact topology for graph-colored contact solves."""
        if contacts is not None and contacts.model_max_contacts_host > 0:
            self._contact_bid_AB = contacts.bid_AB
        else:
            self._contact_bid_AB = None

    def reset(self, problem: DualProblem | None = None, world_mask: wp.array[wp.bool] | None = None):
        """Reset scratch state and cached solution data."""
        self._data.state.reset()
        if self._data.info is not None:
            self._data.info.zero()
        if world_mask is None:
            self._data.solution.zero()
        else:
            if problem is None:
                raise ValueError("A `DualProblem` instance must be provided when a world mask is used.")
            wp.launch(
                kernel=_reset_dvi_solver_data,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    world_mask,
                    problem.data.vio,
                    problem.data.maxdim,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )

    def coldstart(self):
        """Prepare a cold-start solve."""
        self._data.state.reset()
        self._data.solution.zero()

    def warmstart(
        self,
        problem: DualProblem,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        """Prepare a warm-start solve."""
        self._data.state.reset()
        self.set_contacts(contacts)

        match self._warmstart:
            case WarmStartMode.NONE:
                self._data.solution.zero()
            case WarmStartMode.INTERNAL:
                self._warmstart_from_solution(problem)
            case WarmStartMode.CONTAINERS:
                self._warmstart_from_containers(problem, model, data, limits, contacts)
            case _:
                raise ValueError(f"Invalid warmstart mode: {self._warmstart}")

    def solve(self, problem: DualProblem):
        """Solve the cone-complementarity problem defined by ``problem``.

        Kamino supplies the constraint-space system

        ``v_plus = D * lambda + v_f``,

        where ``D`` is Kamino's represented Delassus operator derived from
        ``J * M^-1 * J^T``, ``lambda`` contains joint, limit, and contact
        impulses, and ``v_f`` contains unconstrained motion, stabilization,
        and restitution. The bilateral equation ``D * lambda = -v_f`` is the
        standalone ``N * lambda = b`` formulation with ``b = -v_f``.

        Rows are ordered as bilateral joints, unilateral limits, and contact
        triplets ``[t0, t1, n]``. Contact rows use
        ``v_aug = v_plus + [0, 0, mu * norm(v_t)]``.

        The DVI solution satisfies zero ``v_aug`` on bilateral rows,
        nonnegative complementarity on limit rows, and Coulomb-cone
        complementarity between contact impulses and augmented velocities. DVI
        exploits this difference by partitioning the system into bilateral
        impulses ``lambda_b`` and unilateral limit/contact impulses
        ``lambda_u``. When the bilateral block is available, it is factored and
        solved directly:

        ``D_bb * lambda_b = -(v_f,b + D_bu * lambda_u)``.

        The unilateral block is updated iteratively with projection onto the
        nonnegative and Coulomb cones. Alternating these updates retains the
        ``D_bu`` and ``D_ub`` coupling while using a solver suited to each
        constraint class. Repeating the alternation for ``block_iterations``
        drives ``lambda_b`` and ``lambda_u`` toward a mutually consistent
        solution; a single block without a bilateral re-solve reduces to a
        one-directional solve where the joints never see the final contact and
        limit impulses. The fallback path instead applies projected
        Gauss-Seidel to all rows.

        This differs from Kamino's PADMM backend, which places all constraint
        rows in one proximal-ADMM iteration: it solves a regularized full
        Delassus system for the unconstrained primal update, then projects the
        unilateral components. DVI uses no ADMM penalty or auxiliary-variable
        iteration; its primary split is direct bilateral versus projected
        iterative unilateral solves. Dense and sparse DVI paths implement the
        same split with different Delassus representations.

        Args:
            problem: Unified Kamino dual problem to solve.
        """
        wp.launch(
            kernel=_reset_dvi_status,
            dim=self._size.num_worlds,
            inputs=[self._data.status],
            device=self.device,
        )

        if problem.sparse:
            if self._sparse_path is None:
                raise RuntimeError("Sparse DVI path has not been allocated. Call `finalize()` first.")
            if self._sparse_path.bilateral_nzb_pairs is None:
                self._sparse_path.prepare(problem)
            # Apply projected iterations through matrix-free products
            # D * lambda = J * (M^-1 * (J^T * lambda)) + R * lambda.
            self._sparse_path.solve(problem)
        elif self._has_contact_block_preconditioner and self._size.max_of_max_contacts > 0:
            # For each contact c, form B_c = (D_cc + regularization * I)^-1.
            wp.launch(
                kernel=_compute_dvi_contact_block_inverse,
                dim=(self._size.num_worlds, self._size.max_of_max_contacts),
                inputs=[
                    problem.data.dim,
                    problem.data.mio,
                    problem.data.nc,
                    problem.data.ccgo,
                    problem.data.cio,
                    problem.data.D,
                    self._data.config,
                    self._data.state.contact_block_inv,
                ],
                device=self.device,
            )

        if not problem.sparse:
            if self._bilateral_solver is not None and self._data.bilateral_operator is not None:
                self._solve_with_bilateral_direct_block(problem)
            else:
                # Solve all rows together with projected Gauss-Seidel:
                # lambda_next = projection(lambda - omega * B * v_aug).
                wp.launch(
                    kernel=_solve_dvi_pgs,
                    dim=self._size.num_worlds,
                    inputs=[
                        problem.data.dim,
                        problem.data.mio,
                        problem.data.vio,
                        problem.data.njc,
                        problem.data.nl,
                        problem.data.nc,
                        problem.data.lcgo,
                        problem.data.ccgo,
                        problem.data.cio,
                        problem.data.mu,
                        problem.data.D,
                        problem.data.v_f,
                        self._data.state.contact_block_inv,
                        self._data.config,
                        self._data.status,
                        self._data.solution.lambdas,
                    ],
                    device=self.device,
                )

            # Evaluate the physical post-event velocity v_plus = D * lambda + v_f.
            wp.launch(
                kernel=_compute_dvi_solution_vectors,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    problem.data.dim,
                    problem.data.mio,
                    problem.data.vio,
                    problem.data.D,
                    problem.data.v_f,
                    self._data.state.s,
                    self._data.state.v_aug,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )

        if self._size.max_of_max_contacts > 0:
            # Map physical contact velocity to the dual-cone variable
            # v_aug = v_plus + [0, 0, mu * norm(v_t)].
            wp.launch(
                kernel=_compute_dvi_desaxce_corrections,
                dim=(self._size.num_worlds, self._size.max_of_max_contacts),
                inputs=[
                    problem.data.nc,
                    problem.data.ccgo,
                    problem.data.cio,
                    problem.data.vio,
                    problem.data.mu,
                    self._data.state.s,
                    self._data.state.v_aug,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )

        # Classify the final iterate using all DVI conditions. This replaces
        # provisional iterate-change convergence from the dense fallback;
        # direct and sparse paths reach this check after fixed iteration counts.
        wp.launch(
            kernel=_compute_dvi_status_residuals,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.dim,
                problem.data.vio,
                problem.data.njc,
                problem.data.nl,
                problem.data.nc,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.cio,
                problem.data.mu,
                self._data.config,
                self._data.state.v_aug,
                self._data.solution.lambdas,
                self._data.status,
            ],
            device=self.device,
        )

        if self._collect_info:
            wp.copy(self._data.info.status, self._data.status)

        wp.launch(
            kernel=_unprecondition_dvi_solution,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.state.s,
                self._data.state.v_aug,
                self._data.solution.lambdas,
                self._data.solution.v_plus,
            ],
            device=self.device,
        )

    def _solve_bilateral_block(self, problem: DualProblem, active_dim: wp.array[wp.int32] | None = None):
        """Solve ``D_bb * lambda_b = -(v_f,b + D_bu * lambda_u)``."""
        operator = self._data.bilateral_operator
        state = self._data.state
        wp.launch(
            kernel=_build_bilateral_rhs,
            dim=(self._size.num_worlds, self._size.max_of_num_joint_cts),
            inputs=[
                problem.data.dim,
                problem.data.mio,
                problem.data.vio,
                problem.data.njc,
                problem.data.D,
                problem.data.v_f,
                operator.info.vio,
                state.bilateral_preconditioner,
                self._data.solution.lambdas,
                state.bilateral_rhs,
            ],
            device=self.device,
        )
        full_dim = operator.info.dim
        if active_dim is not None:
            operator.info.dim = active_dim
        try:
            self._bilateral_solver.solve(b=state.bilateral_rhs, x=state.bilateral_solution)
        finally:
            operator.info.dim = full_dim
        wp.launch(
            kernel=_scatter_bilateral_solution,
            dim=(self._size.num_worlds, self._size.max_of_num_joint_cts),
            inputs=[
                problem.data.vio,
                problem.data.njc,
                operator.info.vio,
                state.bilateral_preconditioner,
                state.bilateral_solution,
                self._data.solution.lambdas,
            ],
            device=self.device,
        )

    def _factor_bilateral_block(self, problem: DualProblem):
        """Extract, symmetrically scale, and factor the bilateral block ``D_bb``."""
        operator = self._data.bilateral_operator
        operator.info.dim = operator.info.maxdim
        wp.launch(
            kernel=_copy_bilateral_block,
            dim=(self._size.num_worlds, self._size.max_of_num_joint_cts * self._size.max_of_num_joint_cts),
            inputs=[
                problem.data.dim,
                problem.data.mio,
                problem.data.njc,
                problem.data.D,
                operator.info.mio,
                operator.info.vio,
                operator.mat,
                self._data.state.bilateral_preconditioner,
            ],
            device=self.device,
        )
        self._bilateral_solver.compute(A=operator.mat)

    def _solve_with_bilateral_direct_block(self, problem: DualProblem):
        """Alternate a direct bilateral solve with projected unilateral updates.

        With unilateral impulses fixed, the direct solve satisfies
        ``D_bb * lambda_b = -(v_f,b + D_bu * lambda_u)``. Repeating these
        updates preserves bilateral-unilateral coupling. Between direct solves,
        limits and contacts apply projected updates using the unilateral
        residual ``D_ub * lambda_b + D_uu * lambda_u + v_f,u``. As the block
        count grows the two impulse sets converge to a mutually consistent
        solution; one block without a bilateral re-solve corresponds to a
        one-directional joint-then-contact solve.
        """
        self._factor_bilateral_block(problem)
        self._solve_bilateral_block(problem)
        if not self._has_unilateral_constraints:
            return

        wp.launch(
            kernel=_initialize_dvi_status,
            dim=self._size.num_worlds,
            inputs=[
                self._data.config,
                self._data.status,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_set_dvi_bilateral_active_dim,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.njc,
                problem.data.nl,
                problem.data.nc,
                self._data.state.bilateral_active_dim,
            ],
            device=self.device,
        )

        use_colored_contacts = (
            self._size.max_of_max_contacts > 0 and self.device.is_cuda and self._contact_bid_AB is not None
        )
        if use_colored_contacts:
            wp.launch(
                kernel=_color_dvi_contacts,
                dim=self._size.num_worlds,
                inputs=[
                    problem.data.nc,
                    problem.data.cio,
                    self._contact_bid_AB,
                    self._data.state.contact_colors,
                    self._data.state.contact_num_colors,
                ],
                device=self.device,
            )

        for block_iteration in range(self._max_block_iterations):
            if self._size.max_of_max_limits > 0:
                wp.launch(
                    kernel=_solve_dvi_limits_pgs,
                    dim=self._size.num_worlds,
                    inputs=[
                        problem.data.dim,
                        problem.data.mio,
                        problem.data.vio,
                        problem.data.nl,
                        problem.data.lcgo,
                        problem.data.D,
                        problem.data.v_f,
                        block_iteration,
                        self._data.config,
                        self._data.status,
                        self._data.solution.lambdas,
                    ],
                    device=self.device,
                )

            if self._size.max_of_max_contacts > 0:
                wp.launch(
                    kernel=_compute_dvi_contact_velocities,
                    dim=(self._size.num_worlds, 3 * self._size.max_of_max_contacts),
                    inputs=[
                        problem.data.dim,
                        problem.data.mio,
                        problem.data.vio,
                        problem.data.nc,
                        problem.data.ccgo,
                        problem.data.D,
                        problem.data.v_f,
                        self._data.solution.lambdas,
                        self._data.state.v_aug,
                    ],
                    device=self.device,
                )

                if use_colored_contacts:
                    wp.launch(
                        kernel=_solve_dvi_contacts_colored_gs,
                        dim=self._size.num_worlds * 64,
                        inputs=[
                            problem.data.dim,
                            problem.data.mio,
                            problem.data.vio,
                            problem.data.nc,
                            problem.data.ccgo,
                            problem.data.cio,
                            problem.data.mu,
                            problem.data.D,
                            block_iteration,
                            self._data.state.contact_block_inv,
                            self._data.state.contact_colors,
                            self._data.state.contact_num_colors,
                            self._data.config,
                            self._data.state.v_aug,
                            self._data.solution.lambdas,
                        ],
                        device=self.device,
                        block_dim=64,
                    )
                else:
                    for contact_iteration in range(self._max_contact_iterations):
                        wp.launch(
                            kernel=_compute_dvi_contact_jacobi_delta,
                            dim=(self._size.num_worlds, self._size.max_of_max_contacts),
                            inputs=[
                                problem.data.dim,
                                problem.data.mio,
                                problem.data.vio,
                                problem.data.nc,
                                problem.data.ccgo,
                                problem.data.cio,
                                problem.data.mu,
                                problem.data.D,
                                block_iteration,
                                contact_iteration,
                                self._data.config,
                                self._data.state.contact_block_inv,
                                self._data.state.v_aug,
                                self._data.solution.lambdas,
                                self._data.state.scratch,
                            ],
                            device=self.device,
                        )
                        wp.launch(
                            kernel=_apply_dvi_contact_jacobi_delta,
                            dim=(self._size.num_worlds, 3 * self._size.max_of_max_contacts),
                            inputs=[
                                problem.data.dim,
                                problem.data.mio,
                                problem.data.vio,
                                problem.data.nc,
                                problem.data.ccgo,
                                problem.data.D,
                                block_iteration,
                                contact_iteration,
                                self._data.config,
                                self._data.state.scratch,
                                self._data.state.v_aug,
                            ],
                            device=self.device,
                        )

            if self._should_solve_bilateral_after_block(block_iteration):
                self._solve_bilateral_block(problem, active_dim=self._data.state.bilateral_active_dim)

        self._solve_bilateral_block(problem, active_dim=self._data.state.bilateral_active_dim)

        wp.launch(
            kernel=_set_dvi_direct_status_iterations,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.nl,
                problem.data.nc,
                self._data.config,
                self._data.status,
            ],
            device=self.device,
        )

    def _warmstart_from_solution(self, problem: DualProblem):
        wp.launch(
            kernel=apply_dual_preconditioner_to_solution,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.solution.lambdas,
                self._data.solution.v_plus,
            ],
            device=self.device,
        )

    def _warmstart_from_containers(
        self,
        problem: DualProblem,
        model: ModelKamino,
        data: DataKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
    ):
        self._data.solution.zero()
        if model.size.sum_of_num_joints > 0:
            wp.launch(
                kernel=warmstart_joint_constraints,
                dim=model.size.sum_of_num_joints,
                inputs=[
                    model.time.dt,
                    model.joints.wid,
                    model.joints.num_dynamic_cts,
                    model.joints.num_kinematic_cts,
                    model.joints.dynamic_cts_offset_joint_cts,
                    model.joints.kinematic_cts_offset_joint_cts,
                    model.joints.dynamic_cts_offset_total_cts,
                    model.joints.kinematic_cts_offset_total_cts,
                    data.joints.lambda_j,
                    problem.data.P,
                    self._data.solution.lambdas,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )
        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                kernel=warmstart_limit_constraints,
                dim=limits.model_max_limits_host,
                inputs=[
                    model.time.dt,
                    model.info.total_cts_offset,
                    data.info.limit_cts_group_offset,
                    limits.model_active_limits,
                    limits.wid,
                    limits.lid,
                    limits.reaction,
                    limits.velocity,
                    problem.data.P,
                    self._data.solution.lambdas,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )
        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                kernel=warmstart_contact_constraints,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    model.time.dt,
                    model.info.total_cts_offset,
                    data.info.contact_cts_group_offset,
                    contacts.model_active_contacts,
                    contacts.wid,
                    contacts.cid,
                    contacts.material,
                    contacts.reaction,
                    contacts.velocity,
                    problem.data.P,
                    self._data.solution.lambdas,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )
