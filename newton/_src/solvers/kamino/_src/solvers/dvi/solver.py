# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Projected DVI solver for Kamino dual forward-dynamics problems."""

from __future__ import annotations

import warp as wp

from ....config import DVISolverConfig
from ...core.data import DataKamino
from ...core.model import ModelKamino
from ...core.size import SizeKamino
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
from .apgd import ContactAPGDOptions, ContactAPGDSolver, DenseContactOperator
from .kernels import (
    _FUSED_SINGLE_FAMILY_BLOCK,
    _INEQUALITY_FAMILY_CONTACTS,
    _INEQUALITY_FAMILY_LIMITS,
    _accumulate_dvi_apgd_status,
    _build_bilateral_rhs,
    _compute_dvi_desaxce_corrections,
    _compute_dvi_solution_vectors,
    _compute_dvi_status_residuals,
    _compute_dvi_unilateral_velocities,
    _copy_bilateral_block,
    _initialize_dvi_status,
    _reset_dvi_solver_data,
    _reset_dvi_status,
    _scale_dvi_tangential_warmstart,
    _scatter_bilateral_solution,
    _set_dvi_bilateral_active_dim,
    _set_dvi_contact_active_mask,
    _set_dvi_direct_status_iterations,
    _solve_dvi_inequalities_colored_pgs,
    _unprecondition_dvi_solution,
)
from .sparse import SparseDVIPath
from .sparse_kernels import (
    _color_mapped_dvi_inequalities,
    _map_active_contacts,
    _map_active_limits,
    _map_bounded_constraints,
)
from .types import DVIConfigStruct, DVIData, convert_config_to_struct

wp.set_module_options({"enable_backward": False})

float32 = wp.float32


class DVISolver:
    """Solve Kamino dual problems with projected DVI iterations.

    For Kamino's dual system ``v_plus = D * lambda + v_f``, bilateral rows
    enforce zero effective velocity, limit rows enforce nonnegative
    complementarity in the effective velocity, and contact rows enforce a
    selected Coulomb law. PGS uses the
    non-associated De Saxce correction; APGD solves the associated cone QP.

    Every coupling sweep evaluates limits, then the direct bilateral block,
    then contacts (``L -> B -> C``), with an optional post-stabilization
    bilateral refresh. Dense and matrix-free sparse problems share the same
    solution, warm-start, status, and diagnostics contract.
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
        self._contact_solver: str = "pgs"
        self._contact_law: str = "de_saxce"
        self._contact_apgd: ContactAPGDSolver | None = None
        self._dense_contact_operator: DenseContactOperator | None = None
        self._dense_contact_problem: DualProblem | None = None
        self._max_alternating_iterations: int = 1
        self._has_unilateral_constraints: bool = False
        self._has_limit_constraints: bool = False
        self._has_contact_constraints: bool = False
        self._has_post_stabilization_bilateral: bool = False
        self._limits: LimitsKamino | None = None
        self._contacts: ContactsKamino | None = None
        self._sparse_path: SparseDVIPath | None = None
        self._all_worlds_mask: wp.array[wp.bool] | None = None
        self._device: wp.DeviceLike = None
        self._num_joints: int = 0
        self._joint_wid: wp.array[wp.int32] | None = None
        self._joint_bid_B: wp.array[wp.int32] | None = None
        self._joint_bid_F: wp.array[wp.int32] | None = None
        self._joint_bounded_cts_offset: wp.array[wp.int32] | None = None

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
        self._num_joints = model.size.sum_of_num_joints
        self._joint_wid = model.joints.wid
        self._joint_bid_B = model.joints.bid_B
        self._joint_bid_F = model.joints.bid_F
        self._joint_bounded_cts_offset = model.joints.bounded_cts_offset
        self._config = self._check_config(model, config)
        contact_solvers = {c.contact_solver for c in self._config}
        contact_laws = {c.resolved_contact_law for c in self._config}
        if len(contact_solvers) != 1 or len(contact_laws) != 1:
            raise ValueError("All worlds must use the same DVI contact solver and contact law.")
        self._contact_solver = self._config[0].contact_solver
        self._contact_law = self._config[0].resolved_contact_law
        self._warmstart = warmstart
        self._collect_info = collect_info
        self._max_alternating_iterations = max(c.max_alternating_iterations for c in self._config)
        self._has_limit_constraints = self._size.max_of_max_limits > 0 or self._size.max_of_num_bounded_joint_cts > 0
        self._has_contact_constraints = self._size.max_of_max_contacts > 0
        self._has_unilateral_constraints = self._has_limit_constraints or self._has_contact_constraints
        self._has_post_stabilization_bilateral = any(c.post_stabilization_bilateral for c in self._config)
        self._data = DVIData(size=self._size, collect_info=self._collect_info, device=self._device)
        self._all_worlds_mask = wp.ones(shape=(self._size.num_worlds,), dtype=wp.bool, device=self._device)
        self._allocate_contact_solver(contacts, problem)
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
            contact_solver=self._contact_solver,
            contact_apgd=self._contact_apgd,
            max_alternating_iterations=self._max_alternating_iterations,
            has_unilateral_constraints=self._has_unilateral_constraints,
            has_limit_constraints=self._has_limit_constraints,
            has_contact_constraints=self._has_contact_constraints,
            has_post_stabilization_bilateral=self._has_post_stabilization_bilateral,
            all_worlds_mask=self._all_worlds_mask,
            set_bilateral_active_dim=self._set_bilateral_active_dim,
        )
        self._limits = limits
        self.set_contacts(contacts)
        self._validate_inequality_topology()
        if problem is not None and problem.sparse:
            self._sparse_path.prepare(problem)

        configs = [convert_config_to_struct(c) for c in self._config]
        with wp.ScopedDevice(self._device):
            self._data.config = wp.array(configs, dtype=DVIConfigStruct)

    def _set_bilateral_active_dim(self, problem: DualProblem, block_iteration: int) -> None:
        """Select worlds whose bilateral block is active for a scheduled solve."""
        wp.launch(
            kernel=_set_dvi_bilateral_active_dim,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.njc,
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                block_iteration,
                self._data.config,
                self._data.state.bilateral_active_dim,
            ],
            device=self.device,
        )

    def _allocate_contact_solver(self, contacts: ContactsKamino | None, problem: DualProblem | None) -> None:
        """Allocate the opt-in contact backend and all graph-time workspace."""
        self._contact_apgd = None
        self._dense_contact_operator = None
        self._dense_contact_problem = None
        if self._contact_solver != "apgd":
            return

        if contacts is None:
            if self._size.max_of_max_contacts > 0:
                raise ValueError("The APGD contact solver requires a ContactsKamino container.")
            contact_capacities = [0] * self._size.num_worlds
        else:
            contact_capacities = contacts.world_max_contacts_host
        if len(contact_capacities) != self._size.num_worlds:
            raise ValueError("The APGD contact capacities must contain one entry per world.")

        options = [
            ContactAPGDOptions(
                max_iterations=config.apgd.max_iterations,
                max_backtrack_iterations=config.apgd.max_backtrack_iterations,
                tolerance=config.apgd.tolerance,
                min_iterations=config.apgd.min_iterations,
                early_exit=config.apgd.early_exit,
                use_graph_conditionals=config.apgd.use_graph_conditionals,
            )
            for config in self._config
        ]
        self._contact_apgd = ContactAPGDSolver(contact_capacities, options, device=self.device)
        if problem is not None and not problem.sparse:
            self._dense_contact_operator = self._make_dense_contact_operator(problem)
            self._dense_contact_problem = problem

    def _make_dense_contact_operator(self, problem: DualProblem) -> DenseContactOperator:
        """Bind APGD to the represented dense contact block without copying it."""
        if self._contact_apgd is None:
            raise RuntimeError("The APGD contact solver has not been allocated.")
        if problem.sparse or problem.data.D is None:
            raise TypeError("Dense contact APGD requires a dense DualProblem.")
        return self._contact_apgd.make_dense_operator(
            problem_dim=problem.data.dim,
            problem_mio=problem.data.mio,
            problem_vio=problem.data.vio,
            problem_nc=problem.data.nc,
            problem_cio=problem.data.cio,
            problem_ccgo=problem.data.ccgo,
            matrix=problem.data.D,
            represented_compliance=problem.data.E_hat,
            free_velocity=problem.data.v_f,
        )

    def _allocate_bilateral_solver(self, model: ModelKamino):
        """Allocate the reduced dense operator used for bilateral DVI solves."""
        self._bilateral_solver = None
        self._data.bilateral_operator = None
        if model.size.sum_of_num_bilateral_joint_cts == 0:
            return

        bilateral_joint_cts_per_world = model.info.num_joint_bilateral_cts.numpy().astype(int).tolist()
        # LLT metadata requires positive blocks; assembly makes zero-row worlds disconnected identities.
        factor_dims = [max(1, njc) for njc in bilateral_joint_cts_per_world]

        operator = DenseLinearOperatorData()
        operator.info = DenseSquareMultiLinearInfo()
        operator.info.finalize(factor_dims, dtype=float32, device=self._device)
        operator.mat = wp.zeros(shape=(operator.info.total_mat_size,), dtype=float32, device=self._device)
        self._data.state.bilateral_rhs = wp.zeros(operator.info.total_vec_size, dtype=float32, device=self._device)
        self._data.state.bilateral_solution = wp.zeros(operator.info.total_vec_size, dtype=float32, device=self._device)
        self._data.state.bilateral_preconditioner = wp.zeros(
            operator.info.total_vec_size, dtype=float32, device=self._device
        )
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
        """Cache contact topology for graph-colored inequality solves."""
        self._contacts = contacts
        if self._sparse_path is not None:
            self._sparse_path.contacts = contacts

    def reset(self, problem: DualProblem | None = None, world_mask: wp.array[wp.bool] | None = None):
        """Reset scratch state and cached solution data."""
        if world_mask is None:
            self._data.state.reset()
            if self._data.info is not None:
                self._data.info.zero()
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
        if limits is None:
            limits = self._limits
        else:
            self._limits = limits
            if self._sparse_path is not None:
                self._sparse_path.limits = limits
        if contacts is None:
            contacts = self._contacts
        else:
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

        Kamino supplies the represented constraint-space system

        ``v_plus = D * lambda + v_f``,

        where ``D`` is Kamino's preconditioned Delassus operator derived from
        ``J * M^-1 * J^T``, ``lambda`` contains joint, limit, and contact
        impulses, and ``v_f`` contains unconstrained motion, stabilization,
        and restitution. For compliant constraints the solve uses
        ``v_eff = D * lambda + v_f + E_hat * lambda``, where
        ``E_hat = P * E * P`` is the compliance diagonal represented in solver
        coordinates. The exported physical ``v_plus`` excludes this
        constitutive compliance term. The bilateral equation includes its
        configured constitutive diagonal while retaining ``b = -v_f``.

        Rows are ordered as bilateral joints, unilateral limits, and contact
        triplets ``[t0, t1, n]``. The compatibility PGS path uses the
        non-associated De Saxce velocity
        ``v_aug = v_plus + [0, 0, mu * norm(v_t)]``; APGD instead uses the
        associated contact velocity directly.

        The DVI solution satisfies zero effective constraint velocity on
        bilateral rows, projected bounds and nonnegative complementarity in
        the effective velocity on joint-side unilateral rows, and the selected
        Coulomb-cone contact law. DVI
        partitions these rows into ``L`` (bounded joint rows and joint limits),
        ``B`` (bilateral joint rows), and ``C`` (contact triplets). When the
        bilateral block is available, it is factored once and solved directly:

        ``(D_bb + E_hat_bb) * lambda_b = -(v_f,b + D_bu * lambda_u)``.

        Every coupling sweep then executes ``L -> B -> C`` and skips empty
        families. This retains all cross-family Delassus coupling while using a
        solver suited to each constraint class. Repeating the schedule for
        ``max_alternating_iterations`` drives the families toward a mutually
        consistent solution. By default the terminal contact phase is freshest:
        the final ``B`` solve includes the same-sweep ``L`` update but not the
        final ``C`` increment. ``post_stabilization_bilateral=True`` adds an
        optional terminal ``B`` refresh. When no bilateral block exists, the
        projected ``L`` and ``C`` phases retain the same ordering and skip
        ``B``.

        This differs from Kamino's PADMM backend, which places all constraint
        rows in one proximal-ADMM iteration: it solves a regularized full
        Delassus system for the unconstrained primal update, then projects the
        unilateral components. DVI uses no ADMM penalty or auxiliary-variable
        iteration; DVI instead uses explicit family phases. Dense and sparse
        DVI paths implement the same schedule with different Delassus
        representations.

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

        if not problem.sparse:
            if self._bilateral_solver is not None and self._data.bilateral_operator is not None:
                self._solve_with_bilateral_direct_block(problem)
            elif self._can_use_dense_inequality_pgs():
                self._solve_dense_family_pgs(problem)

            # Evaluate the physical post-event velocity v_plus = D * lambda + v_f.
            wp.launch(
                kernel=_compute_dvi_solution_vectors,
                dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
                inputs=[
                    problem.data.dim,
                    problem.data.mio,
                    problem.data.vio,
                    problem.data.D,
                    problem.data.E_hat,
                    problem.data.v_f,
                    self._data.state.s,
                    self._data.state.v_aug,
                    self._data.solution.lambdas,
                    self._data.solution.v_plus,
                ],
                device=self.device,
            )

        if self._size.max_of_max_contacts > 0 and self._contact_law == "de_saxce":
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
                ],
                device=self.device,
            )

        # Convert represented solver values to physical constraint units before
        # classifying convergence. Contact triplets use one common P entry, so
        # both the associated velocity and the homogeneous De Saxce correction
        # transform component-wise through this operation.
        wp.launch(
            kernel=_unprecondition_dvi_solution,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.vio,
                problem.data.P,
                self._data.status,
                self._data.state.s,
                self._data.state.v_aug,
                self._data.solution.lambdas,
                self._data.solution.v_plus,
            ],
            device=self.device,
        )

        # Classify the final physical iterate using all DVI conditions. This
        # replaces provisional iterate-change convergence from the dense
        # fallback; direct and sparse paths reach this check after fixed counts.
        wp.launch(
            kernel=_compute_dvi_status_residuals,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.dim,
                problem.data.vio,
                problem.data.njc,
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                problem.data.bcgo,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.bcio,
                problem.data.cio,
                problem.data.mu,
                problem.data.P,
                problem.data.bound_lower,
                problem.data.bound_upper,
                self._data.config,
                self._data.state.v_aug,
                self._data.solution.lambdas,
                self._data.status,
            ],
            device=self.device,
        )

        if self._collect_info:
            wp.copy(self._data.info.status, self._data.status)

    def _validate_inequality_topology(self) -> None:
        """Require the topology that graph-colored inequality solves consume.

        Checked at allocation so a model with limit or contact capacity fails
        before the first solve, which may run inside a captured graph.
        """
        if self._size.max_of_max_limits > 0 and self._limits is None:
            raise ValueError("DVI requires `limits` when the model allocates joint limits.")
        if self._size.max_of_max_contacts > 0 and self._contacts is None:
            raise ValueError("DVI requires `contacts` when the model allocates contacts.")

    def _prepare_inequality_coloring(self, problem: DualProblem) -> None:
        """Map and color active inequalities with the multi-world fast path."""
        state = self._data.state
        self._validate_inequality_topology()
        if self._num_joints > 0 and self._size.max_of_num_bounded_joint_cts > 0:
            wp.launch(
                kernel=_map_bounded_constraints,
                dim=self._num_joints,
                inputs=[
                    self._joint_wid,
                    self._joint_bid_B,
                    self._joint_bid_F,
                    self._joint_bounded_cts_offset,
                    problem.data.bcio,
                    problem.data.iio,
                    state.inequality_bodies,
                ],
                device=self.device,
            )
        limits = self._limits
        if limits is not None and limits.model_max_limits_host > 0:
            wp.launch(
                kernel=_map_active_limits,
                dim=limits.model_max_limits_host,
                inputs=[
                    limits.model_active_limits,
                    limits.wid,
                    limits.lid,
                    limits.bids,
                    problem.data.lio,
                    problem.data.iio,
                    problem.data.nbc,
                    state.limit_indices,
                    state.inequality_bodies,
                ],
                device=self.device,
            )
        contacts = self._contacts
        if contacts is not None and contacts.model_max_contacts_host > 0:
            wp.launch(
                kernel=_map_active_contacts,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    contacts.model_active_contacts,
                    contacts.wid,
                    contacts.cid,
                    contacts.bid_AB,
                    problem.data.nbc,
                    problem.data.nl,
                    problem.data.cio,
                    problem.data.iio,
                    state.contact_indices,
                    state.inequality_bodies,
                ],
                device=self.device,
            )
        state.inequality_body_color_masks.zero_()
        wp.launch(
            kernel=_color_mapped_dvi_inequalities,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                problem.data.iio,
                state.inequality_bodies,
                state.inequality_body_color_masks,
                state.inequality_colors,
                state.inequality_num_colors,
                state.inequality_ids_by_color,
                state.inequality_color_starts,
            ],
            device=self.device,
        )

    def _can_use_dense_inequality_pgs(self) -> bool:
        return self._has_unilateral_constraints and self._size.sum_of_num_bilateral_joint_cts == 0

    def _initialize_projected_iterations(self, problem: DualProblem) -> None:
        """Initialize status and graph coloring shared by projected schedules."""
        wp.launch(
            kernel=_initialize_dvi_status,
            dim=self._size.num_worlds,
            inputs=[self._data.config, self._data.status],
            device=self.device,
        )
        self._prepare_inequality_coloring(problem)

    def _refresh_dense_unilateral_velocities(self, problem: DualProblem) -> None:
        """Evaluate all unilateral rows from the latest unified impulse vector."""
        wp.launch(
            kernel=_compute_dvi_unilateral_velocities,
            dim=(self._size.num_worlds, self._size.max_of_max_total_cts),
            inputs=[
                problem.data.dim,
                problem.data.mio,
                problem.data.vio,
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                problem.data.bcgo,
                problem.data.D,
                problem.data.E_hat,
                problem.data.v_f,
                self._data.solution.lambdas,
                self._data.state.v_aug,
            ],
            device=self.device,
        )

    def _launch_dense_inequality_pgs(
        self,
        problem: DualProblem,
        block_iteration: int,
        inequality_family: int,
    ) -> None:
        """Launch one dense projected block for the selected row family."""
        state = self._data.state
        threads_per_world = 64 if self.device.is_cuda else 1
        wp.launch(
            kernel=_solve_dvi_inequalities_colored_pgs,
            dim=self._size.num_worlds * threads_per_world,
            inputs=[
                problem.data.dim,
                problem.data.mio,
                problem.data.vio,
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                problem.data.bcgo,
                problem.data.lcgo,
                problem.data.ccgo,
                problem.data.bcio,
                problem.data.cio,
                problem.data.iio,
                problem.data.mu,
                problem.data.bound_lower,
                problem.data.bound_upper,
                problem.data.D,
                problem.data.E_hat,
                block_iteration,
                inequality_family,
                state.inequality_num_colors,
                state.inequality_ids_by_color,
                state.inequality_color_starts,
                self._data.config,
                state.v_aug,
                self._data.solution.lambdas,
            ],
            device=self.device,
            block_dim=threads_per_world,
        )

    def _set_projected_iteration_status(self, problem: DualProblem, *, single_contact_phase: bool = False) -> None:
        """Record the configured projected work budget for each world."""
        wp.launch(
            kernel=_set_dvi_direct_status_iterations,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.nbc,
                problem.data.nl,
                problem.data.nc,
                single_contact_phase,
                self._data.config,
                self._data.status,
            ],
            device=self.device,
        )

    def _solve_dense_limit_phase(self, problem: DualProblem, block_iteration: int) -> None:
        """Solve bounded joint rows and joint limits from the latest iterate."""
        self._refresh_dense_unilateral_velocities(problem)
        self._launch_dense_inequality_pgs(problem, block_iteration, _INEQUALITY_FAMILY_LIMITS)

    def _solve_dense_contact_phase(self, problem: DualProblem, block_iteration: int) -> None:
        """Solve contact triplets from the latest iterate with the selected backend."""
        if self._contact_solver == "apgd":
            self._solve_dense_contact_apgd(problem, block_iteration)
        else:
            self._refresh_dense_unilateral_velocities(problem)
            self._launch_dense_inequality_pgs(problem, block_iteration, _INEQUALITY_FAMILY_CONTACTS)

    def _solve_dense_contact_apgd(self, problem: DualProblem, block_iteration: int) -> None:
        """Solve the associated dense contact subproblem with L and B fixed."""
        if self._contact_apgd is None:
            raise RuntimeError("The APGD contact solver has not been allocated.")
        if self._dense_contact_operator is None or self._dense_contact_problem is not problem:
            self._dense_contact_operator = self._make_dense_contact_operator(problem)
            self._dense_contact_problem = problem
        wp.launch(
            kernel=_set_dvi_contact_active_mask,
            dim=self._size.num_worlds,
            inputs=[
                problem.data.nc,
                problem.data.ccgo,
                problem.data.vio,
                problem.data.P,
                block_iteration,
                self._data.config,
                self._data.status,
                self._data.state.contact_active_mask,
            ],
            device=self.device,
        )
        apgd_status = self._contact_apgd.solve_dense(
            self._dense_contact_operator,
            problem.data.mu,
            self._data.solution.lambdas,
            phase_mask=self._data.state.contact_active_mask,
        )
        wp.launch(
            kernel=_accumulate_dvi_apgd_status,
            dim=self._size.num_worlds,
            inputs=[apgd_status, self._data.state.contact_active_mask, self._data.status],
            device=self.device,
        )

    def _solve_dense_family_pgs(self, problem: DualProblem) -> None:
        """Apply explicit ``L -> C`` family coupling when no bilateral block exists."""
        self._initialize_projected_iterations(problem)
        single_contact_phase = self._contact_solver == "apgd" and not self._has_limit_constraints
        fused_pgs_family = self._contact_solver == "pgs" and (
            self._has_limit_constraints != self._has_contact_constraints
        )
        if fused_pgs_family:
            if self._has_limit_constraints:
                self._solve_dense_limit_phase(problem, _FUSED_SINGLE_FAMILY_BLOCK)
            else:
                self._solve_dense_contact_phase(problem, _FUSED_SINGLE_FAMILY_BLOCK)
        else:
            family_iterations = 1 if single_contact_phase else self._max_alternating_iterations
            for block_iteration in range(family_iterations):
                if self._has_limit_constraints:
                    self._solve_dense_limit_phase(problem, block_iteration)
                if self._has_contact_constraints:
                    self._solve_dense_contact_phase(problem, block_iteration)
        self._set_projected_iteration_status(problem, single_contact_phase=single_contact_phase)

    def _solve_bilateral_block(self, problem: DualProblem, active_dim: wp.array[wp.int32] | None = None):
        """Solve ``(D_bb + E_hat_bb) * lambda_b = -(v_f,b + D_bu * lambda_u)``."""
        operator = self._data.bilateral_operator
        state = self._data.state
        wp.launch(
            kernel=_build_bilateral_rhs,
            dim=(self._size.num_worlds, self._size.max_of_num_bilateral_joint_cts),
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
            dim=(self._size.num_worlds, self._size.max_of_num_bilateral_joint_cts),
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
        """Extract, symmetrically scale, and factor ``D_bb + E_hat_bb``."""
        operator = self._data.bilateral_operator
        operator.info.dim = operator.info.maxdim
        wp.launch(
            kernel=_copy_bilateral_block,
            dim=(
                self._size.num_worlds,
                self._size.max_of_num_bilateral_joint_cts * self._size.max_of_num_bilateral_joint_cts,
            ),
            inputs=[
                problem.data.dim,
                problem.data.mio,
                problem.data.vio,
                problem.data.njc,
                problem.data.D,
                problem.data.E_hat,
                operator.info.mio,
                operator.info.vio,
                operator.mat,
                self._data.state.bilateral_preconditioner,
            ],
            device=self.device,
        )
        self._bilateral_solver.compute(A=operator.mat)

    def _solve_with_bilateral_direct_block(self, problem: DualProblem):
        """Factor the bilateral block once and execute ``L -> B -> C`` sweeps."""
        self._factor_bilateral_block(problem)
        if not self._has_unilateral_constraints:
            self._solve_bilateral_block(problem)
            return

        self._initialize_projected_iterations(problem)
        self._solve_explicit_family_coupling(problem)
        self._set_projected_iteration_status(problem)

    def _solve_explicit_family_coupling(self, problem: DualProblem) -> None:
        """Apply exactly ``L -> B -> C`` per coupling sweep."""
        for block_iteration in range(self._max_alternating_iterations):
            if self._has_limit_constraints:
                self._solve_dense_limit_phase(problem, block_iteration)

            self._set_bilateral_active_dim(problem, block_iteration)
            self._solve_bilateral_block(problem, active_dim=self._data.state.bilateral_active_dim)

            if self._has_contact_constraints:
                # B changed lambda_b, so C must rebuild its residual from the
                # latest unified lambda rather than incrementing a pre-B value.
                self._solve_dense_contact_phase(problem, block_iteration)

        self._solve_post_stabilization_bilateral(problem)

    def _solve_post_stabilization_bilateral(self, problem: DualProblem) -> None:
        """Apply the configured optional terminal bilateral refresh."""
        if not self._has_post_stabilization_bilateral:
            return
        self._set_bilateral_active_dim(problem, -1)
        self._solve_bilateral_block(problem, active_dim=self._data.state.bilateral_active_dim)

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
                    model.joints.num_friction_cts,
                    model.joints.num_effort_cts,
                    model.joints.dofs_offset,
                    model.joints.dynamic_cts_offset,
                    model.joints.kinematic_cts_offset,
                    model.joints.friction_cts_offset,
                    model.joints.effort_cts_offset,
                    model.joints.friction_cts_axis,
                    model.joints.effort_cts_axis,
                    model.joints.dynamic_cts_offset_total_cts,
                    model.joints.kinematic_cts_offset_total_cts,
                    model.joints.friction_cts_offset_total_cts,
                    model.joints.effort_cts_offset_total_cts,
                    data.joints.lambda_dyn_j,
                    data.joints.lambda_kin_j,
                    data.joints.lambda_f_j,
                    data.joints.lambda_tau_j,
                    data.joints.dq_j,
                    data.joints.inv_m_a,
                    data.joints.dq_b_a,
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
            wp.launch(
                kernel=_scale_dvi_tangential_warmstart,
                dim=contacts.model_max_contacts_host,
                inputs=[
                    model.info.total_cts_offset,
                    data.info.contact_cts_group_offset,
                    contacts.model_active_contacts,
                    contacts.wid,
                    contacts.cid,
                    self._data.config,
                    self._data.solution.lambdas,
                ],
                device=self.device,
            )
