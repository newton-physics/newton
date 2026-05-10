# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit MPM solver."""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as wps

import newton

from ...core.types import override
from ..flags import SolverNotifyFlags
from ..solver import SolverBase
from .implicit_mpm_model import ImplicitMPMModel
from .rasterized_collisions import (
    build_rigidity_operator,
    interpolate_collider_normals,
    project_outside_collider,
    rasterize_collider,
)
from .render_grains import sample_render_grains, update_render_grains
from .solve_rheology import CollisionData, MomentumData, RheologyData, YieldParamVec, solve_rheology

__all__ = ["SolverImplicitMPM"]

from .implicit_mpm_solver_kernels import (
    EPSILON,
    INFINITY,
    YIELD_PARAM_LENGTH,
    advect_particles,
    allocate_by_voxels,
    average_elastic_parameters,
    collision_weight_field,
    compliance_form,
    compute_bounds,
    compute_color_offsets,
    compute_eigenvalues,
    compute_unilateral_strain_offset,
    fill_uniform_color_block_indices,
    free_velocity,
    integrate_collider_fraction,
    integrate_collider_fraction_apic,
    integrate_elastic_parameters,
    integrate_fraction,
    integrate_mass,
    integrate_particle_stress,
    integrate_velocity,
    integrate_velocity_apic,
    integrate_yield_parameters,
    inverse_scale_sym_tensor,
    inverse_scale_vector,
    make_cell_color_kernel,
    make_dynamic_color_block_indices_kernel,
    make_inverse_rotate_vectors,
    make_rotate_vectors,
    mark_active_cells,
    mass_form,
    mat11,
    mat13,
    mat31,
    mat66,
    node_color,
    rotate_matrix_columns,
    rotate_matrix_rows,
    scatter_field_dof_values,
    strain_delta_form,
    strain_rhs,
    update_particle_frames,
    update_particle_strains,
)


def _as_2d_array(array, shape, dtype):
    return wp.array(
        data=None,
        ptr=array.ptr,
        capacity=array.capacity,
        device=array.device,
        shape=shape,
        dtype=dtype,
        grad=None if array.grad is None else _as_2d_array(array.grad, shape, dtype),
    )


def _make_grid_basis_space(grid: fem.Geometry, basis_str: str, family: fem.Polynomial | None = None):
    assert len(basis_str) >= 2

    degree = int(basis_str[1])
    discontinuous = degree == 0 or basis_str[-1] == "d"

    if basis_str[0] == "B":
        element_basis = fem.ElementBasis.BSPLINE
    elif basis_str[0] == "Q":
        element_basis = fem.ElementBasis.LAGRANGE
    elif basis_str[0] == "S":
        element_basis = fem.ElementBasis.SERENDIPITY
    elif basis_str[0] == "P" and discontinuous:
        element_basis = fem.ElementBasis.NONCONFORMING_POLYNOMIAL
    else:
        raise ValueError(
            f"Unsupported basis: {basis_str}. Expected format: Q<degree>[d], S<degree>, or P<degree>[d] for tri-polynomial, serendipity, or non-conforming polynomial respectively."
        )

    return fem.make_polynomial_basis_space(
        grid, degree=degree, element_basis=element_basis, family=family, discontinuous=discontinuous
    )


def _make_pic_basis_space(pic: fem.PicQuadrature, basis_str: str):
    try:
        max_points_per_cell = int(basis_str[3:])
    except ValueError:
        max_points_per_cell = -1

    return fem.PointBasisSpace(pic, max_nodes_per_element=max_points_per_cell, use_evaluation_point_index=True)


_RheologySolverName = Literal[
    "auto",
    "gs",
    "gauss-seidel",
    "gs-soa",
    "gauss-seidel-soa",
    "gs-batched",
    "gauss-seidel-batched",
    "jacobi",
    "cg",
    "cr",
    "gmres",
]
_MPMVelocityBasisName = Literal["Q1", "B2", "B3"]
# Python typing cannot express the accepted ``"pic"`` / ``"picN"`` basis family.
_MPMColliderBasisName = Literal["Q1", "S2", "pic", "pic8", "pic27"] | str
_MPMStrainBasisName = Literal["P0", "P1d", "Q1", "Q1d", "pic", "pic8", "pic27"] | str


def _resolve_solver_spec(
    solver: _RheologySolverName | Sequence[_RheologySolverName], velocity_basis: str
) -> tuple[str, ...]:
    solvers = (solver,) if isinstance(solver, str) else tuple(solver)
    if len(solvers) == 0:
        raise ValueError("Solver sequence must contain at least one solver.")

    def resolve_auto(solver_name: _RheologySolverName) -> str:
        if solver_name == "auto":
            return "gs-batched" if velocity_basis in ("B2", "B3") else "gs"
        return solver_name

    return tuple(resolve_auto(solver_name) for solver_name in solvers)


class ImplicitMPMScratchpad:
    """Per-step spaces, fields, and temporaries for the implicit MPM solver."""

    def __init__(self):
        self.grid = None

        self.velocity_test = None
        self.velocity_trial = None
        self.fraction_test = None

        self.sym_strain_test = None
        self.sym_strain_trial = None
        self.divergence_test = None
        self.divergence_trial = None
        self.fraction_field = None
        self.elastic_parameters_field = None

        self.plastic_strain_delta_field = None
        self.elastic_strain_delta_field = None
        self.strain_yield_parameters_field = None
        self.strain_yield_parameters_test = None

        self.strain_matrix = wps.bsr_zeros(0, 0, mat13)
        self.transposed_strain_matrix = wps.bsr_zeros(0, 0, mat31)

        self.compliance_matrix = wps.bsr_zeros(0, 0, mat66)

        self.color_offsets = None
        self.color_indices = None

        self.inv_mass_matrix = None

        self.collider_fraction_test = None

        self.collider_normal_field = None
        self.collider_distance_field = None

        self.collider_velocity = None
        self.collider_friction = None
        self.collider_adhesion = None

        self.collider_matrix = wps.bsr_zeros(0, 0, block_type=float)
        self.transposed_collider_matrix = wps.bsr_zeros(0, 0, block_type=float)

        self.strain_node_particle_volume = None
        self.strain_node_volume = None
        self.strain_node_collider_volume = None

        self.collider_total_volumes = None
        self.collider_node_volume = None

    def rebuild_function_spaces(
        self,
        pic: fem.PicQuadrature,
        velocity_basis_str: str,
        strain_basis_str: str,
        collider_basis_str: str,
        max_cell_count: int,
        temporary_store: fem.TemporaryStore,
    ):
        """Define velocity and strain function spaces over the given geometry."""

        self.domain = pic.domain

        use_pic_collider_basis = collider_basis_str[:3] == "pic"
        use_pic_strain_basis = strain_basis_str[:3] == "pic"

        if self.domain.geometry is not self.grid:
            self.grid = self.domain.geometry

            # Define function spaces: linear (Q1) for velocity and volume fraction,
            # zero or first order for pressure
            self._velocity_basis = _make_grid_basis_space(self.grid, velocity_basis_str)

            if not use_pic_strain_basis:
                self._strain_basis = _make_grid_basis_space(self.grid, strain_basis_str)

            if not use_pic_collider_basis:
                self._collision_basis = _make_grid_basis_space(
                    self.grid, collider_basis_str, family=fem.Polynomial.EQUISPACED_CLOSED
                )

        # Point-based basis space needs to be rebuilt even when the geo does not change
        if use_pic_strain_basis:
            self._strain_basis = _make_pic_basis_space(pic, strain_basis_str)
        if use_pic_collider_basis:
            self._collision_basis = _make_pic_basis_space(pic, collider_basis_str)

        self._create_velocity_function_space(temporary_store, max_cell_count)
        self._create_collider_function_space(temporary_store, max_cell_count)
        self._create_strain_function_space(temporary_store, max_cell_count)

    def _create_velocity_function_space(self, temporary_store: fem.TemporaryStore, max_cell_count: int = -1):
        """Create velocity and fraction spaces and their partition/restriction."""
        domain = self.domain

        velocity_space = fem.make_collocated_function_space(self._velocity_basis, dtype=wp.vec3)

        # overly conservative
        max_vel_node_count = (
            velocity_space.topology.MAX_NODES_PER_ELEMENT * max_cell_count if max_cell_count >= 0 else -1
        )

        vel_space_partition = fem.make_space_partition(
            space_topology=velocity_space.topology,
            geometry_partition=domain.geometry_partition,
            with_halo=False,
            max_node_count=max_vel_node_count,
            temporary_store=temporary_store,
        )
        vel_space_restriction = fem.make_space_restriction(
            space_partition=vel_space_partition, domain=domain, temporary_store=temporary_store
        )

        self._velocity_space = velocity_space
        self._vel_space_restriction = vel_space_restriction

    def _create_collider_function_space(self, temporary_store: fem.TemporaryStore, max_cell_count: int = -1):
        """Create collider function space and its partition/restriction."""

        if self._velocity_basis == self._collision_basis:
            self._collision_space = self._velocity_space
            self._collision_space_restriction = self._vel_space_restriction
            return

        domain = self.domain

        collision_space = fem.make_collocated_function_space(self._collision_basis, dtype=wp.vec3)

        if isinstance(collision_space.basis, fem.PointBasisSpace):
            max_collision_node_count = collision_space.node_count()
        else:
            # overly conservative
            max_collision_node_count = (
                collision_space.topology.MAX_NODES_PER_ELEMENT * domain.element_count() if max_cell_count >= 0 else -1
            )

        collision_space_partition = fem.make_space_partition(
            space_topology=collision_space.topology,
            geometry_partition=domain.geometry_partition,
            with_halo=False,
            max_node_count=max_collision_node_count,
            temporary_store=temporary_store,
        )
        collision_space_restriction = fem.make_space_restriction(
            space_partition=collision_space_partition, domain=domain, temporary_store=temporary_store
        )

        self._collision_space = collision_space
        self._collision_space_restriction = collision_space_restriction

    def _create_strain_function_space(self, temporary_store: fem.TemporaryStore, max_cell_count: int = -1):
        """Create symmetric strain space (P0 or Q1) and its partition/restriction."""
        domain = self.domain

        sym_strain_space = fem.make_collocated_function_space(
            self._strain_basis,
            dof_mapper=fem.SymmetricTensorMapper(dtype=wp.mat33, mapping=fem.SymmetricTensorMapper.Mapping.DB16),
        )

        max_strain_node_count = (
            sym_strain_space.topology.MAX_NODES_PER_ELEMENT * max_cell_count if max_cell_count >= 0 else -1
        )

        strain_space_partition = fem.make_space_partition(
            space_topology=sym_strain_space.topology,
            geometry_partition=domain.geometry_partition,
            with_halo=False,
            max_node_count=max_strain_node_count,
            temporary_store=temporary_store,
        )

        strain_space_restriction = fem.make_space_restriction(
            space_partition=strain_space_partition, domain=domain, temporary_store=temporary_store
        )

        self._sym_strain_space = sym_strain_space
        self._strain_space_restriction = strain_space_restriction

    def require_velocity_space_fields(self, has_compliant_particles: bool):
        velocity_basis = self._velocity_basis
        velocity_space = self._velocity_space
        vel_space_restriction = self._vel_space_restriction
        domain = vel_space_restriction.domain
        vel_space_partition = vel_space_restriction.space_partition

        if (
            self.velocity_test is not None
            and self.velocity_test.space_restriction.space_partition == vel_space_partition
        ):
            return

        fraction_space = fem.make_collocated_function_space(velocity_basis, dtype=float)

        # test, trial and discrete fields
        if self.velocity_test is None:
            self.velocity_test = fem.make_test(velocity_space, domain=domain, space_restriction=vel_space_restriction)
            self.fraction_test = fem.make_test(fraction_space, space_restriction=vel_space_restriction)

            self.velocity_trial = fem.make_trial(velocity_space, domain=domain, space_partition=vel_space_partition)
            self.fraction_trial = fem.make_trial(fraction_space, domain=domain, space_partition=vel_space_partition)

            self.fraction_field = fem.make_discrete_field(fraction_space, space_partition=vel_space_partition)

        else:
            self.velocity_test.rebind(velocity_space, vel_space_restriction)
            self.fraction_test.rebind(fraction_space, vel_space_restriction)

            self.velocity_trial.rebind(velocity_space, vel_space_partition, domain)
            self.fraction_trial.rebind(fraction_space, vel_space_partition, domain)
            self.fraction_field.rebind(fraction_space, vel_space_partition)

        if has_compliant_particles:
            elastic_parameters_space = fem.make_collocated_function_space(velocity_basis, dtype=wp.vec3)
            if self.elastic_parameters_field is None:
                self.elastic_parameters_field = elastic_parameters_space.make_field(space_partition=vel_space_partition)
            else:
                self.elastic_parameters_field.rebind(elastic_parameters_space, vel_space_partition)

        self.velocity_field = velocity_space.make_field(space_partition=vel_space_partition)

    def require_collision_space_fields(self):
        collision_basis = self._collision_basis
        collision_space = self._collision_space
        collision_space_restriction = self._collision_space_restriction
        domain = collision_space_restriction.domain
        collision_space_partition = collision_space_restriction.space_partition

        if (
            self.collider_fraction_test is not None
            and self.collider_fraction_test.space_restriction.space_partition == collision_space_partition
        ):
            return
        collider_fraction_space = fem.make_collocated_function_space(collision_basis, dtype=float)

        # test, trial and discrete fields
        if self.collider_fraction_test is None:
            self.collider_fraction_test = fem.make_test(
                collider_fraction_space, space_restriction=collision_space_restriction
            )
            self.collider_distance_field = collider_fraction_space.make_field(space_partition=collision_space_partition)

            self.collider_velocity_field = collision_space.make_field(space_partition=collision_space_partition)
            self.collider_normal_field = collision_space.make_field(space_partition=collision_space_partition)

            self.background_impulse_field = fem.UniformField(domain, wp.vec3(0.0))
        else:
            self.collider_fraction_test.rebind(collider_fraction_space, collision_space_restriction)
            self.collider_distance_field.rebind(collider_fraction_space, collision_space_partition)

            self.collider_velocity_field.rebind(collision_space, collision_space_partition)
            self.collider_normal_field.rebind(collision_space, collision_space_partition)

            self.background_impulse_field.domain = domain

        self.impulse_field = collision_space.make_field(space_partition=collision_space_partition)
        self.collider_position_field = collision_space.make_field(space_partition=collision_space_partition)
        self.collider_ids = wp.empty(collision_space_partition.node_count(), dtype=int)

    def require_strain_space_fields(self):
        """Ensure strain-space fields exist and match current spaces."""
        strain_basis = self._strain_basis
        sym_strain_space = self._sym_strain_space
        strain_space_restriction = self._strain_space_restriction
        domain = strain_space_restriction.domain
        strain_space_partition = strain_space_restriction.space_partition

        if (
            self.sym_strain_test is not None
            and self.sym_strain_test.space_restriction.space_partition == strain_space_partition
        ):
            return

        divergence_space = fem.make_collocated_function_space(strain_basis, dtype=float)
        strain_yield_parameters_space = fem.make_collocated_function_space(strain_basis, dtype=YieldParamVec)

        if self.sym_strain_test is None:
            self.sym_strain_test = fem.make_test(sym_strain_space, space_restriction=strain_space_restriction)
            self.divergence_test = fem.make_test(divergence_space, space_restriction=strain_space_restriction)
            self.strain_yield_parameters_test = fem.make_test(
                strain_yield_parameters_space, space_restriction=strain_space_restriction
            )
            self.sym_strain_trial = fem.make_trial(
                sym_strain_space, domain=domain, space_partition=strain_space_partition
            )
            self.divergence_trial = fem.make_trial(
                divergence_space, domain=domain, space_partition=strain_space_partition
            )

            self.elastic_strain_delta_field = sym_strain_space.make_field(space_partition=strain_space_partition)
            self.plastic_strain_delta_field = sym_strain_space.make_field(space_partition=strain_space_partition)
            self.strain_yield_parameters_field = strain_yield_parameters_space.make_field(
                space_partition=strain_space_partition
            )

            self.background_stress_field = fem.UniformField(domain, wp.mat33(0.0))
        else:
            self.sym_strain_test.rebind(sym_strain_space, strain_space_restriction)
            self.divergence_test.rebind(divergence_space, strain_space_restriction)
            self.strain_yield_parameters_test.rebind(strain_yield_parameters_space, strain_space_restriction)

            self.sym_strain_trial.rebind(sym_strain_space, strain_space_partition, domain)
            self.divergence_trial.rebind(divergence_space, strain_space_partition, domain)

            self.elastic_strain_delta_field.rebind(sym_strain_space, strain_space_partition)
            self.plastic_strain_delta_field.rebind(sym_strain_space, strain_space_partition)
            self.strain_yield_parameters_field.rebind(strain_yield_parameters_space, strain_space_partition)

            self.background_stress_field.domain = domain

        self.stress_field = sym_strain_space.make_field(space_partition=strain_space_partition)

    @property
    def collider_node_count(self) -> int:
        return self._collision_space_restriction.space_partition.node_count()

    @property
    def velocity_node_count(self) -> int:
        return self._vel_space_restriction.space_partition.node_count()

    @property
    def velocity_nodes_per_element(self) -> int:
        return self._vel_space_restriction.space_partition.space_topology.MAX_NODES_PER_ELEMENT

    @property
    def strain_node_count(self) -> int:
        return self._strain_space_restriction.space_partition.node_count()

    @property
    def strain_nodes_per_element(self) -> int:
        return self._strain_space_restriction.space_partition.space_topology.MAX_NODES_PER_ELEMENT

    def allocate_temporaries(
        self,
        collider_count: int,
        has_compliant_bodies: bool,
        has_critical_fraction: bool,
        max_colors: int,
        temporary_store: fem.TemporaryStore,
    ):
        """Allocate transient arrays sized to current grid and options."""
        vel_node_count = self.velocity_node_count
        collider_node_count = self.collider_node_count
        strain_node_count = self.strain_node_count

        self.inv_mass_matrix = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)

        self.collider_velocity = fem.borrow_temporary(temporary_store, shape=(collider_node_count,), dtype=wp.vec3)
        self.collider_friction = fem.borrow_temporary(temporary_store, shape=(collider_node_count,), dtype=float)
        self.collider_adhesion = fem.borrow_temporary(temporary_store, shape=(collider_node_count,), dtype=float)
        self.collider_node_volume = fem.borrow_temporary(temporary_store, shape=collider_node_count, dtype=float)

        self.strain_node_particle_volume = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)
        self.unilateral_strain_offset = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)

        wps.bsr_set_zero(self.strain_matrix, rows_of_blocks=strain_node_count, cols_of_blocks=vel_node_count)
        wps.bsr_set_zero(self.compliance_matrix, rows_of_blocks=strain_node_count, cols_of_blocks=strain_node_count)

        if has_critical_fraction:
            self.strain_node_volume = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)
            self.strain_node_collider_volume = fem.borrow_temporary(
                temporary_store, shape=strain_node_count, dtype=float
            )

        if has_compliant_bodies:
            self.collider_total_volumes = fem.borrow_temporary(temporary_store, shape=collider_count, dtype=float)

        if max_colors > 0:
            self.color_indices = fem.borrow_temporary(temporary_store, shape=(2, strain_node_count), dtype=int)
            self.color_offsets = fem.borrow_temporary(temporary_store, shape=max_colors + 1, dtype=int)

    def release_temporaries(self):
        """Release previously allocated temporaries to the store."""
        self.inv_mass_matrix.release()
        self.collider_velocity.release()
        self.collider_friction.release()
        self.collider_adhesion.release()
        self.collider_node_volume.release()
        self.strain_node_particle_volume.release()
        self.unilateral_strain_offset.release()

        if self.strain_node_volume is not None:
            self.strain_node_volume.release()
            self.strain_node_collider_volume.release()

        if self.collider_total_volumes is not None:
            self.collider_total_volumes.release()

        if self.color_indices is not None:
            self.color_indices.release()
            self.color_offsets.release()


class LastStepData:
    """Persistent solver state preserved across time steps.

    Separate from ImplicitMPMScratchpad which is rebuilt when the grid changes.
    Stores warmstart fields for the iterative solver and previous body transforms
    for finite-difference velocity computation.
    """

    def __init__(self):
        self.ws_impulse_field = None  # Warmstart for collision impulses
        self.ws_stress_field = None  # Warmstart for stress field
        self.body_q_prev = None  # Previous body transforms for finite-difference velocities

    def _ws_stress_space(self, scratch: ImplicitMPMScratchpad, smoothed: bool):
        sym_strain_space = scratch.sym_strain_test.space
        if isinstance(sym_strain_space.basis, fem.PointBasisSpace) or not smoothed:
            return sym_strain_space
        else:
            return fem.make_polynomial_space(scratch.grid, degree=1, dof_mapper=sym_strain_space.dof_mapper)

    def require_strain_space_fields(self, scratch: ImplicitMPMScratchpad, smoothed: bool):
        """Ensure strain-space fields exist and match current spaces."""
        if self.ws_stress_field is None:
            self.ws_stress_field = self._ws_stress_space(scratch, smoothed).make_field()

    def rebind_strain_space_fields(self, scratch: ImplicitMPMScratchpad, smoothed: bool):
        if self.ws_stress_field.geometry != scratch.sym_strain_test.space.geometry:
            ws_stress_space = self._ws_stress_space(scratch, smoothed)
            self.ws_stress_field.rebind(
                space=ws_stress_space,
                space_partition=fem.make_space_partition(
                    space_topology=ws_stress_space.topology, geometry_partition=None
                ),
            )

    def require_collision_space_fields(self, scratch: ImplicitMPMScratchpad):
        """Ensure collision-space fields exist and match current spaces."""
        if self.ws_impulse_field is None:
            self.ws_impulse_field = scratch.impulse_field.space.make_field()

    def rebind_collision_space_fields(self, scratch: ImplicitMPMScratchpad):
        if self.ws_impulse_field.geometry != scratch.impulse_field.space.geometry:
            self.ws_impulse_field.rebind(
                space=scratch.impulse_field.space,
                space_partition=fem.make_space_partition(
                    space_topology=scratch.impulse_field.space.topology,
                    geometry_partition=None,
                ),
            )

    def require_collider_previous_position(self, collider_body_q: wp.array | None):
        if collider_body_q is None:
            self.body_q_prev = None
        elif self.body_q_prev is None or self.body_q_prev.shape != collider_body_q.shape:
            self.body_q_prev = wp.clone(collider_body_q)

    def save_collider_current_position(self, collider_body_q: wp.array | None):
        self.require_collider_previous_position(collider_body_q)
        if collider_body_q is not None:
            self.body_q_prev.assign(collider_body_q)


class SolverImplicitMPM(SolverBase):
    """Implicit MPM solver for granular and elasto-plastic materials.

    Implements an implicit Material Point Method (MPM) algorithm roughly
    following [1], extended with a GPU-friendly rheology solver supporting
    pressure-dependent yield (Drucker-Prager), viscosity, dilatancy, and
    isotropic hardening/softening.

    This variant is particularly well-suited for very stiff materials and
    the fully inelastic limit. It is less versatile than traditional explicit
    MPM but offers unconditional stability with respect to the time step.

    Call :meth:`register_custom_attributes` on your :class:`~newton.ModelBuilder`
    before building the model to enable the MPM-specific per-particle material
    parameters and state variables (e.g. ``mpm:young_modulus``,
    ``mpm:friction``, ``mpm:particle_elastic_strain``).

    [1] https://doi.org/10.1145/2897824.2925877

    Args:
        model: The model to simulate.
        config: Solver configuration. See :class:`SolverImplicitMPM.Config`.
        temporary_store: Optional Warp FEM temporary store for reusing scratch
            allocations across steps.
        verbose: Enable verbose solver output. Defaults to ``wp.config.verbose``.
        enable_timers: Enable per-section wall-clock timings.
    """

    @dataclass
    class Config:
        """Configuration for :class:`SolverImplicitMPM`.

        Per-particle properties can be configured using custom attributes on the Model.
        See :meth:`SolverImplicitMPM.register_custom_attributes` for details.
        """

        # numerics
        max_iterations: int = 250
        """Maximum number of iterations for the rheology solver."""
        tolerance: float = 1.0e-4
        """Tolerance for the rheology solver."""
        solver: _RheologySolverName | Sequence[_RheologySolverName] = "auto"
        """Solver to use for the rheology solver. ``"auto"`` selects ``"gs"``
        for Q1 velocity basis and ``"gs-batched"`` for higher-order bases
        (B2, B3).  Accepted values: ``"auto"``, ``"gs"`` (or
        ``"gauss-seidel"``), ``"gs-soa"`` (or ``"gauss-seidel-soa"``),
        ``"gs-batched"`` (or ``"gauss-seidel-batched"``), ``"jacobi"``,
        ``"cg"``, ``"cr"``, ``"gmres"``.  Pass an ordered sequence to
        warmstart solvers left-to-right, e.g. ``("cr", "gs")`` or
        ``("cg", "jacobi", "gs")``."""
        warmstart_mode: Literal["none", "auto", "particles", "grid", "smoothed"] = "auto"
        """Warmstart mode to use for the rheology solver."""
        collider_velocity_mode: Literal["forward", "backward", "instantaneous", "finite_difference"] = "forward"
        """Collider velocity computation mode. ``'forward'`` uses the current velocity,
        ``'backward'`` uses the previous timestep position.

        .. deprecated:: 1.1
            Aliases ``'instantaneous'`` (= ``'forward'``) and ``'finite_difference'``
            (= ``'backward'``) are deprecated and will be removed in a future release.
        """

        # grid
        voxel_size: float = 0.1
        """Size of the grid voxels."""
        grid_type: Literal["sparse", "dense", "fixed"] = "sparse"
        """Type of grid to use."""
        grid_padding: int = 0
        """Number of empty cells to add around particles when allocating the grid."""
        max_active_cell_count: int = -1
        """Maximum number of active cells to use for active subsets of dense grids. -1 means unlimited."""
        transfer_scheme: Literal["apic", "pic"] = "apic"
        """Transfer scheme to use for particle-grid transfers."""
        integration_scheme: Literal["pic", "gimp"] = "pic"
        """Integration scheme controlling shape-function support."""

        # material / background
        critical_fraction: float = 0.0
        """Fraction for particles under which the yield surface collapses."""
        air_drag: float = 1.0
        """Numerical drag for the background air."""

        # experimental
        collider_normal_from_sdf_gradient: bool = False
        """Compute collider normals from sdf gradient rather than closest point"""
        collider_basis: _MPMColliderBasisName = "S2"
        """Collider basis function. Defaults to ``"S2"``; pass ``"Q1"``
        to restore the previous trilinear collider basis. Common values are
        ``"Q1"`` (trilinear), ``"S2"`` (quadratic serendipity), or
        ``"pic"``, ``"pic8"``, ``"pic27"``
        (particle-based with optional max points per cell). Any ``"picN"``
        form with integer ``N`` is accepted."""
        strain_basis: _MPMStrainBasisName = "P0"
        """Strain basis function. Common values are ``"P0"``, ``"P1d"``,
        ``"Q1"``, ``"Q1d"``, or particle-based ``"pic"``, ``"pic8"``,
        ``"pic27"``. Any ``"picN"`` form with integer ``N`` is accepted."""
        velocity_basis: _MPMVelocityBasisName = "Q1"
        """Velocity basis function. Common values are ``"Q1"``, ``"B2"``,
        or ``"B3"``."""

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        """Register MPM-specific custom attributes in the 'mpm' namespace.

        This method registers per-particle material parameters and state variables
        for the implicit MPM solver.

        Attributes registered on Model (per-particle):
            - ``mpm:young_modulus``: Young's modulus in Pa
            - ``mpm:poisson_ratio``: Poisson's ratio for elasticity
            - ``mpm:damping``: Elastic damping relaxation time in seconds
            - ``mpm:friction``: Friction coefficient
            - ``mpm:yield_pressure``: Yield pressure in Pa
            - ``mpm:tensile_yield_ratio``: Tensile yield ratio
            - ``mpm:yield_stress``: Deviatoric yield stress in Pa
            - ``mpm:hardening``: Hardening factor for plasticity
            - ``mpm:hardening_rate``: Hardening rate for plasticity
            - ``mpm:softening_rate``: Softening rate for plasticity
            - ``mpm:dilatancy``: Dilatancy factor for plasticity
            - ``mpm:viscosity``: Viscosity for plasticity [Pa·s]

        Attributes registered on State (per-particle):
            - ``mpm:particle_qd_grad``: Velocity gradient for APIC transfer
            - ``mpm:particle_elastic_strain``: Elastic deformation gradient
            - ``mpm:particle_Jp``: Determinant of plastic deformation gradient
            - ``mpm:particle_stress``: Cauchy stress tensor [Pa]
            - ``mpm:particle_transform``: Overall deformation gradient for rendering
        """
        # Per-particle material parameters
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="young_modulus",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0e15,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="poisson_ratio",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.3,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="damping",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="hardening",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="friction",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.5,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="yield_pressure",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0e15,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="tensile_yield_ratio",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="yield_stress",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="hardening_rate",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="softening_rate",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=1.0,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="dilatancy",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="viscosity",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mpm",
            )
        )

        # Per-particle state attributes (attached to State objects)
        identity = wp.mat33(np.eye(3))
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="particle_qd_grad",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.mat33,
                default=wp.mat33(0.0),
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="particle_elastic_strain",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.mat33,
                default=identity,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="particle_Jp",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.float32,
                default=1.0,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="particle_stress",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.mat33,
                default=wp.mat33(0.0),
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="particle_transform",
                frequency=newton.Model.AttributeFrequency.PARTICLE,
                assignment=newton.Model.AttributeAssignment.STATE,
                dtype=wp.mat33,
                default=identity,
                namespace="mpm",
            )
        )

    def __init__(
        self,
        model: newton.Model,
        config: Config,
        temporary_store: fem.TemporaryStore | None = None,
        verbose: bool | None = None,
        enable_timers: bool = False,
    ):
        super().__init__(model)

        self._mpm_model = ImplicitMPMModel(model, config)

        self.max_iterations = config.max_iterations
        self.tolerance = float(config.tolerance)

        self.temporary_store = temporary_store
        self.verbose = verbose if verbose is not None else wp.config.verbose
        self.enable_timers = enable_timers

        self.velocity_basis = "Q1"
        self.strain_basis = config.strain_basis
        self.velocity_basis = config.velocity_basis

        self.grid_padding = config.grid_padding
        self.grid_type = config.grid_type
        self.solver = _resolve_solver_spec(config.solver, self.velocity_basis)
        self.coloring = any("gauss-seidel" in solver or "gs" in solver for solver in self.solver)
        self.apic = config.transfer_scheme == "apic"
        self.gimp = config.integration_scheme == "gimp"
        self.max_active_cell_count = config.max_active_cell_count

        self.collider_normal_from_sdf_gradient = config.collider_normal_from_sdf_gradient
        self.collider_basis = config.collider_basis

        # Map deprecated aliases to canonical values
        if config.collider_velocity_mode == "finite_difference":
            warnings.warn(
                "collider_velocity_mode='finite_difference' is deprecated, use 'backward' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.collider_velocity_mode = "backward"
        elif config.collider_velocity_mode == "instantaneous":
            warnings.warn(
                "collider_velocity_mode='instantaneous' is deprecated, use 'forward' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.collider_velocity_mode = "forward"
        else:
            if config.collider_velocity_mode not in ("forward", "backward"):
                raise ValueError(f"Invalid collider velocity mode: {config.collider_velocity_mode}")
            self.collider_velocity_mode = config.collider_velocity_mode

        if config.warmstart_mode == "none":
            self._stress_warmstart = ""
        elif config.warmstart_mode == "auto":
            if self.strain_basis in ("P1d", "Q1d"):
                self._stress_warmstart = "particles"
            else:
                self._stress_warmstart = "grid"
        else:
            if config.warmstart_mode not in ("particles", "grid", "smoothed"):
                raise ValueError(f"Invalid warmstart mode: {config.warmstart_mode}")
            self._stress_warmstart = config.warmstart_mode

        self._use_cuda_graph = self.model.device.is_cuda and wp.is_conditional_graph_supported()

        self._timers_use_nvtx = False

        # Pre-allocate scratchpad and last step data so that step() can be graph-captured
        self._scratchpad = None
        self._last_step_data = LastStepData()
        with wp.ScopedDevice(model.device):
            pic = self._particles_to_cells(model.particle_q)
            self._rebuild_scratchpad(pic)
            self._require_velocity_space_fields(self._scratchpad, self._mpm_model.has_compliant_particles)
            self._require_collision_space_fields(self._scratchpad, self._last_step_data)
            self._require_strain_space_fields(self._scratchpad, self._last_step_data)

        self._velocity_nodes_per_strain_sample = (
            self._scratchpad.velocity_nodes_per_element
            if self.strain_basis != "Q1" and self.velocity_basis == "Q1"
            else -1
        )

    def setup_collider(
        self,
        collider_meshes: list[wp.Mesh] | None = None,
        collider_body_ids: list[int] | None = None,
        collider_margins: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_adhesion: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        model: newton.Model | None = None,
        body_com: wp.array | None = None,
        body_mass: wp.array | None = None,
        body_inv_inertia: wp.array | None = None,
        body_q: wp.array | None = None,
    ) -> None:
        """Configure collider geometry and material properties.

        By default, collisions are set up against all shapes in the model with
        ``newton.ShapeFlags.COLLIDE_PARTICLES``. Use this method to customize
        collider sources, materials, or to read colliders from a different model.

        Args:
            collider_meshes: Warp triangular meshes used as colliders.
            collider_body_ids: For dynamic colliders, per-mesh body ids.
            collider_margins: Per-mesh signed distance offsets (m).
            collider_friction: Per-mesh Coulomb friction coefficients.
            collider_adhesion: Per-mesh adhesion (Pa).
            collider_projection_threshold: Per-mesh projection threshold (m).
            model: The model to read collider properties from. Default to solver's model.
            body_com: For dynamic colliders, per-body center of mass.
            body_mass: For dynamic colliders, per-body mass. Pass zeros for kinematic bodies.
            body_inv_inertia: For dynamic colliders, per-body inverse inertia.
            body_q: For dynamic colliders, per-body initial transform.
        """
        self._mpm_model.setup_collider(
            collider_meshes=collider_meshes,
            collider_body_ids=collider_body_ids,
            collider_thicknesses=collider_margins,
            collider_friction=collider_friction,
            collider_adhesion=collider_adhesion,
            collider_projection_threshold=collider_projection_threshold,
            model=model,
            body_com=body_com,
            body_mass=body_mass,
            body_inv_inertia=body_inv_inertia,
            body_q=body_q,
        )

        self._last_step_data.save_collider_current_position(self._mpm_model.collider_body_q)

    @property
    def voxel_size(self) -> float:
        """Grid voxel size used by the solver."""
        return self._mpm_model.voxel_size

    @override
    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ) -> None:
        """Advance the simulation by one time step.

        Transfers particle data to the grid, solves the implicit rheology
        system, and transfers the result back to update particle positions,
        velocities, and stress.

        Args:
            state_in: Input state at the start of the step.
            state_out: Output state written with updated particle data.
                May be the same object as ``state_in`` for in-place stepping.
            control: Control input (unused; material parameters come from the model).
            contacts: Contact information (unused; collisions are handled internally).
            dt: Time step duration [s].
        """
        model = self.model

        with wp.ScopedDevice(model.device):
            pic = self._particles_to_cells(state_in.particle_q)
            scratch = self._rebuild_scratchpad(pic)
            self._step_impl(state_in, state_out, dt, pic, scratch)
            scratch.release_temporaries()

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self._mpm_model.notify_particle_material_changed()

    def collect_collider_impulses(self, state: newton.State) -> tuple[wp.array, wp.array, wp.array]:
        """Collect current collider impulses and their application positions.

        Returns a tuple of 3 arrays:
            - Impulse values in world units.
            - Collider positions in world units.
            - Collider id, that can be mapped back to the model's body ids using the ``collider_body_index`` property.
        """

        # Not stepped yet, read from preallocated scratchpad
        if not hasattr(state, "impulse_field"):
            state = self._scratchpad

        cell_volume = self._mpm_model.voxel_size**3
        return (
            -cell_volume * state.impulse_field.dof_values,
            state.collider_position_field.dof_values,
            state.collider_ids,
        )

    @property
    def collider_body_index(self) -> wp.array:
        """Array mapping collider indices to body indices.

        Returns:
            Per-collider body index array. Value is -1 for colliders that are not bodies.
        """
        return self._mpm_model.collider.collider_body_index

    def project_outside(self, state_in: newton.State, state_out: newton.State, dt: float, gap: float | None = None):
        """Project particles outside of colliders, and adjust their velocity and velocity gradients

        Args:
            state_in: The input state.
            state_out: The output state. Only particle_q, particle_qd, and particle_qd_grad are written.
            dt: The time step, for extrapolating the collider end-of-step positions from its current position and velocity.
            gap: Maximum distance for closest-point queries. If None, the default is the voxel size times sqrt(3).
        """

        if gap is not None:
            # Update max query dist if provided
            prev_gap, self._mpm_model.collider.query_max_dist = self._mpm_model.collider.query_max_dist, gap

        self._last_step_data.require_collider_previous_position(state_in.body_q)
        wp.launch(
            project_outside_collider,
            dim=state_in.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.mpm.particle_qd_grad,
                self.model.particle_flags,
                self.model.particle_mass,
                self._mpm_model.collider,
                state_in.body_q,
                state_in.body_qd if self.collider_velocity_mode == "forward" else None,
                self._last_step_data.body_q_prev if self.collider_velocity_mode == "backward" else None,
                dt,
            ],
            outputs=[
                state_out.particle_q,
                state_out.particle_qd,
                state_out.mpm.particle_qd_grad,
            ],
            device=state_in.particle_q.device,
        )

        if gap is not None:
            # Restore previous max query dist
            self._mpm_model.collider.query_max_dist = prev_gap

    def update_particle_frames(
        self,
        state_prev: newton.State,
        state: newton.State,
        dt: float,
        min_stretch: float = 0.25,
        max_stretch: float = 2.0,
    ) -> None:
        """Update per-particle deformation frames for rendering and projection.

        Integrates the particle deformation gradient using the velocity gradient
        and clamps its principal stretches to the provided bounds for
        robustness.
        """

        wp.launch(
            update_particle_frames,
            dim=state.particle_count,
            inputs=[
                dt,
                min_stretch,
                max_stretch,
                state.mpm.particle_qd_grad,
                state_prev.mpm.particle_transform,
                state.mpm.particle_transform,
            ],
            device=state.mpm.particle_qd_grad.device,
        )

    def sample_render_grains(self, state: newton.State, grains_per_particle: int) -> wp.array:
        """Generate per-particle point samples used for high-resolution rendering.

        Args:
            state: Current Newton state providing particle positions.
            grains_per_particle: Number of grains to sample per particle.

        Returns:
            A ``wp.array`` with shape ``(num_particles, grains_per_particle)`` of
            type ``wp.vec3`` containing grain positions.
        """

        return sample_render_grains(state, self._mpm_model.particle_radius, grains_per_particle)

    def update_render_grains(
        self,
        state_prev: newton.State,
        state: newton.State,
        grains: wp.array,
        dt: float,
    ) -> None:
        """Advect grain samples with the grid velocity and keep them inside the deformed particle.

        Args:
            state_prev: Previous state (t_n).
            state: Current state (t_{n+1}).
            grains: 2D array of grain positions per particle to be updated in place. See ``sample_render_grains``.
            dt: Time step duration.
        """

        return update_render_grains(state_prev, state, grains, self._mpm_model.particle_radius, dt)

    def _allocate_grid(
        self,
        positions: wp.array,
        particle_flags: wp.array,
        voxel_size: float,
        temporary_store: fem.TemporaryStore,
        padding_voxels: int = 0,
    ):
        """Create a grid (sparse or dense) covering all particle positions.

        Uses a sparse ``Nanogrid`` when requested; otherwise computes an axis
        aligned bounding box and instantiates a dense ``Grid3D`` with optional
        padding in voxel units.

        Args:
            positions: Particle positions to bound.
            particle_flags: Per-particle flags; inactive particles are excluded from bounds.
            voxel_size: Grid voxel edge length.
            temporary_store: Temporary storage for intermediate buffers.
            padding_voxels: Additional empty voxels to add around the bounds.

        Returns:
            A geometry partition suitable for FEM field assembly.
        """
        with self._timer("Allocate grid"):
            if self.grid_type == "sparse":
                volume = allocate_by_voxels(positions, voxel_size, padding_voxels=padding_voxels)
                grid = fem.Nanogrid(volume, temporary_store=temporary_store)
            else:
                # Compute bounds and transfer to host
                device = positions.device
                if device.is_cuda:
                    min_dev = fem.borrow_temporary(temporary_store, shape=1, dtype=wp.vec3, device=device)
                    max_dev = fem.borrow_temporary(temporary_store, shape=1, dtype=wp.vec3, device=device)

                    min_dev.fill_(wp.vec3(INFINITY))
                    max_dev.fill_(wp.vec3(-INFINITY))

                    tile_size = 256
                    wp.launch(
                        compute_bounds,
                        dim=((positions.shape[0] + tile_size - 1) // tile_size, tile_size),
                        block_dim=tile_size,
                        inputs=[positions, particle_flags, min_dev, max_dev],
                        device=device,
                    )

                    min_host = fem.borrow_temporary(
                        temporary_store, shape=1, dtype=wp.vec3, device="cpu", pinned=device.is_cuda
                    )
                    max_host = fem.borrow_temporary(
                        temporary_store, shape=1, dtype=wp.vec3, device="cpu", pinned=device.is_cuda
                    )
                    wp.copy(src=min_dev, dest=min_host)
                    wp.copy(src=max_dev, dest=max_host)
                    wp.synchronize_stream()
                    bbox_min, bbox_max = min_host.numpy(), max_host.numpy()
                else:
                    bbox_min, bbox_max = np.min(positions.numpy(), axis=0), np.max(positions.numpy(), axis=0)

                # Round to nearest voxel
                grid_min = np.floor(bbox_min / voxel_size) - padding_voxels
                grid_max = np.ceil(bbox_max / voxel_size) + padding_voxels

                grid = fem.Grid3D(
                    bounds_lo=wp.vec3(grid_min * voxel_size),
                    bounds_hi=wp.vec3(grid_max * voxel_size),
                    res=wp.vec3i((grid_max - grid_min).astype(int)),
                )

        return grid

    def _create_geometry_partition(
        self, grid: fem.Geometry, positions: wp.array, particle_flags: wp.array, max_cell_count: int
    ):
        """Create a geometry partition for the given positions."""

        active_cells = fem.borrow_temporary(self.temporary_store, shape=grid.cell_count(), dtype=int)
        active_cells.zero_()
        fem.interpolate(
            mark_active_cells,
            dim=positions.shape[0],
            at=fem.Cells(grid),
            values={
                "positions": positions,
                "particle_flags": particle_flags,
                "active_cells": active_cells,
            },
            temporary_store=self.temporary_store,
        )

        partition = fem.ExplicitGeometryPartition(
            grid,
            cell_mask=active_cells,
            max_cell_count=max_cell_count,
            max_side_count=0,
            temporary_store=self.temporary_store,
        )
        active_cells.release()

        return partition

    def _rebuild_scratchpad(self, pic: fem.PicQuadrature):
        """(Re)create function spaces and allocate per-step temporaries.

        Allocates the grid based on current particle positions, rebuilds
        velocity and strain spaces as needed, configures collision data, and
        optionally computes a Gauss-Seidel coloring for the strain nodes.
        """

        if self._scratchpad is None:
            self._scratchpad = ImplicitMPMScratchpad()

        scratch = self._scratchpad

        with self._timer("Scratchpad"):
            scratch.rebuild_function_spaces(
                pic,
                strain_basis_str=self.strain_basis,
                velocity_basis_str=self.velocity_basis,
                collider_basis_str=self.collider_basis,
                max_cell_count=self.max_active_cell_count,
                temporary_store=self.temporary_store,
            )

            scratch.allocate_temporaries(
                collider_count=self._mpm_model.collider.collider_mesh.shape[0],
                has_compliant_bodies=self._mpm_model.has_compliant_colliders,
                has_critical_fraction=self._mpm_model.critical_fraction > 0.0,
                max_colors=self._max_colors(),
                temporary_store=self.temporary_store,
            )

            if self.coloring:
                self._compute_coloring(pic, scratch=scratch)

        return scratch

    def _particles_to_cells(self, positions: wp.array) -> fem.PicQuadrature:
        """Rebuild the grid and grid partition around particles, then assign particles to grid cells."""

        # Rebuild grid

        if self._scratchpad is not None and self.grid_type == "fixed":
            grid = self._scratchpad.grid
        else:
            grid = self._allocate_grid(
                positions,
                self.model.particle_flags,
                voxel_size=self._mpm_model.voxel_size,
                temporary_store=self.temporary_store,
                padding_voxels=self.grid_padding,
            )

        # Build active partition
        with self._timer("Build active partition"):
            if self.grid_type == "sparse":
                max_cell_count = -1
                geo_partition = grid
            else:
                max_cell_count = self.max_active_cell_count
                geo_partition = self._create_geometry_partition(
                    grid, positions, self.model.particle_flags, max_cell_count
                )

        # Bin particles to grid cells
        with self._timer("Bin particles"):
            domain = fem.Cells(geo_partition)

            if self.gimp:
                particle_locations = self._particle_grid_locations_gimp(
                    domain, positions, self._mpm_model.particle_radius
                )
            else:
                particle_locations = self._particle_grid_locations(domain, positions)

            pic = fem.PicQuadrature(
                domain=domain,
                positions=particle_locations,
                measures=self._mpm_model.particle_volume,
                temporary_store=self.temporary_store,
                use_domain_element_indices=True,
            )

        return pic

    def _particle_grid_locations(self, domain: fem.GeometryDomain, positions: wp.array) -> wp.array:
        """Convert particle positions to grid locations."""

        cell_lookup = domain.element_partition_lookup

        @fem.cache.dynamic_kernel(suffix=domain.name)
        def particle_locations(
            cell_arg_value: domain.ElementArg,
            domain_index_arg_value: domain.ElementIndexArg,
            positions: wp.array[wp.vec3],
            cell_index: wp.array[fem.ElementIndex],
            cell_coords: wp.array[fem.Coords],
        ):
            p = wp.tid()
            domain_arg = domain.DomainArg(cell_arg_value, domain_index_arg_value)

            sample = cell_lookup(domain_arg, positions[p])

            cell_index[p] = domain.element_partition_index(domain_index_arg_value, sample.element_index)
            cell_coords[p] = sample.element_coords

        device = positions.device

        cell_indices = fem.borrow_temporary(self.temporary_store, shape=positions.shape[0], dtype=fem.ElementIndex)
        cell_coords = fem.borrow_temporary(self.temporary_store, shape=positions.shape[0], dtype=fem.Coords)
        wp.launch(
            particle_locations,
            dim=positions.shape[0],
            inputs=[
                domain.element_arg_value(device=device),
                domain.element_index_arg_value(device=device),
                positions,
                cell_indices,
                cell_coords,
            ],
            device=device,
        )

        return cell_indices, cell_coords

    def _particle_grid_locations_gimp(
        self, domain: fem.GeometryDomain, positions: wp.array, radii: wp.array
    ) -> wp.array:
        """Convert particle positions to grid locations."""

        cell_lookup = domain.element_partition_lookup
        cell_closest_point = domain.element_closest_point

        @wp.func
        def add_cell(
            particle_cell_indices: wp.array[fem.ElementIndex],
            particle_cell_coords: wp.array[fem.Coords],
            particle_cell_fractions: wp.array[float],
            cell_index: int,
            cell_coords: fem.Coords,
            cell_weight: float,
        ):
            for i in range(8):
                if particle_cell_indices[i] == fem.NULL_NODE_INDEX:
                    particle_cell_indices[i] = cell_index
                    particle_cell_coords[i] = cell_coords
                    particle_cell_fractions[i] = cell_weight
                    return

                if particle_cell_indices[i] == cell_index:
                    particle_cell_fractions[i] += cell_weight
                    return

        @fem.cache.dynamic_kernel(suffix=domain.name)
        def particle_locations_gimp(
            cell_arg_value: domain.ElementArg,
            domain_index_arg_value: domain.ElementIndexArg,
            positions: wp.array[wp.vec3],
            radii: wp.array[float],
            cell_index: wp.array2d[fem.ElementIndex],
            cell_coords: wp.array2d[fem.Coords],
            cell_fractions: wp.array2d[float],
        ):
            p = wp.tid()
            domain_arg = domain.DomainArg(cell_arg_value, domain_index_arg_value)

            center = positions[p]
            radius = radii[p]

            tot_weight = float(0.0)

            # Find cell containing each corner of the particle,
            # merging repeated cell indices
            for vtx in range(8):
                i = (vtx & 4) >> 2
                j = (vtx & 2) >> 1
                k = vtx & 1

                pos = center - wp.vec3(radius) + 2.0 * radius * wp.vec3(float(i), float(j), float(k))
                sample = cell_lookup(domain_arg, pos)

                if sample.element_index == fem.NULL_ELEMENT_INDEX:
                    continue

                elem_index = domain.element_partition_index(domain_index_arg_value, sample.element_index)
                cell_weight = wp.min(wp.min(sample.element_coords), 1.0 - wp.max(sample.element_coords))

                if cell_weight > 0.0:
                    tot_weight += cell_weight
                    cell_center_coords, _ = cell_closest_point(cell_arg_value, sample.element_index, center)
                    add_cell(
                        cell_index[p],
                        cell_coords[p],
                        cell_fractions[p],
                        elem_index,
                        cell_center_coords,
                        cell_weight,
                    )

            # Normalize the weights over the cells
            for vtx in range(8):
                if cell_index[p, vtx] != fem.NULL_NODE_INDEX:
                    cell_fractions[p, vtx] /= tot_weight

        device = positions.device

        cell_indices = fem.borrow_temporary(self.temporary_store, shape=(positions.shape[0], 8), dtype=fem.ElementIndex)
        cell_coords = fem.borrow_temporary(self.temporary_store, shape=(positions.shape[0], 8), dtype=fem.Coords)
        cell_fractions = fem.borrow_temporary(self.temporary_store, shape=(positions.shape[0], 8), dtype=float)

        cell_indices.fill_(fem.NULL_NODE_INDEX)

        wp.launch(
            particle_locations_gimp,
            dim=positions.shape[0],
            inputs=[
                domain.element_arg_value(device=device),
                domain.element_index_arg_value(device=device),
                positions,
                radii,
                cell_indices,
                cell_coords,
                cell_fractions,
            ],
            device=device,
        )

        return cell_indices, cell_coords, cell_fractions

    def _step_impl(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: ImplicitMPMScratchpad,
    ):
        """Single implicit MPM step: bin, rasterize, assemble, solve, advect.

        Executes the full pipeline for one time step, including particle
        binning, collider rasterization, RHS assembly, strain/compliance matrix
        computation, warm-starting, coupled rheology/contact solve, strain
        updates, and particle advection.

        Args:
            state_in: Input state at the beginning of the timestep.
            state_out: Output state to write to.
            dt: Timestep length.
            pic: Particle-in-cell quadrature data.
            scratch: Scratchpad for temporary storage.
        """

        cell_volume = self._mpm_model.voxel_size**3
        inv_cell_volume = 1.0 / cell_volume

        mpm_model = self._mpm_model
        last_step_data = self._last_step_data

        self._require_collision_space_fields(scratch, last_step_data)
        self._require_velocity_space_fields(scratch, mpm_model.has_compliant_particles)

        # Rasterize colliders to discrete space
        self._rasterize_colliders(state_in, dt, last_step_data, scratch, inv_cell_volume)

        # Velocity right-hand side and inverse mass matrix
        self._compute_unconstrained_velocity(state_in, dt, pic, scratch, inv_cell_volume)

        # Build collider rigidity matrix
        rigidity_operator = self._build_collider_rigidity_operator(state_in, scratch, cell_volume)

        self._require_strain_space_fields(scratch, last_step_data)

        # Build elasticity compliance matrix and right-hand-side
        self._build_elasticity_system(state_in, dt, pic, scratch, inv_cell_volume)

        # Build strain matrix and offset, setup yield surface parameters
        self._build_plasticity_system(state_in, dt, pic, scratch, inv_cell_volume)

        # Solve implicit system
        self._load_warmstart(state_in, last_step_data, scratch, pic, inv_cell_volume)

        # Solve implicit system
        # Keep _solve_graph alive until end of function as destruction may cause sync point
        _solve_graph = self._solve_rheology(pic, scratch, rigidity_operator, last_step_data, inv_cell_volume)

        self._save_for_next_warmstart(scratch, pic, last_step_data)

        # Update and advect particles
        self._update_particles(state_in, state_out, dt, pic, scratch)

        # Save data for next step or further processing
        self._save_data(state_in, scratch, last_step_data, state_out)

    def _compute_unconstrained_velocity(
        self,
        state_in: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        """Compute the unconstrained (ballistic) velocity at grid nodes, as well as inverse mass matrix."""

        model = self.model
        mpm_model = self._mpm_model

        with self._timer("Unconstrained velocity"):
            velocity_int = fem.integrate(
                integrate_velocity,
                quadrature=pic,
                fields={"u": scratch.velocity_test},
                values={
                    "velocities": state_in.particle_qd,
                    "dt": dt,
                    "gravity": model.gravity,
                    "particle_world": model.particle_world,
                    "particle_density": mpm_model.particle_density,
                    "inv_cell_volume": inv_cell_volume,
                },
                output_dtype=wp.vec3,
                temporary_store=self.temporary_store,
            )

            if self.apic:
                fem.integrate(
                    integrate_velocity_apic,
                    quadrature=pic,
                    fields={"u": scratch.velocity_test},
                    values={
                        "velocity_gradients": state_in.mpm.particle_qd_grad,
                        "particle_density": mpm_model.particle_density,
                        "inv_cell_volume": inv_cell_volume,
                    },
                    output=velocity_int,
                    add=True,
                    temporary_store=self.temporary_store,
                )

            node_particle_mass = fem.integrate(
                integrate_mass,
                quadrature=pic,
                fields={"phi": scratch.fraction_test},
                values={
                    "inv_cell_volume": inv_cell_volume,
                    "particle_density": mpm_model.particle_density,
                },
                output_dtype=float,
                temporary_store=self.temporary_store,
            )

            drag = mpm_model.air_drag * dt

            wp.launch(
                free_velocity,
                dim=scratch.velocity_node_count,
                inputs=[
                    velocity_int,
                    node_particle_mass,
                    drag,
                ],
                outputs=[
                    scratch.inv_mass_matrix,
                    scratch.velocity_field.dof_values,
                ],
            )

    def _rasterize_colliders(
        self,
        state_in: newton.State,
        dt: float,
        last_step_data: LastStepData,
        scratch: ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        # Rasterize collider to grid
        collider_node_count = scratch.collider_node_count
        vel_node_count = scratch.velocity_node_count

        with self._timer("Rasterize collider"):
            # volume associated to each collider node
            fem.integrate(
                integrate_fraction,
                fields={"phi": scratch.collider_fraction_test},
                values={"inv_cell_volume": inv_cell_volume},
                assembly="nodal",
                output=scratch.collider_node_volume,
                temporary_store=self.temporary_store,
            )

            # rasterize sdf and properties to grid
            rasterize_collider(
                self._mpm_model.collider,
                state_in.body_q,
                state_in.body_qd if self.collider_velocity_mode == "forward" else None,
                last_step_data.body_q_prev if self.collider_velocity_mode == "backward" else None,
                self._mpm_model.voxel_size,
                dt,
                scratch.collider_fraction_test.space_restriction,
                scratch.collider_node_volume,
                scratch.collider_position_field,
                scratch.collider_distance_field,
                scratch.collider_normal_field,
                scratch.collider_velocity,
                scratch.collider_friction,
                scratch.collider_adhesion,
                scratch.collider_ids,
                temporary_store=self.temporary_store,
            )

            # normal interpolation
            if self.collider_normal_from_sdf_gradient:
                interpolate_collider_normals(
                    scratch.collider_fraction_test.space_restriction,
                    scratch.collider_distance_field,
                    scratch.collider_normal_field,
                    temporary_store=self.temporary_store,
                )

            # Subgrid collisions
            if self.collider_basis != self.velocity_basis:
                #  Map from collider nodes to velocity nodes
                wps.bsr_set_zero(
                    scratch.collider_matrix, rows_of_blocks=collider_node_count, cols_of_blocks=vel_node_count
                )
                fem.interpolate(
                    collision_weight_field,
                    dest=scratch.collider_matrix,
                    dest_space=scratch.collider_fraction_test.space,
                    at=scratch.collider_fraction_test.space_restriction,
                    reduction="first",
                    fields={"trial": scratch.fraction_trial, "normal": scratch.collider_normal_field},
                    temporary_store=self.temporary_store,
                )

    def _build_collider_rigidity_operator(
        self,
        state_in: newton.State,
        scratch: ImplicitMPMScratchpad,
        cell_volume: float,
    ):
        has_compliant_colliders = self._mpm_model.min_collider_mass < INFINITY

        if not has_compliant_colliders:
            return None

        with self._timer("Collider compliance"):
            rigidity_operator = build_rigidity_operator(
                cell_volume=cell_volume,
                node_volumes=scratch.collider_node_volume,
                node_positions=scratch.collider_position_field.dof_values,
                collider=self._mpm_model.collider,
                body_q=state_in.body_q,
                body_mass=self._mpm_model.collider_body_mass,
                body_inv_inertia=self._mpm_model.collider_body_inv_inertia,
                collider_ids=scratch.collider_ids,
            )

        return rigidity_operator

    def _build_elasticity_system(
        self,
        state_in: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        """Build the elasticity and compliance system."""

        mpm_model = self._mpm_model

        if not mpm_model.has_compliant_particles:
            scratch.elastic_strain_delta_field.dof_values.zero_()
            return

        with self._timer("Elasticity"):
            node_particle_volume = fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": scratch.fraction_test},
                values={"inv_cell_volume": inv_cell_volume},
                output_dtype=float,
                temporary_store=self.temporary_store,
            )

            elastic_parameters_int = fem.integrate(
                integrate_elastic_parameters,
                quadrature=pic,
                fields={"u": scratch.velocity_test},
                values={
                    "material_parameters": mpm_model.material_parameters,
                    "inv_cell_volume": inv_cell_volume,
                },
                output_dtype=wp.vec3,
                temporary_store=self.temporary_store,
            )

            wp.launch(
                average_elastic_parameters,
                dim=scratch.elastic_parameters_field.space_partition.node_count(),
                inputs=[
                    elastic_parameters_int,
                    node_particle_volume,
                    scratch.elastic_parameters_field.dof_values,
                ],
            )

            fem.integrate(
                strain_rhs,
                quadrature=pic,
                fields={
                    "tau": scratch.sym_strain_test,
                    "elastic_parameters": scratch.elastic_parameters_field,
                },
                values={
                    "elastic_strains": state_in.mpm.particle_elastic_strain,
                    "inv_cell_volume": inv_cell_volume,
                    "dt": dt,
                },
                temporary_store=self.temporary_store,
                output=scratch.elastic_strain_delta_field.dof_values,
            )

            fem.integrate(
                compliance_form,
                quadrature=pic,
                fields={
                    "tau": scratch.sym_strain_test,
                    "sig": scratch.sym_strain_trial,
                    "elastic_parameters": scratch.elastic_parameters_field,
                },
                values={
                    "elastic_strains": state_in.mpm.particle_elastic_strain,
                    "inv_cell_volume": inv_cell_volume,
                    "dt": dt,
                },
                output=scratch.compliance_matrix,
                temporary_store=self.temporary_store,
            )

    def _build_plasticity_system(
        self,
        state_in: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        mpm_model = self._mpm_model

        with self._timer("Interpolated yield parameters"):
            fem.integrate(
                integrate_yield_parameters,
                quadrature=pic,
                fields={
                    "u": scratch.strain_yield_parameters_test,
                },
                values={
                    "particle_Jp": state_in.mpm.particle_Jp,
                    "material_parameters": mpm_model.material_parameters,
                    "inv_cell_volume": inv_cell_volume,
                    "dt": dt,
                },
                output=scratch.strain_yield_parameters_field.dof_values,
                temporary_store=self.temporary_store,
            )

            fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": scratch.divergence_test},
                values={"inv_cell_volume": inv_cell_volume},
                output=scratch.strain_node_particle_volume,
                temporary_store=self.temporary_store,
            )

        # Void fraction (unilateral incompressibility offset)
        if mpm_model.critical_fraction > 0.0:
            with self._timer("Unilateral offset"):
                fem.integrate(
                    integrate_fraction,
                    fields={"phi": scratch.divergence_test},
                    values={"inv_cell_volume": inv_cell_volume},
                    output=scratch.strain_node_volume,
                    temporary_store=self.temporary_store,
                )

                if isinstance(scratch.collider_distance_field.space.basis, fem.PointBasisSpace):
                    fem.integrate(
                        integrate_collider_fraction_apic,
                        fields={
                            "phi": scratch.divergence_test,
                            "sdf": scratch.collider_distance_field,
                            "sdf_gradient": scratch.collider_normal_field,
                        },
                        values={
                            "inv_cell_volume": inv_cell_volume,
                        },
                        output=scratch.strain_node_collider_volume,
                        temporary_store=self.temporary_store,
                    )
                else:
                    fem.integrate(
                        integrate_collider_fraction,
                        fields={
                            "phi": scratch.divergence_test,
                            "sdf": scratch.collider_distance_field,
                        },
                        values={
                            "inv_cell_volume": inv_cell_volume,
                        },
                        output=scratch.strain_node_collider_volume,
                        temporary_store=self.temporary_store,
                    )

                wp.launch(
                    compute_unilateral_strain_offset,
                    dim=scratch.strain_node_count,
                    inputs=[
                        mpm_model.critical_fraction,
                        scratch.strain_node_particle_volume,
                        scratch.strain_node_collider_volume,
                        scratch.strain_node_volume,
                        scratch.unilateral_strain_offset,
                    ],
                )
        else:
            scratch.unilateral_strain_offset.zero_()

        # Strain jacobian
        with self._timer("Strain matrix"):
            fem.integrate(
                strain_delta_form,
                quadrature=pic,
                fields={
                    "u": scratch.velocity_trial,
                    "tau": scratch.divergence_test,
                },
                values={
                    "dt": dt,
                    "inv_cell_volume": inv_cell_volume,
                },
                output_dtype=float,
                output=scratch.strain_matrix,
                temporary_store=self.temporary_store,
                bsr_options={"prune_numerical_zeros": self._velocity_nodes_per_strain_sample < 0},
            )

    def _build_strain_eigenbasis(
        self,
        pic: fem.PicQuadrature,
        scratch: ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        if self.strain_basis in ("Q1", "S2"):
            scratch.strain_node_particle_volume += EPSILON
            return None, None
        elif self.strain_basis[:3] == "pic":
            M_diag = scratch.strain_node_particle_volume
            M_diag.assign(self._mpm_model.particle_volume * inv_cell_volume)
            return None, None

        # build mass matrix of PIC integration
        M = fem.integrate(
            mass_form,
            quadrature=pic,
            fields={"p": scratch.divergence_test, "q": scratch.divergence_trial},
            values={"inv_cell_volume": inv_cell_volume},
            output_dtype=float,
        )

        # extract diagonal blocks
        nodes_per_elt = scratch.divergence_test.space.topology.MAX_NODES_PER_ELEMENT
        M_elt_wise = wps.bsr_copy(M, block_shape=(nodes_per_elt, nodes_per_elt))

        if M_elt_wise.block_shape == (1, 1):
            M_values = M_elt_wise.values.view(dtype=mat11)
        else:
            M_values = M_elt_wise.values

        M_ev = wp.empty(shape=(M_elt_wise.nrow, *M_elt_wise.block_shape), dtype=M_elt_wise.scalar_type)
        M_diag = scratch.strain_node_particle_volume.reshape((-1, nodes_per_elt))
        rotated_volume = wp.empty_like(M_diag)

        wp.launch(
            compute_eigenvalues,
            dim=M_elt_wise.nrow,
            inputs=[
                M_elt_wise.offsets,
                M_elt_wise.columns,
                M_values,
                M_diag,
                scratch.strain_yield_parameters_field.dof_values,
            ],
            outputs=[
                M_diag,
                M_ev,
                rotated_volume,
            ],
        )

        return M_ev, rotated_volume.reshape((-1,))

    def _apply_strain_eigenbasis(
        self,
        scratch: ImplicitMPMScratchpad,
        M_ev: wp.array3d[float],
        rotated_volume=None,
    ):
        node_count = scratch.strain_node_count

        if M_ev is not None and M_ev.shape[1] > 1:
            # Rotate matrix and vectors according to eigenbasis

            nodes_per_elt = M_ev.shape[1]
            elt_count = M_ev.shape[0]

            B = scratch.strain_matrix
            strain_mat_tmp = wp.empty_like(B.values)
            wp.launch(rotate_matrix_rows, dim=B.nnz, inputs=[M_ev, B.offsets, B.columns, B.values, strain_mat_tmp])
            B.values = strain_mat_tmp

            C = scratch.compliance_matrix
            compliance_mat_tmp = wp.empty_like(C.values)
            wp.launch(rotate_matrix_rows, dim=C.nnz, inputs=[M_ev, C.offsets, C.columns, C.values, compliance_mat_tmp])
            wp.launch(
                rotate_matrix_columns, dim=C.nnz, inputs=[M_ev, C.offsets, C.columns, compliance_mat_tmp, C.values]
            )

            rotate_vectors = make_rotate_vectors(nodes_per_elt)
            wp.launch_tiled(
                rotate_vectors,
                dim=elt_count,
                block_dim=32,
                inputs=[
                    M_ev,
                    _as_2d_array(scratch.elastic_strain_delta_field.dof_values, shape=(node_count, 6), dtype=float),
                    _as_2d_array(scratch.stress_field.dof_values, shape=(node_count, 6), dtype=float),
                    _as_2d_array(
                        scratch.strain_yield_parameters_field.dof_values,
                        shape=(node_count, YIELD_PARAM_LENGTH),
                        dtype=float,
                    ),
                    _as_2d_array(scratch.unilateral_strain_offset, shape=(node_count, 1), dtype=float),
                ],
            )

        M_diag = scratch.strain_node_particle_volume
        if self._stress_warmstart == "particles":
            # Particle stresses are integrated, need scale with inverse node volume

            wp.launch(
                inverse_scale_sym_tensor,
                dim=node_count,
                inputs=[M_diag, scratch.stress_field.dof_values],
            )

        # Yield parameters are integrated, scale with inverse rotated volume
        # to correctly recover uniform parameters after eigenbasis rotation
        yield_volume = rotated_volume if rotated_volume is not None else M_diag
        wp.launch(
            inverse_scale_vector,
            dim=node_count,
            inputs=[yield_volume, scratch.strain_yield_parameters_field.dof_values],
        )

    def _unapply_strain_eigenbasis(
        self,
        scratch: ImplicitMPMScratchpad,
        M_ev: wp.array3d[float],
    ):
        node_count = scratch.strain_node_count

        # Un-integrate strains by scaling with inverse node volume
        M_diag = scratch.strain_node_particle_volume
        if self._mpm_model.has_compliant_particles:
            wp.launch(
                inverse_scale_sym_tensor,
                dim=node_count,
                inputs=[M_diag, scratch.elastic_strain_delta_field.dof_values],
            )
        if self._mpm_model.has_hardening:
            wp.launch(
                inverse_scale_sym_tensor,
                dim=node_count,
                inputs=[M_diag, scratch.plastic_strain_delta_field.dof_values],
            )

        if M_ev is not None and M_ev.shape[1] > 1:
            # Un-rotate vectors according to eigenbasis

            elt_count = M_ev.shape[0]
            nodes_per_elt = M_ev.shape[1]

            inverse_rotate_vectors = make_inverse_rotate_vectors(nodes_per_elt)
            wp.launch_tiled(
                inverse_rotate_vectors,
                dim=elt_count,
                block_dim=32,
                inputs=[
                    M_ev,
                    _as_2d_array(scratch.stress_field.dof_values, shape=(node_count, 6), dtype=float),
                    _as_2d_array(scratch.elastic_strain_delta_field.dof_values, shape=(node_count, 6), dtype=float),
                    _as_2d_array(scratch.plastic_strain_delta_field.dof_values, shape=(node_count, 6), dtype=float),
                ],
            )

    def _solve_rheology(
        self,
        pic: fem.PicQuadrature,
        scratch: ImplicitMPMScratchpad,
        rigidity_operator: tuple[wps.BsrMatrix, wps.BsrMatrix, wps.BsrMatrix] | None,
        last_step_data: LastStepData,
        inv_cell_volume: float,
    ):
        M_ev, rotated_volume = self._build_strain_eigenbasis(pic, scratch, inv_cell_volume)

        self._apply_strain_eigenbasis(scratch, M_ev, rotated_volume)

        with self._timer("Strain solve"):
            momentum_data = MomentumData(
                inv_volume=scratch.inv_mass_matrix,
                velocity=scratch.velocity_field.dof_values,
            )
            rheology_data = RheologyData(
                strain_mat=scratch.strain_matrix,
                transposed_strain_mat=scratch.transposed_strain_matrix,
                compliance_mat=scratch.compliance_matrix,
                strain_node_volume=scratch.strain_node_particle_volume,
                yield_params=scratch.strain_yield_parameters_field.dof_values,
                unilateral_strain_offset=scratch.unilateral_strain_offset,
                color_offsets=scratch.color_offsets,
                color_blocks=scratch.color_indices,
                elastic_strain_delta=scratch.elastic_strain_delta_field.dof_values,
                plastic_strain_delta=scratch.plastic_strain_delta_field.dof_values,
                stress=scratch.stress_field.dof_values,
                has_viscosity=self._mpm_model.has_viscosity,
                has_dilatancy=self._mpm_model.has_dilatancy,
                strain_velocity_node_count=self._velocity_nodes_per_strain_sample,
            )
            collision_data = CollisionData(
                collider_mat=scratch.collider_matrix,
                transposed_collider_mat=scratch.transposed_collider_matrix,
                collider_friction=scratch.collider_friction,
                collider_adhesion=scratch.collider_adhesion,
                collider_normals=scratch.collider_normal_field.dof_values,
                collider_velocities=scratch.collider_velocity,
                rigidity_operator=rigidity_operator,
                collider_impulse=scratch.impulse_field.dof_values,
                has_colliders=self._mpm_model.collider.collider_mesh.shape[0] > 0,
            )

            # Retain graph to avoid immediate CPU sync
            solve_graph = solve_rheology(
                self.solver,
                self.max_iterations,
                self.tolerance,
                momentum_data,
                rheology_data,
                collision_data,
                temporary_store=self.temporary_store,
                use_graph=self._use_cuda_graph,
                verbose=self.verbose,
            )

        self._unapply_strain_eigenbasis(scratch, M_ev)

        return solve_graph

    def _update_particles(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: ImplicitMPMScratchpad,
    ):
        """Update particle quantities (strains, velocities, ...) from grid fields an advect them."""

        model = self.model
        mpm_model = self._mpm_model

        has_compliant_particles = mpm_model.min_young_modulus < INFINITY
        has_hardening = mpm_model.max_hardening > 0.0

        if self._stress_warmstart == "particles" or has_compliant_particles or has_hardening:
            with self._timer("Particle strain update"):
                # Update particle elastic strain from grid strain delta

                if state_in is state_out:
                    elastic_strain_prev = wp.clone(state_in.mpm.particle_elastic_strain)
                    particle_Jp_prev = wp.clone(state_in.mpm.particle_Jp)
                else:
                    elastic_strain_prev = state_in.mpm.particle_elastic_strain
                    particle_Jp_prev = state_in.mpm.particle_Jp

                state_out.mpm.particle_Jp.zero_()
                state_out.mpm.particle_stress.zero_()
                state_out.mpm.particle_elastic_strain.zero_()

                fem.interpolate(
                    update_particle_strains,
                    at=pic,
                    values={
                        "dt": dt,
                        "particle_flags": model.particle_flags,
                        "particle_density": mpm_model.particle_density,
                        "particle_volume": mpm_model.particle_volume,
                        "elastic_strain_prev": elastic_strain_prev,
                        "elastic_strain": state_out.mpm.particle_elastic_strain,
                        "particle_stress": state_out.mpm.particle_stress,
                        "particle_Jp_prev": particle_Jp_prev,
                        "particle_Jp": state_out.mpm.particle_Jp,
                        "material_parameters": mpm_model.material_parameters,
                    },
                    fields={
                        "grid_vel": scratch.velocity_field,
                        "plastic_strain_delta": scratch.plastic_strain_delta_field,
                        "elastic_strain_delta": scratch.elastic_strain_delta_field,
                        "stress": scratch.stress_field,
                    },
                    temporary_store=self.temporary_store,
                )

        # (A)PIC advection
        with self._timer("Advection"):
            state_out.particle_qd.zero_()
            state_out.mpm.particle_qd_grad.zero_()
            state_out.particle_q.assign(state_in.particle_q)

            fem.interpolate(
                advect_particles,
                at=pic,
                values={
                    "particle_flags": model.particle_flags,
                    "particle_volume": mpm_model.particle_volume,
                    "pos": state_out.particle_q,
                    "vel": state_out.particle_qd,
                    "vel_grad": state_out.mpm.particle_qd_grad,
                    "dt": dt,
                    "max_vel": model.particle_max_velocity,
                },
                fields={
                    "grid_vel": scratch.velocity_field,
                },
                temporary_store=self.temporary_store,
            )

    def _save_data(
        self,
        state_in: newton.State,
        scratch: ImplicitMPMScratchpad,
        last_step_data: LastStepData,
        state_out: newton.State,
    ):
        """Save data for next step or further processing."""

        # Copy current body_q to last_step_data.body_q_prev for next step's velocity computation
        last_step_data.save_collider_current_position(state_in.body_q)

        # Necessary fields for two-way coupling
        state_out.impulse_field = scratch.impulse_field
        state_out.collider_ids = scratch.collider_ids
        state_out.collider_position_field = scratch.collider_position_field
        state_out.collider_distance_field = scratch.collider_distance_field
        state_out.collider_normal_field = scratch.collider_normal_field

        # Necessary fields for grains rendering
        # Re-generated at each step, defined on space partition
        state_out.velocity_field = scratch.velocity_field

    def _require_velocity_space_fields(self, scratch: ImplicitMPMScratchpad, has_compliant_particles: bool):
        """Ensure velocity-space fields exist and match current spaces."""

        scratch.require_velocity_space_fields(has_compliant_particles)

    def _require_collision_space_fields(self, scratch: ImplicitMPMScratchpad, last_step_data: LastStepData):
        """Ensure collision-space fields exist and match current spaces."""
        scratch.require_collision_space_fields()
        last_step_data.require_collision_space_fields(scratch)
        last_step_data.require_collider_previous_position(self._mpm_model.collider_body_q)

    def _require_strain_space_fields(self, scratch: ImplicitMPMScratchpad, last_step_data: LastStepData):
        """Ensure strain-space fields exist and match current spaces."""
        scratch.require_strain_space_fields()
        last_step_data.require_strain_space_fields(scratch, smoothed=self._stress_warmstart == "smoothed")

    def _load_warmstart(
        self,
        state_in: newton.State,
        last_step_data: LastStepData,
        scratch: ImplicitMPMScratchpad,
        pic: fem.PicQuadrature,
        inv_cell_volume: float,
    ):
        with self._timer("Warmstart fields"):
            self._warmstart_fields(last_step_data, scratch, pic)

            if self._stress_warmstart == "particles":
                fem.integrate(
                    integrate_particle_stress,
                    quadrature=pic,
                    fields={
                        "tau": scratch.sym_strain_test,
                    },
                    values={
                        "particle_stress": state_in.mpm.particle_stress,
                        "inv_cell_volume": inv_cell_volume,
                    },
                    output=scratch.stress_field.dof_values,
                )
            elif not self._stress_warmstart:
                scratch.stress_field.dof_values.zero_()

    def _warmstart_fields(
        self,
        last_step_data: LastStepData,
        scratch: ImplicitMPMScratchpad,
        pic: fem.PicQuadrature,
    ):
        """Interpolate previous grid fields into the current grid layout.

        Transfers impulse and stress fields from the previous grid to the new
        grid (handling nonconforming cases), and initializes the output state's
        grid fields to the current scratchpad fields.
        """

        prev_impulse_field = last_step_data.ws_impulse_field
        prev_stress_field = last_step_data.ws_stress_field

        domain = scratch.velocity_test.domain

        if isinstance(prev_impulse_field.space.basis, fem.PointBasisSpace):
            # point-based collisions, simply copy the previous impulses
            scratch.impulse_field.dof_values.assign(prev_impulse_field.dof_values[pic.cell_particle_indices])
        else:
            # Interpolate previous impulse
            prev_impulse_field = fem.NonconformingField(
                domain, prev_impulse_field, background=scratch.background_impulse_field
            )
            fem.interpolate(
                prev_impulse_field,
                dest=scratch.impulse_field,
                at=scratch.collider_fraction_test.space_restriction,
                reduction="first",
                temporary_store=self.temporary_store,
            )

        # Interpolate previous stress
        if isinstance(prev_stress_field.space.basis, fem.PointBasisSpace):
            scratch.stress_field.dof_values.assign(prev_stress_field.dof_values[pic.cell_particle_indices])
        elif self._stress_warmstart in ("grid", "smoothed"):
            prev_stress_field = fem.NonconformingField(
                domain, prev_stress_field, background=scratch.background_stress_field
            )
            fem.interpolate(
                prev_stress_field,
                dest=scratch.stress_field,
                at=scratch.sym_strain_test.space_restriction,
                reduction="first",
                temporary_store=self.temporary_store,
            )

    def _save_for_next_warmstart(
        self, scratch: ImplicitMPMScratchpad, pic: fem.PicQuadrature, last_step_data: LastStepData
    ):
        with self._timer("Save warmstart fields"):
            last_step_data.rebind_collision_space_fields(scratch)

            if isinstance(last_step_data.ws_impulse_field.space.basis, fem.PointBasisSpace):
                # point-based collisions, simply copy the previous impulses
                last_step_data.ws_impulse_field.dof_values[pic.cell_particle_indices].assign(
                    scratch.impulse_field.dof_values
                )
            else:
                last_step_data.ws_impulse_field.dof_values.zero_()
                wp.launch(
                    scatter_field_dof_values,
                    dim=scratch.impulse_field.space_partition.node_count(),
                    inputs=[
                        scratch.impulse_field.space_partition.space_node_indices(),
                        scratch.impulse_field.dof_values,
                        last_step_data.ws_impulse_field.dof_values,
                    ],
                )

            last_step_data.rebind_strain_space_fields(scratch, smoothed=self._stress_warmstart == "smoothed")
            if isinstance(last_step_data.ws_stress_field.space.basis, fem.PointBasisSpace):
                last_step_data.ws_stress_field.dof_values[pic.cell_particle_indices].assign(
                    scratch.stress_field.dof_values
                )
            else:
                last_step_data.ws_stress_field.dof_values.zero_()

                fem.interpolate(
                    scratch.stress_field,
                    dest=last_step_data.ws_stress_field,
                )

    def _max_colors(self):
        if not self.coloring:
            return 0
        return 27 if self.strain_basis == "Q1" else self._scratchpad.velocity_nodes_per_element

    def _compute_coloring(
        self,
        pic: fem.PicQuadrature,
        scratch: ImplicitMPMScratchpad,
    ):
        """Compute Gauss-Seidel coloring of strain nodes to avoid write conflicts.

        Writes scratch.color_offsets, scratch.color_indices.
        """

        space_partition = scratch._strain_space_restriction.space_partition
        grid = space_partition.geo_partition.geometry

        is_pic = self.strain_basis[:3] == "pic"

        if not is_pic:
            nodes_per_color_element = scratch.strain_nodes_per_element
            is_dg = space_partition.space_topology.node_count() == nodes_per_color_element * grid.cell_count()

        if is_pic or is_dg:
            # cell-based coloring

            # nodes in each element solved sequentially
            stencil_size = int(np.round(np.cbrt(scratch.velocity_nodes_per_element)))
            if isinstance(grid, fem.Nanogrid):
                voxels = grid._cell_ijk
                res = wp.vec3i(0)
            else:
                voxels = None
                res = grid.res

            colored_element_count = space_partition.geo_partition.cell_count()
            partition_arg = space_partition.geo_partition.cell_arg_value(device=scratch.color_indices.device)

            colors = fem.borrow_temporary(self.temporary_store, shape=colored_element_count * 2 + 1, dtype=int)
            color_indices = scratch.color_indices.flatten()
            wp.launch(
                make_cell_color_kernel(space_partition.geo_partition),
                dim=colored_element_count,
                inputs=[
                    partition_arg,
                    stencil_size,
                    voxels,
                    res,
                    colors,
                    color_indices,
                ],
            )

        elif self.strain_basis == "Q1":
            nodes_per_color_element = 1
            stencil_size = 3
            if isinstance(grid, fem.Nanogrid):
                voxels = grid._node_ijk
                res = wp.vec3i(0)
            else:
                voxels = None
                res = grid.res + wp.vec3i(1)

            colored_element_count = space_partition.node_count()
            space_node_indices = space_partition.space_node_indices()

            colors = fem.borrow_temporary(self.temporary_store, shape=colored_element_count * 2 + 1, dtype=int)
            color_indices = scratch.color_indices.flatten()
            wp.launch(
                node_color,
                dim=colored_element_count,
                inputs=[
                    space_node_indices,
                    stencil_size,
                    voxels,
                    res,
                    colors,
                    color_indices,
                ],
            )
        else:
            raise RuntimeError("Unsupported strain basis for coloring")

        wp.utils.radix_sort_pairs(
            keys=colors,
            values=color_indices,
            count=colored_element_count,
        )

        unique_colors = colors[colored_element_count:]
        color_count = unique_colors[colored_element_count:]
        color_node_counts = color_indices[colored_element_count:]

        wp.utils.runlength_encode(
            colors,
            value_count=colored_element_count,
            run_values=unique_colors,
            run_lengths=color_node_counts,
            run_count=color_count,
        )
        wp.launch(
            compute_color_offsets,
            dim=1,
            inputs=[self._max_colors(), color_count, unique_colors, color_node_counts, scratch.color_offsets],
        )

        # build color ranges from cell/node color indices
        if is_pic:
            wp.launch(
                make_dynamic_color_block_indices_kernel(space_partition.geo_partition),
                dim=colored_element_count,
                inputs=[partition_arg, pic.cell_particle_offsets, scratch.color_indices],
            )
        else:
            wp.launch(
                fill_uniform_color_block_indices,
                dim=colored_element_count,
                inputs=[nodes_per_color_element, scratch.color_indices],
            )

        colors.release()

    def _timer(self, name: str):
        return wp.ScopedTimer(
            name,
            active=self.enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        )
