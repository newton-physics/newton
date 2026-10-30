# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit MPM solver."""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as wps

import newton

from ...core.types import override
from ...geometry.particle_surface import ParticleSurface
from ...sim import ModelFlags, StateFlags
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from .density_measurement import (
    critical_offset_values,
    deformed_particle_volumes,
    feedback_offset_values,
    make_free_box_mask_kernel,
    make_grid_capacity_kernel,
    make_grid_readback_kernel,
    make_grid_volume_kernel,
    make_particle_capacity_kernel,
    make_particle_volume_kernel,
    point_density_offset,
)
from .implicit_mpm_model import ImplicitMPMModel
from .particle_surface_colliders import extrapolate_surface_sdf_into_colliders
from .rasterized_collisions import (
    Collider,
    build_rigidity_operator,
    collider_distance_gradients,
    interpolate_collider_normals,
    point_contact_radii,
    project_outside_collider,
    rasterize_collider,
    speculative_stop_distance,
    stabilize_collider_velocity,
)
from .render_grains import sample_render_grains, update_render_grains
from .solve_rheology import CollisionData, MomentumData, RheologyData, YieldParamVec, solve_rheology

__all__ = ["SolverImplicitMPM"]

from .implicit_mpm_solver_kernels import (
    EPSILON,
    GIMP_MIN_OVERLAP_FRACTION,
    INFINITY,
    NODE_KEY_SENTINEL,
    YIELD_PARAM_LENGTH,
    YieldMomentVec,
    _rebuild_capacity,
    add_critical_strain_offset,
    advect_particles,
    allocate_by_voxels,
    average_elastic_parameters,
    build_active_particle_mask,
    collision_weight_field,
    compliance_form,
    compute_bounds,
    compute_color_offsets,
    compute_lattice_node_keys,
    compute_nodal_diagonalization,
    fill_uniform_color_block_indices,
    free_velocity,
    gather_point_dof_values,
    gather_sorted_node_values,
    gimp_domain_corners,
    gimp_half_size,
    gimp_store_point_stress,
    inertial_delassus_scale,
    integrate_active_fraction,
    integrate_elastic_parameters,
    integrate_fraction,
    integrate_gimp_point_stress,
    integrate_mass,
    integrate_particle_stress,
    integrate_transformed_yield_parameters,
    integrate_velocity,
    integrate_velocity_apic,
    integrate_yield_parameters,
    inverse_scale_sym_tensor,
    inverse_scale_vector,
    invert_point_particle_indices,
    load_values_by_lattice_key,
    make_cell_color_kernel,
    make_contact_cell_keys_kernel,
    make_discontinuous_color_block_indices_kernel,
    make_dynamic_color_block_indices_kernel,
    make_inverse_rotate_vectors,
    make_rotate_vectors,
    mark_active_cells,
    mark_active_cells_by_environment,
    mass_form,
    mat13,
    mat31,
    mat66,
    max_particle_density,
    node_color,
    record_volume_rebuild_status,
    reset_mpm_collider_history,
    reset_mpm_grid_warmstart,
    reset_mpm_particle_history,
    reset_mpm_point_warmstart,
    reset_particle_values,
    reset_values_by_lattice_key,
    rotate_matrix_columns,
    rotate_matrix_rows,
    scatter_field_dof_values,
    scatter_point_dof_values,
    strain_delta_form,
    strain_rhs,
    transfer_gimp_impulses,
    update_particle_frames,
    update_particle_strains,
    voxel_coordinates,
)

# Disabled until Warp includes the fix for NVIDIA/warp#2036 (see #4506).
_ROW_COMPRESSED_CONTACT_CONSTRUCTION = False


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


def _sparse_grid_rebuild_error(status: int) -> RuntimeError:
    capacity_flags = (
        (wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED, "active voxels"),
        (wp.Volume.REBUILD_LEAF_CAPACITY_EXCEEDED, "leaf nodes"),
        (wp.Volume.REBUILD_LOWER_CAPACITY_EXCEEDED, "lower internal nodes"),
        (wp.Volume.REBUILD_UPPER_CAPACITY_EXCEEDED, "upper internal nodes"),
    )
    exceeded = [name for flag, name in capacity_flags if status & flag]
    if exceeded:
        details = ", ".join(exceeded)
        suggestions = []
        if status & wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED:
            suggestions.append("increase Config.max_active_cell_count")
        if status & wp.Volume.REBUILD_LEAF_CAPACITY_EXCEEDED:
            suggestions.append("set or increase Config.max_leaf_node_count")
        if status & wp.Volume.REBUILD_LOWER_CAPACITY_EXCEEDED:
            suggestions.append("set or increase Config.max_lower_node_count")
        if status & wp.Volume.REBUILD_UPPER_CAPACITY_EXCEEDED:
            suggestions.append("set or increase Config.max_upper_node_count")
        if status & (
            wp.Volume.REBUILD_LEAF_CAPACITY_EXCEEDED
            | wp.Volume.REBUILD_LOWER_CAPACITY_EXCEEDED
            | wp.Volume.REBUILD_UPPER_CAPACITY_EXCEEDED
        ):
            suggestions.append("reduce the active grid's spatial spread")
        suggestion = " or ".join(suggestions)
        return RuntimeError(
            f"Implicit MPM sparse grid rebuild capacity was exceeded for {details} (status {status}). "
            f"To avoid overflow, {suggestion}."
        )
    return RuntimeError(f"Implicit MPM sparse grid rebuild failed with status {status}.")


@wp.kernel(enable_backward=False)
def _clear_sparse_grid_rebuild_status_masked(
    world_mask: wp.array[wp.bool],
    status: wp.array[wp.uint32],
    accumulated_status: wp.array[wp.uint32],
):
    for world in range(world_mask.shape[0]):
        if world_mask[world]:
            status[0] = wp.uint32(0)
            accumulated_status[0] = wp.uint32(0)
            return


def _validate_sparse_grid_node_capacity(name: str, value: int) -> int:
    """Validate one optional NanoVDB hierarchy capacity."""
    if isinstance(value, bool):
        raise ValueError(f"Config.{name} must be -1 or a positive integer, got {value!r}.")
    try:
        capacity = operator.index(value)
    except TypeError as error:
        raise ValueError(f"Config.{name} must be -1 or a positive integer, got {value!r}.") from error
    if capacity != -1 and not 0 < capacity <= np.iinfo(np.uint32).max:
        raise ValueError(
            f"Config.{name} must be -1 or a positive integer no greater than {np.iinfo(np.uint32).max}, got {capacity}."
        )
    return capacity


def _parse_grid_basis_name(basis_str: str) -> tuple[str, int, bool]:
    """Parse and validate Newton's compact grid-basis spelling."""
    match = re.fullmatch(r"([BQSP])([0-9]+)(d?)", basis_str)
    if match is None:
        raise ValueError(
            f"Unsupported basis: {basis_str}. Expected format: B<degree>[d], Q<degree>[d], "
            "S<degree>[d], P0, or P<positive-degree>d."
        )

    basis_type, degree_text, discontinuous_suffix = match.groups()
    degree = int(degree_text)
    discontinuous = degree == 0 or discontinuous_suffix == "d"
    if basis_type == "P" and not discontinuous:
        raise ValueError(f"Unsupported basis: {basis_str}. Non-conforming polynomial (P) bases must be discontinuous.")
    return basis_type, degree, discontinuous


def _make_grid_basis_space(grid: fem.Geometry, basis_str: str, family: fem.Polynomial | None = None):
    basis_type, degree, discontinuous = _parse_grid_basis_name(basis_str)

    if basis_type == "B":
        element_basis = fem.ElementBasis.BSPLINE
    elif basis_type == "Q":
        element_basis = fem.ElementBasis.LAGRANGE
    elif basis_type == "S":
        element_basis = fem.ElementBasis.SERENDIPITY
    elif basis_type == "P":
        element_basis = fem.ElementBasis.NONCONFORMING_POLYNOMIAL

    return fem.make_polynomial_basis_space(
        grid, degree=degree, element_basis=element_basis, family=family, discontinuous=discontinuous
    )


class _PicQuadrature(fem.PicQuadrature):
    """PIC quadrature whose cache name includes its domain.

    Warp 1.18 names PIC quadratures by index scope only, so point bases built on
    different geometries share generated kernels. Remove once Newton requires
    Warp 1.19, which includes the domain name itself (GH-1984).
    """

    @property
    def name(self):
        return f"{super().name}_{self.domain.name}"


def _make_pic_basis_space(pic: fem.PicQuadrature, basis_str: str):
    try:
        max_points_per_cell = int(basis_str[3:])
    except ValueError:
        max_points_per_cell = -1

    if max_points_per_cell < 0 and wp.get_device().is_capturing:
        raise RuntimeError("Uncapped 'pic' requires a host read; use an explicit 'picN' limit for graph capture.")

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
    "conjugate-gradient",
    "cr",
    "conjugate-residual",
    "gmres",
    "generalized-minimal-residual",
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
        self.particle_cells = None
        self.contact_pic = None
        self.collision_pic = None
        self.density_pic = None
        self.pic_domain_element_indices = False
        self.gimp_particle_contact_nodes = None
        self.gimp_domain_corners = None

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
        self.strain_basis_transform_field = None

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
        self.delassus_scale = None
        self.strain_basis_transform = None
        self.strain_stress_transform = None
        self.critical_strain_offset = None

        self.collider_total_volumes = None
        self.collider_node_volume = None

        self.velocity_environment_offsets = None
        self.collider_environment_offsets = None
        self.strain_environment_offsets = None

    def rebuild_function_spaces(
        self,
        pic: fem.PicQuadrature,
        velocity_basis_str: str,
        strain_basis_str: str,
        collider_basis_str: str,
        max_cell_count: int,
        environment_first: bool,
        temporary_store: fem.TemporaryStore,
        contact_pic: fem.PicQuadrature | None = None,
    ):
        """Define velocity and strain function spaces over the given geometry."""

        self.domain = pic.domain

        use_pic_collider_basis = collider_basis_str[:3] == "pic"
        use_pic_strain_basis = strain_basis_str[:3] == "pic"

        # The rebuildable sparse grid (like the fixed grid) reuses the same geometry object
        # across steps, so refresh retained topologies before rebuilding their partitions.
        if self.domain.geometry is not self.grid:
            self.grid = self.domain.geometry

            # Define function spaces: linear (Q1) for velocity and volume fraction,
            # zero or first order for pressure
            self._velocity_basis = _make_grid_basis_space(self.grid, velocity_basis_str)

            if not use_pic_strain_basis:
                family = fem.Polynomial.GAUSS_LEGENDRE if strain_basis_str == "Q1d" else None
                self._strain_basis = _make_grid_basis_space(self.grid, strain_basis_str, family=family)

            if not use_pic_collider_basis:
                self._collision_basis = _make_grid_basis_space(
                    self.grid, collider_basis_str, family=fem.Polynomial.EQUISPACED_CLOSED
                )
        else:
            topologies = {id(self._velocity_basis.topology): self._velocity_basis.topology}
            if not use_pic_strain_basis:
                topologies[id(self._strain_basis.topology)] = self._strain_basis.topology
            if not use_pic_collider_basis:
                topologies[id(self._collision_basis.topology)] = self._collision_basis.topology
            for topology in topologies.values():
                if hasattr(topology, "rebuild"):
                    topology.rebuild()

        # Point-based basis space needs to be rebuilt even when the geo does not change
        if use_pic_strain_basis:
            self._strain_basis = _make_pic_basis_space(pic, strain_basis_str)
        if use_pic_collider_basis:
            self.collision_pic = contact_pic if contact_pic is not None else pic
            self._collision_basis = _make_pic_basis_space(self.collision_pic, collider_basis_str)

        self._create_velocity_function_space(temporary_store, max_cell_count, environment_first)
        self._create_collider_function_space(temporary_store, max_cell_count, environment_first)
        self._create_strain_function_space(temporary_store, max_cell_count, environment_first)

    def _create_velocity_function_space(
        self, temporary_store: fem.TemporaryStore, max_cell_count: int, environment_first: bool
    ):
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
            environment_first=environment_first,
            temporary_store=temporary_store,
        )
        vel_space_restriction = fem.make_space_restriction(
            space_partition=vel_space_partition, domain=domain, temporary_store=temporary_store
        )

        self._velocity_space = velocity_space
        self._vel_space_restriction = vel_space_restriction
        self.velocity_environment_offsets = vel_space_partition.env_offsets if environment_first else None

    def _create_collider_function_space(
        self, temporary_store: fem.TemporaryStore, max_cell_count: int, environment_first: bool
    ):
        """Create collider function space and its partition/restriction."""

        if self._velocity_basis == self._collision_basis:
            self._collision_space = self._velocity_space
            self._collision_space_restriction = self._vel_space_restriction
            self.collider_environment_offsets = self.velocity_environment_offsets
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
            environment_first=environment_first,
            temporary_store=temporary_store,
        )
        collision_space_restriction = fem.make_space_restriction(
            space_partition=collision_space_partition, domain=domain, temporary_store=temporary_store
        )

        self._collision_space = collision_space
        self._collision_space_restriction = collision_space_restriction
        self.collider_environment_offsets = collision_space_partition.env_offsets if environment_first else None

    def _create_strain_function_space(
        self, temporary_store: fem.TemporaryStore, max_cell_count: int, environment_first: bool
    ):
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
            environment_first=environment_first,
            temporary_store=temporary_store,
        )

        strain_space_restriction = fem.make_space_restriction(
            space_partition=strain_space_partition, domain=domain, temporary_store=temporary_store
        )

        self._sym_strain_space = sym_strain_space
        self._strain_space_restriction = strain_space_restriction
        self.strain_environment_offsets = strain_space_partition.env_offsets if environment_first else None

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

    def require_strain_space_fields(self, transform_basis: bool):
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
        if transform_basis:
            mode_space = fem.make_collocated_function_space(
                strain_basis, dtype=wp.types.vector(length=self.strain_nodes_per_element, dtype=wp.float32)
            )
            if self.strain_basis_transform_field is None:
                self.strain_basis_transform_field = mode_space.make_field(space_partition=strain_space_partition)
            else:
                self.strain_basis_transform_field.rebind(mode_space, strain_space_partition)
        strain_yield_parameters_test_space = fem.make_collocated_function_space(strain_basis, dtype=YieldMomentVec)

        if self.sym_strain_test is None:
            self.sym_strain_test = fem.make_test(sym_strain_space, space_restriction=strain_space_restriction)
            self.divergence_test = fem.make_test(divergence_space, space_restriction=strain_space_restriction)
            self.strain_yield_parameters_test = fem.make_test(
                strain_yield_parameters_test_space, space_restriction=strain_space_restriction
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
            self.strain_yield_parameters_test.rebind(strain_yield_parameters_test_space, strain_space_restriction)

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
        has_critical_offset: bool,
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

        if has_critical_offset:
            self.critical_strain_offset = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)

        if has_compliant_bodies:
            self.collider_total_volumes = fem.borrow_temporary(temporary_store, shape=collider_count, dtype=float)

        if max_colors > 0:
            # Cell-based coloring sorts one entry per partition cell, and cells without
            # particles make that count exceed the particle-based strain node count.
            partition_cell_count = self._strain_space_restriction.space_partition.geo_partition.cell_count()
            color_block_capacity = max(strain_node_count, partition_cell_count)
            self.color_indices = fem.borrow_temporary(temporary_store, shape=(2, color_block_capacity), dtype=int)
            self.color_offsets = fem.borrow_temporary(temporary_store, shape=max_colors + 1, dtype=int)

        if self.contact_pic is not None:
            self.gimp_particle_contact_nodes = fem.borrow_temporary(
                temporary_store, shape=self.particle_cells.shape[0], dtype=int
            )

    def release_temporaries(self):
        """Release previously allocated temporaries to the store."""
        self.inv_mass_matrix.release()
        self.collider_velocity.release()
        self.collider_friction.release()
        self.collider_adhesion.release()
        self.collider_node_volume.release()
        self.strain_node_particle_volume.release()
        self.unilateral_strain_offset.release()

        if self.critical_strain_offset is not None:
            self.critical_strain_offset.release()
            self.critical_strain_offset = None

        if self.collider_total_volumes is not None:
            self.collider_total_volumes.release()

        if self.color_indices is not None:
            self.color_indices.release()
            self.color_offsets.release()

        if self.gimp_particle_contact_nodes is not None:
            self.gimp_particle_contact_nodes.release()
        if self.gimp_domain_corners is not None:
            for array in self.gimp_domain_corners:
                if array is not None:
                    array.release()


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
        self.gimp_stress = None
        self.gimp_warmstart_state = None
        # Grid collider impulses keyed by lattice node, for grids rebuilt in place
        self.impulse_keys = None
        self.impulse_nodes = None
        self.impulse_values = None

    def require_keyed_impulses(self, node_count: int, device: wp.Device):
        """Allocate sorted lattice keys and impulses for ``node_count`` collider nodes."""
        if self.impulse_values is None or self.impulse_values.shape[0] != node_count:
            self.impulse_keys = wp.full(2 * node_count, NODE_KEY_SENTINEL, dtype=wp.int64, device=device)
            self.impulse_nodes = wp.empty(2 * node_count, dtype=int, device=device)
            self.impulse_values = wp.zeros(node_count, dtype=wp.vec3, device=device)

    def require_gimp_stress_fields(self, particle_count: int, device: wp.Device):
        """Allocate persistent stress history for each particle's cell overlaps."""
        if self.gimp_stress is None:
            self.gimp_stress = (
                wp.zeros((particle_count, 8), dtype=wp.mat33, device=device),
                wp.zeros((particle_count, 8), dtype=float, device=device),
                wp.zeros((particle_count, 8), dtype=wp.vec3, device=device),
            )

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

    def save_collider_current_position(
        self,
        collider_body_q: wp.array[wp.transform] | None,
        body_world: wp.array[wp.int32] | None = None,
        world_mask: wp.array[wp.bool] | None = None,
        world_count: int | None = None,
    ):
        had_previous = self.body_q_prev is not None and (
            collider_body_q is None or self.body_q_prev.shape == collider_body_q.shape
        )
        self.require_collider_previous_position(collider_body_q)
        if collider_body_q is not None:
            if world_mask is None or body_world is None or not had_previous:
                self.body_q_prev.assign(collider_body_q)
            else:
                if world_count is None:
                    raise ValueError("world_count is required with world_mask.")
                wp.launch(
                    reset_mpm_collider_history,
                    dim=collider_body_q.shape[0],
                    inputs=[body_world, world_mask, world_count, collider_body_q, self.body_q_prev],
                    device=collider_body_q.device,
                )


class SolverImplicitMPM(SolverBase, CouplingInterface):
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

    Multi-world models retain the shared FEM topology by default. Set
    :attr:`Config.separate_worlds` to use an independent FEM environment for
    each world. In isolated mode, every MPM particle must belong to a local
    world and particles must be stored contiguously by world. World-local
    colliders affect only that world; global static colliders and global
    colliders backed by kinematic bodies affect every world, while global
    colliders backed by dynamic bodies are rejected.

    A sparse grid is rebuildable when :attr:`Config.max_active_cell_count` is
    positive, :attr:`Config.grid_padding` is zero, and the strain and collider
    bases support rebuilding. Every velocity basis supports rebuilding; the
    ``"B2"`` and ``"B3"`` bases reserve 64 velocity nodes per active cell,
    compared with 8 for ``"Q1"``. Cell and node capacities are totals across
    all FEM environments, and resolved capacities must satisfy
    ``upper <= lower <= leaf <= active``.

    Outer graph capture requires CUDA, an enabled memory pool, conditional
    graph support, ``enable_timers=False``, positive active-cell capacity, and
    either a fixed or rebuildable sparse grid. Construction creates persistent
    capture resources internally. Run an uncaptured warm-up step for rheology
    modes that build an inner graph lazily, and keep model topology, solver
    configuration, captured buffers, and step arguments fixed across replays.
    Keep :meth:`reset` outside capture and call
    :meth:`check_sparse_grid_rebuild_status` after replay to report
    sparse-grid capacity failures.

    [1] https://doi.org/10.1145/2897824.2925877

    Args:
        model: The model to simulate.
        config: Solver configuration. See :class:`SolverImplicitMPM.Config`.
        temporary_store: Optional Warp FEM temporary store for reusing scratch
            allocations across steps.
        verbose: If True, enable verbose solver output. If False, suppress details. If None, enable verbose output when
            ``wp.config.log_level`` is configured for debug logging.
        enable_timers: Enable per-section wall-clock timings. This must be
            ``False`` when :meth:`step` is recorded in an outer graph
            because section timers synchronize the device.

    Raises:
        ValueError: If an isolated multi-world model contains global MPM
            particles, particle world IDs outside ``[-1, model.world_count)``,
            or particles that are not stored contiguously by world.
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
        carry_residual_strain: Literal["none", "unconverged", "sampled"] = "none"
        """Residual strain carried into the elastic strain, beyond the strain consistent with the solved stress.

        ``"none"`` updates the elastic strain from the solved stress only.
        ``"unconverged"`` uses the kinematic strain increment minus the plastic strain
        increment at each strain-basis node, which adds the flow-rule residual left
        by an unconverged solve. ``"sampled"`` samples ``dt * grad(v)`` minus the
        plastic strain increment at each particle, which also keeps the strain that
        the strain basis cannot represent.

        With ``"none"``, soft materials relax when solves stop at ``max_iterations``.
        The residual modes keep that strain but can destabilize stiff materials at
        low iteration counts. Use ``"none"`` for stiff materials and ``"unconverged"`` for
        compliant elastic materials.

        .. experimental::

            The residual strain modes may change without prior notice.
        """
        solver: _RheologySolverName | Sequence[_RheologySolverName] = "auto"
        """Solver to use for the rheology solver. ``"auto"`` selects ``"gs"``
        for Q1 velocity basis and ``"gs-batched"`` for higher-order bases
        (B2, B3).  Accepted values: ``"auto"``, ``"gs"`` (or
        ``"gauss-seidel"``), ``"gs-soa"`` (or ``"gauss-seidel-soa"``),
        ``"gs-batched"`` (or ``"gauss-seidel-batched"``), ``"jacobi"``,
        ``"conjugate-gradient"`` (or ``"cg"``), ``"conjugate-residual"``
        (or ``"cr"``), ``"generalized-minimal-residual"`` (or ``"gmres"``).
        Pass an ordered sequence to warmstart solvers left-to-right, e.g.
        ``("conjugate-residual", "gauss-seidel")`` or
        ``("conjugate-gradient", "jacobi", "gauss-seidel")``."""
        warmstart_mode: Literal["none", "auto", "particles", "grid", "smoothed"] = "auto"
        """Warmstart mode to use for the rheology solver.

        ``"auto"`` uses particle-backed stress for rebuildable sparse grids
        and ``P1d``/``Q1d`` strain bases, and grid-backed stress otherwise.
        Grid-backed ``"grid"`` and ``"smoothed"`` modes are not supported for
        rebuildable sparse grids because their topology changes in place.
        """
        collider_velocity_mode: Literal["forward", "backward"] = "forward"
        """Collider velocity computation mode. ``'forward'`` uses the current velocity,
        ``'backward'`` uses the previous timestep position."""

        # grid
        voxel_size: float = 0.1
        """Size of the grid voxels."""
        grid_type: Literal["sparse", "dense", "fixed"] = "sparse"
        """Type of grid to use.

        A capacity-bounded ``"sparse"`` grid is rebuilt in place. Dense grids
        read dynamic bounds on the host.
        """
        grid_padding: int = 0
        """Number of empty cells to add around particles when allocating the grid."""
        max_active_cell_count: int = -1
        """Maximum number of active grid cells across all worlds.

        A positive value reserves persistent sparse-grid capacity and bounds
        active subsets of dense and fixed grids.
        ``-1`` retains per-step sparse-grid allocation and uses exact active
        counts elsewhere. Call :meth:`check_sparse_grid_rebuild_status` after
        graph replay to detect sparse-grid overflow.
        """
        max_leaf_node_count: int = -1
        """Maximum NanoVDB leaf-node count across all worlds.

        This independently bounds leaf topology for a rebuildable sparse grid.
        ``-1`` reserves one leaf per :attr:`max_active_cell_count`, which is the
        worst case for arbitrarily scattered active cells. Set an explicit
        value only when application-level spatial bounds guarantee a tighter
        limit. All node-capacity settings are validated at construction but
        used only for rebuildable sparse grids.
        """
        max_lower_node_count: int = -1
        """Maximum NanoVDB lower internal-node count across all worlds.

        ``-1`` estimates the initial packed topology and reserves 16 times its
        lower-node count, capped by the resolved leaf-node capacity. An
        explicit value budgets spatial-spread headroom independently from
        active cells. Only used for rebuildable sparse grids.
        """
        max_upper_node_count: int = -1
        """Maximum NanoVDB upper internal-node count across all worlds.

        ``-1`` estimates the initial packed topology and reserves 16 times its
        upper-node count, capped by the resolved lower-node capacity. Upper
        nodes cover regions 4096 voxels wide and are substantially larger than
        lower or leaf nodes, so applications with known spatial bounds can use
        this field to budget them explicitly. Only used for rebuildable sparse
        grids.
        """
        transfer_scheme: Literal["apic", "pic"] = "apic"
        """Transfer scheme to use for particle-grid transfers."""
        integration_scheme: Literal["pic", "gimp"] = "pic"
        """Integration scheme controlling shape-function support.

        GIMP integrates each particle's cube-shaped domain over its grid-cell
        overlaps. The domain edge is capped at one voxel; all support cells
        participate in transfers and particle stress warmstarts. Particle contacts
        sample velocity at particle centers with the velocity basis weights.
        """

        # material / background
        critical_fraction: float = 0.0
        """Packing fraction below which cohesion is disabled and recompression is allowed.

        Each particle compares its measured packing with the available space
        outside colliders, both from :attr:`critical_fraction_measurement`.
        Underfilled material may recompress under external forces without
        an attractive correction.
        """
        air_drag: float = 1.0
        """Numerical drag for the background air."""

        # experimental
        gs_colored_contacts: bool | None = None
        """Solve particle contacts inside the strain-cell Gauss-Seidel color loop.

        Requires discontinuous grid strain, particle contacts, and GS or GS-SOA as the
        only rheology solver; sequences such as ``("cr", "gs")`` use the existing contact
        solver. By default (``None``), enable this for supported configurations and use
        the existing contact solver otherwise. Set ``False`` to opt out; ``True`` raises
        :class:`ValueError` for unsupported configurations.
        Reuses the strain-cell coloring for the selected
        velocity basis. With GIMP, contacts remain at particle centers and use
        velocity-basis weights; material transfers still integrate particle-cell overlaps.
        Dynamic collider velocities are updated once after each full color sweep.

        .. experimental::

            Cell-local contact solver ordering may change.
        """
        collider_recovery_fraction: float = 0.0
        """Fraction of particle contact penetration corrected per step.

        Values in ``[0, 1]`` add an outward normal target velocity of
        ``fraction * penetration / dt``. Zero disables penetration recovery.
        Requires a particle collider basis (``"pic"`` or ``"picN"``).

        .. experimental::

            Collider recovery may change without prior notice.
        """
        collider_contact_gap: float = 0.0
        """Predictive contact activation distance [m].

        Positive values activate contacts before penetration for any
        supported collider basis. Separated collider samples may approach
        at ``gap / dt`` so they can reach the contact band during the step.
        Zero retains the default activation distance.

        .. experimental::

            Predictive contact gaps may change without prior notice.
        """
        volume_recovery_fraction: float = 0.0
        """Fraction of particle overpacking corrected per step.

        Values in ``[0, 1]`` add the nonpositive offset
        ``fraction * min(available / measured - 1, 0)`` to the divergence
        constraint, the volumetric strain that would bring the measured
        particle volume fraction back to the space left free by colliders,
        with fractions from :attr:`volume_recovery_measurement`. The strain
        quadrature samples the offsets like the elastic strain, so they vary
        continuously as particles move. The target filling fraction is one.
        Overfilled regions produce negative offsets, requesting expansion.
        Underfilled regions, such as free surfaces, receive no correction.
        Zero disables this feedback.

        .. experimental::

            Density-based divergence feedback may change without prior notice.
        """
        volume_recovery_measurement: Literal["grid", "particle"] = "grid"
        """Measurement of the packing fractions for :attr:`volume_recovery_fraction`.

        Each particle contributes its deformed volume, the reference volume
        times ``det(F_e)`` of its elastic deformation gradient. The measured
        fraction and the available fraction, the part of the same kernel
        outside colliders, use matching kernels, so packed material reads
        full against walls. ``"grid"`` deposits volumes into cells with
        quadratic B-splines and reads them back linearly at each particle,
        averaging over about two voxels. ``"particle"`` sums neighbor volumes
        with a cubic B-spline of one voxel radius around each particle, which
        keeps the band that reads as loose at free surfaces thin, at the cost
        of a neighbor search and a slight overestimate in disordered packings.

        .. experimental::

            Density measurements may change without prior notice.
        """
        critical_fraction_measurement: Literal["grid", "particle"] = "particle"
        """Measurement of the packing fractions for :attr:`critical_fraction`.

        Accepts the measurements of :attr:`volume_recovery_measurement`. The
        default ``"particle"`` keeps free surfaces from reading as loose over
        a thick band, which would let them recompress without friction.

        .. experimental::

            Density measurements may change without prior notice.
        """
        collider_normal_from_sdf_gradient: bool = False
        """Compute collider normals from the signed-distance gradient rather than the closest point.

        Grid collider bases differentiate the distance rasterized at their nodes. Particle
        collider bases (``"pic"`` and ``"picN"``) differentiate the collider distance
        rasterized at cell centers with quadratic B-spline weights, so normals turn smoothly
        around concave and convex collider edges, such as the seams between assembled boxes.
        """
        collider_basis: _MPMColliderBasisName = "S2"
        """Collider basis function. Defaults to ``"S2"``; pass ``"Q1"``
        to restore the previous trilinear collider basis. Common values are
        ``"Q1"`` (trilinear), ``"S2"`` (quadratic serendipity), or
        ``"pic"``, ``"pic8"``, ``"pic27"``
        (particle-based with optional max points per cell). Any ``"picN"``
        form with integer ``N`` is accepted. Particle bases require an explicit
        per-cell limit for graph capture (``"picN"``).

        Particle contacts activate within 0.25 voxels of the surface, measured
        from the particle radius. Activated contacts allow a normal approach
        speed up to the gap beyond that band divided by the timestep, so
        material stops in the band where grid contacts activate, and particles
        near walls keep sampling the volume around them instead of bunching
        against the surface.
        """
        strain_basis: _MPMStrainBasisName = "P0"
        """Strain basis function. Common values are ``"P0"``, ``"P1d"``,
        ``"Q1"``, ``"Q1d"``, or particle-based ``"pic"``, ``"pic8"``,
        ``"pic27"``. Any ``"picN"`` form with integer ``N`` is accepted.
        Particle bases require an explicit per-cell limit (``"picN"``) for graph capture.

        Discontinuous polynomial bases use nodal mass orthogonalization.
        P1d/Q1d recover the full-cell reference basis; Q1d uses Gauss--Legendre
        reference nodes. Rank-deficient cells drop null modes and orthogonalize
        the supported subspace. Each mode is scaled so the unit field has unit
        coefficients, keeping yield bounds in physical units in partially filled
        cells. Pressure-related parameters use positive basis weights, shear
        strength uses absolute weights, and viscosity uses squared weights.

        .. experimental::

            Nodal mass orthogonalization of discontinuous strain bases may change.
        """
        velocity_basis: _MPMVelocityBasisName = "Q1"
        """Velocity basis function. Common values are ``"Q1"``, ``"B2"``,
        or ``"B3"``."""

        separate_worlds: bool = False
        """Use independent FEM environments for each world in a multi-world model.

        The default ``False`` retains the legacy shared-grid behavior. Set to
        ``True`` to isolate grid mass, momentum, stress, and collider response
        by world. Isolated multi-world models require every MPM particle to
        belong to a local world and particles to be stored contiguously by
        world. See :ref:`implicit-mpm-worlds` for how this mode interprets
        world assignment and collider ownership.

        .. experimental::

            Isolated multi-world MPM configuration and behavior may change
            without prior notice.
        """

        @classmethod
        def create_from_usd(cls, scene_prim: Any) -> SolverImplicitMPM.Config:
            """Create a solver configuration from ``NewtonMPMSceneAPI``.

            Only authored USD values override :class:`Config` defaults, so the
            Python defaults remain the source of simulator-default values.
            Authored ``-inf`` values on dimensional attributes using the USD
            simulator-default convention retain those Python defaults. Length
            values authored in stage units are converted to meters. Following
            :meth:`ModelBuilder.add_usd`, unauthored stage unit metadata is
            interpreted as one meter and one kilogram per stage unit.

            Args:
                scene_prim: A ``UsdPhysics.Scene`` prim with
                    ``NewtonMPMSceneAPI`` applied, or its typed schema object.

            Returns:
                A validated MPM solver configuration.

            Raises:
                TypeError: If ``scene_prim`` is not a USD physics scene prim.
                ValueError: If the API is absent or an authored value is invalid.
            """
            try:
                from pxr import UsdGeom, UsdPhysics
            except ImportError as error:
                raise ImportError("Creating an MPM config from USD requires usd-core.") from error

            from ...usd import utils as usd  # noqa: PLC0415

            prim = scene_prim.GetPrim() if hasattr(scene_prim, "GetPrim") else scene_prim
            if not prim or not prim.IsValid() or not prim.IsA(UsdPhysics.Scene):
                raise TypeError("scene_prim must be a valid UsdPhysics.Scene prim.")
            path = str(prim.GetPath())
            if not usd.has_applied_api_schema(prim, "NewtonMPMSceneAPI"):
                raise ValueError(f"{path}: NewtonMPMSceneAPI is not applied.")

            config = cls()

            def authored(name: str):
                attr = prim.GetAttribute(name)
                if not attr or not attr.HasAuthoredValue():
                    return None
                value = attr.Get()
                if value is None:
                    raise ValueError(f"{path}: authored attribute {name!r} has no value.")
                return value

            def finite_float(name: str, value: Any, *, minimum: float | None = None) -> float:
                try:
                    result = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}: {name} must be a finite number, got {value!r}.") from error
                if not math.isfinite(result) or (minimum is not None and result < minimum):
                    qualifier = f" greater than or equal to {minimum}" if minimum is not None else ""
                    raise ValueError(f"{path}: {name} must be a finite number{qualifier}, got {value!r}.")
                return result

            def requests_simulator_default(name: str, value: Any) -> bool:
                try:
                    result = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}: {name} must be a number, got {value!r}.") from error
                return result == float("-inf")

            def integer(name: str, value: Any, *, allow_minus_one: bool = False) -> int:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"{path}: {name} must be an integer, got {value!r}.")
                try:
                    result = operator.index(value)
                except TypeError as error:
                    raise ValueError(f"{path}: {name} must be an integer, got {value!r}.") from error
                if result < 0 and not (allow_minus_one and result == -1):
                    suffix = " or -1" if allow_minus_one else ""
                    raise ValueError(f"{path}: {name} must be non-negative{suffix}, got {result}.")
                return result

            def token(name: str, value: Any, allowed: set[str]) -> str:
                result = str(value)
                if result not in allowed:
                    raise ValueError(f"{path}: {name} must be one of {sorted(allowed)}, got {result!r}.")
                return result

            value = authored("newton:maxSolverIterations")
            if value is not None:
                value = integer("newton:maxSolverIterations", value, allow_minus_one=True)
                if value == 0:
                    raise ValueError(f"{path}: newton:maxSolverIterations must be positive or -1, got 0.")
                if value != -1:
                    config.max_iterations = value

            value = authored("newton:mpm:tolerance")
            if value is not None:
                config.tolerance = finite_float("newton:mpm:tolerance", value, minimum=0.0)
                if config.tolerance == 0.0:
                    raise ValueError(f"{path}: newton:mpm:tolerance must be positive.")

            value = authored("newton:mpm:rheologySolvers")
            if value is not None:
                solvers = tuple(str(item) for item in value)
                solver_names = {
                    "auto": "auto",
                    "gauss-seidel": "gauss-seidel",
                    "gauss-seidel-soa": "gauss-seidel-soa",
                    "gauss-seidel-batched": "gauss-seidel-batched",
                    "jacobi": "jacobi",
                    "conjugate-gradient": "cg",
                    "conjugate-residual": "cr",
                    "generalized-minimal-residual": "gmres",
                }
                if not solvers or any(item not in solver_names for item in solvers):
                    raise ValueError(
                        f"{path}: newton:mpm:rheologySolvers must be a non-empty ordered array of supported tokens, "
                        f"got {solvers!r}."
                    )
                config.solver = tuple(solver_names[item] for item in solvers)

            token_fields = (
                ("newton:mpm:gridType", "grid_type", {"sparse", "dense", "fixed"}),
                ("newton:mpm:transferScheme", "transfer_scheme", {"apic", "pic"}),
                ("newton:mpm:integrationScheme", "integration_scheme", {"pic", "gimp"}),
            )
            for usd_name, field_name, allowed in token_fields:
                value = authored(usd_name)
                if value is not None:
                    setattr(config, field_name, token(usd_name, value, allowed))

            stage = prim.GetStage()
            linear_unit = (
                float(UsdGeom.GetStageMetersPerUnit(stage)) if UsdGeom.StageHasAuthoredMetersPerUnit(stage) else 1.0
            )
            if not math.isfinite(linear_unit) or linear_unit <= 0.0:
                raise ValueError(f"{path}: metersPerUnit must be finite and positive, got {linear_unit!r}.")
            mass_unit = (
                float(UsdPhysics.GetStageKilogramsPerUnit(stage))
                if UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage)
                else 1.0
            )
            if not math.isfinite(mass_unit) or mass_unit <= 0.0:
                raise ValueError(f"{path}: kilogramsPerUnit must be finite and positive, got {mass_unit!r}.")
            value = authored("newton:mpm:voxelSize")
            if value is not None and not requests_simulator_default("newton:mpm:voxelSize", value):
                config.voxel_size = finite_float("newton:mpm:voxelSize", value, minimum=0.0) * linear_unit
                if not math.isfinite(config.voxel_size) or config.voxel_size <= 0.0:
                    raise ValueError(f"{path}: newton:mpm:voxelSize must convert to a finite positive SI distance.")

            value = authored("newton:mpm:gridPadding")
            if value is not None:
                config.grid_padding = integer("newton:mpm:gridPadding", value)

            value = authored("newton:mpm:maxActiveCellCount")
            if value is not None:
                try:
                    config.max_active_cell_count = _validate_sparse_grid_node_capacity("max_active_cell_count", value)
                except ValueError as error:
                    raise ValueError(f"{path}: invalid newton:mpm:maxActiveCellCount: {error}") from error

            value = authored("newton:mpm:criticalFraction")
            if value is not None:
                config.critical_fraction = finite_float("newton:mpm:criticalFraction", value, minimum=0.0)
                if config.critical_fraction > 1.0:
                    raise ValueError(f"{path}: newton:mpm:criticalFraction must not exceed 1.0.")
            value = authored("newton:mpm:airDrag")
            if value is not None and not requests_simulator_default("newton:mpm:airDrag", value):
                authored_air_drag = finite_float("newton:mpm:airDrag", value, minimum=0.0)
                config.air_drag = authored_air_drag * mass_unit
                if not math.isfinite(config.air_drag) or (authored_air_drag != 0.0 and config.air_drag == 0.0):
                    raise ValueError(f"{path}: newton:mpm:airDrag must convert to a finite, representable SI value.")

            def read_basis(prefix: str, field_name: str, allowed_types: set[str]) -> None:
                """Compose the schema's basis fields into Newton's compact basis name."""
                type_name = f"newton:mpm:{prefix}BasisType"
                order_name = f"newton:mpm:{prefix}BasisOrder"
                discontinuous_name = f"newton:mpm:{prefix}DiscontinuousBasis"
                attrs = (
                    prim.GetAttribute(type_name),
                    prim.GetAttribute(order_name),
                    prim.GetAttribute(discontinuous_name),
                )
                if not any(attr and attr.HasAuthoredValue() for attr in attrs):
                    return

                basis_type = token(
                    type_name,
                    attrs[0].Get(),
                    allowed_types,
                )
                order = integer(order_name, attrs[1].Get())
                discontinuous = attrs[2].Get()
                if not isinstance(discontinuous, (bool, np.bool_)):
                    raise ValueError(f"{path}: {discontinuous_name} must be bool, got {discontinuous!r}.")

                if basis_type == "particle":
                    # The current runtime has no USD representation for its
                    # optional picN capacity suffix. Plain "pic" is the stable
                    # particle-basis spelling; order and continuity do not apply.
                    basis = "pic"
                else:
                    if basis_type in ("bspline", "serendipity") and not 1 <= order <= 3:
                        raise ValueError(
                            f"{path}: {type_name}={basis_type!r} supports {order_name} values from 1 through 3, "
                            f"got {order}."
                        )
                    prefix_by_type = {
                        "linear": "P",
                        "trilinear": "Q",
                        "bspline": "B",
                        "serendipity": "S",
                    }
                    basis = f"{prefix_by_type[basis_type]}{order}"
                    # Degree-zero spaces are inherently discontinuous, so the
                    # authored flag only changes positive-order basis names.
                    if bool(discontinuous) and order > 0:
                        basis += "d"
                    _parse_grid_basis_name(basis)
                setattr(config, field_name, basis)

            read_basis(
                "collider",
                "collider_basis",
                {"linear", "trilinear", "bspline", "serendipity", "particle"},
            )
            read_basis("strain", "strain_basis", {"linear", "trilinear", "particle"})

            velocity_type_name = "newton:mpm:velocityBasisType"
            velocity_order_name = "newton:mpm:velocityBasisOrder"
            velocity_attrs = (
                prim.GetAttribute(velocity_type_name),
                prim.GetAttribute(velocity_order_name),
            )
            if any(attr and attr.HasAuthoredValue() for attr in velocity_attrs):
                velocity_type = token(
                    velocity_type_name,
                    velocity_attrs[0].Get(),
                    {"trilinear", "bspline"},
                )
                velocity_order = integer(velocity_order_name, velocity_attrs[1].Get())
                if not 1 <= velocity_order <= 3:
                    raise ValueError(f"{path}: {velocity_order_name} must be from 1 through 3, got {velocity_order}.")
                velocity_prefix = "Q" if velocity_type == "trilinear" else "B"
                config.velocity_basis = f"{velocity_prefix}{velocity_order}"

            return config

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

        Attributes registered on Model (per-shape):
            - ``mpm:collider_adhesion``: Collider adhesion in Pa, used when the shape is a collider
              (see :meth:`setup_collider`)
            - ``mpm:collider_projection_threshold``: Collider projection threshold in m; a negative value (default)
              selects the solver default, which scales with the voxel size

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

        # Per-shape collider material
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="collider_adhesion",
                frequency=newton.Model.AttributeFrequency.SHAPE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="mpm",
            )
        )
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="collider_projection_threshold",
                frequency=newton.Model.AttributeFrequency.SHAPE,
                assignment=newton.Model.AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=-1.0,
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

    _rheology_acceleration = "nesterov"
    """Acceleration of the rheology iterations, ``"nesterov"`` or ``"none"``, see ``solve_rheology``."""

    def __init__(
        self,
        model: newton.Model,
        config: Config,
        *,
        temporary_store: fem.TemporaryStore | None = None,
        verbose: bool | None = None,
        enable_timers: bool = False,
    ):
        super().__init__(model)

        self._initial_particle_count = int(model.particle_count)
        self._initial_world_count = int(model.world_count)

        if self._initial_particle_count == 0:
            raise ValueError("SolverImplicitMPM requires at least one particle.")

        self._separate_worlds = bool(config.separate_worlds and self._initial_world_count > 1)
        self._environment_count = self._initial_world_count if self._separate_worlds else 1
        self._particle_environment = model.particle_world if self._separate_worlds else None
        self._particle_world_ranges = None

        if self._separate_worlds:
            particle_world = model.particle_world.numpy()
            if np.any(particle_world < -1) or np.any(particle_world >= model.world_count):
                raise ValueError(
                    "SolverImplicitMPM found invalid MPM particle world IDs; expected values in "
                    "[-1, model.world_count)."
                )
            if np.any(particle_world == -1):
                raise ValueError(
                    "SolverImplicitMPM cannot isolate a multi-world model containing global MPM particles; "
                    "replicate the particles into each world or set Config.separate_worlds=False for legacy "
                    "coupled behavior."
                )

            world_boundaries = np.searchsorted(particle_world, np.arange(model.world_count + 1))
            particle_world_ranges = tuple((int(begin), int(end)) for begin, end in pairwise(world_boundaries))
            ranges_cover_particles = world_boundaries[0] == 0 and world_boundaries[-1] == particle_world.shape[0]
            ranges_match_worlds = all(
                np.all(particle_world[begin:end] == world) for world, (begin, end) in enumerate(particle_world_ranges)
            )
            if not ranges_cover_particles or not ranges_match_worlds:
                raise ValueError(
                    "SolverImplicitMPM requires MPM particles to be stored contiguously by world when "
                    "Config.separate_worlds=True."
                )
            self._particle_world_ranges = particle_world_ranges

        self._mpm_model = ImplicitMPMModel(model, config)

        self.max_iterations = config.max_iterations
        self.tolerance = float(config.tolerance)
        if config.carry_residual_strain not in ("none", "unconverged", "sampled"):
            raise ValueError(f"Invalid carry_residual_strain: {config.carry_residual_strain}")
        self.carry_residual_strain = config.carry_residual_strain

        self.temporary_store = temporary_store
        self.verbose = verbose if verbose is not None else wp.config.log_level <= wp.LOG_DEBUG
        self.enable_timers = enable_timers

        self.volume_recovery_fraction = float(config.volume_recovery_fraction)
        if not math.isfinite(self.volume_recovery_fraction) or not 0.0 <= self.volume_recovery_fraction <= 1.0:
            raise ValueError("volume_recovery_fraction must be finite and in [0, 1]")
        for name in ("volume_recovery_measurement", "critical_fraction_measurement"):
            if getattr(config, name) not in ("grid", "particle"):
                raise ValueError(f"{name} must be 'grid' or 'particle', got {getattr(config, name)!r}")
        self.volume_recovery_measurement = config.volume_recovery_measurement
        self.critical_fraction_measurement = config.critical_fraction_measurement

        self.velocity_basis = "Q1"
        self.strain_basis = config.strain_basis
        self.velocity_basis = config.velocity_basis

        self.grid_padding = config.grid_padding
        self.grid_type = config.grid_type
        self.max_active_cell_count = config.max_active_cell_count
        self.max_leaf_node_count = _validate_sparse_grid_node_capacity(
            "max_leaf_node_count", config.max_leaf_node_count
        )
        self.max_lower_node_count = _validate_sparse_grid_node_capacity(
            "max_lower_node_count", config.max_lower_node_count
        )
        self.max_upper_node_count = _validate_sparse_grid_node_capacity(
            "max_upper_node_count", config.max_upper_node_count
        )
        strain_basis = config.strain_basis
        collider_basis = config.collider_basis
        strain_rebuild_safe = strain_basis[:3] == "pic" or strain_basis in ("P0", "P1d", "Q1d", "Q1")
        collider_rebuild_safe = collider_basis[:3] == "pic" or collider_basis in ("Q1", "S2", "S3")
        self._sparse_rebuildable = (
            self.grid_type == "sparse"
            and self.max_active_cell_count > 0
            and self.grid_padding == 0
            and strain_rebuild_safe
            and collider_rebuild_safe
        )
        self._grid_status = None
        self._grid_accumulated_status = None
        self._grid_point_mask = None
        self.solver = _resolve_solver_spec(config.solver, self.velocity_basis)
        self.coloring = any("gauss-seidel" in solver or "gs" in solver for solver in self.solver)
        self.apic = config.transfer_scheme == "apic"
        self.gimp = config.integration_scheme == "gimp"
        self.collider_normal_from_sdf_gradient = config.collider_normal_from_sdf_gradient
        self.collider_basis = config.collider_basis
        supports_colored_contacts = (
            not self.strain_basis.startswith("pic")
            and _parse_grid_basis_name(self.strain_basis)[2]
            and self.collider_basis.startswith("pic")
            and self.solver in (("gs",), ("gauss-seidel",), ("gs-soa",), ("gauss-seidel-soa",))
        )
        self.gs_colored_contacts = (
            supports_colored_contacts if config.gs_colored_contacts is None else bool(config.gs_colored_contacts)
        )
        if self.gs_colored_contacts and not supports_colored_contacts:
            raise ValueError("Colored contacts require discontinuous grid strain, particle contacts, and GS or GS-SOA")
        self.collider_recovery_fraction = float(config.collider_recovery_fraction)
        self.collider_contact_gap = float(config.collider_contact_gap)
        if not math.isfinite(self.collider_recovery_fraction) or not 0.0 <= self.collider_recovery_fraction <= 1.0:
            raise ValueError("collider_recovery_fraction must be finite and in [0, 1]")
        if not math.isfinite(self.collider_contact_gap) or self.collider_contact_gap < 0.0:
            raise ValueError("collider_contact_gap must be finite and nonnegative")
        if self.collider_recovery_fraction > 0.0 and not self.collider_basis.startswith("pic"):
            raise ValueError("Collider recovery requires a particle collider basis")

        if config.collider_velocity_mode not in ("forward", "backward"):
            raise ValueError(f"Invalid collider velocity mode: {config.collider_velocity_mode}")
        self.collider_velocity_mode = config.collider_velocity_mode

        warmstart_mode = config.warmstart_mode
        if warmstart_mode not in ("none", "auto", "particles", "grid", "smoothed"):
            raise ValueError(f"Invalid warmstart mode: {warmstart_mode}")
        if warmstart_mode == "auto":
            warmstart_mode = "particles" if self._sparse_rebuildable or self.strain_basis in ("P1d", "Q1d") else "grid"
        if self._sparse_rebuildable and warmstart_mode in ("grid", "smoothed"):
            raise ValueError(
                f"Config.warmstart_mode={config.warmstart_mode!r} is not supported with rebuildable sparse grids "
                "because their topology changes in place; use 'none', 'auto', or 'particles'."
            )
        self._stress_warmstart = "" if warmstart_mode == "none" else warmstart_mode

        self._use_cuda_graph = self.model.device.is_cuda and wp.is_conditional_graph_supported()

        self._timers_use_nvtx = False

        # Materialize graph-persistent topology and buffers before callers can capture step().
        self._last_step_data = LastStepData()
        with wp.ScopedDevice(model.device):
            self._scratchpad = ImplicitMPMScratchpad()
            pic = self._particles_to_cells(model.particle_q)
            self._rebuild_scratchpad(pic)
            self._require_velocity_space_fields(self._scratchpad, self._mpm_model.has_compliant_particles)
            self._require_collision_space_fields(self._scratchpad, self._last_step_data)
            self._require_strain_space_fields(self._scratchpad, self._last_step_data)
            if self._sparse_rebuildable and self.collider_basis[:3] != "pic":
                # Allocated before any capture, so graphs do not record the initialization.
                self._last_step_data.require_keyed_impulses(self._scratchpad.collider_node_count, model.device)

        self._velocity_nodes_per_strain_sample = (
            self._scratchpad.velocity_nodes_per_element
            if self.strain_basis != "Q1" and self.velocity_basis == "Q1"
            else -1
        )

    def setup_collider(
        self,
        *,
        collider_meshes: list[wp.Mesh] | None = None,
        collider_body_ids: list[int] | None = None,
        collider_margins: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_adhesion: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        collider_particle_ids: list[list[int] | wp.array[int] | None] | None = None,
        model: newton.Model | None = None,
        body_com: wp.array | None = None,
        body_mass: wp.array | None = None,
        body_inv_inertia: wp.array | None = None,
        body_q: wp.array | None = None,
        collider_world_ids: list[int] | None = None,
        collider_material_counts: list[int | None] | None = None,
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
            collider_particle_ids: For deformable mesh colliders, model particle ids corresponding to each mesh vertex.
            collider_material_counts: Number of materials to allocate per collider (default 1), to be assigned
                per face with :meth:`collider_face_materials`. Body colliders allocate
                ``max(shape count, count)`` materials, the first ones being the shape materials.
            model: The model to read collider properties from. Default to solver's model.
            body_com: For dynamic colliders, per-body center of mass.
            body_mass: For dynamic colliders, per-body effective mass. When omitted, bodies flagged
                with :attr:`newton.BodyFlags.KINEMATIC` have zero effective mass. An explicit array
                is authoritative.
            body_inv_inertia: For dynamic colliders, per-body inverse inertia.
            body_q: For dynamic colliders, per-body initial transform.
            collider_world_ids: Per-collider Newton world IDs. Custom meshes default to global
                (``-1``). In isolated mode, body-backed colliders infer their body's world and
                require any supplied ID to match. Shared-mode default discovery globalizes
                colliders. IDs must be ``-1`` or in ``[0, model.world_count)``.

                .. experimental::

                    Per-world MPM collider filtering may change without prior notice.

        Raises:
            ValueError: If collider-aligned inputs have different lengths, a world ID is invalid,
                an isolated external collider model has a different ``world_count``, or an isolated
                global body-backed collider is dynamic. Replicate a global dynamic collider per
                world, make it static or kinematic, or disable ``Config.separate_worlds``.
        """
        self._mpm_model.setup_collider(
            collider_meshes=collider_meshes,
            collider_body_ids=collider_body_ids,
            collider_thicknesses=collider_margins,
            collider_friction=collider_friction,
            collider_adhesion=collider_adhesion,
            collider_projection_threshold=collider_projection_threshold,
            collider_particle_ids=collider_particle_ids,
            model=model,
            body_com=body_com,
            body_mass=body_mass,
            body_inv_inertia=body_inv_inertia,
            body_q=body_q,
            collider_world_ids=collider_world_ids,
            collider_material_counts=collider_material_counts,
        )

        self._last_step_data.save_collider_current_position(self._mpm_model.collider_body_q)

    def collider_face_materials(self, collider_id: int) -> wp.array[int]:
        """Return a writable view of the per-face material ids of a collider.

        Entries are collider-local material indices, i.e. ``k`` selects the material at index ``k`` of the views
        returned by :meth:`collider_materials`. Values outside ``[0, count)`` are clamped to that range. Writing to
        the view takes effect on the next step and is safe inside captured graphs. Face order follows the collider
        mesh; for body colliders, faces of successive shapes are concatenated.

        Args:
            collider_id: Collider index, in the order the colliders were discovered or passed to
                :meth:`setup_collider`.

        Returns:
            View into the solver's face material table. It is invalidated by the next :meth:`setup_collider` call.
        """
        offset, count = self._mpm_model.collider_face_ranges[collider_id]
        return self._mpm_model.collider.face_material_index[offset : offset + count]

    def collider_materials(self, collider_id: int) -> dict[str, wp.array[float]]:
        """Return writable views of the material values of a collider.

        The views are indexed by the collider-local material index ``k``, as in :meth:`collider_face_materials`.
        Body colliders own one material per collision shape, in shape order, followed by any extra materials
        requested through ``collider_material_counts``. Mesh colliders own ``collider_material_counts[collider_id]``
        materials (1 by default).

        Args:
            collider_id: Collider index, see :meth:`collider_face_materials`.

        Returns:
            Dictionary with keys ``"friction"`` (Coulomb coefficient), ``"adhesion"`` [Pa],
            ``"projection_threshold"`` [m] and ``"margin"`` [m].
            The views are invalidated by the next :meth:`setup_collider` call.

        Warning:
            The collider's query range is derived at setup from the largest margin among its materials.
            Setting a margin above that construction-time maximum can lead to unexpected failures,
            such as missed contacts; call :meth:`setup_collider` again with larger ``collider_margins`` instead.

            The contact face is selected by geometric distance, so margins that differ between nearby
            faces apply to the nearest face only; a face with a larger margin can be missed when a
            closer face has a smaller one.
        """
        first, count = self._mpm_model.collider_material_ranges[collider_id]
        collider = self._mpm_model.collider
        return {
            "friction": collider.material_friction[first : first + count],
            "adhesion": collider.material_adhesion[first : first + count],
            "projection_threshold": collider.material_projection_threshold[first : first + count],
            "margin": collider.material_thickness[first : first + count],
        }

    @property
    def voxel_size(self) -> float:
        """Grid voxel size used by the solver."""
        return self._mpm_model.voxel_size

    def check_sparse_grid_rebuild_status(self) -> None:
        """Raise if a rebuildable sparse grid exceeded its reserved capacity.

        Rebuildable sparse-grid failures accumulate until a valid
        :meth:`reset`. Call this method outside graph capture after replay; the
        check synchronizes the solver device. It is a no-op when the solver has
        no asynchronous sparse-grid status buffer.

        Raises:
            RuntimeError: If rebuild status is inspected during graph capture,
                or if a rebuild reported a capacity or topology failure.
        """
        if self._grid_status is None:
            return
        if self.model.device.is_capturing:
            raise RuntimeError("Cannot inspect sparse grid rebuild status during graph capture")

        status = int(self._grid_status.numpy()[0])
        if self._grid_accumulated_status is not None:
            status |= int(self._grid_accumulated_status.numpy()[0])
        if status != wp.Volume.REBUILD_SUCCESS:
            raise _sparse_grid_rebuild_error(status)

    def _clear_sparse_grid_rebuild_status(self) -> None:
        """Clear the latest and accumulated sparse-grid rebuild status.

        Call this outside graph capture after handling a reported rebuild
        failure. A subsequent :meth:`check_sparse_grid_rebuild_status` then
        reports only failures produced after this boundary.

        Raises:
            RuntimeError: If called while the solver device is capturing a graph.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("Cannot clear sparse grid rebuild status during graph capture")
        if self._grid_status is not None:
            self._grid_status.zero_()
        if self._grid_accumulated_status is not None:
            self._grid_accumulated_status.zero_()

    @staticmethod
    def _validate_reset_array(
        array: wp.array | None,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype,
        device: wp.Device,
    ) -> None:
        if not isinstance(array, wp.array):
            raise ValueError(f"state.{name} must be a Warp array.")
        if array.shape != shape:
            raise ValueError(f"state.{name} has shape {array.shape}, expected {shape}.")
        if array.dtype != dtype:
            raise TypeError(f"state.{name} has dtype {array.dtype}, expected {dtype}.")
        if array.device != device:
            raise ValueError(f"state.{name} is on device {array.device}, expected {device}.")

    def _validate_reset_inputs(self, state: newton.State) -> None:
        if state is None:
            raise ValueError("'state' argument is required.")

        particle_count = int(self.model.particle_count)
        if particle_count != self._initial_particle_count:
            raise RuntimeError(
                "SolverImplicitMPM model.particle_count changed after construction: "
                f"expected {self._initial_particle_count}, got {particle_count}."
            )
        world_count = int(self.model.world_count)
        if world_count != self._initial_world_count:
            raise RuntimeError(
                "SolverImplicitMPM model.world_count changed after construction: "
                f"expected {self._initial_world_count}, got {world_count}."
            )

        device = self.model.device
        particle_shape = (self._initial_particle_count,)
        self._validate_reset_array(
            state.particle_q,
            name="particle_q",
            shape=particle_shape,
            dtype=wp.vec3,
            device=device,
        )
        self._validate_reset_array(
            state.particle_qd,
            name="particle_qd",
            shape=particle_shape,
            dtype=wp.vec3,
            device=device,
        )
        if not hasattr(state, "mpm"):
            raise ValueError("state is missing the 'mpm' custom-attribute namespace.")
        for name, dtype in (
            ("particle_elastic_strain", wp.mat33),
            ("particle_transform", wp.mat33),
            ("particle_qd_grad", wp.mat33),
            ("particle_stress", wp.mat33),
            ("particle_Jp", wp.float32),
        ):
            self._validate_reset_array(
                getattr(state.mpm, name, None),
                name=f"mpm.{name}",
                shape=particle_shape,
                dtype=dtype,
                device=device,
            )

        if self.model.body_count > 0:
            self._validate_reset_array(
                state.body_q,
                name="body_q",
                shape=(self.model.body_count,),
                dtype=wp.transform,
                device=device,
            )

    def _reset_grid_warmstart_partition(self, name, field, scratch_field) -> fem.SpacePartition:
        scratch_partition = scratch_field.space_partition
        if field.space.topology == scratch_partition.space_topology:
            partition = scratch_partition
        else:
            partition = fem.make_space_partition(
                space_topology=field.space.topology,
                geometry_partition=scratch_partition.geo_partition,
                with_halo=False,
                max_node_count=field.dof_values.shape[0],
                environment_first=True,
                device=self.model.device,
                temporary_store=self.temporary_store,
            )

        environment_offsets = getattr(partition, "env_offsets", None)
        expected_shape = (self._initial_world_count + 1,)
        if environment_offsets is None or environment_offsets.shape != expected_shape:
            raise RuntimeError(
                f"Masked reset cannot selectively clear last-step {name}: its grid partition does not expose "
                f"environment offsets with shape {expected_shape}."
            )
        return partition

    def _validate_reset_warmstart_fields(
        self, world_mask: wp.array[wp.bool] | None
    ) -> tuple[fem.SpacePartition | None, fem.SpacePartition | None]:
        reset_partitions = []
        for name, field, scratch_field in (
            ("ws_impulse_field", self._last_step_data.ws_impulse_field, self._scratchpad.impulse_field),
            ("ws_stress_field", self._last_step_data.ws_stress_field, self._scratchpad.stress_field),
        ):
            if field is None:
                reset_partitions.append(None)
                continue

            expected_shape = (field.space_partition.node_count(),)
            if field.dof_values.shape != expected_shape:
                raise ValueError(
                    f"last-step {name}.dof_values has shape {field.dof_values.shape}, expected {expected_shape}."
                )

            if world_mask is None or isinstance(field.space.basis, fem.PointBasisSpace):
                reset_partitions.append(None)
                continue
            if self._initial_world_count > 1 and not self._separate_worlds:
                raise RuntimeError(
                    "Masked reset cannot selectively clear grid-backed warm starts when "
                    "Config.separate_worlds=False for a multi-world model; set separate_worlds=True or reset all worlds."
                )
            reset_partitions.append(self._reset_grid_warmstart_partition(name, field, scratch_field))

        return tuple(reset_partitions)

    def _clear_reset_warmstarts(
        self,
        world_mask: wp.array[wp.bool] | None,
        reset_partitions: tuple[fem.SpacePartition | None, fem.SpacePartition | None],
    ) -> None:
        last_step_data = self._last_step_data
        if last_step_data.impulse_values is not None:
            if world_mask is None:
                last_step_data.impulse_keys.fill_(NODE_KEY_SENTINEL)
                last_step_data.impulse_values.zero_()
            else:
                wp.launch(
                    reset_values_by_lattice_key,
                    dim=last_step_data.impulse_values.shape[0],
                    inputs=[
                        world_mask,
                        self._initial_world_count,
                        last_step_data.impulse_keys,
                        last_step_data.impulse_values,
                    ],
                    device=self.model.device,
                )
        if self._last_step_data.gimp_stress is not None:
            valid = self._last_step_data.gimp_stress[1]
            if world_mask is None:
                valid.zero_()
            else:
                wp.launch(
                    reset_mpm_point_warmstart,
                    dim=self.model.particle_count,
                    inputs=[
                        self.model.particle_world,
                        world_mask,
                        self._initial_world_count,
                        valid.view(wp.types.vector(length=8, dtype=float)),
                    ],
                    device=self.model.device,
                )
        for field, partition in zip(
            (self._last_step_data.ws_impulse_field, self._last_step_data.ws_stress_field),
            reset_partitions,
            strict=True,
        ):
            if field is None:
                continue
            if world_mask is None:
                field.dof_values.zero_()
            elif self._initial_particle_count > 0 and isinstance(field.space.basis, fem.PointBasisSpace):
                point_values = field.dof_values
                wp.launch(
                    reset_mpm_point_warmstart,
                    dim=self._initial_particle_count,
                    inputs=[self.model.particle_world, world_mask, self._initial_world_count, point_values],
                    device=self.model.device,
                )
            elif partition is not None and partition.node_count() > 0:
                wp.launch(
                    reset_mpm_grid_warmstart,
                    dim=partition.node_count(),
                    inputs=[
                        world_mask,
                        self._initial_world_count,
                        partition.env_offsets,
                        partition.space_node_indices(),
                        field.dof_values,
                    ],
                    device=self.model.device,
                )

    @override
    def reset(
        self,
        state: newton.State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset implicit MPM history for all or selected worlds.

        Particle history is reset when ``flags`` includes either
        :attr:`~newton.StateFlags.PARTICLE_Q` or
        :attr:`~newton.StateFlags.PARTICLE_QD`. If ``flags`` is ``None``, all
        particle history is reset. Particle- and grid-backed warm starts are
        cleared for selected worlds. Grid-backed warm starts cannot be
        selectively cleared on a shared multi-world grid; use
        :attr:`Config.separate_worlds` or a full reset. A full reset clears
        every warm-start field. Sparse-grid rebuild status is always cleared at
        a valid reset boundary, and the previous-collider-pose cache is
        refreshed from ``state``. When present, the final mask entry selects
        global particle-backed history and collider poses whose world index is
        ``-1``.

        Args:
            state: Simulation state whose MPM history is modified in place.
            world_mask: Optional one-dimensional Warp boolean mask on the
                model device with shape ``(model.world_count + 1,)``. The final
                entry selects global objects whose world index is ``-1``. If
                ``None``, reset all worlds and global objects.

                .. experimental::

                    Selective per-world MPM reset behavior may change without prior notice.
            flags: Optional state bitmask. If ``None``, reset all particle
                history.
        """
        self._validate_reset_inputs(state)
        world_mask = self._normalize_reset_world_mask(world_mask)

        reset_partitions = self._validate_reset_warmstart_fields(world_mask)
        state_flags = int(StateFlags.ALL if flags is None else flags)
        reset_particle_history = bool(state_flags & int(StateFlags.PARTICLE))

        # Clearing first ensures a capture-time rejection cannot leave state
        # partially reset.
        if world_mask is None:
            self._clear_sparse_grid_rebuild_status()
        elif self._grid_status is not None:
            wp.launch(
                _clear_sparse_grid_rebuild_status_masked,
                dim=1,
                inputs=[world_mask, self._grid_status, self._grid_accumulated_status],
                device=self.model.device,
            )

        with wp.ScopedDevice(self.model.device):
            self._clear_reset_warmstarts(world_mask, reset_partitions)

            if reset_particle_history and self.model.particle_count > 0:
                if world_mask is None:
                    identity = wp.mat33(np.eye(3))
                    state.mpm.particle_elastic_strain.fill_(identity)
                    state.mpm.particle_transform.fill_(identity)
                    state.mpm.particle_qd_grad.zero_()
                    state.mpm.particle_stress.zero_()
                    state.mpm.particle_Jp.assign(self.model.mpm.particle_Jp)
                else:
                    wp.launch(
                        reset_mpm_particle_history,
                        dim=self.model.particle_count,
                        inputs=[
                            self.model.particle_world,
                            world_mask,
                            self._initial_world_count,
                            self.model.mpm.particle_Jp,
                            state.mpm.particle_elastic_strain,
                            state.mpm.particle_transform,
                            state.mpm.particle_qd_grad,
                            state.mpm.particle_stress,
                            state.mpm.particle_Jp,
                        ],
                        device=self.model.device,
                    )

            self._last_step_data.save_collider_current_position(
                state.body_q,
                body_world=self.model.body_world,
                world_mask=world_mask,
                world_count=self._initial_world_count,
            )
            self._last_step_data.gimp_warmstart_state = state

    @override
    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance the simulation by one time step.

        Transfers particle data to the grid, solves the implicit rheology
        system, and transfers the result back to update particle positions,
        velocities, and stress.

        Args:
            state_in: Input state at the start of the step. If
                ``state_in.particle_f`` is allocated, it is applied as an
                external force [N] on each active, dynamic particle, in
                addition to gravity.
            state_out: Output state written with updated particle data.
                May be the same object as ``state_in`` for in-place stepping.
            control: Control input (unused; material parameters come from the model).
            contacts: Contact information (unused; collisions are handled internally).
            dt: Time step duration [s].
            observables: Optional solver observable arrays allocated by :meth:`observables`.
                This solver declares no supported observables, so only an empty
                container is accepted.
        """
        self.validate_observables(observables, contacts)
        model = self.model

        with wp.ScopedDevice(model.device):
            last_step_data = self._last_step_data
            # Overlap history belongs to the state it was produced for, or the
            # last reset state; fresh history is already invalid.
            if (
                last_step_data.gimp_stress is not None
                and last_step_data.gimp_warmstart_state is not None
                and last_step_data.gimp_warmstart_state is not state_in
            ):
                last_step_data.gimp_stress[1].zero_()
            pic = self._particles_to_cells(state_in.particle_q)
            scratch = self._rebuild_scratchpad(pic)
            self._step_impl(state_in, state_out, dt, pic, scratch)
            scratch.release_temporaries()
            if last_step_data.gimp_stress is not None:
                last_step_data.gimp_warmstart_state = state_out

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        if flags & ModelFlags.MODEL_PROPERTIES:
            self._mpm_model.notify_particle_material_changed()

    @override
    def coupling_eval_gravity_acceleration(
        self,
        out_body_acceleration: wp.array[wp.vec3] | None,
        out_particle_acceleration: wp.array[wp.vec3] | None,
    ) -> None:
        """Evaluate gravity acceleration applied internally by the MPM solver."""
        if out_body_acceleration is not None:
            out_body_acceleration.zero_()
        if out_particle_acceleration is not None:
            super().coupling_eval_gravity_acceleration(None, out_particle_acceleration)

    def coupling_notify_input_state_update(
        self,
        state: newton.State,
        flags: StateFlags | int,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """Synchronize deformable collider meshes after particle input-state updates."""
        del dt
        flags = int(flags)
        update_points = bool(flags & StateFlags.PARTICLE_Q)
        update_velocities = bool(flags & StateFlags.PARTICLE_QD)
        if not (update_points or update_velocities) or not self._mpm_model.deformable_collider_vertex_ranges:
            return

        sync_points = update_points and state.particle_q is not None
        sync_velocities = update_velocities and state.particle_qd is not None
        if not (sync_points or sync_velocities):
            return

        # On iteration restart the source state is the same as at the start of
        # the outer step, so the collider mesh and its BVH are still valid from
        # the first call this step — skip the resync and refit.
        if iteration_restart:
            return

        for collider_id, vertex_start, vertex_end in self._mpm_model.deformable_collider_vertex_ranges:
            vertex_count = vertex_end - vertex_start
            if vertex_count <= 0:
                continue

            mesh = self._mpm_model._collider_meshes[collider_id]
            if sync_points:
                wp.launch(
                    _sync_mpm_proxy_particle_points_kernel,
                    dim=vertex_count,
                    inputs=[
                        vertex_start,
                        state.particle_q,
                        self._mpm_model.collider.collider_particle_ids,
                        mesh.points,
                    ],
                    device=self.model.device,
                )
                mesh.refit()
            if sync_velocities:
                wp.launch(
                    _sync_mpm_proxy_particle_velocities_kernel,
                    dim=vertex_count,
                    inputs=[
                        vertex_start,
                        state.particle_qd,
                        self._mpm_model.collider.collider_particle_ids,
                        mesh.velocities,
                    ],
                    device=self.model.device,
                )

    def coupling_rewind_proxy_body(
        self,
        body_local_to_proxy_global: wp.array[int],
        state: newton.State,
        coupling_forces: wp.array[wp.spatial_vector],
        body_gravity_acceleration: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Remove lagged velocity-level proxy wrenches from collider velocities."""
        del body_gravity_acceleration
        if state.body_q is None or state.body_qd is None or body_local_to_proxy_global.shape[0] == 0:
            return

        wp.launch(
            _rewind_mpm_proxy_bodies_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                body_local_to_proxy_global,
                coupling_forces,
                state.body_q,
                self.model.body_inv_inertia,
                self.model.body_inv_mass,
                state.body_qd,
            ],
            device=self.model.device,
        )

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector],
        state: newton.State,
        state_out: newton.State,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Convert MPM collider grid impulses to proxy-body wrenches."""
        del body_qd_before, state_out, contacts
        if dt <= 0.0:
            raise ValueError("MPM proxy wrench harvesting requires a positive dt")
        out_body_f.zero_()

        impulses, positions, collider_ids = self.collect_collider_impulses(state)
        if collider_ids.shape[0] == 0:
            return
        body_q = state.body_q if state.body_q is not None else self.model.body_q

        wp.launch(
            _harvest_mpm_proxy_wrenches_kernel,
            dim=collider_ids.shape[0],
            inputs=[
                float(dt),
                collider_ids,
                impulses,
                positions,
                self.collider_body_index,
                body_local_to_proxy_global,
                int(newton.BodyFlags.PROXY),
                self.model.body_flags,
                self.model.body_com,
                body_q,
                out_body_f,
            ],
            device=self.model.device,
        )

    def coupling_rewind_proxy_particle(
        self,
        particle_local_to_proxy_global: wp.array[int],
        state: newton.State,
        coupling_forces: wp.array[wp.vec3],
        particle_gravity_acceleration: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Remove lagged velocity-level proxy forces from proxy particle velocities."""
        if state.particle_qd is None or particle_local_to_proxy_global.shape[0] == 0:
            return

        wp.launch(
            _rewind_mpm_proxy_particles_kernel,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                particle_local_to_proxy_global,
                int(newton.ParticleFlags.PROXY),
                int(newton.ParticleFlags.ACTIVE),
                self.model.particle_flags,
                self._mpm_model.particle_flags,
                particle_gravity_acceleration,
                state.particle_f,
                coupling_forces,
                self.model.particle_inv_mass,
                state.particle_qd,
            ],
            device=self.model.device,
        )
        if (
            not hasattr(self, "_proxy_particle_qd_before")
            or self._proxy_particle_qd_before.shape != state.particle_qd.shape
        ):
            self._proxy_particle_qd_before = wp.empty_like(state.particle_qd)
        wp.copy(self._proxy_particle_qd_before, state.particle_qd)

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        particle_qd_before: wp.array[wp.vec3],
        state: newton.State,
        state_out: newton.State,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Convert MPM proxy momentum changes and collider impulses to forces."""
        if dt <= 0.0:
            raise ValueError("MPM proxy particle-force harvesting requires a positive dt")
        if particle_local_to_proxy_global.shape[0] == 0:
            return

        super().coupling_harvest_proxy_particle_forces(
            particle_local_to_proxy_global,
            out_particle_f,
            particle_qd_before=particle_qd_before,
            state=state,
            state_out=state_out,
            contacts=contacts,
            dt=dt,
        )

        if not self._mpm_model.deformable_collider_vertex_ranges:
            return

        wp.launch(
            _clear_inactive_mpm_proxy_particle_forces_kernel,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[
                particle_local_to_proxy_global,
                self._mpm_model.particle_flags,
                int(newton.ParticleFlags.PROXY),
                int(newton.ParticleFlags.ACTIVE),
                out_particle_f,
            ],
            device=self.model.device,
        )

        impulses, positions, collider_ids = self.collect_collider_impulses(state)
        if collider_ids.shape[0] == 0:
            return

        wp.launch(
            _harvest_mpm_proxy_particle_forces_kernel,
            dim=collider_ids.shape[0],
            inputs=[
                float(dt),
                collider_ids,
                impulses,
                positions,
                self._mpm_model.collider,
                particle_local_to_proxy_global,
                int(newton.ParticleFlags.PROXY),
                self._mpm_model.particle_flags,
                out_particle_f,
            ],
            device=self.model.device,
        )

    def collect_collider_impulses(self, state: newton.State | None) -> tuple[wp.array, wp.array, wp.array]:
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

        collider = self._mpm_model.collider
        prev_gap = collider.query_max_dist
        # Particle contacts measure distances from the particle surface, one radius closer than its center
        collider.query_max_dist = (gap if gap is not None else prev_gap) + self._contact_radius_reach()

        self._last_step_data.require_collider_previous_position(state_in.body_q)
        wp.launch(
            project_outside_collider,
            dim=state_in.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.mpm.particle_qd_grad,
                self._mpm_model.particle_flags,
                self.model.particle_mass,
                self._particle_environment,
                self._mpm_model.collider,
                state_in.body_q,
                state_in.body_qd if self.collider_velocity_mode == "forward" else None,
                self._last_step_data.body_q_prev if self.collider_velocity_mode == "backward" else None,
                dt,
                # Particle contacts keep particles one radius from collider surfaces
                self._mpm_model.particle_radius if self.collider_basis[:3] == "pic" else None,
            ],
            outputs=[
                state_out.particle_q,
                state_out.particle_qd,
                state_out.mpm.particle_qd_grad,
            ],
            device=state_in.particle_q.device,
        )

        collider.query_max_dist = prev_gap

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
        robustness. Inactive particles keep their previous frame.
        """

        wp.launch(
            update_particle_frames,
            dim=state.particle_count,
            inputs=[
                dt,
                min_stretch,
                max_stretch,
                self._mpm_model.particle_flags,
                state.mpm.particle_qd_grad,
                state_prev.mpm.particle_transform,
                state.mpm.particle_transform,
            ],
            device=state.mpm.particle_qd_grad.device,
        )

    def sample_render_grains(self, state: newton.State, grains_per_particle: int) -> wp.array2d[wp.vec3]:
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
        grains: wp.array2d[wp.vec3],
        dt: float,
    ) -> None:
        """Advect grain samples with the grid velocity and keep them inside the deformed particle.

        Args:
            state_prev: Previous state (t_n).
            state: Current state (t_{n+1}).
            grains: 2D array of grain positions per particle to be updated in place. See ``sample_render_grains``.
            dt: Time step duration.
        """

        with wp.ScopedDevice(grains.device):
            return update_render_grains(
                state_prev,
                state,
                grains,
                self._mpm_model.particle_radius,
                dt,
                particle_environment=self._particle_environment,
                temporary_store=self.temporary_store,
            )

    def create_particle_surface(
        self,
        voxel_size: float | None = None,
        *,
        max_grid_cells: int | None = None,
        **kwargs,
    ) -> ParticleSurface:
        """Create a reusable particle surface extraction context.

        Args:
            voxel_size: Voxel size for the density grid [m].
                Defaults to ``0.45 * solver_voxel_size``.
            max_grid_cells: Maximum active sparse-grid cell count across all
                worlds. When set, extraction uses graph-capturable preallocated
                buffers. When ``None``, it uses tight sparse allocations.
            **kwargs: Forwarded to :class:`newton.geometry.ParticleSurface`.

        Returns:
            A :class:`newton.geometry.ParticleSurface` context for use with
            :meth:`extract_particle_surface`.
        """
        if voxel_size is None:
            voxel_size = self._mpm_model.voxel_size * 0.45
        world_count = max(self.model.world_count, 1)
        if "world_count" in kwargs and kwargs["world_count"] != world_count:
            raise ValueError(f"world_count must match the model world count ({world_count})")
        kwargs["world_count"] = world_count
        return ParticleSurface(
            voxel_size=voxel_size,
            max_grid_cells=max_grid_cells,
            device=self.model.device,
            **kwargs,
        )

    def extract_particle_surface(
        self,
        state: newton.State,
        surface: ParticleSurface,
        *,
        compute_normals: bool = True,
        extrapolate_into_colliders: bool = False,
        collider_extrapolation_depth: float | None = None,
        collider_extrapolation_onset: float = 0.0,
        collider_extrapolation_redistance_iterations: int = 1,
        particle_flags: wp.array[wp.int32] | None = None,
    ) -> ParticleSurface.ExtractionMesh:
        """Extract a triangle mesh from the current particle state.

        Args:
            state: Current simulation state.
            surface: Reusable extraction context from
                :meth:`create_particle_surface`.
            compute_normals: Whether to compute per-vertex normals.
            extrapolate_into_colliders: Mirror-extrapolate the particle SDF
                into collider interiors before meshing.  Requires
                a surface created with ``field_mode="sdf"``.
            collider_extrapolation_depth: Maximum distance [m] to extrapolate
                into colliders. Defaults to the smaller of
                four surface voxels and the allocated topology halo.
            collider_extrapolation_onset: Signed collider distance [m] where
                extrapolation starts.  ``0`` starts at the collider surface.
            collider_extrapolation_redistance_iterations: Number of
                redistancing iterations to apply after collider extrapolation.
                Set to 0 to disable redistancing.
            particle_flags: Optional per-particle flags selecting the active
                particles to surface.  Defaults to the model particle flags.

        Returns:
            Mesh buffers and device-resident logical counts.
        """
        if particle_flags is None:
            particle_flags = self._mpm_model.particle_flags
        if extrapolate_into_colliders and surface.field_mode != "sdf":
            raise ValueError("Collider extrapolation requires ParticleSurface(field_mode='sdf')")

        if not extrapolate_into_colliders:
            return surface.extract(
                state.particle_q,
                radii=self._mpm_model.particle_radius,
                compute_normals=compute_normals,
                particle_flags=particle_flags,
                particle_world=self.model.particle_world if surface.world_count > 1 else None,
            )

        sparse_field = surface.update_field(
            state.particle_q,
            radii=self._mpm_model.particle_radius,
            particle_flags=particle_flags,
            particle_world=self.model.particle_world if surface.world_count > 1 else None,
        )
        if sparse_field is None:
            return surface.resurface(compute_normals=compute_normals)

        return extrapolate_surface_sdf_into_colliders(
            surface,
            self._mpm_model.collider,
            state.body_q,
            max_depth=collider_extrapolation_depth,
            onset=collider_extrapolation_onset,
            redistance_iterations=collider_extrapolation_redistance_iterations,
            compute_normals=compute_normals,
        )

    def _allocate_grid(
        self,
        positions: wp.array,
        particle_flags: wp.array,
        voxel_size: float,
        temporary_store: fem.TemporaryStore,
        padding_voxels: int = 0,
        *,
        point_environment: wp.array | None = None,
        particle_range_scale: int = 1,
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
        if point_environment is None:
            point_environment = self._particle_environment
        with self._timer("Allocate grid"):
            if self.grid_type == "sparse":
                if self._separate_worlds:
                    if self._sparse_rebuildable:
                        if self._grid_status is None:
                            self._grid_status = wp.zeros(1, dtype=wp.uint32, device=positions.device)
                            self._grid_accumulated_status = wp.zeros(1, dtype=wp.uint32, device=positions.device)
                        point_mask = self._update_grid_point_mask(positions, particle_flags)
                        guard_cells = 3
                        capacity_kwargs = _rebuild_capacity(
                            positions,
                            voxel_size,
                            16.0,
                            self.max_active_cell_count,
                            point_mask=point_mask,
                            point_environment=point_environment,
                            environment_count=self._environment_count,
                            guard_cells=guard_cells,
                            temporary_store=temporary_store,
                            max_leaf_node_count=self.max_leaf_node_count,
                            max_lower_node_count=self.max_lower_node_count,
                            max_upper_node_count=self.max_upper_node_count,
                        )
                        grid = fem.Nanogrid.from_environment_voxels(
                            positions,
                            point_environment,
                            self._environment_count,
                            point_mask=point_mask,
                            voxel_size=voxel_size,
                            temporary_store=temporary_store,
                            device=positions.device,
                            rebuildable=True,
                            status=self._grid_status,
                            **capacity_kwargs,
                        )
                        self.check_sparse_grid_rebuild_status()
                    else:
                        cell_ijks = [
                            voxel_coordinates(
                                positions[begin * particle_range_scale : end * particle_range_scale],
                                voxel_size,
                                padding_voxels=padding_voxels,
                            )
                            if begin != end
                            else wp.empty(0, dtype=wp.vec3i, device=positions.device)
                            for begin, end in self._particle_world_ranges
                        ]
                        cell_count = sum(cell_ijk.shape[0] for cell_ijk in cell_ijks)
                        cell_ijk = wp.empty(cell_count, dtype=wp.vec3i, device=positions.device)
                        cell_environment = wp.empty(cell_count, dtype=wp.int32, device=positions.device)
                        cell_offset = 0
                        for environment, environment_cell_ijk in enumerate(cell_ijks):
                            environment_cell_count = environment_cell_ijk.shape[0]
                            if environment_cell_count:
                                wp.copy(
                                    cell_ijk,
                                    environment_cell_ijk,
                                    dest_offset=cell_offset,
                                    count=environment_cell_count,
                                )
                                cell_environment[cell_offset : cell_offset + environment_cell_count].fill_(environment)
                            cell_offset += environment_cell_count
                        grid = fem.Nanogrid.from_environment_voxels(
                            cell_ijk,
                            cell_environment,
                            self._environment_count,
                            voxel_size=voxel_size,
                            temporary_store=temporary_store,
                            device=positions.device,
                        )
                else:
                    point_mask = None
                    if self._sparse_rebuildable:
                        if self._grid_status is None:
                            self._grid_status = wp.zeros(1, dtype=wp.uint32, device=positions.device)
                            self._grid_accumulated_status = wp.zeros(1, dtype=wp.uint32, device=positions.device)
                        point_mask = self._update_grid_point_mask(positions, particle_flags)
                    volume = allocate_by_voxels(
                        positions,
                        voxel_size,
                        padding_voxels=padding_voxels,
                        rebuildable=self._sparse_rebuildable,
                        max_active_voxels=self.max_active_cell_count if self._sparse_rebuildable else None,
                        status=self._grid_status,
                        point_mask=point_mask,
                        max_leaf_node_count=self.max_leaf_node_count,
                        max_lower_node_count=self.max_lower_node_count,
                        max_upper_node_count=self.max_upper_node_count,
                    )
                    if self._sparse_rebuildable:
                        self.check_sparse_grid_rebuild_status()
                        grid = fem.Nanogrid(volume, temporary_store=temporary_store, rebuildable=True)
                    else:
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
                    env_count=self._environment_count,
                )

        return grid

    def _update_grid_point_mask(
        self,
        positions: wp.array[wp.vec3],
        particle_flags: wp.array[wp.int32],
    ) -> wp.array[wp.int32]:
        if self._grid_point_mask is None:
            self._grid_point_mask = wp.empty(
                shape=particle_flags.shape,
                dtype=wp.int32,
                device=particle_flags.device,
            )
        elif self._grid_point_mask.shape != particle_flags.shape:
            raise RuntimeError("Implicit MPM particle count changed after sparse grid initialization")

        wp.launch(
            build_active_particle_mask,
            dim=particle_flags.shape[0],
            inputs=[positions, particle_flags, self._grid_point_mask],
            device=particle_flags.device,
        )
        return self._grid_point_mask

    def _create_geometry_partition(
        self,
        grid: fem.Geometry,
        positions: wp.array,
        particle_flags: wp.array,
        max_cell_count: int,
        point_environment: wp.array | None = None,
    ):
        """Create a geometry partition for the given positions."""

        active_cells = fem.borrow_temporary(self.temporary_store, shape=grid.cell_count(), dtype=int)
        active_cells.zero_()
        if self._separate_worlds:
            active_cell_integrand = mark_active_cells_by_environment
            active_cell_values = {
                "positions": positions,
                "particle_flags": particle_flags,
                "particle_environment": point_environment
                if point_environment is not None
                else self._particle_environment,
                "active_cells": active_cells,
            }
        else:
            active_cell_integrand = mark_active_cells
            active_cell_values = {
                "positions": positions,
                "particle_flags": particle_flags,
                "active_cells": active_cells,
            }
        fem.interpolate(
            active_cell_integrand,
            dim=positions.shape[0],
            at=fem.Cells(grid),
            values=active_cell_values,
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

        scratch = self._scratchpad

        with self._timer("Scratchpad"):
            scratch.rebuild_function_spaces(
                pic,
                strain_basis_str=self.strain_basis,
                velocity_basis_str=self.velocity_basis,
                collider_basis_str=self.collider_basis,
                max_cell_count=self.max_active_cell_count,
                environment_first=self._separate_worlds,
                contact_pic=scratch.contact_pic,
                temporary_store=self.temporary_store,
            )

            scratch.allocate_temporaries(
                collider_count=self._mpm_model.collider.collider_mesh.shape[0],
                has_compliant_bodies=self._mpm_model.has_compliant_colliders,
                has_critical_offset=self._mpm_model.critical_fraction > 0.0,
                max_colors=self._max_colors(),
                temporary_store=self.temporary_store,
            )

            if self.coloring:
                self._compute_coloring(pic, scratch=scratch)

        return scratch

    def _particles_to_cells(self, positions: wp.array) -> fem.PicQuadrature:
        """Rebuild the grid and grid partition around particles, then assign particles to grid cells."""

        grid_positions, grid_flags, grid_environment, point_stride = self._grid_points(positions)

        # Rebuild grid

        # The fixed grid and the rebuildable sparse grid both persist across steps: the
        # fixed grid is static, the sparse grid is refreshed in place from the current
        # particles. Plain sparse (no rebuild support) reallocates the grid each step.
        if self._scratchpad.grid is not None and (self.grid_type == "fixed" or self._sparse_rebuildable):
            grid = self._scratchpad.grid
            if self._sparse_rebuildable:
                point_mask = self._update_grid_point_mask(grid_positions, grid_flags)
                grid.rebuild(
                    grid_positions,
                    point_envs=grid_environment,
                    status=self._grid_status,
                    point_mask=point_mask,
                )
                wp.launch(
                    record_volume_rebuild_status,
                    dim=1,
                    inputs=[self._grid_status, self._grid_accumulated_status],
                    device=positions.device,
                )
        else:
            grid = self._allocate_grid(
                grid_positions,
                grid_flags,
                voxel_size=self._mpm_model.voxel_size,
                temporary_store=self.temporary_store,
                padding_voxels=self.grid_padding,
                point_environment=grid_environment,
                particle_range_scale=point_stride,
            )

        # Build active partition. Plain sparse uses the whole grid; fixed and rebuildable
        # sparse use a capacity-bounded partition that masks to the active cells (the
        # rebuildable grid's cell buffers are capacity-sized and include unused slots).
        with self._timer("Build active partition"):
            if self.grid_type == "sparse" and not self._sparse_rebuildable:
                max_cell_count = -1
                geo_partition = grid
            else:
                max_cell_count = self.max_active_cell_count
                geo_partition = self._create_geometry_partition(
                    grid, grid_positions, grid_flags, max_cell_count, grid_environment
                )

        # Bin particles to grid cells
        with self._timer("Bin particles"):
            domain = fem.Cells(geo_partition)

            # Whole-grid sparse domains use identical geometry and domain indices; fixed and
            # rebuildable grids have explicit partitions that require domain indices.
            use_domain_element_indices = not (
                self._separate_worlds and self.grid_type == "sparse" and not self._sparse_rebuildable
            )
            self._scratchpad.pic_domain_element_indices = use_domain_element_indices

            if self.gimp:
                particle_locations = self._particle_grid_locations_gimp(
                    domain,
                    positions,
                    self._mpm_model.particle_radius,
                    self._mpm_model.voxel_size,
                    self._particle_environment,
                )
                pic = _PicQuadrature(
                    domain=domain,
                    positions=particle_locations,
                    measures=self._mpm_model.particle_volume,
                    temporary_store=self.temporary_store,
                    use_domain_element_indices=use_domain_element_indices,
                )
                self._scratchpad.particle_cells = particle_locations[0]
                center_pic = None
                if self.collider_basis[:3] == "pic" or "particle" in self._density_measurements():
                    center_pic = _PicQuadrature(
                        domain=domain,
                        positions=self._particle_cell_locations(domain, positions, self._particle_environment),
                        measures=self._mpm_model.particle_volume,
                        temporary_store=self.temporary_store,
                        use_domain_element_indices=use_domain_element_indices,
                    )
                if self.collider_basis[:3] == "pic":
                    self._scratchpad.contact_pic = center_pic
                # Particle-centered density measurements gather neighbors through one point per particle center
                self._scratchpad.density_pic = center_pic
            else:
                particle_locations = self._particle_cell_locations(domain, positions, self._particle_environment)
                self._scratchpad.particle_cells = particle_locations[0].reshape((-1, 1))
                pic = _PicQuadrature(
                    domain=domain,
                    positions=particle_locations,
                    measures=self._mpm_model.particle_volume,
                    temporary_store=self.temporary_store,
                    use_domain_element_indices=use_domain_element_indices,
                )
                self._scratchpad.density_pic = pic

        return pic

    def _grid_points(self, positions: wp.array):
        """Points whose cells are activated: particle centers, or the corners of the GIMP particle domains."""
        flags = self._mpm_model.particle_flags
        if not self.gimp:
            return positions, flags, self._particle_environment, 1

        count = positions.shape[0]
        device = positions.device
        corners = fem.borrow_temporary(self.temporary_store, shape=8 * count, dtype=wp.vec3, device=device)
        corner_flags = fem.borrow_temporary(self.temporary_store, shape=8 * count, dtype=wp.int32, device=device)
        corner_environment = (
            None
            if self._particle_environment is None
            else fem.borrow_temporary(self.temporary_store, shape=8 * count, dtype=int, device=device)
        )
        self._scratchpad.gimp_domain_corners = (corners, corner_flags, corner_environment)
        wp.launch(
            gimp_domain_corners,
            dim=(count, 8),
            inputs=[
                positions,
                self._mpm_model.particle_radius,
                flags,
                self._particle_environment,
                self._mpm_model.voxel_size,
            ],
            outputs=[corners, corner_flags, corner_environment],
            device=device,
        )
        return corners, corner_flags, corner_environment, 8

    def _particle_cell_locations(
        self,
        domain: fem.GeometryDomain,
        positions: wp.array,
        particle_environment: wp.array | None,
    ):
        """Bin particle centers to the cells containing them; particles outside every cell are left unbinned."""

        cell_lookup = domain.element_partition_lookup
        cell_coordinates = domain.element_coordinates
        separate_worlds = self._separate_worlds

        @fem.cache.dynamic_kernel(suffix=f"{domain.name}_{'isolated' if separate_worlds else 'shared'}_inside")
        def particle_locations(
            cell_arg_value: domain.ElementArg,
            domain_index_arg_value: domain.ElementIndexArg,
            positions: wp.array[wp.vec3],
            particle_environment: wp.array[int],
            cell_index: wp.array[fem.ElementIndex],
            cell_coords: wp.array[fem.Coords],
        ):
            p = wp.tid()
            domain_arg = domain.DomainArg(cell_arg_value, domain_index_arg_value)
            if wp.static(separate_worlds):
                sample = cell_lookup(domain_arg, positions[p], int(particle_environment[p]))
            else:
                sample = cell_lookup(domain_arg, positions[p])

            cell_index[p] = fem.NULL_ELEMENT_INDEX
            cell_coords[p] = fem.Coords(fem.OUTSIDE)
            if sample.element_index != fem.NULL_ELEMENT_INDEX:
                # Grid lookups clamp outside points to boundary cells, where coordinates extrapolate.
                coords = cell_coordinates(cell_arg_value, sample.element_index, positions[p])
                if wp.min(coords) >= 0.0 and wp.max(coords) <= 1.0:
                    cell_index[p] = domain.element_partition_index(domain_index_arg_value, sample.element_index)
                    cell_coords[p] = coords

        device = positions.device
        cell_indices = fem.borrow_temporary(self.temporary_store, shape=positions.shape, dtype=fem.ElementIndex)
        cell_coords = fem.borrow_temporary(self.temporary_store, shape=positions.shape, dtype=fem.Coords)
        wp.launch(
            particle_locations,
            dim=positions.shape[0],
            inputs=[
                domain.element_arg_value(device=device),
                domain.element_index_arg_value(device=device),
                positions,
                particle_environment,
            ],
            outputs=[cell_indices, cell_coords],
            device=device,
        )
        return cell_indices, cell_coords

    def _particle_grid_locations_gimp(
        self,
        domain: fem.GeometryDomain,
        positions: wp.array,
        particle_radius: wp.array,
        voxel_size: float,
        particle_environment: wp.array | None,
    ):
        """Split each particle's cubic domain over the cells it overlaps (uGIMP).

        The domain is the cube with the particle diameter, capped at one voxel so that it spans at most two cells
        per axis. Each overlapped cell receives the overlap volume fraction, with its quadrature point at the overlap
        centroid, which integrates multilinear shape functions and their gradients exactly over the domain.
        """

        cell_lookup = domain.element_partition_lookup
        separate_worlds = self._separate_worlds
        grid_offset = 0.5 if self.grid_type == "sparse" else 0.0

        @fem.cache.dynamic_kernel(suffix=f"{domain.name}_{'isolated' if separate_worlds else 'shared'}_overlap")
        def particle_locations_gimp(
            cell_arg_value: domain.ElementArg,
            domain_index_arg_value: domain.ElementIndexArg,
            positions: wp.array[wp.vec3],
            particle_radius: wp.array[float],
            particle_environment: wp.array[int],
            voxel_size: float,
            cell_index: wp.array2d[fem.ElementIndex],
            cell_coords: wp.array2d[fem.Coords],
            cell_fractions: wp.array2d[float],
        ):
            p = wp.tid()
            domain_arg = domain.DomainArg(cell_arg_value, domain_index_arg_value)
            center = positions[p]

            if wp.static(separate_worlds):
                home = cell_lookup(domain_arg, center, int(particle_environment[p]))
            else:
                home = cell_lookup(domain_arg, center)
            if home.element_index == fem.NULL_ELEMENT_INDEX:
                return

            uc = home.element_coords
            size = 2.0 * gimp_half_size(particle_radius[p], voxel_size) / voxel_size
            if size <= 0.0:
                cell_index[p, 0] = domain.element_partition_index(domain_index_arg_value, home.element_index)
                cell_coords[p, 0] = uc
                cell_fractions[p, 0] = 1.0
                return

            # Per axis, the domain [lo, hi] (home-cell coordinates) splits at most once, at `split`:
            # segment 0 is [lo, split] and segment 1 is [split, hi], one of them in a neighbor cell.
            lo = uc - wp.vec3(0.5 * size)
            hi = uc + wp.vec3(0.5 * size)
            split = wp.vec3()
            for a in range(3):
                split[a] = wp.where(lo[a] < 0.0, 0.0, wp.where(hi[a] > 1.0, 1.0, hi[a]))

            # Cell origins are integer grid coordinates; round away lookup roundoff.
            origin = center / voxel_size - uc + wp.vec3(grid_offset)
            home_ijk = wp.vec3i(int(wp.round(origin[0])), int(wp.round(origin[1])), int(wp.round(origin[2])))

            tot_weight = float(0.0)
            for combo in range(8):
                weight = float(1.0)
                centroid = wp.vec3()
                for a in range(3):
                    if (combo >> a) & 1 == 0:
                        weight *= (split[a] - lo[a]) / size
                        centroid[a] = 0.5 * (lo[a] + split[a])
                    else:
                        weight *= (hi[a] - split[a]) / size
                        centroid[a] = 0.5 * (split[a] + hi[a])
                if weight <= GIMP_MIN_OVERLAP_FRACTION:
                    continue

                # Grid cells are axis-aligned cubes of edge voxel_size
                pos = center + voxel_size * (centroid - uc)
                if wp.static(separate_worlds):
                    sample = cell_lookup(domain_arg, pos, int(particle_environment[p]))
                else:
                    sample = cell_lookup(domain_arg, pos)

                # Lookups outside the active cells snap onto a cell face; those parts are left out
                coords = sample.element_coords
                if sample.element_index == fem.NULL_ELEMENT_INDEX or wp.min(coords) <= 0.0 or wp.max(coords) >= 1.0:
                    continue

                # Slot = parity of the overlapped cell's index along each axis: distinct for the (at most two
                # per axis) cells a domain overlaps, and unchanged while the particle keeps overlapping a cell.
                # The cell is the home cell plus the exact segment offset, independent of centroid rounding.
                ijk = home_ijk
                for a in range(3):
                    bit = (combo >> a) & 1
                    if lo[a] < 0.0:
                        ijk[a] = home_ijk[a] + bit - 1
                    elif hi[a] > 1.0:
                        ijk[a] = home_ijk[a] + bit
                slot = ((ijk[0] & 1) << 2) | ((ijk[1] & 1) << 1) | (ijk[2] & 1)
                cell_index[p, slot] = domain.element_partition_index(domain_index_arg_value, sample.element_index)
                cell_coords[p, slot] = coords
                cell_fractions[p, slot] = weight
                tot_weight += weight

            for k in range(8):
                if cell_index[p, k] != fem.NULL_ELEMENT_INDEX:
                    cell_fractions[p, k] /= tot_weight

        device = positions.device

        cell_indices = fem.borrow_temporary(self.temporary_store, shape=(positions.shape[0], 8), dtype=fem.ElementIndex)
        cell_coords = fem.borrow_temporary(self.temporary_store, shape=(positions.shape[0], 8), dtype=fem.Coords)
        cell_fractions = fem.borrow_temporary(self.temporary_store, shape=(positions.shape[0], 8), dtype=float)

        cell_indices.fill_(fem.NULL_ELEMENT_INDEX)

        wp.launch(
            particle_locations_gimp,
            dim=positions.shape[0],
            inputs=[
                domain.element_arg_value(device=device),
                domain.element_index_arg_value(device=device),
                positions,
                particle_radius,
                particle_environment,
                voxel_size,
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
        _solve_graph = self._solve_rheology(scratch, rigidity_operator)

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
                    "particle_f": state_in.particle_f,
                    "particle_inv_mass": model.particle_inv_mass,
                    "particle_density": mpm_model.particle_density,
                    "particle_flags": mpm_model.particle_flags,
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
                        "particle_flags": mpm_model.particle_flags,
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
                    "particle_flags": mpm_model.particle_flags,
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

        # Particle contacts close their gap in the rasterized velocity
        speculative = self.collider_basis[:3] == "pic"

        with self._timer("Rasterize collider"):
            self._mpm_model.collider.query_max_dist = max(
                self._mpm_model.collider.query_max_dist,
                self._contact_radius_reach()
                + max(self.collider_contact_gap, speculative_stop_distance(self._mpm_model.voxel_size)),
            )
            # volume associated to each collider node
            fem.integrate(
                integrate_fraction,
                fields={"phi": scratch.collider_fraction_test},
                values={"inv_cell_volume": inv_cell_volume},
                assembly="nodal",
                output=scratch.collider_node_volume,
                temporary_store=self.temporary_store,
            )

            # Point contacts keep their particle's radius from collider surfaces
            node_radii = None
            if speculative:
                node_radii = fem.borrow_temporary(self.temporary_store, shape=collider_node_count, dtype=float)
                wp.launch(
                    point_contact_radii,
                    dim=collider_node_count,
                    inputs=[
                        scratch.collider_fraction_test.space_restriction.space_partition.space_node_indices(),
                        scratch.collision_pic.cell_particle_indices,
                        self._mpm_model.particle_radius,
                    ],
                    outputs=[node_radii],
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
                node_environment_offsets=scratch.collider_environment_offsets,
                contact_gap=self.collider_contact_gap,
                node_radii=node_radii,
            )

            # normal interpolation
            if self.collider_normal_from_sdf_gradient:
                distance_gradients = None
                if speculative:
                    # Particle nodes differentiate the collider distance rasterized at cell centers
                    distance_gradients = collider_distance_gradients(
                        self._mpm_model.collider,
                        state_in.body_q,
                        scratch.grid,
                        self._mpm_model.voxel_size,
                        scratch.collider_position_field.dof_values,
                        scratch.collider_environment_offsets,
                        self._separate_worlds,
                        self.temporary_store,
                    )
                interpolate_collider_normals(
                    scratch.collider_fraction_test.space_restriction,
                    scratch.collider_distance_field,
                    scratch.collider_normal_field,
                    temporary_store=self.temporary_store,
                    collider_velocity=scratch.collider_velocity,
                    dt=dt,
                    distance_gradients=distance_gradients,
                    stop_distance=speculative_stop_distance(self._mpm_model.voxel_size),
                )

            contact_nodes = scratch.gimp_particle_contact_nodes
            if contact_nodes is not None:
                contact_nodes.fill_(-1)
                wp.launch(
                    invert_point_particle_indices,
                    dim=scratch.collider_normal_field.space_partition.node_count(),
                    inputs=[
                        scratch.contact_pic.cell_particle_indices,
                        scratch.collider_normal_field.space_partition.space_node_indices(),
                        scratch.collider_normal_field.dof_values,
                        contact_nodes,
                    ],
                    device=self.model.device,
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
                    # Preserve first-sample arithmetic; older Warp falls back to triplets.
                    bsr_options={"construction": "auto"} if self._use_local_contact_construction(scratch) else None,
                )

            if self.collider_recovery_fraction > 0.0 or self.collider_contact_gap > 0.0:
                wp.launch(
                    stabilize_collider_velocity,
                    dim=collider_node_count,
                    inputs=[
                        dt,
                        self.collider_recovery_fraction,
                        not speculative,
                        scratch.collider_distance_field.dof_values,
                        scratch.collider_normal_field.dof_values,
                        scratch.collider_velocity,
                    ],
                )

    def _use_local_contact_construction(self, scratch: ImplicitMPMScratchpad) -> bool:
        """Prefer row compression for validated CUDA contact-map layouts."""
        # Compact maps with many inactive partition rows can still favor triplets,
        # even when Warp packs active-row candidate capacity.
        return (
            _ROW_COMPRESSED_CONTACT_CONSTRUCTION
            and self.model.device.is_cuda
            and self.velocity_basis == "Q1"
            and self.collider_basis in ("S2", "S3")
            and scratch.collider_node_count <= scratch.collider_fraction_test.space_restriction.node_count()
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
            body_q = state_in.body_q
            if body_q is None:
                body_q = wp.empty(0, dtype=wp.transform, device=self.model.device)

            rigidity_operator = build_rigidity_operator(
                cell_volume=cell_volume,
                node_volumes=scratch.collider_node_volume,
                node_positions=scratch.collider_position_field.dof_values,
                collider=self._mpm_model.collider,
                body_q=body_q,
                body_mass=self._mpm_model.collider_body_mass,
                body_inv_inertia=self._mpm_model.collider_body_inv_inertia,
                particle_mass=self._mpm_model.model.particle_mass,
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
                integrate_active_fraction,
                quadrature=pic,
                fields={"phi": scratch.fraction_test},
                values={
                    "inv_cell_volume": inv_cell_volume,
                    "particle_flags": mpm_model.material_particle_flags,
                },
                output_dtype=float,
                temporary_store=self.temporary_store,
            )

            elastic_parameters_int = fem.integrate(
                integrate_elastic_parameters,
                quadrature=pic,
                fields={"u": scratch.velocity_test},
                values={
                    "material_parameters": mpm_model.material_parameters,
                    "particle_flags": mpm_model.material_particle_flags,
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
                    "particle_flags": mpm_model.material_particle_flags,
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
                    "particle_flags": mpm_model.material_particle_flags,
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

        # A full cell's inertial Delassus scale, dt^2 / (rho dx^2), sets the rheology regularization
        max_density = wp.zeros(shape=(1,), dtype=float)
        wp.launch(
            max_particle_density,
            dim=mpm_model.particle_density.shape[0],
            inputs=[mpm_model.particle_density, mpm_model.material_particle_flags, max_density],
        )
        scratch.delassus_scale = wp.empty(shape=(1,), dtype=float)
        dt_over_dx_sq = dt * dt / (self.voxel_size * self.voxel_size)
        wp.launch(inertial_delassus_scale, dim=1, inputs=[dt_over_dx_sq, max_density, scratch.delassus_scale])

        with self._timer("Strain-node particle volumes"):
            fem.integrate(
                integrate_active_fraction,
                quadrature=pic,
                fields={"phi": scratch.divergence_test},
                values={
                    "inv_cell_volume": inv_cell_volume,
                    "particle_flags": mpm_model.material_particle_flags,
                },
                output=scratch.strain_node_particle_volume,
                temporary_store=self.temporary_store,
            )

        scratch.unilateral_strain_offset.zero_()
        if self._measures_density():
            self._build_density_strain_offset(state_in, scratch, pic)

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
                    "particle_flags": mpm_model.material_particle_flags,
                },
                output_dtype=float,
                output=scratch.strain_matrix,
                temporary_store=self.temporary_store,
                bsr_options={"prune_numerical_zeros": self._velocity_nodes_per_strain_sample < 0},
            )

        scratch.strain_basis_transform, scratch.strain_stress_transform = self._build_strain_basis_transform(
            pic, scratch, inv_cell_volume
        )
        with self._timer("Interpolated yield parameters"):
            integrand = integrate_yield_parameters
            fields = {"u": scratch.strain_yield_parameters_test}
            if scratch.strain_basis_transform is not None:
                integrand = integrate_transformed_yield_parameters
                fields["modes"] = scratch.strain_basis_transform_field
            moments = fem.integrate(
                integrand,
                quadrature=pic,
                fields=fields,
                values={
                    "particle_Jp": state_in.mpm.particle_Jp,
                    "material_parameters": mpm_model.material_parameters,
                    "particle_flags": mpm_model.material_particle_flags,
                    "inv_cell_volume": inv_cell_volume,
                    "dt": dt,
                },
                output_dtype=YieldMomentVec,
                assembly="generic",
                kernel_options={"max_unroll": 4, "enable_backward": False},
                temporary_store=self.temporary_store,
            )
            values = moments.view(dtype=float)
            parameters = values[:, :YIELD_PARAM_LENGTH]
            wp.launch(
                inverse_scale_vector,
                dim=(scratch.strain_node_count, YIELD_PARAM_LENGTH),
                inputs=[values[:, YIELD_PARAM_LENGTH:], parameters],
            )
            scratch.strain_yield_parameters_field.dof_values = parameters.view(YieldParamVec)

    def _contact_radius_reach(self) -> float:
        """Largest particle radius, which particle collider bases subtract from collider distances."""
        if self.collider_basis[:3] != "pic":
            return 0.0
        return self._mpm_model.max_particle_radius

    def _measures_density(self) -> bool:
        """Whether critical packing or density feedback needs a density measurement."""
        return self._mpm_model.critical_fraction > 0.0 or self.volume_recovery_fraction > 0.0

    def _density_measurements(self) -> set[str]:
        """Measurements needed this step, among ``"grid"`` and ``"particle"``."""
        measurements = set()
        if self._mpm_model.critical_fraction > 0.0:
            measurements.add(self.critical_fraction_measurement)
        if self.volume_recovery_fraction > 0.0:
            measurements.add(self.volume_recovery_measurement)
        return measurements

    def _build_density_strain_offset(
        self,
        state_in: newton.State,
        scratch: ImplicitMPMScratchpad,
        pic: fem.PicQuadrature,
    ):
        """Measure packing independently of the mass and strain quadrature.

        Each particle gets a measured and an available volume fraction from the measurement of
        each consumer: density feedback pushes overfilled particles apart, and the critical bound
        lets underfilled particles compress. The strain quadrature samples both offsets like the
        elastic strain, so they share the weights of the strain mass and stay bounded in every
        strain mode.
        """
        model = self._mpm_model
        device = state_in.particle_q.device
        particle_count = state_in.particle_q.shape[0]
        volumes = fem.borrow_temporary_like(model.material_particle_volume, self.temporary_store)
        wp.launch(
            deformed_particle_volumes,
            dim=volumes.shape[0],
            inputs=[model.material_particle_volume, state_in.mpm.particle_elastic_strain, volumes],
            device=device,
        )

        grid = scratch.grid
        environments = self._particle_environment if self._separate_worlds else None
        collider = model.collider
        old_query_distance = collider.query_max_dist
        # Wall masks need distances up to the cell half-diagonal
        collider.query_max_dist = max(old_query_distance, model.voxel_size)
        try:
            masks = fem.borrow_temporary(self.temporary_store, shape=grid.cell_count(), dtype=wp.uint32, device=device)
            collider_inputs = [collider, state_in.body_q, None, None, self._separate_worlds]
            wp.launch(
                make_free_box_mask_kernel(grid),
                dim=grid.cell_count(),
                inputs=[grid.cell_arg_value(device), *collider_inputs, model.voxel_size, masks],
                device=device,
            )

            fractions = {}
            for measurement in self._density_measurements():
                measured = fem.borrow_temporary(self.temporary_store, shape=particle_count, dtype=float, device=device)
                available = fem.borrow_temporary_like(measured, self.temporary_store)
                if measurement == "grid":
                    cell_values = fem.borrow_temporary(
                        self.temporary_store, shape=grid.cell_count(), dtype=float, device=device
                    )
                    cell_values.zero_()
                    wp.launch(
                        make_grid_volume_kernel(grid),
                        dim=particle_count,
                        inputs=[
                            grid.cell_arg_value(device),
                            state_in.particle_q,
                            volumes,
                            model.material_particle_flags,
                            environments,
                            model.voxel_size,
                            cell_values,
                        ],
                        device=device,
                    )
                    readback = make_grid_readback_kernel(grid)
                    readback_inputs = [
                        grid.cell_arg_value(device),
                        state_in.particle_q,
                        model.material_particle_flags,
                        environments,
                    ]
                    wp.launch(
                        readback,
                        dim=particle_count,
                        inputs=[*readback_inputs, cell_values, model.voxel_size, measured],
                        device=device,
                    )
                    wp.launch(
                        make_grid_capacity_kernel(grid),
                        dim=grid.cell_count(),
                        inputs=[grid.cell_arg_value(device), masks, *collider_inputs, model.voxel_size, cell_values],
                        device=device,
                    )
                    wp.launch(
                        readback,
                        dim=particle_count,
                        inputs=[*readback_inputs, cell_values, model.voxel_size, available],
                        device=device,
                    )
                else:
                    center_pic = scratch.density_pic
                    wp.launch(
                        make_particle_volume_kernel(center_pic),
                        dim=particle_count,
                        inputs=[
                            grid.cell_arg_value(device),
                            center_pic.domain.element_index_arg_value(device),
                            center_pic.cell_particle_offsets,
                            center_pic.cell_particle_indices,
                            state_in.particle_q,
                            volumes,
                            model.material_particle_flags,
                            environments,
                            model.voxel_size,
                            measured,
                        ],
                        device=device,
                    )
                    wp.launch(
                        make_particle_capacity_kernel(grid),
                        dim=particle_count,
                        inputs=[
                            grid.cell_arg_value(device),
                            state_in.particle_q,
                            model.material_particle_flags,
                            environments,
                            masks,
                            *collider_inputs,
                            model.voxel_size,
                            available,
                        ],
                        device=device,
                    )
                fractions[measurement] = (measured, available)
        finally:
            collider.query_max_dist = old_query_distance

        values = fem.borrow_temporary(self.temporary_store, shape=particle_count, dtype=float, device=device)
        laws = []
        if model.critical_fraction > 0.0:
            scratch.critical_strain_offset.zero_()
            laws.append(
                (
                    critical_offset_values,
                    self.critical_fraction_measurement,
                    model.critical_fraction,
                    scratch.critical_strain_offset,
                )
            )
        if self.volume_recovery_fraction > 0.0:
            laws.append(
                (
                    feedback_offset_values,
                    self.volume_recovery_measurement,
                    self.volume_recovery_fraction,
                    scratch.unilateral_strain_offset,
                )
            )
        for law, measurement, parameter, output in laws:
            measured, available = fractions[measurement]
            wp.launch(
                law,
                dim=particle_count,
                inputs=[measured, available, model.material_particle_flags, parameter, values],
                device=device,
            )
            fem.integrate(
                point_density_offset,
                quadrature=pic,
                fields={"tau": scratch.divergence_test},
                values={
                    "values": values,
                    "flags": model.material_particle_flags,
                    "inv_cell_volume": 1.0 / model.voxel_size**3,
                },
                output=output,
                add=True,
                temporary_store=self.temporary_store,
            )

    def _build_strain_basis_transform(
        self,
        pic: fem.PicQuadrature,
        scratch: ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        if self.strain_basis[:3] == "pic":
            # Point nodes already hold their particle volumes from the strain-node integration.
            return None, None
        nodes_per_element = scratch.divergence_test.space.topology.MAX_NODES_PER_ELEMENT
        if scratch.strain_basis_transform_field is None:
            scratch.strain_node_particle_volume += EPSILON
            return None, None

        mass = fem.integrate(
            mass_form,
            quadrature=pic,
            fields={"p": scratch.divergence_test, "q": scratch.divergence_trial},
            values={
                "inv_cell_volume": inv_cell_volume,
                "particle_flags": self._mpm_model.material_particle_flags,
            },
            output_dtype=float,
        )
        mass_blocks = wps.bsr_copy(mass, block_shape=(nodes_per_element, nodes_per_element))
        transform = wp.empty((mass_blocks.nrow, nodes_per_element, nodes_per_element), dtype=float)
        stress_transform = wp.empty_like(transform)
        diagonal_mass = scratch.strain_node_particle_volume.reshape((-1, nodes_per_element))
        wp.launch(
            compute_nodal_diagonalization,
            dim=mass_blocks.nrow,
            inputs=[mass_blocks.offsets, mass_blocks.columns, mass_blocks.values, diagonal_mass],
            outputs=[diagonal_mass, transform, stress_transform],
        )
        modes = scratch.strain_basis_transform_field
        modes.dof_values = (
            transform.transpose((0, 2, 1)).contiguous().reshape((-1, nodes_per_element)).view(modes.dof_dtype)
        )
        return transform, stress_transform

    def _apply_strain_basis_transform(
        self,
        scratch: ImplicitMPMScratchpad,
        basis_transform: wp.array3d[float] | None,
        stress_transform: wp.array3d[float] | None,
    ):
        node_count = scratch.strain_node_count

        if basis_transform is not None:
            # Transform the assembled operators and their dual vectors.

            nodes_per_elt = basis_transform.shape[1]
            elt_count = basis_transform.shape[0]

            B = scratch.strain_matrix
            strain_mat_tmp = wp.empty_like(B.values)
            wp.launch(
                rotate_matrix_rows, dim=B.nnz, inputs=[basis_transform, B.offsets, B.columns, B.values, strain_mat_tmp]
            )
            # Captured assembly retains this pointer; replacing it can free the
            # warm-up allocation before the first graph replay.
            wp.copy(B.values, strain_mat_tmp)

            C = scratch.compliance_matrix
            compliance_mat_tmp = wp.empty_like(C.values)
            wp.launch(
                rotate_matrix_rows,
                dim=C.nnz,
                inputs=[basis_transform, C.offsets, C.columns, C.values, compliance_mat_tmp],
            )
            wp.launch(
                rotate_matrix_columns,
                dim=C.nnz,
                inputs=[basis_transform, C.offsets, C.columns, compliance_mat_tmp, C.values],
            )

            rotate_vectors = make_rotate_vectors(nodes_per_elt)
            wp.launch_tiled(
                rotate_vectors,
                dim=elt_count,
                block_dim=32,
                inputs=[
                    basis_transform,
                    stress_transform if self._stress_warmstart != "particles" else basis_transform,
                    _as_2d_array(scratch.elastic_strain_delta_field.dof_values, shape=(node_count, 6), dtype=float),
                    _as_2d_array(scratch.stress_field.dof_values, shape=(node_count, 6), dtype=float),
                    _as_2d_array(scratch.unilateral_strain_offset, shape=(node_count, 1), dtype=float),
                ],
            )

        if scratch.critical_strain_offset is not None:
            wp.launch(
                add_critical_strain_offset,
                dim=node_count,
                inputs=[basis_transform, scratch.critical_strain_offset, scratch.unilateral_strain_offset],
            )

        M_diag = scratch.strain_node_particle_volume
        if self._stress_warmstart == "particles":
            # Particle stresses are integrated, need scale with inverse node volume

            wp.launch(
                inverse_scale_sym_tensor,
                dim=node_count,
                inputs=[M_diag, scratch.stress_field.dof_values],
            )

    def _unapply_strain_basis_transform(
        self,
        scratch: ImplicitMPMScratchpad,
        basis_transform: wp.array3d[float] | None,
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
        if self._mpm_model.has_hardening or self.carry_residual_strain == "sampled":
            wp.launch(
                inverse_scale_sym_tensor,
                dim=node_count,
                inputs=[M_diag, scratch.plastic_strain_delta_field.dof_values],
            )

        if basis_transform is not None:
            # Reconstruct fields in the original nodal basis.

            elt_count = basis_transform.shape[0]
            nodes_per_elt = basis_transform.shape[1]

            inverse_rotate_vectors = make_inverse_rotate_vectors(nodes_per_elt)
            wp.launch_tiled(
                inverse_rotate_vectors,
                dim=elt_count,
                block_dim=32,
                inputs=[
                    basis_transform,
                    _as_2d_array(scratch.stress_field.dof_values, shape=(node_count, 6), dtype=float),
                    _as_2d_array(scratch.elastic_strain_delta_field.dof_values, shape=(node_count, 6), dtype=float),
                    _as_2d_array(scratch.plastic_strain_delta_field.dof_values, shape=(node_count, 6), dtype=float),
                ],
            )

    def _solve_rheology(
        self,
        scratch: ImplicitMPMScratchpad,
        rigidity_operator: tuple[wps.BsrMatrix, wps.BsrMatrix, wps.BsrMatrix] | None,
    ):
        basis_transform = scratch.strain_basis_transform
        self._apply_strain_basis_transform(scratch, basis_transform, scratch.strain_stress_transform)

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
                strain_environment_offsets=scratch.strain_environment_offsets,
                has_viscosity=self._mpm_model.has_viscosity,
                has_dilatancy=self._mpm_model.has_dilatancy,
                strain_velocity_node_count=-1 if self._separate_worlds else self._velocity_nodes_per_strain_sample,
                strain_nodes_per_cell=(
                    scratch.strain_nodes_per_element
                    if not self.strain_basis.startswith("pic") and _parse_grid_basis_name(self.strain_basis)[2]
                    else 0
                ),
                include_unconverged_strain=self.carry_residual_strain == "unconverged",
            )
            contact_temporaries = []
            contact_cell_offsets = None
            contact_cell_rows = None
            if self.gs_colored_contacts:
                contact_pic = scratch.collision_pic
                partition = scratch._strain_space_restriction.space_partition
                cell_count = partition.node_count() // scratch.strain_nodes_per_element
                node_count = scratch.collider_node_count
                contact_cell_offsets = fem.borrow_temporary(self.temporary_store, shape=cell_count + 1, dtype=int)
                keys = fem.borrow_temporary(self.temporary_store, shape=2 * node_count, dtype=int)
                contact_cell_rows = fem.borrow_temporary(self.temporary_store, shape=2 * node_count, dtype=int)
                contact_temporaries = [contact_cell_offsets, keys, contact_cell_rows]
                contact_cell_offsets.zero_()
                wp.launch(
                    make_contact_cell_keys_kernel(
                        contact_pic.domain,
                        partition,
                        scratch.strain_nodes_per_element,
                        scratch.pic_domain_element_indices,
                    ),
                    dim=node_count,
                    inputs=[
                        contact_pic.domain.element_index_arg_value(device=self.model.device),
                        partition.partition_arg_value(device=self.model.device),
                        scratch._collision_space_restriction.space_partition.space_node_indices(),
                        contact_pic.cell_particle_indices,
                        contact_pic.cell_indices,
                        scratch.collider_matrix.offsets,
                        keys,
                        contact_cell_rows,
                        contact_cell_offsets,
                    ],
                    device=self.model.device,
                )
                wp.utils.radix_sort_pairs(keys=keys, values=contact_cell_rows, count=node_count)
                wp.utils.array_scan(contact_cell_offsets, contact_cell_offsets, inclusive=True)
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
                colored_cell_offsets=contact_cell_offsets,
                colored_cell_rows=contact_cell_rows,
                colored_velocity_nodes_per_cell=scratch.velocity_nodes_per_element,
            )

            # Retain graph to avoid immediate CPU sync
            solve_graph = solve_rheology(
                self.solver,
                self.max_iterations,
                self.tolerance,
                momentum_data,
                rheology_data,
                collision_data,
                scratch.delassus_scale,
                temporary_store=self.temporary_store,
                use_graph=self._use_cuda_graph,
                verbose=self.verbose,
                acceleration=self._rheology_acceleration,
            )

        for temporary in contact_temporaries:
            temporary.release()
        self._unapply_strain_basis_transform(scratch, basis_transform)

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

                # In place, the copy for skipped particles is a no-op, so no clone is needed for stress
                for previous, values in (
                    (particle_Jp_prev, state_out.mpm.particle_Jp),
                    (state_in.mpm.particle_stress, state_out.mpm.particle_stress),
                    (elastic_strain_prev, state_out.mpm.particle_elastic_strain),
                ):
                    wp.launch(
                        reset_particle_values,
                        dim=model.particle_count,
                        inputs=[
                            scratch.particle_cells,
                            mpm_model.material_particle_flags,
                            mpm_model.particle_density,
                            True,
                            previous,
                            values,
                        ],
                    )

                fem.interpolate(
                    update_particle_strains,
                    at=pic,
                    values={
                        "dt": dt,
                        "kinematic_elastic_strain": self.carry_residual_strain == "sampled",
                        "particle_flags": mpm_model.material_particle_flags,
                        "particle_density": mpm_model.particle_density,
                        "particle_volume": mpm_model.material_particle_volume,
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
            for previous, values in (
                (state_in.particle_qd, state_out.particle_qd),
                (state_in.mpm.particle_qd_grad, state_out.mpm.particle_qd_grad),
            ):
                wp.launch(
                    reset_particle_values,
                    dim=model.particle_count,
                    inputs=[
                        scratch.particle_cells,
                        mpm_model.particle_flags,
                        mpm_model.particle_density,
                        False,
                        previous,
                        values,
                    ],
                )
            state_out.particle_q.assign(state_in.particle_q)

            fem.interpolate(
                advect_particles,
                at=pic,
                values={
                    "particle_flags": mpm_model.particle_flags,
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
        scratch.require_strain_space_fields(
            transform_basis=self.strain_basis[:3] != "pic"
            and self.strain_basis not in ("Q1", "S2")
            and scratch.strain_nodes_per_element > 1
        )
        last_step_data.require_strain_space_fields(scratch, smoothed=self._stress_warmstart == "smoothed")
        if (
            self.gimp
            and self._stress_warmstart
            and (
                self._stress_warmstart == "particles"
                or isinstance(scratch.stress_field.space.basis, fem.PointBasisSpace)
            )
        ):
            last_step_data.require_gimp_stress_fields(self.model.particle_count, self.model.device)

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

            if (
                self.gimp
                and self._stress_warmstart
                and (
                    self._stress_warmstart == "particles"
                    or isinstance(scratch.stress_field.space.basis, fem.PointBasisSpace)
                )
            ):
                point_stress, point_valid, point_centers = last_step_data.gimp_stress
                fem.integrate(
                    integrate_gimp_point_stress,
                    quadrature=pic,
                    fields={"tau": scratch.sym_strain_test},
                    values={
                        "inv_cell_volume": inv_cell_volume,
                        "gimp_cells": scratch.particle_cells,
                        "point_stress": point_stress,
                        "point_valid": point_valid,
                        "point_centers": point_centers,
                        "voxel_size": self._mpm_model.voxel_size,
                        "particle_stress": state_in.mpm.particle_stress,
                        "particle_flags": self._mpm_model.material_particle_flags,
                    },
                    output=scratch.stress_field.dof_values,
                    temporary_store=self.temporary_store,
                )
                if self._stress_warmstart != "particles":
                    wp.launch(
                        inverse_scale_sym_tensor,
                        dim=scratch.strain_node_count,
                        inputs=[scratch.strain_node_particle_volume, scratch.stress_field.dof_values],
                    )
            elif self._stress_warmstart == "particles":
                fem.integrate(
                    integrate_particle_stress,
                    quadrature=pic,
                    fields={
                        "tau": scratch.sym_strain_test,
                    },
                    values={
                        "particle_stress": state_in.mpm.particle_stress,
                        "particle_flags": self._mpm_model.material_particle_flags,
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

        # The rebuildable sparse grid is refreshed in place, so the previous step's grid
        # topology no longer exists: a grid-to-grid (nonconforming) warmstart would read
        # stale cells. Skip those transfers (they only accelerate convergence); the
        # particle/point paths below still apply since they go through the PIC quadrature.
        grid_to_grid_warmstart = not self._sparse_rebuildable

        if isinstance(prev_impulse_field.space.basis, fem.PointBasisSpace):
            # point-based collisions, simply copy the previous impulses
            if scratch.contact_pic is not None:
                scratch.impulse_field.dof_values.zero_()
                wp.launch(
                    transfer_gimp_impulses,
                    dim=self.model.particle_count,
                    inputs=[
                        scratch.gimp_particle_contact_nodes,
                        prev_impulse_field.dof_values,
                        scratch.impulse_field.dof_values,
                        False,
                    ],
                )
            else:
                wp.launch(
                    gather_point_dof_values,
                    dim=scratch.impulse_field.space_partition.node_count(),
                    inputs=[
                        scratch.impulse_field.space_partition.space_node_indices(),
                        pic.cell_particle_indices,
                        prev_impulse_field.dof_values,
                        scratch.impulse_field.dof_values,
                    ],
                )
        elif grid_to_grid_warmstart:
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
        elif last_step_data.impulse_values is not None:
            # Collider nodes keep their positions across in-place rebuilds; match them by lattice key.
            wp.launch(
                load_values_by_lattice_key,
                dim=scratch.collider_node_count,
                inputs=[
                    scratch.collider_position_field.dof_values,
                    scratch.collider_environment_offsets,
                    2.0 / self._mpm_model.voxel_size,
                    last_step_data.impulse_keys,
                    last_step_data.impulse_values,
                    last_step_data.impulse_values.shape[0],
                    scratch.impulse_field.dof_values,
                ],
            )
        else:
            scratch.impulse_field.dof_values.zero_()

        # Interpolate previous stress
        if isinstance(prev_stress_field.space.basis, fem.PointBasisSpace):
            if self.gimp:
                # GIMP has unused subpoint slots with particle index -1. Its
                # point stresses are transferred through the per-overlap cache.
                scratch.stress_field.dof_values.zero_()
            else:
                wp.launch(
                    gather_point_dof_values,
                    dim=scratch.stress_field.space_partition.node_count(),
                    inputs=[
                        scratch.stress_field.space_partition.space_node_indices(),
                        pic.cell_particle_indices,
                        prev_stress_field.dof_values,
                        scratch.stress_field.dof_values,
                    ],
                )
        elif self._stress_warmstart in ("grid", "smoothed") and grid_to_grid_warmstart:
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
        elif not self._sparse_rebuildable:
            pass
        else:
            # No grid-to-grid stress warmstart available for the rebuilt grid; start cold.
            scratch.stress_field.dof_values.zero_()

    def _save_keyed_impulses(self, scratch: ImplicitMPMScratchpad, last_step_data: LastStepData):
        """Save grid collider impulses sorted by lattice key; an in-place rebuild discards their node numbering."""
        node_count = scratch.collider_node_count
        last_step_data.require_keyed_impulses(node_count, self.model.device)
        wp.launch(
            compute_lattice_node_keys,
            dim=node_count,
            inputs=[
                scratch.collider_position_field.dof_values,
                scratch.collider_environment_offsets,
                2.0 / self._mpm_model.voxel_size,
            ],
            outputs=[last_step_data.impulse_keys, last_step_data.impulse_nodes],
        )
        wp.utils.radix_sort_pairs(last_step_data.impulse_keys, last_step_data.impulse_nodes, node_count)
        wp.launch(
            gather_sorted_node_values,
            dim=node_count,
            inputs=[last_step_data.impulse_nodes, scratch.impulse_field.dof_values],
            outputs=[last_step_data.impulse_values],
        )

    def _save_for_next_warmstart(
        self, scratch: ImplicitMPMScratchpad, pic: fem.PicQuadrature, last_step_data: LastStepData
    ):
        with self._timer("Save warmstart fields"):
            if (
                self.gimp
                and self._stress_warmstart
                and (
                    self._stress_warmstart == "particles"
                    or isinstance(scratch.stress_field.space.basis, fem.PointBasisSpace)
                )
            ):
                point_stress, point_valid, point_centers = last_step_data.gimp_stress
                point_valid.zero_()
                fem.interpolate(
                    gimp_store_point_stress,
                    at=pic,
                    fields={"stress": scratch.stress_field},
                    values={
                        "gimp_cells": scratch.particle_cells,
                        "point_stress": point_stress,
                        "point_valid": point_valid,
                        "point_centers": point_centers,
                        "particle_flags": self._mpm_model.material_particle_flags,
                    },
                    temporary_store=self.temporary_store,
                )
            last_step_data.rebind_collision_space_fields(scratch)

            if isinstance(last_step_data.ws_impulse_field.space.basis, fem.PointBasisSpace):
                # point-based collisions, simply copy the previous impulses
                if scratch.contact_pic is not None:
                    wp.launch(
                        transfer_gimp_impulses,
                        dim=self.model.particle_count,
                        inputs=[
                            scratch.gimp_particle_contact_nodes,
                            last_step_data.ws_impulse_field.dof_values,
                            scratch.impulse_field.dof_values,
                            True,
                        ],
                    )
                else:
                    wp.launch(
                        scatter_point_dof_values,
                        dim=scratch.impulse_field.space_partition.node_count(),
                        inputs=[
                            scratch.impulse_field.space_partition.space_node_indices(),
                            pic.cell_particle_indices,
                            scratch.impulse_field.dof_values,
                            last_step_data.ws_impulse_field.dof_values,
                        ],
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
                if self._sparse_rebuildable:
                    self._save_keyed_impulses(scratch, last_step_data)

            last_step_data.rebind_strain_space_fields(scratch, smoothed=self._stress_warmstart == "smoothed")
            if isinstance(last_step_data.ws_stress_field.space.basis, fem.PointBasisSpace):
                if self.gimp:
                    # The per-overlap cache above replaces the particle-indexed
                    # scatter, which cannot represent multiple GIMP subpoints.
                    last_step_data.ws_stress_field.dof_values.zero_()
                else:
                    wp.launch(
                        scatter_point_dof_values,
                        dim=scratch.stress_field.space_partition.node_count(),
                        inputs=[
                            scratch.stress_field.space_partition.space_node_indices(),
                            pic.cell_particle_indices,
                            scratch.stress_field.dof_values,
                            last_step_data.ws_stress_field.dof_values,
                        ],
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
            # Rebuildable sparse grids report capacity-sized node counts, so read
            # discontinuity from the basis name instead of comparing node counts.
            is_dg = _parse_grid_basis_name(self.strain_basis)[2]

        if is_pic or is_dg:
            # cell-based coloring

            # nodes in each element solved sequentially
            stencil_size = int(np.round(np.cbrt(scratch.velocity_nodes_per_element)))
            if isinstance(grid, fem.Nanogrid):
                voxels = grid._cell_ijk
                voxel_envs = grid.cell_env
                env_offsets = grid.env_offsets
                res = wp.vec3i(0)
            else:
                voxels = None
                voxel_envs = None
                env_offsets = None
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
                    voxel_envs,
                    env_offsets,
                    res,
                    colors,
                    color_indices,
                ],
            )

        elif self.strain_basis == "Q1":
            nodes_per_color_element = 1
            stencil_size = 3
            node_env_offsets = None
            env_offsets = None
            if isinstance(grid, fem.Nanogrid):
                voxels = grid._node_ijk
                node_env_offsets = getattr(space_partition, "env_offsets", None)
                env_offsets = grid.env_offsets
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
                    node_env_offsets,
                    env_offsets,
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
                make_dynamic_color_block_indices_kernel(
                    space_partition.geo_partition, space_partition, scratch.pic_domain_element_indices
                ),
                dim=colored_element_count,
                inputs=[
                    partition_arg,
                    space_partition.partition_arg_value(device=scratch.color_indices.device),
                    pic.cell_particle_offsets,
                    space_partition.space_topology.MAX_NODES_PER_ELEMENT,
                    scratch.color_indices,
                ],
            )
        elif is_dg:
            wp.launch(
                make_discontinuous_color_block_indices_kernel(space_partition.geo_partition, space_partition),
                dim=colored_element_count,
                inputs=[
                    partition_arg,
                    space_partition.partition_arg_value(device=scratch.color_indices.device),
                    nodes_per_color_element,
                    scratch.color_indices,
                ],
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


@wp.kernel(enable_backward=False)
def _sync_mpm_proxy_particle_points_kernel(
    collider_particle_offset: int,
    particle_q: wp.array[wp.vec3],
    collider_particle_ids: wp.array[int],
    collider_points: wp.array[wp.vec3],
):
    local_vertex = wp.tid()
    dst_particle = collider_particle_ids[collider_particle_offset + local_vertex]
    collider_points[local_vertex] = particle_q[dst_particle]


@wp.kernel(enable_backward=False)
def _sync_mpm_proxy_particle_velocities_kernel(
    collider_particle_offset: int,
    particle_qd: wp.array[wp.vec3],
    collider_particle_ids: wp.array[int],
    collider_velocities: wp.array[wp.vec3],
):
    local_vertex = wp.tid()
    dst_particle = collider_particle_ids[collider_particle_offset + local_vertex]
    collider_velocities[local_vertex] = particle_qd[dst_particle]


@wp.kernel(enable_backward=False)
def _rewind_mpm_proxy_particles_kernel(
    dt: float,
    particle_local_to_proxy_global: wp.array[int],
    proxy_flag: int,
    active_flag: int,
    particle_flags: wp.array[wp.int32],
    transfer_flags: wp.array[wp.int32],
    particle_gravity_acceleration: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    coupling_forces: wp.array[wp.vec3],
    particle_inv_mass: wp.array[float],
    particle_qd: wp.array[wp.vec3],
):
    local_particle = wp.tid()
    proxy_global = particle_local_to_proxy_global[local_particle]
    if proxy_global < 0 or (particle_flags[local_particle] & proxy_flag) == 0:
        return

    # The source velocity already includes these forces. Gravity and particle_f are
    # applied by the solver on active particles only, and are removed likewise.
    force = coupling_forces[proxy_global]
    is_active = (transfer_flags[local_particle] & active_flag) != 0
    if particle_f and is_active:
        force += particle_f[local_particle]
    delta_v = dt * particle_inv_mass[local_particle] * force
    if is_active:
        delta_v = delta_v + dt * particle_gravity_acceleration[local_particle]

    particle_qd[local_particle] = particle_qd[local_particle] - delta_v


@wp.kernel(enable_backward=False)
def _harvest_mpm_proxy_particle_forces_kernel(
    dt: float,
    collider_ids: wp.array[int],
    collider_impulses: wp.array[wp.vec3],
    collider_impulse_pos: wp.array[wp.vec3],
    collider: Collider,
    particle_local_to_proxy_global: wp.array[int],
    proxy_flag: int,
    particle_flags: wp.array[wp.int32],
    out_particle_f: wp.array[wp.vec3],
):
    i = wp.tid()
    cid = collider_ids[i]

    if cid < 0 or cid + 1 >= collider.collider_particle_offsets.shape[0]:
        return

    vertex_offset = collider.collider_particle_offsets[cid]
    vertex_end = collider.collider_particle_offsets[cid + 1]
    if vertex_end <= vertex_offset:
        return

    mesh = collider.collider_mesh[cid]
    max_dist = collider.query_max_dist + collider.collider_max_thickness[cid]
    query = wp.mesh_query_point_no_sign(mesh, collider_impulse_pos[i], max_dist)
    if not query.result:
        return

    indices = wp.mesh_get(mesh).indices
    tri = query.face
    local_i = indices[3 * tri + 0]
    local_j = indices[3 * tri + 1]
    local_k = indices[3 * tri + 2]

    dst_i = collider.collider_particle_ids[vertex_offset + local_i]
    dst_j = collider.collider_particle_ids[vertex_offset + local_j]
    dst_k = collider.collider_particle_ids[vertex_offset + local_k]

    f = collider_impulses[i] / dt
    w_i = query.u
    w_j = query.v
    w_k = 1.0 - w_i - w_j

    if dst_i >= 0 and dst_i < particle_local_to_proxy_global.shape[0]:
        proxy_global_i = particle_local_to_proxy_global[dst_i]
        if proxy_global_i >= 0 and (particle_flags[dst_i] & proxy_flag) != 0 and w_i > 0.0:
            wp.atomic_add(out_particle_f, proxy_global_i, w_i * f)
    if dst_j >= 0 and dst_j < particle_local_to_proxy_global.shape[0]:
        proxy_global_j = particle_local_to_proxy_global[dst_j]
        if proxy_global_j >= 0 and (particle_flags[dst_j] & proxy_flag) != 0 and w_j > 0.0:
            wp.atomic_add(out_particle_f, proxy_global_j, w_j * f)
    if dst_k >= 0 and dst_k < particle_local_to_proxy_global.shape[0]:
        proxy_global_k = particle_local_to_proxy_global[dst_k]
        if proxy_global_k >= 0 and (particle_flags[dst_k] & proxy_flag) != 0 and w_k > 0.0:
            wp.atomic_add(out_particle_f, proxy_global_k, w_k * f)


@wp.kernel(enable_backward=False)
def _clear_inactive_mpm_proxy_particle_forces_kernel(
    particle_local_to_proxy_global: wp.array[int],
    particle_flags: wp.array[wp.int32],
    proxy_flag: int,
    active_flag: int,
    out_particle_f: wp.array[wp.vec3],
):
    local_particle = wp.tid()
    proxy_global = particle_local_to_proxy_global[local_particle]
    if proxy_global < 0 or proxy_global >= out_particle_f.shape[0]:
        return
    particle_flag = particle_flags[local_particle]
    if (particle_flag & proxy_flag) != 0 and (particle_flag & active_flag) == 0:
        out_particle_f[proxy_global] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def _rewind_mpm_proxy_bodies_kernel(
    dt: float,
    body_local_to_proxy_global: wp.array[int],
    coupling_forces: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_inv_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_qd: wp.array[wp.spatial_vector],
):
    local_body = wp.tid()
    proxy_global = body_local_to_proxy_global[local_body]
    if proxy_global < 0:
        return

    f = coupling_forces[proxy_global]
    delta_v = dt * body_inv_mass[local_body] * wp.spatial_top(f)
    rot = wp.transform_get_rotation(body_q[local_body])
    delta_w = dt * wp.quat_rotate(
        rot,
        body_inv_inertia[local_body] * wp.quat_rotate_inv(rot, wp.spatial_bottom(f)),
    )

    body_qd[local_body] = body_qd[local_body] - wp.spatial_vector(delta_v, delta_w)


@wp.kernel(enable_backward=False)
def _harvest_mpm_proxy_wrenches_kernel(
    dt: float,
    collider_ids: wp.array[int],
    collider_impulses: wp.array[wp.vec3],
    collider_impulse_pos: wp.array[wp.vec3],
    collider_body_ids: wp.array[int],
    body_local_to_proxy_global: wp.array[int],
    proxy_flag: int,
    body_flags: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    out_body_f: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    cid = collider_ids[i]

    if cid < 0 or cid >= collider_body_ids.shape[0]:
        return

    local_body = collider_body_ids[cid]
    if local_body < 0 or local_body >= body_local_to_proxy_global.shape[0]:
        return

    proxy_global = body_local_to_proxy_global[local_body]
    if proxy_global < 0 or proxy_global >= out_body_f.shape[0] or (body_flags[local_body] & proxy_flag) == 0:
        return

    f_world = collider_impulses[i] / dt
    center = wp.transform_point(body_q[local_body], body_com[local_body])
    r = collider_impulse_pos[i] - center
    wp.atomic_add(out_body_f, proxy_global, wp.spatial_vector(f_world, wp.cross(r, f_world)))
