# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural Delassus assembly for ALM penalty initialization."""

from __future__ import annotations

import warp as wp

from ...core.types import vec6f
from ...linalg import DenseSquareMultiLinearInfo
from .joint_factorize import make_batched_body_solve_kernel
from .system import BatchedPrimalBodySystem

__all__ = ["BatchedStructuralDelassus"]

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _permute_body_response_rows(
    body_dimensions: wp.array[wp.int32],
    body_vector_offsets: wp.array[wp.int32],
    joint_dimensions: wp.array[wp.int32],
    response_offsets: wp.array[wp.int32],
    response_leading_dimensions: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    component, reordered_row, column = wp.tid()
    if reordered_row >= body_dimensions[component] or column >= joint_dimensions[component]:
        return
    original_row = permutation[body_vector_offsets[component] + reordered_row]
    response_offset = response_offsets[component]
    leading_dimension = response_leading_dimensions[component]
    destination[response_offset + reordered_row * leading_dimension + column] = source[
        response_offset + original_row * leading_dimension + column
    ]


@wp.kernel
def _unpermute_body_response_rows(
    body_dimensions: wp.array[wp.int32],
    body_vector_offsets: wp.array[wp.int32],
    joint_dimensions: wp.array[wp.int32],
    response_offsets: wp.array[wp.int32],
    response_leading_dimensions: wp.array[wp.int32],
    permutation: wp.array[wp.int32],
    source: wp.array[wp.float32],
    destination: wp.array[wp.float32],
):
    component, reordered_row, column = wp.tid()
    if reordered_row >= body_dimensions[component] or column >= joint_dimensions[component]:
        return
    original_row = permutation[body_vector_offsets[component] + reordered_row]
    response_offset = response_offsets[component]
    leading_dimension = response_leading_dimensions[component]
    destination[response_offset + original_row * leading_dimension + column] = source[
        response_offset + reordered_row * leading_dimension + column
    ]


@wp.kernel
def _build_joint_basis_body_right_hand_sides(
    body_dimensions: wp.array[wp.int32],
    joint_dimensions: wp.array[wp.int32],
    joint_vector_offsets: wp.array[wp.int32],
    response_offsets: wp.array[wp.int32],
    response_leading_dimensions: wp.array[wp.int32],
    vector_row: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    body_value: wp.array[wp.float32],
):
    component, row, column = wp.tid()
    if row >= body_dimensions[component] or column >= joint_dimensions[component]:
        return
    value = wp.float32(0.0)
    flat_row = vector_row[joint_vector_offsets[component] + column]
    body = row // 6
    axis = row - 6 * body
    if body_first[flat_row] == body:
        value += jacobian_first[flat_row][axis]
    if body_second[flat_row] == body:
        value += jacobian_second[flat_row][axis]
    body_value[response_offsets[component] + row * response_leading_dimensions[component] + column] = value


@wp.func
def _load_body_response(
    response: wp.array[wp.float32],
    response_offset: wp.int32,
    leading_dimension: wp.int32,
    body: wp.int32,
    column: wp.int32,
) -> vec6f:
    result = vec6f(0.0)
    if body >= 0:
        offset = response_offset + 6 * body * leading_dimension + column
        for axis in range(6):
            result[axis] = response[offset + axis * leading_dimension]
    return result


@wp.kernel
def _assemble_from_body_response(
    joint_dimensions: wp.array[wp.int32],
    joint_matrix_offsets: wp.array[wp.int32],
    joint_vector_offsets: wp.array[wp.int32],
    response_offsets: wp.array[wp.int32],
    response_leading_dimensions: wp.array[wp.int32],
    vector_row: wp.array[wp.int32],
    body_first: wp.array[wp.int32],
    body_second: wp.array[wp.int32],
    jacobian_first: wp.array[vec6f],
    jacobian_second: wp.array[vec6f],
    response: wp.array[wp.float32],
    matrix: wp.array[wp.float32],
):
    component, row, column = wp.tid()
    joint_dimension = joint_dimensions[component]
    if row >= joint_dimension or column >= joint_dimension:
        return
    flat_row = vector_row[joint_vector_offsets[component] + row]
    value = wp.float32(0.0)
    first = body_first[flat_row]
    second = body_second[flat_row]
    response_offset = response_offsets[component]
    leading_dimension = response_leading_dimensions[component]
    if first >= 0:
        value += wp.dot(
            jacobian_first[flat_row],
            _load_body_response(response, response_offset, leading_dimension, first, column),
        )
    if second >= 0:
        value += wp.dot(
            jacobian_second[flat_row],
            _load_body_response(response, response_offset, leading_dimension, second, column),
        )
    matrix[joint_matrix_offsets[component] + row * joint_dimension + column] = value


class BatchedStructuralDelassus:
    """Assemble structural Delassus blocks for ALM penalty initialization."""

    def __init__(
        self,
        body_system: BatchedPrimalBodySystem,
        body_first_global: wp.array[wp.int32],
        body_second_global: wp.array[wp.int32],
        jacobian_first: wp.array[vec6f],
        jacobian_second: wp.array[vec6f],
    ):
        self.body_system = body_system
        self.device = body_system.device
        self.row_count = body_first_global.shape[0]
        self.jacobian_first = jacobian_first
        self.jacobian_second = jacobian_second

        if any(array.shape[0] != self.row_count for array in (body_second_global, jacobian_first, jacobian_second)):
            raise ValueError("Structural row arrays must have identical lengths.")

        first_global = body_first_global.numpy().astype(int).tolist()
        second_global = body_second_global.numpy().astype(int).tolist()
        row_components = [-1] * self.row_count
        body_first_local: list[int] = []
        body_second_local: list[int] = []
        component_row_counts = [0] * body_system.num_blocks
        component_row_local = [-1] * self.row_count
        for row, (first, second) in enumerate(zip(first_global, second_global, strict=True)):
            if first < 0 and second < 0:
                raise ValueError("Each structural row must reference at least one body.")
            if first >= body_system.num_bodies or second >= body_system.num_bodies:
                raise ValueError("Structural row body indices must reference packed bodies.")
            first_local = body_system.body_local_host[first] if first >= 0 else -1
            second_local = body_system.body_local_host[second] if second >= 0 else -1
            body_first_local.append(first_local)
            body_second_local.append(second_local)
            if first_local < 0 and second_local < 0:
                continue
            component = body_system.body_block_host[first] if first_local >= 0 else body_system.body_block_host[second]
            if second_local >= 0 and body_system.body_block_host[second] != component:
                raise ValueError("A structural row cannot connect independent body components.")
            row_components[row] = component
            component_row_local[row] = component_row_counts[component]
            component_row_counts[component] += 1

        storage_dimensions = [max(1, count) for count in component_row_counts]
        vector_offsets = [0]
        for dimension in storage_dimensions:
            vector_offsets.append(vector_offsets[-1] + dimension)
        row_vector_index = [-1] * self.row_count
        for row, (component, local) in enumerate(zip(row_components, component_row_local, strict=True)):
            if component >= 0:
                row_vector_index[row] = vector_offsets[component] + local

        self.component_row_counts = tuple(component_row_counts)
        self.component_world_host = body_system.block_world_host
        self.body_first_local = wp.array(body_first_local, dtype=wp.int32, device=self.device)
        self.body_second_local = wp.array(body_second_local, dtype=wp.int32, device=self.device)

        self.info = DenseSquareMultiLinearInfo()
        self.info.finalize(dimensions=storage_dimensions, dtype=wp.float32, itype=wp.int32, device=self.device)
        self.info.dim = wp.array(component_row_counts, dtype=wp.int32, device=self.device)

        response_leading_dimensions = storage_dimensions
        response_sizes = [
            6 * body_count * joint_dimension
            for body_count, joint_dimension in zip(
                body_system.block_body_counts, response_leading_dimensions, strict=True
            )
        ]
        response_offsets = [0]
        for size in response_sizes:
            response_offsets.append(response_offsets[-1] + size)
        self.response_offsets = wp.array(response_offsets[:-1], dtype=wp.int32, device=self.device)
        self.response_leading_dimensions = wp.array(response_leading_dimensions, dtype=wp.int32, device=self.device)
        self.body_response = wp.zeros(response_offsets[-1], dtype=wp.float32, device=self.device)
        self.body_response_intermediate = wp.zeros_like(self.body_response)
        self.body_response_permuted = (
            wp.zeros_like(self.body_response) if body_system.linear_solver.uses_reordering else self.body_response
        )
        vector_row = [-1] * self.info.total_vec_size
        for row, vector_index in enumerate(row_vector_index):
            if vector_index >= 0:
                vector_row[vector_index] = row
        self.vector_row = wp.array(vector_row, dtype=wp.int32, device=self.device)
        self.matrix = wp.zeros(self.info.total_mat_size, dtype=wp.float32, device=self.device)

        right_hand_side_tile_size = 4
        self._right_hand_side_block_count = (
            self.info.max_dimension + right_hand_side_tile_size - 1
        ) // right_hand_side_tile_size
        self._body_solve_kernel = make_batched_body_solve_kernel(
            body_system.linear_solver.block_size,
            right_hand_side_tile_size,
        )

    def assemble(self) -> None:
        """Assemble the structural Delassus from the factored body system."""
        wp.launch(
            _build_joint_basis_body_right_hand_sides,
            dim=(self.body_system.num_blocks, self.body_system.info.max_dimension, self.info.max_dimension),
            inputs=[
                self.body_system.info.dim,
                self.info.dim,
                self.info.vio,
                self.response_offsets,
                self.response_leading_dimensions,
                self.vector_row,
                self.body_first_local,
                self.body_second_local,
                self.jacobian_first,
                self.jacobian_second,
            ],
            outputs=[self.body_response],
            device=self.device,
        )
        if self.body_response_permuted is not self.body_response:
            wp.launch(
                _permute_body_response_rows,
                dim=(self.body_system.num_blocks, self.body_system.info.max_dimension, self.info.max_dimension),
                inputs=[
                    self.body_system.info.dim,
                    self.body_system.info.vio,
                    self.info.dim,
                    self.response_offsets,
                    self.response_leading_dimensions,
                    self.body_system.linear_solver.P,
                    self.body_response,
                ],
                outputs=[self.body_response_permuted],
                device=self.device,
            )
        wp.launch_tiled(
            self._body_solve_kernel,
            dim=(self.body_system.num_blocks, self._right_hand_side_block_count),
            block_dim=128,
            inputs=[
                self.body_system.info.dim,
                self.body_system.info.mio,
                self.info.dim,
                self.response_offsets,
                self.response_leading_dimensions,
                self.body_system.linear_solver.L,
                self.body_response_permuted,
            ],
            outputs=[self.body_response_intermediate],
            device=self.device,
        )
        if self.body_response_permuted is not self.body_response:
            wp.launch(
                _unpermute_body_response_rows,
                dim=(self.body_system.num_blocks, self.body_system.info.max_dimension, self.info.max_dimension),
                inputs=[
                    self.body_system.info.dim,
                    self.body_system.info.vio,
                    self.info.dim,
                    self.response_offsets,
                    self.response_leading_dimensions,
                    self.body_system.linear_solver.P,
                    self.body_response_permuted,
                ],
                outputs=[self.body_response],
                device=self.device,
            )
        wp.launch(
            _assemble_from_body_response,
            dim=(self.body_system.num_blocks, self.info.max_dimension, self.info.max_dimension),
            inputs=[
                self.info.dim,
                self.info.mio,
                self.info.vio,
                self.response_offsets,
                self.response_leading_dimensions,
                self.vector_row,
                self.body_first_local,
                self.body_second_local,
                self.jacobian_first,
                self.jacobian_second,
                self.body_response,
            ],
            outputs=[self.matrix],
            device=self.device,
        )
