# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Filter native MuJoCo Warp contacts for heterogeneous geometry slots."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from mujoco_warp import Data, Model


@wp.struct
class _ContactField:
    source: wp.array2d[wp.uint32]
    scratch: wp.array2d[wp.uint32]


@wp.kernel
def _prepare_filter(
    nacon: wp.array[int],
    ncollision: wp.array[int],
    capacity: int,
    narrowphase_overflow: int,
    broadphase_overflow: int,
    warn_overflow: bool,
    overflow: wp.array[int],
    original_count: wp.array[int],
    rejected: wp.array[int],
):
    world = wp.tid()
    count = nacon[0]
    broadphase_count = ncollision[0]
    if count > capacity:
        overflow[world] = overflow[world] | narrowphase_overflow
    if broadphase_count > capacity:
        overflow[world] = overflow[world] | broadphase_overflow
    if world == 0:
        original_count[0] = count
        rejected[0] = int(count > capacity)
        if warn_overflow:
            if count > capacity:
                wp.printf("narrowphase overflow before contact filtering: increase naconmax to at least %d\n", count)
            if broadphase_count > capacity:
                wp.printf(
                    "broadphase overflow before contact filtering: increase naconmax to at least %d\n", broadphase_count
                )


@wp.kernel
def _mark_contacts(
    count: wp.array[int],
    geoms: wp.array[wp.vec2i],
    worlds: wp.array[int],
    allowed_pairs: wp.array2d[bool],
    world_to_filter: wp.array[int],
    ngeom: int,
    keep: wp.array[int],
    rejected: wp.array[int],
):
    contact = wp.tid()
    retain = int(0)
    if contact < count[0]:
        pair = geoms[contact]
        first = wp.min(pair[0], pair[1])
        second = wp.max(pair[0], pair[1])
        world = worlds[contact]
        if first >= 0 and second < ngeom and first != second and world >= 0 and world < world_to_filter.shape[0]:
            variant = world_to_filter[world]
            if variant >= 0 and variant < allowed_pairs.shape[0]:
                index = first * (2 * ngeom - first - 1) // 2 + second - first - 1
                retain = int(allowed_pairs[variant, index])
        if retain == 0:
            wp.atomic_max(rejected, 0, 1)
    keep[contact] = retain


@wp.kernel
def _gather_contacts(fields: wp.array[_ContactField], keep: wp.array[int], offsets: wp.array[int]):
    field, contact = wp.tid()
    if keep[contact] != 0:
        buffers = fields[field]
        destination = offsets[contact]
        for word in range(buffers.source.shape[1]):
            buffers.scratch[destination, word] = buffers.source[contact, word]


@wp.kernel
def _restore_contacts(fields: wp.array[_ContactField], keep: wp.array[int], offsets: wp.array[int]):
    field, contact = wp.tid()
    last = keep.shape[0] - 1
    count = offsets[last] + keep[last]
    if contact < count:
        buffers = fields[field]
        for word in range(buffers.source.shape[1]):
            buffers.source[contact, word] = buffers.scratch[contact, word]


@wp.kernel
def _update_count(keep: wp.array[int], offsets: wp.array[int], nacon: wp.array[int]):
    last = keep.shape[0] - 1
    nacon[0] = offsets[last] + keep[last]


class HeterogeneousContactFilter:
    """Stably compact native contacts using per-world allowed geometry pairs.

    MuJoCo Warp invokes ``contactfilter`` after narrowphase and before collision
    wake processing and constraint construction. This callback preserves the
    order and every field of retained contacts, including sensor metadata. Its
    buffers are allocated once and remain valid during CUDA graph replay.

    The native candidate masks must include the union of permitted pairs. This
    callback does not reduce broadphase or narrowphase work, and their buffers
    must accommodate contacts involving inactive internal geometry slots.

    Args:
        data: MuJoCo Warp data whose contact buffers will be filtered.
        allowed_pairs: Boolean lookup rows in upper-triangle order, excluding
            the diagonal. Each row has ``ngeom * (ngeom - 1) // 2`` entries.
        ngeom: Number of geometry slots in the compiled MuJoCo model.
        world_to_filter: Optional lookup-row index for each world, allowing
            worlds with the same filtering rules to share one lookup row.
            Without this map, ``allowed_pairs`` has one row per world.

    Rebuild this object if the data or its contact buffers are reallocated.
    Lookup values may change in place; existing captured graphs observe them.
    """

    def __init__(
        self,
        data: Data,
        allowed_pairs: wp.array2d[bool],
        ngeom: int,
        *,
        world_to_filter: wp.array[int] | None = None,
    ):
        from mujoco_warp import OverflowType

        if isinstance(ngeom, bool) or not isinstance(ngeom, int) or ngeom < 0:
            raise ValueError("ngeom must be a non-negative integer.")
        device = data.nacon.device
        pair_count = ngeom * (ngeom - 1) // 2
        if (
            allowed_pairs.ndim != 2
            or allowed_pairs.dtype != wp.bool
            or allowed_pairs.shape[1] != pair_count
            or allowed_pairs.device != device
        ):
            raise ValueError("allowed_pairs must be a boolean array with one column per geom pair on the data device.")
        if world_to_filter is None:
            if allowed_pairs.shape[0] != data.nworld:
                raise ValueError("allowed_pairs must have one row per world without world_to_filter.")
            world_to_filter = wp.array(list(range(data.nworld)), dtype=int, device=device)
        elif (
            world_to_filter.ndim != 1
            or world_to_filter.dtype != wp.int32
            or world_to_filter.shape[0] != data.nworld
            or world_to_filter.device != device
        ):
            raise ValueError("world_to_filter must contain one int32 row index per world on the data device.")
        else:
            rows = world_to_filter.numpy()
            if (rows < 0).any() or (rows >= allowed_pairs.shape[0]).any():
                raise ValueError("world_to_filter contains an invalid allowed_pairs row.")

        self._data = data
        self._device = device
        self._capacity = data.naconmax
        self._ngeom = ngeom
        self._allowed_pairs = allowed_pairs
        self._world_to_filter = world_to_filter
        self._narrowphase_overflow = int(OverflowType.NARROWPHASE)
        self._broadphase_overflow = int(OverflowType.BROADPHASE)
        self._original_count = wp.zeros(1, dtype=int, device=device)
        self._rejected = wp.zeros(1, dtype=int, device=device)
        self._keep = wp.zeros(self._capacity, dtype=int, device=device)
        self._offsets = wp.empty_like(self._keep)
        self._conditional_capture = device.is_cuda and wp.is_conditional_graph_supported()
        self._field_names = tuple(field.name for field in dataclasses.fields(data.contact))
        self._source_arrays = tuple(getattr(data.contact, name) for name in self._field_names)
        self._buffers = []
        descriptors = []
        for name, source in zip(self._field_names, self._source_arrays, strict=True):
            if source.size == 0:
                continue
            if not source.is_contiguous or source.shape[0] != self._capacity or source.device != device:
                raise ValueError(
                    f"Contact field {name!r} must be contiguous and have naconmax rows on the data device."
                )
            row_bytes = source.strides[0]
            if row_bytes % 4:
                raise ValueError(f"Contact field {name!r} must have rows aligned to 32-bit words.")
            if row_bytes == 0 or self._capacity == 0:
                continue
            # Byte views avoid a separate copy kernel for each scalar/vector
            # type and automatically cover additional Contact dataclass fields.
            words = wp.array(
                ptr=source.ptr,
                dtype=wp.uint32,
                shape=(self._capacity, row_bytes // 4),
                device=device,
            )
            scratch = wp.empty_like(words)
            descriptor = _ContactField()
            descriptor.source = words
            descriptor.scratch = scratch
            descriptors.append(descriptor)
            self._buffers.append((words, scratch))
        self._fields = wp.array(descriptors, dtype=_ContactField, device=device)

    def _compact(self) -> None:
        dim = (self._fields.shape[0], self._capacity)
        wp.launch(_gather_contacts, dim=dim, inputs=[self._fields, self._keep, self._offsets], device=self._device)
        wp.launch(_restore_contacts, dim=dim, inputs=[self._fields, self._keep, self._offsets], device=self._device)
        wp.launch(_update_count, dim=1, inputs=[self._keep, self._offsets, self._data.nacon], device=self._device)

    def __call__(self, model: Model, data: Data) -> None:
        if (
            data is not self._data
            or data.naconmax != self._capacity
            or any(
                getattr(data.contact, name) is not source
                for name, source in zip(self._field_names, self._source_arrays, strict=True)
            )
        ):
            raise ValueError("MuJoCo contact buffers changed; rebuild HeterogeneousContactFilter.")
        wp.launch(
            _prepare_filter,
            dim=data.nworld,
            inputs=[
                data.nacon,
                data.ncollision,
                self._capacity,
                self._narrowphase_overflow,
                self._broadphase_overflow,
                bool(model.opt.warn_overflow),
                data.overflow,
                self._original_count,
                self._rejected,
            ],
            device=self._device,
        )
        if self._capacity == 0:
            data.nacon.zero_()
            return
        wp.launch(
            _mark_contacts,
            dim=self._capacity,
            inputs=[
                self._original_count,
                data.contact.geom,
                data.contact.worldid,
                self._allowed_pairs,
                self._world_to_filter,
                self._ngeom,
                self._keep,
                self._rejected,
            ],
            device=self._device,
        )
        # Warp's scan allocates temporary workspace during capture. CUDA
        # conditional bodies cannot contain allocation nodes, so only the
        # larger contact-field copies are placed behind the conditional.
        wp.utils.array_scan(self._keep, self._offsets, inclusive=False)
        if self._conditional_capture and model.opt.graph_conditional and self._device.is_capturing:
            wp.capture_if(self._rejected, on_true=self._compact)
        else:
            # capture_if outside CUDA capture reads the condition back to the
            # host. Keep eager execution asynchronous instead.
            self._compact()
