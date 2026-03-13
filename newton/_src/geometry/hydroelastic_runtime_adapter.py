"""Runtime extraction helpers for :class:`HydroelasticSDF` model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..sim.model import Model
from .flags import HydroelasticContactWorkflow, HydroelasticType, ShapeFlags, hydroelastic_type_from_flags
from .hydroelastic_pressure_fields import PressureFieldData, build_immutable_pressure_field_from_sdf
from .types import GeoType


@dataclass(frozen=True)
class HydroelasticRuntimeData:
    """Container for prebuilt hydroelastic arrays used by :class:`HydroelasticSDF`."""

    num_hydroelastic_pairs: int
    total_num_tiles: int
    max_num_blocks_per_shape: int
    shape_sdf_shape2blocks: np.ndarray
    shape_contact_workflow: np.ndarray
    shape_pressure_index: np.ndarray
    compact_pressure_field_data: list[PressureFieldData]
    pressure_field_volume: list[wp.Volume]


def _read_shape_workflow(model: Model, shape_count: int) -> np.ndarray:
    """Read per-shape pressure workflow settings with legacy defaults."""
    if hasattr(model, "shape_hydroelastic_contact_workflow") and model.shape_hydroelastic_contact_workflow is not None:
        return model.shape_hydroelastic_contact_workflow.numpy().astype(np.int32)
    return np.full(shape_count, int(HydroelasticContactWorkflow.CLASSIC), dtype=np.int32)


def _read_shape_pressure_sines(model: Model, shape_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read per-shape sine triplets with safe fallback arrays."""
    if hasattr(model, "shape_hydro_pressure_sine_amplitude") and model.shape_hydro_pressure_sine_amplitude is not None:
        amp = model.shape_hydro_pressure_sine_amplitude.numpy().astype(np.float32)
    else:
        amp = np.zeros((shape_count, 3), dtype=np.float32)

    if hasattr(model, "shape_hydro_pressure_sine_cycles") and model.shape_hydro_pressure_sine_cycles is not None:
        cycles = model.shape_hydro_pressure_sine_cycles.numpy().astype(np.float32)
    else:
        cycles = np.ones((shape_count, 3), dtype=np.float32)

    if hasattr(model, "shape_hydro_pressure_sine_phase") and model.shape_hydro_pressure_sine_phase is not None:
        phase = model.shape_hydro_pressure_sine_phase.numpy().astype(np.float32)
    else:
        phase = np.zeros((shape_count, 3), dtype=np.float32)

    return amp, cycles, phase


def _collect_hydroelastic_indices(shape_flags: np.ndarray, shape_hydro_mode: np.ndarray) -> list[int]:
    """Collect shape indices eligible for hydroelastic contact.

    Collidable shapes are filtered by the ``COLLIDE_SHAPES`` flag and must have a
    non-NONE hydroelastic mode.
    """
    return [
        idx
        for idx in range(len(shape_flags))
        if (int(shape_flags[idx]) & ShapeFlags.COLLIDE_SHAPES)
        and (shape_hydro_mode[idx] != int(HydroelasticType.NONE))
    ]


def _validate_hydroelastic_shape_requirements(
    model: Model,
    hydroelastic_indices: list[int],
    shape_hydro_mode: np.ndarray,
    shape_flags: np.ndarray,
    shape_sdf_index: np.ndarray,
    sdf_data: np.ndarray,
) -> None:
    """Validate SDF constraints for hydroelastic shapes."""
    shape_type = model.shape_type.numpy()
    shape_scale = model.shape_scale.numpy()

    for idx in hydroelastic_indices:
        mode = shape_hydro_mode[idx]
        stype = int(shape_type[idx])
        is_rigid_terrain = mode == int(HydroelasticType.RIGID) and stype in (
            int(GeoType.PLANE),
            int(GeoType.HFIELD),
        )
        if is_rigid_terrain:
            continue

        sdf_idx = int(shape_sdf_index[idx])
        if sdf_idx < 0:
            raise ValueError(f"Hydroelastic shape {idx} requires SDF data but has no attached/generated SDF.")
        if not sdf_data[sdf_idx]["scale_baked"]:
            sx, sy, sz = shape_scale[idx]
            if not (np.isclose(sx, 1.0) and np.isclose(sy, 1.0) and np.isclose(sz, 1.0)):
                raise ValueError(
                    f"Hydroelastic shape {idx} uses non-unit scale but its SDF is not scale-baked. "
                    "Build a scale-baked SDF for hydroelastic use."
                )


def _compute_shape_pair_count(shape_pairs: np.ndarray, shape_hydro_mode: np.ndarray) -> int:
    """Count hydroelastic pairs where at least one side is compliant."""
    num_hydroelastic_pairs = 0
    for shape_a, shape_b in shape_pairs:
        mode_a = shape_hydro_mode[int(shape_a)]
        mode_b = shape_hydro_mode[int(shape_b)]
        both_hydro = mode_a != int(HydroelasticType.NONE) and mode_b != int(HydroelasticType.NONE)
        has_compliant = mode_a == int(HydroelasticType.COMPLIANT) or mode_b == int(HydroelasticType.COMPLIANT)
        if both_hydro and has_compliant:
            num_hydroelastic_pairs += 1
    return int(num_hydroelastic_pairs)


def _compute_shape_block_stats(
    model: Model,
    hydroelastic_indices: list[int],
    shape_sdf_index: np.ndarray,
    sdf_index2blocks: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Compute per-shape SDF block span and aggregate tile counts."""
    shape_sdf_shape2blocks = np.zeros((model.shape_count, 2), dtype=np.int32)
    for shape_idx in range(model.shape_count):
        sdf_idx = int(shape_sdf_index[shape_idx])
        if sdf_idx >= 0 and sdf_idx < len(sdf_index2blocks):
            shape_sdf_shape2blocks[shape_idx] = sdf_index2blocks[sdf_idx]

    total_num_tiles = 0
    max_num_blocks_per_shape = 0
    for idx in hydroelastic_indices:
        start_block, end_block = shape_sdf_shape2blocks[idx]
        num_blocks = int(end_block - start_block)
        total_num_tiles += num_blocks
        max_num_blocks_per_shape = max(max_num_blocks_per_shape, num_blocks)

    return shape_sdf_shape2blocks, int(total_num_tiles), int(max_num_blocks_per_shape)


def _collect_shape_sdf_metadata(model: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect per-SDF shape metadata for pressure-field reconstruction."""
    shape_count = model.shape_count
    shape_type = model.shape_type.numpy()
    shape_scale = model.shape_scale.numpy()
    shape_source_ptr = model.shape_source_ptr.numpy()
    shape_sdf_index = model.shape_sdf_index.numpy()
    num_sdfs = len(model.sdf_volume)

    sdf_shape_type = np.full(num_sdfs, -1, dtype=np.int32)
    sdf_shape_scale = np.ones((num_sdfs, 3), dtype=np.float32)
    sdf_shape_source_ptr = np.zeros(num_sdfs, dtype=np.uint64)

    for shape_idx in range(shape_count):
        sdf_idx = int(shape_sdf_index[shape_idx])
        if sdf_idx < 0 or sdf_idx >= num_sdfs:
            continue
        if sdf_shape_type[sdf_idx] != -1:
            continue
        sdf_shape_type[sdf_idx] = int(shape_type[shape_idx])
        sdf_shape_scale[sdf_idx] = np.asarray(shape_scale[shape_idx], dtype=np.float32)
        sdf_shape_source_ptr[sdf_idx] = np.uint64(shape_source_ptr[shape_idx])

    return sdf_shape_type, sdf_shape_scale, sdf_shape_source_ptr


def _build_pressure_tables(
    model: Model,
    shape_hydro_mode: np.ndarray,
    shape_pressure_sine_amp: np.ndarray,
    shape_pressure_sine_cycles: np.ndarray,
    shape_pressure_sine_phase: np.ndarray,
    shape_contact_workflow: np.ndarray,
    hydroelastic_indices: list[int],
) -> tuple[np.ndarray, list[PressureFieldData], list[wp.Volume]]:
    """Build deduplicated immutable pressure fields for compliant shapes."""
    shape_pressure_index = np.full(model.shape_count, -1, dtype=np.int32)
    compact_pressure_field_data: list[PressureFieldData] = []
    pressure_field_volume: list[wp.Volume] = []
    pressure_profile_cache: dict[tuple[Any, ...], int] = {}

    sdf_data = model.sdf_data.numpy()
    sdf_shape_type, sdf_shape_scale, sdf_shape_source_ptr = _collect_shape_sdf_metadata(model)
    shape_sdf_index = model.shape_sdf_index.numpy()
    num_sdfs = len(model.sdf_volume)

    for shape_idx in hydroelastic_indices:
        if shape_hydro_mode[shape_idx] != int(HydroelasticType.COMPLIANT):
            continue

        sdf_idx = int(shape_sdf_index[shape_idx])
        if sdf_idx < 0 or sdf_idx >= num_sdfs:
            continue

        amp = np.asarray(shape_pressure_sine_amp[shape_idx], dtype=np.float32)
        cyc = np.asarray(shape_pressure_sine_cycles[shape_idx], dtype=np.float32)
        phs = np.asarray(shape_pressure_sine_phase[shape_idx], dtype=np.float32)
        workflow = int(shape_contact_workflow[shape_idx])

        if workflow != int(HydroelasticContactWorkflow.PRESSURE):
            amp = np.zeros(3, dtype=np.float32)
            cyc = np.ones(3, dtype=np.float32)
            phs = np.zeros(3, dtype=np.float32)

        if np.all(np.abs(amp) <= 1.0e-8):
            amp = np.zeros(3, dtype=np.float32)
            cyc = np.ones(3, dtype=np.float32)
            phs = np.zeros(3, dtype=np.float32)

        key = (
            int(sdf_idx),
            float(amp[0]),
            float(amp[1]),
            float(amp[2]),
            float(cyc[0]),
            float(cyc[1]),
            float(cyc[2]),
            float(phs[0]),
            float(phs[1]),
            float(phs[2]),
        )
        pressure_idx = pressure_profile_cache.get(key)
        if pressure_idx is None:
            pressure_shape_type: int | None = None
            pressure_shape_scale: np.ndarray | None = None
            pressure_shape_source_ptr: int | None = None
            if sdf_shape_type[sdf_idx] >= 0:
                pressure_shape_type = int(sdf_shape_type[sdf_idx])
                pressure_shape_scale = sdf_shape_scale[sdf_idx]
                pressure_shape_source_ptr = int(sdf_shape_source_ptr[sdf_idx])

            pressure_data, pressure_volume = build_immutable_pressure_field_from_sdf(
                model.sdf_data,
                sdf_data[sdf_idx],
                sdf_idx,
                model.device,
                shape_type=pressure_shape_type,
                shape_scale=pressure_shape_scale,
                shape_source_ptr=pressure_shape_source_ptr,
                pressure_sine_amplitude=amp,
                pressure_sine_cycles=cyc,
                pressure_sine_phase=phs,
            )
            pressure_idx = len(compact_pressure_field_data)
            pressure_profile_cache[key] = pressure_idx
            compact_pressure_field_data.append(pressure_data)
            if pressure_volume is not None:
                pressure_field_volume.append(pressure_volume)

        shape_pressure_index[shape_idx] = pressure_idx

    return shape_pressure_index, compact_pressure_field_data, pressure_field_volume


def collect_hydroelastic_runtime_data(model: Model) -> HydroelasticRuntimeData | None:
    """Prepare hydroelastic runtime arrays from a finalized model."""
    shape_flags = model.shape_flags.numpy()
    shape_hydro_mode = np.array([hydroelastic_type_from_flags(int(flags)) for flags in shape_flags], dtype=np.int32)
    if not np.any(shape_hydro_mode != int(HydroelasticType.NONE)):
        return None

    shape_pairs = model.shape_contact_pairs.numpy()
    num_hydroelastic_pairs = _compute_shape_pair_count(shape_pairs, shape_hydro_mode)
    if num_hydroelastic_pairs == 0:
        return None

    shape_sdf_index = model.shape_sdf_index.numpy()
    sdf_index2blocks = model.sdf_index2blocks.numpy()
    sdf_data = model.sdf_data.numpy()
    if np.ndim(sdf_index2blocks) != 2 or sdf_index2blocks.shape[1] != 2:
        raise ValueError("model.sdf_index2blocks must have shape [num_sdfs, 2].")

    shape_contact_workflow = _read_shape_workflow(model, model.shape_count)
    shape_pressure_sine_amp, shape_pressure_sine_cycles, shape_pressure_sine_phase = _read_shape_pressure_sines(
        model, model.shape_count
    )
    hydroelastic_indices = _collect_hydroelastic_indices(shape_flags, shape_hydro_mode)

    _validate_hydroelastic_shape_requirements(
        model,
        hydroelastic_indices,
        shape_hydro_mode,
        shape_flags,
        shape_sdf_index,
        sdf_data,
    )

    shape_sdf_shape2blocks, total_num_tiles, max_num_blocks_per_shape = _compute_shape_block_stats(
        model,
        hydroelastic_indices,
        shape_sdf_index,
        sdf_index2blocks,
    )

    shape_pressure_index, compact_pressure_field_data, pressure_field_volume = _build_pressure_tables(
        model,
        shape_hydro_mode,
        shape_pressure_sine_amp,
        shape_pressure_sine_cycles,
        shape_pressure_sine_phase,
        shape_contact_workflow,
        hydroelastic_indices,
    )

    return HydroelasticRuntimeData(
        num_hydroelastic_pairs=num_hydroelastic_pairs,
        total_num_tiles=total_num_tiles,
        max_num_blocks_per_shape=max_num_blocks_per_shape,
        shape_sdf_shape2blocks=shape_sdf_shape2blocks,
        shape_contact_workflow=shape_contact_workflow,
        shape_pressure_index=shape_pressure_index,
        compact_pressure_field_data=compact_pressure_field_data,
        pressure_field_volume=pressure_field_volume,
    )
