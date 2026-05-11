# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Digital Instron
#
# Build averaged physical Instron traces and compare them to a quasi-static
# hydroelastic pressure-field sweep of a shoe midsole.
#
# Commands:
#   uv run --extra examples -m newton.examples digital_instron --mode preprocess
#   uv run --extra examples -m newton.examples digital_instron --mode run --test-case rearfoot --viewer null
#
###########################################################################

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticContactWorkflow, HydroelasticSDF, HydroelasticType
from projects.digital_instron.diagnostics import (
    force_jump_metrics as _force_jump_metrics,
)
from projects.digital_instron.diagnostics import (
    residual_motion_metrics as _residual_motion_metrics,
)
from projects.digital_instron.objectives import (
    force_fit_objective as _force_fit_objective,
)
from projects.digital_instron.objectives import (
    loop_metrics as _loop_metrics,
)
from projects.digital_instron.objectives import (
    mean_numeric_components as _mean_numeric_components,
)
from projects.digital_instron.objectives import (
    shared_material_objective as _shared_material_objective,
)
from projects.digital_instron.objectives import (
    shared_material_objective_breakdown as _shared_material_objective_breakdown,
)
from projects.digital_instron.objectives import (
    trapz as _trapz,
)
from projects.digital_instron.objectives import (
    trial_objective_components as _trial_objective_components,
)

logger = logging.getLogger(__name__)

# Keep the old private helper names available while the tests and callers move
# to the extracted modules.
_COMPAT_OBJECTIVE_HELPERS = (
    _mean_numeric_components,
    _shared_material_objective,
    _trapz,
    _trial_objective_components,
)

DATA_DIR = Path("DigitalInstron")
FULLFOOT_CSV = DATA_DIR / "04-07-2025_FR3_185ms_Fullfoot_100Cycle.steps.tracking.csv"
REARFOOT_CSV = DATA_DIR / "04-07-2025_FR3_140ms_Rearfoot_100Cycle.steps.tracking.csv"
MIDSOLE_OBJ = DATA_DIR / "puma-fast-r-nitro-elite-3-3d-internal-wt-LR.obj"
FULLFOOT_EFFECTOR_STL = DATA_DIR / "Instron Shoe Last Size 9 6drop merged attachment 1.STL"
PROCESSED_DIR = DATA_DIR / "processed"

CSV_TIME = "Cycle Elapsed Time (s)"
CSV_CYCLE = "Total Cycles"
CSV_POSITION = "Position (mm)"
CSV_FORCE = "Force (N)"
CSV_ENERGY = "Cycle Energy(Energy Calculation) (J)"
CSV_VELOCITY = "Velocity(Velocity Calculation) (m/s)"


@dataclass(frozen=True)
class TraceSummary:
    path: str
    output_path: str
    cycle_start: int
    cycle_end: int
    cycles: list[int]
    samples: int
    displacement_peak_mm: float
    force_peak_n: float
    force_min_n: float
    measured_loop_area_j: float


@dataclass
class TrialResponse:
    test_case: str
    trace_csv: str
    output_csv: str
    trace: dict[str, np.ndarray]
    displacement_m: np.ndarray
    time_s: np.ndarray
    displacement_velocity_m_s: np.ndarray
    measured_force_n: np.ndarray
    raw_contact_force_n: np.ndarray
    contact_stats: list[dict[str, float | int]]
    midsole_stats: dict[str, Any]


@dataclass
class TrialScene:
    test_case: str
    trace_csv: str
    output_csv: str
    trace: dict[str, np.ndarray]
    displacement_m: np.ndarray
    time_s: np.ndarray
    displacement_velocity_m_s: np.ndarray
    measured_force_n: np.ndarray
    model: Any
    state: Any
    state_next: Any
    pipeline: Any
    contacts: Any
    solver: Any
    control: Any
    shoe_body: int
    shoe_anchor_xy: tuple[float, float]
    shoe_anchor_quat: tuple[float, float, float, float]
    settled_rotation_locked: bool
    indenter_body: int
    indenter_shape: int
    midsole_shape: int
    base_shape: int
    base_top_z: float
    indenter_stop_z: float
    indenter_rest_z: float
    midsole_stats: dict[str, Any]
    candidate_index: int = 0


@dataclass(frozen=True)
class MaterialCandidate:
    kh: float
    kd: float
    layer_eta_lock: float
    layer_densification_power: float


@dataclass(frozen=True)
class CachedTrialSetup:
    midsole_mesh: newton.Mesh
    midsole_stats: dict[str, Any]
    midsole_vertices: np.ndarray
    midsole_extents: np.ndarray
    midsole_top: float
    rearfoot_local_top_z: float | None
    rearfoot_local_vertex_count: int
    heel_x: float
    indenter_mesh: newton.Mesh | None
    indenter_stats: dict[str, Any] | None
    traces: dict[str, dict[str, Any]]
    gravity: float


@dataclass
class MaterialBatchTrialResponses:
    responses_by_candidate: list[list[TrialResponse]]
    invalid_candidates: dict[int, dict[str, Any]]


@dataclass
class MaterialSearchResult:
    material: dict[str, Any]
    responses: list[TrialResponse]
    candidate_count: int
    history: list[dict[str, Any]]


@wp.kernel
def _set_body_translation_z(body_q: wp.array[wp.transform], body_idx: int, z: float):
    q = body_q[body_idx]
    p = wp.transform_get_translation(q)
    body_q[body_idx] = wp.transform(wp.vec3(p[0], p[1], z), wp.transform_get_rotation(q))


def _set_body_z_velocity(state, body_idx: int, z: float, linear_z: float = 0.0) -> None:
    _set_body_z(state, body_idx, z)
    body_qd = state.body_qd.numpy()
    body_qd[body_idx] = 0.0
    body_qd[body_idx, 2] = float(linear_z)
    state.body_qd.assign(body_qd)


def _stabilize_shoe_fixture(scene: TrialScene, args, *, zero_vertical_velocity: bool = False) -> None:
    if not bool(args.fixture_lock_shoe_horizontal):
        return

    free_axis = "none" if scene.settled_rotation_locked else str(args.fixture_free_rotation_axis)
    axis_index = {"x": 0, "y": 1, "z": 2}.get(free_axis)
    body_q = scene.state.body_q.numpy()
    shoe_q = body_q[scene.shoe_body].copy()
    shoe_q[0] = float(scene.shoe_anchor_xy[0])
    shoe_q[1] = float(scene.shoe_anchor_xy[1])
    if axis_index is None:
        shoe_q[3:7] = np.asarray(scene.shoe_anchor_quat, dtype=shoe_q.dtype)
    else:
        rotation = np.zeros(4, dtype=shoe_q.dtype)
        rotation[axis_index] = shoe_q[3 + axis_index]
        rotation[3] = shoe_q[6]
        norm = float(np.linalg.norm(rotation))
        shoe_q[3:7] = rotation / norm if norm > 1.0e-12 else np.asarray(scene.shoe_anchor_quat, dtype=shoe_q.dtype)
    body_q[scene.shoe_body] = shoe_q
    scene.state.body_q.assign(body_q)

    body_qd = scene.state.body_qd.numpy()
    body_qd[scene.shoe_body, 0] = 0.0
    body_qd[scene.shoe_body, 1] = 0.0
    angular_velocity = body_qd[scene.shoe_body, 3:].copy()
    body_qd[scene.shoe_body, 3:] = 0.0
    if axis_index is not None and not zero_vertical_velocity:
        body_qd[scene.shoe_body, 3 + axis_index] = angular_velocity[axis_index] * float(
            np.clip(args.fixture_shoe_angular_damping, 0.0, 1.0)
        )
    if zero_vertical_velocity:
        body_qd[scene.shoe_body, 2] = 0.0
    else:
        body_qd[scene.shoe_body, 2] *= float(np.clip(args.fixture_shoe_velocity_damping, 0.0, 1.0))
    scene.state.body_qd.assign(body_qd)


def _shoe_motion_stats(scene: TrialScene, args) -> dict[str, float]:
    body_qd = scene.state.body_qd.numpy()
    shoe_qd = body_qd[scene.shoe_body]
    angular_velocity = np.asarray(shoe_qd[3:], dtype=np.float64)
    free_axis = "none" if scene.settled_rotation_locked else str(args.fixture_free_rotation_axis)
    axis_index = {"x": 0, "y": 1, "z": 2}.get(free_axis)
    free_axis_velocity = float(angular_velocity[axis_index]) if axis_index is not None else 0.0
    return {
        "shoe_vertical_velocity_m_s": float(shoe_qd[2]),
        "shoe_abs_vertical_velocity_m_s": float(abs(shoe_qd[2])),
        "shoe_angular_velocity_x_rad_s": float(angular_velocity[0]),
        "shoe_angular_velocity_y_rad_s": float(angular_velocity[1]),
        "shoe_angular_velocity_z_rad_s": float(angular_velocity[2]),
        "shoe_angular_velocity_norm_rad_s": float(np.linalg.norm(angular_velocity)),
        "shoe_free_axis_angular_velocity_rad_s": free_axis_velocity,
        "shoe_abs_free_axis_angular_velocity_rad_s": float(abs(free_axis_velocity)),
    }


def _body_points_to_world(points: np.ndarray, body_ids: np.ndarray, body_q: np.ndarray) -> np.ndarray:
    world = np.asarray(points, dtype=np.float64).copy()
    for body_value in np.unique(body_ids):
        body = int(body_value)
        if body < 0:
            continue
        mask = body_ids == body
        pos = body_q[body, :3].astype(np.float64)
        quat = body_q[body, 3:7].astype(np.float64)
        qv = quat[:3]
        qw = quat[3]
        v = world[mask]
        world[mask] = v + 2.0 * np.cross(qv, np.cross(qv, v) + qw * v) + pos
    return world


def _scene_solver_body_indices(scene: TrialScene) -> list[int]:
    return [
        int(body_idx)
        for body_idx in (getattr(scene, "shoe_body", -1), getattr(scene, "indenter_body", -1))
        if int(body_idx) >= 0
    ]


def _check_solver_state_finite(state, scenes: list[TrialScene]) -> None:
    """Raise RuntimeError if solver produced non-finite state values."""
    body_q = state.body_q.numpy()
    divergent_scenes = [scene for scene in scenes if not np.isfinite(body_q[_scene_solver_body_indices(scene)]).all()]
    if not divergent_scenes:
        return
    scene_info = " / ".join(
        f"{scene.test_case} candidate {getattr(scene, 'candidate_index', 'unknown')}" for scene in divergent_scenes
    )
    raise RuntimeError(
        f"Solver diverged: non-finite body positions detected. {scene_info}. "
        "This usually means the material stiffness (kh) exceeds the solver's "
        "stability limit for the current timestep. Try reducing kh or increasing substeps."
    )


def _runtime_error_sample_index(message: str) -> int | None:
    match = re.search(r"(?:at sample|sample) (\d+)", message)
    return int(match.group(1)) if match is not None else None


def _candidate_invalid_reason(message: str) -> str:
    if "Solver diverged" in message:
        return "solver diverged"
    if "no raw pressure-field contact force" in message:
        return "zero contact"
    if "adjacent force jump" in message or "adjacent force jumps" in message:
        return "adjacent force jumps"
    return message


def _invalid_material_candidate(
    candidate: MaterialCandidate,
    invalid_reason: str,
    *,
    divergence_sample_index: int | None = None,
) -> dict[str, Any]:
    material = {
        "kh": float(candidate.kh),
        "kd": float(candidate.kd),
        "layer_eta_lock": float(candidate.layer_eta_lock),
        "layer_densification_power": float(candidate.layer_densification_power),
        "combined_objective": float("inf"),
        "invalid_reason": invalid_reason,
    }
    if divergence_sample_index is not None:
        material["divergence_sample_index"] = int(divergence_sample_index)
    return material


def _invalid_candidate_history_entry(candidate_material: dict[str, Any]) -> dict[str, float | int | str]:
    entry: dict[str, float | int | str] = {
        "kh": float(candidate_material["kh"]),
        "kd": float(candidate_material["kd"]),
        "invalid_reason": str(candidate_material.get("invalid_reason", "")),
    }
    if "divergence_sample_index" in candidate_material:
        entry["divergence_sample_index"] = int(candidate_material["divergence_sample_index"])
    return entry


def _read_instron_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _cycles_in_rows(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(float(row[CSV_CYCLE])) for row in rows if row.get(CSV_CYCLE)})


def _interp_cycle(rows: list[dict[str, str]], phase: np.ndarray) -> dict[str, np.ndarray]:
    time = np.asarray([float(row[CSV_TIME]) for row in rows], dtype=np.float64)
    duration = float(np.max(time))
    if duration <= 0.0:
        raise ValueError("Cycle duration must be positive.")

    order = np.argsort(time)
    time = time[order]
    phase_in = time / duration

    out: dict[str, np.ndarray] = {"time_s": phase * duration}
    for name, column in (
        ("position_mm_raw", CSV_POSITION),
        ("force_n_raw", CSV_FORCE),
        ("cycle_energy_j", CSV_ENERGY),
        ("velocity_m_s", CSV_VELOCITY),
    ):
        values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)[order]
        out[name] = np.interp(phase, phase_in, values)
    return out


def average_instron_cycles(
    path: Path,
    output_path: Path,
    *,
    cycle_start: int = 90,
    cycle_end: int = 100,
    samples: int = 501,
) -> TraceSummary:
    rows = _read_instron_rows(path)
    by_cycle: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        cycle = int(float(row[CSV_CYCLE]))
        if cycle_start <= cycle <= cycle_end:
            by_cycle.setdefault(cycle, []).append(row)

    cycles = sorted(by_cycle)
    if not cycles:
        raise ValueError(f"No cycles in [{cycle_start}, {cycle_end}] found in {path}.")

    phase = np.linspace(0.0, 1.0, samples)
    interpolated = [_interp_cycle(by_cycle[cycle], phase) for cycle in cycles]

    averaged: dict[str, np.ndarray] = {"phase": phase}
    for key in interpolated[0]:
        averaged[key] = np.mean([cycle[key] for cycle in interpolated], axis=0)

    # Instron exported compression as more-negative position and force for these files.
    position = averaged["position_mm_raw"]
    force_raw = averaged["force_n_raw"]
    displacement_mm = np.max(position) - position
    force_positive = np.maximum(-force_raw, 0.0)
    averaged["displacement_mm"] = displacement_mm
    averaged["displacement_m"] = displacement_mm * 0.001
    averaged["force_n"] = force_positive

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "time_s",
        "displacement_m",
        "displacement_mm",
        "force_n",
        "position_mm_raw",
        "force_n_raw",
        "velocity_m_s",
        "cycle_energy_j",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(samples):
            writer.writerow({field: f"{float(averaged[field][i]):.12g}" for field in fieldnames})

    loop_area_j = float(abs(np.trapezoid(force_positive, displacement_mm * 0.001)))
    summary = TraceSummary(
        path=str(path),
        output_path=str(output_path),
        cycle_start=cycle_start,
        cycle_end=cycle_end,
        cycles=cycles,
        samples=samples,
        displacement_peak_mm=float(np.max(displacement_mm)),
        force_peak_n=float(np.max(force_positive)),
        force_min_n=float(np.min(force_positive)),
        measured_loop_area_j=loop_area_j,
    )

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary.__dict__, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_averaged_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {key: np.asarray([float(row[key]) for row in rows], dtype=np.float64) for key in rows[0]}


def _sample_indices_for_stride(displacement_m: np.ndarray, stride: int) -> np.ndarray:
    stride = max(int(stride), 1)
    if stride == 1 or displacement_m.size <= 2:
        return np.arange(displacement_m.size, dtype=np.int64)
    peak_index = int(np.argmax(displacement_m))
    indices = set(range(0, int(displacement_m.size), stride))
    indices.update({0, peak_index, int(displacement_m.size) - 1})
    return np.asarray(sorted(indices), dtype=np.int64)


def _stride_trace(trace: dict[str, np.ndarray], stride: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    indices = _sample_indices_for_stride(trace["displacement_m"], stride)
    if indices.size == trace["displacement_m"].size:
        return trace, indices
    return {key: value[indices] for key, value in trace.items()}, indices


def _load_trimesh_module():
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("digital_instron requires trimesh. Run with `uv run --extra examples`.") from exc
    return trimesh


def _load_single_trimesh(path: Path):
    trimesh = _load_trimesh_module()
    tri = trimesh.load(str(path), force="mesh", process=False)
    if hasattr(tri, "geometry"):
        geometries = []
        for geom in tri.geometry.values():
            vertices = np.asarray(getattr(geom, "vertices", []))
            faces = np.asarray(getattr(geom, "faces", []))
            if vertices.size and faces.size:
                geometries.append(geom)
        if not geometries:
            raise ValueError(f"Mesh file '{path}' did not contain triangle geometry.")
        tri = trimesh.util.concatenate(tuple(geometries))
    return tri


def _canonicalize_vertices(vertices_mm: np.ndarray, *, scale: float, center_xy: bool = True) -> np.ndarray:
    vertices = np.asarray(vertices_mm, dtype=np.float32) * float(scale)
    extents = np.ptp(vertices, axis=0)
    thickness_axis = int(np.argmin(extents))
    length_axis = int(np.argmax(extents))
    width_axis = ({0, 1, 2} - {thickness_axis, length_axis}).pop()
    vertices = vertices[:, [length_axis, width_axis, thickness_axis]]

    if center_xy:
        center_xy_vec = 0.5 * (np.min(vertices[:, :2], axis=0) + np.max(vertices[:, :2], axis=0))
        vertices[:, :2] -= center_xy_vec
    vertices[:, 2] -= float(np.min(vertices[:, 2]))
    return vertices.astype(np.float32, copy=False)


def _rotation_matrix_rpy_degrees(rotation_degrees: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rotation_degrees, dtype=np.float64))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rot_y = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rot_z = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _rotate_and_rebase_vertices(
    vertices: np.ndarray,
    *,
    rotation_degrees: tuple[float, float, float],
    center_xy: bool = True,
) -> np.ndarray:
    rotation = _rotation_matrix_rpy_degrees(rotation_degrees)
    rotated = np.asarray(vertices, dtype=np.float64) @ rotation.T
    if center_xy:
        center_xy_vec = 0.5 * (np.min(rotated[:, :2], axis=0) + np.max(rotated[:, :2], axis=0))
        rotated[:, :2] -= center_xy_vec
    rotated[:, 2] -= float(np.min(rotated[:, 2]))
    return rotated.astype(np.float32)


def _crop_mesh_by_vertex_z_max(
    vertices: np.ndarray,
    faces: np.ndarray,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    z_max = float(z_max)
    face_vertices = vertices[faces]
    keep_faces = np.max(face_vertices[:, :, 2], axis=1) <= z_max
    kept = int(np.count_nonzero(keep_faces))
    if kept == 0:
        raise ValueError(f"Cropping mesh at z <= {z_max:g} m removed all triangles.")

    cropped_faces = faces[keep_faces]
    used_vertices, remapped_faces = np.unique(cropped_faces.reshape(-1), return_inverse=True)
    cropped_vertices = vertices[used_vertices]
    cropped_faces = remapped_faces.reshape((-1, 3)).astype(np.int32)
    cropped_vertices[:, 2] -= float(np.min(cropped_vertices[:, 2]))
    return (
        cropped_vertices.astype(np.float32, copy=False),
        cropped_faces,
        {
            "crop_z_max_m": z_max,
            "crop_input_triangles": int(faces.shape[0]),
            "crop_output_triangles": kept,
            "crop_removed_triangles": int(faces.shape[0] - kept),
            "crop_input_vertices": int(vertices.shape[0]),
            "crop_output_vertices": int(cropped_vertices.shape[0]),
        },
    )


@lru_cache(maxsize=16)
def _load_newton_mesh_cached(
    path: str,
    device: str,
    scale: float,
    sdf_resolution: int,
    narrow_band: float,
    rotation_degrees: tuple[float, float, float] | None,
    z_max: float | None,
) -> tuple[newton.Mesh, dict[str, Any], np.ndarray]:
    _ = device
    path_obj = Path(path)
    tri = _load_single_trimesh(path_obj)
    vertices = _canonicalize_vertices(np.asarray(tri.vertices, dtype=np.float32), scale=scale)
    if rotation_degrees is not None:
        vertices = _rotate_and_rebase_vertices(vertices, rotation_degrees=rotation_degrees)
    faces = np.asarray(tri.faces, dtype=np.int32)
    crop_stats = None
    if z_max is not None:
        vertices, faces, crop_stats = _crop_mesh_by_vertex_z_max(vertices, faces, float(z_max))
    bounds = np.stack([np.min(vertices, axis=0), np.max(vertices, axis=0)], axis=0)
    stats = {
        "path": str(path_obj),
        "vertices": int(vertices.shape[0]),
        "triangles": int(faces.shape[0]),
        "watertight": bool(tri.is_watertight),
        "winding_consistent": bool(tri.is_winding_consistent),
        "extents_m": np.ptp(vertices, axis=0).astype(float).tolist(),
        "bounds_m": bounds.astype(float).tolist(),
    }
    if rotation_degrees is not None:
        stats["rotation_rpy_deg"] = [float(value) for value in rotation_degrees]
    if crop_stats is not None:
        stats["fixture_crop"] = crop_stats
    mesh = newton.Mesh(vertices, faces.reshape(-1), compute_inertia=False)
    mesh.build_sdf(
        max_resolution=int(sdf_resolution),
        narrow_band_range=(-float(narrow_band), float(narrow_band)),
        margin=float(narrow_band),
    )
    return mesh, stats, vertices


def _load_newton_mesh(
    path: Path,
    *,
    scale: float,
    sdf_resolution: int,
    narrow_band: float,
    rotation_degrees: tuple[float, float, float] | None = None,
    z_max: float | None = None,
) -> tuple[newton.Mesh, dict[str, Any], np.ndarray]:
    rotation_key = None if rotation_degrees is None else tuple(float(value) for value in rotation_degrees)
    mesh, stats, vertices = _load_newton_mesh_cached(
        str(path),
        str(wp.get_device()),
        float(scale),
        int(sdf_resolution),
        float(narrow_band),
        rotation_key,
        None if z_max is None else float(z_max),
    )
    return mesh, dict(stats), vertices


def _load_trial_setup(
    args,
    cases: list[str],
    sample_stride: int,
) -> CachedTrialSetup:
    midsole_mesh, midsole_stats, midsole_vertices = _load_newton_mesh(
        Path(args.midsole_mesh),
        scale=float(args.mesh_scale),
        sdf_resolution=int(args.sdf_resolution),
        narrow_band=float(args.narrow_band),
    )
    midsole_extents = np.asarray(midsole_stats["extents_m"], dtype=np.float64)
    midsole_top = float(midsole_extents[2])

    heel_sign = -1.0 if args.heel_side == "min" else 1.0
    heel_x = heel_sign * 0.30 * float(midsole_extents[0])

    rearfoot_local_top_z: float | None = None
    rearfoot_local_vertex_count = 0
    if "rearfoot" in cases:
        rearfoot_local_top_z, rearfoot_local_vertex_count = _local_top_z_under_radius(
            midsole_vertices,
            (heel_x, 0.0),
            float(args.punch_radius),
        )

    indenter_mesh: newton.Mesh | None = None
    indenter_stats: dict[str, Any] | None = None
    if "fullfoot" in cases:
        indenter_rotation = (
            float(args.fullfoot_rotation_deg[0]),
            float(args.fullfoot_rotation_deg[1]),
            float(args.fullfoot_rotation_deg[2]),
        )
        indenter_mesh, indenter_stats, _indenter_vertices = _load_newton_mesh(
            Path(args.indenter_mesh),
            scale=float(args.mesh_scale),
            sdf_resolution=int(args.sdf_resolution),
            narrow_band=float(args.narrow_band),
            rotation_degrees=indenter_rotation,
            z_max=float(args.fullfoot_indenter_fixture_crop_z) if bool(args.fullfoot_indenter_crop_fixture) else None,
        )

    traces: dict[str, dict[str, Any]] = {}
    for test_case in cases:
        trace_csv, output_csv = _resolve_trial_paths(args, test_case)
        trace, sample_indices = _stride_trace(_load_averaged_trace(trace_csv), sample_stride)
        displacement_m = trace["displacement_m"]
        time_s = trace["time_s"]
        traces[test_case] = {
            "trace_csv": trace_csv,
            "output_csv": output_csv,
            "trace": trace,
            "sample_indices": sample_indices,
            "displacement_m": displacement_m,
            "time_s": time_s,
            "displacement_velocity_m_s": np.gradient(displacement_m, time_s),
            "measured_force_n": trace["force_n"],
        }

    gravity = -9.81 if bool(args.fixture_gravity) else 0.0

    return CachedTrialSetup(
        midsole_mesh=midsole_mesh,
        midsole_stats=midsole_stats,
        midsole_vertices=midsole_vertices,
        midsole_extents=midsole_extents,
        midsole_top=midsole_top,
        rearfoot_local_top_z=rearfoot_local_top_z,
        rearfoot_local_vertex_count=rearfoot_local_vertex_count,
        heel_x=heel_x,
        indenter_mesh=indenter_mesh,
        indenter_stats=indenter_stats,
        traces=traces,
        gravity=gravity,
    )


def _local_top_z_under_radius(vertices: np.ndarray, center_xy: tuple[float, float], radius: float) -> tuple[float, int]:
    center = np.asarray(center_xy, dtype=np.float32)
    dist_xy = np.linalg.norm(vertices[:, :2] - center[None, :], axis=1)
    mask = dist_xy <= float(radius)
    if not np.any(mask):
        nearest = int(np.argmin(dist_xy))
        return float(vertices[nearest, 2]), 1
    return float(np.max(vertices[mask, 2])), int(np.count_nonzero(mask))


def _make_compliant_cfg(
    kh: float,
    kd: float,
    *,
    sdf_resolution: int | None = None,
    narrow_band: float | None = None,
    pressure_profile: str = "poisson",
    pressure_layer_params: tuple[float, float, float, float] = (0.65, 0.0, 0.0, 0.0),
    pressure_sine_amplitude: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pressure_sine_cycles: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pressure_sine_phase: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> newton.ModelBuilder.ShapeConfig:
    kwargs = {}
    if sdf_resolution is not None:
        if narrow_band is None:
            raise ValueError("narrow_band is required when sdf_resolution is set.")
        kwargs["sdf_max_resolution"] = int(sdf_resolution)
        kwargs["sdf_narrow_band_range"] = (-float(narrow_band), float(narrow_band))
    return newton.ModelBuilder.ShapeConfig(
        hydroelastic_type=HydroelasticType.COMPLIANT,
        hydroelastic_contact_workflow=HydroelasticContactWorkflow.PRESSURE,
        hydro_pressure_profile=pressure_profile,
        hydro_pressure_layer_params=pressure_layer_params,
        hydro_pressure_sine_amplitude=pressure_sine_amplitude,
        hydro_pressure_sine_cycles=pressure_sine_cycles,
        hydro_pressure_sine_phase=pressure_sine_phase,
        kh=float(kh),
        kd=float(kd),
        mu=0.8,
        gap=float(narrow_band) if narrow_band is not None else 0.01,
        margin=1.0e-5,
        **kwargs,
    )


def _make_rigid_cfg(
    *,
    sdf_resolution: int | None = None,
    narrow_band: float | None = None,
) -> newton.ModelBuilder.ShapeConfig:
    kwargs = {}
    if sdf_resolution is not None:
        if narrow_band is None:
            raise ValueError("narrow_band is required when sdf_resolution is set.")
        kwargs["sdf_max_resolution"] = int(sdf_resolution)
        kwargs["sdf_narrow_band_range"] = (-float(narrow_band), float(narrow_band))
    return newton.ModelBuilder.ShapeConfig(
        hydroelastic_type=HydroelasticType.RIGID,
        hydroelastic_contact_workflow=HydroelasticContactWorkflow.CLASSIC,
        kh=1.0e12,
        kd=0.0,
        mu=0.8,
        gap=float(narrow_band) if narrow_band is not None else 0.01,
        margin=1.0e-5,
        **kwargs,
    )


def _pair_contact_force_z(contacts, model, state, shape_a: int, shape_b: int) -> tuple[float, dict[str, float | int]]:
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count == 0 or contacts.rigid_contact_stiffness is None:
        return 0.0, {
            "rigid_contact_count": count,
            "pair_contact_count": 0,
            "active_pair_contact_count": 0,
            "pair_stiffness_sum": 0.0,
            "pair_depth_min_m": 0.0,
            "pair_depth_max_m": 0.0,
        }

    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    mask = ((shape0 == shape_a) & (shape1 == shape_b)) | ((shape0 == shape_b) & (shape1 == shape_a))
    if not np.any(mask):
        return 0.0, {
            "rigid_contact_count": count,
            "pair_contact_count": 0,
            "active_pair_contact_count": 0,
            "pair_stiffness_sum": 0.0,
            "pair_depth_min_m": 0.0,
            "pair_depth_max_m": 0.0,
        }

    normals = contacts.rigid_contact_normal.numpy()[:count][mask]
    p0 = contacts.rigid_contact_point0.numpy()[:count][mask]
    p1 = contacts.rigid_contact_point1.numpy()[:count][mask]
    stiffness = contacts.rigid_contact_stiffness.numpy()[:count][mask]

    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()
    b0 = shape_body[shape0[mask]]
    b1 = shape_body[shape1[mask]]
    p0w = _body_points_to_world(p0, b0, body_q)
    p1w = _body_points_to_world(p1, b1, body_q)

    depth = np.einsum("ij,ij->i", p0w - p1w, -normals) / 2.0
    active = (stiffness > 0.0) & (depth < 0.0)
    stats = {
        "rigid_contact_count": count,
        "pair_contact_count": int(np.count_nonzero(mask)),
        "active_pair_contact_count": int(np.count_nonzero(active)),
        "pair_stiffness_sum": float(np.sum(stiffness)),
        "pair_depth_min_m": float(np.min(depth)) if depth.size else 0.0,
        "pair_depth_max_m": float(np.max(depth)) if depth.size else 0.0,
    }
    if not np.any(active):
        return 0.0, stats
    force_mag = stiffness[active] * (-depth[active])
    force_vec = force_mag[:, None] * (-normals[active])
    return float(abs(np.sum(force_vec[:, 2]))), stats


def _pair_contact_stats_z(contacts, model, state, shape_a: int, shape_b: int) -> dict[str, float | int]:
    if (
        not all(
            hasattr(contacts, attr)
            for attr in (
                "rigid_contact_count",
                "rigid_contact_shape0",
                "rigid_contact_shape1",
                "rigid_contact_normal",
                "rigid_contact_point0",
                "rigid_contact_point1",
                "rigid_contact_stiffness",
            )
        )
        or not hasattr(model, "shape_body")
        or not hasattr(state, "body_q")
    ):
        return {
            "rigid_contact_count": 0,
            "pair_contact_count": 0,
            "active_pair_contact_count": 0,
            "pair_stiffness_sum": 0.0,
            "pair_depth_min_m": 0.0,
            "pair_depth_max_m": 0.0,
        }

    count = int(contacts.rigid_contact_count.numpy()[0])
    if count == 0 or contacts.rigid_contact_stiffness is None:
        return {
            "rigid_contact_count": count,
            "pair_contact_count": 0,
            "active_pair_contact_count": 0,
            "pair_stiffness_sum": 0.0,
            "pair_depth_min_m": 0.0,
            "pair_depth_max_m": 0.0,
        }

    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    mask = ((shape0 == shape_a) & (shape1 == shape_b)) | ((shape0 == shape_b) & (shape1 == shape_a))
    if not np.any(mask):
        return {
            "rigid_contact_count": count,
            "pair_contact_count": 0,
            "active_pair_contact_count": 0,
            "pair_stiffness_sum": 0.0,
            "pair_depth_min_m": 0.0,
            "pair_depth_max_m": 0.0,
        }

    normals = contacts.rigid_contact_normal.numpy()[:count][mask]
    p0 = contacts.rigid_contact_point0.numpy()[:count][mask]
    p1 = contacts.rigid_contact_point1.numpy()[:count][mask]
    stiffness = contacts.rigid_contact_stiffness.numpy()[:count][mask]

    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()
    b0 = shape_body[shape0[mask]]
    b1 = shape_body[shape1[mask]]
    p0w = _body_points_to_world(p0, b0, body_q)
    p1w = _body_points_to_world(p1, b1, body_q)

    depth = np.einsum("ij,ij->i", p0w - p1w, -normals) / 2.0
    active = (stiffness > 0.0) & (depth < 0.0)
    return {
        "rigid_contact_count": count,
        "pair_contact_count": int(np.count_nonzero(mask)),
        "active_pair_contact_count": int(np.count_nonzero(active)),
        "pair_stiffness_sum": float(np.sum(stiffness)),
        "pair_depth_min_m": float(np.min(depth)) if depth.size else 0.0,
        "pair_depth_max_m": float(np.max(depth)) if depth.size else 0.0,
    }


def _set_body_z(state, body_idx: int, z: float) -> None:
    wp.launch(
        _set_body_translation_z,
        dim=1,
        inputs=[state.body_q, body_idx, float(z)],
        device=wp.get_device(),
    )


def _shoe_box_inertia(extents: np.ndarray, mass: float) -> wp.mat33:
    lx = max(float(extents[0]), 1.0e-4)
    ly = max(float(extents[1]), 1.0e-4)
    lz = max(float(extents[2]), 1.0e-4)
    ixx = mass * (ly * ly + lz * lz) / 12.0
    iyy = mass * (lx * lx + lz * lz) / 12.0
    izz = mass * (lx * lx + ly * ly) / 12.0
    return wp.mat33(
        float(ixx),
        0.0,
        0.0,
        0.0,
        float(iyy),
        0.0,
        0.0,
        0.0,
        float(izz),
    )


def _make_fixture_solver(args, model):
    solver_name = str(getattr(args, "fixture_solver", "semi-implicit"))
    if solver_name == "xpbd":
        return newton.solvers.SolverXPBD(model, iterations=int(args.fixture_solver_iterations))
    if solver_name == "semi-implicit":
        return newton.solvers.SolverSemiImplicit(model, angular_damping=float(args.fixture_solver_angular_damping))
    if solver_name == "mujoco":
        return newton.solvers.SolverMuJoCo(
            model,
            njmax=int(getattr(args, "fixture_mujoco_njmax", 16384)),
            nconmax=int(getattr(args, "fixture_mujoco_nconmax", 4096)),
            use_mujoco_contacts=False,
            use_mujoco_cpu=bool(getattr(args, "fixture_mujoco_cpu", False)),
            iterations=int(getattr(args, "fixture_mujoco_iterations", 100)),
            solver=str(getattr(args, "fixture_mujoco_solver", "newton")),
            integrator=str(getattr(args, "fixture_mujoco_integrator", "implicitfast")),
        )
    raise ValueError(f"Unknown fixture solver: {solver_name}")


def _validate_fixture_contact_capacity(scene: TrialScene, args) -> None:
    if str(getattr(args, "fixture_solver", "")) != "mujoco":
        return
    if bool(getattr(args, "fixture_mujoco_cpu", False)):
        return
    if not bool(getattr(args, "fail_on_fixture_contact_overflow", True)):
        return

    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    limit = int(getattr(args, "fixture_mujoco_nconmax", 4096))
    if count <= limit:
        return

    reduction_hint = (
        " Enable --contact-reduction so the hydroelastic solver feeds aggregate contacts to MuJoCo."
        if bool(getattr(args, "no_contact_reduction", False))
        else " Inspect hydroelastic reduction settings before increasing --fixture-mujoco-nconmax."
    )
    raise RuntimeError(
        f"{scene.test_case} generated {count} Newton contacts, exceeding the MJWarp bridge limit "
        f"of {limit}. Continuing would truncate contacts and invalidate fixture mechanics."
        f"{reduction_hint}"
    )


def _sample_dt(time_s: np.ndarray, sample_index: int) -> float:
    if time_s.size < 2:
        return 1.0 / 240.0
    if sample_index <= 0:
        return max(float(time_s[1] - time_s[0]), 1.0e-5)
    return max(float(time_s[sample_index] - time_s[sample_index - 1]), 1.0e-5)


def _indenter_substep_commands(
    *,
    start_z: float,
    target_z: float,
    trace_velocity_z: float,
    dt: float,
    substeps: int,
    stop_z: float,
) -> list[tuple[float, float, int]]:
    substeps = max(int(substeps), 1)
    trajectory_velocity_z = (float(target_z) - float(start_z)) / max(float(dt), 1.0e-9)
    velocity_z = trajectory_velocity_z if abs(trajectory_velocity_z) > 1.0e-12 else float(trace_velocity_z)
    return [
        _clamp_indenter_to_platen_stop(
            float(start_z) + (float(index + 1) / float(substeps)) * (float(target_z) - float(start_z)),
            velocity_z,
            stop_z,
        )
        for index in range(substeps)
    ]


def _fixture_step(
    scene: TrialScene,
    args,
    *,
    indenter_z: float,
    indenter_velocity_z: float,
    dt: float,
    substeps: int,
) -> None:
    _fixture_step_many(
        [scene],
        args,
        indenter_z=[indenter_z],
        indenter_velocity_z=[indenter_velocity_z],
        dt=dt,
        substeps=substeps,
    )


def _fixture_step_many(
    scenes: list[TrialScene],
    args,
    *,
    indenter_z: list[float],
    indenter_velocity_z: list[float],
    dt: float,
    substeps: int,
) -> None:
    if len(scenes) != len(indenter_z) or len(scenes) != len(indenter_velocity_z):
        raise ValueError("scenes, indenter_z, and indenter_velocity_z must have the same length.")
    if not scenes:
        return
    substeps = max(int(substeps), 1)
    sub_dt = float(dt) / float(substeps)
    body_q = scenes[0].state.body_q.numpy()
    commands_by_scene = []
    for scene, target_z, target_velocity_z in zip(scenes, indenter_z, indenter_velocity_z, strict=True):
        clamped_z, clamped_velocity_z, _ = _clamp_indenter_to_platen_stop(
            target_z,
            target_velocity_z,
            scene.indenter_stop_z,
        )
        commands_by_scene.append(
            _indenter_substep_commands(
                start_z=float(body_q[scene.indenter_body, 2]),
                target_z=clamped_z,
                trace_velocity_z=clamped_velocity_z,
                dt=dt,
                substeps=substeps,
                stop_z=scene.indenter_stop_z,
            )
        )

    state = scenes[0].state
    state_next = scenes[0].state_next
    pipeline = scenes[0].pipeline
    contacts = scenes[0].contacts
    solver = scenes[0].solver
    control = scenes[0].control
    for substep_index in range(substeps):
        for scene, commands in zip(scenes, commands_by_scene, strict=True):
            substep_indenter_z, substep_indenter_velocity_z, _ = commands[substep_index]
            _stabilize_shoe_fixture(scene, args)
            _set_body_z_velocity(state, scene.indenter_body, substep_indenter_z, substep_indenter_velocity_z)
        state.clear_forces()
        pipeline.collide(state, contacts)
        for scene in scenes:
            _validate_fixture_contact_capacity(scene, args)
        solver.step(state, state_next, control, contacts, sub_dt)
        _check_solver_state_finite(state_next, scenes)
        state, state_next = state_next, state
        for scene, commands in zip(scenes, commands_by_scene, strict=True):
            substep_indenter_z, substep_indenter_velocity_z, _ = commands[substep_index]
            scene.state = state
            scene.state_next = state_next
            _stabilize_shoe_fixture(scene, args)
            _set_body_z_velocity(state, scene.indenter_body, substep_indenter_z, substep_indenter_velocity_z)
    if state.body_count and state_next.body_count:
        state.body_f.assign(state_next.body_f.numpy())


def _read_body_force_z(state, body_idx: int) -> float:
    """Read the magnitude of the Z-component of the net force on a body from solver state."""
    body_f = state.body_f.numpy()
    return float(abs(body_f[body_idx, 2]))


def _measure_trial_force(
    scene: TrialScene,
    *,
    include_diagnostics: bool = False,
) -> tuple[float, dict[str, float | int]]:
    top_solver_force = _read_body_force_z(scene.state, scene.indenter_body)
    bottom_solver_force = _read_body_force_z(scene.state, scene.shoe_body)
    top_force = top_solver_force
    bottom_force = bottom_solver_force
    top_stats = _pair_contact_stats_z(
        scene.contacts, scene.model, scene.state, scene.indenter_shape, scene.midsole_shape
    )
    bottom_stats = _pair_contact_stats_z(
        scene.contacts, scene.model, scene.state, scene.base_shape, scene.midsole_shape
    )
    top_geometry_force: float | None = None
    bottom_geometry_force: float | None = None

    if top_force <= 1.0e-9:
        top_geometry_force, top_stats = _pair_contact_force_z(
            scene.contacts,
            scene.model,
            scene.state,
            scene.indenter_shape,
            scene.midsole_shape,
        )
        top_force = max(top_force, top_geometry_force)
    if bottom_force <= 1.0e-9:
        bottom_geometry_force, bottom_stats = _pair_contact_force_z(
            scene.contacts,
            scene.model,
            scene.state,
            scene.base_shape,
            scene.midsole_shape,
        )
        bottom_force = max(bottom_force, bottom_geometry_force)

    stats = {
        **top_stats,
        "top_contact_force_n": float(top_force),
        "top_solver_force_n": float(top_solver_force),
        **{f"top_{key}": value for key, value in top_stats.items()},
        "bottom_contact_force_n": float(bottom_force),
        "bottom_solver_force_n": float(bottom_solver_force),
        **{f"bottom_{key}": value for key, value in bottom_stats.items()},
    }

    if not include_diagnostics:
        return top_force, stats

    if top_geometry_force is None:
        top_geometry_force, _ = _pair_contact_force_z(
            scene.contacts,
            scene.model,
            scene.state,
            scene.indenter_shape,
            scene.midsole_shape,
        )
    if bottom_geometry_force is None:
        bottom_geometry_force, _ = _pair_contact_force_z(
            scene.contacts,
            scene.model,
            scene.state,
            scene.base_shape,
            scene.midsole_shape,
        )

    top_disagreement = abs(top_solver_force - top_geometry_force)
    top_disagreement_ratio = min(top_disagreement / max(abs(top_solver_force), 1.0e-9), 1.0)
    if top_disagreement_ratio > 0.2:
        logger.warning(
            "Top contact force disagreement is %.1f%% (solver=%.6g N, geometry=%.6g N)",
            100.0 * top_disagreement_ratio,
            top_solver_force,
            top_geometry_force,
        )

    bottom_disagreement = abs(bottom_solver_force - bottom_geometry_force)
    bottom_disagreement_ratio = min(bottom_disagreement / max(abs(bottom_solver_force), 1.0e-9), 1.0)
    if bottom_disagreement_ratio > 0.2:
        logger.warning(
            "Bottom contact force disagreement is %.1f%% (solver=%.6g N, geometry=%.6g N)",
            100.0 * bottom_disagreement_ratio,
            bottom_solver_force,
            bottom_geometry_force,
        )

    stats.update(
        {
            "top_geometry_force_n": float(top_geometry_force),
            "top_force_disagreement_n": float(top_disagreement),
            "top_force_disagreement_ratio": float(top_disagreement_ratio),
            "bottom_geometry_force_n": float(bottom_geometry_force),
            "bottom_force_disagreement_n": float(bottom_disagreement),
            "bottom_force_disagreement_ratio": float(bottom_disagreement_ratio),
        }
    )
    return top_force, stats


def _fixture_settle_indenter_z(scene: TrialScene, args) -> float:
    return float(scene.indenter_rest_z) + max(float(getattr(args, "fixture_settle_indenter_clearance", 0.0)), 0.0)


def _solve_gravity_preload(scene: TrialScene, args) -> dict[str, float | int]:
    if not bool(getattr(args, "fixture_dynamic_shoe", False)) or not bool(getattr(args, "fixture_gravity", False)):
        return {
            "fixture_gravity_preload_enabled": 0,
            "fixture_gravity_preload_force_n": 0.0,
            "fixture_gravity_preload_target_n": 0.0,
            "fixture_gravity_preload_error_n": 0.0,
            "fixture_gravity_preload_iterations": 0,
        }

    gravity = 9.81
    target_force = max(float(getattr(args, "shoe_mass", 0.0)) * gravity, 0.0)
    if target_force <= 0.0:
        return {
            "fixture_gravity_preload_enabled": 0,
            "fixture_gravity_preload_force_n": 0.0,
            "fixture_gravity_preload_target_n": 0.0,
            "fixture_gravity_preload_error_n": 0.0,
            "fixture_gravity_preload_iterations": 0,
        }

    body_q = scene.state.body_q.numpy()
    center_z = float(body_q[scene.shoe_body, 2])
    search_m = max(float(getattr(args, "fixture_gravity_preload_search_m", 0.0)), 1.0e-6)
    iterations = max(int(getattr(args, "fixture_gravity_preload_iterations", 0)), 1)
    indenter_clear_z = _fixture_settle_indenter_z(scene, args)

    def bottom_force_at(shoe_z: float) -> float:
        _set_body_z_velocity(scene.state, scene.shoe_body, float(shoe_z), 0.0)
        _set_body_z_velocity(scene.state, scene.indenter_body, indenter_clear_z, 0.0)
        _stabilize_shoe_fixture(scene, args, zero_vertical_velocity=True)
        scene.pipeline.collide(scene.state, scene.contacts)
        _bottom_force, _bottom_stats = _pair_contact_force_z(
            scene.contacts,
            scene.model,
            scene.state,
            scene.base_shape,
            scene.midsole_shape,
        )
        return float(_bottom_force)

    high_z = center_z + search_m
    low_z = center_z - search_m
    high_force = bottom_force_at(high_z)
    low_force = bottom_force_at(low_z)
    expand_count = 0
    while high_force > target_force and expand_count < 6:
        search_m *= 2.0
        high_z = center_z + search_m
        high_force = bottom_force_at(high_z)
        expand_count += 1
    while low_force < target_force and expand_count < 12:
        search_m *= 2.0
        low_z = center_z - search_m
        low_force = bottom_force_at(low_z)
        expand_count += 1

    best_z = high_z if abs(high_force - target_force) <= abs(low_force - target_force) else low_z
    best_force = high_force if best_z == high_z else low_force
    if low_force >= target_force >= high_force:
        for _ in range(iterations):
            mid_z = 0.5 * (low_z + high_z)
            mid_force = bottom_force_at(mid_z)
            if abs(mid_force - target_force) < abs(best_force - target_force):
                best_z = mid_z
                best_force = mid_force
            if mid_force > target_force:
                low_z = mid_z
            else:
                high_z = mid_z

    final_force = bottom_force_at(best_z)
    return {
        "fixture_gravity_preload_enabled": 1,
        "fixture_gravity_preload_force_n": float(final_force),
        "fixture_gravity_preload_target_n": float(target_force),
        "fixture_gravity_preload_error_n": float(final_force - target_force),
        "fixture_gravity_preload_iterations": int(iterations + expand_count),
    }


def _settle_fixture(scene: TrialScene, args) -> dict[str, float | int]:
    duration = max(float(args.fixture_settle_duration), 0.0)
    substeps = max(int(args.fixture_settle_substeps), 1)
    if duration <= 0.0:
        _set_body_z_velocity(scene.state, scene.indenter_body, _fixture_settle_indenter_z(scene, args), 0.0)
        _stabilize_shoe_fixture(scene, args, zero_vertical_velocity=True)
        preload_stats = _solve_gravity_preload(scene, args)
        return {
            "fixture_settle_steps": 0,
            "fixture_settle_converged": 0,
            "fixture_settle_final_speed_m_s": 0.0,
            **preload_stats,
        }

    dt = 1.0 / max(float(args.fixture_settle_rate_hz), 1.0)
    max_steps = max(int(np.ceil(duration / dt)), 1)
    velocity_tol = max(float(args.fixture_settle_velocity_tol), 0.0)
    final_speed = float("inf")
    converged = False
    steps = 0
    for step_index in range(1, max_steps + 1):
        steps = step_index
        _fixture_step(
            scene,
            args,
            indenter_z=_fixture_settle_indenter_z(scene, args),
            indenter_velocity_z=0.0,
            dt=dt,
            substeps=substeps,
        )
        body_qd = scene.state.body_qd.numpy()
        final_speed = float(np.linalg.norm(body_qd[scene.shoe_body, :3]))
        if final_speed <= velocity_tol:
            converged = True
            break

    _set_body_z_velocity(scene.state, scene.indenter_body, _fixture_settle_indenter_z(scene, args), 0.0)
    _stabilize_shoe_fixture(scene, args, zero_vertical_velocity=True)
    preload_stats = _solve_gravity_preload(scene, args)
    if bool(args.fixture_lock_rotation_after_settle):
        body_q = scene.state.body_q.numpy()
        scene.shoe_anchor_quat = tuple(float(v) for v in body_q[scene.shoe_body, 3:7])
        scene.settled_rotation_locked = True
    return {
        "fixture_settle_steps": int(steps),
        "fixture_settle_converged": int(converged),
        "fixture_settle_final_speed_m_s": float(final_speed),
        **preload_stats,
    }


def _settle_fixture_many(scenes: list[TrialScene], args) -> list[dict[str, float | int]]:
    if not scenes:
        return []
    duration = max(float(args.fixture_settle_duration), 0.0)
    if duration <= 0.0:
        for scene in scenes:
            _set_body_z_velocity(scene.state, scene.indenter_body, _fixture_settle_indenter_z(scene, args), 0.0)
            _stabilize_shoe_fixture(scene, args, zero_vertical_velocity=True)
        preload_stats = [_solve_gravity_preload(scene, args) for scene in scenes]
        return [
            {
                "fixture_settle_steps": 0,
                "fixture_settle_converged": 0,
                "fixture_settle_final_speed_m_s": 0.0,
                **preload_stats[scene_index],
            }
            for scene_index, _scene in enumerate(scenes)
        ]

    dt = 1.0 / max(float(args.fixture_settle_rate_hz), 1.0)
    max_steps = max(int(np.ceil(duration / dt)), 1)
    velocity_tol = max(float(args.fixture_settle_velocity_tol), 0.0)
    final_speed = np.full(len(scenes), np.inf, dtype=np.float64)
    converged = np.zeros(len(scenes), dtype=bool)
    steps = 0
    for step_index in range(1, max_steps + 1):
        steps = step_index
        _fixture_step_many(
            scenes,
            args,
            indenter_z=[_fixture_settle_indenter_z(scene, args) for scene in scenes],
            indenter_velocity_z=[0.0 for _scene in scenes],
            dt=dt,
            substeps=max(int(args.fixture_settle_substeps), 1),
        )
        body_qd = scenes[0].state.body_qd.numpy()
        for scene_index, scene in enumerate(scenes):
            final_speed[scene_index] = float(np.linalg.norm(body_qd[scene.shoe_body, :3]))
        converged |= final_speed <= velocity_tol
        if bool(np.all(converged)):
            break

    for scene in scenes:
        _set_body_z_velocity(scene.state, scene.indenter_body, _fixture_settle_indenter_z(scene, args), 0.0)
        _stabilize_shoe_fixture(scene, args, zero_vertical_velocity=True)
        if bool(args.fixture_lock_rotation_after_settle):
            body_q = scene.state.body_q.numpy()
            scene.shoe_anchor_quat = tuple(float(v) for v in body_q[scene.shoe_body, 3:7])
            scene.settled_rotation_locked = True
    preload_stats = [_solve_gravity_preload(scene, args) for scene in scenes]
    return [
        {
            "fixture_settle_steps": int(steps),
            "fixture_settle_converged": int(converged[scene_index]),
            "fixture_settle_final_speed_m_s": float(final_speed[scene_index]),
            **preload_stats[scene_index],
        }
        for scene_index, _scene in enumerate(scenes)
    ]


def _advance_and_measure_trial_force(
    scene: TrialScene,
    args,
    *,
    sample_index: int,
    displacement: float,
    displacement_velocity: float,
    relax_steps: int,
    include_diagnostics: bool = False,
) -> tuple[float, dict[str, float | int]]:
    indenter_z = float(scene.indenter_rest_z - displacement)
    indenter_velocity_z = -float(displacement_velocity)
    indenter_z, indenter_velocity_z, indenter_clamped = _clamp_indenter_to_platen_stop(
        indenter_z,
        indenter_velocity_z,
        scene.indenter_stop_z,
    )
    if relax_steps > 0:
        _fixture_step(
            scene,
            args,
            indenter_z=indenter_z,
            indenter_velocity_z=indenter_velocity_z,
            dt=_sample_dt(scene.time_s, sample_index),
            substeps=int(relax_steps),
        )
        if bool(args.fixture_quasistatic_replay):
            _stabilize_shoe_fixture(scene, args, zero_vertical_velocity=True)
            _set_body_z_velocity(scene.state, scene.indenter_body, indenter_z, 0.0)
    else:
        _set_body_z_velocity(scene.state, scene.indenter_body, indenter_z, indenter_velocity_z)
    if include_diagnostics and relax_steps == 0:
        scene.pipeline.collide(scene.state, scene.contacts)
    force, stats = _measure_trial_force(scene, include_diagnostics=include_diagnostics)
    stats.update(_shoe_motion_stats(scene, args))
    stats["indenter_platen_stop_active"] = int(indenter_clamped)
    stats["indenter_command_z_m"] = float(scene.indenter_rest_z - displacement)
    stats["indenter_applied_z_m"] = float(indenter_z)
    stats["indenter_stop_z_m"] = float(scene.indenter_stop_z)
    return force, stats


def _contact_sample_stats(time_s: np.ndarray) -> dict[str, float | int]:
    if time_s.size < 2:
        return {
            "contact_sample_count": int(time_s.size),
            "trace_duration_s": 0.0,
            "contact_mean_dt_s": 0.0,
            "contact_evaluation_rate_hz": 0.0,
        }
    dt = np.diff(time_s)
    mean_dt = float(np.mean(dt))
    return {
        "contact_sample_count": int(time_s.size),
        "trace_duration_s": float(time_s[-1] - time_s[0]),
        "contact_mean_dt_s": mean_dt,
        "contact_evaluation_rate_hz": float(1.0 / mean_dt) if mean_dt > 0.0 else 0.0,
    }


def _fullfoot_auto_contact_search_limit(args) -> tuple[float, float]:
    requested_search_max = max(float(args.fullfoot_contact_search_max), 0.0)
    rest_clearance = float(args.fullfoot_start_clearance) - float(args.fullfoot_z_offset)
    return requested_search_max, rest_clearance


def _fullfoot_indenter_rest_z(args, *, shoe_z: float, midsole_top: float) -> float:
    return float(shoe_z) + float(midsole_top) + float(args.fullfoot_start_clearance) - float(args.fullfoot_z_offset)


def _rearfoot_indenter_rest_z(args, *, shoe_z: float, local_top_z: float) -> float:
    return float(shoe_z) + float(local_top_z) + float(args.punch_half_height)


def _fixture_shoe_rest_z(args, *, base_top_z: float) -> float:
    return float(base_top_z) - max(float(getattr(args, "fixture_bottom_platen_overlap", 0.0)), 0.0)


def _compression_relax_steps(args) -> int:
    value = getattr(args, "compression_relax_steps", None)
    if value is None:
        return 4
    return int(value)


def _indenter_platen_stop_z(args, *, test_case: str, base_top_z: float) -> float:
    clearance = max(float(getattr(args, "fixture_platen_stop_clearance", 0.0)), 0.0)
    if test_case == "rearfoot":
        return float(base_top_z) + float(args.punch_half_height) + clearance
    if test_case == "fullfoot":
        return float(base_top_z) + clearance
    raise ValueError(f"Unknown test case: {test_case}")


def _clamp_indenter_to_platen_stop(
    indenter_z: float,
    indenter_velocity_z: float,
    stop_z: float,
) -> tuple[float, float, int]:
    if float(indenter_z) >= float(stop_z):
        return float(indenter_z), float(indenter_velocity_z), 0
    return float(stop_z), max(float(indenter_velocity_z), 0.0), 1


def _find_trial_contact_offset(
    args,
    *,
    test_case: str,
    pipeline,
    contacts,
    model,
    state,
    indenter_body: int,
    indenter_shape: int,
    midsole_shape: int,
    indenter_rest_z: float,
    displacement_m: np.ndarray,
    measured_force_n: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    if test_case == "fullfoot":
        enabled = bool(args.fullfoot_auto_contact_offset)
        prefix = "fullfoot_auto"
        search_max = max(float(args.fullfoot_contact_search_max), 0.0)
        rest_clearance = float(args.fullfoot_start_clearance) - float(args.fullfoot_z_offset)
        min_start_clearance = float(args.fullfoot_min_start_clearance)
        max_initial_force = float(args.fullfoot_initial_force_max)
        search_steps = int(args.fullfoot_contact_search_steps)
        tolerance = float(args.fullfoot_contact_search_tolerance)
        stop_after_target = bool(args.fullfoot_stop_search_after_target)
    elif test_case == "rearfoot":
        enabled = bool(getattr(args, "rearfoot_auto_contact_offset", True))
        prefix = "rearfoot_auto"
        search_max = max(float(getattr(args, "rearfoot_contact_search_max", args.fullfoot_contact_search_max)), 0.0)
        rest_clearance = 0.0
        min_start_clearance = 0.0
        max_initial_force = float(getattr(args, "rearfoot_initial_force_max", args.fullfoot_initial_force_max))
        search_steps = int(getattr(args, "rearfoot_contact_search_steps", args.fullfoot_contact_search_steps))
        tolerance = float(getattr(args, "rearfoot_contact_search_tolerance", args.fullfoot_contact_search_tolerance))
        stop_after_target = bool(
            getattr(args, "rearfoot_stop_search_after_target", args.fullfoot_stop_search_after_target)
        )
    else:
        raise ValueError(f"Unknown test case: {test_case}")

    if not enabled:
        return 0.0, {
            f"{prefix}_contact_offset_m": 0.0,
            f"{prefix}_contact_found": 0,
            f"{prefix}_contact_search_max_m": float(search_max),
        }

    contact_threshold = 1.0  # N: force level indicating initial geometric contact
    best_offset = 0.0
    valid_candidates = 0
    best_stats: dict[str, float | int] = {
        f"{prefix}_contact_offset_m": 0.0,
        f"{prefix}_contact_found": 0,
        f"{prefix}_contact_search_max_m": float(search_max),
        f"{prefix}_contact_search_requested_max_m": float(search_max),
        f"{prefix}_rest_clearance_m": float(rest_clearance),
        f"{prefix}_min_start_clearance_m": float(min_start_clearance),
        f"{prefix}_initial_force_max_n": max_initial_force,
        f"{prefix}_contact_threshold_n": contact_threshold,
    }
    for offset in np.linspace(0.0, float(search_max), max(int(search_steps), 1)):
        _set_body_z(state, indenter_body, indenter_rest_z - float(offset))
        pipeline.collide(state, contacts)
        initial_force, _initial_stats = _pair_contact_force_z(contacts, model, state, indenter_shape, midsole_shape)
        if initial_force > max_initial_force:
            continue
        valid_candidates += 1
        if initial_force >= contact_threshold:
            best_offset = float(offset)
            best_stats = {
                f"{prefix}_contact_offset_m": best_offset,
                f"{prefix}_contact_found": int(initial_force > 1.0e-9),
                f"{prefix}_contact_search_max_m": float(search_max),
                f"{prefix}_contact_search_requested_max_m": float(search_max),
                f"{prefix}_rest_clearance_m": float(rest_clearance),
                f"{prefix}_min_start_clearance_m": float(min_start_clearance),
                f"{prefix}_initial_force_max_n": max_initial_force,
                f"{prefix}_contact_threshold_n": contact_threshold,
                f"{prefix}_initial_contact_force_n": float(initial_force),
                **{f"{prefix}_{key}": value for key, value in _initial_stats.items()},
            }
            break

    if valid_candidates == 0:
        _set_body_z(state, indenter_body, indenter_rest_z)
        pipeline.collide(state, contacts)
        initial_force, _initial_stats = _pair_contact_force_z(contacts, model, state, indenter_shape, midsole_shape)
        best_stats = {
            f"{prefix}_contact_offset_m": 0.0,
            f"{prefix}_contact_found": int(initial_force > 1.0e-9),
            f"{prefix}_contact_search_max_m": float(search_max),
            f"{prefix}_contact_search_requested_max_m": float(search_max),
            f"{prefix}_rest_clearance_m": float(rest_clearance),
            f"{prefix}_min_start_clearance_m": float(min_start_clearance),
            f"{prefix}_valid_candidates": 0,
            f"{prefix}_initial_force_n": float(initial_force),
            f"{prefix}_initial_force_max_n": max_initial_force,
            f"{prefix}_contact_threshold_n": contact_threshold,
            **{f"{prefix}_{key}": value for key, value in _initial_stats.items()},
        }

    _set_body_z(state, indenter_body, indenter_rest_z)
    return best_offset, best_stats


def _find_fullfoot_contact_offset(
    args,
    pipeline,
    contacts,
    model,
    state,
    indenter_body: int,
    indenter_shape: int,
    midsole_shape: int,
    indenter_rest_z: float,
    displacement_m: np.ndarray,
    measured_force_n: np.ndarray,
) -> tuple[float, dict[str, float | int]]:
    return _find_trial_contact_offset(
        args,
        test_case="fullfoot",
        pipeline=pipeline,
        contacts=contacts,
        model=model,
        state=state,
        indenter_body=indenter_body,
        indenter_shape=indenter_shape,
        midsole_shape=midsole_shape,
        indenter_rest_z=indenter_rest_z,
        displacement_m=displacement_m,
        measured_force_n=measured_force_n,
    )


def _write_sim_results(path: Path, rows: list[dict[str, float | str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _pressure_memory_state_model(args) -> dict[str, Any]:
    if not bool(getattr(args, "pressure_memory", False)):
        return {"type": "memoryless_v1", "parameters": {}}
    return {
        "type": "max_compression_memory_v1",
        "parameters": {
            "key": "shape_pair_normal_bin_coarse_local_cell",
            "aggregate_source": "all_penetrating_hydroelastic_faces_before_reduction_export",
            "grid": [int(v) for v in getattr(args, "pressure_memory_grid", (16, 8, 1))],
            "unloading_loss": float(getattr(args, "pressure_memory_unloading_loss", 0.45)),
            "recovery_tau_s": float(getattr(args, "pressure_memory_recovery_tau_s", 0.25)),
            "dt_s": float(getattr(args, "pressure_memory_dt_s", 1.0 / 240.0)),
        },
    }


def _build_trial_scene(
    args,
    *,
    test_case: str,
    trace_csv: Path,
    output_csv: Path,
    sample_stride: int = 1,
    cached_setup: CachedTrialSetup | None = None,
) -> TrialScene:
    if args.device:
        wp.set_device(args.device)
    if not wp.get_device().is_cuda:
        raise RuntimeError("digital_instron pressure-field run requires CUDA for SDF volumes.")

    if cached_setup is not None:
        trace_data = cached_setup.traces[test_case]
        trace = trace_data["trace"]
        sample_indices = trace_data["sample_indices"]
        displacement_m = trace_data["displacement_m"]
        time_s = trace_data["time_s"]
        displacement_velocity_m_s = trace_data["displacement_velocity_m_s"]
        measured_force_n = trace_data["measured_force_n"]
        midsole_mesh = cached_setup.midsole_mesh
        midsole_stats = dict(cached_setup.midsole_stats)
        midsole_vertices = cached_setup.midsole_vertices
        midsole_extents = cached_setup.midsole_extents
        midsole_top = cached_setup.midsole_top
        gravity = cached_setup.gravity
    else:
        trace, sample_indices = _stride_trace(_load_averaged_trace(trace_csv), sample_stride)
        displacement_m = trace["displacement_m"]
        time_s = trace["time_s"]
        displacement_velocity_m_s = np.gradient(displacement_m, time_s)
        measured_force_n = trace["force_n"]

        midsole_mesh, midsole_stats, midsole_vertices = _load_newton_mesh(
            Path(args.midsole_mesh),
            scale=float(args.mesh_scale),
            sdf_resolution=int(args.sdf_resolution),
            narrow_band=float(args.narrow_band),
        )
        midsole_extents = np.asarray(midsole_stats["extents_m"], dtype=np.float64)
        midsole_top = float(midsole_extents[2])
        gravity = -9.81 if bool(args.fixture_gravity) else 0.0

    midsole_stats["pressure_profile"] = str(args.pressure_profile)
    midsole_stats["pressure_layer_params"] = {
        "eta_lock": float(args.layer_eta_lock),
        "densification_power": float(args.layer_densification_power),
        "cubic": float(args.layer_cubic),
        "quintic": float(args.layer_quintic),
    }
    midsole_stats["pressure_sine_amplitude"] = [float(v) for v in args.pressure_sine_amplitude]
    midsole_stats["pressure_sine_cycles"] = [float(v) for v in args.pressure_sine_cycles]
    midsole_stats["pressure_sine_phase"] = [float(v) for v in args.pressure_sine_phase]
    midsole_stats["trace_sample_stride"] = int(max(sample_stride, 1))
    midsole_stats["trace_sample_indices"] = sample_indices.astype(int).tolist()

    builder = newton.ModelBuilder(gravity=gravity)
    compliant_mesh_cfg = _make_compliant_cfg(
        float(args.kh),
        float(args.kd),
        narrow_band=float(args.narrow_band),
        pressure_profile=str(args.pressure_profile),
        pressure_layer_params=(
            float(args.layer_eta_lock),
            float(args.layer_densification_power),
            float(args.layer_cubic),
            float(args.layer_quintic),
        ),
        pressure_sine_amplitude=tuple(float(v) for v in args.pressure_sine_amplitude),
        pressure_sine_cycles=tuple(float(v) for v in args.pressure_sine_cycles),
        pressure_sine_phase=tuple(float(v) for v in args.pressure_sine_phase),
    )
    rigid_mesh_cfg = _make_rigid_cfg(narrow_band=float(args.narrow_band))
    rigid_primitive_cfg = _make_rigid_cfg(
        sdf_resolution=int(args.sdf_resolution),
        narrow_band=float(args.narrow_band),
    )

    shoe_mass = float(args.shoe_mass)
    if shoe_mass <= 0.0:
        raise ValueError("--shoe-mass must be positive for the dynamic Digital Instron fixture.")
    base_hz = 0.025
    base_top_z = 0.0
    shoe_rest_z = _fixture_shoe_rest_z(args, base_top_z=base_top_z)
    shoe_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, shoe_rest_z), wp.quat_identity()),
        mass=shoe_mass,
        com=wp.vec3(0.0, 0.0, 0.5 * midsole_top),
        inertia=_shoe_box_inertia(midsole_extents, shoe_mass),
        lock_inertia=True,
        is_kinematic=not bool(getattr(args, "fixture_dynamic_shoe", False)),
        label="shoe",
    )
    midsole_shape = builder.add_shape_mesh(body=shoe_body, mesh=midsole_mesh, cfg=compliant_mesh_cfg, label="midsole")
    midsole_stats["fixture"] = {
        "gravity_m_s2": gravity,
        "shoe_body": "dynamic" if bool(getattr(args, "fixture_dynamic_shoe", False)) else "kinematic_specimen",
        "shoe_mass_kg": shoe_mass,
        "bottom_platen_overlap_m": float(max(float(getattr(args, "fixture_bottom_platen_overlap", 0.0)), 0.0)),
        "shoe_horizontal_lock": bool(args.fixture_lock_shoe_horizontal),
        "shoe_free_rotation_axis": str(args.fixture_free_rotation_axis),
        "lock_rotation_after_settle": bool(args.fixture_lock_rotation_after_settle),
        "kd": float(args.kd),
        "compression_relax_steps": _compression_relax_steps(args),
        "quasistatic_replay": bool(args.fixture_quasistatic_replay),
        "bottom_platen": "static_box",
        "top_indenter": "kinematic",
    }

    base_shape = builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -base_hz), wp.quat_identity()),
        hx=max(0.12, 0.6 * float(midsole_extents[0])),
        hy=max(0.06, 0.8 * float(midsole_extents[1])),
        hz=base_hz,
        cfg=rigid_primitive_cfg,
        label="base",
    )

    rearfoot_local_top_z: float | None = None
    if test_case == "rearfoot":
        if cached_setup is not None:
            heel_x = cached_setup.heel_x
            local_top_z = cached_setup.rearfoot_local_top_z
            local_vertex_count = cached_setup.rearfoot_local_vertex_count
            assert local_top_z is not None
        else:
            heel_sign = -1.0 if args.heel_side == "min" else 1.0
            heel_x = heel_sign * 0.30 * float(midsole_extents[0])
            local_top_z, local_vertex_count = _local_top_z_under_radius(
                midsole_vertices,
                (heel_x, 0.0),
                float(args.punch_radius),
            )
        rearfoot_local_top_z = local_top_z
        indenter_body = builder.add_body(
            xform=wp.transform(wp.vec3(heel_x, 0.0, local_top_z), wp.quat_identity()),
            is_kinematic=True,
            label="rearfoot_punch_body",
        )
        indenter_shape = builder.add_shape_cylinder(
            body=indenter_body,
            radius=float(args.punch_radius),
            half_height=float(args.punch_half_height),
            cfg=rigid_primitive_cfg,
            label="rearfoot_punch",
        )
        indenter_rest_z = _rearfoot_indenter_rest_z(args, shoe_z=shoe_rest_z, local_top_z=local_top_z)
        midsole_stats["rearfoot_punch"] = {
            "center_m": [float(heel_x), 0.0],
            "radius_m": float(args.punch_radius),
            "local_top_z_m": local_top_z,
            "local_top_vertex_count": local_vertex_count,
            "global_top_z_m": midsole_top,
        }
    elif test_case == "fullfoot":
        indenter_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, midsole_top), wp.quat_identity()),
            is_kinematic=True,
            label="fullfoot_effector_body",
        )
        if cached_setup is not None:
            indenter_mesh = cached_setup.indenter_mesh
            indenter_stats = cached_setup.indenter_stats
        else:
            indenter_rotation = tuple(float(value) for value in args.fullfoot_rotation_deg)
            indenter_mesh, indenter_stats, _indenter_vertices = _load_newton_mesh(
                Path(args.indenter_mesh),
                scale=float(args.mesh_scale),
                sdf_resolution=int(args.sdf_resolution),
                narrow_band=float(args.narrow_band),
                rotation_degrees=indenter_rotation,
                z_max=float(args.fullfoot_indenter_fixture_crop_z)
                if bool(args.fullfoot_indenter_crop_fixture)
                else None,
            )
        indenter_shape = builder.add_shape_mesh(
            body=indenter_body,
            mesh=indenter_mesh,
            cfg=rigid_mesh_cfg,
            label="fullfoot_effector",
        )
        indenter_rest_z = _fullfoot_indenter_rest_z(args, shoe_z=shoe_rest_z, midsole_top=midsole_top)
        midsole_stats["indenter"] = indenter_stats
        midsole_stats["fullfoot_rotation_deg"] = [float(value) for value in args.fullfoot_rotation_deg]
        midsole_stats["fullfoot_start_clearance_m"] = float(args.fullfoot_start_clearance)
        midsole_stats["fullfoot_z_offset_m"] = float(args.fullfoot_z_offset)
        midsole_stats["fullfoot_indenter_crop_fixture"] = bool(args.fullfoot_indenter_crop_fixture)
        midsole_stats["fullfoot_indenter_fixture_crop_z_m"] = (
            float(args.fullfoot_indenter_fixture_crop_z) if bool(args.fullfoot_indenter_crop_fixture) else None
        )
    else:
        raise ValueError(f"Unknown test case: {test_case}")

    model = builder.finalize(device=wp.get_device())
    state = model.state()
    state_next = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_next)
    pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=int(args.rigid_contact_max),
        broad_phase="explicit",
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=not bool(args.no_contact_reduction),
            output_contact_surface=True,
            buffer_fraction=1.0,
            buffer_mult_contact=int(args.buffer_mult_contact),
            buffer_mult_iso=int(args.buffer_mult_iso),
            pressure_memory_enabled=bool(args.pressure_memory),
            pressure_memory_grid=tuple(int(v) for v in args.pressure_memory_grid),
            pressure_memory_unloading_loss=float(args.pressure_memory_unloading_loss),
            pressure_memory_recovery_tau_s=float(args.pressure_memory_recovery_tau_s),
            pressure_memory_dt_s=float(args.pressure_memory_dt_s),
        ),
    )
    contacts = pipeline.contacts()
    solver = _make_fixture_solver(args, model)

    scene = TrialScene(
        test_case=test_case,
        trace_csv=str(trace_csv),
        output_csv=str(output_csv),
        trace=trace,
        displacement_m=displacement_m,
        time_s=time_s,
        displacement_velocity_m_s=displacement_velocity_m_s,
        measured_force_n=measured_force_n,
        model=model,
        state=state,
        state_next=state_next,
        pipeline=pipeline,
        contacts=contacts,
        solver=solver,
        control=control,
        shoe_body=shoe_body,
        shoe_anchor_xy=(0.0, 0.0),
        shoe_anchor_quat=(0.0, 0.0, 0.0, 1.0),
        settled_rotation_locked=False,
        indenter_body=indenter_body,
        indenter_shape=indenter_shape,
        midsole_shape=midsole_shape,
        base_shape=base_shape,
        base_top_z=base_top_z,
        indenter_stop_z=_indenter_platen_stop_z(args, test_case=test_case, base_top_z=base_top_z),
        indenter_rest_z=indenter_rest_z,
        midsole_stats=midsole_stats,
    )

    _set_body_z_velocity(scene.state, scene.indenter_body, scene.indenter_rest_z, 0.0)
    midsole_stats.update(_settle_fixture(scene, args))
    settled_shoe_z = float(scene.state.body_q.numpy()[scene.shoe_body, 2])
    if test_case == "rearfoot":
        if rearfoot_local_top_z is None:
            raise RuntimeError("rearfoot local top height was not initialized.")
        indenter_rest_z = _rearfoot_indenter_rest_z(args, shoe_z=settled_shoe_z, local_top_z=rearfoot_local_top_z)
    elif test_case == "fullfoot":
        indenter_rest_z = _fullfoot_indenter_rest_z(args, shoe_z=settled_shoe_z, midsole_top=midsole_top)
    scene.indenter_rest_z = indenter_rest_z
    _set_body_z_velocity(scene.state, scene.indenter_body, scene.indenter_rest_z, 0.0)
    midsole_stats["fixture"]["settled_shoe_z_m"] = settled_shoe_z
    midsole_stats["fixture"]["indenter_rest_z_after_settle_m"] = float(scene.indenter_rest_z)
    midsole_stats["fixture"]["base_top_z_m"] = float(scene.base_top_z)
    midsole_stats["fixture"]["indenter_stop_z_m"] = float(scene.indenter_stop_z)
    midsole_stats["fixture"]["platen_stop_clearance_m"] = float(args.fixture_platen_stop_clearance)

    if test_case == "fullfoot":
        auto_offset, auto_stats = _find_fullfoot_contact_offset(
            args,
            pipeline,
            contacts,
            model,
            scene.state,
            indenter_body,
            indenter_shape,
            midsole_shape,
            indenter_rest_z,
            displacement_m,
            measured_force_n,
        )
        indenter_rest_z -= auto_offset
        indenter_rest_z, _, auto_stop_clamped = _clamp_indenter_to_platen_stop(
            indenter_rest_z, 0.0, scene.indenter_stop_z
        )
        scene.indenter_rest_z = indenter_rest_z
        _set_body_z_velocity(scene.state, scene.indenter_body, scene.indenter_rest_z, 0.0)
        midsole_stats.update(auto_stats)
        midsole_stats["fullfoot_auto_contact_platen_stop_active"] = int(auto_stop_clamped)

    return scene


def _simulate_trial_contact_response(
    args,
    *,
    test_case: str,
    trace_csv: Path,
    output_csv: Path,
    sample_stride: int = 1,
    cached_setup: CachedTrialSetup | None = None,
) -> TrialResponse:
    build_kwargs = {
        "test_case": test_case,
        "trace_csv": trace_csv,
        "output_csv": output_csv,
        "sample_stride": sample_stride,
    }
    if cached_setup is not None:
        build_kwargs["cached_setup"] = cached_setup
    scene = _build_trial_scene(args, **build_kwargs)

    contact_force_n = np.zeros_like(scene.displacement_m)
    contact_stats: list[dict[str, float | int]] = []
    for i, displacement in enumerate(scene.displacement_m):
        contact_force_n[i], step_contact_stats = _advance_and_measure_trial_force(
            scene,
            args,
            sample_index=i,
            displacement=float(displacement),
            displacement_velocity=float(scene.displacement_velocity_m_s[i]),
            relax_steps=_compression_relax_steps(args),
            include_diagnostics=False,
        )
        contact_stats.append(step_contact_stats)
        early_reason = _calibration_early_out_reason(
            args,
            test_case=scene.test_case,
            measured_force_n=scene.measured_force_n,
            contact_force_n=contact_force_n,
            contact_stats=contact_stats,
            sample_index=i,
        )
        if early_reason is not None:
            raise RuntimeError(early_reason)

    if bool(args.fail_on_zero_contact) and float(np.max(contact_force_n)) <= 1.0e-9:
        raise RuntimeError(
            f"{scene.test_case} generated no raw pressure-field contact force. "
            "This is a setup/contact-path issue, not a material-fit issue. "
            "Try a larger --fullfoot-contact-search-max or inspect mesh alignment/contact support."
        )

    return TrialResponse(
        test_case=scene.test_case,
        trace_csv=scene.trace_csv,
        output_csv=scene.output_csv,
        trace=scene.trace,
        displacement_m=scene.displacement_m,
        time_s=scene.time_s,
        displacement_velocity_m_s=scene.displacement_velocity_m_s,
        measured_force_n=scene.measured_force_n,
        raw_contact_force_n=contact_force_n,
        contact_stats=contact_stats,
        midsole_stats=scene.midsole_stats,
    )


def _calibration_early_out_reason(
    args,
    *,
    test_case: str,
    measured_force_n: np.ndarray,
    contact_force_n: np.ndarray,
    contact_stats: list[dict[str, float | int]],
    sample_index: int,
) -> str | None:
    if not bool(getattr(args, "calibration_early_out", False)):
        return None

    force = float(contact_force_n[sample_index])
    if not np.isfinite(force):
        return f"{test_case} candidate became unstable: non-finite simulated force at sample {sample_index}."

    measured_peak = max(float(np.max(measured_force_n)), 1.0e-9)
    force_multiplier = float(getattr(args, "calibration_early_out_force_multiplier", 0.0))
    if force_multiplier > 0.0 and abs(force) > force_multiplier * measured_peak:
        return (
            f"{test_case} candidate became unstable: simulated force {force:.6g} N exceeded "
            f"{force_multiplier:.6g}x measured peak {measured_peak:.6g} N at sample {sample_index}."
        )

    jump_limit = float(getattr(args, "calibration_early_out_force_jump_limit", 0.0))
    if jump_limit > 0.0 and sample_index > 0:
        jump = abs(force - float(contact_force_n[sample_index - 1]))
        if jump > jump_limit:
            return (
                f"{test_case} candidate became unstable: adjacent force jump {jump:.6g} N exceeded "
                f"{jump_limit:.6g} N at sample {sample_index}."
            )

    stats = contact_stats[-1] if contact_stats else {}
    max_vz = float(getattr(args, "calibration_early_out_max_shoe_vz", 0.0))
    if max_vz > 0.0 and abs(float(stats.get("shoe_vertical_velocity_m_s", 0.0))) > max_vz:
        return (
            f"{test_case} candidate became unstable: shoe vertical velocity "
            f"{float(stats.get('shoe_vertical_velocity_m_s', 0.0)):.6g} m/s exceeded {max_vz:.6g} m/s "
            f"at sample {sample_index}."
        )

    max_omega = float(getattr(args, "calibration_early_out_max_shoe_omega", 0.0))
    omega = abs(float(stats.get("shoe_free_axis_angular_velocity_rad_s", 0.0)))
    if max_omega > 0.0 and omega > max_omega:
        return (
            f"{test_case} candidate became unstable: shoe free-axis angular velocity {omega:.6g} rad/s exceeded "
            f"{max_omega:.6g} rad/s at sample {sample_index}."
        )

    return None


def _evaluate_trial_response(
    response: TrialResponse,
    args,
) -> dict[str, Any]:
    displacement_m = response.displacement_m
    measured_force_n = response.measured_force_n
    sim_force_n = response.raw_contact_force_n

    rows: list[dict[str, float | str]] = []
    for i, displacement in enumerate(displacement_m):
        step_contact_stats = response.contact_stats[i]
        rows.append(
            {
                "phase": float(response.trace["phase"][i]),
                "time_s": float(response.time_s[i]),
                "displacement_m": float(displacement),
                "displacement_mm": float(displacement * 1000.0),
                "displacement_velocity_m_s": float(response.displacement_velocity_m_s[i]),
                "measured_force_n": float(measured_force_n[i]),
                "raw_contact_force_n": float(response.raw_contact_force_n[i]),
                "sim_force_n": float(sim_force_n[i]),
                "force_error_n": float(sim_force_n[i] - measured_force_n[i]),
                "kh": float(args.kh),
                "kd": float(args.kd),
                **step_contact_stats,
            }
        )

    measured_peak = float(np.max(measured_force_n))
    sim_peak = float(np.max(sim_force_n))
    raw_contact_peak = float(np.max(response.raw_contact_force_n))
    objective, peak_error, loop_rmse = _force_fit_objective(displacement_m, measured_force_n, sim_force_n)
    metrics = _loop_metrics(displacement_m, measured_force_n, sim_force_n)
    jump_metrics = _force_jump_metrics(sim_force_n, float(args.force_jump_limit))
    motion_metrics = _residual_motion_metrics(response.contact_stats)
    state_model = _pressure_memory_state_model(args)

    result_csv = Path(response.output_csv)
    _write_sim_results(result_csv, rows)
    summary = {
        "mode": "run",
        "test_case": response.test_case,
        "trace_csv": response.trace_csv,
        "output_csv": str(result_csv),
        "midsole_stats": response.midsole_stats,
        **_contact_sample_stats(response.time_s),
        "kh": float(args.kh),
        "kd": float(args.kd),
        "fixture_solver": str(getattr(args, "fixture_solver", "")),
        "fixture_mujoco_nconmax": int(getattr(args, "fixture_mujoco_nconmax", 0)),
        "fixture_mujoco_njmax": int(getattr(args, "fixture_mujoco_njmax", 0)),
        "contact_reduction_enabled": not bool(getattr(args, "no_contact_reduction", False)),
        "rigid_contact_max": int(getattr(args, "rigid_contact_max", 0)),
        "buffer_mult_contact": int(getattr(args, "buffer_mult_contact", 0)),
        "buffer_mult_iso": int(getattr(args, "buffer_mult_iso", 0)),
        "measured_peak_force_n": measured_peak,
        "sim_peak_force_n": sim_peak,
        "raw_contact_peak_force_n": raw_contact_peak,
        "raw_contact_peak_relative_error": abs(raw_contact_peak - measured_peak) / max(measured_peak, 1.0e-9),
        "peak_relative_error": peak_error,
        "loop_rmse_n": loop_rmse,
        "force_r2": float(metrics["force_r2"]),
        "equal_weight_objective": objective,
        **jump_metrics,
        **motion_metrics,
        "state_model": state_model,
        "metrics": metrics,
        "max_rigid_contact_count": int(max(s["rigid_contact_count"] for s in response.contact_stats)),
        "max_pair_contact_count": int(max(s["pair_contact_count"] for s in response.contact_stats)),
        "max_active_pair_contact_count": int(max(s["active_pair_contact_count"] for s in response.contact_stats)),
        "max_top_contact_force_n": float(max(s["top_contact_force_n"] for s in response.contact_stats)),
        "max_bottom_contact_force_n": float(max(s["bottom_contact_force_n"] for s in response.contact_stats)),
        "max_bottom_active_pair_contact_count": int(
            max(s["bottom_active_pair_contact_count"] for s in response.contact_stats)
        ),
        "min_pair_depth_m": float(min(s["pair_depth_min_m"] for s in response.contact_stats)),
        "max_pair_stiffness_sum": float(max(s["pair_stiffness_sum"] for s in response.contact_stats)),
        "note": (
            "Pressure-field state model is memoryless; hysteresis loop area will require material state "
            "or dissipation beyond static kh fitting."
            if state_model["type"] == "memoryless_v1"
            else "Pressure-field state model uses max-compression memory with recovery before contact reduction export."
        ),
    }
    summary_path = result_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if bool(args.fail_on_force_jump) and int(jump_metrics["force_jump_violation_count"]) > 0:
        raise RuntimeError(
            f"{response.test_case} generated {jump_metrics['force_jump_violation_count']} adjacent force jumps "
            f"above {jump_metrics['force_jump_limit_n']:.6g} N. "
            f"Max jump {jump_metrics['max_adjacent_force_jump_n']:.6g} N at sample "
            f"{jump_metrics['max_adjacent_force_jump_index']} "
            f"({jump_metrics['max_adjacent_force_jump_from_n']:.6g} -> "
            f"{jump_metrics['max_adjacent_force_jump_to_n']:.6g} N)."
        )
    return summary


def _resolve_trial_paths(args, test_case: str) -> tuple[Path, Path]:
    trace_csv = Path(args.trace_csv) if args.trace_csv is not None else _default_trace_for_case(test_case)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None and args.mode == "run"
        else PROCESSED_DIR / f"{test_case}_sim_pressure_field.csv"
    )
    return trace_csv, output_csv


def run_digital_instron(args) -> dict[str, Any]:
    trace_csv, output_csv = _resolve_trial_paths(args, args.test_case)
    response = _simulate_trial_contact_response(
        args, test_case=args.test_case, trace_csv=trace_csv, output_csv=output_csv
    )
    return _evaluate_trial_response(response, args)


def _fit_shared_material_parameters(responses: list[TrialResponse], args) -> dict[str, Any]:
    objective, objective_components = _shared_material_objective_breakdown(responses, args)
    contact_offset_m = 0.0
    contact_threshold_n = 1.0
    contact_metadata_found = False
    for prefix in ("fullfoot_auto", "rearfoot_auto"):
        for response in responses:
            stats = response.midsole_stats or {}
            offset_key = f"{prefix}_contact_offset_m"
            threshold_key = f"{prefix}_contact_threshold_n"
            if offset_key in stats and threshold_key in stats:
                contact_offset_m = float(stats[offset_key])
                contact_threshold_n = float(stats[threshold_key])
                contact_metadata_found = True
                break
        if contact_metadata_found:
            break
    return {
        "kh": float(args.kh),
        "kd": float(args.kd),
        "layer_eta_lock": float(getattr(args, "layer_eta_lock", 0.65)),
        "layer_densification_power": float(getattr(args, "layer_densification_power", 0.0)),
        "contact_offset_m": contact_offset_m,
        "contact_threshold_n": contact_threshold_n,
        "combined_objective": float(objective),
        "objective_components": objective_components,
    }


def _simulate_calibration_responses(
    args, *, sample_stride: int = 1, cached_setup: CachedTrialSetup | None = None
) -> list[TrialResponse]:
    responses: list[TrialResponse] = []
    for test_case in [case.strip() for case in args.calibration_cases.split(",") if case.strip()]:
        if test_case not in {"rearfoot", "fullfoot"}:
            raise ValueError(f"Unknown calibration case: {test_case}")
        trace_csv, output_csv = _resolve_trial_paths(args, test_case)
        trial_kwargs = {
            "test_case": test_case,
            "trace_csv": trace_csv,
            "output_csv": output_csv,
            "sample_stride": sample_stride,
        }
        if cached_setup is not None:
            trial_kwargs["cached_setup"] = cached_setup
        responses.append(_simulate_trial_contact_response(args, **trial_kwargs))
    return responses


def _batched_rearfoot_trial_responses(
    args,
    candidates: list[MaterialCandidate],
    *,
    trace_csv: Path,
    output_csv: Path,
    sample_stride: int,
) -> list[TrialResponse]:
    if args.device:
        wp.set_device(args.device)
    if not wp.get_device().is_cuda:
        raise RuntimeError("digital_instron pressure-field run requires CUDA for SDF volumes.")
    if not candidates:
        return []

    trace, sample_indices = _stride_trace(_load_averaged_trace(trace_csv), sample_stride)
    displacement_m = trace["displacement_m"]
    time_s = trace["time_s"]
    displacement_velocity_m_s = np.gradient(displacement_m, time_s)
    measured_force_n = trace["force_n"]

    midsole_mesh, midsole_stats_base, midsole_vertices = _load_newton_mesh(
        Path(args.midsole_mesh),
        scale=float(args.mesh_scale),
        sdf_resolution=int(args.sdf_resolution),
        narrow_band=float(args.narrow_band),
    )
    midsole_extents = np.asarray(midsole_stats_base["extents_m"], dtype=np.float64)
    midsole_top = float(midsole_extents[2])
    heel_sign = -1.0 if args.heel_side == "min" else 1.0
    heel_x = heel_sign * 0.30 * float(midsole_extents[0])
    rearfoot_local_top_z, local_vertex_count = _local_top_z_under_radius(
        midsole_vertices,
        (heel_x, 0.0),
        float(args.punch_radius),
    )

    gravity = -9.81 if bool(args.fixture_gravity) else 0.0
    builder = newton.ModelBuilder(gravity=gravity)
    rigid_primitive_cfg = _make_rigid_cfg(
        sdf_resolution=int(args.sdf_resolution),
        narrow_band=float(args.narrow_band),
    )
    base_hz = 0.025
    base_top_z = 0.0
    shoe_rest_z = _fixture_shoe_rest_z(args, base_top_z=base_top_z)
    shoe_mass = float(args.shoe_mass)
    if shoe_mass <= 0.0:
        raise ValueError("--shoe-mass must be positive for the dynamic Digital Instron fixture.")

    world_records: list[dict[str, Any]] = []
    for world_index, candidate in enumerate(candidates):
        builder.begin_world()
        compliant_mesh_cfg = _make_compliant_cfg(
            float(candidate.kh),
            float(candidate.kd),
            narrow_band=float(args.narrow_band),
            pressure_profile=str(args.pressure_profile),
            pressure_layer_params=(
                float(candidate.layer_eta_lock),
                float(candidate.layer_densification_power),
                float(args.layer_cubic),
                float(args.layer_quintic),
            ),
            pressure_sine_amplitude=tuple(float(v) for v in args.pressure_sine_amplitude),
            pressure_sine_cycles=tuple(float(v) for v in args.pressure_sine_cycles),
            pressure_sine_phase=tuple(float(v) for v in args.pressure_sine_phase),
        )
        shoe_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, shoe_rest_z), wp.quat_identity()),
            mass=shoe_mass,
            com=wp.vec3(0.0, 0.0, 0.5 * midsole_top),
            inertia=_shoe_box_inertia(midsole_extents, shoe_mass),
            lock_inertia=True,
            is_kinematic=not bool(getattr(args, "fixture_dynamic_shoe", False)),
            label=f"candidate_{world_index}/shoe",
        )
        midsole_shape = builder.add_shape_mesh(
            body=shoe_body,
            mesh=midsole_mesh,
            cfg=compliant_mesh_cfg,
            label=f"candidate_{world_index}/midsole",
        )
        base_shape = builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -base_hz), wp.quat_identity()),
            hx=max(0.12, 0.6 * float(midsole_extents[0])),
            hy=max(0.06, 0.8 * float(midsole_extents[1])),
            hz=base_hz,
            cfg=rigid_primitive_cfg,
            label=f"candidate_{world_index}/base",
        )
        indenter_body = builder.add_body(
            xform=wp.transform(wp.vec3(heel_x, 0.0, rearfoot_local_top_z), wp.quat_identity()),
            is_kinematic=True,
            label=f"candidate_{world_index}/rearfoot_punch_body",
        )
        indenter_shape = builder.add_shape_cylinder(
            body=indenter_body,
            radius=float(args.punch_radius),
            half_height=float(args.punch_half_height),
            cfg=rigid_primitive_cfg,
            label=f"candidate_{world_index}/rearfoot_punch",
        )
        builder.end_world()
        world_records.append(
            {
                "candidate": candidate,
                "shoe_body": shoe_body,
                "midsole_shape": midsole_shape,
                "base_shape": base_shape,
                "indenter_body": indenter_body,
                "indenter_shape": indenter_shape,
                "indenter_rest_z": _rearfoot_indenter_rest_z(
                    args,
                    shoe_z=shoe_rest_z,
                    local_top_z=rearfoot_local_top_z,
                ),
            }
        )

    model = builder.finalize(device=wp.get_device())
    state = model.state()
    state_next = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_next)
    pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=int(args.rigid_contact_max),
        broad_phase="explicit",
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=not bool(args.no_contact_reduction),
            output_contact_surface=True,
            buffer_fraction=1.0,
            buffer_mult_contact=int(args.buffer_mult_contact),
            buffer_mult_iso=int(args.buffer_mult_iso),
            contact_buffer_fraction=float(getattr(args, "calibration_multiworld_contact_buffer_fraction", 0.25)),
            pressure_memory_enabled=bool(args.pressure_memory),
            pressure_memory_grid=tuple(int(v) for v in args.pressure_memory_grid),
            pressure_memory_unloading_loss=float(args.pressure_memory_unloading_loss),
            pressure_memory_recovery_tau_s=float(args.pressure_memory_recovery_tau_s),
            pressure_memory_dt_s=float(args.pressure_memory_dt_s),
        ),
    )
    contacts = pipeline.contacts()
    solver = _make_fixture_solver(args, model)

    scenes: list[TrialScene] = []
    for record in world_records:
        candidate = record["candidate"]
        midsole_stats = copy.deepcopy(midsole_stats_base)
        midsole_stats["pressure_profile"] = str(args.pressure_profile)
        midsole_stats["pressure_layer_params"] = {
            "eta_lock": float(candidate.layer_eta_lock),
            "densification_power": float(candidate.layer_densification_power),
            "cubic": float(args.layer_cubic),
            "quintic": float(args.layer_quintic),
        }
        midsole_stats["pressure_sine_amplitude"] = [float(v) for v in args.pressure_sine_amplitude]
        midsole_stats["pressure_sine_cycles"] = [float(v) for v in args.pressure_sine_cycles]
        midsole_stats["pressure_sine_phase"] = [float(v) for v in args.pressure_sine_phase]
        midsole_stats["trace_sample_stride"] = int(max(sample_stride, 1))
        midsole_stats["trace_sample_indices"] = sample_indices.astype(int).tolist()
        midsole_stats["fixture"] = {
            "gravity_m_s2": gravity,
            "shoe_body": "dynamic" if bool(getattr(args, "fixture_dynamic_shoe", False)) else "kinematic_specimen",
            "shoe_mass_kg": shoe_mass,
            "bottom_platen_overlap_m": float(max(float(getattr(args, "fixture_bottom_platen_overlap", 0.0)), 0.0)),
            "shoe_horizontal_lock": bool(args.fixture_lock_shoe_horizontal),
            "shoe_free_rotation_axis": str(args.fixture_free_rotation_axis),
            "lock_rotation_after_settle": bool(args.fixture_lock_rotation_after_settle),
            "kd": float(candidate.kd),
            "compression_relax_steps": _compression_relax_steps(args),
            "quasistatic_replay": bool(args.fixture_quasistatic_replay),
            "bottom_platen": "static_box",
            "top_indenter": "kinematic",
        }
        midsole_stats["rearfoot_punch"] = {
            "center_m": [float(heel_x), 0.0],
            "radius_m": float(args.punch_radius),
            "local_top_z_m": float(rearfoot_local_top_z),
            "local_top_vertex_count": int(local_vertex_count),
            "global_top_z_m": midsole_top,
        }
        scene = TrialScene(
            test_case="rearfoot",
            trace_csv=str(trace_csv),
            output_csv=str(output_csv),
            trace=trace,
            displacement_m=displacement_m,
            time_s=time_s,
            displacement_velocity_m_s=displacement_velocity_m_s,
            measured_force_n=measured_force_n,
            model=model,
            state=state,
            state_next=state_next,
            pipeline=pipeline,
            contacts=contacts,
            solver=solver,
            control=control,
            shoe_body=int(record["shoe_body"]),
            shoe_anchor_xy=(0.0, 0.0),
            shoe_anchor_quat=(0.0, 0.0, 0.0, 1.0),
            settled_rotation_locked=False,
            indenter_body=int(record["indenter_body"]),
            indenter_shape=int(record["indenter_shape"]),
            midsole_shape=int(record["midsole_shape"]),
            base_shape=int(record["base_shape"]),
            base_top_z=base_top_z,
            indenter_stop_z=_indenter_platen_stop_z(args, test_case="rearfoot", base_top_z=base_top_z),
            indenter_rest_z=float(record["indenter_rest_z"]),
            midsole_stats=midsole_stats,
        )
        _set_body_z_velocity(scene.state, scene.indenter_body, scene.indenter_rest_z, 0.0)
        scenes.append(scene)

    settle_stats = _settle_fixture_many(scenes, args)
    body_q = state.body_q.numpy()
    for scene, stats in zip(scenes, settle_stats, strict=True):
        scene.midsole_stats.update(stats)
        settled_shoe_z = float(body_q[scene.shoe_body, 2])
        scene.indenter_rest_z = _rearfoot_indenter_rest_z(args, shoe_z=settled_shoe_z, local_top_z=rearfoot_local_top_z)
        _set_body_z_velocity(scene.state, scene.indenter_body, scene.indenter_rest_z, 0.0)
        scene.midsole_stats["fixture"]["settled_shoe_z_m"] = settled_shoe_z
        scene.midsole_stats["fixture"]["indenter_rest_z_after_settle_m"] = float(scene.indenter_rest_z)
        scene.midsole_stats["fixture"]["base_top_z_m"] = float(scene.base_top_z)
        scene.midsole_stats["fixture"]["indenter_stop_z_m"] = float(scene.indenter_stop_z)
        scene.midsole_stats["fixture"]["platen_stop_clearance_m"] = float(args.fixture_platen_stop_clearance)

    contact_force_n = [np.zeros_like(displacement_m) for _scene in scenes]
    contact_stats: list[list[dict[str, float | int]]] = [[] for _scene in scenes]
    for sample_index, displacement in enumerate(displacement_m):
        _fixture_step_many(
            scenes,
            args,
            indenter_z=[float(scene.indenter_rest_z - float(displacement)) for scene in scenes],
            indenter_velocity_z=[-float(displacement_velocity_m_s[sample_index]) for _scene in scenes],
            dt=_sample_dt(time_s, sample_index),
            substeps=_compression_relax_steps(args),
        )
        if bool(args.fixture_quasistatic_replay):
            for scene in scenes:
                _stabilize_shoe_fixture(scene, args, zero_vertical_velocity=True)
                _set_body_z_velocity(scene.state, scene.indenter_body, scene.indenter_rest_z - float(displacement), 0.0)
        scenes[0].pipeline.collide(scenes[0].state, scenes[0].contacts)
        for scene_index, scene in enumerate(scenes):
            force, stats = _measure_trial_force(scene, include_diagnostics=False)
            stats.update(_shoe_motion_stats(scene, args))
            stats["indenter_platen_stop_active"] = int(
                scene.indenter_rest_z - float(displacement) < scene.indenter_stop_z
            )
            stats["indenter_command_z_m"] = float(scene.indenter_rest_z - float(displacement))
            stats["indenter_applied_z_m"] = float(
                max(scene.indenter_rest_z - float(displacement), scene.indenter_stop_z)
            )
            stats["indenter_stop_z_m"] = float(scene.indenter_stop_z)
            contact_force_n[scene_index][sample_index] = force
            contact_stats[scene_index].append(stats)

    responses = []
    for scene_index, scene in enumerate(scenes):
        if bool(args.fail_on_zero_contact) and float(np.max(contact_force_n[scene_index])) <= 1.0e-9:
            raise RuntimeError(
                f"{scene.test_case} generated no raw pressure-field contact force. "
                "This is a setup/contact-path issue, not a material-fit issue."
            )
        responses.append(
            TrialResponse(
                test_case=scene.test_case,
                trace_csv=scene.trace_csv,
                output_csv=scene.output_csv,
                trace=scene.trace,
                displacement_m=scene.displacement_m,
                time_s=scene.time_s,
                displacement_velocity_m_s=scene.displacement_velocity_m_s,
                measured_force_n=scene.measured_force_n,
                raw_contact_force_n=contact_force_n[scene_index],
                contact_stats=contact_stats[scene_index],
                midsole_stats=scene.midsole_stats,
            )
        )
    return responses


def _batched_calibration_trial_responses(
    args,
    candidates: list[MaterialCandidate],
    *,
    cases: list[str],
    sample_stride: int,
    viewer: Any | None = None,
    cached_setup: CachedTrialSetup | None = None,
) -> MaterialBatchTrialResponses:
    if args.device:
        wp.set_device(args.device)
    if not wp.get_device().is_cuda:
        raise RuntimeError("digital_instron pressure-field run requires CUDA for SDF volumes.")
    if not candidates:
        return MaterialBatchTrialResponses([], {})
    if not cases:
        raise ValueError("At least one calibration case is required for multiworld batching.")

    if cached_setup is not None:
        traces = cached_setup.traces
        midsole_mesh = cached_setup.midsole_mesh
        midsole_extents = cached_setup.midsole_extents
        midsole_top = cached_setup.midsole_top
        heel_x = cached_setup.heel_x
        rearfoot_local_top_z = cached_setup.rearfoot_local_top_z
        indenter_mesh = cached_setup.indenter_mesh
        gravity = cached_setup.gravity
    else:
        traces: dict[str, dict[str, Any]] = {}
        for test_case in cases:
            if test_case not in {"rearfoot", "fullfoot"}:
                raise ValueError(f"Unknown calibration case: {test_case}")
            trace_csv, output_csv = _resolve_trial_paths(args, test_case)
            trace, sample_indices = _stride_trace(_load_averaged_trace(trace_csv), sample_stride)
            displacement_m = trace["displacement_m"]
            time_s = trace["time_s"]
            traces[test_case] = {
                "trace_csv": trace_csv,
                "output_csv": output_csv,
                "trace": trace,
                "sample_indices": sample_indices,
                "displacement_m": displacement_m,
                "time_s": time_s,
                "displacement_velocity_m_s": np.gradient(displacement_m, time_s),
                "measured_force_n": trace["force_n"],
            }

        midsole_mesh, midsole_stats_base, midsole_vertices = _load_newton_mesh(
            Path(args.midsole_mesh),
            scale=float(args.mesh_scale),
            sdf_resolution=int(args.sdf_resolution),
            narrow_band=float(args.narrow_band),
        )
        midsole_extents = np.asarray(midsole_stats_base["extents_m"], dtype=np.float64)
        midsole_top = float(midsole_extents[2])
        heel_sign = -1.0 if args.heel_side == "min" else 1.0
        heel_x = heel_sign * 0.30 * float(midsole_extents[0])
        rearfoot_local_top_z, rearfoot_local_vertex_count = _local_top_z_under_radius(
            midsole_vertices,
            (heel_x, 0.0),
            float(args.punch_radius),
        )

        indenter_mesh = None
        indenter_stats = None
        if "fullfoot" in cases:
            indenter_rotation = tuple(float(value) for value in args.fullfoot_rotation_deg)
            indenter_mesh, indenter_stats, _indenter_vertices = _load_newton_mesh(
                Path(args.indenter_mesh),
                scale=float(args.mesh_scale),
                sdf_resolution=int(args.sdf_resolution),
                narrow_band=float(args.narrow_band),
                rotation_degrees=indenter_rotation,
                z_max=float(args.fullfoot_indenter_fixture_crop_z)
                if bool(args.fullfoot_indenter_crop_fixture)
                else None,
            )

        gravity = -9.81 if bool(args.fixture_gravity) else 0.0
    builder = newton.ModelBuilder(gravity=gravity)
    rigid_mesh_cfg = _make_rigid_cfg(narrow_band=float(args.narrow_band))
    rigid_primitive_cfg = _make_rigid_cfg(
        sdf_resolution=int(args.sdf_resolution),
        narrow_band=float(args.narrow_band),
    )
    base_hz = 0.025
    base_top_z = 0.0
    shoe_rest_z = _fixture_shoe_rest_z(args, base_top_z=base_top_z)
    shoe_mass = float(args.shoe_mass)
    if shoe_mass <= 0.0:
        raise ValueError("--shoe-mass must be positive for the dynamic Digital Instron fixture.")

    records_by_candidate: list[list[dict[str, Any]]] = []
    for candidate_index, candidate in enumerate(candidates):
        builder.begin_world()
        compliant_mesh_cfg = _make_compliant_cfg(
            float(candidate.kh),
            float(candidate.kd),
            narrow_band=float(args.narrow_band),
            pressure_profile=str(args.pressure_profile),
            pressure_layer_params=(
                float(candidate.layer_eta_lock),
                float(candidate.layer_densification_power),
                float(args.layer_cubic),
                float(args.layer_quintic),
            ),
            pressure_sine_amplitude=tuple(float(v) for v in args.pressure_sine_amplitude),
            pressure_sine_cycles=tuple(float(v) for v in args.pressure_sine_cycles),
            pressure_sine_phase=tuple(float(v) for v in args.pressure_sine_phase),
        )
        candidate_records: list[dict[str, Any]] = []
        fixture_spacing_y = max(0.25, 3.0 * float(midsole_extents[1]))
        for case_index, test_case in enumerate(cases):
            label_prefix = f"candidate_{candidate_index}/{test_case}"
            fixture_origin_x = 0.0
            fixture_origin_y = fixture_spacing_y * float(case_index)
            shoe_body = builder.add_body(
                xform=wp.transform(wp.vec3(fixture_origin_x, fixture_origin_y, shoe_rest_z), wp.quat_identity()),
                mass=shoe_mass,
                com=wp.vec3(0.0, 0.0, 0.5 * midsole_top),
                inertia=_shoe_box_inertia(midsole_extents, shoe_mass),
                lock_inertia=True,
                is_kinematic=not bool(getattr(args, "fixture_dynamic_shoe", False)),
                label=f"{label_prefix}/shoe",
            )
            midsole_shape = builder.add_shape_mesh(
                body=shoe_body,
                mesh=midsole_mesh,
                cfg=compliant_mesh_cfg,
                label=f"{label_prefix}/midsole",
            )
            base_shape = builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(fixture_origin_x, fixture_origin_y, -base_hz), wp.quat_identity()),
                hx=max(0.12, 0.6 * float(midsole_extents[0])),
                hy=max(0.06, 0.8 * float(midsole_extents[1])),
                hz=base_hz,
                cfg=rigid_primitive_cfg,
                label=f"{label_prefix}/base",
            )
            if test_case == "rearfoot":
                indenter_body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3(fixture_origin_x + heel_x, fixture_origin_y, rearfoot_local_top_z),
                        wp.quat_identity(),
                    ),
                    is_kinematic=True,
                    label=f"{label_prefix}/punch_body",
                )
                indenter_shape = builder.add_shape_cylinder(
                    body=indenter_body,
                    radius=float(args.punch_radius),
                    half_height=float(args.punch_half_height),
                    cfg=rigid_primitive_cfg,
                    label=f"{label_prefix}/punch",
                )
                indenter_rest_z = _rearfoot_indenter_rest_z(
                    args,
                    shoe_z=shoe_rest_z,
                    local_top_z=rearfoot_local_top_z,
                )
            elif test_case == "fullfoot":
                if indenter_mesh is None:
                    raise RuntimeError("fullfoot indenter mesh was not initialized.")
                indenter_body = builder.add_body(
                    xform=wp.transform(wp.vec3(fixture_origin_x, fixture_origin_y, midsole_top), wp.quat_identity()),
                    is_kinematic=True,
                    label=f"{label_prefix}/effector_body",
                )
                indenter_shape = builder.add_shape_mesh(
                    body=indenter_body,
                    mesh=indenter_mesh,
                    cfg=rigid_mesh_cfg,
                    label=f"{label_prefix}/effector",
                )
                indenter_rest_z = _fullfoot_indenter_rest_z(args, shoe_z=shoe_rest_z, midsole_top=midsole_top)
            else:
                raise ValueError(f"Unknown test case: {test_case}")
            candidate_records.append(
                {
                    "candidate": candidate,
                    "candidate_index": candidate_index,
                    "test_case": test_case,
                    "shoe_body": shoe_body,
                    "midsole_shape": midsole_shape,
                    "base_shape": base_shape,
                    "indenter_body": indenter_body,
                    "indenter_shape": indenter_shape,
                    "indenter_rest_z": indenter_rest_z,
                    "fixture_origin_xy": (fixture_origin_x, fixture_origin_y),
                }
            )
        builder.end_world()
        records_by_candidate.append(candidate_records)

    model = builder.finalize(device=wp.get_device())
    state = model.state()
    state_next = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_next)
    pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=int(args.rigid_contact_max),
        broad_phase="explicit",
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=not bool(args.no_contact_reduction),
            output_contact_surface=True,
            buffer_fraction=1.0,
            buffer_mult_contact=int(args.buffer_mult_contact),
            buffer_mult_iso=int(args.buffer_mult_iso),
            contact_buffer_fraction=float(getattr(args, "calibration_multiworld_contact_buffer_fraction", 0.25)),
            pressure_memory_enabled=bool(args.pressure_memory),
            pressure_memory_grid=tuple(int(v) for v in args.pressure_memory_grid),
            pressure_memory_unloading_loss=float(args.pressure_memory_unloading_loss),
            pressure_memory_recovery_tau_s=float(args.pressure_memory_recovery_tau_s),
            pressure_memory_dt_s=float(args.pressure_memory_dt_s),
        ),
    )
    contacts = pipeline.contacts()
    solver = _make_fixture_solver(args, model)
    if viewer is not None:
        viewer.set_model(model)
        viewer.show_contacts = True
        viewer.show_hydro_contact_surface = True
        viewer.set_camera(pos=wp.vec3(0.45, -0.65, 0.35), pitch=-30.0, yaw=135.0)

    scenes_by_candidate: list[list[TrialScene]] = []
    all_scenes: list[TrialScene] = []
    for candidate_records in records_by_candidate:
        candidate_scenes: list[TrialScene] = []
        for record in candidate_records:
            candidate = record["candidate"]
            test_case = str(record["test_case"])
            trace_data = traces[test_case]
            midsole_stats = copy.deepcopy(midsole_stats_base)
            midsole_stats["pressure_profile"] = str(args.pressure_profile)
            midsole_stats["pressure_layer_params"] = {
                "eta_lock": float(candidate.layer_eta_lock),
                "densification_power": float(candidate.layer_densification_power),
                "cubic": float(args.layer_cubic),
                "quintic": float(args.layer_quintic),
            }
            midsole_stats["pressure_sine_amplitude"] = [float(v) for v in args.pressure_sine_amplitude]
            midsole_stats["pressure_sine_cycles"] = [float(v) for v in args.pressure_sine_cycles]
            midsole_stats["pressure_sine_phase"] = [float(v) for v in args.pressure_sine_phase]
            midsole_stats["trace_sample_stride"] = int(max(sample_stride, 1))
            midsole_stats["trace_sample_indices"] = trace_data["sample_indices"].astype(int).tolist()
            midsole_stats["fixture"] = {
                "gravity_m_s2": gravity,
                "shoe_body": "dynamic" if bool(getattr(args, "fixture_dynamic_shoe", False)) else "kinematic_specimen",
                "shoe_mass_kg": shoe_mass,
                "bottom_platen_overlap_m": float(max(float(getattr(args, "fixture_bottom_platen_overlap", 0.0)), 0.0)),
                "shoe_horizontal_lock": bool(args.fixture_lock_shoe_horizontal),
                "shoe_free_rotation_axis": str(args.fixture_free_rotation_axis),
                "lock_rotation_after_settle": bool(args.fixture_lock_rotation_after_settle),
                "kd": float(candidate.kd),
                "compression_relax_steps": _compression_relax_steps(args),
                "quasistatic_replay": bool(args.fixture_quasistatic_replay),
                "bottom_platen": "static_box",
                "top_indenter": "kinematic",
                "candidate_world": int(record["candidate_index"]),
            }
            if test_case == "rearfoot":
                midsole_stats["rearfoot_punch"] = {
                    "center_m": [float(heel_x), 0.0],
                    "radius_m": float(args.punch_radius),
                    "local_top_z_m": float(rearfoot_local_top_z),
                    "local_top_vertex_count": int(rearfoot_local_vertex_count),
                    "global_top_z_m": midsole_top,
                }
            else:
                midsole_stats["indenter"] = copy.deepcopy(indenter_stats)
                midsole_stats["fullfoot_rotation_deg"] = [float(value) for value in args.fullfoot_rotation_deg]
                midsole_stats["fullfoot_start_clearance_m"] = float(args.fullfoot_start_clearance)
                midsole_stats["fullfoot_z_offset_m"] = float(args.fullfoot_z_offset)
                midsole_stats["fullfoot_indenter_crop_fixture"] = bool(args.fullfoot_indenter_crop_fixture)
                midsole_stats["fullfoot_indenter_fixture_crop_z_m"] = (
                    float(args.fullfoot_indenter_fixture_crop_z) if bool(args.fullfoot_indenter_crop_fixture) else None
                )
            scene = TrialScene(
                test_case=test_case,
                trace_csv=str(trace_data["trace_csv"]),
                output_csv=str(trace_data["output_csv"]),
                trace=trace_data["trace"],
                displacement_m=trace_data["displacement_m"],
                time_s=trace_data["time_s"],
                displacement_velocity_m_s=trace_data["displacement_velocity_m_s"],
                measured_force_n=trace_data["measured_force_n"],
                model=model,
                state=state,
                state_next=state_next,
                pipeline=pipeline,
                contacts=contacts,
                solver=solver,
                control=control,
                shoe_body=int(record["shoe_body"]),
                shoe_anchor_xy=tuple(float(v) for v in record["fixture_origin_xy"]),
                shoe_anchor_quat=(0.0, 0.0, 0.0, 1.0),
                settled_rotation_locked=False,
                indenter_body=int(record["indenter_body"]),
                indenter_shape=int(record["indenter_shape"]),
                midsole_shape=int(record["midsole_shape"]),
                base_shape=int(record["base_shape"]),
                base_top_z=base_top_z,
                indenter_stop_z=_indenter_platen_stop_z(args, test_case=test_case, base_top_z=base_top_z),
                indenter_rest_z=float(record["indenter_rest_z"]),
                midsole_stats=midsole_stats,
                candidate_index=int(record["candidate_index"]),
            )
            _set_body_z_velocity(scene.state, scene.indenter_body, scene.indenter_rest_z, 0.0)
            candidate_scenes.append(scene)
            all_scenes.append(scene)
        scenes_by_candidate.append(candidate_scenes)

    settle_stats = _settle_fixture_many(all_scenes, args)
    body_q = state.body_q.numpy()
    for scene, stats in zip(all_scenes, settle_stats, strict=True):
        scene.midsole_stats.update(stats)
        settled_shoe_z = float(body_q[scene.shoe_body, 2])
        if scene.test_case == "rearfoot":
            scene.indenter_rest_z = _rearfoot_indenter_rest_z(
                args,
                shoe_z=settled_shoe_z,
                local_top_z=rearfoot_local_top_z,
            )
        else:
            scene.indenter_rest_z = _fullfoot_indenter_rest_z(args, shoe_z=settled_shoe_z, midsole_top=midsole_top)
        _set_body_z_velocity(scene.state, scene.indenter_body, scene.indenter_rest_z, 0.0)
        scene.midsole_stats["fixture"]["settled_shoe_z_m"] = settled_shoe_z
        scene.midsole_stats["fixture"]["indenter_rest_z_after_settle_m"] = float(scene.indenter_rest_z)
        scene.midsole_stats["fixture"]["base_top_z_m"] = float(scene.base_top_z)
        scene.midsole_stats["fixture"]["indenter_stop_z_m"] = float(scene.indenter_stop_z)
        scene.midsole_stats["fixture"]["platen_stop_clearance_m"] = float(args.fixture_platen_stop_clearance)

    for scene in all_scenes:
        auto_offset, auto_stats = _find_trial_contact_offset(
            args,
            test_case=scene.test_case,
            pipeline=pipeline,
            contacts=contacts,
            model=model,
            state=scene.state,
            indenter_body=scene.indenter_body,
            indenter_shape=scene.indenter_shape,
            midsole_shape=scene.midsole_shape,
            indenter_rest_z=scene.indenter_rest_z,
            displacement_m=scene.displacement_m,
            measured_force_n=scene.measured_force_n,
        )
        scene.indenter_rest_z -= auto_offset
        scene.indenter_rest_z, _, auto_stop_clamped = _clamp_indenter_to_platen_stop(
            scene.indenter_rest_z,
            0.0,
            scene.indenter_stop_z,
        )
        _set_body_z_velocity(scene.state, scene.indenter_body, scene.indenter_rest_z, 0.0)
        scene.midsole_stats.update(auto_stats)
        scene.midsole_stats[f"{scene.test_case}_auto_contact_platen_stop_active"] = int(auto_stop_clamped)

    responses_by_scene: dict[int, TrialResponse] = {}
    invalid_candidates: dict[int, dict[str, Any]] = {}

    def mark_candidate_invalid(candidate_index: int, invalid_reason: str, sample_index: int | None) -> None:
        invalid_candidates.setdefault(
            candidate_index,
            _invalid_material_candidate(
                candidates[candidate_index],
                invalid_reason,
                divergence_sample_index=sample_index,
            ),
        )

    def divergent_candidate_indices(scenes: list[TrialScene]) -> list[int]:
        body_q = scenes[0].state_next.body_q.numpy()
        return sorted(
            {
                scene.candidate_index
                for scene in scenes
                if not np.isfinite(body_q[_scene_solver_body_indices(scene)]).all()
            }
        )

    for test_case in cases:
        case_scenes = [scene for scene in all_scenes if scene.test_case == test_case]
        if not case_scenes:
            continue
        displacement_m = case_scenes[0].displacement_m
        time_s = case_scenes[0].time_s
        displacement_velocity_m_s = case_scenes[0].displacement_velocity_m_s
        contact_force_n = [np.zeros_like(displacement_m) for _scene in case_scenes]
        contact_stats: list[list[dict[str, float | int]]] = [[] for _scene in case_scenes]
        for sample_index, displacement in enumerate(displacement_m):
            if viewer is not None and not viewer.is_running():
                break
            active_scene_indices = [
                scene_index
                for scene_index, scene in enumerate(case_scenes)
                if scene.candidate_index not in invalid_candidates
            ]
            if not active_scene_indices:
                break
            active_scenes = [case_scenes[scene_index] for scene_index in active_scene_indices]
            relax_steps = _compression_relax_steps(args)
            try:
                if relax_steps > 0:
                    _fixture_step_many(
                        active_scenes,
                        args,
                        indenter_z=[float(scene.indenter_rest_z - float(displacement)) for scene in active_scenes],
                        indenter_velocity_z=[
                            -float(displacement_velocity_m_s[sample_index]) for _scene in active_scenes
                        ],
                        dt=_sample_dt(time_s, sample_index),
                        substeps=relax_steps,
                    )
                    if bool(args.fixture_quasistatic_replay):
                        for scene in active_scenes:
                            _stabilize_shoe_fixture(scene, args, zero_vertical_velocity=True)
                            _set_body_z_velocity(
                                scene.state,
                                scene.indenter_body,
                                scene.indenter_rest_z - float(displacement),
                                0.0,
                            )
                else:
                    for scene in active_scenes:
                        _set_body_z_velocity(
                            scene.state,
                            scene.indenter_body,
                            scene.indenter_rest_z - float(displacement),
                            -float(displacement_velocity_m_s[sample_index]),
                        )
            except RuntimeError as exc:
                if "Solver diverged" not in str(exc):
                    raise
                failed_indices = divergent_candidate_indices(active_scenes) or [
                    scene.candidate_index for scene in active_scenes
                ]
                for candidate_index in failed_indices:
                    mark_candidate_invalid(candidate_index, "solver diverged", sample_index)
                continue
            case_scenes[0].pipeline.collide(case_scenes[0].state, case_scenes[0].contacts)
            for scene_index, scene in enumerate(case_scenes):
                if scene.candidate_index in invalid_candidates:
                    continue
                force, stats = _measure_trial_force(scene, include_diagnostics=False)
                stats.update(_shoe_motion_stats(scene, args))
                stats["indenter_platen_stop_active"] = int(
                    scene.indenter_rest_z - float(displacement) < scene.indenter_stop_z
                )
                stats["indenter_command_z_m"] = float(scene.indenter_rest_z - float(displacement))
                stats["indenter_applied_z_m"] = float(
                    max(scene.indenter_rest_z - float(displacement), scene.indenter_stop_z)
                )
                stats["indenter_stop_z_m"] = float(scene.indenter_stop_z)
                contact_force_n[scene_index][sample_index] = force
                contact_stats[scene_index].append(stats)
                early_reason = _calibration_early_out_reason(
                    args,
                    test_case=scene.test_case,
                    sample_index=sample_index,
                    measured_force_n=scene.measured_force_n,
                    contact_force_n=contact_force_n[scene_index],
                    contact_stats=contact_stats[scene_index],
                )
                if early_reason is not None:
                    mark_candidate_invalid(scene.candidate_index, _candidate_invalid_reason(early_reason), sample_index)
            if viewer is not None:
                viewer.log_scalar(f"CalibrationBatch/{test_case}/sample_index", float(sample_index))
                viewer.log_scalar(f"CalibrationBatch/{test_case}/displacement_mm", float(displacement * 1000.0))
                for scene_index, scene in enumerate(case_scenes):
                    if scene.candidate_index in invalid_candidates:
                        continue
                    viewer.log_scalar(
                        f"CalibrationBatch/{test_case}/candidate_{scene.candidate_index}/measured_force_n",
                        float(scene.measured_force_n[sample_index]),
                    )
                    viewer.log_scalar(
                        f"CalibrationBatch/{test_case}/candidate_{scene.candidate_index}/raw_contact_force_n",
                        float(contact_force_n[scene_index][sample_index]),
                    )
                viewer.begin_frame(float(case_scenes[0].time_s[sample_index]))
                viewer.log_state(case_scenes[0].state)
                viewer.log_contacts(case_scenes[0].contacts, case_scenes[0].state)
                if case_scenes[0].pipeline.hydroelastic_sdf is not None:
                    viewer.log_hydro_contact_surface(case_scenes[0].pipeline.hydroelastic_sdf.get_contact_surface())
                viewer.end_frame()
                if viewer.is_paused():
                    continue

        for scene_index, scene in enumerate(case_scenes):
            if scene.candidate_index in invalid_candidates:
                continue
            responses_by_scene[id(scene)] = TrialResponse(
                test_case=scene.test_case,
                trace_csv=scene.trace_csv,
                output_csv=scene.output_csv,
                trace=scene.trace,
                displacement_m=scene.displacement_m,
                time_s=scene.time_s,
                displacement_velocity_m_s=scene.displacement_velocity_m_s,
                measured_force_n=scene.measured_force_n,
                raw_contact_force_n=contact_force_n[scene_index],
                contact_stats=contact_stats[scene_index],
                midsole_stats=scene.midsole_stats,
            )

    if viewer is not None and bool(getattr(args, "view_loop", True)) and all_scenes:
        hold_scene = all_scenes[0]
        hold_time = float(max(scene.time_s[-1] for scene in all_scenes if scene.time_s.size))
        hold_dt = 1.0 / 60.0
        hold_frame = 0
        while viewer.is_running():
            hold_scene.pipeline.collide(hold_scene.state, hold_scene.contacts)
            viewer.begin_frame(hold_time + hold_frame * hold_dt)
            viewer.log_state(hold_scene.state)
            viewer.log_contacts(hold_scene.contacts, hold_scene.state)
            if hold_scene.pipeline.hydroelastic_sdf is not None:
                viewer.log_hydro_contact_surface(hold_scene.pipeline.hydroelastic_sdf.get_contact_surface())
            viewer.end_frame()
            if not viewer.is_paused():
                hold_frame += 1

    responses_by_candidate = [
        [responses_by_scene[id(scene)] for scene in candidate_scenes if id(scene) in responses_by_scene]
        for candidate_scenes in scenes_by_candidate
    ]
    return MaterialBatchTrialResponses(responses_by_candidate, invalid_candidates)


def _fit_range(enabled: bool, center: float, minimum: float, maximum: float) -> tuple[float, float]:
    if not enabled:
        return float(center), float(center)
    lower = float(min(minimum, maximum))
    upper = float(max(minimum, maximum))
    if lower == upper:
        return lower, upper
    return lower, upper


def _fit_values(
    *,
    enabled: bool,
    center: float,
    minimum: float,
    maximum: float,
    steps: int,
    log_space: bool = False,
) -> np.ndarray:
    if not enabled:
        return np.asarray([float(center)], dtype=np.float64)
    lower, upper = _fit_range(True, center, minimum, maximum)
    steps = max(int(steps), 1)
    if steps == 1 or lower == upper:
        if log_space and lower > 0.0 and upper > 0.0:
            return np.asarray([float(np.sqrt(lower * upper))], dtype=np.float64)
        return np.asarray([0.5 * (lower + upper)], dtype=np.float64)
    if log_space and lower > 0.0 and upper > 0.0:
        return np.geomspace(lower, upper, steps)
    return np.linspace(lower, upper, steps)


def _material_candidate_grid(
    args,
    *,
    kh_min: float | None = None,
    kh_max: float | None = None,
    kd_min: float | None = None,
    kd_max: float | None = None,
    layer_eta_lock_min: float | None = None,
    layer_eta_lock_max: float | None = None,
    layer_densification_power_min: float | None = None,
    layer_densification_power_max: float | None = None,
    log_kh: bool | None = None,
) -> list[MaterialCandidate]:
    if log_kh is None:
        log_kh = str(getattr(args, "calibration_kh_spacing", "linear")) == "log"
    kh_lower, kh_upper = _fit_range(
        bool(args.fit_kh),
        float(args.kh),
        float(args.fit_kh_min if kh_min is None else kh_min),
        float(args.fit_kh_max if kh_max is None else kh_max),
    )
    kd_lower, kd_upper = _fit_range(
        bool(getattr(args, "fit_kd", True)),
        float(args.kd),
        float(getattr(args, "fit_kd_min", 0.0) if kd_min is None else kd_min),
        float(getattr(args, "fit_kd_max", 50.0) if kd_max is None else kd_max),
    )
    eta_lower, eta_upper = _fit_range(
        bool(getattr(args, "fit_layer_eta_lock", False)),
        float(args.layer_eta_lock),
        float(getattr(args, "fit_layer_eta_lock_min", 0.45) if layer_eta_lock_min is None else layer_eta_lock_min),
        float(getattr(args, "fit_layer_eta_lock_max", 0.85) if layer_eta_lock_max is None else layer_eta_lock_max),
    )
    densification_lower, densification_upper = _fit_range(
        bool(getattr(args, "fit_layer_densification_power", False)),
        float(args.layer_densification_power),
        float(
            getattr(args, "fit_layer_densification_power_min", 0.0)
            if layer_densification_power_min is None
            else layer_densification_power_min
        ),
        float(
            getattr(args, "fit_layer_densification_power_max", 2.0)
            if layer_densification_power_max is None
            else layer_densification_power_max
        ),
    )
    kh_values = (
        _fit_values(
            enabled=True,
            center=float(args.kh),
            minimum=kh_lower,
            maximum=kh_upper,
            steps=int(args.fit_kh_steps),
            log_space=bool(log_kh),
        )
        if args.fit_kh
        else np.asarray([float(args.kh)], dtype=np.float64)
    )
    kd_values = (
        _fit_values(
            enabled=True,
            center=float(args.kd),
            minimum=kd_lower,
            maximum=kd_upper,
            steps=int(getattr(args, "fit_kd_steps", 3)),
            log_space=False,
        )
        if getattr(args, "fit_kd", True)
        else np.asarray([float(args.kd)], dtype=np.float64)
    )
    layer_eta_lock_values = (
        _fit_values(
            enabled=True,
            center=float(args.layer_eta_lock),
            minimum=eta_lower,
            maximum=eta_upper,
            steps=int(getattr(args, "fit_layer_eta_lock_steps", 3)),
            log_space=False,
        )
        if getattr(args, "fit_layer_eta_lock", False)
        else np.asarray([float(args.layer_eta_lock)], dtype=np.float64)
    )
    layer_densification_power_values = (
        _fit_values(
            enabled=True,
            center=float(args.layer_densification_power),
            minimum=densification_lower,
            maximum=densification_upper,
            steps=int(getattr(args, "fit_layer_densification_power_steps", 3)),
            log_space=False,
        )
        if getattr(args, "fit_layer_densification_power", False)
        else np.asarray([float(args.layer_densification_power)], dtype=np.float64)
    )
    return [
        MaterialCandidate(
            kh=float(kh),
            kd=float(kd),
            layer_eta_lock=float(layer_eta_lock),
            layer_densification_power=float(layer_densification_power),
        )
        for kh in kh_values
        for kd in kd_values
        for layer_eta_lock in layer_eta_lock_values
        for layer_densification_power in layer_densification_power_values
    ]


def _iter_candidate_batches(candidates: list[MaterialCandidate], batch_size: int) -> list[list[MaterialCandidate]]:
    batch_size = max(int(batch_size), 1)
    return [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]


def _evaluate_material_candidate(
    args, candidate: MaterialCandidate, cached_setup: CachedTrialSetup | None = None
) -> tuple[dict[str, Any], list[TrialResponse]]:
    candidate_args = copy.copy(args)
    candidate_args.kh = candidate.kh
    candidate_args.kd = candidate.kd
    candidate_args.layer_eta_lock = candidate.layer_eta_lock
    candidate_args.layer_densification_power = candidate.layer_densification_power
    try:
        sim_kwargs = {"sample_stride": int(candidate_args.calibration_search_sample_stride)}
        if cached_setup is not None:
            sim_kwargs["cached_setup"] = cached_setup
        responses = _simulate_calibration_responses(candidate_args, **sim_kwargs)
    except RuntimeError as exc:
        invalid_messages = (
            "Solver diverged",
            "generated no raw pressure-field contact force",
            "adjacent force jumps",
            "candidate became unstable",
        )
        if not any(message in str(exc) for message in invalid_messages):
            raise
        return _invalid_material_candidate(
            candidate,
            _candidate_invalid_reason(str(exc)),
            divergence_sample_index=_runtime_error_sample_index(str(exc)),
        ), []
    material = _fit_shared_material_parameters(responses, candidate_args)
    return material, responses


def _can_evaluate_batch_multiworld(args, candidates: list[MaterialCandidate]) -> bool:
    if len(candidates) <= 1:
        return False
    if int(getattr(args, "calibration_workers", 1)) != 1:
        return False
    cases_text = str(getattr(args, "calibration_cases", ""))
    if not cases_text:
        return False
    cases = [case.strip() for case in cases_text.split(",") if case.strip()]
    return bool(cases) and all(case in {"rearfoot", "fullfoot"} for case in cases)


def _evaluate_material_candidate_batch_multiworld(
    args,
    candidates: list[MaterialCandidate],
    cached_setup: CachedTrialSetup | None = None,
) -> list[tuple[dict[str, Any], list[TrialResponse]]]:
    if not _can_evaluate_batch_multiworld(args, candidates):
        return [_evaluate_material_candidate(args, candidate, cached_setup=cached_setup) for candidate in candidates]

    cases = [case.strip() for case in str(args.calibration_cases).split(",") if case.strip()]
    candidate_args = [copy.copy(args) for _candidate in candidates]
    batch_kwargs: dict[str, Any] = {
        "cases": cases,
        "sample_stride": int(args.calibration_search_sample_stride),
    }
    if cached_setup is not None:
        batch_kwargs["cached_setup"] = cached_setup
    try:
        batch_result = _batched_calibration_trial_responses(args, candidates, **batch_kwargs)
    except RuntimeError as exc:
        invalid_messages = (
            "Solver diverged",
            "generated no raw pressure-field contact force",
            "adjacent force jumps",
            "candidate became unstable",
        )
        if not any(message in str(exc) for message in invalid_messages):
            raise
        return [_evaluate_material_candidate(args, candidate, cached_setup=cached_setup) for candidate in candidates]

    results: list[tuple[dict[str, Any], list[TrialResponse]]] = []
    for candidate_index, (args_i, candidate, responses) in enumerate(
        zip(candidate_args, candidates, batch_result.responses_by_candidate, strict=True)
    ):
        if candidate_index in batch_result.invalid_candidates:
            fallback_material, fallback_responses = _evaluate_material_candidate(
                args, candidate, cached_setup=cached_setup
            )
            fallback_material["calibration_batch_mode"] = "singleworld_fallback"
            results.append((fallback_material, fallback_responses))
            continue
        args_i.kh = candidate.kh
        args_i.kd = candidate.kd
        args_i.layer_eta_lock = candidate.layer_eta_lock
        args_i.layer_densification_power = candidate.layer_densification_power
        zero_contact_response = next(
            (
                response
                for response in responses
                if bool(args_i.fail_on_zero_contact) and float(np.max(response.raw_contact_force_n)) <= 1.0e-9
            ),
            None,
        )
        if zero_contact_response is not None:
            results.append(
                (
                    _invalid_material_candidate(candidate, "zero contact"),
                    [],
                )
            )
            continue
        material = _fit_shared_material_parameters(responses, args_i)
        material["calibration_batch_mode"] = "multiworld"
        results.append((material, responses))
    return results


def _candidate_key(candidate: MaterialCandidate) -> tuple[float, float, float, float]:
    return (
        round(float(candidate.kh), 9),
        round(float(candidate.kd), 9),
        round(float(candidate.layer_eta_lock), 9),
        round(float(candidate.layer_densification_power), 9),
    )


def _evaluate_candidate_batches(
    args,
    candidates: list[MaterialCandidate],
    *,
    workers: int,
    cached_setup: CachedTrialSetup | None = None,
) -> list[tuple[dict[str, Any], list[TrialResponse]]]:
    results: list[tuple[dict[str, Any], list[TrialResponse]]] = []
    for batch in _iter_candidate_batches(candidates, int(args.calibration_batch_size)):
        if workers == 1 or len(batch) == 1:
            if _can_evaluate_batch_multiworld(args, batch):
                if cached_setup is not None:
                    results.extend(
                        _evaluate_material_candidate_batch_multiworld(args, batch, cached_setup=cached_setup)
                    )
                else:
                    results.extend(_evaluate_material_candidate_batch_multiworld(args, batch))
            else:
                if cached_setup is not None:
                    results.extend(
                        _evaluate_material_candidate(args, candidate, cached_setup=cached_setup) for candidate in batch
                    )
                else:
                    results.extend(_evaluate_material_candidate(args, candidate) for candidate in batch)
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                if cached_setup is not None:
                    results.extend(
                        executor.map(
                            lambda candidate: _evaluate_material_candidate(args, candidate, cached_setup=cached_setup),
                            batch,
                        )
                    )
                else:
                    results.extend(
                        executor.map(
                            lambda candidate: _evaluate_material_candidate(args, candidate),
                            batch,
                        )
                    )
    return results


def _refined_range(
    *,
    value: float,
    lower: float,
    upper: float,
    steps: int,
    radius: float,
    log_space: bool,
) -> tuple[float, float]:
    if lower == upper:
        return lower, upper
    steps = max(int(steps), 2)
    radius = max(float(radius), 0.5)
    if log_space and lower > 0.0 and upper > 0.0 and value > 0.0:
        lo = float(np.log(lower))
        hi = float(np.log(upper))
        center = float(np.log(value))
        step = (hi - lo) / float(steps - 1)
        return float(np.exp(max(lo, center - radius * step))), float(np.exp(min(hi, center + radius * step)))
    step = (upper - lower) / float(steps - 1)
    return max(lower, value - radius * step), min(upper, value + radius * step)


def _clip_candidate(args, candidate: MaterialCandidate) -> MaterialCandidate:
    kh_min, kh_max = _fit_range(bool(args.fit_kh), float(args.kh), float(args.fit_kh_min), float(args.fit_kh_max))
    kd_min, kd_max = _fit_range(
        bool(getattr(args, "fit_kd", True)),
        float(args.kd),
        float(getattr(args, "fit_kd_min", 0.0)),
        float(getattr(args, "fit_kd_max", 50.0)),
    )
    eta_min, eta_max = _fit_range(
        bool(getattr(args, "fit_layer_eta_lock", False)),
        float(args.layer_eta_lock),
        float(getattr(args, "fit_layer_eta_lock_min", 0.45)),
        float(getattr(args, "fit_layer_eta_lock_max", 0.85)),
    )
    densification_min, densification_max = _fit_range(
        bool(getattr(args, "fit_layer_densification_power", False)),
        float(args.layer_densification_power),
        float(getattr(args, "fit_layer_densification_power_min", 0.0)),
        float(getattr(args, "fit_layer_densification_power_max", 2.0)),
    )
    return MaterialCandidate(
        kh=float(np.clip(candidate.kh, kh_min, kh_max)),
        kd=float(np.clip(candidate.kd, kd_min, kd_max)),
        layer_eta_lock=float(np.clip(candidate.layer_eta_lock, eta_min, eta_max)),
        layer_densification_power=float(
            np.clip(candidate.layer_densification_power, densification_min, densification_max)
        ),
    )


def _coordinate_candidate_step(args) -> MaterialCandidate:
    log_kh = str(getattr(args, "calibration_kh_spacing", "linear")) == "log"
    kh_min, kh_max = _fit_range(bool(args.fit_kh), float(args.kh), float(args.fit_kh_min), float(args.fit_kh_max))
    if log_kh and kh_min > 0.0 and kh_max > 0.0:
        kh_step = float(np.exp(0.25 * (np.log(kh_max) - np.log(kh_min))))
    else:
        kh_step = 0.25 * (kh_max - kh_min)
    kd_min, kd_max = _fit_range(
        bool(getattr(args, "fit_kd", True)),
        float(args.kd),
        float(getattr(args, "fit_kd_min", 0.0)),
        float(getattr(args, "fit_kd_max", 50.0)),
    )
    eta_min, eta_max = _fit_range(
        bool(getattr(args, "fit_layer_eta_lock", False)),
        float(args.layer_eta_lock),
        float(getattr(args, "fit_layer_eta_lock_min", 0.45)),
        float(getattr(args, "fit_layer_eta_lock_max", 0.85)),
    )
    densification_min, densification_max = _fit_range(
        bool(getattr(args, "fit_layer_densification_power", False)),
        float(args.layer_densification_power),
        float(getattr(args, "fit_layer_densification_power_min", 0.0)),
        float(getattr(args, "fit_layer_densification_power_max", 2.0)),
    )
    return MaterialCandidate(
        kh=float(kh_step),
        kd=float(0.25 * (kd_max - kd_min)),
        layer_eta_lock=float(0.25 * (eta_max - eta_min)),
        layer_densification_power=float(0.25 * (densification_max - densification_min)),
    )


def _coordinate_probe_candidates(args, center: MaterialCandidate, step: MaterialCandidate) -> list[MaterialCandidate]:
    candidates = [center]
    if bool(args.fit_kh):
        log_kh = str(getattr(args, "calibration_kh_spacing", "linear")) == "log"
        if log_kh:
            candidates.extend(
                [
                    MaterialCandidate(
                        center.kh / max(step.kh, 1.0),
                        center.kd,
                        center.layer_eta_lock,
                        center.layer_densification_power,
                    ),
                    MaterialCandidate(
                        center.kh * max(step.kh, 1.0),
                        center.kd,
                        center.layer_eta_lock,
                        center.layer_densification_power,
                    ),
                ]
            )
        else:
            candidates.extend(
                [
                    MaterialCandidate(
                        center.kh - step.kh, center.kd, center.layer_eta_lock, center.layer_densification_power
                    ),
                    MaterialCandidate(
                        center.kh + step.kh, center.kd, center.layer_eta_lock, center.layer_densification_power
                    ),
                ]
            )
    if bool(getattr(args, "fit_kd", True)):
        candidates.extend(
            [
                MaterialCandidate(
                    center.kh, center.kd - step.kd, center.layer_eta_lock, center.layer_densification_power
                ),
                MaterialCandidate(
                    center.kh, center.kd + step.kd, center.layer_eta_lock, center.layer_densification_power
                ),
            ]
        )
    if bool(getattr(args, "fit_layer_eta_lock", False)):
        candidates.extend(
            [
                MaterialCandidate(
                    center.kh, center.kd, center.layer_eta_lock - step.layer_eta_lock, center.layer_densification_power
                ),
                MaterialCandidate(
                    center.kh, center.kd, center.layer_eta_lock + step.layer_eta_lock, center.layer_densification_power
                ),
            ]
        )
    if bool(getattr(args, "fit_layer_densification_power", False)):
        candidates.extend(
            [
                MaterialCandidate(
                    center.kh,
                    center.kd,
                    center.layer_eta_lock,
                    center.layer_densification_power - step.layer_densification_power,
                ),
                MaterialCandidate(
                    center.kh,
                    center.kd,
                    center.layer_eta_lock,
                    center.layer_densification_power + step.layer_densification_power,
                ),
            ]
        )
    unique: dict[tuple[float, float, float, float], MaterialCandidate] = {}
    for candidate in candidates:
        clipped_candidate = _clip_candidate(args, candidate)
        unique[_candidate_key(clipped_candidate)] = clipped_candidate
    return list(unique.values())


def _shrink_coordinate_step(step: MaterialCandidate, factor: float) -> MaterialCandidate:
    factor = max(float(factor), 1.0e-6)
    return MaterialCandidate(
        kh=1.0 + (float(step.kh) - 1.0) * factor if float(step.kh) >= 1.0 else float(step.kh) * factor,
        kd=float(step.kd) * factor,
        layer_eta_lock=float(step.layer_eta_lock) * factor,
        layer_densification_power=float(step.layer_densification_power) * factor,
    )


def _search_material_candidates_coordinate(args, *, workers: int) -> MaterialSearchResult:
    cases = [case.strip() for case in str(getattr(args, "calibration_cases", "")).split(",") if case.strip()]
    try:
        cached_setup = _load_trial_setup(args, cases, int(args.calibration_search_sample_stride))
    except AttributeError:
        cached_setup = None

    center = _clip_candidate(
        args,
        MaterialCandidate(
            kh=float(args.kh),
            kd=float(args.kd),
            layer_eta_lock=float(args.layer_eta_lock),
            layer_densification_power=float(args.layer_densification_power),
        ),
    )
    step = _coordinate_candidate_step(args)
    evaluated: set[tuple[float, float, float, float]] = set()
    best_candidate: tuple[dict[str, Any], list[TrialResponse]] | None = None
    best_objective = float("inf")
    history: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []

    iterations = max(int(getattr(args, "calibration_optimizer_iterations", 8)), 1)
    shrink = float(getattr(args, "calibration_optimizer_shrink", 0.5))
    for iteration in range(iterations):
        probes = [
            candidate
            for candidate in _coordinate_probe_candidates(args, center, step)
            if _candidate_key(candidate) not in evaluated
        ]
        evaluated.update(_candidate_key(candidate) for candidate in probes)
        if not probes:
            step = _shrink_coordinate_step(step, shrink)
            continue

        previous_best_objective = best_objective
        stage_best_material: dict[str, Any] | None = None
        stage_best_objective = float("inf")
        invalid_candidate_count = 0
        invalid_candidate_history: list[dict[str, float | int | str]] = []
        for candidate_material, candidate_responses in _evaluate_candidate_batches(
            args, probes, workers=workers, cached_setup=cached_setup
        ):
            objective = float(candidate_material["combined_objective"])
            if not np.isfinite(objective):
                invalid_candidate_count += 1
                invalid_candidate_history.append(_invalid_candidate_history_entry(candidate_material))
                invalid_reason = str(candidate_material.get("invalid_reason", ""))
                if invalid_reason and invalid_reason not in invalid_reasons:
                    invalid_reasons.append(invalid_reason)
                continue
            logger.info(
                "  candidate  kh=%.4g  kd=%.4g  eta=%.4g  dens=%.4g  objective=%.6g",
                float(candidate_material["kh"]),
                float(candidate_material["kd"]),
                float(candidate_material.get("layer_eta_lock", 0.0)),
                float(candidate_material.get("layer_densification_power", 0.0)),
                objective,
            )
            if objective < stage_best_objective:
                stage_best_objective = objective
                stage_best_material = candidate_material
            if objective < best_objective:
                best_objective = objective
                best_candidate = (candidate_material, candidate_responses)
        if stage_best_material is not None:
            logger.info(
                "  best stage objective=%.6g  kh=%.4g  kd=%.4g  eta=%.4g  dens=%.4g",
                stage_best_objective,
                float(stage_best_material["kh"]),
                float(stage_best_material["kd"]),
                float(stage_best_material.get("layer_eta_lock", 0.0)),
                float(stage_best_material.get("layer_densification_power", 0.0)),
            )

        improved = bool(stage_best_material is not None and stage_best_objective < previous_best_objective)
        if improved and stage_best_material is not None:
            center = MaterialCandidate(
                kh=float(stage_best_material["kh"]),
                kd=float(stage_best_material["kd"]),
                layer_eta_lock=float(stage_best_material.get("layer_eta_lock", args.layer_eta_lock)),
                layer_densification_power=float(
                    stage_best_material.get("layer_densification_power", args.layer_densification_power)
                ),
            )
        if not improved:
            step = _shrink_coordinate_step(step, shrink)

        stage_history: dict[str, Any] = {
            "stage": iteration + 1,
            "method": "coordinate",
            "candidate_count": len(probes),
            "invalid_candidate_count": invalid_candidate_count,
            "status": "ok" if stage_best_material is not None else "all_invalid",
        }
        if stage_best_material is not None:
            stage_history.update(
                {
                    "best_kh": float(stage_best_material["kh"]),
                    "best_kd": float(stage_best_material["kd"]),
                    "best_layer_eta_lock": float(stage_best_material.get("layer_eta_lock", args.layer_eta_lock)),
                    "best_layer_densification_power": float(
                        stage_best_material.get("layer_densification_power", args.layer_densification_power)
                    ),
                    "best_objective": float(stage_best_material["combined_objective"]),
                }
            )
        if invalid_candidate_history:
            stage_history["invalid_candidates"] = invalid_candidate_history
        history.append(stage_history)

    if best_candidate is None:
        if invalid_reasons:
            raise RuntimeError(
                "Material calibration could not evaluate any valid candidates. "
                f"First invalid candidate reason: {invalid_reasons[0]}"
            )
        raise RuntimeError("Material calibration did not evaluate any candidates.")
    return MaterialSearchResult(
        material=best_candidate[0],
        responses=best_candidate[1],
        candidate_count=len(evaluated),
        history=history,
    )


def _search_material_candidates(args, *, workers: int) -> MaterialSearchResult:
    if str(args.calibration_search_method) == "coordinate":
        return _search_material_candidates_coordinate(args, workers=workers)

    cases = [case.strip() for case in str(getattr(args, "calibration_cases", "")).split(",") if case.strip()]
    try:
        cached_setup = _load_trial_setup(args, cases, int(args.calibration_search_sample_stride))
    except AttributeError:
        cached_setup = None

    global_kh_min, global_kh_max = _fit_range(
        bool(args.fit_kh),
        float(args.kh),
        float(args.fit_kh_min),
        float(args.fit_kh_max),
    )
    global_kd_min, global_kd_max = _fit_range(
        bool(getattr(args, "fit_kd", True)),
        float(args.kd),
        float(getattr(args, "fit_kd_min", 0.0)),
        float(getattr(args, "fit_kd_max", 50.0)),
    )
    global_eta_min, global_eta_max = _fit_range(
        bool(getattr(args, "fit_layer_eta_lock", False)),
        float(args.layer_eta_lock),
        float(getattr(args, "fit_layer_eta_lock_min", 0.45)),
        float(getattr(args, "fit_layer_eta_lock_max", 0.85)),
    )
    global_densification_min, global_densification_max = _fit_range(
        bool(getattr(args, "fit_layer_densification_power", False)),
        float(args.layer_densification_power),
        float(getattr(args, "fit_layer_densification_power_min", 0.0)),
        float(getattr(args, "fit_layer_densification_power_max", 2.0)),
    )
    kh_min, kh_max = global_kh_min, global_kh_max
    kd_min, kd_max = global_kd_min, global_kd_max
    eta_min, eta_max = global_eta_min, global_eta_max
    densification_min, densification_max = global_densification_min, global_densification_max
    best_candidate: tuple[dict[str, Any], list[TrialResponse]] | None = None
    best_objective = float("inf")
    evaluated: set[tuple[float, float, float, float]] = set()
    history: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    search_method = str(args.calibration_search_method)
    stages = 1 if search_method == "grid" else max(int(args.calibration_refine_stages), 1)
    log_kh = str(args.calibration_kh_spacing) == "log"

    for stage_index in range(stages):
        stage_candidates = _material_candidate_grid(
            args,
            kh_min=kh_min,
            kh_max=kh_max,
            kd_min=kd_min,
            kd_max=kd_max,
            layer_eta_lock_min=eta_min,
            layer_eta_lock_max=eta_max,
            layer_densification_power_min=densification_min,
            layer_densification_power_max=densification_max,
            log_kh=log_kh,
        )
        new_candidates = [candidate for candidate in stage_candidates if _candidate_key(candidate) not in evaluated]
        evaluated.update(_candidate_key(candidate) for candidate in new_candidates)
        if not new_candidates:
            continue

        stage_best_material: dict[str, Any] | None = None
        stage_best_objective = float("inf")
        invalid_candidate_count = 0
        invalid_candidate_history: list[dict[str, float | int | str]] = []
        for candidate_material, candidate_responses in _evaluate_candidate_batches(
            args, new_candidates, workers=workers, cached_setup=cached_setup
        ):
            objective = float(candidate_material["combined_objective"])
            if not np.isfinite(objective):
                invalid_candidate_count += 1
                invalid_candidate_history.append(_invalid_candidate_history_entry(candidate_material))
                invalid_reason = str(candidate_material.get("invalid_reason", ""))
                if invalid_reason and invalid_reason not in invalid_reasons:
                    invalid_reasons.append(invalid_reason)
                continue
            logger.info(
                "  candidate  kh=%.4g  kd=%.4g  eta=%.4g  dens=%.4g  objective=%.6g",
                float(candidate_material["kh"]),
                float(candidate_material["kd"]),
                float(candidate_material.get("layer_eta_lock", 0.0)),
                float(candidate_material.get("layer_densification_power", 0.0)),
                objective,
            )
            if objective < stage_best_objective:
                stage_best_objective = objective
                stage_best_material = candidate_material
            if objective < best_objective:
                best_objective = objective
                best_candidate = (candidate_material, candidate_responses)
        if stage_best_material is not None:
            logger.info(
                "  best stage objective=%.6g  kh=%.4g  kd=%.4g  eta=%.4g  dens=%.4g",
                stage_best_objective,
                float(stage_best_material["kh"]),
                float(stage_best_material["kd"]),
                float(stage_best_material.get("layer_eta_lock", 0.0)),
                float(stage_best_material.get("layer_densification_power", 0.0)),
            )

        stage_history: dict[str, Any] = {
            "stage": stage_index + 1,
            "method": search_method,
            "candidate_count": len(new_candidates),
            "invalid_candidate_count": invalid_candidate_count,
            "kh_min": float(kh_min),
            "kh_max": float(kh_max),
            "kd_min": float(kd_min),
            "kd_max": float(kd_max),
            "layer_eta_lock_min": float(eta_min),
            "layer_eta_lock_max": float(eta_max),
            "layer_densification_power_min": float(densification_min),
            "layer_densification_power_max": float(densification_max),
            "status": "ok" if stage_best_material is not None else "all_invalid",
        }
        if stage_best_material is not None:
            stage_history.update(
                {
                    "best_kh": float(stage_best_material["kh"]),
                    "best_kd": float(stage_best_material["kd"]),
                    "best_layer_eta_lock": float(stage_best_material.get("layer_eta_lock", args.layer_eta_lock)),
                    "best_layer_densification_power": float(
                        stage_best_material.get("layer_densification_power", args.layer_densification_power)
                    ),
                    "best_objective": float(stage_best_material["combined_objective"]),
                }
            )
        if invalid_candidate_history:
            stage_history["invalid_candidates"] = invalid_candidate_history
        history.append(stage_history)
        if stage_best_material is not None:
            if search_method == "adaptive" and stage_index < stages - 1:
                kh_min, kh_max = _refined_range(
                    value=float(stage_best_material["kh"]),
                    lower=global_kh_min,
                    upper=global_kh_max,
                    steps=int(args.fit_kh_steps),
                    radius=float(args.calibration_refine_radius),
                    log_space=log_kh,
                )
                kd_min, kd_max = _refined_range(
                    value=float(stage_best_material["kd"]),
                    lower=global_kd_min,
                    upper=global_kd_max,
                    steps=int(getattr(args, "fit_kd_steps", 3)),
                    radius=float(args.calibration_refine_radius),
                    log_space=False,
                )
                eta_min, eta_max = _refined_range(
                    value=float(stage_best_material.get("layer_eta_lock", args.layer_eta_lock)),
                    lower=global_eta_min,
                    upper=global_eta_max,
                    steps=int(getattr(args, "fit_layer_eta_lock_steps", 3)),
                    radius=float(args.calibration_refine_radius),
                    log_space=False,
                )
                densification_min, densification_max = _refined_range(
                    value=float(stage_best_material.get("layer_densification_power", args.layer_densification_power)),
                    lower=global_densification_min,
                    upper=global_densification_max,
                    steps=int(getattr(args, "fit_layer_densification_power_steps", 3)),
                    radius=float(args.calibration_refine_radius),
                    log_space=False,
                )

    if best_candidate is None:
        if invalid_reasons:
            raise RuntimeError(
                "Material calibration could not evaluate any valid candidates. "
                f"First invalid candidate reason: {invalid_reasons[0]}"
            )
        raise RuntimeError("Material calibration did not evaluate any candidates.")
    return MaterialSearchResult(
        material=best_candidate[0],
        responses=best_candidate[1],
        candidate_count=len(evaluated),
        history=history,
    )


def _write_material_json(
    path: Path,
    args,
    *,
    material: dict[str, float],
    summaries: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "material_model": "digital_instron_pressure_field_v1",
        "description": (
            "Shared pressure-field midsole material fit from rearfoot and fullfoot Instron trials. "
            "contact_offset_m is a geometric nuisance parameter, not a material property."
        ),
        "source_midsole_mesh": str(args.midsole_mesh),
        "source_indenter_mesh": str(args.indenter_mesh),
        "contact_offset_m": float(material["contact_offset_m"]),
        "contact_threshold_n": float(material["contact_threshold_n"]),
        "parameters": {
            "kh": float(material["kh"]),
            "kd": float(material["kd"]),
            "pressure_profile": str(args.pressure_profile),
            "layer_eta_lock": float(args.layer_eta_lock),
            "layer_densification_power": float(args.layer_densification_power),
            "layer_cubic": float(args.layer_cubic),
            "layer_quintic": float(args.layer_quintic),
            "pressure_sine_amplitude": [float(v) for v in args.pressure_sine_amplitude],
            "pressure_sine_cycles": [float(v) for v in args.pressure_sine_cycles],
            "pressure_sine_phase": [float(v) for v in args.pressure_sine_phase],
            "mu": 0.8,
            "narrow_band_m": float(args.narrow_band),
            "sdf_resolution": int(args.sdf_resolution),
        },
        "calibration": {
            "test_cases": list(summaries),
            "combined_objective": float(material["combined_objective"]),
            "objective_components": material.get("objective_components", {}),
            "trials": {
                test_case: {
                    "trace_csv": summary["trace_csv"],
                    "output_csv": summary["output_csv"],
                    "measured_peak_force_n": summary["measured_peak_force_n"],
                    "sim_peak_force_n": summary["sim_peak_force_n"],
                    "raw_contact_peak_force_n": summary["raw_contact_peak_force_n"],
                    "raw_contact_peak_relative_error": summary["raw_contact_peak_relative_error"],
                    "peak_relative_error": summary["peak_relative_error"],
                    "loop_rmse_n": summary["loop_rmse_n"],
                    "force_r2": summary["force_r2"],
                    "max_abs_shoe_vertical_velocity_m_s": summary["max_abs_shoe_vertical_velocity_m_s"],
                    "rms_shoe_vertical_velocity_m_s": summary["rms_shoe_vertical_velocity_m_s"],
                    "max_abs_shoe_free_axis_angular_velocity_rad_s": summary[
                        "max_abs_shoe_free_axis_angular_velocity_rad_s"
                    ],
                    "rms_shoe_free_axis_angular_velocity_rad_s": summary["rms_shoe_free_axis_angular_velocity_rad_s"],
                    "max_adjacent_force_jump_n": summary["max_adjacent_force_jump_n"],
                    "force_jump_limit_n": summary["force_jump_limit_n"],
                    "force_jump_violation_count": summary["force_jump_violation_count"],
                    "metrics": summary["metrics"],
                }
                for test_case, summary in summaries.items()
            },
        },
        "state_model": _pressure_memory_state_model(args),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _bundle_payload_from_material(
    args,
    *,
    material_payload: dict[str, Any],
) -> dict[str, Any]:
    parameters = dict(material_payload.get("parameters", {}))
    calibration = dict(material_payload.get("calibration", {}))
    state_model = dict(material_payload.get("state_model", _pressure_memory_state_model(args)))
    geometry = {
        "midsole_mesh": str(material_payload.get("source_midsole_mesh", args.midsole_mesh)),
        "fullfoot_indenter_mesh": str(material_payload.get("source_indenter_mesh", args.indenter_mesh)),
        "mesh_scale": float(args.mesh_scale),
        "coordinate_inference": {
            "length_axis": "longest_source_axis",
            "width_axis": "remaining_source_axis",
            "vertical_axis": "shortest_source_axis",
        },
        "fullfoot_rotation_deg": [float(value) for value in args.fullfoot_rotation_deg],
        "fullfoot_start_clearance_m": float(args.fullfoot_start_clearance),
        "fullfoot_min_start_clearance_m": float(args.fullfoot_min_start_clearance),
        "fullfoot_indenter_crop_fixture": bool(args.fullfoot_indenter_crop_fixture),
        "fullfoot_indenter_fixture_crop_z_m": (
            float(args.fullfoot_indenter_fixture_crop_z) if bool(args.fullfoot_indenter_crop_fixture) else None
        ),
        "rearfoot_punch": {
            "radius_m": float(args.punch_radius),
            "half_height_m": float(args.punch_half_height),
            "heel_side": str(args.heel_side),
        },
    }
    fixture = {
        "type": "digital_instron_box_shoe_missile_v1",
        "shoe_body": (
            "dynamic_rigid_body_with_compliant_midsole_mesh"
            if bool(getattr(args, "fixture_dynamic_shoe", False))
            else "kinematic_compliant_midsole_specimen"
        ),
        "shoe_horizontal_lock": bool(args.fixture_lock_shoe_horizontal),
        "shoe_free_rotation_axis": str(args.fixture_free_rotation_axis),
        "gravity": "enabled",
        "bottom_platen": "static_box",
        "platen_stop_clearance_m": float(args.fixture_platen_stop_clearance),
        "bottom_platen_overlap_m": float(max(float(getattr(args, "fixture_bottom_platen_overlap", 0.0)), 0.0)),
        "top_indenter": "kinematic_position_prescribed",
        "reaction_force_source": "top_indenter",
        "compression_replay": {
            "mode": "vertical_dynamic_relaxation",
            "substeps_per_sample": _compression_relax_steps(args),
        },
        "displacement_zero": "auto_contact_search_against_peak_force",
        "settling": {
            "mode": "gravity_preload_equilibrium",
            "implemented": True,
            "duration_s": float(args.fixture_settle_duration),
            "velocity_tolerance_m_s": float(args.fixture_settle_velocity_tol),
            "indenter_clearance_m": float(args.fixture_settle_indenter_clearance),
            "gravity_preload_search_m": float(args.fixture_gravity_preload_search_m),
            "gravity_preload_iterations": int(args.fixture_gravity_preload_iterations),
        },
        "auto_contact_search": {
            "enabled": bool(args.fullfoot_auto_contact_offset),
            "max_m": float(args.fullfoot_contact_search_max),
            "steps": int(args.fullfoot_contact_search_steps),
            "initial_force_max_n": float(args.fullfoot_initial_force_max),
            "tolerance": float(args.fullfoot_contact_search_tolerance),
            "pre_embedding": "disallowed",
        },
    }
    material = {
        "model": str(material_payload.get("material_model", "digital_instron_pressure_field_v1")),
        "parameters": parameters,
        "pressure_profile": {
            "type": str(parameters.get("pressure_profile", args.pressure_profile)),
            "layer": {
                "eta_lock": float(parameters.get("layer_eta_lock", args.layer_eta_lock)),
                "densification_power": float(
                    parameters.get("layer_densification_power", args.layer_densification_power)
                ),
                "cubic": float(parameters.get("layer_cubic", args.layer_cubic)),
                "quintic": float(parameters.get("layer_quintic", args.layer_quintic)),
            },
            "sine_amplitude": list(parameters.get("pressure_sine_amplitude", args.pressure_sine_amplitude)),
            "sine_cycles": list(parameters.get("pressure_sine_cycles", args.pressure_sine_cycles)),
            "sine_phase": list(parameters.get("pressure_sine_phase", args.pressure_sine_phase)),
        },
    }
    return {
        "schema_version": 1,
        "asset_type": "shoe_pressure_field_bundle",
        "geometry": geometry,
        "fixture": fixture,
        "material": material,
        "state_model": state_model,
        "calibration": calibration,
        "protomotions": {
            "consumer": "ProtoMotions Newton pressure-field shoe contact",
            "status": "ready_for_integration",
        },
    }


def _write_bundle_json(
    path: Path,
    args,
    *,
    material_payload: dict[str, Any],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _bundle_payload_from_material(args, material_payload=material_payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _plot_hysteresis_comparison(summaries: dict[str, dict[str, Any]], output_path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(summaries), figsize=(6 * len(summaries), 4), squeeze=False)
    for ax, (test_case, summary) in zip(axes[0], summaries.items(), strict=False):
        with Path(summary["output_csv"]).open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        displacement_mm = np.asarray([float(row["displacement_mm"]) for row in rows], dtype=np.float64)
        measured_force_n = np.asarray([float(row["measured_force_n"]) for row in rows], dtype=np.float64)
        sim_force_n = np.asarray([float(row["sim_force_n"]) for row in rows], dtype=np.float64)

        ax.plot(displacement_mm, measured_force_n, color="black", linewidth=2.0, label="physical")
        ax.plot(displacement_mm, sim_force_n, color="#1f77b4", linewidth=2.0, label="digital")
        ax.set_title(
            f"{test_case}: RMSE {summary['loop_rmse_n']:.1f} N, peak err {100.0 * summary['peak_relative_error']:.2f}%"
        )
        ax.set_xlabel("Displacement [mm]")
        ax.set_ylabel("Force [N]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def calibrate_digital_instron(args) -> dict[str, Any]:
    workers = max(int(args.calibration_workers), 1)
    search_result = _search_material_candidates(args, workers=workers)
    material = search_result.material
    args.kh = float(material["kh"])
    args.kd = float(material["kd"])
    if "layer_eta_lock" in material:
        args.layer_eta_lock = float(material["layer_eta_lock"])
    if "layer_densification_power" in material:
        args.layer_densification_power = float(material["layer_densification_power"])
    if (
        int(args.calibration_final_sample_stride) == int(args.calibration_search_sample_stride)
        and search_result.responses
    ):
        responses = search_result.responses
    else:
        responses = _simulate_calibration_responses(
            args,
            sample_stride=int(args.calibration_final_sample_stride),
        )
    material = _fit_shared_material_parameters(responses, args)
    summaries = {response.test_case: _evaluate_trial_response(response, args) for response in responses}

    material_json = Path(args.material_output_json)
    _write_material_json(material_json, args, material=material, summaries=summaries)
    material_payload = json.loads(material_json.read_text(encoding="utf-8"))
    bundle_json = Path(args.bundle_output_json)
    _write_bundle_json(bundle_json, args, material_payload=material_payload)
    plot_path = _plot_hysteresis_comparison(summaries, Path(args.plot_output))

    summary = {
        "mode": "calibrate",
        "material_json": str(material_json),
        "bundle_json": str(bundle_json),
        "plot_output": plot_path,
        "candidate_count": search_result.candidate_count,
        "calibration_search_method": str(args.calibration_search_method),
        "calibration_objective": str(args.calibration_objective),
        "calibration_kh_spacing": str(args.calibration_kh_spacing),
        "calibration_refine_stages": int(args.calibration_refine_stages),
        "calibration_search_history": search_result.history,
        "calibration_cases": [case.strip() for case in args.calibration_cases.split(",") if case.strip()],
        "calibration_search_sample_stride": int(args.calibration_search_sample_stride),
        "calibration_final_sample_stride": int(args.calibration_final_sample_stride),
        "calibration_batch_size": int(args.calibration_batch_size),
        "calibration_workers": workers,
        "calibration_early_out": bool(getattr(args, "calibration_early_out", False)),
        "calibration_optimizer_iterations": int(getattr(args, "calibration_optimizer_iterations", 0)),
        "fixture_solver": str(getattr(args, "fixture_solver", "")),
        "fixture_mujoco_nconmax": int(getattr(args, "fixture_mujoco_nconmax", 0)),
        "fixture_mujoco_njmax": int(getattr(args, "fixture_mujoco_njmax", 0)),
        "objective_components": material.get("objective_components", {}),
        "parameters": {
            "kh": float(material["kh"]),
            **material,
            "pressure_profile": str(args.pressure_profile),
            "layer_eta_lock": float(args.layer_eta_lock),
            "layer_densification_power": float(args.layer_densification_power),
            "layer_cubic": float(args.layer_cubic),
            "layer_quintic": float(args.layer_quintic),
            "pressure_sine_amplitude": [float(v) for v in args.pressure_sine_amplitude],
            "pressure_sine_cycles": [float(v) for v in args.pressure_sine_cycles],
            "pressure_sine_phase": [float(v) for v in args.pressure_sine_phase],
        },
        "trials": summaries,
    }
    summary_path = PROCESSED_DIR / "digital_instron_calibration.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def export_bundle(args) -> dict[str, Any]:
    material_json = Path(args.material_output_json)
    if not material_json.exists():
        raise FileNotFoundError(f"Material JSON does not exist: {material_json}")
    material_payload = json.loads(material_json.read_text(encoding="utf-8"))
    bundle_json = Path(args.bundle_output_json)
    bundle_payload = _write_bundle_json(bundle_json, args, material_payload=material_payload)
    return {
        "mode": "export-bundle",
        "material_json": str(material_json),
        "bundle_json": str(bundle_json),
        "asset_type": bundle_payload["asset_type"],
        "state_model": bundle_payload["state_model"],
    }


def _apply_material_parameters(args) -> dict[str, Any] | None:
    if not bool(args.load_material):
        return None

    material_json = Path(args.material_output_json)
    if not material_json.exists():
        return None

    material_payload = json.loads(material_json.read_text(encoding="utf-8"))
    parameters = dict(material_payload.get("parameters", {}))
    if not parameters:
        return None

    if "kh" in parameters:
        args.kh = float(parameters["kh"])
    if "kd" in parameters:
        args.kd = float(parameters["kd"])
    if "pressure_profile" in parameters:
        args.pressure_profile = str(parameters["pressure_profile"])
    if "layer_eta_lock" in parameters:
        args.layer_eta_lock = float(parameters["layer_eta_lock"])
    if "layer_densification_power" in parameters:
        args.layer_densification_power = float(parameters["layer_densification_power"])
    if "layer_cubic" in parameters:
        args.layer_cubic = float(parameters["layer_cubic"])
    if "layer_quintic" in parameters:
        args.layer_quintic = float(parameters["layer_quintic"])
    if "pressure_sine_amplitude" in parameters:
        args.pressure_sine_amplitude = tuple(float(v) for v in parameters["pressure_sine_amplitude"])
    if "pressure_sine_cycles" in parameters:
        args.pressure_sine_cycles = tuple(float(v) for v in parameters["pressure_sine_cycles"])
    if "pressure_sine_phase" in parameters:
        args.pressure_sine_phase = tuple(float(v) for v in parameters["pressure_sine_phase"])
    if "narrow_band_m" in parameters:
        args.narrow_band = float(parameters["narrow_band_m"])
    if "sdf_resolution" in parameters:
        args.sdf_resolution = int(parameters["sdf_resolution"])

    return {
        "path": str(material_json),
        "kh": float(args.kh),
        "kd": float(args.kd),
        "pressure_profile": str(args.pressure_profile),
    }


def _create_viewer(args):
    import newton.viewer  # noqa: PLC0415

    if args.device:
        wp.set_device(args.device)

    if args.viewer == "gl":
        return newton.viewer.ViewerGL(headless=args.headless)
    if args.viewer == "usd":
        return newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    if args.viewer == "rerun":
        return newton.viewer.ViewerRerun(address=args.rerun_address)
    if args.viewer == "viser":
        return newton.viewer.ViewerViser()
    if args.viewer == "null":
        return newton.viewer.ViewerNull(num_frames=args.num_frames)
    raise ValueError(f"Invalid viewer: {args.viewer}")


def view_digital_instron(args) -> dict[str, Any]:
    if args.device:
        wp.set_device(args.device)
    loaded_material = _apply_material_parameters(args)
    trace_csv, output_csv = _resolve_trial_paths(args, args.test_case)
    scene = _build_trial_scene(args, test_case=args.test_case, trace_csv=trace_csv, output_csv=output_csv)
    viewer = _create_viewer(args)
    viewer.set_model(scene.model)
    viewer.show_contacts = True
    viewer.show_hydro_contact_surface = True
    viewer.set_camera(pos=wp.vec3(0.35, -0.45, 0.22), pitch=-25.0, yaw=135.0)

    sample_count = len(scene.displacement_m)
    stride = max(int(args.view_sample_stride), 1)
    frame = 0
    last_force = 0.0
    last_stats: dict[str, float | int] = {}
    while viewer.is_running():
        sample_index = min(frame * stride, sample_count - 1)
        displacement = float(scene.displacement_m[sample_index])
        last_force, last_stats = _advance_and_measure_trial_force(
            scene,
            args,
            sample_index=sample_index,
            displacement=displacement,
            displacement_velocity=float(scene.displacement_velocity_m_s[sample_index]),
            relax_steps=_compression_relax_steps(args),
            include_diagnostics=False,
        )

        viewer.log_scalar("Instron/measured_force_n", float(scene.measured_force_n[sample_index]))
        viewer.log_scalar("Instron/raw_contact_force_n", last_force)
        viewer.log_scalar("Instron/displacement_mm", displacement * 1000.0)

        viewer.begin_frame(float(scene.time_s[sample_index]))
        viewer.log_state(scene.state)
        viewer.log_contacts(scene.contacts, scene.state)
        if scene.pipeline.hydroelastic_sdf is not None:
            viewer.log_hydro_contact_surface(scene.pipeline.hydroelastic_sdf.get_contact_surface())
        viewer.end_frame()

        if not viewer.is_paused():
            frame += 1
            if frame * stride >= sample_count:
                if bool(args.view_loop):
                    frame = 0
                else:
                    break

    viewer.close()
    return {
        "mode": "view",
        "test_case": scene.test_case,
        "trace_csv": scene.trace_csv,
        "sample_count": sample_count,
        "sample_stride": stride,
        "loaded_material": loaded_material,
        "last_sample_index": min(frame * stride, sample_count - 1),
        "last_displacement_mm": float(scene.displacement_m[min(frame * stride, sample_count - 1)] * 1000.0),
        "last_raw_contact_force_n": float(last_force),
        "last_contact_stats": last_stats,
        "midsole_stats": scene.midsole_stats,
    }


def view_calibration_batch(args) -> dict[str, Any]:
    if args.device:
        wp.set_device(args.device)
    loaded_material = _apply_material_parameters(args)
    original_early_out = bool(getattr(args, "calibration_early_out", False))
    args.calibration_early_out = False
    candidates = _material_candidate_grid(args)
    batch_index = max(int(getattr(args, "calibration_view_batch_index", 0)), 0)
    batches = _iter_candidate_batches(candidates, int(args.calibration_batch_size))
    if not batches:
        raise RuntimeError("Calibration batch view did not generate any candidates.")
    if batch_index >= len(batches):
        raise ValueError(f"--calibration-view-batch-index {batch_index} exceeds last batch index {len(batches) - 1}.")

    viewer = _create_viewer(args)
    cases = [case.strip() for case in str(args.calibration_cases).split(",") if case.strip()]
    batch_result = _batched_calibration_trial_responses(
        args,
        batches[batch_index],
        cases=cases,
        sample_stride=int(args.calibration_search_sample_stride),
        viewer=viewer,
    )
    responses_by_candidate = batch_result.responses_by_candidate
    viewer.close()

    trial_peaks = []
    for candidate_index, responses in enumerate(responses_by_candidate):
        for response in responses:
            trial_peaks.append(
                {
                    "candidate_index": candidate_index,
                    "test_case": response.test_case,
                    "peak_force_n": float(np.max(response.raw_contact_force_n)),
                    "sample_count": int(response.displacement_m.size),
                    "auto_contact": {
                        key: value
                        for key, value in response.midsole_stats.items()
                        if "auto" in key and ("contact_offset" in key or "contact_found" in key)
                    },
                }
            )

    return {
        "mode": "view-calibration",
        "batch_index": batch_index,
        "candidate_count": len(batches[batch_index]),
        "calibration_cases": cases,
        "calibration_search_sample_stride": int(args.calibration_search_sample_stride),
        "calibration_batch_size": int(args.calibration_batch_size),
        "calibration_early_out_disabled_for_view": original_early_out,
        "loaded_material": loaded_material,
        "trials": trial_peaks,
    }


def _default_trace_for_case(test_case: str) -> Path:
    if test_case == "fullfoot":
        return PROCESSED_DIR / "fullfoot_avg_cycles_90_100.csv"
    return PROCESSED_DIR / "rearfoot_avg_cycles_90_100.csv"


def preprocess_defaults(args) -> list[TraceSummary]:
    return [
        average_instron_cycles(
            FULLFOOT_CSV,
            PROCESSED_DIR / "fullfoot_avg_cycles_90_100.csv",
            cycle_start=int(args.cycle_start),
            cycle_end=int(args.cycle_end),
            samples=int(args.samples),
        ),
        average_instron_cycles(
            REARFOOT_CSV,
            PROCESSED_DIR / "rearfoot_avg_cycles_90_100.csv",
            cycle_start=int(args.cycle_start),
            cycle_end=int(args.cycle_end),
            samples=int(args.samples),
        ),
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument(
        "--mode",
        choices=["preprocess", "run", "calibrate", "view", "view-calibration", "export-bundle"],
        default="preprocess",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable INFO-level logging during calibration.")
    parser.add_argument("--test-case", choices=["fullfoot", "rearfoot"], default="rearfoot")
    parser.add_argument(
        "--calibration-cases",
        type=str,
        default="rearfoot,fullfoot",
        help="Comma-separated trial cases for --mode calibrate.",
    )
    parser.add_argument(
        "--report-json", action="store_true", help="Print the full raw summary JSON instead of the compact report."
    )
    parser.add_argument("--cycle-start", type=int, default=90)
    parser.add_argument("--cycle-end", type=int, default=100)
    parser.add_argument("--samples", type=int, default=501)
    parser.add_argument("--trace-csv", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--midsole-mesh", type=str, default=str(MIDSOLE_OBJ))
    parser.add_argument("--indenter-mesh", type=str, default=str(FULLFOOT_EFFECTOR_STL))
    parser.add_argument("--mesh-scale", type=float, default=0.001)
    parser.add_argument(
        "--fixture-gravity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gravity for the dynamic shoe fixture.",
    )
    parser.add_argument("--shoe-mass", type=float, default=0.165, help="Dynamic shoe body mass [kg].")
    parser.add_argument(
        "--fixture-lock-shoe-horizontal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lock shoe x/y translation and constrained rotations while allowing vertical gravity settling.",
    )
    parser.add_argument(
        "--fixture-dynamic-shoe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make the midsole specimen dynamic in vertical motion so gravity and platen contact support it.",
    )
    parser.add_argument(
        "--fixture-free-rotation-axis",
        choices=["none", "x", "y", "z"],
        default="none",
        help="Shoe-local rotation axis left free by the fixture lock; default keeps the Instron specimen level.",
    )
    parser.add_argument(
        "--fixture-lock-rotation-after-settle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug option: use the free rotation axis during gravity settling, then lock the settled shoe orientation.",
    )
    parser.add_argument(
        "--fixture-shoe-velocity-damping",
        type=float,
        default=1.0,
        help="Per-substep damping applied to shoe vertical velocity during fixture dynamics.",
    )
    parser.add_argument(
        "--fixture-shoe-angular-damping",
        type=float,
        default=1.0,
        help="Per-substep damping applied to the free shoe angular velocity during fixture dynamics.",
    )
    parser.add_argument("--fixture-settle-duration", type=float, default=0.5, help="Maximum gravity settling time [s].")
    parser.add_argument("--fixture-settle-rate-hz", type=float, default=120.0, help="Settling simulation rate [Hz].")
    parser.add_argument("--fixture-settle-substeps", type=int, default=4, help="Solver substeps per settling step.")
    parser.add_argument(
        "--fixture-settle-indenter-clearance",
        type=float,
        default=0.01,
        help="Temporary clearance [m] used to keep the top indenter off the midsole during gravity preload settling.",
    )
    parser.add_argument(
        "--fixture-gravity-preload-search-m",
        type=float,
        default=0.02,
        help="Vertical search half-width [m] for the initial gravity preload equilibrium solve.",
    )
    parser.add_argument(
        "--fixture-gravity-preload-iterations",
        type=int,
        default=24,
        help="Bisection iterations for matching initial bottom contact force to shoe weight.",
    )
    parser.add_argument(
        "--fixture-settle-velocity-tol",
        type=float,
        default=1.0e-3,
        help="Early-exit velocity threshold [m/s] for gravity settling.",
    )
    parser.add_argument(
        "--fixture-solver",
        choices=["mujoco", "semi-implicit", "xpbd"],
        default="mujoco",
        help="Solver used for dynamic fixture replay; mujoco/semi-implicit use contact kh/kd as forces.",
    )
    parser.add_argument(
        "--fixture-solver-iterations", type=int, default=8, help="XPBD iterations for fixture dynamics."
    )
    parser.add_argument(
        "--fixture-solver-angular-damping",
        type=float,
        default=0.05,
        help="Angular damping passed to SolverSemiImplicit for fixture dynamics.",
    )
    parser.add_argument("--fixture-mujoco-iterations", type=int, default=100, help="MuJoCo solver iterations.")
    parser.add_argument(
        "--fixture-mujoco-nconmax",
        type=int,
        default=4096,
        help="MuJoCo contact-point capacity for dense fixture pressure-field contact.",
    )
    parser.add_argument(
        "--fixture-mujoco-njmax",
        type=int,
        default=16384,
        help="MuJoCo constraint capacity for dense fixture pressure-field contact.",
    )
    parser.add_argument(
        "--fixture-mujoco-solver",
        choices=["cg", "newton"],
        default="newton",
        help="MuJoCo solver type for fixture dynamics.",
    )
    parser.add_argument(
        "--fixture-mujoco-integrator",
        choices=["euler", "rk4", "implicit", "implicitfast"],
        default="implicitfast",
        help="MuJoCo integrator for fixture dynamics.",
    )
    parser.add_argument(
        "--fixture-mujoco-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use MuJoCo CPU backend instead of mujoco_warp for fixture dynamics.",
    )
    parser.add_argument(
        "--fixture-platen-stop-clearance",
        type=float,
        default=1.0e-4,
        help="Minimum separation [m] kept between the shoe/indenter reference surfaces and the bottom platen.",
    )
    parser.add_argument(
        "--fixture-bottom-platen-overlap",
        type=float,
        default=0.0,
        help="Optional initial bottom platen overlap [m] into the compliant midsole pressure field; default lets gravity create preload.",
    )
    parser.add_argument(
        "--compression-relax-steps",
        type=int,
        default=None,
        help=(
            "Solver substeps after each prescribed indenter sample; 0 gives quasi-static collision replay. "
            "Defaults to 4 for dynamic run/calibrate/view."
        ),
    )
    parser.add_argument(
        "--fixture-quasistatic-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug option: zero shoe velocities after each prescribed sample solve to suppress replay bounce.",
    )
    parser.add_argument("--heel-side", choices=["min", "max"], default="min")
    parser.add_argument("--punch-radius", type=float, default=0.022)
    parser.add_argument("--punch-half-height", type=float, default=0.025)
    parser.add_argument(
        "--fullfoot-rotation-deg",
        type=float,
        nargs=3,
        default=(90.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="RPY rotation [deg] applied to the fullfoot indenter mesh after axis canonicalization.",
    )
    parser.add_argument(
        "--fullfoot-indenter-crop-fixture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Crop fixture/post triangles from the fullfoot indenter mesh before building its collision SDF.",
    )
    parser.add_argument(
        "--fullfoot-indenter-fixture-crop-z",
        type=float,
        default=0.08,
        help="Maximum post-rotation local z [m] retained for the fullfoot indenter when fixture cropping is enabled.",
    )
    parser.add_argument(
        "--fullfoot-start-clearance",
        type=float,
        default=0.0,
        help="Initial clearance [m] above the midsole top before applying the fullfoot displacement trace.",
    )
    parser.add_argument(
        "--fullfoot-min-start-clearance",
        type=float,
        default=0.0,
        help="Minimum allowed fullfoot indenter clearance [m] after auto-contact search; prevents pre-embedding.",
    )
    parser.add_argument(
        "--fullfoot-initial-force-max",
        type=float,
        default=25.0,
        help="Maximum allowed fullfoot force [N] at zero displacement during auto-contact offset search.",
    )
    parser.add_argument(
        "--fullfoot-z-offset",
        type=float,
        default=0.0,
        help="Manual downward offset [m] for the fullfoot indenter rest pose.",
    )
    parser.add_argument(
        "--fullfoot-auto-contact-offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search downward for fullfoot contact at peak displacement before the sweep.",
    )
    parser.add_argument("--fullfoot-contact-search-max", type=float, default=0.08)
    parser.add_argument("--fullfoot-contact-search-steps", type=int, default=101)
    parser.add_argument("--fullfoot-contact-search-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--fullfoot-stop-search-after-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop fullfoot offset search once the target peak force is reached within tolerance.",
    )
    parser.add_argument(
        "--rearfoot-auto-contact-offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search downward for rearfoot punch contact at peak displacement before the sweep.",
    )
    parser.add_argument("--rearfoot-contact-search-max", type=float, default=0.02)
    parser.add_argument("--rearfoot-contact-search-steps", type=int, default=41)
    parser.add_argument("--rearfoot-contact-search-tolerance", type=float, default=0.02)
    parser.add_argument("--rearfoot-initial-force-max", type=float, default=10.0)
    parser.add_argument(
        "--rearfoot-stop-search-after-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop rearfoot offset search once the target peak force is reached within tolerance.",
    )
    parser.add_argument(
        "--fail-on-zero-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise an error when a trial generates no raw contact force.",
    )
    parser.add_argument("--kh", type=float, default=2.0e6)
    parser.add_argument(
        "--fit-kh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit Newton contact stiffness directly during calibration.",
    )
    parser.add_argument("--fit-kh-min", type=float, default=2.0e5)
    parser.add_argument("--fit-kh-max", type=float, default=2.0e8)
    parser.add_argument("--fit-kh-steps", type=int, default=7)
    parser.add_argument(
        "--calibration-search-method",
        choices=["adaptive", "grid", "coordinate"],
        default="adaptive",
        help=(
            "Material search method. adaptive does coarse-to-fine refinement; grid evaluates one fixed grid; "
            "coordinate does derivative-free local pattern search."
        ),
    )
    parser.add_argument(
        "--calibration-kh-spacing",
        choices=["log", "linear"],
        default="log",
        help="Spacing for fitted kh candidates.",
    )
    parser.add_argument(
        "--calibration-refine-stages",
        type=int,
        default=2,
        help="Number of coarse-to-fine stages for --calibration-search-method adaptive.",
    )
    parser.add_argument(
        "--calibration-refine-radius",
        type=float,
        default=0.75,
        help="Half-width, in previous grid intervals, used when refining around the best candidate.",
    )
    parser.add_argument(
        "--calibration-search-sample-stride",
        type=int,
        default=4,
        help="Use every Nth whole-loop trace sample while searching material candidates; peak and endpoints are kept.",
    )
    parser.add_argument(
        "--calibration-final-sample-stride",
        type=int,
        default=1,
        help="Use every Nth whole-loop trace sample when re-running the winning material for output files.",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=1,
        help="Number of material candidates grouped per calibration dispatch batch.",
    )
    parser.add_argument(
        "--calibration-workers",
        type=int,
        default=1,
        help="Parallel worker count for evaluating material candidates in a batch.",
    )
    parser.add_argument(
        "--calibration-multiworld-contact-buffer-fraction",
        type=float,
        default=0.25,
        help="Reduced-contact face-buffer fraction used by multi-world calibration batches.",
    )
    parser.add_argument(
        "--calibration-view-batch-index",
        type=int,
        default=0,
        help="Candidate batch index to visualize in --mode view-calibration.",
    )
    parser.add_argument(
        "--calibration-optimizer-iterations",
        type=int,
        default=8,
        help="Coordinate-search iterations for --calibration-search-method coordinate.",
    )
    parser.add_argument(
        "--calibration-optimizer-shrink",
        type=float,
        default=0.5,
        help="Step shrink factor when a coordinate-search iteration does not improve the material.",
    )
    parser.add_argument(
        "--calibration-early-out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject unstable calibration candidates during replay instead of finishing the whole trial.",
    )
    parser.add_argument(
        "--calibration-early-out-force-multiplier",
        type=float,
        default=4.0,
        help="Reject candidates whose absolute simulated force exceeds this multiple of the measured peak; 0 disables.",
    )
    parser.add_argument(
        "--calibration-early-out-force-jump-limit",
        type=float,
        default=200.0,
        help="Reject candidates whose adjacent simulated force jump exceeds this limit [N]; 0 disables.",
    )
    parser.add_argument(
        "--calibration-early-out-max-shoe-vz",
        type=float,
        default=0.0,
        help="Reject candidates above this absolute shoe vertical velocity [m/s]; 0 disables.",
    )
    parser.add_argument(
        "--calibration-early-out-max-shoe-omega",
        type=float,
        default=0.0,
        help="Reject candidates above this absolute free-axis shoe angular velocity [rad/s]; 0 disables.",
    )
    parser.add_argument(
        "--calibration-objective",
        choices=["force", "goal"],
        default="force",
        help="Candidate ranking objective. force preserves peak/RMSE fitting; goal adds R2, hysteresis, and shoe-motion penalties.",
    )
    parser.add_argument("--goal-rmse-weight", type=float, default=1.0)
    parser.add_argument("--goal-r2-weight", type=float, default=1.0)
    parser.add_argument("--goal-hysteresis-weight", type=float, default=1.0)
    parser.add_argument("--goal-shoe-vz-weight", type=float, default=0.25)
    parser.add_argument("--goal-shoe-omega-weight", type=float, default=0.25)
    parser.add_argument(
        "--goal-shoe-vz-target",
        type=float,
        default=1.0e-3,
        help="Reference RMS shoe vertical velocity [m/s] used to normalize goal-objective motion penalty.",
    )
    parser.add_argument(
        "--goal-shoe-omega-target",
        type=float,
        default=1.0e-2,
        help="Reference RMS shoe free-axis angular velocity [rad/s] used to normalize goal-objective motion penalty.",
    )
    parser.add_argument(
        "--pressure-profile",
        choices=["poisson", "layer"],
        default="layer",
        help="Immutable pressure field profile for the compliant midsole.",
    )
    parser.add_argument(
        "--layer-eta-lock",
        type=float,
        default=0.65,
        help="Layer pressure lock-up strain for --pressure-profile layer.",
    )
    parser.add_argument(
        "--layer-densification-power",
        type=float,
        default=0.0,
        help="Layer pressure densification exponent for --pressure-profile layer.",
    )
    parser.add_argument(
        "--fit-layer-eta-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit layer pressure lock-up strain during calibration.",
    )
    parser.add_argument("--fit-layer-eta-lock-min", type=float, default=0.45)
    parser.add_argument("--fit-layer-eta-lock-max", type=float, default=0.85)
    parser.add_argument("--fit-layer-eta-lock-steps", type=int, default=3)
    parser.add_argument(
        "--fit-layer-densification-power",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit layer pressure densification exponent during calibration.",
    )
    parser.add_argument("--fit-layer-densification-power-min", type=float, default=0.0)
    parser.add_argument("--fit-layer-densification-power-max", type=float, default=2.0)
    parser.add_argument("--fit-layer-densification-power-steps", type=int, default=3)
    parser.add_argument(
        "--layer-cubic",
        type=float,
        default=0.0,
        help="Cubic stiffening coefficient for --pressure-profile layer.",
    )
    parser.add_argument(
        "--layer-quintic",
        type=float,
        default=0.0,
        help="Quintic stiffening coefficient for --pressure-profile layer.",
    )
    parser.add_argument(
        "--pressure-sine-amplitude",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("AX", "AY", "AZ"),
        help="Per-axis pressure sine modulation amplitude for the midsole.",
    )
    parser.add_argument(
        "--pressure-sine-cycles",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("CX", "CY", "CZ"),
        help="Per-axis pressure sine modulation cycle count for the midsole.",
    )
    parser.add_argument(
        "--pressure-sine-phase",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("PX", "PY", "PZ"),
        help="Per-axis pressure sine modulation phase [rad] for the midsole.",
    )
    parser.add_argument("--kd", type=float, default=0.0, help="Newton kd for the compliant midsole.")
    parser.add_argument(
        "--fit-kd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit Newton kd during calibration.",
    )
    parser.add_argument("--fit-kd-min", type=float, default=0.0)
    parser.add_argument("--fit-kd-max", type=float, default=500.0)
    parser.add_argument("--fit-kd-steps", type=int, default=7)
    parser.add_argument("--sdf-resolution", type=int, default=128)
    parser.add_argument("--narrow-band", type=float, default=0.01)
    parser.add_argument("--rigid-contact-max", type=int, default=200000)
    parser.add_argument("--buffer-mult-contact", type=int, default=16)
    parser.add_argument("--buffer-mult-iso", type=int, default=16)
    parser.add_argument(
        "--contact-reduction",
        dest="no_contact_reduction",
        action="store_false",
        default=False,
        help="Enable reduced hydroelastic contacts for faster approximate Digital Instron runs.",
    )
    parser.add_argument(
        "--no-contact-reduction",
        dest="no_contact_reduction",
        action="store_true",
        help="Use unreduced hydroelastic contacts for higher-fidelity force calibration.",
    )
    parser.add_argument(
        "--fail-on-fixture-contact-overflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise before MuJoCo fixture replay would truncate Newton-generated contacts.",
    )
    parser.add_argument(
        "--pressure-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable stateful max-compression pressure memory in reduced hydroelastic contact export.",
    )
    parser.add_argument(
        "--pressure-memory-grid",
        type=int,
        nargs=3,
        metavar=("GX", "GY", "GZ"),
        default=(16, 8, 1),
        help="Coarse local-frame pressure-memory grid per shape pair and normal bin.",
    )
    parser.add_argument(
        "--pressure-memory-unloading-loss",
        type=float,
        default=0.45,
        help="Fraction of remembered compression proxy subtracted during unloading. Range: [0, 1].",
    )
    parser.add_argument(
        "--pressure-memory-recovery-tau-s",
        type=float,
        default=0.25,
        help="Pressure-memory recovery time constant while a cell remains active [s].",
    )
    parser.add_argument(
        "--pressure-memory-dt-s",
        type=float,
        default=1.0 / 240.0,
        help="Collision-step interval used for pressure-memory recovery [s].",
    )
    parser.add_argument(
        "--force-jump-limit",
        type=float,
        default=200.0,
        help="Maximum allowed adjacent-sample simulated top-force jump [N] before diagnostics report a violation.",
    )
    parser.add_argument(
        "--fail-on-force-jump",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise an error when adjacent simulated top-force jumps exceed --force-jump-limit.",
    )
    parser.add_argument(
        "--material-output-json",
        type=str,
        default=str(PROCESSED_DIR / "pressure_field_material.json"),
    )
    parser.add_argument(
        "--load-material",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load calibrated material parameters from --material-output-json for view mode.",
    )
    parser.add_argument(
        "--bundle-output-json",
        type=str,
        default=str(PROCESSED_DIR / "shoe_pressure_field_bundle.json"),
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default=str(PROCESSED_DIR / "digital_instron_hysteresis.png"),
    )
    parser.add_argument("--view-sample-stride", type=int, default=2)
    parser.add_argument(
        "--view-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop the prescribed Instron displacement trace in --mode view.",
    )
    return parser


def _normalize_cli_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        argv = sys.argv[1:]
    modes = {"preprocess", "run", "calibrate", "view", "view-calibration", "export-bundle"}
    if argv and argv[0] in modes:
        return ["--mode", argv[0], *argv[1:]]
    return argv


def _apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "compression_relax_steps", None) is None:
        args.compression_relax_steps = 4
    return args


def _format_float(value: Any, *, precision: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{precision}g}"


def _format_cli_report(summary: dict[str, Any] | list[Any]) -> str:
    if isinstance(summary, list):
        lines = ["Preprocess complete"]
        for item in summary:
            trace_name = Path(str(item.get("output_path", ""))).name
            lines.append(
                f"  {trace_name}: {item.get('samples', 'n/a')} samples, "
                f"peak {_format_float(item.get('force_peak_n'))} N"
            )
        return "\n".join(lines)

    mode = str(summary.get("mode", ""))
    if mode == "calibrate":
        parameters = dict(summary.get("parameters", {}))
        lines = [
            "Calibration complete",
            f"  final kh: {_format_float(parameters.get('kh'))} N/m",
            f"  final kd: {_format_float(parameters.get('kd'))}",
            f"  objective: {_format_float(parameters.get('combined_objective'))}",
            f"  cases: {', '.join(str(case) for case in summary.get('calibration_cases', [])) or 'n/a'}",
            f"  objective mode: {summary.get('calibration_objective', 'force')}",
            f"  search: {summary.get('calibration_search_method', 'n/a')} "
            f"({summary.get('calibration_kh_spacing', 'n/a')} kh, {summary.get('calibration_refine_stages', 'n/a')} stages)",
            f"  candidates: {summary.get('candidate_count', 'n/a')} "
            f"(batch {summary.get('calibration_batch_size', 'n/a')}, workers {summary.get('calibration_workers', 'n/a')})",
            f"  sample stride: search {summary.get('calibration_search_sample_stride', 'n/a')}, "
            f"final {summary.get('calibration_final_sample_stride', 'n/a')}",
        ]
        history = list(summary.get("calibration_search_history", []))
        if history:
            lines.append("  search stages:")
            for stage in history:
                invalid_count = int(stage.get("invalid_candidate_count", 0))
                invalid_text = f", invalid {invalid_count}" if invalid_count else ""
                if stage.get("status") == "all_invalid":
                    lines.append(
                        f"    {stage.get('stage')}: all invalid, candidates {stage.get('candidate_count', 'n/a')}"
                    )
                else:
                    lines.append(
                        f"    {stage.get('stage')}: kh {_format_float(stage.get('best_kh'))}, "
                        f"kd {_format_float(stage.get('best_kd'))}, "
                        f"objective {_format_float(stage.get('best_objective'))}, "
                        f"candidates {stage.get('candidate_count', 'n/a')}{invalid_text}"
                    )
        trials = dict(summary.get("trials", {}))
        if trials:
            lines.append("  trials:")
            for test_case, trial in trials.items():
                lines.append(
                    f"    {test_case}: RMSE {_format_float(trial.get('loop_rmse_n'))} N, "
                    f"R2 {_format_float(trial.get('force_r2'))}, "
                    f"peak err {_format_float(100.0 * float(trial.get('peak_relative_error', 0.0)), precision=3)}%, "
                    f"sim peak {_format_float(trial.get('sim_peak_force_n'))} N, "
                    f"max jump {_format_float(trial.get('max_adjacent_force_jump_n'))} N"
                )
        lines.extend(
            [
                f"  material: {summary.get('material_json', 'n/a')}",
                f"  bundle: {summary.get('bundle_json', 'n/a')}",
                f"  plot: {summary.get('plot_output', 'n/a')}",
            ]
        )
        return "\n".join(lines)

    if mode == "view":
        loaded_material = summary.get("loaded_material") or {}
        lines = [
            f"View complete ({summary.get('test_case', 'n/a')})",
            f"  material: {loaded_material.get('path', 'not loaded')}",
            f"  kh: {_format_float(loaded_material.get('kh'))} N/m",
            f"  kd: {_format_float(loaded_material.get('kd'))}",
            f"  last displacement: {_format_float(summary.get('last_displacement_mm'))} mm",
            f"  last top force: {_format_float(summary.get('last_raw_contact_force_n'))} N",
        ]
        stats = dict(summary.get("last_contact_stats", {}))
        if stats:
            lines.append(
                f"  contacts: top {stats.get('top_active_pair_contact_count', stats.get('active_pair_contact_count', 'n/a'))}, "
                f"bottom {stats.get('bottom_active_pair_contact_count', 'n/a')}"
            )
        return "\n".join(lines)

    if mode == "view-calibration":
        loaded_material = summary.get("loaded_material") or {}
        lines = [
            "Calibration batch view complete",
            f"  batch index: {summary.get('batch_index', 'n/a')}",
            f"  candidates: {summary.get('candidate_count', 'n/a')}",
            f"  cases: {', '.join(str(case) for case in summary.get('calibration_cases', [])) or 'n/a'}",
            f"  sample stride: {summary.get('calibration_search_sample_stride', 'n/a')}",
            f"  material: {loaded_material.get('path', 'not loaded')}",
        ]
        trials = list(summary.get("trials", []))
        if trials:
            lines.append("  trials:")
            for trial in trials:
                lines.append(
                    f"    candidate {trial.get('candidate_index')}, {trial.get('test_case')}: "
                    f"peak {_format_float(trial.get('peak_force_n'))} N"
                )
        return "\n".join(lines)

    if mode == "export-bundle":
        return "\n".join(
            [
                "Bundle export complete",
                f"  material: {summary.get('material_json', 'n/a')}",
                f"  bundle: {summary.get('bundle_json', 'n/a')}",
                f"  state model: {dict(summary.get('state_model', {})).get('type', 'n/a')}",
            ]
        )

    if mode == "run":
        return "\n".join(
            [
                f"Run complete ({summary.get('test_case', 'n/a')})",
                f"  RMSE: {_format_float(summary.get('loop_rmse_n'))} N",
                f"  R2: {_format_float(summary.get('force_r2'))}",
                f"  peak error: {_format_float(100.0 * float(summary.get('peak_relative_error', 0.0)), precision=3)}%",
                f"  sim peak: {_format_float(summary.get('sim_peak_force_n'))} N",
                f"  shoe motion: vz rms {_format_float(summary.get('rms_shoe_vertical_velocity_m_s'))} m/s, "
                f"omega rms {_format_float(summary.get('rms_shoe_free_axis_angular_velocity_rad_s'))} rad/s",
                f"  max force jump: {_format_float(summary.get('max_adjacent_force_jump_n'))} N "
                f"(limit {_format_float(summary.get('force_jump_limit_n'))} N)",
                f"  output: {summary.get('output_csv', 'n/a')}",
            ]
        )

    return json.dumps(summary, indent=2)


def _print_cli_report(args, summary: dict[str, Any] | list[Any]) -> None:
    if bool(args.report_json):
        print(json.dumps(summary, indent=2))
    else:
        print(_format_cli_report(summary))


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = _apply_mode_defaults(parser.parse_args(_normalize_cli_argv(argv)))
    if bool(getattr(args, "verbose", False)):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.mode == "preprocess":
        summaries = preprocess_defaults(args)
        _print_cli_report(args, [summary.__dict__ for summary in summaries])
        return

    if args.mode == "calibrate":
        summary = calibrate_digital_instron(args)
        _print_cli_report(args, summary)
        return

    if args.mode == "export-bundle":
        summary = export_bundle(args)
        _print_cli_report(args, summary)
        return

    if args.mode == "view":
        summary = view_digital_instron(args)
        _print_cli_report(args, summary)
        return

    if args.mode == "view-calibration":
        summary = view_calibration_batch(args)
        _print_cli_report(args, summary)
        return

    if args.trace_csv is None:
        args.trace_csv = str(_default_trace_for_case(args.test_case))
    if args.output_csv is None:
        args.output_csv = str(PROCESSED_DIR / f"{args.test_case}_sim_pressure_field.csv")
    summary = run_digital_instron(args)
    _print_cli_report(args, summary)


if __name__ == "__main__":
    main()
