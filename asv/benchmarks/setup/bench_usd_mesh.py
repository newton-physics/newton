# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from asv_runner.benchmarks.mark import skip_benchmark_if

try:
    from pxr import Sdf, Usd, UsdGeom, Vt

    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False

import newton.usd

# Grid resolutions chosen so the corner counts bracket a typical robot visual
# mesh set: a 64x64 grid is ~25k corners (one link), 256x256 is ~390k (a full
# articulation's visuals).
_GRID_RESOLUTIONS = [64, 256]

# Fraction of corners carrying both a hard crease and a UV seam. ``0.0`` is the
# common case for exported visual meshes -- every vertex is already referenced
# by a single corner, so no corner needs splitting. ``0.25`` forces a quarter
# of the corners through the sequential clustering fallback.
_SPLIT_FRACTIONS = [0.0, 0.25]


def _build_facevarying_mesh(stage, grid: int, split_fraction: float):
    """Author a textured grid mesh whose faceVarying normals force ``split_fraction`` splits."""
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(grid + 1), np.arange(grid + 1), indexing="ij")
    points = np.stack([xs.ravel(), ys.ravel(), np.sin(xs.ravel() * 0.3)], axis=1).astype(np.float32)

    quad = np.arange(grid * grid)
    row, col = quad // grid, quad % grid
    lower_left = row * (grid + 1) + col
    upper_left = lower_left + (grid + 1)
    triangles = np.stack(
        [
            np.stack([lower_left, upper_left, upper_left + 1], -1),
            np.stack([lower_left, upper_left + 1, lower_left + 1], -1),
        ],
        axis=1,
    ).reshape(-1, 3)
    indices = triangles.ravel().astype(np.int32)

    corner_normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(indices), 1))
    # Jitter stays well inside the 25-degree default threshold, so it never
    # splits on its own; the creases below are what force the fallback.
    corner_normals += rng.normal(scale=0.01, size=corner_normals.shape).astype(np.float32)
    creased = rng.random(len(indices)) < split_fraction
    corner_normals[creased] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    corner_normals /= np.linalg.norm(corner_normals, axis=1, keepdims=True)

    # A UV derived from the vertex position alone gives every corner of a vertex
    # the same value, which would leave the UV comparison unexercised; offsetting
    # the creased corners puts a texture seam on the same corners.
    corner_uvs = (points[indices][:, :2] / grid).astype(np.float32)
    corner_uvs[creased] += np.float32(0.5)

    mesh = UsdGeom.Mesh.Define(stage, "/grid")
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices))
    api = UsdGeom.PrimvarsAPI(mesh)
    normals = api.CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.faceVarying)
    normals.Set(Vt.Vec3fArray.FromNumpy(corner_normals))
    uvs = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    uvs.Set(Vt.Vec2fArray.FromNumpy(corner_uvs))
    return mesh.GetPrim()


class GetMeshFaceVaryingSplit:
    """Time ``newton.usd.get_mesh`` on meshes whose faceVarying normals drive vertex splitting."""

    params = (_GRID_RESOLUTIONS, _SPLIT_FRACTIONS)
    param_names = ["grid", "split_fraction"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1
    timeout = 1800

    def setup(self, grid, split_fraction):
        if not USD_AVAILABLE:
            return
        self.stage = Usd.Stage.CreateInMemory()
        self.prim = _build_facevarying_mesh(self.stage, grid, split_fraction)
        # Warm the import path so the first timed call is not paying one-time
        # import/attribute-resolution costs.
        newton.usd.get_mesh(self.prim, load_normals=True, load_uvs=True, compute_inertia=False)

    @skip_benchmark_if(not USD_AVAILABLE)
    def time_get_mesh(self, grid, split_fraction):
        newton.usd.get_mesh(self.prim, load_normals=True, load_uvs=True, compute_inertia=False)
