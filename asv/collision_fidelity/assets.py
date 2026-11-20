# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Preview and builder dependencies are optional; keep geometry imports light.
# ruff: noqa: PLC0415
"""Procedural runtime meshes for the collision-fidelity scenes.

Experimental helpers. All geometry uses metres and Z up. Generated garments
are synthetic shell models, not measured fabric products. Generated geometry is
the rest state, not a simulated drape. Panel generation uses the existing NumPy and SciPy dependencies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CLOTH_NAMES",
    "SOLID_NAMES",
    "AddedAsset",
    "AssetMesh",
    "add_cloth_asset",
    "add_solid_asset",
    "cloth_mesh",
    "solid_mesh",
    "validate_asset",
]

CLOTH_NAMES = ("patch", "towel", "sheet", "trousers", "dress", "tshirt", "skirt", "tote", "pouch", "gusset_bag")
SOLID_NAMES = ("block", "pad", "sphere", "capsule", "torus", "branched")
_RESOLUTION = {"coarse": 1, "medium": 2, "fine": 3}
_COLORS = {
    "trousers": (0.18, 0.36, 0.69),
    "dress": (0.75, 0.22, 0.37),
    "tshirt": (0.25, 0.65, 0.66),
    "skirt": (0.59, 0.35, 0.70),
    "tote": (0.82, 0.62, 0.28),
    "pouch": (0.28, 0.60, 0.42),
    "gusset_bag": (0.74, 0.47, 0.26),
    "towel": (0.30, 0.62, 0.80),
    "sheet": (0.75, 0.80, 0.88),
}


@dataclass
class AssetMesh:
    """Rest geometry in SI units. Array indices are local to this mesh."""

    name: str
    vertices: np.ndarray
    """Particle positions [m], shape [vertex_count, 3]."""
    triangles: np.ndarray
    """Surface triangle indices, shape [triangle_count, 3]."""
    tetrahedra: np.ndarray | None = None
    """Volume element indices, shape [tet_count, 4], or None for cloth."""
    landmarks: dict[str, list[int]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class AddedAsset:
    """Builder particle range and global landmark indices for one asset."""

    mesh: AssetMesh
    particle_start: int
    particle_end: int
    landmark_indices: dict[str, list[int]]


def _sewn_panel(name, points, openings, depth, level):
    from scipy.spatial import Delaunay

    original_seams = [(i, (i + 1) % len(points)) for i in range(len(points)) if i not in openings]
    outline = np.asarray(points, dtype=float)
    spacing = float(np.max(np.ptp(outline, axis=0))) / (16 + 8 * level)
    boundary, seams = [], set()
    for i, start in enumerate(outline):
        end = outline[(i + 1) % len(outline)]
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing)))
        for j in range(count):
            a = len(boundary)
            boundary.append(start + (end - start) * j / count)
            if i not in openings:
                seams.add((a, a + 1))
    boundary_count = len(boundary)
    seams = {tuple(sorted((a, b % boundary_count))) for a, b in seams}

    def inside(q):
        result = np.zeros(len(q), dtype=bool)
        for i, a in enumerate(outline):
            b = outline[(i + 1) % len(outline)]
            if a[1] == b[1]:
                continue
            crosses = (a[1] > q[:, 1]) != (b[1] > q[:, 1])
            x_at_y = a[0] + (q[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            result ^= crosses & (q[:, 0] < x_at_y)
        return result

    candidates = []
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    for row, z in enumerate(np.arange(lo[1], hi[1] + spacing, spacing * np.sqrt(3) / 2)):
        for x in np.arange(lo[0] + (row % 2) * spacing / 2, hi[0] + spacing, spacing):
            candidates.append((x, z))
    candidates = np.asarray(candidates)
    distance = np.full(len(candidates), np.inf)
    for i, a in enumerate(outline):
        b = outline[(i + 1) % len(outline)]
        d = b - a
        u = np.clip((candidates - a) @ d / (d @ d), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(candidates - a - u[:, None] * d, axis=1))
    p = np.vstack((boundary, candidates[inside(candidates) & (distance > 0.58 * spacing)]))
    t = Delaunay(p).simplices
    oriented = (p[t[:, 1], 0] - p[t[:, 0], 0]) * (p[t[:, 2], 1] - p[t[:, 0], 1]) - (p[t[:, 1], 1] - p[t[:, 0], 1]) * (
        p[t[:, 2], 0] - p[t[:, 0], 0]
    )
    valid = inside(p[t].mean(axis=1)) & (np.abs(oriented) > 1e-10 * spacing * spacing)
    t, oriented = t[valid], oriented[valid]
    flip = oriented < 0
    t[flip] = t[flip][:, ::-1]
    edges = np.sort(np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    actual_boundary = {tuple(e) for e in unique[counts == 1]}
    expected_boundary = {tuple(sorted((i, (i + 1) % boundary_count))) for i in range(boundary_count)}
    if actual_boundary != expected_boundary:
        raise ValueError(
            f"Panel triangulation did not preserve all boundary segments: {name}; missing={expected_boundary - actual_boundary}; extra={actual_boundary - expected_boundary}"
        )
    seam_ids = {v for e in seams for v in e}
    # A chord between two seam vertices must not become a four-face edge when
    # the back panel is sewn on. Split such chords and give them interior nodes.
    points_refined = p.tolist()
    split = {}
    for tri in t:
        for a, b in zip(tri, np.roll(tri, -1), strict=True):
            key = tuple(sorted((int(a), int(b))))
            if a in seam_ids and b in seam_ids and key not in seams and key not in split:
                split[key] = len(points_refined)
                points_refined.append(((p[a] + p[b]) * 0.5).tolist())
    triangles_refined = []
    for tri in t:
        boundary = []
        for a, b in zip(tri, np.roll(tri, -1), strict=True):
            boundary.append(int(a))
            key = tuple(sorted((int(a), int(b))))
            if key in split:
                boundary.append(split[key])
        if len(boundary) == 3:
            triangles_refined.append(tuple(tri))
        else:
            center = len(points_refined)
            points_refined.append(p[tri].mean(axis=0).tolist())
            triangles_refined.extend((a, boundary[(i + 1) % len(boundary)], center) for i, a in enumerate(boundary))
    p, t = np.asarray(points_refined), np.asarray(triangles_refined, dtype=np.int32)
    distances = np.full(len(p), np.inf)
    for a, b in original_seams:
        start, end = np.asarray(points[a]), np.asarray(points[b])
        direction = end - start
        u = np.clip(((p - start) @ direction) / (direction @ direction), 0, 1)
        distances = np.minimum(distances, np.linalg.norm(p - (start + u[:, None] * direction), axis=1))
    # Lift the two rest panels apart while keeping the stitched curves coincident.
    offset = depth * np.sin(0.5 * np.pi * np.clip(distances / (2 * depth), 0, 1))
    vertices = np.column_stack((p[:, 0], -offset, p[:, 1])).tolist()
    back = []
    for i, (x, z) in enumerate(p):
        if i in seam_ids:
            back.append(i)
        else:
            back.append(len(vertices))
            vertices.append((x, offset[i], z))
    faces = np.vstack((t, np.asarray(back)[t[:, ::-1]]))
    return (
        np.asarray(vertices),
        faces,
        {"seam": sorted(seam_ids)},
        {
            "construction": "two original cut panels, welded seam vertices, unsewn openings",
            "panel_outline_xz_m": points,
            "open_panel_edges": sorted(openings),
            "panel_half_depth_m": depth,
            "panel_mesher": "Delaunay triangular lattice; boundary segments verified; no simulation",
            "panel_target_spacing_m": spacing,
        },
    )


def _quad_grid(vertices, faces, index, positions, flip=False):
    rows = []
    for row in positions:
        out = []
        for p in row:
            key = tuple(np.round(p, 10))
            if key not in index:
                index[key] = len(vertices)
                vertices.append(p)
            out.append(index[key])
        rows.append(out)
    for j in range(len(rows) - 1):
        for i in range(len(rows[j]) - 1):
            a, b, c, d = rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]
            pair = [(a, b, c), (a, c, d)]
            faces.extend([tuple(reversed(t)) for t in pair] if flip else pair)
    return rows


def _bag(name, level):
    width, height, depth = {"tote": (0.30, 0.35, 0.10), "pouch": (0.15, 0.20, 0.018), "gusset_bag": (0.25, 0.30, 0.12)}[
        name
    ]
    nx, ny, nz = 12 * level, max(2, 4 * level), 12 * level
    xs, ys, zs = (
        np.linspace(-width / 2, width / 2, nx + 1),
        np.linspace(-depth / 2, depth / 2, ny + 1),
        np.linspace(0, height, nz + 1),
    )
    vertices, faces, index = [], [], {}
    gusset = 0.025 if name == "gusset_bag" else 0.0

    def xmap(x, y):
        return x * (1 - 2 * gusset / width * (1 - abs(2 * y / depth)))

    for side in (-1, 1):
        _quad_grid(vertices, faces, index, [[(x, side * depth / 2, z) for x in xs] for z in zs], flip=side == 1)
        _quad_grid(
            vertices, faces, index, [[(xmap(side * width / 2, y), y, z) for y in ys] for z in zs], flip=side == -1
        )
    _quad_grid(vertices, faces, index, [[(xmap(x, y), y, 0.0) for x in xs] for y in ys], flip=True)
    handles = []
    if name != "pouch":
        # Each root spans one coarse rim edge. Refinement adds matching nodes.
        strip_width, radius, rise = width / 12, width / 4, 0.14
        for side in (-1, 1):
            positions = []
            for theta in np.linspace(0, np.pi, 20 * level + 1):
                center = np.array((-radius * np.cos(theta), side * depth / 2, height + rise * np.sin(theta)))
                normal = np.array((rise * np.cos(theta), 0, -radius * np.sin(theta)))
                normal /= np.linalg.norm(normal)
                positions.append(
                    [tuple(center + w * normal) for w in np.linspace(-strip_width / 2, strip_width / 2, level + 1)]
                )
            # Put roots on exact existing rim segments (both end edges weld).
            for end, center_x in ((0, -width / 4), (-1, width / 4)):
                root_center_x = center_x + width / 24
                root_x = np.linspace(root_center_x - strip_width / 2, root_center_x + strip_width / 2, level + 1)
                if end == -1:
                    root_x = root_x[::-1]
                positions[end] = [(float(x), side * depth / 2, height) for x in root_x]
            rows = _quad_grid(vertices, faces, index, positions, flip=side == 1)
            handles.extend(rows[len(rows) // 2])
    v = np.asarray(vertices)
    landmarks = {
        "handle_top": handles,
        "rim": np.flatnonzero(np.isclose(v[:, 2], height)).tolist(),
        "bottom": np.flatnonzero(np.isclose(v[:, 2], 0)).tolist(),
    }
    return (
        v,
        np.asarray(faces),
        landmarks,
        {
            "construction": "welded front, back, side and bottom panels; open rim; rim-attached handle strips",
            "bag_body_dimensions_m": [width, depth, height],
            "gusset_indent_m": gusset,
            "body_grid": [nx, ny, nz],
            "handle_count": 0 if name == "pouch" else 2,
        },
    )


def _complete(name, vertices, triangles, *, tetrahedra=None, landmarks=None, metadata=None, resolution="medium"):
    v, t = np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)
    tetrahedra = None if tetrahedra is None else np.asarray(tetrahedra, dtype=np.int32)
    lm = dict(landmarks or {})
    zmin, zmax = float(v[:, 2].min()), float(v[:, 2].max())
    lm.setdefault("top", np.flatnonzero(v[:, 2] >= zmax - max((zmax - zmin) * 0.035, 1.0e-5)).tolist())
    lm.setdefault("bottom", np.flatnonzero(v[:, 2] <= zmin + max((zmax - zmin) * 0.035, 1.0e-5)).tolist())
    lm.setdefault("left", [int(np.argmin(v[:, 0]))])
    lm.setdefault("right", [int(np.argmax(v[:, 0]))])
    payload = v.astype("<f4").tobytes() + t.astype("<i4").tobytes()
    if tetrahedra is not None:
        payload += tetrahedra.astype("<i4").tobytes()
    meta = {
        "name": name,
        "family": "cloth" if tetrahedra is None else "solid",
        "units": "m",
        "up_axis": "Z",
        "source": "original procedural benchmark geometry",
        "license": "Apache-2.0",
        "generator_revision": 2,
        "resolution": resolution,
        "dimensions_m": np.ptp(v, axis=0).tolist(),
        "bounds_m": [v.min(axis=0).tolist(), v.max(axis=0).tolist()],
        "vertex_count": len(v),
        "triangle_count": len(t),
        "tet_count": 0 if tetrahedra is None else len(tetrahedra),
        "geometry_sha256": hashlib.sha256(payload).hexdigest(),
        "color_rgb": _COLORS.get(name, (0.43, 0.65, 0.78)),
        "rest_shape": "generated geometry, no settling or simulated draping",
        "landmarks": lm,
        "synthetic_material": {
            "density_kg_m2": 0.25,
            "particle_radius_m": 0.002,
            "tri_ke": 1000.0,
            "tri_ka": 1000.0,
            "tri_kd": 0.01,
            "edge_ke": 0.01,
            "edge_kd": 0.001,
        }
        if tetrahedra is None
        else {
            "density_kg_m3": 100.0,
            "particle_radius_m": 0.001,
            "k_mu_pa": 20000.0,
            "k_lambda_pa": 20000.0,
            "k_damp_pa_s": 0.01,
        },
    }
    meta.update(metadata or {})
    return AssetMesh(name, v, t, tetrahedra, lm, meta)


def cloth_mesh(name: str, *, resolution: str = "medium") -> AssetMesh:
    """Create original shell geometry [m], with Z up and its bottom at Z=0.

    Garment seams share actual particle indices. Waist, cuff, neck, armhole,
    hem and bag-mouth openings remain open. Geometry resolution does not alter
    physical shell thickness; the builder contact radius is a separate setting.
    """
    level = _RESOLUTION[resolution]
    if name in ("patch", "towel", "sheet"):
        width, height = {"patch": (0.1, 0.1), "towel": (0.4, 0.6), "sheet": (1.0, 1.0)}[name]
        nx, nz = (8 * level, 8 * level) if name == "patch" else (12 * level, 16 * level)
        v, t = [], []
        _quad_grid(
            v,
            t,
            {},
            [[(x, 0.0, z) for x in np.linspace(-width / 2, width / 2, nx + 1)] for z in np.linspace(0, height, nz + 1)],
        )
        landmarks, metadata = {}, {"construction": "single rectangular cloth panel", "grid": [nx, nz]}
    elif name in ("tote", "pouch", "gusset_bag"):
        v, t, landmarks, metadata = _bag(name, level)
    elif name == "trousers":
        points = [
            (-0.18, 0.85),
            (-0.20, 0.48),
            (-0.18, 0),
            (-0.035, 0),
            (-0.025, 0.40),
            (0, 0.46),
            (0.025, 0.40),
            (0.035, 0),
            (0.18, 0),
            (0.20, 0.48),
            (0.18, 0.85),
        ]
        v, t, landmarks, metadata = _sewn_panel(name, points, {2, 7, 10}, 0.025, level)
    elif name == "dress":
        points = [
            (-0.38, 0),
            (0.38, 0),
            (0.18, 0.62),
            (0.15, 0.78),
            (0.18, 0.94),
            (0.105, 1.0),
            (0.065, 0.88),
            (-0.065, 0.88),
            (-0.105, 1.0),
            (-0.18, 0.94),
            (-0.15, 0.78),
            (-0.18, 0.62),
        ]
        v, t, landmarks, metadata = _sewn_panel(name, points, {0, 2, 3, 5, 6, 7, 9, 10}, 0.035, level)
    elif name == "tshirt":
        points = [
            (-0.24, 0),
            (0.24, 0),
            (0.24, 0.40),
            (0.41, 0.31),
            (0.49, 0.49),
            (0.25, 0.65),
            (0.10, 0.68),
            (0.07, 0.61),
            (-0.07, 0.61),
            (-0.10, 0.68),
            (-0.25, 0.65),
            (-0.49, 0.49),
            (-0.41, 0.31),
            (-0.24, 0.40),
        ]
        v, t, landmarks, metadata = _sewn_panel(name, points, {0, 3, 6, 7, 8, 11}, 0.035, level)
    elif name == "skirt":
        points = [(-0.34, 0), (0.34, 0), (0.17, 0.55), (-0.17, 0.55)]
        v, t, landmarks, metadata = _sewn_panel(name, points, {0, 2}, 0.045, level)
    else:
        raise ValueError(f"Unknown cloth asset {name!r}; choose from {CLOTH_NAMES}")
    return _complete(name, v, t, landmarks=landmarks, metadata=metadata, resolution=resolution)


def _surface(tets):
    face_map = {}
    for a, b, c, d in tets:
        for face in ((a, c, b), (a, b, d), (a, d, c), (b, c, d)):
            key = tuple(sorted(face))
            if key in face_map:
                face_map[key] = None
            else:
                face_map[key] = face
    return np.asarray([f for f in face_map.values() if f is not None], dtype=np.int32)


def _hex_tets(corners):
    a, b, c, d, e, f, g, h = corners
    return [(a, b, c, g), (a, c, d, g), (a, d, h, g), (a, h, e, g), (a, e, f, g), (a, f, b, g)]


def _tet_grid(axes, transform=None, keep=None, periodic=False):
    points = np.asarray([(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]])
    shape = tuple(len(a) for a in axes)
    ids = np.arange(len(points)).reshape(shape)
    tets = []
    for i in range(shape[0] if periodic else shape[0] - 1):
        ii = (i + 1) % shape[0]
        for j in range(shape[1] - 1):
            for k in range(shape[2] - 1):
                corners = [
                    ids[i, j, k],
                    ids[ii, j, k],
                    ids[ii, j + 1, k],
                    ids[i, j + 1, k],
                    ids[i, j, k + 1],
                    ids[ii, j, k + 1],
                    ids[ii, j + 1, k + 1],
                    ids[i, j + 1, k + 1],
                ]
                if keep is None or keep(points[corners].mean(axis=0)):
                    tets.extend(_hex_tets(corners))
    tets = np.asarray(tets, dtype=np.int32)
    used, inverse = np.unique(tets, return_inverse=True)
    points, tets = points[used], inverse.reshape(-1, 4)
    if transform is not None:
        points = transform(points)
    determinants = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]])
    inverted = determinants < 0
    tets[inverted, 2], tets[inverted, 3] = tets[inverted, 3].copy(), tets[inverted, 2].copy()
    points[:, 2] -= points[:, 2].min()
    return points, tets


def solid_mesh(name: str, *, resolution: str = "medium") -> AssetMesh:
    """Create an original conforming tet mesh in metres, with bottom Z=0."""
    level = _RESOLUTION[resolution]
    if name in ("block", "pad"):
        dims, cells = {"block": ((0.06, 0.04, 0.03), (4, 3, 2)), "pad": ((0.12, 0.08, 0.01), (8, 6, 1))}[name]
        axes = [np.linspace(-d / 2, d / 2, n * level + 1) for d, n in zip(dims, cells, strict=True)]
        v, tet = _tet_grid(axes)
        construction = "structured hexahedral grid split into six conforming tetrahedra per cell"
    elif name in ("sphere", "capsule"):
        n = 4 * level
        axes = [np.linspace(-1, 1, n + 1)] * 3

        def transform(p):
            length = np.linalg.norm(p, axis=1)
            radius = 0.025 if name == "sphere" else 0.0175
            scale = np.divide(np.max(np.abs(p), axis=1), length, out=np.zeros_like(length), where=length > 0)
            q = radius * p * scale[:, None]
            if name == "capsule":
                q[:, 2] += 0.0225 * p[:, 2]
            return q

        v, tet = _tet_grid(axes, transform)
        construction = "cube grid radially mapped to sphere" + (
            " and axially stretched to a rounded capsule surrogate" if name == "capsule" else ""
        )
    elif name == "torus":
        axes = [
            np.linspace(0, 2 * np.pi, 16 * level, endpoint=False),
            np.linspace(-1, 1, 2 * level + 1),
            np.linspace(-1, 1, 2 * level + 1),
        ]

        def transform(p):
            uv = p[:, 1:]
            r = np.linalg.norm(uv, axis=1)
            s = np.divide(np.max(np.abs(uv), axis=1), r, out=np.zeros_like(r), where=r > 0)
            uv = 0.01 * uv * s[:, None]
            return np.column_stack(((0.03 + uv[:, 0]) * np.cos(p[:, 0]), (0.03 + uv[:, 0]) * np.sin(p[:, 0]), uv[:, 1]))

        v, tet = _tet_grid(axes, transform, periodic=True)
        construction = "periodic swept disk grid with major radius 0.03 m and tube radius 0.01 m"
    elif name == "branched":
        axes = [
            np.linspace(-0.06, 0.06, 12 * level + 1),
            np.linspace(-0.06, 0.06, 12 * level + 1),
            np.linspace(-0.012, 0.012, 2 * level + 1),
        ]
        v, tet = _tet_grid(axes, keep=lambda p: abs(p[0]) < 0.02 or abs(p[1]) < 0.02)
        construction = "original four-arm cross, rectangular arms; conforming union of voxel cells"
    else:
        raise ValueError(f"Unknown solid asset {name!r}; choose from {SOLID_NAMES}")
    return _complete(
        name, v, _surface(tet), tetrahedra=tet, resolution=resolution, metadata={"construction": construction}
    )


def add_cloth_asset(
    builder,
    name: str,
    *,
    resolution: str = "medium",
    pos=(0.0, 0.0, 0.0),
    rot=None,
    scale: float = 1.0,
    density: float = 0.25,
    particle_radius: float = 0.002,
    **material,
) -> AddedAsset:
    """Add an original shell; density is kg/m², radius and position are m.

    Extra keyword arguments go to ModelBuilder.add_cloth_mesh. They may set
    elastic constants, color and velocity. This helper never pins particles.
    """
    import warp as wp

    mesh = cloth_mesh(name, resolution=resolution)
    start = len(builder.particle_q)
    kwargs = {
        "tri_ke": 1000.0,
        "tri_ka": 1000.0,
        "tri_kd": 0.01,
        "edge_ke": 0.01,
        "edge_kd": 0.001,
        "vel": (0.0, 0.0, 0.0),
        "color": mesh.metadata["color_rgb"],
    }
    kwargs.update(material)
    builder.add_cloth_mesh(
        pos=pos,
        rot=wp.quat_identity() if rot is None else rot,
        scale=scale,
        vertices=mesh.vertices.tolist(),
        indices=mesh.triangles.ravel().tolist(),
        density=density,
        particle_radius=particle_radius,
        **kwargs,
    )
    return AddedAsset(
        mesh, start, len(builder.particle_q), {k: [start + i for i in ids] for k, ids in mesh.landmarks.items()}
    )


def add_solid_asset(
    builder,
    name: str,
    *,
    resolution: str = "medium",
    pos=(0.0, 0.0, 0.0),
    rot=None,
    scale: float = 1.0,
    density: float = 100.0,
    particle_radius: float = 0.001,
    **material,
) -> AddedAsset:
    """Add original tet geometry; density is kg/m³, radius and position are m."""
    import warp as wp

    mesh = solid_mesh(name, resolution=resolution)
    start = len(builder.particle_q)
    kwargs = {
        "k_mu": 20000.0,
        "k_lambda": 20000.0,
        "k_damp": 0.01,
        "vel": (0.0, 0.0, 0.0),
        "color": mesh.metadata["color_rgb"],
    }
    kwargs.update(material)
    builder.add_soft_mesh(
        pos=pos,
        rot=wp.quat_identity() if rot is None else rot,
        scale=scale,
        vertices=mesh.vertices.tolist(),
        indices=mesh.tetrahedra.ravel().tolist(),
        density=density,
        particle_radius=particle_radius,
        **kwargs,
    )
    return AddedAsset(
        mesh, start, len(builder.particle_q), {k: [start + i for i in ids] for k, ids in mesh.landmarks.items()}
    )


def validate_asset(mesh: AssetMesh) -> dict:
    """Check finite coordinates, connectivity, manifold edges and tet volume."""
    v, t = mesh.vertices.astype(float), mesh.triangles
    for name, indices, width in (("triangles", t, 3), ("tetrahedra", mesh.tetrahedra, 4)):
        if indices is not None and (
            indices.ndim != 2
            or indices.shape[1] != width
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(v))
        ):
            raise ValueError(f"{name} must contain {width} valid integer vertex indices per element")
    area = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    lengths = np.stack([np.linalg.norm(v[t[:, (i + 1) % 3]] - v[t[:, i]], axis=1) for i in range(3)], axis=1)
    angles = []
    for i in range(3):
        a, b, c = lengths[:, i], lengths[:, (i + 1) % 3], lengths[:, (i + 2) % 3]
        cosine = np.divide(a * a + b * b - c * c, 2 * a * b, out=np.ones_like(a), where=a * b > 0)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    minimum_angles = np.min(angles, axis=0)
    edges = np.sort(np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]])), axis=1)
    directed_edges = np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(directed_edges[:, 0] < directed_edges[:, 1], 1, -1))
    parent = np.arange(len(v))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    elements = t if mesh.tetrahedra is None else mesh.tetrahedra
    for element in elements:
        for j in element[1:]:
            parent[find(j)] = find(element[0])
    components = len({find(i) for i in range(len(v))})
    result = {
        "finite": bool(np.isfinite(v).all()),
        "min_triangle_area_m2": float(area.min()),
        "surface_area_m2": float(area.sum()),
        "min_triangle_angle_degrees": float(minimum_angles.min()),
        "triangles_with_angle_below_10_degrees": int(np.sum(minimum_angles < 10)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "boundary_edges": int(np.sum(counts == 1)),
        "inconsistent_winding_edges": int(np.sum((counts == 2) & (orientation != 0))),
        "connected_components": components,
    }
    if mesh.tetrahedra is not None:
        tet = mesh.tetrahedra
        volumes = np.linalg.det(v[tet[:, 1:]] - v[tet[:, :1]]) / 6
        result.update(
            min_tet_volume_m3=float(volumes.min()),
            volume_m3=float(volumes.sum()),
            nonpositive_tets=int(np.sum(volumes <= 0)),
        )
    result["valid"] = bool(
        result["finite"]
        and area.min() > 1e-14
        and result["nonmanifold_edges"] == 0
        and result["inconsistent_winding_edges"] == 0
        and components == 1
        and result.get("nonpositive_tets", 0) == 0
    )
    return result
