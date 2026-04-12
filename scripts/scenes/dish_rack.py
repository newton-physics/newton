# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dish rack scene: GSO kitchenware dropped onto a GSO dish drainer.

All assets come from the ``kevinzakka/mujoco_scanned_objects`` port of Google
Scanned Objects.  Each object ships with a textured render mesh
(``model.obj`` + ``texture.png``) and a pre-computed V-HACD convex
decomposition; we use the full textured mesh here and let MuJoCo Warp take
its automatic convex hull for collision.

Assets live under ``scripts/assets/gso/<name>/``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.solvers

# --- Timing / solver defaults ---

DT_OUTER = 0.01
TOL = 1e-3
DT_INNER_MIN = 1e-6
LOG_EVERY = 250

# --- GSO asset catalog -------------------------------------------------------
#
# Each entry: (folder stem under scripts/assets/gso, target mass [kg], scale).
# Target masses are real-world ballpark; GSO meshes are unit-density, so we
# rescale inertia from the convex hull to match.  Scale is applied at
# shape-add time (render + collision); inertia is rescaled accordingly.

_GSO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "gso"
_OBJAVERSE_ROOT = Path(__file__).resolve().parents[1] / "assets" / "objaverse"
_STL_ROOT = Path(__file__).resolve().parents[1] / "assets" / "dish-rack-2.snapshot.2"

# Standing slotted dish rack (STL).  Raw units are millimeters, Z-up,
# bbox (461 × 94 × 373) mm.  Scale 1e-3 converts to metres directly.
DRAINER_PATH = _STL_ROOT / "Escurridor platos.STL"
DRAINER_SCALE = 1e-3

# Uniform post-tuning scale bump applied to every object.
_GLOBAL_OBJECT_SCALE = 1.30

# name -> (asset path, target mass [kg], base scale applied to mesh verts).
# Effective scale = base * _GLOBAL_OBJECT_SCALE.
_ASSET_CATALOG: dict[str, tuple[Path, float, float]] = {
    "plate": (_GSO_ROOT / "Threshold_Dinner_Plate_Square_Rim_White_Porcelain" / "model.obj", 0.20, 0.45),
    "mug":   (_GSO_ROOT / "Cole_Hardware_Mug_Classic_Blue" / "model.obj",                    0.30, 0.90),
    "bowl":  (_GSO_ROOT / "Threshold_Bead_Cereal_Bowl_White" / "model.obj",                  0.18, 0.65),
    # Objaverse GLBs: raw units are arbitrary; fork long-axis 256 -> 0.15 m.
    "fork":  (_OBJAVERSE_ROOT / "fork" / "model.glb",  0.05, 0.15 / 256.0),
    "spoon": (_OBJAVERSE_ROOT / "spoon" / "model.glb", 0.04, 0.18 / 50.0),
}

OBJECT_MIX: tuple[tuple[str, int], ...] = (
    ("mug",   1),
    ("bowl",  1),
    ("fork",  1),
)
OBJECTS_PER_WORLD = sum(count for _, count in OBJECT_MIX)

# --- Drainer footprint + drop volume [m] ------------------------------------
#
# After Y-up -> Z-up rotation the basket bbox is
#   x ∈ [-0.946, 0.946], z ∈ [0, 1.425], y ∈ [-0.584, 0.582]  (raw units),
# then scaled by DRAINER_SCALE.  It's centered on its bbox midpoint so the
# bottom sits at z = -0.5 * scaled_height; we lift by that offset.

# STL is Y-up with bbox (461, 94, 373) mm: long axis X (tray length),
# Y=peg height, Z=tray depth.  After Y-up -> Z-up rotation, Z becomes peg
# height (9.4 cm) and Y becomes tray depth (37 cm).
DRAINER_HALF_X = 0.461 * 0.5
DRAINER_HALF_Y = 0.373 * 0.5
DRAINER_TOP_Z = 0.094

# Objects drop from above the rack, centered over its footprint.
DROP_XY_HALF_X = DRAINER_HALF_X * 0.6
DROP_XY_HALF_Y = DRAINER_HALF_Y * 0.4
DROP_Z_MIN = DRAINER_TOP_Z + 0.15
DROP_Z_MAX = DROP_Z_MIN + 0.40


# --- Mesh loader -------------------------------------------------------------

_MESH_CACHE: dict[str, newton.Mesh] = {}


def _load_mesh_from_path(
    asset_path: Path, target_mass: float, scale: float,
    *, y_up_to_z_up: bool = False,
) -> newton.Mesh:
    """Load an OBJ or GLB as a :class:`newton.Mesh` with clean inertia.

    The mesh is centered on its bbox midpoint so the body origin is the
    geometric center.  Textures come from a sibling ``texture.png`` if present,
    otherwise from GLB-embedded ``baseColorTexture``.  Collision under MuJoCo
    Warp uses the automatic convex hull.  Inertia is computed from the convex
    hull, rescaled for `scale` (volume ∝ scale³), then rescaled to
    `target_mass` (inertia ∝ scale⁵).  With `y_up_to_z_up=True` the mesh is
    rotated so +Y (asset up) maps to +Z (Newton up).
    """
    cache_key = f"{asset_path}:{scale}:{y_up_to_z_up}"
    if cache_key in _MESH_CACHE:
        return _MESH_CACHE[cache_key]

    import trimesh  # noqa: PLC0415

    if not asset_path.exists():
        raise FileNotFoundError(f"Asset mesh not found: {asset_path}")

    raw = trimesh.load(str(asset_path), process=False, force="mesh")

    raw_v = np.asarray(raw.vertices, dtype=np.float64)
    if y_up_to_z_up:
        # Rotate -90° about X: (x, y, z) -> (x, -z, y).  Asset-up Y -> world-up Z.
        raw_v = np.stack([raw_v[:, 0], -raw_v[:, 2], raw_v[:, 1]], axis=1)
    bbox_center = 0.5 * (raw_v.min(axis=0) + raw_v.max(axis=0))
    verts = (raw_v - bbox_center).astype(np.float32)
    faces = np.asarray(raw.faces, dtype=np.int32).flatten()

    uvs = None
    texture = None
    if hasattr(raw.visual, "uv") and raw.visual.uv is not None:
        uvs = np.asarray(raw.visual.uv, dtype=np.float32)
    png_path = asset_path.with_name("texture.png")
    if png_path.exists():
        from PIL import Image  # noqa: PLC0415
        texture = np.asarray(Image.open(png_path))
    elif uvs is not None and hasattr(raw.visual, "material"):
        embedded = getattr(raw.visual.material, "baseColorTexture", None)
        if embedded is not None:
            texture = np.asarray(embedded)

    mesh = newton.Mesh(
        verts, faces,
        uvs=uvs,
        texture=texture,
        compute_inertia=False,
    )

    hull = trimesh.Trimesh(vertices=verts, faces=raw.faces).convex_hull
    hull_v = np.asarray(hull.vertices, dtype=np.float32)
    hull_i = np.asarray(hull.faces, dtype=np.int32).flatten()
    from newton._src.geometry.inertia import compute_inertia_mesh  # noqa: PLC0415
    mass_hull, com_hull, inertia_hull, _ = compute_inertia_mesh(
        1.0, hull_v, hull_i, is_solid=True,
    )
    if float(mass_hull) > 0.0:
        hull_mass_scaled = float(mass_hull) * (scale ** 3)
        mass_ratio = target_mass / hull_mass_scaled
        inertia_scale = (scale ** 5) * mass_ratio
        mesh.mass = target_mass
        mesh.com = wp.vec3(*(np.asarray(com_hull) * scale))
        mesh.inertia = wp.mat33(np.asarray(inertia_hull).reshape(3, 3) * inertia_scale)

    _MESH_CACHE[cache_key] = mesh
    return mesh


def _load_object_mesh(name: str) -> tuple[newton.Mesh, float]:
    asset_path, target_mass, base_scale = _ASSET_CATALOG[name]
    scale = base_scale * _GLOBAL_OBJECT_SCALE
    return _load_mesh_from_path(asset_path, target_mass, scale), scale


def _spawn_object(
    builder: newton.ModelBuilder, xform: wp.transform, cfg, name: str,
) -> int:
    body = builder.add_body(xform=xform)
    mesh, scale = _load_object_mesh(name)
    builder.add_shape_mesh(
        body, mesh=mesh, cfg=cfg,
        scale=wp.vec3(scale, scale, scale),
    )
    return body


OBJECT_SPAWNERS: dict[str, Callable[[newton.ModelBuilder, wp.transform, object], int]] = {
    name: (lambda b, x, c, n=name: _spawn_object(b, x, c, n))
    for name in _ASSET_CATALOG
}


def _object_sequence() -> list[str]:
    seq: list[str] = []
    for name, count in OBJECT_MIX:
        seq.extend([name] * count)
    return seq


# --- Dish drainer (static rack) ---------------------------------------------

_COACD_ROOT = _STL_ROOT / "coacd"


def _load_coacd_parts(rotated_bbox_center: np.ndarray) -> list[newton.Mesh]:
    """Load the CoACD decomposition of the STL as per-part :class:`newton.Mesh`.

    Each part's verts are rotated Y-up → Z-up and shifted by the shared
    rotated-bbox center so all parts align with the centered visual mesh.
    """
    import trimesh  # noqa: PLC0415

    part_paths = sorted(_COACD_ROOT.glob("part_*.obj"))
    if not part_paths:
        raise FileNotFoundError(
            f"No CoACD parts in {_COACD_ROOT} -- regenerate the decomposition."
        )

    out: list[newton.Mesh] = []
    for p in part_paths:
        raw = trimesh.load(str(p), process=False, force="mesh")
        v = np.asarray(raw.vertices, dtype=np.float64)
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)  # Y-up -> Z-up
        v = (v - rotated_bbox_center).astype(np.float32)
        f = np.asarray(raw.faces, dtype=np.int32).flatten()
        out.append(newton.Mesh(v, f, compute_inertia=False))
    return out


def _rotated_bbox_center() -> np.ndarray:
    """Bbox midpoint of the STL after Y-up → Z-up rotation (pre-scale)."""
    import trimesh  # noqa: PLC0415

    raw = trimesh.load(str(DRAINER_PATH), process=False, force="mesh")
    v = np.asarray(raw.vertices, dtype=np.float64)
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    return 0.5 * (v.min(axis=0) + v.max(axis=0))


def _add_drainer(builder: newton.ModelBuilder, cfg) -> None:
    """Static slotted dish rack with CoACD-decomposed collision.

    Render uses the full STL; physics uses 64 convex parts so the pegs and
    slots actually contain plates under MuJoCo Warp's convex-per-shape model.
    """
    visual_mesh = _load_mesh_from_path(
        DRAINER_PATH, target_mass=1.0, scale=DRAINER_SCALE, y_up_to_z_up=True,
    )
    collision_parts = _load_coacd_parts(_rotated_bbox_center())

    xform = wp.transform(
        p=wp.vec3(0.0, 0.0, DRAINER_TOP_Z * 0.5),
        q=wp.quat_identity(),
    )
    scale_vec = wp.vec3(DRAINER_SCALE, DRAINER_SCALE, DRAINER_SCALE)

    visual_cfg = cfg.copy()
    visual_cfg.has_shape_collision = False
    builder.add_shape_mesh(
        body=-1, xform=xform, mesh=visual_mesh, cfg=visual_cfg, scale=scale_vec,
    )

    collision_cfg = cfg.copy()
    collision_cfg.is_visible = False
    for piece in collision_parts:
        builder.add_shape_mesh(
            body=-1, xform=xform, mesh=piece, cfg=collision_cfg, scale=scale_vec,
        )


# --- Model builders ----------------------------------------------------------

def build_template() -> newton.ModelBuilder:
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(template)

    # ke/kd translate to MuJoCo solref = (2/kd, kd/(2*sqrt(ke))) via
    # convert_solref() in newton/_src/solvers/mujoco/kernels.py:171.
    # Target: timeconst = 20 ms (= 2 * DT_OUTER, MuJoCo's stability floor)
    # with dampratio ~2 so seam-induced contact energy (from the CoACD
    # decomposition) bleeds out instead of re-exciting adjacent hulls.
    cfg_obj = newton.ModelBuilder.ShapeConfig(ke=500, kd=100, mu=0.5, margin=0.002)
    cfg_rack = newton.ModelBuilder.ShapeConfig(ke=500, kd=100, mu=0.8, margin=0.002)

    _add_drainer(template, cfg_rack)

    sequence = _object_sequence()
    for i, name in enumerate(sequence):
        z = DROP_Z_MIN + i * 0.10
        xform = wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity())
        OBJECT_SPAWNERS[name](template, xform, cfg_obj)

    return template


def build_model(n_worlds: int) -> newton.Model:
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()
    return builder.finalize()


def _random_unit_quaternion(rng) -> tuple[float, float, float, float]:
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    a1 = 2.0 * math.pi * u2
    a2 = 2.0 * math.pi * u3
    return (
        s1 * math.sin(a1), s1 * math.cos(a1),
        s2 * math.sin(a2), s2 * math.cos(a2),
    )


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """N worlds with per-world randomized object transforms (fixed base seed)."""
    model = build_model(n_worlds)

    joint_q_np = model.joint_q.numpy()
    body_q_np = model.body_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds
    bodies_per_world = model.body_count // n_worlds

    for w in range(n_worlds):
        rng = np.random.default_rng(seed + w)
        for b in range(bodies_per_world):
            x = rng.uniform(-DROP_XY_HALF_X, DROP_XY_HALF_X)
            y = rng.uniform(-DROP_XY_HALF_Y, DROP_XY_HALF_Y)
            z = rng.uniform(DROP_Z_MIN, DROP_Z_MAX)
            qx, qy, qz, qw = _random_unit_quaternion(rng)

            base = w * coords_per_world + b * 7
            joint_q_np[base + 0] = x
            joint_q_np[base + 1] = y
            joint_q_np[base + 2] = z
            joint_q_np[base + 3] = qx
            joint_q_np[base + 4] = qy
            joint_q_np[base + 5] = qz
            joint_q_np[base + 6] = qw

            body_idx = w * bodies_per_world + b
            body_q_np[body_idx] = (x, y, z, qx, qy, qz, qw)

    model.joint_q.assign(joint_q_np)
    model.body_q.assign(body_q_np)

    return model


def make_solver(
    model: newton.Model,
    tol: float = TOL,
    dt_mode: str = "per_world",
) -> newton.solvers.SolverMuJoCoCENIC:
    return newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=tol,
        dt_inner_init=DT_OUTER,
        dt_inner_min=DT_INNER_MIN,
        dt_inner_max=DT_OUTER,
        dt_mode=dt_mode,
        nconmax=2048,
        njmax=8192,
    )
