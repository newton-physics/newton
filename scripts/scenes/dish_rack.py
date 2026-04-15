# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dish rack scene: LBM (Toyota) meshes dropped onto the LBM dish drying rack.

All geometry comes from ``lbm_eval_models`` (Toyota Research Institute).
Visual and collision use the same glTF meshes; MuJoCo Warp auto-hulls each
part for collision.  The rack is three welded parts:

  * ``base``           -- thin slab tray (collision + visual)
  * ``wireframe``      -- the wire cage (visual only; its convex hull is a
    solid box which would block objects from entering the rack -- true
    wire-level contact requires SDF/hydroelastic and is a follow-up)
  * ``utensil_holder`` -- small cup at one end (collision + visual)

Masses and inertia tensors for dynamic objects come from LBM's SDFormat
``.sdf`` descriptors; rack parts are static (body=-1) so their inertial data
is ignored.  glTF is Y-up; meshes are rotated to Drake/Newton Z-up at load.

Assets live under ``scripts/assets/lbm/``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
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

# --- LBM asset catalog -------------------------------------------------------

_LBM_ROOT = Path(__file__).resolve().parents[1] / "assets" / "lbm"

_RACK_BASE_GLTF = _LBM_ROOT / "drying_racks" / "assets" / "sweet_home_dish_drying_rack_base.gltf"
_RACK_WIREFRAME_GLTF = _LBM_ROOT / "drying_racks" / "assets" / "sweet_home_dish_drying_rack_wireframe.gltf"
_RACK_UTENSIL_GLTF = _LBM_ROOT / "drying_racks" / "assets" / "sweet_home_dish_drying_rack_utensil_holder.gltf"

# Welded transforms from sweet_home_dish_drying_rack.dmd.yaml (Drake Z-up).
# Base is the root; wireframe attaches at base's nominal_wireframe_origin__z_up
# frame (translation + tiny pitch); utensil holder attaches to wireframe::origin
# with an explicit offset and -4.5 deg pitch.
_WIREFRAME_OFFSET_P = wp.vec3(-0.00527, -0.01075, 0.009861)
_WIREFRAME_OFFSET_PITCH_RAD = 0.013858
_UTENSIL_OFFSET_P = wp.vec3(-0.1518, 0.0735, 0.11335)
_UTENSIL_OFFSET_PITCH_RAD = math.radians(-4.5)

# name -> (gltf path, sdf path, target mass [kg] or None to use LBM mass,
#          scale, material).  When target_mass is None the LBM-provided mass
#          and inertia are used verbatim (recommended); overriding rescales
#          inertia proportionally.  Material keys into per-shape ShapeConfig.
_LBM_MUG_DIR = _LBM_ROOT / "mugs"
_LBM_FORK_DIR = _LBM_ROOT / "forks"

_ASSET_CATALOG: dict[str, tuple[Path, Path, float | None, float, str]] = {
    "mug": (
        _LBM_MUG_DIR / "assets" / "mug_inomata_ceramic_dense_patterned_yellow_mesh_collision.gltf",
        _LBM_MUG_DIR / "mug_inomata_ceramic_dense_patterned_yellow_mesh_collision.sdf",
        None,
        1.0,
        "ceramic",
    ),
    "fork": (
        _LBM_FORK_DIR / "assets" / "cambridge_jubilee_stainless_plastic_black_fork.gltf",
        _LBM_FORK_DIR / "cambridge_jubilee_stainless_plastic_black_fork.sdf",
        None,
        1.0,
        "metal",
    ),
}

OBJECT_MIX: tuple[tuple[str, int], ...] = (
    ("mug", 1),
    ("fork", 2),
)
OBJECTS_PER_WORLD = sum(count for _, count in OBJECT_MIX)

# --- Drainer footprint + drop volume [m] ------------------------------------
#
# LBM rack bboxes after glTF Y-up -> Z-up rotation (x, y, z) <- (x, -z, y):
#   base:      x ∈ [-0.154, 0.154], y ∈ [-0.213, 0.213], z ∈ [0, 0.032]
#   wireframe: x ∈ [-0.154, 0.154], y ∈ [-0.222, 0.222], z ∈ [0, 0.114]
# Wireframe is offset +0.009861 m in z via the weld, so the rack top is at
# ~0.124 m.  Footprint used for drop zone is wireframe's xy extent.

DRAINER_HALF_X = 0.154
DRAINER_HALF_Y = 0.222
DRAINER_TOP_Z = 0.124

# Objects drop from above the rack, centered over its footprint.  The rack's
# long axis is y in LBM orientation (44 cm) so the drop zone is wider in y.
DROP_XY_HALF_X = 0.06
DROP_XY_HALF_Y = 0.08
DROP_Z_MIN = DRAINER_TOP_Z + 0.15
DROP_Z_MAX = DROP_Z_MIN + 0.40


# --- LBM mesh loader ---------------------------------------------------------

_MESH_CACHE: dict[Path, newton.Mesh] = {}


def _gltf_y_up_to_z_up(verts: np.ndarray) -> np.ndarray:
    """Rotate -90° about X: glTF (+Y up) -> Drake/Newton (+Z up).

    Maps (x, y, z) -> (x, -z, y) so a mesh authored with +Y as vertical lands
    with +Z vertical in Newton's world frame.  LBM SDFormat poses and inertia
    tensors are already Z-up (Drake convention) so no further adjustment is
    needed for values parsed from the ``.sdf`` file.
    """
    return np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)


def _parse_lbm_inertial(sdf_path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Parse ``<inertial>`` from an LBM SDFormat file.

    Returns ``(mass [kg], com [m, Z-up, len-3], inertia [kg m^2, 3x3, Z-up])``.
    The inertia is read as a symmetric tensor about the COM.
    """
    root = ET.parse(str(sdf_path)).getroot()
    inertial = root.find(".//inertial")
    if inertial is None:
        raise ValueError(f"No <inertial> in {sdf_path}")
    pose = np.fromstring(inertial.findtext("pose", "0 0 0 0 0 0"), sep=" ")
    com = pose[:3].astype(np.float64)
    mass = float(inertial.findtext("mass", "0"))
    it = inertial.find("inertia")
    ixx = float(it.findtext("ixx", "0"))
    ixy = float(it.findtext("ixy", "0"))
    ixz = float(it.findtext("ixz", "0"))
    iyy = float(it.findtext("iyy", "0"))
    iyz = float(it.findtext("iyz", "0"))
    izz = float(it.findtext("izz", "0"))
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return mass, com, inertia


# Per-mesh convex-hull vertex budget.  MuJoCo Warp auto-hulls every mesh to
# this many vertices via QHull's TA flag.  The default of 64 (Newton's
# :attr:`newton.Mesh.MAX_HULL_VERTICES`) produces near-coplanar hulls for
# thin LBM parts (fork, rack base, utensil cup), which yield redundant
# contact points along ridges and visible jitter even at 0.1 ms dt.  Raising
# the budget makes the hull actually match the source geometry.  Cost is
# per-contact narrowphase work, roughly O(hull_verts); 1024 still runs
# real-time for this scene because contact pairs are few.
_HULL_BUDGET = 10000  # effectively unlimited (QHull returns the full hull)


def _load_lbm_mesh(
    gltf_path: Path,
    sdf_path: Path | None = None,
    *,
    target_mass: float | None = None,
    maxhullvert: int = _HULL_BUDGET,
) -> newton.Mesh:
    """Load an LBM glTF as a Z-up :class:`newton.Mesh`.

    If ``sdf_path`` is given, the inertial block (mass, COM, inertia) is
    attached to the mesh verbatim.  Passing ``target_mass`` overrides the LBM
    mass and rescales inertia linearly (same shape, different density).

    Textures are left to the viewer (glTF materials reference sibling PNG/KTX2
    files by URI; trimesh resolves those automatically when present).
    """
    if gltf_path in _MESH_CACHE and sdf_path is None and target_mass is None:
        return _MESH_CACHE[gltf_path]

    import trimesh

    if not gltf_path.exists():
        raise FileNotFoundError(f"LBM glTF not found: {gltf_path}")

    raw = trimesh.load(str(gltf_path), process=False, force="mesh")
    verts = _gltf_y_up_to_z_up(np.asarray(raw.vertices, dtype=np.float64)).astype(np.float32)
    faces = np.asarray(raw.faces, dtype=np.int32).flatten()

    uvs = None
    texture = None
    if hasattr(raw.visual, "uv") and raw.visual.uv is not None:
        uvs = np.asarray(raw.visual.uv, dtype=np.float32)
    if uvs is not None and hasattr(raw.visual, "material"):
        embedded = getattr(raw.visual.material, "baseColorTexture", None)
        if embedded is not None:
            texture = np.asarray(embedded)

    mesh = newton.Mesh(
        verts,
        faces,
        uvs=uvs,
        texture=texture,
        compute_inertia=False,
        maxhullvert=maxhullvert,
    )

    if sdf_path is not None:
        lbm_mass, lbm_com, lbm_inertia = _parse_lbm_inertial(sdf_path)
        if target_mass is not None and lbm_mass > 0.0:
            lbm_inertia = lbm_inertia * (target_mass / lbm_mass)
            lbm_mass = target_mass
        mesh.mass = lbm_mass
        mesh.com = wp.vec3(*lbm_com)
        mesh.inertia = wp.mat33(lbm_inertia)

    if sdf_path is None and target_mass is None:
        _MESH_CACHE[gltf_path] = mesh
    return mesh


def _load_object_mesh(name: str) -> tuple[newton.Mesh, float]:
    gltf_path, sdf_path, target_mass, scale, _material = _ASSET_CATALOG[name]
    return _load_lbm_mesh(gltf_path, sdf_path, target_mass=target_mass), scale


def _spawn_object(
    builder: newton.ModelBuilder,
    xform: wp.transform,
    cfg_by_material: dict[str, newton.ModelBuilder.ShapeConfig],
    name: str,
    voxel: float,
) -> int:
    """Spawn a dynamic object with an SDF built on its collision mesh."""
    body = builder.add_body(xform=xform)
    mesh, scale = _load_object_mesh(name)
    if mesh.sdf is None:
        mesh.build_sdf(target_voxel_size=voxel)
    scale_vec = wp.vec3(scale, scale, scale)
    material = _ASSET_CATALOG[name][4]
    builder.add_shape_mesh(body, mesh=mesh, cfg=cfg_by_material[material], scale=scale_vec)
    return body


OBJECT_SPAWNERS: dict[str, Callable[[newton.ModelBuilder, wp.transform, object, float], int]] = {
    name: (lambda b, x, c, v, n=name: _spawn_object(b, x, c, n, v)) for name in _ASSET_CATALOG
}


def _object_sequence() -> list[str]:
    seq: list[str] = []
    for name, count in OBJECT_MIX:
        seq.extend([name] * count)
    return seq


# --- Dish drainer (static LBM rack) -----------------------------------------


def _pitch_quat(angle_rad: float) -> wp.quat:
    """Rotation about +Y by ``angle_rad`` (Drake RPY 'pitch' axis)."""
    return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle_rad)


def _add_drainer(builder: newton.ModelBuilder, cfg, voxel: float) -> None:
    """Static LBM dish rack: three welded parts, all colliding.

    Three glTF parts from the LBM dish drying rack are added to body=-1 with
    the welded transforms encoded in ``sweet_home_dish_drying_rack.dmd.yaml``:

      * ``base``           -- visual + collision (tray slab).
      * ``wireframe``      -- visual + collision.  Collision uses the
        mesh SDF built at ``voxel`` spacing, so the wire cage keeps its
        genuine hollow interior and dishes nest between the wires.
      * ``utensil_holder`` -- visual + collision (small end cup).

    The base sits at world origin (z=0 is its bottom face).  Other parts'
    transforms compound through the welds.
    """
    # Wireframe: base -> nominal_wireframe_origin__z_up (translation + pitch).
    q_wf = _pitch_quat(_WIREFRAME_OFFSET_PITCH_RAD)
    wf_xform = wp.transform(p=_WIREFRAME_OFFSET_P, q=q_wf)

    # Utensil holder: wireframe::origin -> explicit X_PC (translation + pitch).
    q_ut_local = _pitch_quat(_UTENSIL_OFFSET_PITCH_RAD)
    ut_local_xform = wp.transform(p=_UTENSIL_OFFSET_P, q=q_ut_local)
    ut_xform = wp.transform_multiply(wf_xform, ut_local_xform)

    base_mesh = _load_lbm_mesh(_RACK_BASE_GLTF)
    wf_mesh = _load_lbm_mesh(_RACK_WIREFRAME_GLTF)
    ut_mesh = _load_lbm_mesh(_RACK_UTENSIL_GLTF)

    for m in (base_mesh, wf_mesh, ut_mesh):
        if m.sdf is None:
            m.build_sdf(target_voxel_size=voxel)

    unit_scale = wp.vec3(1.0, 1.0, 1.0)
    base_xform = wp.transform_identity()

    # Base: visual + collision.
    builder.add_shape_mesh(
        body=-1,
        xform=base_xform,
        mesh=base_mesh,
        cfg=cfg,
        scale=unit_scale,
    )

    # Wireframe: visual + collision.  Convex hull acts as a filled outer
    # envelope -- dishes rest on top rather than between the wires.
    builder.add_shape_mesh(
        body=-1,
        xform=wf_xform,
        mesh=wf_mesh,
        cfg=cfg,
        scale=unit_scale,
    )

    # Utensil holder: visual + collision.
    builder.add_shape_mesh(
        body=-1,
        xform=ut_xform,
        mesh=ut_mesh,
        cfg=cfg,
        scale=unit_scale,
    )


# --- Model builders ----------------------------------------------------------


def build_template(
    *,
    # MuJoCo Warp has no CCD.  Halo = obj_margin + rack_margin must exceed
    # max per-step displacement (v_max * dt_inner_max) or thin bodies tunnel
    # through the tray.  Objects drop from up to ~0.7 m above the floor and
    # hit at ~3.8 m/s, so dt_max = 5 ms needs a halo of at least 19 mm; we
    # budget 35 mm (15 mm object + 20 mm rack) for headroom.
    obj_margin: float = 0.015,
    rack_margin: float = 0.020,
) -> newton.ModelBuilder:
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(template)

    # Contact stiffness / damping tuned for the shared dt_inner = 2 ms floor
    # (see make_solver / make_solver_fixed).  ke/kd map to MuJoCo solref via
    # convert_solref() in newton/_src/solvers/mujoco/kernels.py:171:
    #   timeconst = 2/kd        -- must stay above 2*dt for stability
    #   dampratio = kd/(2*sqrt(ke))
    # MuJoCo folds the constraint mass matrix in internally, so dampratio is
    # dimensionless and mass-independent here.  With dt_inner=2 ms we need
    # timeconst >= 4 ms, i.e. kd <= 500.  Pick kd=400 (timeconst=5 ms, 2.5x
    # margin).  For critical damping: ke = (kd/2)^2 = 40000.
    _KD = 400.0
    _KE = 40000.0  # kd/(2*sqrt(ke)) = 1.0 -> critically damped contact

    # SDF-SDF contact for both rack and objects: the wireframe rack is
    # genuinely hollow (a wire cage), so convex hulls cannot represent it --
    # objects would either phase through the sides or rest on a filled box.
    # SDF lets MuJoCo Warp query the actual mesh geometry, so dishes nest
    # between the wires like they would on a real rack.  Both sides of the
    # pair must have SDFs configured.  Voxel sizes chosen per feature scale:
    # wire diameter is ~3 mm so the rack needs ~1 mm voxels; bulk objects
    # (mugs, bowls) use 3 mm to keep memory sane.
    _RACK_VOXEL = 0.001   # 1 mm -- resolves wire cage
    _OBJ_VOXEL = 0.003    # 3 mm -- bulk objects

    cfg_rack = newton.ModelBuilder.ShapeConfig(
        ke=_KE,
        kd=_KD,
        mu=0.3,
        margin=rack_margin,
        collision_group=-1,
    )
    cfg_by_material = {
        "ceramic": newton.ModelBuilder.ShapeConfig(
            ke=_KE,
            kd=_KD,
            mu=0.4,
            mu_rolling=5e-3,
            mu_torsional=5e-3,
            margin=obj_margin,
            collision_group=1,
        ),
        "rubber": newton.ModelBuilder.ShapeConfig(
            ke=_KE,
            kd=_KD,
            mu=0.9,
            mu_rolling=1e-2,
            mu_torsional=1e-2,
            margin=obj_margin,
            collision_group=1,
        ),
        "metal": newton.ModelBuilder.ShapeConfig(
            ke=_KE,
            kd=_KD,
            mu=0.25,
            mu_rolling=5e-3,
            mu_torsional=5e-3,
            margin=obj_margin,
            collision_group=1,
        ),
    }
    _add_drainer(template, cfg_rack, voxel=_RACK_VOXEL)

    sequence = _object_sequence()
    for i, name in enumerate(sequence):
        z = DROP_Z_MIN + i * 0.10
        xform = wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity())
        OBJECT_SPAWNERS[name](template, xform, cfg_by_material, _OBJ_VOXEL)

    return template


def build_model(n_worlds: int, **tpl_kwargs) -> newton.Model:
    template = build_template(**tpl_kwargs)
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    ground_cfg = newton.ModelBuilder.ShapeConfig(collision_group=-1)
    builder.add_ground_plane(cfg=ground_cfg)
    return builder.finalize()


def _random_unit_quaternion(rng) -> tuple[float, float, float, float]:
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    a1 = 2.0 * math.pi * u2
    a2 = 2.0 * math.pi * u3
    return (
        s1 * math.sin(a1),
        s1 * math.cos(a1),
        s2 * math.sin(a2),
        s2 * math.cos(a2),
    )


def build_model_randomized(n_worlds: int, seed: int = 42, **tpl_kwargs) -> newton.Model:
    """N worlds with per-world randomized object transforms (fixed base seed)."""
    model = build_model(n_worlds, **tpl_kwargs)

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
        dt_inner_max=0.01,
        dt_mode=dt_mode,
        nconmax=2048,
        njmax=8192,
        cone="elliptic",
        iterations=100,
        impratio=10.0,
        ccd_iterations=100,
    )


# Inner step for the fixed-step baseline.  2 ms stays below solref
# timeconst = 2/kd = 5 ms (kd=400), matching CENIC's dt_inner_max.
FIXED_DT_INNER = 0.002


def make_solver_fixed(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    """Known-good fixed-step MuJoCo Warp solver.

    Same MuJoCo settings as :func:`make_solver` so only the adaptive-step
    wrapper differs -- use this as an oracle when debugging CENIC behavior.
    Caller is responsible for substepping ``step(..., dt=FIXED_DT_INNER)``
    inside each ``DT_OUTER`` control period.
    """
    return newton.solvers.SolverMuJoCo(
        model,
        njmax=8192,
        nconmax=2048,
        cone="elliptic",
        iterations=100,
        impratio=10.0,
        ccd_iterations=100,
        solver="newton",
    )
