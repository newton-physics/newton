# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Any

import numpy as np
import warp as wp

import newton

from ..core.types import override

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
except ImportError:
    Gf = Sdf = Usd = UsdGeom = UsdLux = UsdShade = Vt = None

from .lookdev import _PARAMS, LookdevMode, _coerce_mode, _rotate_z_up_to
from .viewer import _DEFAULT_LAYER_ID, ViewerBase

# Map the USD up-axis string to the lookdev light-rig axis index.
_UP_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

# Lookdev studio-rig authoring (see ViewerUSD ``lookdev=``). The GL key/fill/rim
# intensities feed an ACES tone-map; USD renderers apply their own, so this scale
# maps them into the range Newton's RTX studio rig uses (distant ~3000), folding
# in the per-look exposure. First-pass calibration — tune by eye in Kit / ovrtx.
_LOOKDEV_DISTANT_INTENSITY_SCALE = 750.0
# Dome intensity. Matches Kit's default DomeLight intensity and the ZetCG
# ExhibitionHall reference — the operating point at which the HDR sky is visible
# as the RTX background. The textures hold true scene radiance, so this is the
# normalization/exposure point where the renderer tone-maps it to the GL look
# (not a literal multiplier); at intensity 1 the same textures read as black.
_LOOKDEV_DOME_INTENSITY = 1000.0
# Shadow-catcher floor half-size, as a multiple of the scene AABB diagonal so
# the floor reaches past the scene at any scale (never a world constant). Its
# altitude is the simulation ground level, not a scene-relative offset.
_LOOKDEV_FLOOR_HALF_SIZE_FACTOR = 50.0
# The committed sky gradients under ``assets/`` are Radiance HDR equirects
# holding the scene radiance the GL sky shader feeds to its tone-map:
# ``pow(sky_srgb, 2.2) * exposure`` (sRGB->linear, then exposure), with
# ``sky_upper`` at the zenith and ``sky_lower`` at the nadir/ground (DARK
# ~0.005-0.062, LIGHT ~0.32-0.63). Both modes share the same non-linear ramp (cf.
# the studio prototype): the transition remapped into a sphere-height band with a
# 2x lift (Kit renders the dome darker than GL's ACES) and a 0.3x zenith
# darkening. LIGHT additionally darkens its ground (``sky_lower``) for USD only,
# since Kit compresses its bright gradient toward white; GL is untouched. The
# renderer tone-maps the radiance itself at ``_LOOKDEV_DOME_INTENSITY``, so no
# tone-map is baked in. Re-bake if the sky colors, exposure, lift, band, or the
# LIGHT ground override change.
_LOOKDEV_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")


# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):
    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()

    dir = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)


def _usd_add_xform(prim):
    prim = UsdGeom.Xform(prim)
    prim.ClearXformOpOrder()

    prim.AddTranslateOp()
    prim.AddOrientOp()
    prim.AddScaleOp()


def _usd_set_xform(
    xform,
    pos: tuple | None = None,
    rot: tuple | None = None,
    scale: tuple | None = None,
    time: float = 0.0,
):
    xform = UsdGeom.Xform(xform)

    xform_ops = xform.GetOrderedXformOps()

    if pos is not None:
        xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), time)
    if rot is not None:
        xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])), time)
    if scale is not None:
        xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])), time)


def _lookdev_sky_texture_path(mode: LookdevMode) -> str:
    """Path to the committed sky-gradient equirect (Radiance HDR) for ``mode``."""
    return os.path.join(_LOOKDEV_ASSET_DIR, f"lookdev_sky_{LookdevMode(mode).name.lower()}.hdr")


def _lookdev_up_vector(up_axis: int) -> tuple[float, float, float]:
    v = [0.0, 0.0, 0.0]
    v[up_axis] = 1.0
    return (v[0], v[1], v[2])


def _lookdev_dome_euler(up_axis: int) -> tuple[float, float, float]:
    """``rotateXYZ`` Euler angles [deg] that carry the dome's default +Z pole
    onto the scene up-axis, so a Z-up scene needs no rotation and a Y-up scene
    matches the ZetCG reference's ``rotateXYZ (270, 0, 0)``.

    Authored as a ``rotateXYZ`` op (not a quaternion ``orient`` op): Kit/RTX
    honours ``rotateXYZ`` on a DomeLight but ignores ``orient``, so an orient op
    leaves the sky gradient stuck on its default axis.
    """
    return {0: (0.0, 90.0, 0.0), 1: (-90.0, 0.0, 0.0), 2: (0.0, 0.0, 0.0)}[up_axis]


def _relative_asset_path(src: str, base_dir: str) -> str:
    """``src`` as an explicitly layer-relative USD asset path (forward slashes,
    leading ``./``) anchored at ``base_dir`` — the exported stage's directory.

    A leading ``./`` (or ``../``) is required: USD treats a bare ``foo/bar.png``
    as a search path, not a path relative to the stage, so it would not resolve.
    Falls back to ``src`` when there is no common base (e.g. different Windows
    drives).
    """
    try:
        rel = os.path.relpath(src, base_dir).replace(os.sep, "/")
    except ValueError:
        return src
    return rel if rel.startswith("../") else "./" + rel


def _ground_plane_altitude(model, up_axis: int) -> float | None:
    """World altitude (along ``up_axis``) of the model's ground plane, or ``None``.

    Returns the up-axis position of the first ``GeoType.PLANE`` shape (e.g. from
    ``ModelBuilder.add_ground_plane``) so the shadow-catcher floor can sit on the
    simulation ground rather than float at the scene's lowest geometry.
    """
    types = model.shape_type.numpy()
    idxs = np.where(types == int(newton.GeoType.PLANE))[0]
    if len(idxs) == 0:
        return None
    idx = int(idxs[0])
    local = model.shape_transform.numpy()[idx, :3]
    body = int(model.shape_body.numpy()[idx])
    if body < 0:  # static plane: the shape transform is already world-space
        return float(local[up_axis])
    state = model.state()
    body_q = state.body_q.numpy() if state.body_q is not None else None
    if body_q is None:
        return float(local[up_axis])
    # Horizontal ground planes: body translation + local offset along up-axis.
    return float(body_q[body, up_axis] + local[up_axis])


def _author_lookdev(
    stage,
    root_path: str,
    mode: LookdevMode,
    up_axis: int,
    aabb,
    floor_height: float = 0.0,
    sky_texture: str | None = None,
) -> None:
    """Author the lookdev studio rig into ``stage`` under ``root_path``.

    Writes a ``_Lookdev`` scope with a sky ``DomeLight`` textured with the mode's
    gradient (background + image-based ambient), three ``DistantLight`` key/fill/
    rim lights (only the key casts shadows, matching the GL viewer), and a matte
    shadow-catcher floor sized to ``aabb``. All values come from
    :data:`.lookdev._PARAMS`, so the USD look tracks the GL look. The dome and
    directional lights are scale-invariant; the floor scales with the scene
    diagonal.

    Args:
        stage: The ``Usd.Stage`` to author into.
        root_path: Prim path the rig is parented under (e.g. ``"/root"``).
        mode: The concrete :class:`~newton.viewer.LookdevMode` to author.
        up_axis: Active up-axis index (0=X, 1=Y, 2=Z).
        aabb: ``(min, max)`` scene bounds (length-3, model space), or ``None`` to
            skip the floor (no boundable geometry).
        floor_height: World altitude (along ``up_axis``) for the shadow-catcher
            floor — the simulation ground level, or 0.
        sky_texture: Asset path for the dome's gradient texture. Defaults to the
            committed absolute path; callers should pass a path relative to the
            stage so it resolves in Kit / ovrtx.
    """
    params = _PARAMS[mode]
    exposure = float(params["exposure"])
    if sky_texture is None:
        sky_texture = _lookdev_sky_texture_path(mode)

    scope_path = f"{root_path}/_Lookdev"
    UsdGeom.Scope.Define(stage, scope_path)

    # Sky dome: background gradient + image-based ambient. Author the pole
    # rotation as a rotateXYZ (Euler) op on the DomeLight, matching the ZetCG
    # reference that Kit renders correctly -- Kit/RTX honours rotateXYZ on a dome
    # but ignores a quaternion orient op (which leaves the gradient stuck on its
    # default horizontal axis). See :func:`_lookdev_dome_euler`.
    dome = UsdLux.DomeLight.Define(stage, f"{scope_path}/Sky")
    dome_x = UsdGeom.Xformable(dome)
    dome_x.ClearXformOpOrder()
    dome_x.AddRotateXYZOp().Set(Gf.Vec3f(*_lookdev_dome_euler(up_axis)))
    tex = dome.CreateTextureFileAttr()
    tex.Set(Sdf.AssetPath(sky_texture))
    # HDR holds linear radiance, so tag it raw (no sRGB decode). ``automatic``
    # lets the renderer detect the equirect projection, matching how a real
    # environment dome is authored.
    tex.SetColorSpace("raw")
    dome.CreateTextureFormatAttr().Set(UsdLux.Tokens.automatic)
    dome.CreateIntensityAttr().Set(_LOOKDEV_DOME_INTENSITY)
    # Make the dome camera-visible so RTX renders it as the viewport background.
    # A scene dome is the default background source, so nothing else is needed;
    # set explicitly because the RTX UI has defaulted this to false in some
    # versions.
    dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, custom=True).Set(True)

    # Three-point lights (key/fill/rim). Shadows are left entirely at the
    # renderer default -- no ShadowAPI and no soft-shadow angle -- so Kit does
    # not apply inconsistent per-light shadow overrides. A DistantLight emits
    # along -Z; the lookdev direction points toward the source, so aim local +Z
    # along it (light travels the opposite way).
    for name, (direction, color, intensity) in zip(("Key", "Fill", "Rim"), params["lights"], strict=True):
        d = _rotate_z_up_to(up_axis, direction)
        n = max((d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5, 1e-12)
        d = (d[0] / n, d[1] / n, d[2] / n)
        light = UsdLux.DistantLight.Define(stage, f"{scope_path}/{name}")
        light_x = UsdGeom.Xformable(light)
        light_x.ClearXformOpOrder()
        light_x.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), Gf.Vec3d(*d)).GetQuat()))
        light.CreateColorAttr().Set(Gf.Vec3f(*color))
        light.CreateIntensityAttr().Set(float(intensity) * exposure * _LOOKDEV_DISTANT_INTENSITY_SCALE)

    if aabb is not None:
        _author_lookdev_floor(stage, scope_path, mode, up_axis, aabb, floor_height)


def _author_lookdev_floor(stage, scope_path: str, mode: LookdevMode, up_axis: int, aabb, floor_height: float) -> None:
    """Author the matte shadow-catcher floor quad + material (see :func:`_author_lookdev`)."""
    lo, hi = np.asarray(aabb[0], dtype=np.float64), np.asarray(aabb[1], dtype=np.float64)
    diagonal = float(np.linalg.norm(hi - lo)) or 1.0
    half = _LOOKDEV_FLOOR_HALF_SIZE_FACTOR * diagonal
    height = float(floor_height)  # simulation ground level, so objects rest on it
    center = 0.5 * (lo + hi)

    plane_axes = [a for a in range(3) if a != up_axis]
    corners = []
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = [0.0, 0.0, 0.0]
        p[up_axis] = height
        p[plane_axes[0]] = float(center[plane_axes[0]]) + su * half
        p[plane_axes[1]] = float(center[plane_axes[1]]) + sv * half
        corners.append(Gf.Vec3f(*p))
    normal = Gf.Vec3f(*_lookdev_up_vector(up_axis))

    mesh = UsdGeom.Mesh.Define(stage, f"{scope_path}/Floor")
    mesh.CreatePointsAttr(corners)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([normal, normal, normal, normal])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)

    # Flag the floor as an Omniverse RTX "matte object" (shadow catcher): in
    # Kit / ovrtx it then renders invisibly — the sky/dome shows through — while
    # still catching the scene's shadows, grounding the objects on the gradient
    # background instead of an opaque plate. Requires the renderer's Matte Object
    # (and, for shadows only, Shadow Catcher) post-process to be enabled;
    # non-RTX Hydra delegates (usdview/Storm) ignore it and draw the plane using
    # the bound material below.
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("isMatteObject", Sdf.ValueTypeNames.Bool).Set(True)

    mat_path = f"{scope_path}/FloorMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    surface = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_PARAMS[mode]["ground_color"]))
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(_PARAMS[mode]["ground_roughness"]))
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh).Bind(material)


class ViewerUSD(ViewerBase):
    """
    USD viewer backend for Newton physics simulations.

    This backend creates a USD stage and manages mesh prototypes and instanced rendering
    using PointInstancers. It supports time-sampled transforms for efficient playback
    and visualization of simulation data.
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 60,
        up_axis: str = "Z",
        num_frames: int | None = 100,
        scaling: float = 1.0,
        points_as_spheres: bool = True,
        lookdev: LookdevMode | None = None,
    ):
        """
        Initialize the USD viewer backend for Newton physics simulations.

        Args:
            output_path: Path to the output USD file.
            fps: Frames per second for time sampling. Default is 60.
            up_axis: USD up axis, either 'Y' or 'Z'. Default is 'Z'.
            num_frames: Maximum number of frames to record. Default is 100. If None, recording is unlimited.
            scaling: Uniform scaling applied to the scene root. Default is 1.0.
            points_as_spheres: When True, :meth:`log_points` renders points as a
                :class:`~pxr.UsdGeom.PointInstancer` of :class:`~pxr.UsdGeom.Sphere`
                prototypes scaled by ``radii``, instead of flat
                :class:`~pxr.UsdGeom.Points` splats. Default is True.
            lookdev: Built-in :class:`~newton.viewer.LookdevMode` to author into
                the stage as a portable studio rig (textured sky dome, three
                directional lights, matte shadow-catcher floor) approximating
                the GL viewer's look, or ``None`` (the default) to author no
                lighting/environment.

        Raises:
            ImportError: If the usd-core package is not installed.
        """
        if Usd is None:
            raise ImportError("usd-core package is required for ViewerUSD. Install with: pip install usd-core")

        super().__init__()

        self.output_path = os.path.abspath(output_path)
        self.fps = fps
        self.up_axis = up_axis
        self.scaling = scaling
        self.num_frames = num_frames
        self.points_as_spheres = points_as_spheres
        self._lookdev_mode = _coerce_mode(lookdev)

        # Create USD stage. If this output path is already registered in the
        # current process, reuse and clear the existing layer instead of
        # calling CreateNew() again (which raises for duplicate identifiers).
        existing_layer = Sdf.Layer.Find(self.output_path)
        if existing_layer is not None:
            existing_layer.Clear()
            self.stage = Usd.Stage.Open(existing_layer)
        else:
            self.stage = Usd.Stage.CreateNew(self.output_path)
        self.stage.SetTimeCodesPerSecond(fps)  # number of timeCodes per second for data storage
        self.stage.SetFramesPerSecond(fps)  # display frame rate (timeline FPS in DCC tools)
        self.stage.SetStartTimeCode(0)

        axis_token = {
            "X": UsdGeom.Tokens.x,
            "Y": UsdGeom.Tokens.y,
            "Z": UsdGeom.Tokens.z,
        }.get(self.up_axis.strip().upper())

        UsdGeom.SetStageUpAxis(self.stage, axis_token)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)

        self.root = UsdGeom.Xform.Define(self.stage, "/root")

        # apply root scaling
        self.root.ClearXformOpOrder()
        s = self.root.AddScaleOp()
        s.Set(Gf.Vec3d(float(scaling), float(scaling), float(scaling)), 0.0)

        self.stage.SetDefaultPrim(self.root.GetPrim())

        # Track current frame
        self._frame_index = 0
        self._frame_count = 0

        self.set_model(None)

    @override
    def set_model(self, model: newton.Model | None):
        super().set_model(model)
        # Author the studio rig once the model (and thus its bounds/up-axis) is
        # known. set_model clears the stage on a model swap, so re-author here.
        if self._lookdev_mode is not None and model is not None:
            up_axis = _UP_AXIS_INDEX.get(self.up_axis.strip().upper(), 2)
            ground = _ground_plane_altitude(model, up_axis)
            _author_lookdev(
                self.stage,
                "/root",
                self._lookdev_mode,
                up_axis,
                self._get_world_bounds(),
                floor_height=ground if ground is not None else 0.0,
                sky_texture=_relative_asset_path(
                    _lookdev_sky_texture_path(self._lookdev_mode), os.path.dirname(self.output_path)
                ),
            )

    @override
    def _should_show_shape(self, flags: int, is_static: bool, geo_type: int | None = None) -> bool:
        # The lookdev shadow-catcher floor replaces the ``GeoType.PLANE``
        # collider from ``ModelBuilder.add_ground_plane`` — hide that plane while
        # lookdev is active (mirrors ViewerGL) so the export has a single ground.
        if geo_type == newton.GeoType.PLANE and self._lookdev_mode is not None and not self.show_collision:
            return False
        return super()._should_show_shape(flags, is_static, geo_type=geo_type)

    @override
    def _init_extra_layer_state(self, layer) -> None:
        super()._init_extra_layer_state(layer)
        layer._meshes = {}  # mesh_name -> prototype path
        layer._instance_groups = {}  # instance_name -> group prim for individually referenced meshes
        layer._instancers = {}  # instancer_name -> UsdGeom.PointInstancer
        layer._points = {}  # point_name -> UsdGeom.Points
        layer._texture_materials: dict[str, Any] = {}  # mesh_name -> UsdShade.Material

    def _reset_stage(self):
        self.stage.GetRootLayer().Clear()
        self.stage.SetTimeCodesPerSecond(self.fps)
        self.stage.SetFramesPerSecond(self.fps)
        self.stage.SetStartTimeCode(0)
        axis_token = {
            "X": UsdGeom.Tokens.x,
            "Y": UsdGeom.Tokens.y,
            "Z": UsdGeom.Tokens.z,
        }.get(self.up_axis.strip().upper())
        UsdGeom.SetStageUpAxis(self.stage, axis_token)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.root = UsdGeom.Xform.Define(self.stage, "/root")
        self.root.ClearXformOpOrder()
        s = self.root.AddScaleOp()
        s.Set(Gf.Vec3d(float(self.scaling), float(self.scaling), float(self.scaling)), 0.0)
        self.stage.SetDefaultPrim(self.root.GetPrim())
        self._frame_index = 0
        self._frame_count = 0

    @staticmethod
    def _save_texture_atomic(tex_array: np.ndarray, tex_path: str) -> None:
        """Write a texture image atomically so readers never see a partial PNG."""
        from PIL import Image

        tex_dir = os.path.dirname(tex_path)
        base_name = os.path.basename(tex_path)
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=f".{base_name}.", suffix=".tmp", dir=tex_dir)
            os.close(fd)
            Image.fromarray(tex_array).save(tmp_path, format="PNG")
            os.replace(tmp_path, tex_path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _remove_active_layer_prims(self):
        names = set(self._meshes) | set(self._instance_groups) | set(self._instancers) | set(self._points)
        for name in sorted(names, key=lambda item: self._get_path(item).count("/"), reverse=True):
            if self._is_layer_owned_path(name):
                self.stage.RemovePrim(self._get_path(name))

        for mesh_name in list(self._texture_materials):
            if self._is_layer_owned_path(mesh_name):
                self.stage.RemovePrim(self._texture_material_path(mesh_name))

    def _has_user_layers(self) -> bool:
        return any(layer_id != _DEFAULT_LAYER_ID for layer_id in self._layers)

    def clear_model(self):
        if hasattr(self, "stage") and self.stage is not None:
            if self._active_layer_id == _DEFAULT_LAYER_ID and not self._has_user_layers():
                self._reset_stage()
            else:
                self._remove_active_layer_prims()

        super().clear_model()

    @override
    def begin_frame(self, time: float):
        """
        Begin a new frame at the given simulation time.

        Args:
            time: The simulation time for the new frame.
        """
        super().begin_frame(time)
        self._frame_index = int(time * self.fps)
        self._frame_count += 1

        # Update stage end time if needed
        if self._frame_index > self.stage.GetEndTimeCode():
            self.stage.SetEndTimeCode(self._frame_index)

    @override
    def end_frame(self):
        """
        End the current frame.

        This method is a placeholder for any end-of-frame logic required by the backend.
        """
        pass

    @override
    def is_running(self):
        """
        Check if the viewer is still running.

        Returns:
            bool: False if the frame limit is exceeded, True otherwise.
        """
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    @override
    def close(self):
        """
        Finalize and save the USD stage.

        This should be called when all logging is complete to ensure the USD file is written.
        """
        self.stage.GetRootLayer().Save()
        self.stage = None

        if self.output_path:
            print(f"USD output saved in: {os.path.abspath(self.output_path)}")

    def _get_path(self, name):
        # Handle both absolute and relative paths correctly
        if name.startswith("/"):
            return "/root" + name
        else:
            return "/root/" + name

    @override
    def log_mesh(
        self,
        name: str,
        points: wp.array[wp.vec3],
        indices: wp.array[wp.int32] | wp.array[wp.uint32],
        normals: wp.array[wp.vec3] | None = None,
        uvs: wp.array[wp.vec2] | None = None,
        texture: np.ndarray | str | None = None,
        hidden: bool = False,
        backface_culling: bool = True,
        color: tuple[float, float, float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ):
        """
        Create a USD mesh prototype from vertex and index data.

        Args:
            name: Mesh name or Sdf.Path string.
            points: Vertex positions as a warp array of wp.vec3.
            indices: Triangle indices as a warp array of wp.uint32.
            normals: Vertex normals as a warp array of wp.vec3.
            uvs: UV coordinates as a warp array of wp.vec2.
            texture: Optional texture path/URL or image array.
            hidden: If True, mesh will be hidden.
            backface_culling: If True, enable backface culling.
            color: Optional base color as an RGB tuple with values in
                [0, 1]. Used when no texture is provided.
            roughness: Surface roughness in ``[0, 1]``. ``0`` is perfectly
                smooth, ``1`` is fully rough.
            metallic: Metallicity in ``[0, 1]``. ``0`` is dielectric, ``1``
                is metal.
        """

        name = self._qualify(name)

        # Convert warp arrays to numpy
        points_np = points.numpy().astype(np.float32)
        indices_np = indices.numpy().astype(np.uint32)

        if name not in self._meshes:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            mesh_prim = UsdGeom.Mesh.Define(self.stage, self._get_path(name))

            # setup topology once (do not set every frame)
            face_vertex_counts = [3] * (len(indices_np) // 3)
            mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.GetFaceVertexIndicesAttr().Set(indices_np)

            # Store the prototype path
            self._meshes[name] = mesh_prim

        mesh_prim = self._meshes[name]
        mesh_prim.GetPointsAttr().Set(points_np, self._frame_index)

        # Set normals if provided
        if normals is not None:
            normals_np = normals.numpy().astype(np.float32)
            mesh_prim.GetNormalsAttr().Set(normals_np, self._frame_index)
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set UVs if provided
        if uvs is not None:
            uvs_np = uvs.numpy().astype(np.float32) if isinstance(uvs, wp.array) else np.asarray(uvs, dtype=np.float32)
            pv_api = UsdGeom.PrimvarsAPI(mesh_prim)
            st_pv = pv_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st_pv.Set(uvs_np)

        # Create and bind a textured material only when both texture and UVs are
        # provided — a UsdUVTexture shader with no "st" primvar would sample
        # undefined data and produce incorrect results.
        if texture is not None and uvs is not None and name not in self._texture_materials:
            self._create_texture_material(name, mesh_prim, texture)

        # how to hide the prototype mesh but not the instances in USD?
        mesh_prim.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    def _create_texture_material(self, mesh_name: str, mesh_prim, texture):
        """Create a UsdPreviewSurface material with a diffuse texture and bind it to *mesh_prim*."""
        from pxr import Sdf as _Sdf
        from pxr import UsdShade

        from ..utils.texture import load_texture  # noqa: PLC0415

        # Resolve texture to a file path on disk
        if isinstance(texture, str):
            tex_path = os.path.abspath(texture)
        else:
            tex_array = load_texture(texture)
            if tex_array is None:
                return
            tex_dir = os.path.dirname(self.output_path)
            safe_name = mesh_name.replace("/", "_").replace("\\", "_")
            tex_path = os.path.join(tex_dir, f"_tex_{safe_name}.png")
            try:
                self._save_texture_atomic(tex_array, tex_path)
            except Exception as exc:
                warnings.warn(
                    f"ViewerUSD: failed to export texture for mesh '{mesh_name}': {exc}. "
                    "Mesh will render without texture.",
                    stacklevel=2,
                )
                return

        mat_path = self._texture_material_path(mesh_name)
        self._ensure_scopes_for_path(self.stage, mat_path)

        material = UsdShade.Material.Define(self.stage, mat_path)
        surface = UsdShade.Shader.Define(self.stage, f"{mat_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        diff_input = surface.CreateInput("diffuseColor", _Sdf.ValueTypeNames.Color3f)
        surface.CreateInput("roughness", _Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

        tex_reader = UsdShade.Shader.Define(self.stage, f"{mat_path}/DiffuseTexture")
        tex_reader.CreateIdAttr("UsdUVTexture")
        tex_reader.CreateInput("file", _Sdf.ValueTypeNames.Asset).Set(tex_path)
        tex_reader.CreateInput("sourceColorSpace", _Sdf.ValueTypeNames.Token).Set("auto")
        tex_reader.CreateInput("wrapS", _Sdf.ValueTypeNames.Token).Set("repeat")
        tex_reader.CreateInput("wrapT", _Sdf.ValueTypeNames.Token).Set("repeat")
        tex_reader.CreateOutput("rgb", _Sdf.ValueTypeNames.Float3)
        diff_input.ConnectToSource(tex_reader.ConnectableAPI(), "rgb")

        st_reader = UsdShade.Shader.Define(self.stage, f"{mat_path}/PrimvarSt")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", _Sdf.ValueTypeNames.Token).Set("st")
        st_reader.CreateOutput("result", _Sdf.ValueTypeNames.Float2)
        tex_reader.CreateInput("st", _Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")

        UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
        UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(material)
        self._texture_materials[mesh_name] = material

    @staticmethod
    def _texture_material_path(mesh_name: str) -> str:
        safe = mesh_name.replace("/", "_").lstrip("_")
        return f"/root/Materials/mat_{safe}"

    # log a set of instances as individual mesh prims, slower but makes it easier
    # to do post-editing of instance materials etc. default for Newton shapes
    @override
    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms: wp.array[wp.transform] | None,
        scales: wp.array[wp.vec3] | None,
        colors: wp.array[wp.vec3] | None,
        materials: wp.array[wp.vec4] | None,
        hidden: bool = False,
    ):
        """
        Log a batch of mesh instances for rendering.

        Args:
            name: Unique name for the instancer.
            mesh: Name of the base mesh.
            xforms: Array of transforms.
            scales: Array of scales.
            colors: Array of colors.
            materials: Array of materials.
            hidden: Whether the instances are hidden.
        """
        name = self._qualify(name)
        mesh = self._qualify(mesh)

        # Get prototype path
        if mesh not in self._meshes:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        self._ensure_scopes_for_path(self.stage, self._get_path(name) + "/scope")
        group_prim = self.stage.GetPrimAtPath(self._get_path(name))
        if group_prim:
            self._instance_groups[name] = group_prim
            UsdGeom.Imageable(group_prim).GetVisibilityAttr().Set(
                "inherited" if not hidden else "invisible", self._frame_index
            )

        if xforms is not None:
            xforms = xforms.numpy()
        else:
            xforms = np.empty((0, 7), dtype=np.float32)

        if scales is not None:
            scales = scales.numpy()
        else:
            scales = np.ones((len(xforms), 3), dtype=np.float32)

        if colors is not None:
            colors = colors.numpy()

        for i in range(len(xforms)):
            instance_path = self._get_path(name) + f"/instance_{i}"
            instance = self.stage.GetPrimAtPath(instance_path)

            if not instance:
                instance = self.stage.DefinePrim(instance_path)
                instance.GetReferences().AddInternalReference(self._get_path(mesh))

                _usd_add_xform(instance)

            UsdGeom.Imageable(instance).GetVisibilityAttr().Set(
                "inherited" if not hidden else "invisible", self._frame_index
            )

            # update transform
            if xforms is not None:
                pos = xforms[i][:3]
                rot = xforms[i][3:7]

                _usd_set_xform(instance, pos, rot, scales[i], self._frame_index)

            # update color
            if colors is not None:
                displayColor = UsdGeom.PrimvarsAPI(instance).GetPrimvar("displayColor")
                displayColor.Set(colors[i], self._frame_index)

    # log a set of instances as a point instancer, faster but less flexible
    def log_instances_point_instancer(
        self,
        name: str,
        mesh: str,
        xforms: wp.array[wp.transform] | None,
        scales: wp.array[wp.vec3] | np.ndarray | None,
        colors: (
            wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | np.ndarray | None
        ),
        materials: wp.array[wp.vec4] | None,
    ):
        """
        Create or update a PointInstancer for mesh instances.

        Args:
            name: Instancer name or Sdf.Path string.
            mesh: Mesh prototype name (must be previously logged).
            xforms: Instance transforms as a warp array of wp.transform.
            scales: Instance scales as a warp array of wp.vec3.
            colors: Instance colors as a warp array of wp.vec3.
            materials: Instance materials as a warp array of wp.vec4.

        Raises:
            RuntimeError: If the mesh prototype is not found.
        """
        name = self._qualify(name)
        mesh = self._qualify(mesh)

        # Get prototype path
        if mesh not in self._meshes:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        num_instances = len(xforms)

        # Create instancer if it doesn't exist
        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            instancer = UsdGeom.PointInstancer.Define(self.stage, self._get_path(name))
            instancer.CreateIdsAttr().Set(list(range(num_instances)))
            instancer.CreateProtoIndicesAttr().Set([0] * num_instances)
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            # Set the prototype relationship
            instancer.GetPrototypesRel().AddTarget(self._get_path(mesh))

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        # Convert transforms to USD format
        if xforms is not None:
            xforms_np = xforms.numpy()

            # Extract positions from warp transforms using vectorized operations
            # Warp transform format: [x, y, z, qx, qy, qz, qw]
            positions = xforms_np[:, :3].astype(np.float32)

            # Convert quaternion format: Warp (x, y, z, w) → USD (w, (x,y,z))
            # USD expects quaternions as Gf.Quath(real, imag_vec3)
            quat_w = xforms_np[:, 6].astype(np.float32)
            quat_xyz = xforms_np[:, 3:6].astype(np.float32)

            # Create orientations list with proper USD quaternion format
            orientations = []
            for i in range(num_instances):
                quat = Gf.Quath(
                    float(quat_w[i]), Gf.Vec3h(float(quat_xyz[i, 0]), float(quat_xyz[i, 1]), float(quat_xyz[i, 2]))
                )
                orientations.append(quat)

            # Handle scales with numpy operations
            if scales is None:
                scales = np.ones((num_instances, 3), dtype=np.float32)
            elif isinstance(scales, wp.array):
                scales = scales.numpy().astype(np.float32)

            # Set attributes at current time
            instancer.GetPositionsAttr().Set(positions, self._frame_index)
            instancer.GetOrientationsAttr().Set(orientations, self._frame_index)

            if scales is not None:
                instancer.GetScalesAttr().Set(scales, self._frame_index)

            if colors is not None:
                # Promote colors to proper numpy array format
                colors_np = self._promote_colors_to_array(colors, num_instances)

                # Set color per-instance
                displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                displayColor.Set(colors_np, self._frame_index)

                # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                indices = Vt.IntArray(range(num_instances))
                displayColor.SetIndices(indices, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited", self._frame_index)

    # Abstract methods that need basic implementations
    @override
    def log_lines(
        self,
        name: str,
        starts: wp.array[wp.vec3] | None,
        ends: wp.array[wp.vec3] | None,
        colors: (wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None),
        width: float = 0.01,
        hidden: bool = False,
    ):
        """Debug helper to add a line list as a set of capsules

        Args:
            name: Unique name for the line batch.
            starts: The vertices of the lines (wp.array)
            ends: The vertices of the lines (wp.array)
            colors: The colors of the lines (wp.array)
            width: The width of the lines.
            hidden: Whether the lines are hidden.
        """

        name = self._qualify(name)

        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, self._get_path(name))

            instancer = UsdGeom.PointInstancer.Define(self.stage, self._get_path(name))

            # define nested capsule prim
            instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
            instancer_capsule.GetRadiusAttr().Set(width)

            instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        if starts is not None and ends is not None:
            num_lines = int(len(starts))
            if num_lines > 0:
                # bring to host
                starts = starts.numpy()
                ends = ends.numpy()

                line_positions = []
                line_rotations = []
                line_scales = []

                for i in range(num_lines):
                    pos0 = starts[i]
                    pos1 = ends[i]

                    (pos, rot, scale) = _compute_segment_xform(
                        Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])),
                        Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])),
                    )

                    line_positions.append(pos)
                    line_rotations.append(rot)
                    line_scales.append(scale)

                instancer.GetPositionsAttr().Set(line_positions, self._frame_index)
                instancer.GetOrientationsAttr().Set(line_rotations, self._frame_index)
                instancer.GetScalesAttr().Set(line_scales, self._frame_index)
                instancer.GetProtoIndicesAttr().Set([0] * num_lines, self._frame_index)
                instancer.CreateIdsAttr().Set(list(range(num_lines)))

                if colors is not None:
                    # Promote colors to proper numpy array format
                    colors_np = self._promote_colors_to_array(colors, num_lines)

                    # Set color per-instance
                    displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                    displayColor.Set(colors_np, self._frame_index)

                    # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                    indices = Vt.IntArray(range(num_lines))
                    displayColor.SetIndices(indices, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    @override
    def log_points(
        self,
        name: str,
        points: wp.array[wp.vec3] | None,
        radii: wp.array[wp.float32] | float | None = None,
        colors: (wp.array[wp.vec3] | wp.array[wp.float32] | tuple[float, float, float] | list[float] | None) = None,
        hidden: bool = False,
    ):
        """Log points as a USD primitive.

        By default, each point is an instance of a ``UsdGeom.Sphere`` prototype
        under a ``UsdGeom.PointInstancer``. Set ``points_as_spheres=False`` to
        write flat ``UsdGeom.Points`` splats instead.

        Args:
            name: Unique name for the point primitive.
            points: Point positions.
            radii: Point radii or a single shared radius.
            colors: Optional per-point colors or a shared RGB triplet.
            hidden: Whether the point primitive is hidden.

        Returns:
            ``Sdf.Path`` of the created/updated primitive.
        """
        name = self._qualify(name)

        if points is None:
            path = self._get_path(name)
            if self.points_as_spheres:
                instancer = UsdGeom.PointInstancer.Get(self.stage, path)
            else:
                instancer = UsdGeom.Points.Get(self.stage, path)
            if instancer:
                instancer.GetVisibilityAttr().Set("invisible", self._frame_index)
                return instancer.GetPath()
            return

        num_points = len(points)

        if radii is None:
            radii = 0.1

        if np.isscalar(radii):
            radius_interp = "constant"
        else:
            radius_interp = "vertex"

        colors, color_interp = self._normalize_point_colors(colors, num_points)

        path = self._get_path(name)

        if self.points_as_spheres:
            if name not in self._instancers:
                self._ensure_scopes_for_path(self.stage, path)
                instancer = UsdGeom.PointInstancer.Define(self.stage, path)
                sphere = UsdGeom.Sphere.Define(self.stage, instancer.GetPath().AppendChild("sphere"))
                sphere.GetRadiusAttr().Set(1.0)
                instancer.CreatePrototypesRel().SetTargets((sphere.GetPath(),))
                if colors is not None:
                    UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                        "displayColor", Sdf.ValueTypeNames.Color3fArray, color_interp, 1
                    )
                self._instancers[name] = instancer

            instancer = self._instancers[name]

            # Proto indices must be updated every frame: particle count can vary due to stream compaction.
            instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * num_points), self._frame_index)
            instancer.GetPositionsAttr().Set(points.numpy(), self._frame_index)

            # PointInstancer scales are vec3; broadcast scalar or per-point radius to (N, 3).
            if np.isscalar(radii):
                scales = np.full((num_points, 3), radii, dtype=np.float32)
            else:
                r = radii.numpy() if isinstance(radii, wp.array) else np.array(radii, dtype=np.float32)
                scales = np.stack([r, r, r], axis=1)
            instancer.GetScalesAttr().Set(scales, self._frame_index)

            if colors is not None:
                primvar = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                if not primvar:
                    primvar = UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                        "displayColor", Sdf.ValueTypeNames.Color3fArray, color_interp, 1
                    )
                primvar.Set(colors, self._frame_index)

            instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)
            return instancer.GetPath()

        instancer = UsdGeom.Points.Get(self.stage, path)
        if not instancer:
            self._ensure_scopes_for_path(self.stage, path)
            instancer = UsdGeom.Points.Define(self.stage, path)
            self._points[name] = instancer

            UsdGeom.Primvar(instancer.GetWidthsAttr()).SetInterpolation(radius_interp)
            UsdGeom.Primvar(instancer.GetDisplayColorAttr()).SetInterpolation(color_interp)
        else:
            self._points[name] = instancer

        instancer.GetPointsAttr().Set(points.numpy(), self._frame_index)

        # convert radii to widths for USD
        if np.isscalar(radii):
            widths = (radii * 2.0,)
        elif isinstance(radii, wp.array):
            widths = radii.numpy() * 2.0
        else:
            widths = np.array(radii) * 2.0

        instancer.GetWidthsAttr().Set(widths, self._frame_index)

        if colors is not None:
            instancer.GetDisplayColorAttr().Set(colors, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)
        return instancer.GetPath()

    @override
    def log_array(self, name: str, array: wp.array[Any] | np.ndarray):
        """
        Log array data (not implemented for USD backend).

        This method is a placeholder and does not log array data in the USD backend.

        Args:
            name: Unique path/name for the array signal.
            array: Array data to visualize.
        """
        pass

    @override
    def log_scalar(self, name: str, value: int | float | bool | np.number, *, clear: bool = False, smoothing: int = 1):
        """
        Log scalar value (not implemented for USD backend).

        This method is a placeholder and does not log scalar values in the USD backend.

        Args:
            name: Unique path/name for the scalar signal.
            value: Scalar value to visualize.
            clear: Ignored by this backend.
            smoothing: Ignored by this backend.
        """
        pass

    @override
    def apply_forces(self, state: newton.State):
        """USD backend does not apply interactive forces.

        Args:
            state: Current simulation state.
        """
        pass

    def _promote_colors_to_array(self, colors, num_items):
        """
        Helper method to promote colors to a numpy array format.

        Parameters:
            colors: Input colors in various formats (wp.array, list/tuple, np.ndarray)
            num_items: Number of items that need colors

        Returns:
            np.ndarray: Colors as numpy array with shape (num_items, 3)
        """
        if colors is None:
            return None

        if isinstance(colors, wp.array):
            # Convert warp array to numpy
            return colors.numpy()
        elif isinstance(colors, list | tuple) and len(colors) == 3 and all(np.isscalar(x) for x in colors):
            # Single color (list/tuple of 3 floats) - promote to array with one value per item
            return np.tile(colors, (num_items, 1))
        elif isinstance(colors, np.ndarray):
            # Already numpy array - pass through
            return colors
        else:
            # Fallback for other formats
            return np.array(colors)

    @staticmethod
    def _is_single_rgb_triplet(colors) -> bool:
        """Returns True when colors represent one RGB triplet."""
        if isinstance(colors, np.ndarray):
            return colors.ndim == 1 and colors.shape[0] == 3

        if isinstance(colors, list | tuple):
            return len(colors) == 3 and all(np.isscalar(x) for x in colors)

        return False

    def _normalize_point_colors(self, colors, num_points):
        """Normalize point colors and return (values, interpolation token)."""
        if colors is None:
            return None, UsdGeom.Tokens.constant

        if isinstance(colors, wp.array):
            colors = colors.numpy()

        if self._is_single_rgb_triplet(colors):
            colors_arr = np.asarray(colors, dtype=np.float32)
            return colors_arr.reshape(1, 3), UsdGeom.Tokens.constant

        if isinstance(colors, np.ndarray):
            return colors, UsdGeom.Tokens.vertex

        if isinstance(colors, list | tuple):
            # Keep list/tuple inputs as-is for existing valid per-point color inputs.
            if len(colors) == num_points:
                return colors, UsdGeom.Tokens.vertex
            return np.asarray(colors), UsdGeom.Tokens.vertex

        return np.asarray(colors), UsdGeom.Tokens.vertex

    @staticmethod
    def _ensure_scopes_for_path(stage: Usd.Stage, prim_path_str: str):
        """
        Ensure that all parent prims in the hierarchy exist as 'Scope' prims.

        If a prim does not exist at the given path, this method creates all
        non-existent parent prims in its hierarchy as 'Scope' prims. This is
        useful for ensuring a valid hierarchy before defining a prim.

        Parameters:
            stage: The USD stage to operate on.
            prim_path_str: The Sdf.Path string for the target prim.
        """
        # Convert the string to an Sdf.Path object for robust manipulation
        prim_path = Sdf.Path(prim_path_str)

        # First, check if the target prim already exists.
        if stage.GetPrimAtPath(prim_path):
            return

        # We only want to create the parent hierarchy, not the final prim itself.
        parent_path = prim_path.GetParentPath()

        # GetPrefixes() provides a convenient list of all ancestor paths.
        # For "/A/B/C", it returns ["/", "/A", "/A/B"].
        for path in parent_path.GetPrefixes():
            # The absolute root path ('/') always exists, so we can skip it.
            if path == Sdf.Path.absoluteRootPath:
                continue

            # Check if a prim exists at the current ancestor path.
            if not stage.GetPrimAtPath(path):
                stage.DefinePrim(path, "Scope")
