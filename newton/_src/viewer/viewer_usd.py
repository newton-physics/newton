# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Any

import numpy as np
import warp as wp

import newton

from ..core.types import override

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
except ImportError:
    Gf = Sdf = Usd = UsdGeom = Vt = None

from .utils import promote_to_clamped_float_array
from .viewer import ViewerBase


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
    ):
        """
        Initialize the USD viewer backend for Newton physics simulations.

        Args:
            output_path: Path to the output USD file.
            fps: Frames per second for time sampling. Default is 60.
            up_axis: USD up axis, either 'Y' or 'Z'. Default is 'Z'.
            num_frames: Maximum number of frames to record. Default is 100. If None, recording is unlimited.
            scaling: Uniform scaling applied to the scene root. Default is 1.0.

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

        # Track meshes and instancers
        self._meshes = {}  # mesh_name -> prototype_path
        self._instancers = {}  # instancer_name -> UsdGeomPointInstancer
        self._points = {}  # point_name -> UsdGeomPoints
        self._texture_materials: dict[str, Any] = {}  # mesh_name -> UsdShade.Material
        self._preview_materials: dict[tuple, Any] = {}  # material key -> UsdShade.Material
        self._texture_paths: dict[str, str] = {}
        self._mesh_appearance: dict[str, dict[str, Any]] = {}
        self._instance_appearance: dict[str, dict[str, np.ndarray]] = {}

        # Track current frame
        self._frame_index = 0
        self._frame_count = 0

        self.set_model(None)

    def clear_model(self):
        if hasattr(self, "stage") and self.stage is not None:
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
            self._meshes = {}
            self._instancers = {}
            self._points = {}
            self._texture_materials = {}
            self._preview_materials = {}
            self._texture_paths = {}
            self._mesh_appearance = {}
            self._instance_appearance = {}
            self._frame_index = 0
            self._frame_count = 0

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
        opacity: float | None = None,
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
            opacity: Optional display opacity in [0, 1].
            color: Optional base color as an RGB tuple with values in
                [0, 1]. Used when no texture is provided.
            roughness: Surface roughness in ``[0, 1]``. ``0`` is perfectly
                smooth, ``1`` is fully rough.
            metallic: Metallicity in ``[0, 1]``. ``0`` is dielectric, ``1``
                is metal.
        """

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

        valid_texture = texture is not None and uvs is not None
        self._mesh_appearance[name] = {
            "texture": texture if valid_texture else None,
            "color": color,
            "roughness": roughness,
            "metallic": metallic,
            "opacity": opacity,
        }

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

        # Create and bind a material only when there is authored appearance
        # data. A UsdUVTexture shader with no "st" primvar would sample
        # undefined data, so textures require UVs.
        material = None
        if valid_texture:
            material = self._get_preview_surface_material(
                name,
                texture=texture,
                opacity=opacity,
                roughness=roughness,
                metallic=metallic,
            )
            self._texture_materials[name] = material
        elif color is not None or roughness is not None or metallic is not None or opacity is not None:
            material = self._get_preview_surface_material(
                name,
                color=color,
                opacity=opacity,
                roughness=roughness,
                metallic=metallic,
            )

        if material is not None:
            self._bind_material(mesh_prim.GetPrim(), material)

        # how to hide the prototype mesh but not the instances in USD?
        mesh_prim.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

        if opacity is not None:
            primvars = UsdGeom.PrimvarsAPI(mesh_prim)
            display_opacity = primvars.GetPrimvar("displayOpacity")
            if not display_opacity:
                display_opacity = primvars.CreatePrimvar(
                    "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant, 1
                )
            display_opacity.Set([float(np.clip(opacity, 0.0, 1.0))], self._frame_index)

    def _create_texture_material(self, mesh_name: str, mesh_prim, texture):
        """Create a UsdPreviewSurface material with a diffuse texture and bind it to *mesh_prim*."""
        material = self._get_preview_surface_material(mesh_name, texture=texture)
        if material is not None:
            self._bind_material(mesh_prim.GetPrim(), material)
            self._texture_materials[mesh_name] = material
        return material

    def _resolve_texture_path(self, mesh_name: str, texture) -> str | None:
        """Resolve a texture path or export an image array next to the USD file."""
        if isinstance(texture, str):
            return os.path.abspath(texture)

        if mesh_name in self._texture_paths:
            return self._texture_paths[mesh_name]

        from ..utils.texture import load_texture  # noqa: PLC0415

        tex_array = load_texture(texture)
        if tex_array is None:
            return None

        tex_dir = os.path.dirname(self.output_path)
        safe_name = mesh_name.replace("/", "_").replace("\\", "_")
        tex_path = os.path.join(tex_dir, f"_tex_{safe_name}.png")
        try:
            from PIL import Image

            Image.fromarray(tex_array).save(tex_path)
        except Exception as exc:
            warnings.warn(
                f"ViewerUSD: failed to export texture for mesh '{mesh_name}': {exc}. Mesh will render without texture.",
                stacklevel=2,
            )
            return None

        self._texture_paths[mesh_name] = tex_path
        return tex_path

    @staticmethod
    def _material_float(value: float | None, default: float) -> float:
        if value is None:
            return float(default)
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _material_color(color: tuple[float, float, float] | np.ndarray | None) -> tuple[float, float, float]:
        if color is None:
            return (0.5, 0.5, 0.5)
        color_np = np.asarray(color, dtype=np.float32).reshape(3)
        return tuple(float(v) for v in np.clip(color_np, 0.0, 1.0))

    @staticmethod
    def _material_key_value(value: float) -> float:
        return round(float(value), 6)

    def _preview_surface_opacity_value(self, requested_opacity: float) -> float:
        return requested_opacity

    def _preview_surface_ior_value(self, requested_opacity: float) -> float | None:
        del requested_opacity
        return None

    def _get_preview_surface_material(
        self,
        mesh_name: str,
        *,
        texture=None,
        color: tuple[float, float, float] | np.ndarray | None = None,
        opacity: float | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ):
        """Return a cached UsdPreviewSurface material for the requested appearance."""
        from pxr import Sdf as _Sdf
        from pxr import UsdShade

        tex_path = self._resolve_texture_path(mesh_name, texture) if texture is not None else None
        color_value = self._material_color(color)
        requested_opacity_value = self._material_float(opacity, 1.0)
        opacity_value = self._preview_surface_opacity_value(requested_opacity_value)
        roughness_value = self._material_float(roughness, 0.5)
        metallic_value = self._material_float(metallic, 0.0)
        ior_value = self._preview_surface_ior_value(requested_opacity_value)

        key = (
            "preview",
            os.path.normcase(tex_path) if tex_path is not None else None,
            tuple(self._material_key_value(v) for v in color_value),
            self._material_key_value(opacity_value),
            self._material_key_value(roughness_value),
            self._material_key_value(metallic_value),
            self._material_key_value(ior_value) if ior_value is not None else None,
        )
        if key in self._preview_materials:
            return self._preview_materials[key]

        digest = hashlib.blake2s(repr(key).encode("utf-8"), digest_size=8).hexdigest()
        safe = mesh_name.replace("/", "_").replace("\\", "_").lstrip("_") or "mesh"
        mat_path = f"/root/Materials/mat_{safe}_{digest}"
        self._ensure_scopes_for_path(self.stage, mat_path)

        material = UsdShade.Material.Define(self.stage, mat_path)
        surface = UsdShade.Shader.Define(self.stage, f"{mat_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        diff_input = surface.CreateInput("diffuseColor", _Sdf.ValueTypeNames.Color3f)
        surface.CreateInput("roughness", _Sdf.ValueTypeNames.Float).Set(roughness_value)
        surface.CreateInput("metallic", _Sdf.ValueTypeNames.Float).Set(metallic_value)
        surface.CreateInput("opacity", _Sdf.ValueTypeNames.Float).Set(opacity_value)
        surface.CreateInput("opacityMode", _Sdf.ValueTypeNames.Token).Set("transparent")
        surface.CreateInput("opacityThreshold", _Sdf.ValueTypeNames.Float).Set(0.0)
        if ior_value is not None:
            surface.CreateInput("ior", _Sdf.ValueTypeNames.Float).Set(ior_value)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

        if tex_path is not None:
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
            tex_reader.CreateInput("st", _Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.ConnectableAPI(), "result"
            )
        else:
            diff_input.Set(Gf.Vec3f(*color_value))

        self._preview_materials[key] = material
        return material

    @staticmethod
    def _bind_material(prim, material):
        from pxr import UsdShade

        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(material)

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
        *,
        opacities: wp.array[wp.float32] | None = None,
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
            opacities: Array of opacity values.
            hidden: Whether the instances are hidden.
        """
        # Get prototype path
        if mesh not in self._meshes:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        self._ensure_scopes_for_path(self.stage, self._get_path(name) + "/scope")

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
        if materials is not None:
            materials = materials.numpy()
        if opacities is not None:
            opacities = promote_to_clamped_float_array(opacities, len(xforms), value_name="Opacity")

        appearance_changed = colors is not None or materials is not None or opacities is not None
        appearance = self._instance_appearance.setdefault(name, {})
        if colors is not None:
            appearance["colors"] = np.asarray(colors, dtype=np.float32)
        if materials is not None:
            appearance["materials"] = np.asarray(materials, dtype=np.float32)
        if opacities is not None:
            appearance["opacities"] = np.asarray(opacities, dtype=np.float32)

        cached_colors = appearance.get("colors")
        cached_materials = appearance.get("materials")
        cached_opacities = appearance.get("opacities")
        if cached_colors is not None and len(cached_colors) != len(xforms):
            cached_colors = None
        if cached_materials is not None and len(cached_materials) != len(xforms):
            cached_materials = None
        if cached_opacities is not None and len(cached_opacities) != len(xforms):
            cached_opacities = None

        mesh_appearance = self._mesh_appearance.get(mesh, {})
        mesh_texture = mesh_appearance.get("texture")
        mesh_opacity = mesh_appearance.get("opacity")

        for i in range(len(xforms)):
            instance_path = self._get_path(name) + f"/instance_{i}"
            instance = self.stage.GetPrimAtPath(instance_path)
            created_instance = False

            if not instance:
                instance = self.stage.DefinePrim(instance_path)
                instance.GetReferences().AddInternalReference(self._get_path(mesh))
                created_instance = True

                UsdGeom.Imageable(instance).GetVisibilityAttr().Set("inherited" if not hidden else "invisible")
                _usd_add_xform(instance)

            # update transform
            if xforms is not None:
                pos = xforms[i][:3]
                rot = xforms[i][3:7]

                _usd_set_xform(instance, pos, rot, scales[i], self._frame_index)

            # update color
            if colors is not None:
                primvars = UsdGeom.PrimvarsAPI(instance)
                displayColor = primvars.GetPrimvar("displayColor")
                if not displayColor:
                    displayColor = primvars.CreatePrimvar(
                        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant, 1
                    )
                displayColor.Set(colors[i], self._frame_index)
            if opacities is not None:
                primvars = UsdGeom.PrimvarsAPI(instance)
                displayOpacity = primvars.GetPrimvar("displayOpacity")
                if not displayOpacity:
                    displayOpacity = primvars.CreatePrimvar(
                        "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.constant, 1
                    )
                displayOpacity.Set([float(opacities[i])], self._frame_index)

            material_color = cached_colors[i] if cached_colors is not None else mesh_appearance.get("color")
            material_params = cached_materials[i] if cached_materials is not None else None
            material_opacity = cached_opacities[i] if cached_opacities is not None else mesh_opacity
            if (appearance_changed or created_instance) and (
                material_color is not None or material_opacity is not None or material_params is not None
            ):
                use_texture = mesh_texture is not None and (
                    material_params is None or len(material_params) < 4 or material_params[3] > 0.5
                )
                roughness = (
                    float(material_params[0]) if material_params is not None and len(material_params) > 0 else None
                )
                metallic = (
                    float(material_params[1]) if material_params is not None and len(material_params) > 1 else None
                )
                material = self._get_preview_surface_material(
                    mesh,
                    texture=mesh_texture if use_texture else None,
                    color=None if use_texture else material_color,
                    opacity=float(material_opacity) if material_opacity is not None else None,
                    roughness=roughness,
                    metallic=metallic,
                )
                self._bind_material(instance, material)

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
        *,
        opacities: wp.array[wp.float32] | None = None,
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
            opacities: Instance opacities as a warp array of wp.float32.

        Raises:
            RuntimeError: If the mesh prototype is not found.
        """
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
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex, 1
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

            if opacities is not None:
                opacities_np = promote_to_clamped_float_array(opacities, num_instances, value_name="Opacity")
                displayOpacity = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayOpacity")
                displayOpacity.Set(opacities_np, self._frame_index)
                indices = Vt.IntArray(range(num_instances))
                displayOpacity.SetIndices(indices, self._frame_index)

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
        """Log points as a USD `Points` primitive.

        Args:
            name: Unique name for the point primitive.
            points: Point positions.
            radii: Point radii or a single shared radius.
            colors: Optional per-point colors or a shared RGB triplet.
            hidden: Whether the point primitive is hidden.

        Returns:
            Sdf.Path of the created/updated points primitive.
        """
        if points is None:
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
        instancer = UsdGeom.Points.Get(self.stage, path)
        if not instancer:
            self._ensure_scopes_for_path(self.stage, path)
            instancer = UsdGeom.Points.Define(self.stage, path)

            UsdGeom.Primvar(instancer.GetWidthsAttr()).SetInterpolation(radius_interp)
            UsdGeom.Primvar(instancer.GetDisplayColorAttr()).SetInterpolation(color_interp)

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
            return None, "constant"

        if isinstance(colors, wp.array):
            colors = colors.numpy()

        if self._is_single_rgb_triplet(colors):
            colors_arr = np.asarray(colors, dtype=np.float32)
            return colors_arr.reshape(1, 3), "constant"

        if isinstance(colors, np.ndarray):
            return colors, "vertex"

        if isinstance(colors, list | tuple):
            # Keep list/tuple inputs as-is for existing valid per-point color inputs.
            if len(colors) == num_points:
                return colors, "vertex"
            return np.asarray(colors), "vertex"

        return np.asarray(colors), "vertex"

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
