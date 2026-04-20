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

from __future__ import annotations

import ctypes
import math
import os
import tempfile
import warnings
from time import perf_counter
from typing import Any

import numpy as np
import warp as wp

from ..core.types import override

try:
    from pxr import Gf, UsdGeom
except ImportError:
    Gf = UsdGeom = None

from .camera import Camera
from .picking import Picking
from .viewer_gui import ViewerGui
from .viewer_usd import ViewerUSD, _compute_segment_xform
from .wind import Wind

PROFILE_ENABLED = os.environ.get("NEWTON_PROFILE", "0") != "0"


@wp.kernel(enable_backward=False)
def write_transforms(xform: wp.array[wp.transform], scale: wp.array[wp.vec3], offset: int, m_out: wp.array[wp.mat44d]):
    tid = wp.tid()
    xf32 = xform[tid]
    sc32 = scale[tid]
    # convert to float64
    p64 = wp.vec3d(wp.float64(xf32[0]), wp.float64(xf32[1]), wp.float64(xf32[2]))
    q64 = wp.quatd(wp.float64(xf32[3]), wp.float64(xf32[4]), wp.float64(xf32[5]), wp.float64(xf32[6]))
    s64 = wp.vec3d(wp.float64(sc32[0]), wp.float64(sc32[1]), wp.float64(sc32[2]))
    # NOTE: transpose needed
    m_out[offset + tid] = wp.transpose(wp.transform_compose(p64, q64, s64))


@wp.kernel(enable_backward=False)
def update_and_write_shape_transforms(
    shape_xforms: wp.array[wp.transform],
    shape_parents: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_worlds: wp.array[int],
    world_offsets: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    mat44_offset: int,
    m_out: wp.array[wp.mat44d],
):
    """Fused kernel: compute world transform from body state then write as mat44d.

    Combines the work of ``update_shape_xforms`` and ``write_transforms`` into a
    single pass, eliminating the intermediate ``world_xforms`` write and read.
    """
    tid = wp.tid()
    xf = shape_xforms[tid]
    parent = shape_parents[tid]
    if parent >= 0:
        world_xf = wp.transform_multiply(body_q[parent], xf)
    else:
        world_xf = xf
    if world_offsets:
        w = shape_worlds[tid]
        if w >= 0 and w < world_offsets.shape[0]:
            world_xf = wp.transform(world_xf.p + world_offsets[w], world_xf.q)
    # promote to f64
    p = world_xf.p
    q = world_xf.q
    sc = scales[tid]
    p64 = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))
    q64 = wp.quatd(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]), wp.float64(q[3]))
    s64 = wp.vec3d(wp.float64(sc[0]), wp.float64(sc[1]), wp.float64(sc[2]))
    # NOTE: transpose needed
    m_out[mat44_offset + tid] = wp.transpose(wp.transform_compose(p64, q64, s64))


class ViewerRTX(ViewerUSD):
    """Real-time ray-traced viewer using NVIDIA OVRTX.

    Builds a USD scene during the first simulation frame using the ViewerUSD
    base class, serializes it to disk, then creates an OVRTX renderer for
    real-time path-traced rendering.  Subsequent frames update rigid-body
    transforms (and deforming-mesh vertices) via the OVRTX attribute API
    and present the rendered image in a pyglet / OpenGL window.
    """

    _PHASE_BUILD = 0
    _PHASE_RENDER = 1
    _PICKING_LINE_NAME = "picking_line"
    _PICKING_LINE_RADIUS = 0.01
    _PICKING_LINE_COLOR = (0.0, 1.0, 1.0)

    # Available lighting environment presets.
    ENVIRONMENTS = ("default", "studio", "none")

    def __init__(
        self,
        width=1280,
        height=720,
        fps=60,
        up_axis="Z",
        num_frames=None,
        scaling=1.0,
        headless=False,
        paused=False,
        environment="default",
        vsync=False,
        async_rendering=True,
    ):
        # FIXME: Disable USD checks in OVRTX that refuse to load the library if `usd-core` is present.
        # This is a temporary workaround until OVRTX is shipped with namespaced USD libs.
        # Note that disabling the check may cause crashes when the version of `usd-core` is the same as
        # as the version of the USD libs shipped with OVRTX.
        os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")

        try:
            import ovrtx  # noqa: F401
        except ImportError as e:
            raise ImportError("ovrtx package is required for ViewerRTX. Install with: pip install ovrtx") from e

        if UsdGeom is None:
            raise ImportError("usd-core package is required for ViewerRTX. Install with: pip install usd-core")

        self._environment = environment.lower()
        if self._environment not in self.ENVIRONMENTS:
            raise ValueError(
                f"Unknown RTX environment {self._environment!r}. Choose from: {', '.join(self.ENVIRONMENTS)}"
            )

        self._tmp_usd_file = tempfile.NamedTemporaryFile(suffix=".usd", delete=False)
        self._tmp_usd_file.close()
        self._tmp_usd_path = self._tmp_usd_file.name

        # Pre-initialize fields that clear_model() (called from super().__init__) touches
        self._ui_callbacks: dict[str, list] = {"side": [], "stats": [], "free": [], "panel": [], "rendering": []}
        self._ui_callbacks["rendering"].append(self._ui_populate_rendering_panel)
        self._paused = paused
        self._rtx = None
        self._pending_line_batches: dict[str, tuple[Any, Any, Any, float, bool]] = {}
        self._line_batch_paths: dict[str, str] = {}
        self._line_batch_proto_paths: dict[str, str] = {}
        self._line_batch_widths: dict[str, float] = {}
        self._line_batch_handles: dict[str, Any] = {}
        self._pending_point_batches: dict[str, tuple[Any, Any, Any, bool]] = {}
        self._point_batch_paths: dict[str, str] = {}
        self._point_batch_colors: dict[str, np.ndarray] = {}
        self._point_batch_synced_counts: dict[str, int] = {}

        super().__init__(
            output_path=self._tmp_usd_path,
            fps=fps,
            up_axis=up_axis,
            num_frames=num_frames,
            scaling=scaling,
        )

        self._width = width
        self._height = height
        self._headless = headless

        self._phase = self._PHASE_BUILD

        # Pyglet window state
        self._window = None
        self._pyglet = None
        self._pyglet_gl = None
        self._pyglet_app = None
        self._should_close = False
        self._camera_dirty = True

        # Instance prim paths collected during build phase, keyed by instancer name.
        # Iteration order (insertion order) defines the layout inside the binding.
        self._instance_prim_paths: dict[str, list[str]] = {}
        self._all_instance_paths: list[str] = []
        self._transform_binding = None

        # Flat concatenated shape arrays built once at the end of the build phase.
        # Ordered to match the mat44d buffer layout so all shapes can be updated in
        # a single kernel launch.  None until _build_flat_shape_arrays() runs.
        self._flat_shape_xforms: wp.array | None = None
        self._flat_shape_parents: wp.array | None = None
        self._flat_shape_worlds: wp.array | None = None
        self._flat_shape_scales: wp.array | None = None
        self._flat_total_shapes: int = 0
        self._flat_mat44_offset: int = 0  # offset of the flat section in the mat44d buffer

        # Mesh prim paths for deforming-mesh point updates
        self._mesh_prim_paths: dict[str, str] = {}

        # Per-frame pending data (cleared each begin_frame)
        self._pending_xforms: dict[str, tuple[Any, Any]] = {}
        self._pending_mesh_points: dict[str, np.ndarray] = {}
        self._pending_mesh_normals: dict[str, np.ndarray] = {}

        # Build-time line instancers plus runtime-authored capsule layers

        # Camera (shared Camera class with ViewerGL)
        self.camera = Camera(width=self._width, height=self._height, up_axis=up_axis)
        self._camera_prim_path = "/World/Camera"
        self._render_product_path = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"

        # Input / timing state
        self._keys_down: set[int] = set()
        self._last_perf_time: float | None = None
        self.gui = None

        # async rendering
        self._render_result = None

        # last frame's render products (used for screenshots)
        self._render_products = None

        # Window is deferred until _init_ovrtx to avoid pyglet/Warp
        # kernel compilation deadlock on Windows.

        # initial value for vsync (applied once window is created)
        self._vsync_init = vsync

        self._async = async_rendering

    # ------------------------------------------------------------------ window

    def _init_window(self):
        """Create a pyglet window with GL texture + shader for fast framebuffer blitting."""
        import ctypes  # noqa: PLC0415

        import pyglet

        pyglet.options["debug_gl"] = False
        from pyglet import gl

        self._window = pyglet.window.Window(
            width=self._width,
            height=self._height,
            caption="Newton RTX Viewer",
            resizable=True,
            visible=not self._headless,
            vsync=self._vsync_init,
        )

        # cache the imported pyglet modules to avoid reimporting later
        self._pyglet = pyglet
        self._pyglet_gl = pyglet.gl
        self._pyglet_app = pyglet.app

        # ---- GL texture + shader for zero-copy blit --------------------------
        self._window.switch_to()

        tex_id = (gl.GLuint * 1)()
        gl.glGenTextures(1, tex_id)
        self._gl_texture = tex_id[0]
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gl_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            self.camera.width,
            self.camera.height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self._tex_resource = wp.GLTextureResource(
            self._gl_texture, gl.GL_TEXTURE_2D, flags=wp.TextureResourceFlags.WRITE_DISCARD
        )

        # Compile fullscreen-triangle shader (linear→sRGB gamma + Y-flip in fragment)
        _VS = b"""#version 330
out vec2 uv;
void main() {
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
\x00"""
        _FS = b"""#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(tex, vec2(uv.x, 1.0 - uv.y));
    fragColor = c;
}
\x00"""

        def _compile_shader(src, stype):
            s = gl.glCreateShader(stype)
            src_p = ctypes.c_char_p(src)
            src_pp = (ctypes.c_char_p * 1)(src_p)
            gl.glShaderSource(s, 1, ctypes.cast(src_pp, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))), None)
            gl.glCompileShader(s)
            status = (gl.GLint * 1)()
            gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS, status)
            if not status[0]:
                log_len = (gl.GLint * 1)()
                gl.glGetShaderiv(s, gl.GL_INFO_LOG_LENGTH, log_len)
                log = (ctypes.c_char * log_len[0])()
                gl.glGetShaderInfoLog(s, log_len[0], None, log)
                raise RuntimeError(f"Shader compilation failed:\n{log.value.decode()}")
            return s

        vs = _compile_shader(_VS, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_FS, gl.GL_FRAGMENT_SHADER)
        self._gl_program = gl.glCreateProgram()
        gl.glAttachShader(self._gl_program, vs)
        gl.glAttachShader(self._gl_program, fs)
        gl.glLinkProgram(self._gl_program)
        link_status = (gl.GLint * 1)()
        gl.glGetProgramiv(self._gl_program, gl.GL_LINK_STATUS, link_status)
        if not link_status[0]:
            log_len = (gl.GLint * 1)()
            gl.glGetProgramiv(self._gl_program, gl.GL_INFO_LOG_LENGTH, log_len)
            log = (ctypes.c_char * log_len[0])()
            gl.glGetProgramInfoLog(self._gl_program, log_len[0], None, log)
            raise RuntimeError(f"Shader program linking failed:\n{log.value.decode()}")
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)

        # Empty VAO required by core profile for the fullscreen triangle
        vao = (gl.GLuint * 1)()
        gl.glGenVertexArrays(1, vao)
        self._gl_vao = vao[0]

        # ---- input callbacks ------------------------------------------------
        @self._window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if self.gui:
                self.gui.handle_mouse_drag(x, y, dx, dy, buttons, self._to_framebuffer_coords)

        @self._window.event
        def on_mouse_press(x, y, button, modifiers):
            if self.gui:
                self.gui.handle_mouse_press(x, y, button, self._to_framebuffer_coords)

        @self._window.event
        def on_mouse_release(x, y, button, modifiers):
            if self.gui:
                self.gui.handle_mouse_release(x, y, button)

        @self._window.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            if self.gui:
                self.gui.handle_mouse_scroll(scroll_y)

        @self._window.event
        def on_key_press(symbol, modifiers):
            if not (self.gui and self.gui.should_ignore_keyboard_input()):
                self._keys_down.add(symbol)
            if self.gui:
                self.gui.handle_key_press(symbol, close_fn=self._window.close)

        @self._window.event
        def on_key_release(symbol, modifiers):
            self._keys_down.discard(symbol)

        @self._window.event
        def on_resize(width, height):
            self._width = width
            self._height = height

        @self._window.event
        def on_close():
            self._should_close = True

        self.gui = ViewerGui(self, self._window)

    @property
    def ui(self):
        if self.gui is None:
            return None
        return self.gui.ui

    @property
    def vsync(self) -> bool:
        """
        Get the current vsync state.

        Returns:
            bool: True if vsync is enabled, False otherwise.
        """
        if self._window is not None:
            return self._window.vsync
        else:
            return self._vsync_init

    @vsync.setter
    def vsync(self, enabled: bool):
        """
        Set the vsync state.

        Args:
            enabled: Enable or disable vsync.
        """
        if self._window is not None:
            self._window.set_vsync(enabled)
        else:
            self._vsync_init = enabled

    # ------------------------------------------------------------------ camera

    def _compute_camera_matrix(self):
        """Return a 4x4 row-major world-transform for the camera prim (USD convention)."""
        fwd = np.array(self.camera.get_front(), dtype=np.float64)
        right = np.array(self.camera.get_right(), dtype=np.float64)
        up = np.array(self.camera.get_up(), dtype=np.float64)

        mat = np.eye(4, dtype=np.float64)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = -fwd  # USD cameras look along local -Z
        mat[3, :3] = np.array(self.camera.pos, dtype=np.float64)
        return mat

    def _to_framebuffer_coords(self, x: float, y: float) -> tuple[float, float]:
        """Map a window-space mouse point to render-target pixel coordinates.

        Accounts for the letterbox/pillarbox viewport so that picking works
        correctly after a window resize.
        """
        if self._window is None:
            return float(x), float(y)
        win_w, win_h = self._window.get_size()
        if win_w <= 0 or win_h <= 0:
            return float(x), float(y)
        render_aspect = self.camera.width / max(self.camera.height, 1)
        window_aspect = win_w / max(win_h, 1)
        if window_aspect >= render_aspect:
            # Pillarbox: black bars left/right
            vp_h = win_h
            vp_w = win_h * render_aspect
            vp_x = (win_w - vp_w) / 2.0
            vp_y = 0.0
        else:
            # Letterbox: black bars top/bottom
            vp_w = win_w
            vp_h = win_w / render_aspect
            vp_x = 0.0
            vp_y = (win_h - vp_h) / 2.0
        rx = (x - vp_x) / vp_w * self.camera.width
        ry = (y - vp_y) / vp_h * self.camera.height
        return float(rx), float(ry)

    def _frame_camera_on_model(self):
        """Frame the camera to show all visible objects in the scene."""
        self.gui.frame_camera_on_model()

    # -------------------------------------------------------- USD scene helpers

    def _add_camera_lights_and_render_product(self):
        """Insert camera, lights, and RenderProduct into the stage before serialisation."""
        from pxr import Sdf

        # ---- Camera ----------------------------------------------------------
        cam = UsdGeom.Camera.Define(self.stage, self._camera_prim_path)

        aspect = self.camera.width / max(self.camera.height, 1)
        # camera.fov is vertical FOV, so derive focal length from the vertical aperture.
        v_aperture = 20.955
        h_aperture = v_aperture * aspect
        focal_length = v_aperture / (2.0 * math.tan(math.radians(self.camera.fov) / 2.0))

        cam.GetFocalLengthAttr().Set(focal_length)
        cam.GetHorizontalApertureAttr().Set(h_aperture)
        cam.GetVerticalApertureAttr().Set(v_aperture)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(self.camera.near, self.camera.far))

        xform = UsdGeom.Xform(cam.GetPrim())
        xform.ClearXformOpOrder()
        mat_op = xform.AddTransformOp()
        cam_mat = self._compute_camera_matrix()
        gf_mat = Gf.Matrix4d(*cam_mat.flatten().tolist())
        mat_op.Set(gf_mat)

        # ---- Lights ----------------------------------------------------------
        if self._environment == "studio":
            self._add_studio_lights()
        elif self._environment == "default":
            self._add_default_lights()

        # ---- Render hierarchy (must match Kit convention for OVRTX) ------------
        # Structure: /Render/OmniverseKit/HydraTextures/<product>
        #            /Render/Vars/LdrColor
        #            /Render/OmniverseGlobalRenderSettings
        self.stage.DefinePrim("/Render")
        self.stage.DefinePrim("/Render/OmniverseKit")
        self.stage.DefinePrim("/Render/OmniverseKit/HydraTextures")

        rp = self.stage.DefinePrim(self._render_product_path, "RenderProduct")
        rp.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.Create(
                prependedItems=[
                    "OmniRtxSettingsCommonAdvancedAPI_1",
                    "OmniRtxSettingsRtAdvancedAPI_1",
                    "OmniRtxSettingsPtAdvancedAPI_1",
                    "OmniRtxPostColorGradingAPI_1",
                    "OmniRtxPostChromaticAberrationAPI_1",
                    "OmniRtxPostBloomPhysicalAPI_1",
                    "OmniRtxPostMatteObjectAPI_1",
                    "OmniRtxPostCompositingAPI_1",
                    "OmniRtxPostDofAPI_1",
                    "OmniRtxPostMotionBlurAPI_1",
                    "OmniRtxPostTvNoiseAPI_1",
                    "OmniRtxPostTonemapIrayReinhardAPI_1",
                    "OmniRtxPostDebugSettingsAPI_1",
                    "OmniRtxDebugSettingsAPI_1",
                ]
            ),
        )
        rp.CreateRelationship("camera").SetTargets([Sdf.Path(self._camera_prim_path)])
        rp.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2, custom=False).Set(
            Gf.Vec2i(self.camera.width, self.camera.height)
        )

        # RenderVar lives at /Render/Vars/LdrColor (NOT nested under the product)
        rv_path = "/Render/Vars/LdrColor"
        rv = self.stage.DefinePrim(rv_path, "RenderVar")
        rv.CreateAttribute("sourceName", Sdf.ValueTypeNames.String, custom=False).Set("LdrColor")
        rp.CreateRelationship("orderedVars").SetTargets([Sdf.Path(rv_path)])

        # ---- RTX render settings on the RenderProduct -------------------------
        rp.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set("RealTimePathTracing")
        rp.CreateAttribute("omni:rtx:ambientOcclusion:denoiserMode", Sdf.ValueTypeNames.Token).Set("none")
        rp.CreateAttribute("omni:rtx:background:source:texture:textureMode", Sdf.ValueTypeNames.Token).Set(
            "repeatMirrored"
        )
        rp.CreateAttribute("omni:rtx:background:source:type", Sdf.ValueTypeNames.Token).Set("domeLight")
        rp.CreateAttribute("omni:rtx:debug:view:pixelDebug:enableFixedTextPos", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:directLighting:sampledLighting:denoisingTechnique", Sdf.ValueTypeNames.Token).Set(
            "None"
        )
        rp.CreateAttribute("omni:rtx:dlss:frameGeneration", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:indirectDiffuse:denoiser:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:post:aa:limitedOps", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:post:registeredCompositing:invertColorCorrection", Sdf.ValueTypeNames.Bool).Set(
            True
        )
        rp.CreateAttribute("omni:rtx:post:registeredCompositing:invertToneMap", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:pt:maxSamplesPerLaunch", Sdf.ValueTypeNames.Int).Set(2073600)
        rp.CreateAttribute("omni:rtx:pt:mgpu:maxPixelsPerRegionExponent", Sdf.ValueTypeNames.Int).Set(12)
        rp.CreateAttribute("omni:rtx:pt:denoising:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:pt:samplesPerPixel", Sdf.ValueTypeNames.UInt).Set(1)
        rp.CreateAttribute("omni:rtx:reflections:denoiser:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:rt:ambientLight:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))
        rp.CreateAttribute("omni:rtx:rt:demoire", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:rt:lightcache:spatialCache:dontResolveConflicts", Sdf.ValueTypeNames.Bool).Set(
            True
        )
        rp.CreateAttribute("omni:rtx:rt:sss:samples", Sdf.ValueTypeNames.Int).Set(1)
        rp.CreateAttribute("omni:rtx:rtpt:maxVolumeBounces", Sdf.ValueTypeNames.Int).Set(15)
        rp.CreateAttribute("omni:rtx:rtpt:modulatingRoughnessThreshold", Sdf.ValueTypeNames.Float).Set(0.08)
        rp.CreateAttribute("omni:rtx:scene:hydra:mdlMaterialWarmup", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:viewTile:limit", Sdf.ValueTypeNames.UInt).Set(4294967295)

        # Disable the quality convergence loop to minimize step() latency
        rp.CreateAttribute("omni:rtx:quality", Sdf.ValueTypeNames.Int, custom=False).Set(0)
        rp.CreateAttribute("omni:rtx:waitForEvents", Sdf.ValueTypeNames.TokenArray).Set([])

        # ---- RenderSettings --------------------------------------------------
        rs = self.stage.DefinePrim("/Render/OmniverseGlobalRenderSettings", "RenderSettings")
        rs.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.Create(
                prependedItems=[
                    "OmniRtxSettingsGlobalRtAdvancedAPI_1",
                    "OmniRtxSettingsGlobalPtAdvancedAPI_1",
                ]
            ),
        )
        rs.CreateRelationship("products").SetTargets([Sdf.Path(self._render_product_path)])

    def _add_default_lights(self):
        """Default lighting: dome light + distant directional light."""
        from pxr import UsdLux

        dome = UsdLux.DomeLight.Define(self.stage, "/root/_RTXDomeLight")
        dome.GetIntensityAttr().Set(150.0)

        distant = UsdLux.DistantLight.Define(self.stage, "/root/_RTXDistantLight")
        distant.GetIntensityAttr().Set(900.0)
        distant.GetAngleAttr().Set(0.53)
        dx = UsdGeom.Xform(distant.GetPrim())
        dx.ClearXformOpOrder()
        rot = dx.AddRotateXYZOp()
        if self.camera.up_axis == 2:
            rot.Set(Gf.Vec3f(-45.0, 30.0, 0.0))
        else:
            rot.Set(Gf.Vec3f(-45.0, 0.0, 30.0))

    def _add_studio_lights(self):
        """Studio lighting rig from dome + warm distant + cool fill sphere."""
        from pxr import Sdf, UsdLux

        # Dome light — cool-tinted low ambient
        dome_xf = UsdGeom.Xform.Define(self.stage, "/root/_RTXDomeLight")
        dome_xf.ClearXformOpOrder()
        dome = UsdLux.DomeLight.Define(self.stage, "/root/_RTXDomeLight/_RTXDomeLight")
        dome.GetColorAttr().Set(Gf.Vec3f(0.250, 0.319, 0.409))
        dome.GetIntensityAttr().Set(200.0)

        # Distant light — warm key, angled from above-behind
        dist_xf = UsdGeom.Xform.Define(self.stage, "/root/_RTXDistantLight")
        dist_xf.ClearXformOpOrder()
        dist_xf.AddRotateXYZOp().Set(Gf.Vec3f(41.4, 0.0, -175.7))
        distant = UsdLux.DistantLight.Define(self.stage, "/root/_RTXDistantLight/_RTXDistantLight")
        distant.GetColorAttr().Set(Gf.Vec3f(1.0, 0.906, 0.722))
        distant.GetIntensityAttr().Set(3000.0)

        # Cool fill sphere light (blue-white)
        fill_xf = UsdGeom.Xform.Define(self.stage, "/root/_RTXFillLight")
        fill_xf.ClearXformOpOrder()
        fill_xf.AddTranslateOp().Set(Gf.Vec3d(5.0, 0.0, 5.5))
        fill = UsdLux.SphereLight.Define(self.stage, "/root/_RTXFillLight/_RTXFillLight")
        fill.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["ShapingAPI"]))
        fill.GetColorAttr().Set(Gf.Vec3f(0.468, 0.684, 1.0))
        fill.GetIntensityAttr().Set(300000.0)
        fill.GetRadiusAttr().Set(0.5)

    def _apply_ground_material(self):
        """Bind a dark, shiny UsdPreviewSurface material to ground-plane meshes."""
        from pxr import Sdf, UsdShade

        plane_prims = [prim for name, prim in self._meshes.items() if "plane" in name.lower()]
        if not plane_prims:
            return

        mat_path = "/root/Materials/mat_ground"
        self._ensure_scopes_for_path(self.stage, mat_path)

        material = UsdShade.Material.Define(self.stage, mat_path)
        surface = UsdShade.Shader.Define(self.stage, f"{mat_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.05, 0.06))
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.15)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

        for prim in plane_prims:
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim())
            UsdShade.MaterialBindingAPI(prim).Bind(material)

    def add_background_usd(self, path: str):
        """Add a reference to a background USD (e.g. Gaussian splat scan).

        Must be called before the first frame (during the build phase).

        Args:
            path: Absolute or relative path to a USD file.
        """
        if self._phase != self._PHASE_BUILD:
            raise RuntimeError("add_background_usd() must be called before the first simulation frame")
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Background USD not found: {path}")
        bg_prim = self.stage.DefinePrim("/root/background")
        bg_prim.GetReferences().AddReference(path)

    # ------------------------------------------------------------- OVRTX init

    def _init_ovrtx(self):
        """Serialise the USD stage, create the OVRTX renderer and load the scene."""

        self._add_camera_lights_and_render_product()
        self._apply_ground_material()

        self.stage.GetRootLayer().Save()

        try:
            import ovrtx

            config = ovrtx.RendererConfig()
            config.log_level = "error"
            self._rtx = ovrtx.Renderer(config=config)
            self._rtx.add_usd(self._tmp_usd_path)

            # Flat prim-path list for a single transform binding
            self._all_instance_paths = []
            for paths in self._instance_prim_paths.values():
                self._all_instance_paths.extend(paths)

            if self._all_instance_paths:
                from ovrtx import PrimMode, Semantic

                self._transform_binding = self._rtx.bind_attribute(
                    prim_paths=self._all_instance_paths,
                    attribute_name="omni:xform",
                    semantic=Semantic.XFORM_MAT4x4,
                    prim_mode=PrimMode.MUST_EXIST,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create OVRTX renderer: {e}") from e

        # Create the presentation window now that all Warp kernels have been
        # compiled.  Doing this earlier causes a deadlock on Windows because
        # the Win32 message pump and Warp's JIT compilation fight for the
        # main thread.  Skip entirely in headless mode — a hidden window still
        # requires a display server and an OpenGL context.
        if not self._headless:
            try:
                self._init_window()
            except Exception as e:
                raise RuntimeError(f"Failed to create window: {e}") from e

        self._build_flat_shape_arrays()

        self._phase = self._PHASE_RENDER

    def _build_flat_shape_arrays(self):
        """Concatenate per-batch shape arrays into flat warp arrays matching the mat44d layout.

        Called once at the end of the build phase.  The resulting arrays are static
        (topology does not change per-frame) and allow all shape transforms to be
        updated with a single ``update_and_write_shape_transforms`` kernel launch instead of
        one launch per shape batch.
        """
        # _shape_instances is keyed by geometry hash (int), not by name; build a reverse map.
        name_to_shapes = {s.name: s for s in self._shape_instances.values()}

        chunks_xforms = []
        chunks_parents = []
        chunks_worlds = []
        chunks_scales = []
        flat_mat44_offset = 0
        found_shape = False

        for name, paths in self._instance_prim_paths.items():
            count = len(paths)
            shapes = name_to_shapes.get(name)
            if shapes is not None:
                found_shape = True
                chunks_xforms.append(shapes.xforms.numpy())
                chunks_parents.append(shapes.parents.numpy())
                chunks_worlds.append(shapes.worlds.numpy())
                chunks_scales.append(shapes.scales.numpy())
            elif not found_shape:
                # Non-shape entry (e.g. future pre-shape prim) before any shapes;
                # keep track so the flat section starts at the right mat44d offset.
                flat_mat44_offset += count

        if not chunks_xforms:
            return

        dev = self.device
        self._flat_shape_xforms = wp.array(np.concatenate(chunks_xforms, axis=0), dtype=wp.transform, device=dev)
        self._flat_shape_parents = wp.array(np.concatenate(chunks_parents, axis=0), dtype=int, device=dev)
        self._flat_shape_worlds = wp.array(np.concatenate(chunks_worlds, axis=0), dtype=int, device=dev)
        self._flat_shape_scales = wp.array(np.concatenate(chunks_scales, axis=0), dtype=wp.vec3, device=dev)
        self._flat_total_shapes = len(self._flat_shape_xforms)
        self._flat_mat44_offset = flat_mat44_offset

    # ------------------------------------------------ ViewerUSD overrides

    @override
    def set_model(self, model, max_worlds=None):
        super().set_model(model, max_worlds=max_worlds)
        if model is not None:
            from pyglet.math import Vec3 as PyVec3

            axis_idx = (
                model.up_axis
                if isinstance(model.up_axis, int)
                else {"X": 0, "Y": 1, "Z": 2}.get(str(model.up_axis).upper(), 2)
            )
            self.camera.up_axis = axis_idx
            if axis_idx == 0:
                self.camera.pos = PyVec3(2.0, 0.0, 10.0)
            elif axis_idx == 2:
                self.camera.pos = PyVec3(10.0, 0.0, 2.0)
            else:
                self.camera.pos = PyVec3(0.0, 2.0, 10.0)

        self.picking = Picking(model, world_offsets=self.world_offsets)
        self.wind = Wind(model)

        if model is not None:
            try:
                from ..geometry import raycast as _raycast_module  # noqa: PLC0415

                wp.load_module(module=_raycast_module, device=model.device)
                wp.load_module(module="newton._src.viewer.kernels", device=model.device)
            except Exception as exc:
                warnings.warn(
                    f"ViewerRTX: Failed to precompile Warp kernels for device {model.device}: {exc}",
                    category=RuntimeWarning,
                    stacklevel=2,
                )

    @override
    def set_world_offsets(self, spacing):
        super().set_world_offsets(spacing)
        if self.picking is not None:
            self.picking.world_offsets = self.world_offsets

    @override
    def set_camera(self, pos, pitch: float, yaw: float):
        from pyglet.math import Vec3 as PyVec3

        try:
            self.camera.pos = PyVec3(float(pos[0]), float(pos[1]), float(pos[2]))
        except (TypeError, IndexError, KeyError):
            pass
        self.camera.pitch = pitch
        self.camera.yaw = yaw
        self._camera_dirty = True

    def _ensure_picking_line_primitive(self):
        if self._phase != self._PHASE_BUILD or self._PICKING_LINE_NAME in self._instance_prim_paths:
            return

        if Gf is None or UsdGeom is None:
            return

        path = self._get_path(self._PICKING_LINE_NAME)
        self._ensure_scopes_for_path(self.stage, path)

        xform = UsdGeom.Xform.Define(self.stage, path)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        xform.AddScaleOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

        capsule = UsdGeom.Capsule.Define(self.stage, xform.GetPath().AppendChild("capsule"))
        capsule.GetAxisAttr().Set(UsdGeom.Tokens.z)
        capsule.GetRadiusAttr().Set(self._PICKING_LINE_RADIUS)
        capsule.GetHeightAttr().Set(1.0)
        capsule.GetDisplayColorAttr().Set([Gf.Vec3f(*self._PICKING_LINE_COLOR)])

        # Use an emissive material so the picking line stays visibly cyan even under scene shadows.
        from pxr import Sdf, UsdShade

        mat_path = "/root/Materials/mat_picking_line"
        self._ensure_scopes_for_path(self.stage, mat_path)
        material = UsdShade.Material.Define(self.stage, mat_path)
        surface = UsdShade.Shader.Define(self.stage, f"{mat_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self._PICKING_LINE_COLOR))
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim())
        UsdShade.MaterialBindingAPI(capsule).Bind(material)

        self._instance_prim_paths[self._PICKING_LINE_NAME] = [path]

    def _remove_runtime_line_batch_layer(self, name: str):
        if self._rtx is None:
            return

        handle = self._line_batch_handles.pop(name, None)
        if handle is not None:
            self._rtx.remove_usd(handle)

    def _ensure_point_batch_primitive(self, name: str):
        if Gf is None or UsdGeom is None:
            return None

        path = self._point_batch_paths.get(name, self._get_path(name))
        instancer = UsdGeom.PointInstancer.Get(self.stage, path)
        if not instancer:
            from pxr import Sdf

            self._ensure_scopes_for_path(self.stage, path)
            instancer = UsdGeom.PointInstancer.Define(self.stage, path)
            sphere = UsdGeom.Sphere.Define(self.stage, instancer.GetPath().AppendChild("sphere"))
            sphere.GetRadiusAttr().Set(1.0)
            instancer.GetPrototypesRel().SetTargets([sphere.GetPath()])
            primvars = UsdGeom.PrimvarsAPI(instancer)
            if not primvars.GetPrimvar("displayColor"):
                primvars.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1)

        self._point_batch_paths[name] = path
        return instancer

    def _build_point_batch_arrays(
        self, points, radii, colors
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        empty_positions = np.zeros((0, 3), dtype=np.float32)
        empty_scales = np.zeros((0, 3), dtype=np.float32)
        empty_proto_indices = np.zeros(0, dtype=np.int32)
        empty_ids = np.zeros(0, dtype=np.int64)

        if points is None:
            return empty_positions, empty_scales, empty_proto_indices, empty_ids, None, None

        points_np = points.numpy() if isinstance(points, wp.array) else points
        positions = np.asarray(points_np, dtype=np.float32).reshape((-1, 3))
        num_points = len(positions)
        if num_points == 0:
            return empty_positions, empty_scales, empty_proto_indices, empty_ids, None, None

        if radii is None:
            radii_np = np.full(num_points, 0.1, dtype=np.float32)
        elif np.isscalar(radii):
            radii_np = np.full(num_points, float(radii), dtype=np.float32)
        else:
            radii_np = np.asarray(radii.numpy() if isinstance(radii, wp.array) else radii, dtype=np.float32).reshape(-1)
            if radii_np.shape[0] == 1 and num_points > 1:
                radii_np = np.full(num_points, float(radii_np[0]), dtype=np.float32)
            elif radii_np.shape[0] != num_points:
                raise ValueError("Number of point radii must match the number of points.")

        scales = np.repeat(radii_np[:, None], 3, axis=1)
        proto_indices = np.zeros(num_points, dtype=np.int32)
        ids = np.arange(num_points, dtype=np.int64)

        colors_np = None
        color_indices = None
        if colors is not None:
            color_values, _ = self._normalize_point_colors(colors, num_points)
            colors_np = np.asarray(color_values, dtype=np.float32)
            if colors_np.ndim == 1:
                if colors_np.shape[0] != 3:
                    raise ValueError("Point colors must be an RGB triplet or an array of RGB triplets.")
                colors_np = colors_np.reshape(1, 3)
            elif colors_np.ndim != 2 or colors_np.shape[1] != 3:
                raise ValueError("Point colors must have shape (N, 3).")

            if colors_np.shape[0] == 1:
                color_indices = np.zeros(num_points, dtype=np.int32)
            elif colors_np.shape[0] == num_points:
                color_indices = np.arange(num_points, dtype=np.int32)
            else:
                raise ValueError("Number of point colors must match the number of points.")

        return positions, scales, proto_indices, ids, colors_np, color_indices

    def _rebuild_runtime_line_batch_layer(self, name: str, starts, ends, colors, width: float, hidden: bool) -> bool:
        if self._rtx is None or Gf is None or UsdGeom is None:
            return False

        self._remove_runtime_line_batch_layer(name)

        (
            positions,
            orientations,
            scales,
            _proto_indices,
            _ids,
            colors_np,
            color_indices,
        ) = self._build_line_batch_arrays(starts, ends, colors, hidden)

        if hidden or len(positions) == 0:
            return True

        try:
            from pxr import Sdf as _Sdf
            from pxr import Usd as _Usd
            from pxr import UsdGeom as _UsdGeom
            from pxr import UsdShade as _UsdShade
        except ImportError:
            return False

        target_path = self._get_path(name)
        prim_name = target_path.rsplit("/", 1)[-1]
        stage = _Usd.Stage.CreateInMemory()
        root = _UsdGeom.Xform.Define(stage, f"/{prim_name}")
        stage.SetDefaultPrim(root.GetPrim())

        # OVRTX reliably renders fully authored runtime capsule prims, whereas
        # runtime PointInstancer color updates fell back to gray.
        material_paths: dict[tuple[float, float, float], str] = {}

        for i in range(len(positions)):
            color_index = int(color_indices[i]) if len(color_indices) > i else min(i, len(colors_np) - 1)
            color = colors_np[color_index].astype(np.float32)
            color_key = (float(color[0]), float(color[1]), float(color[2]))

            seg_path = f"/{prim_name}/seg_{i}"
            xform = _UsdGeom.Xform.Define(stage, seg_path)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*positions[i].astype(np.float64).tolist()))
            orient = orientations[i].astype(np.float32)
            xform.AddOrientOp().Set(Gf.Quatf(float(orient[3]), float(orient[0]), float(orient[1]), float(orient[2])))
            xform.AddScaleOp().Set(Gf.Vec3d(*scales[i].astype(np.float64).tolist()))

            capsule = _UsdGeom.Capsule.Define(stage, f"{seg_path}/capsule")
            capsule.GetAxisAttr().Set(_UsdGeom.Tokens.z)
            capsule.GetRadiusAttr().Set(float(width))
            capsule.GetHeightAttr().Set(1.0)
            capsule.GetDisplayColorAttr().Set([Gf.Vec3f(*color.tolist())])

            mat_path = material_paths.get(color_key)
            if mat_path is None:
                mat_path = f"/{prim_name}/Materials/mat_{len(material_paths)}"
                material_paths[color_key] = mat_path
                material = _UsdShade.Material.Define(stage, mat_path)
                surface = _UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
                surface.CreateIdAttr("UsdPreviewSurface")
                surface.CreateInput("diffuseColor", _Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color.tolist()))
                surface.CreateInput("emissiveColor", _Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color.tolist()))
                surface.CreateInput("roughness", _Sdf.ValueTypeNames.Float).Set(1.0)
                surface.CreateInput("metallic", _Sdf.ValueTypeNames.Float).Set(0.0)
                material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

            _UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim())
            _UsdShade.MaterialBindingAPI(capsule).Bind(_UsdShade.Material.Get(stage, mat_path))

        handle = self._rtx.add_usd_layer(stage.GetRootLayer().ExportToString(), path_prefix=target_path)
        self._line_batch_handles[name] = handle
        return True

    @staticmethod
    def _build_line_instance_buffers(
        starts_np: np.ndarray, ends_np: np.ndarray, capacity: int
    ) -> tuple[np.ndarray, np.ndarray]:
        xforms_np = np.zeros((capacity, 7), dtype=np.float32)
        xforms_np[:, 6] = 1.0
        scales_np = np.zeros((capacity, 3), dtype=np.float32)

        if capacity <= 0 or Gf is None:
            return xforms_np, scales_np

        count = min(capacity, len(starts_np), len(ends_np))
        for i in range(count):
            pos0 = np.asarray(starts_np[i], dtype=np.float32)
            pos1 = np.asarray(ends_np[i], dtype=np.float32)
            delta = pos1 - pos0
            height = float(np.linalg.norm(delta))

            xforms_np[i, :3] = 0.5 * (pos0 + pos1)
            if height <= 1.0e-8:
                continue

            direction = delta / height
            rot = Gf.Rotation()
            rot.SetRotateInto(
                Gf.Vec3d(0.0, 0.0, 1.0),
                Gf.Vec3d(float(direction[0]), float(direction[1]), float(direction[2])),
            )
            quat = rot.GetQuat()
            imag = quat.GetImaginary()
            xforms_np[i, 3] = float(imag[0])
            xforms_np[i, 4] = float(imag[1])
            xforms_np[i, 5] = float(imag[2])
            xforms_np[i, 6] = float(quat.GetReal())
            scales_np[i] = (1.0, 1.0, height)

        return xforms_np, scales_np

    def _queue_picking_line_transform(self, starts_np: np.ndarray, ends_np: np.ndarray):
        xforms_np, scales_np = self._build_line_instance_buffers(starts_np, ends_np, capacity=1)
        self._pending_xforms[self._PICKING_LINE_NAME] = (
            wp.array(xforms_np, dtype=wp.transform, device=self.device),
            wp.array(scales_np, dtype=wp.vec3, device=self.device),
        )

    def _hide_picking_line(self):
        self._queue_picking_line_transform(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )

    def _build_line_batch_arrays(
        self, starts, ends, colors, hidden: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        empty_positions = np.zeros((0, 3), dtype=np.float32)
        empty_orientations = np.zeros((0, 4), dtype=np.float16)
        empty_scales = np.zeros((0, 3), dtype=np.float32)
        empty_proto_indices = np.zeros(0, dtype=np.int32)
        empty_ids = np.zeros(0, dtype=np.int64)
        empty_colors = np.zeros((0, 3), dtype=np.float32)
        empty_color_indices = np.zeros(0, dtype=np.int32)

        if hidden or starts is None or ends is None or colors is None or Gf is None:
            return (
                empty_positions,
                empty_orientations,
                empty_scales,
                empty_proto_indices,
                empty_ids,
                empty_colors,
                empty_color_indices,
            )

        starts_np = (
            starts.numpy().astype(np.float32) if isinstance(starts, wp.array) else np.asarray(starts, dtype=np.float32)
        )
        ends_np = ends.numpy().astype(np.float32) if isinstance(ends, wp.array) else np.asarray(ends, dtype=np.float32)
        num_lines = min(len(starts_np), len(ends_np))
        if num_lines == 0:
            return (
                empty_positions,
                empty_orientations,
                empty_scales,
                empty_proto_indices,
                empty_ids,
                empty_colors,
                empty_color_indices,
            )

        colors_np = self._promote_colors_to_array(colors, num_lines)
        colors_np = np.asarray(colors_np, dtype=np.float32)
        if colors_np.ndim == 1:
            if colors_np.shape[0] != 3:
                raise ValueError("Line colors must be an RGB triplet or an array of RGB triplets.")
            colors_np = np.tile(colors_np, (num_lines, 1))
        elif colors_np.shape[0] == 1 and num_lines > 1:
            colors_np = np.repeat(colors_np, num_lines, axis=0)
        elif colors_np.shape[0] != num_lines:
            raise ValueError("Number of line colors must match the number of lines.")

        positions = np.zeros((num_lines, 3), dtype=np.float32)
        orientations = np.zeros((num_lines, 4), dtype=np.float16)
        orientations[:, 3] = np.float16(1.0)
        scales = np.zeros((num_lines, 3), dtype=np.float32)

        for i in range(num_lines):
            pos0 = starts_np[i]
            pos1 = ends_np[i]
            delta = pos1 - pos0
            if float(np.linalg.norm(delta)) <= 1.0e-8:
                positions[i] = 0.5 * (pos0 + pos1)
                continue

            pos, rot, scale = _compute_segment_xform(
                Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])),
                Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])),
            )
            imag = rot.GetImaginary()
            positions[i] = (float(pos[0]), float(pos[1]), float(pos[2]))
            # quath memory layout is imaginary xyz first, then real w.
            orientations[i] = (
                np.float16(imag[0]),
                np.float16(imag[1]),
                np.float16(imag[2]),
                np.float16(rot.GetReal()),
            )
            scales[i] = (float(scale[0]), float(scale[1]), float(scale[2]))

        proto_indices = np.zeros(num_lines, dtype=np.int32)
        ids = np.arange(num_lines, dtype=np.int64)
        color_indices = np.arange(num_lines, dtype=np.int32)
        return positions, orientations, scales, proto_indices, ids, colors_np, color_indices

    @override
    def is_key_down(self, key: str | int) -> bool:
        pyglet = self._pyglet
        if pyglet is None:
            return False

        if isinstance(key, str):
            key = key.lower()
            if len(key) == 1 and key.isalpha():
                key_code = getattr(pyglet.window.key, key.upper(), None)
            elif len(key) == 1 and key.isdigit():
                key_code = getattr(pyglet.window.key, f"_{key}", None)
            else:
                special_keys = {
                    "space": pyglet.window.key.SPACE,
                    "escape": pyglet.window.key.ESCAPE,
                    "esc": pyglet.window.key.ESCAPE,
                    "enter": pyglet.window.key.ENTER,
                    "return": pyglet.window.key.ENTER,
                    "tab": pyglet.window.key.TAB,
                    "shift": pyglet.window.key.LSHIFT,
                    "ctrl": pyglet.window.key.LCTRL,
                    "alt": pyglet.window.key.LALT,
                    "up": pyglet.window.key.UP,
                    "down": pyglet.window.key.DOWN,
                    "left": pyglet.window.key.LEFT,
                    "right": pyglet.window.key.RIGHT,
                    "backspace": pyglet.window.key.BACKSPACE,
                    "delete": pyglet.window.key.DELETE,
                }
                key_code = special_keys.get(key, None)
            if key_code is None:
                return False
        else:
            key_code = key

        return key_code in self._keys_down

    @override
    def log_gizmo(self, name, transform):
        self._gizmo_log[name] = transform

    @override
    def log_state(self, state):
        self._last_state = state
        if self.model is None:
            return

        if self._phase == self._PHASE_BUILD:
            # Build phase: delegate fully to base so USD prims are set up normally.
            super().log_state(state)
        else:
            # Render phase: flat arrays (built at end of build phase) handle all shape
            # transform updates in a single kernel launch — no per-batch work needed here.
            for shapes in self._shape_instances.values():
                shapes.colors_changed = False

            self._log_gaussian_shapes(state)
            self._log_non_shape_state(state)
            self.model_changed = False

        self._ensure_picking_line_primitive()
        self._render_picking_line(state)

    def _render_picking_line(self, state):
        if not self.picking_enabled or self.picking is None or not self.picking.is_picking():
            if self._phase == self._PHASE_RENDER:
                self._hide_picking_line()
            return

        pick_body_idx = self.picking.pick_body.numpy()[0]
        if pick_body_idx < 0:
            if self._phase == self._PHASE_RENDER:
                self._hide_picking_line()
            return

        pick_state = self.picking.pick_state.numpy()
        pick_target = pick_state[0]["picking_target_world"]
        picked_point = pick_state[0]["picked_point_world"]

        body_world = self.model.body_world if self.model is not None else None
        if self.world_offsets is not None and self.world_offsets.shape[0] > 0:
            if body_world is not None:
                body_world_idx = body_world.numpy()[pick_body_idx]
                if body_world_idx >= 0 and body_world_idx < self.world_offsets.shape[0]:
                    world_offset = self.world_offsets.numpy()[body_world_idx]
                    pick_target = pick_target + world_offset
                    picked_point = picked_point + world_offset

        self._queue_picking_line_transform(
            np.asarray([[picked_point[0], picked_point[1], picked_point[2]]], dtype=np.float32),
            np.asarray([[pick_target[0], pick_target[1], pick_target[2]]], dtype=np.float32),
        )

    @override
    def apply_forces(self, state):
        if self.picking_enabled and self.picking is not None:
            self.picking._apply_picking_force(state)

        if self.wind is not None:
            self.wind._apply_wind_force(state)

    @override
    def begin_frame(self, time):
        with wp.ScopedTimer("ViewerRTX::begin_frame", active=PROFILE_ENABLED, use_nvtx=True):
            super().begin_frame(time)
            self._pending_xforms.clear()
            self._pending_mesh_points.clear()
            self._pending_mesh_normals.clear()
            self._pending_line_batches.clear()
            self._pending_point_batches.clear()
            self._gizmo_log = {}

            if self._window and not self._headless:
                try:
                    self._window.switch_to()
                    self._window.dispatch_events()
                except Exception as exc:
                    warnings.warn(
                        f"ViewerRTX: error dispatching window events: {exc}",
                        category=RuntimeWarning,
                        stacklevel=2,
                    )
                    self._should_close = True

            now = perf_counter()
            if self._last_perf_time is not None:
                dt = min(now - self._last_perf_time, 0.1)
                if self.gui:
                    self.gui.update_camera_from_keys(dt, lambda k: k in self._keys_down)
                if self.wind is not None:
                    self.wind.update(dt)
            self._last_perf_time = now

    @override
    def end_frame(self):
        if self._phase == self._PHASE_BUILD:
            self._init_ovrtx()

        with wp.ScopedTimer("ViewerRTX::end_frame", active=PROFILE_ENABLED, use_nvtx=True):
            self._update_ovrtx_camera()
            self._update_ovrtx_transforms()
            self._update_ovrtx_line_batches()
            self._update_ovrtx_point_batches()
            self._update_ovrtx_mesh_points()
            self._render_and_display()

    @override
    def log_mesh(
        self, name, points, indices, normals=None, uvs=None, texture=None, hidden=False, backface_culling=True
    ):
        if self._phase == self._PHASE_BUILD:
            super().log_mesh(name, points, indices, normals, uvs, texture, hidden, backface_culling)
            self._mesh_prim_paths[name] = self._get_path(name)
        elif name in self._mesh_prim_paths:
            pts = (
                points.numpy().astype(np.float32)
                if isinstance(points, wp.array)
                else np.asarray(points, dtype=np.float32)
            )
            self._pending_mesh_points[name] = pts
            if normals is not None:
                self._pending_mesh_normals[name] = (
                    normals.numpy().astype(np.float32)
                    if isinstance(normals, wp.array)
                    else np.asarray(normals, dtype=np.float32)
                )

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        if self._phase == self._PHASE_BUILD:
            super().log_instances(name, mesh, xforms, scales, colors, materials, hidden)
            if xforms is not None:
                count = len(xforms)
                paths = [self._get_path(name) + f"/instance_{i}" for i in range(count)]
                self._instance_prim_paths[name] = paths
        else:
            if xforms is not None:
                if scales is None:
                    scales = wp.ones(len(xforms), dtype=wp.vec3, device=xforms.device)
                self._pending_xforms[name] = (xforms, scales)

    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        if self._phase == self._PHASE_BUILD:
            super().log_lines(name, starts, ends, colors, width, hidden)
            self._line_batch_paths[name] = self._get_path(name)
            self._line_batch_proto_paths[name] = self._get_path(name) + "/capsule"
            self._line_batch_widths[name] = float(width)
            return

        if name not in self._line_batch_paths:
            self._rebuild_runtime_line_batch_layer(name, starts, ends, colors, float(width), bool(hidden))
        elif name in self._line_batch_paths:
            self._pending_line_batches[name] = (starts, ends, colors, float(width), bool(hidden))

    @override
    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        if self._phase == self._PHASE_BUILD:
            if points is None:
                return None

            instancer = self._ensure_point_batch_primitive(name)
            if instancer is None:
                return None

            positions, scales, proto_indices, ids, colors_np, color_indices = self._build_point_batch_arrays(
                points, radii, colors
            )

            instancer.GetPositionsAttr().Set(positions, self._frame_index)
            instancer.GetScalesAttr().Set(scales, self._frame_index)
            instancer.GetProtoIndicesAttr().Set(proto_indices, self._frame_index)
            instancer.CreateIdsAttr().Set(ids, self._frame_index)

            if colors_np is not None and color_indices is not None:
                from pxr import Vt

                display_color = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                display_color.Set(colors_np, self._frame_index)
                display_color.SetIndices(Vt.IntArray(color_indices.tolist()), self._frame_index)
                self._point_batch_colors[name] = np.array(colors_np, copy=True)

            self._point_batch_synced_counts[name] = len(positions)
            instancer.GetVisibilityAttr().Set(
                "inherited" if not hidden and len(positions) > 0 else "invisible",
                self._frame_index,
            )
            return instancer.GetPath()

        if name in self._point_batch_paths:
            self._pending_point_batches[name] = (points, radii, colors, bool(hidden))
            return self._point_batch_paths[name]

    # --------------------------------------------------------- OVRTX updates

    def _update_ovrtx_camera(self):
        if self._rtx is None or not self._camera_dirty:
            return
        with wp.ScopedTimer("ViewerRTX::update_camera", active=PROFILE_ENABLED, use_nvtx=True):
            import ovrtx.math

            mat = self._compute_camera_matrix()
            cam_mat = ovrtx.math.Matrix4d()
            for i in range(4):
                for j in range(4):
                    cam_mat[i][j] = mat[i, j]

            from ovrtx import Semantic

            self._rtx.write_attribute(
                prim_paths=[self._camera_prim_path],
                attribute_name="omni:xform",
                tensor=cam_mat.to_dltensor(),
                semantic=Semantic.XFORM_MAT4x4,
            )
            self._camera_dirty = False

    def _update_ovrtx_transforms(self):
        has_flat_shape_arrays = self._flat_total_shapes > 0
        if not self._transform_binding or (not has_flat_shape_arrays and not self._pending_xforms):
            return
        with wp.ScopedTimer("ViewerRTX::update_transforms", active=PROFILE_ENABLED, use_nvtx=True):
            from ovrtx import Device

            rtx_device = Device.CUDA if self.device.is_cuda else Device.CPU
            with self._transform_binding.map(device=rtx_device) as mapping:
                matrices = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)  # (N, 4, 4) float64

                body_q = self._last_state.body_q if self._last_state is not None else None
                world_offsets = self.world_offsets

                if has_flat_shape_arrays:
                    # Single kernel launch for all shape batches.
                    wp.launch(
                        update_and_write_shape_transforms,
                        dim=self._flat_total_shapes,
                        inputs=[
                            self._flat_shape_xforms,
                            self._flat_shape_parents,
                            body_q,
                            self._flat_shape_worlds,
                            world_offsets,
                            self._flat_shape_scales,
                            self._flat_mat44_offset,
                            matrices,
                        ],
                        device=matrices.device,
                    )

                # Handle any remaining pre-computed transforms (e.g. picking line).
                if self._pending_xforms:
                    offset = 0
                    for name, paths in self._instance_prim_paths.items():
                        count = len(paths)
                        if name in self._pending_xforms:
                            xf, sc = self._pending_xforms[name]
                            n = min(count, len(xf))
                            wp.launch(
                                write_transforms,
                                dim=n,
                                inputs=[xf, sc, offset, matrices],
                                device=matrices.device,
                            )
                        offset += count

                if matrices.device.is_cuda:
                    mapping.unmap(stream=matrices.device.stream.cuda_stream)

    @staticmethod
    def _make_laned_array_dltensor(values_np: np.ndarray, lanes: int):
        """Create a 1D DLTensor with a fixed lane count per element."""
        from ovrtx._src.dlpack import DLTensor

        flat = np.ascontiguousarray(values_np).reshape(-1)
        dl = DLTensor.from_dlpack(flat)
        n = len(flat) // lanes
        dl.dtype.lanes = lanes
        dl.ndim = 1
        shape_arr = (ctypes.c_int64 * 1)(n)
        dl.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
        dl._laned_shape = shape_arr  # prevent GC
        return dl

    def _write_ovrtx_array_attribute(self, prim_path: str, attribute_name: str, values: Any):
        if self._rtx is None:
            return

        if isinstance(values, wp.array):
            # XXX For now OVRTX only supports array writes on CPU
            values = values.numpy()

        self._rtx.write_array_attribute([prim_path], attribute_name, [np.ascontiguousarray(values)])

    @staticmethod
    def _make_point3f_dltensor(points_np):
        """Create a DLTensor with float3 (lanes=3) dtype from an (N,3) float32 array.

        OVRTX Fabric stores 'points' as point3f[] where each element is 12 bytes
        (float32 x 3 lanes). A plain DLTensor.from_dlpack on a (N,3) float32 array
        produces scalar float32 elements (4 bytes), causing an element-size mismatch.
        """
        return ViewerRTX._make_laned_array_dltensor(np.asarray(points_np, dtype=np.float32), lanes=3)

    def _update_ovrtx_mesh_points(self):
        if self._rtx is None or (not self._pending_mesh_points and not self._pending_mesh_normals):
            return
        with wp.ScopedTimer("ViewerRTX::update_mesh_points", active=PROFILE_ENABLED, use_nvtx=True):
            for mesh_name, points_np in self._pending_mesh_points.items():
                prim_path = self._mesh_prim_paths.get(mesh_name)
                if prim_path is None:
                    continue
                dl = self._make_point3f_dltensor(points_np)
                self._rtx.write_array_attribute(
                    prim_paths=[prim_path],
                    attribute_name="points",
                    tensors=[dl],
                )
            for mesh_name, normals_np in self._pending_mesh_normals.items():
                prim_path = self._mesh_prim_paths.get(mesh_name)
                if prim_path is None:
                    continue
                dl = self._make_point3f_dltensor(normals_np)
                self._rtx.write_array_attribute(
                    prim_paths=[prim_path],
                    attribute_name="normals",
                    tensors=[dl],
                )

    def _update_ovrtx_line_batches(self):
        if self._rtx is None or not self._pending_line_batches:
            return

        with wp.ScopedTimer("ViewerRTX::update_line_batches", active=PROFILE_ENABLED, use_nvtx=True):
            for name, (starts, ends, colors, width, hidden) in self._pending_line_batches.items():
                prim_path = self._line_batch_paths.get(name)
                proto_path = self._line_batch_proto_paths.get(name)
                if prim_path is None or proto_path is None:
                    continue

                if self._line_batch_widths.get(name) != width:
                    self._rtx.write_attribute(
                        prim_paths=[proto_path],
                        attribute_name="radius",
                        tensor=np.asarray([width], dtype=np.float32),
                    )
                    self._line_batch_widths[name] = width

                (
                    positions,
                    orientations,
                    scales,
                    proto_indices,
                    ids,
                    colors_np,
                    color_indices,
                ) = self._build_line_batch_arrays(starts, ends, colors, hidden)

                is_visible = not hidden and len(positions) > 0
                self._rtx.write_attribute(
                    prim_paths=[prim_path],
                    attribute_name="visibility",
                    tensor=["inherited" if is_visible else "invisible"],
                )
                if not is_visible:
                    continue

                self._rtx.write_array_attribute(
                    [prim_path], "positions", [self._make_laned_array_dltensor(positions.astype(np.float32), lanes=3)]
                )
                self._rtx.write_array_attribute(
                    [prim_path],
                    "orientations",
                    [self._make_laned_array_dltensor(orientations.astype(np.float16), lanes=4)],
                )
                self._rtx.write_array_attribute(
                    [prim_path], "scales", [self._make_laned_array_dltensor(scales.astype(np.float32), lanes=3)]
                )
                self._rtx.write_array_attribute([prim_path], "protoIndices", [proto_indices])
                self._rtx.write_array_attribute([prim_path], "ids", [ids])
                self._rtx.write_array_attribute(
                    [prim_path],
                    "primvars:displayColor",
                    [self._make_laned_array_dltensor(colors_np.astype(np.float32), lanes=3)],
                )
                self._rtx.write_array_attribute([prim_path], "primvars:displayColor:indices", [color_indices])

    def _update_ovrtx_point_batches(self):
        if self._rtx is None or not self._pending_point_batches:
            return

        with wp.ScopedTimer("ViewerRTX::update_point_batches", active=PROFILE_ENABLED, use_nvtx=True):
            for name, (points, radii, colors, hidden) in self._pending_point_batches.items():
                prim_path = self._point_batch_paths.get(name)
                if prim_path is None:
                    continue

                if points is None:
                    positions = np.zeros((0, 3), dtype=np.float32)
                else:
                    positions = np.asarray(
                        points.numpy() if isinstance(points, wp.array) else points, dtype=np.float32
                    ).reshape((-1, 3))
                count = len(positions)
                self._rtx.write_attribute(
                    prim_paths=[prim_path],
                    attribute_name="visibility",
                    tensor=["inherited" if not hidden and count > 0 else "invisible"],
                )

                if hidden or count == 0:
                    continue

                if count == self._point_batch_synced_counts.get(name, count):
                    self._write_ovrtx_array_attribute(prim_path, "positions", positions)
                    self._point_batch_synced_counts[name] = count
                    continue

                point_colors = colors if colors is not None else self._point_batch_colors.get(name)
                positions, scales, proto_indices, ids, colors_np, color_indices = self._build_point_batch_arrays(
                    points, radii, point_colors
                )

                self._write_ovrtx_array_attribute(prim_path, "positions", positions)
                self._write_ovrtx_array_attribute(prim_path, "scales", scales)
                self._write_ovrtx_array_attribute(prim_path, "protoIndices", proto_indices)
                self._write_ovrtx_array_attribute(prim_path, "ids", ids)

                if colors_np is not None and color_indices is not None:
                    self._write_ovrtx_array_attribute(prim_path, "primvars:displayColor", colors_np)
                    self._write_ovrtx_array_attribute(prim_path, "primvars:displayColor:indices", color_indices)
                    self._point_batch_colors[name] = np.array(colors_np, copy=True)

                self._point_batch_synced_counts[name] = count

    # ------------------------------------------------------- render + display

    def _render_and_display(self):
        if self._rtx is None or self._should_close:
            return

        with wp.ScopedTimer("ViewerRTX::render_and_display", active=PROFILE_ENABLED, use_nvtx=True):
            from ovrtx import Device

            self._render_products = None

            if self._async:
                # wait for async rendering to complete
                with wp.ScopedTimer("ViewerRTX::rtx_wait", active=PROFILE_ENABLED, use_nvtx=True):
                    if self._render_result is not None:
                        self._render_products = self._render_result.wait()
            else:
                # render synchronously
                with wp.ScopedTimer("ViewerRTX::rtx_step", active=PROFILE_ENABLED, use_nvtx=True):
                    self._render_products = self._rtx.step(
                        render_products={self._render_product_path},
                        delta_time=1.0 / self.fps,
                    )

            # blit to window if not headless
            if self._render_products is not None and self._window is not None and self._window.context is not None:
                for _pname, product in self._render_products.items():
                    for frame in product.frames:
                        if "LdrColor" in frame.render_vars:
                            with wp.ScopedTimer("ViewerRTX::fb_map", active=PROFILE_ENABLED, use_nvtx=True):
                                with frame.render_vars["LdrColor"].map(device=Device.CUDA) as mapping:
                                    pixels = wp.from_dlpack(mapping.tensor, dtype=wp.vec4ub)
                                    with wp.ScopedTimer(
                                        "ViewerRTX::blit_to_window", active=PROFILE_ENABLED, use_nvtx=True
                                    ):
                                        self._blit_to_window(pixels)
                                    mapping.unmap(stream=pixels.device.stream.cuda_stream)

            if self._async:
                # kick off next async rendering frame
                with wp.ScopedTimer("ViewerRTX::rtx_step_async", active=PROFILE_ENABLED, use_nvtx=True):
                    self._render_result = self._rtx.step_async(
                        render_products={self._render_product_path},
                        delta_time=1.0 / self.fps,
                    )

    def _blit_to_window(self, pixels: wp.array | wp.Texture2D):
        """Upload *pixels* to a GL texture and draw a fullscreen triangle (GPU sRGB + flip)."""
        gl = self._pyglet_gl

        with wp.ScopedTimer("ViewerRTX::gl_tex_copy", active=PROFILE_ENABLED, use_nvtx=True):
            # copy OVRTX output to OpenGL texture
            frame_tex = self._tex_resource.map()
            frame_tex.copy_from(pixels)
            self._tex_resource.unmap()

        self._window.switch_to()
        fb_w, fb_h = self._window.get_framebuffer_size()

        # Compute a letterbox viewport that preserves the OVRTX render aspect ratio.
        render_aspect = self.camera.width / max(self.camera.height, 1)
        window_aspect = fb_w / max(fb_h, 1)
        if window_aspect >= render_aspect:
            # Window is wider than render — pillarbox (black bars left/right)
            vp_h = fb_h
            vp_w = int(fb_h * render_aspect)
            vp_x = (fb_w - vp_w) // 2
            vp_y = 0
        else:
            # Window is taller than render — letterbox (black bars top/bottom)
            vp_w = fb_w
            vp_h = int(fb_w / render_aspect)
            vp_x = 0
            vp_y = (fb_h - vp_h) // 2

        with wp.ScopedTimer("ViewerRTX::gl_draw", active=PROFILE_ENABLED, use_nvtx=True):
            # Clear the full window to black, then draw into the letterbox region
            gl.glViewport(0, 0, fb_w, fb_h)
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glViewport(vp_x, vp_y, vp_w, vp_h)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._gl_texture)
            gl.glUseProgram(self._gl_program)
            gl.glBindVertexArray(self._gl_vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            # Restore full viewport for ImGui (which spans the entire window)
            gl.glViewport(0, 0, fb_w, fb_h)

        if self.gui:
            with wp.ScopedTimer("ViewerRTX::gui_render", active=PROFILE_ENABLED, use_nvtx=True):
                self.gui.render_frame(update_fps=True)

        with wp.ScopedTimer("ViewerRTX::swap_buffers", active=PROFILE_ENABLED, use_nvtx=True):
            self._window.flip()

    def _capture_screenshot_pixels(self) -> np.ndarray:
        if self._render_products is not None:
            products = self._render_products
        elif self._render_result is not None:
            products = self._render_result.wait()
        else:
            raise RuntimeError("save_screenshot() requires at least one completed render frame")

        from ovrtx import Device

        for _pname, product in products.items():
            for frame in product.frames:
                if "LdrColor" in frame.render_vars:
                    with frame.render_vars["LdrColor"].map(device=Device.CPU) as mapping:
                        pixels = np.array(np.from_dlpack(mapping.tensor), copy=True)
                    return pixels

        raise RuntimeError("save_screenshot() could not find the LdrColor render output")

    def save_screenshot(self, path: str) -> None:
        """Save the last rendered frame to an image file.

        The file format is inferred from the extension (e.g. ``.png``, ``.jpg``).
        Call this after at least one completed frame has been rendered (e.g.
        after the simulation loop). Works in headless mode.
        """
        from PIL import Image

        pixels = self._capture_screenshot_pixels()
        pil_img = Image.fromarray(pixels)
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".JPG", ".jpeg", ".JPEG"}:
            pil_img = pil_img.convert("RGB")
            pil_img.save(path, quality=92)
        else:
            pil_img.save(path)

    # ----------------------------------------------------------- viewer API

    def register_ui_callback(self, callback, position="side"):
        if position not in self._ui_callbacks:
            valid = list(self._ui_callbacks.keys())
            raise ValueError(f"Invalid position {position!r}. Valid: {valid}")
        self._ui_callbacks[position].append(callback)

    def clear_model(self):
        self._ui_callbacks["side"] = []
        self._ui_callbacks["free"] = []
        self.picking = None
        self.wind = None
        if self._rtx is not None:
            for handle in self._line_batch_handles.values():
                self._rtx.remove_usd(handle)
        self._line_batch_paths = {}
        self._line_batch_proto_paths = {}
        self._line_batch_widths = {}
        self._line_batch_handles = {}
        self._pending_line_batches = {}
        self._pending_point_batches = {}
        self._point_batch_paths = {}
        self._point_batch_colors = {}
        self._point_batch_synced_counts = {}
        self._last_state = None
        self._last_control = None
        super().clear_model()

    def _ui_populate_rendering_panel(self, imgui):
        """Render RTX-specific items inside the Rendering Options panel section."""
        _changed, self._async = imgui.checkbox("Asynchronous Rendering", self._async)

    @override
    def is_paused(self) -> bool:
        return self._paused

    @override
    def is_running(self) -> bool:
        if self._should_close:
            return False
        # In headless mode num_frames acts as a hard stop (e.g. for CI tests).
        # In windowed mode the user closes the window themselves; ignoring
        # num_frames prevents the window from disappearing unexpectedly.
        if self._headless and self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    @override
    def close(self):
        # wait for async rendering results before closing
        if self._render_result is not None:
            self._render_result.wait()
            self._render_result = None

        # release render products
        if self._render_products is not None:
            self._render_products = None

        # release transform binding
        if self._transform_binding is not None:
            self._transform_binding.unbind()
            self._transform_binding = None

        # release ovrtx renderer
        self._rtx = None

        if self.ui:
            self.ui.shutdown()

        if self._window is not None:
            if not self._headless:
                try:
                    self._pyglet_app.event_loop.dispatch_event("on_exit")
                    self._pyglet_app.platform_event_loop.stop()
                except Exception:
                    pass
            self._window.close()
            self._window = None

        if self._tmp_usd_path and os.path.exists(self._tmp_usd_path):
            try:
                os.unlink(self._tmp_usd_path)
            except OSError:
                pass
            self._tmp_usd_path = ""
