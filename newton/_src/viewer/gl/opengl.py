# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

import ctypes
import io
import os
import sys

import numpy as np
import warp as wp

from newton.utils import create_sphere_mesh

from ...utils.mesh import compute_vertex_normals
from ...utils.texture import normalize_texture
from .shaders import (
    FrameShader,
    ShaderLine,
    ShaderShape,
    ShaderSky,
    ShadowShader,
)

ENABLE_CUDA_INTEROP = False
ENABLE_GL_CHECKS = False

wp.set_module_options({"enable_backward": False})


def check_gl_error():
    if not ENABLE_GL_CHECKS:
        return

    from pyglet import gl

    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        error_strings = {
            gl.GL_INVALID_ENUM: "GL_INVALID_ENUM",
            gl.GL_INVALID_VALUE: "GL_INVALID_VALUE",
            gl.GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
            gl.GL_INVALID_FRAMEBUFFER_OPERATION: "GL_INVALID_FRAMEBUFFER_OPERATION",
            gl.GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY",
        }
        error_name = error_strings.get(error, f"Unknown error code: {error}")

        import traceback  # noqa: PLC0415

        stack = traceback.format_stack()
        print(f"OpenGL error: {error_name} ({error:#x})")
        print(f"Called from: {''.join(stack[-2:-1])}")


def _upload_texture_from_file(gl, texture_image: np.ndarray) -> int:
    image = normalize_texture(
        texture_image,
        flip_vertical=True,
        require_channels=True,
        scale_unit_range=True,
    )
    if image is None:
        return 0
    channels = image.shape[2]
    if image.size == 0:
        return 0
    max_size = gl.GLint()
    gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE, max_size)
    if image.shape[0] > max_size.value or image.shape[1] > max_size.value:
        return 0
    texture_id = gl.GLuint()
    gl.glGenTextures(1, texture_id)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    format_enum = gl.GL_RGBA if channels == 4 else gl.GL_RGB
    row_stride = image.shape[1] * channels
    prev_alignment = None
    if row_stride % 4 != 0:
        prev_alignment = gl.GLint()
        gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT, prev_alignment)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        format_enum,
        image.shape[1],
        image.shape[0],
        0,
        format_enum,
        gl.GL_UNSIGNED_BYTE,
        image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    )
    if prev_alignment is not None:
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, prev_alignment.value)
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture_id


@wp.struct
class RenderVertex:
    pos: wp.vec3
    normal: wp.vec3
    uv: wp.vec2


@wp.struct
class LineVertex:
    pos: wp.vec3
    color: wp.vec3


@wp.kernel
def fill_vertex_data(
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    uvs: wp.array(dtype=wp.vec2),
    vertices: wp.array(dtype=RenderVertex),
):
    tid = wp.tid()

    vertices[tid].pos = points[tid]

    if normals:
        vertices[tid].normal = normals[tid]

    if uvs:
        vertices[tid].uv = uvs[tid]


@wp.kernel
def fill_line_vertex_data(
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
    vertices: wp.array(dtype=LineVertex),
):
    tid = wp.tid()

    # Each line has 2 vertices (begin and end)
    vertex_idx = tid * 2

    # First vertex (line begin)
    vertices[vertex_idx].pos = starts[tid]
    vertices[vertex_idx].color = colors[tid]

    # Second vertex (line end)
    vertices[vertex_idx + 1].pos = ends[tid]
    vertices[vertex_idx + 1].color = colors[tid]


class MeshGL:
    """Encapsulates mesh data and OpenGL buffers for a shape."""

    def __init__(self, num_points, num_indices, device, hidden=False, backface_culling=True):
        """Initialize mesh data with vertices and indices."""
        gl = RendererGL.gl

        self.num_points = num_points
        self.num_indices = num_indices

        # Store references to input buffers and rendering data
        self.device = device
        self.hidden = hidden
        self.backface_culling = backface_culling

        self.vertices = wp.zeros(num_points, dtype=RenderVertex, device=self.device)
        self.indices = None
        self.normals = None  # scratch buffer used during normal recomputation
        self.texture_id = None

        # Set up vertex attributes in the packed format the shaders expect
        self.vertex_byte_size = 12 + 12 + 8
        self.index_byte_size = 4

        self.vbo_size = self.vertex_byte_size * num_points
        self.ebo_size = self.index_byte_size * num_indices

        # Create OpenGL buffers
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.GLuint()
        gl.glGenBuffers(1, self.vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vbo_size, None, gl.GL_STATIC_DRAW)

        self.ebo = gl.GLuint()
        gl.glGenBuffers(1, self.ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_size, None, gl.GL_STATIC_DRAW)

        # positions (location 0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        # normals (location 1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)

        # uv coordinates (location 2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(6 * 4))
        gl.glEnableVertexAttribArray(2)

        # set constant instance transform
        gl.glDisableVertexAttribArray(3)
        gl.glDisableVertexAttribArray(4)
        gl.glDisableVertexAttribArray(5)
        gl.glDisableVertexAttribArray(6)
        gl.glDisableVertexAttribArray(7)
        gl.glDisableVertexAttribArray(8)
        gl.glDisableVertexAttribArray(9)

        #   column 0  (1,0,0,0)
        gl.glVertexAttrib4f(3, 1.0, 0.0, 0.0, 0.0)
        #   column 1  (0,1,0,0)
        gl.glVertexAttrib4f(4, 0.0, 1.0, 0.0, 0.0)
        #   column 2  (0,0,1,0)
        gl.glVertexAttrib4f(5, 0.0, 0.0, 1.0, 0.0)
        #   column 3  (0,0,0,1)
        gl.glVertexAttrib4f(6, 0.0, 0.0, 0.0, 1.0)

        # albedo
        gl.glVertexAttrib3f(7, 0.7, 0.5, 0.3)
        # material, roughness, metallic, checker, texture_enable
        gl.glVertexAttrib4f(8, 0.5, 0.0, 0.0, 0.0)

        gl.glBindVertexArray(0)

        # Create CUDA-GL interop buffer for efficient updates
        if ENABLE_CUDA_INTEROP and self.device.is_cuda:
            self.vertex_cuda_buffer = wp.RegisteredGLBuffer(int(self.vbo.value), self.device)
        else:
            self.vertex_cuda_buffer = None
        self._points = None

    def destroy(self):
        """Clean up OpenGL resources."""
        gl = RendererGL.gl
        try:
            if hasattr(self, "vao"):
                gl.glDeleteVertexArrays(1, self.vao)
            if hasattr(self, "vbo"):
                gl.glDeleteBuffers(1, self.vbo)
            if hasattr(self, "ebo"):
                gl.glDeleteBuffers(1, self.ebo)
            if hasattr(self, "texture_id") and self.texture_id is not None:
                gl.glDeleteTextures(1, self.texture_id)
        except Exception:
            # Ignore any errors if the GL context has already been torn down
            pass

    def update(self, points, indices, normals, uvs, texture=None):
        """Update vertex positions in the VBO.

        Args:
            points: New point positions (warp array or numpy array)
            scale: Scaling factor for positions
        """
        gl = RendererGL.gl

        if len(points) != len(self.vertices):
            raise RuntimeError("Number of points does not match")

        self._points = points

        # only update indices the first time (no topology changes)
        if self.indices is None:
            self.indices = wp.clone(indices).view(dtype=wp.uint32)
            self.num_indices = int(len(self.indices))

            host_indices = self.indices.numpy()
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferData(
                gl.GL_ELEMENT_ARRAY_BUFFER, host_indices.nbytes, host_indices.ctypes.data, gl.GL_STATIC_DRAW
            )

        # If normals are missing, compute them before packing vertex data.
        if points is not None and normals is None:
            self.recompute_normals()
            normals = self.normals

        # update gfx vertices
        wp.launch(
            fill_vertex_data,
            dim=len(self.vertices),
            inputs=[points, normals, uvs],
            outputs=[self.vertices],
            device=self.device,
        )

        # upload vertices to GL
        if ENABLE_CUDA_INTEROP and self.vertices.device.is_cuda:
            # upload points via CUDA if possible
            vbo_vertices = self.vertex_cuda_buffer.map(dtype=RenderVertex, shape=self.vertices.shape)
            wp.copy(vbo_vertices, self.vertices)
            self.vertex_cuda_buffer.unmap()

        else:
            host_vertices = self.vertices.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_vertices.nbytes, host_vertices.ctypes.data, gl.GL_STATIC_DRAW)

        self.update_texture(texture)

    def recompute_normals(self):
        if self._points is None or self.indices is None:
            return
        self.normals = compute_vertex_normals(
            self._points,
            self.indices,
            normals=self.normals,
            device=self.device,
        )

    def update_texture(self, texture=None):
        gl = RendererGL.gl
        texture_image = None
        if texture is not None:
            from ...utils.texture import load_texture  # noqa: PLC0415

            texture_image = load_texture(texture)

        if texture_image is None:
            if self.texture_id is not None:
                try:
                    gl.glDeleteTextures(1, self.texture_id)
                except Exception:
                    pass
                self.texture_id = None
            return

        if self.texture_id is not None:
            try:
                gl.glDeleteTextures(1, self.texture_id)
            except Exception:
                pass
            self.texture_id = None

        texture_id = _upload_texture_from_file(gl, texture_image)
        if not texture_id:
            return
        self.texture_id = texture_id

    def render(self):
        if not self.hidden:
            gl = RendererGL.gl

            if self.backface_culling:
                gl.glEnable(gl.GL_CULL_FACE)
            else:
                gl.glDisable(gl.GL_CULL_FACE)

            gl.glActiveTexture(gl.GL_TEXTURE1)
            if self.texture_id is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            else:
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            gl.glBindVertexArray(self.vao)
            gl.glDrawElements(gl.GL_TRIANGLES, self.num_indices, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)


class LinesGL:
    """Encapsulates line data and OpenGL buffers for line rendering."""

    def __init__(self, max_lines, device, hidden=False):
        """Initialize line data with the specified maximum number of lines.

        Args:
            max_lines: Maximum number of lines that can be rendered
            device: Warp device to use
            hidden: Whether the lines are initially hidden
        """
        gl = RendererGL.gl

        self.max_lines = max_lines
        self.max_vertices = max_lines * 2  # Each line has 2 vertices
        self.num_lines = 0  # Current number of active lines to render

        # Store references to input buffers and rendering data
        self.device = device
        self.hidden = hidden

        self.vertices = wp.zeros(self.max_vertices, dtype=LineVertex, device=self.device)

        # Set up vertex attributes for lines (position + color)
        self.vertex_byte_size = 12 + 12  # 3 floats for pos + 3 floats for color
        self.vbo_size = self.vertex_byte_size * self.max_vertices

        # Create OpenGL buffers
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, self.vao)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.GLuint()
        gl.glGenBuffers(1, self.vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vbo_size, None, gl.GL_DYNAMIC_DRAW)

        # positions (location 0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        # colors (location 1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertex_byte_size, ctypes.c_void_p(3 * 4))
        gl.glEnableVertexAttribArray(1)

        gl.glBindVertexArray(0)

        # Create CUDA-GL interop buffer for efficient updates
        if ENABLE_CUDA_INTEROP and self.device.is_cuda:
            self.vertex_cuda_buffer = wp.RegisteredGLBuffer(int(self.vbo.value), self.device)
        else:
            self.vertex_cuda_buffer = None

    def destroy(self):
        """Clean up OpenGL resources."""
        gl = RendererGL.gl
        try:
            if hasattr(self, "vao"):
                gl.glDeleteVertexArrays(1, self.vao)
            if hasattr(self, "vbo"):
                gl.glDeleteBuffers(1, self.vbo)
        except Exception:
            # Ignore any errors if the GL context has already been torn down
            pass

    def update(self, starts, ends, colors):
        """Update line data in the VBO.

        Args:
            starts: Array of line start positions (warp array of vec3) or None
            ends: Array of line end positions (warp array of vec3) or None
            colors: Array of line colors (warp array of vec3) or None
        """
        gl = RendererGL.gl

        # Handle None values by setting line count to zero
        if starts is None or ends is None or colors is None:
            self.num_lines = 0
            return

        # Update current line count
        self.num_lines = len(starts)

        if self.num_lines > self.max_lines:
            raise RuntimeError(f"Number of lines ({self.num_lines}) exceeds maximum ({self.max_lines})")
        if len(ends) != self.num_lines:
            raise RuntimeError("Number of line ends does not match line begins")
        if len(colors) != self.num_lines:
            raise RuntimeError("Number of line colors does not match line begins")

        # Only update vertex data if we have lines to render
        if self.num_lines > 0:
            # Update line vertex data using the kernel
            wp.launch(
                fill_line_vertex_data,
                dim=self.num_lines,
                inputs=[starts, ends, colors],
                outputs=[self.vertices],
                device=self.device,
            )

        # Upload vertices to GL
        if ENABLE_CUDA_INTEROP and self.vertices.device.is_cuda:
            # Upload points via CUDA if possible
            vbo_vertices = self.vertex_cuda_buffer.map(dtype=LineVertex, shape=self.vertices.shape)
            wp.copy(vbo_vertices, self.vertices)
            self.vertex_cuda_buffer.unmap()
        else:
            host_vertices = self.vertices.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_vertices.nbytes, host_vertices.ctypes.data, gl.GL_DYNAMIC_DRAW)

    def render(self):
        if not self.hidden and self.num_lines > 0:
            gl = RendererGL.gl

            gl.glDisable(gl.GL_CULL_FACE)  # Lines don't need culling

            gl.glBindVertexArray(self.vao)
            # Only render vertices for the current number of lines
            current_vertices = self.num_lines * 2
            gl.glDrawArrays(gl.GL_LINES, 0, current_vertices)
            gl.glBindVertexArray(0)


@wp.kernel
def update_vbo_transforms(
    instance_transforms: wp.array(dtype=wp.transform),
    instance_scalings: wp.array(dtype=wp.vec3),
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    """Update VBO with simple instance transformation matrices."""
    tid = wp.tid()

    # Get transform and scaling
    transform = instance_transforms[tid]

    if instance_scalings:
        s = instance_scalings[tid]
    else:
        s = wp.vec3(1.0, 1.0, 1.0)

    # Extract position and rotation
    p = wp.transform_get_translation(transform)
    q = wp.transform_get_rotation(transform)

    # Build rotation matrix
    R = wp.quat_to_matrix(q)

    # Apply scaling
    vbo_transforms[tid] = wp.mat44(
        R[0, 0] * s[0],
        R[1, 0] * s[0],
        R[2, 0] * s[0],
        0.0,
        R[0, 1] * s[1],
        R[1, 1] * s[1],
        R[2, 1] * s[1],
        0.0,
        R[0, 2] * s[2],
        R[1, 2] * s[2],
        R[2, 2] * s[2],
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


@wp.kernel
def update_vbo_transforms_from_points(
    points: wp.array(dtype=wp.vec3),
    widths: wp.array(dtype=wp.float32),
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    """Update VBO with simple instance transformation matrices."""
    tid = wp.tid()

    # Get transform and scaling
    p = points[tid]

    if widths:
        s = widths[tid]
    else:
        s = 1.0

    # Build rotation matrix
    R = wp.identity(n=3, dtype=wp.float32)

    # Apply scaling
    vbo_transforms[tid] = wp.mat44(
        R[0, 0] * s,
        R[1, 0] * s,
        R[2, 0] * s,
        0.0,
        R[0, 1] * s,
        R[1, 1] * s,
        R[2, 1] * s,
        0.0,
        R[0, 2] * s,
        R[1, 2] * s,
        R[2, 2] * s,
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


class MeshInstancerGL:
    """
    Handles instanced rendering for a mesh.
    Note the vertices must be in the 8-dimensional format:
        [3D point, 3D normal, UV texture coordinates]
    """

    def __init__(self, num_instances, mesh):
        self.mesh = mesh
        self.device = mesh.device
        self.hidden = False
        self.instance_transform_buffer = None
        self.instance_color_buffer = None
        self.instance_material_buffer = None

        self.instance_transform_cuda_buffer = None

        self.allocate(num_instances)
        self.active_instances = num_instances

    def __del__(self):
        gl = RendererGL.gl

        if self.instance_transform_cuda_buffer is not None:
            try:
                gl.glDeleteBuffers(1, self.instance_transform_cuda_buffer)
            except Exception:
                # Ignore any errors (e.g., context already destroyed)
                pass

        if hasattr(self, "vao") and self.vao is not None:
            try:
                gl.glDeleteVertexArrays(1, self.vao)
                gl.glDeleteBuffers(1, self.instance_transform_buffer)
                gl.glDeleteBuffers(1, self.instance_color_buffer)
                gl.glDeleteBuffers(1, self.instance_material_buffer)
            except Exception:
                # Ignore any errors during interpreter shutdown
                pass

    def allocate(self, num_instances):
        gl = RendererGL.gl

        self.world_xforms = wp.zeros(num_instances, dtype=wp.mat44, device=self.device)

        self.vao = gl.GLuint()
        self.instance_transform_buffer = gl.GLuint()
        self.instance_color_buffer = gl.GLuint()
        self.instance_material_buffer = gl.GLuint()
        self.num_instances = num_instances

        gl.glGenVertexArrays(1, self.vao)
        gl.glBindVertexArray(self.vao)

        # -------------------------
        # index buffer

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.mesh.ebo)

        # ------------------------
        # mesh buffers

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.mesh.vbo)

        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.mesh.vertex_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(
            1,
            3,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.mesh.vertex_byte_size,
            ctypes.c_void_p(3 * 4),
        )
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(
            2,
            2,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            self.mesh.vertex_byte_size,
            ctypes.c_void_p(6 * 4),
        )
        gl.glEnableVertexAttribArray(2)

        self.transform_byte_size = 16 * 4  # sizeof(mat44)
        self.color_byte_size = 3 * 4  # sizeof(vec3)
        self.material_byte_size = 4 * 4  # sizeof(vec4)

        self.instance_transform_buffer_size = self.transform_byte_size * self.num_instances
        self.instance_color_buffer_size = self.color_byte_size * self.num_instances
        self.instance_material_buffer_size = self.material_byte_size * self.num_instances

        # ------------------------
        # transform buffer

        gl.glGenBuffers(1, self.instance_transform_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.instance_transform_buffer_size, None, gl.GL_DYNAMIC_DRAW)

        # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
        for i in range(4):
            gl.glVertexAttribPointer(
                3 + i, 4, gl.GL_FLOAT, gl.GL_FALSE, self.transform_byte_size, ctypes.c_void_p(i * 16)
            )
            gl.glEnableVertexAttribArray(3 + i)
            gl.glVertexAttribDivisor(3 + i, 1)

        # ------------------------
        # colors

        gl.glGenBuffers(1, self.instance_color_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.instance_color_buffer_size, None, gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(7, 3, gl.GL_FLOAT, gl.GL_FALSE, self.color_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(7)
        gl.glVertexAttribDivisor(7, 1)

        # ------------------------
        # materials buffer
        host_materials = np.zeros(self.num_instances * 4, dtype=np.float32)

        gl.glGenBuffers(1, self.instance_material_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_material_buffer)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, self.instance_material_buffer_size, host_materials.ctypes.data, gl.GL_STATIC_DRAW
        )

        gl.glVertexAttribPointer(8, 4, gl.GL_FLOAT, gl.GL_FALSE, self.material_byte_size, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(8)
        gl.glVertexAttribDivisor(8, 1)

        gl.glBindVertexArray(0)

        # Create CUDA buffer for instance transforms
        if ENABLE_CUDA_INTEROP and self.device.is_cuda:
            self._instance_transform_cuda_buffer = wp.RegisteredGLBuffer(
                int(self.instance_transform_buffer.value), self.device, flags=wp.RegisteredGLBuffer.WRITE_DISCARD
            )
        else:
            self._instance_transform_cuda_buffer = None

    def update_from_transforms(
        self,
        transforms: wp.array = None,
        scalings: wp.array = None,
        colors: wp.array = None,
        materials: wp.array = None,
    ):
        if transforms is None:
            active_count = 0
        else:
            active_count = len(transforms)

            if active_count > self.num_instances:
                raise ValueError(
                    f"Active instance count ({active_count}) exceeds allocated capacity ({self.num_instances})."
                )
            if scalings is not None and len(scalings) != active_count:
                raise ValueError("Number of scalings must match number of transforms")

        if active_count > 0:
            wp.launch(
                update_vbo_transforms,
                dim=active_count,
                inputs=[
                    transforms,
                    scalings,
                ],
                outputs=[
                    self.world_xforms,
                ],
                device=self.device,
                record_tape=False,
            )

        self.active_instances = active_count
        # Upload the full buffer; only the first `active_instances` rows are rendered
        self._update_vbo(self.world_xforms, colors, materials)

    # helper to update instance transforms from points
    def update_from_points(self, points, widths, colors):
        if points is None:
            active = 0
        else:
            active = len(points)

        if active > self.num_instances:
            raise ValueError("Active point count exceeds allocated capacity. Reallocate before updating.")

        self.active_instances = active

        if self.active_instances > 0 and (points is not None or widths is not None):
            wp.launch(
                update_vbo_transforms_from_points,
                dim=self.active_instances,
                inputs=[
                    points,
                    widths,
                ],
                outputs=[
                    self.world_xforms,
                ],
                device=self.device,
                record_tape=False,
            )

        self._update_vbo(self.world_xforms, colors, None)

    # upload to vbo
    def _update_vbo(self, xforms, colors, materials):
        gl = RendererGL.gl

        if ENABLE_CUDA_INTEROP and self.device.is_cuda:
            vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self.num_instances,))
            wp.copy(vbo_transforms, xforms)
            self._instance_transform_cuda_buffer.unmap()
        else:
            host_transforms = xforms.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_transforms.nbytes, host_transforms.ctypes.data, gl.GL_DYNAMIC_DRAW)

        # update other properties through CPU for now
        if colors is not None:
            host_colors = colors.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_colors.nbytes, host_colors.ctypes.data, gl.GL_STATIC_DRAW)

        if materials is not None:
            host_materials = materials.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_material_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_materials.nbytes, host_materials.ctypes.data, gl.GL_STATIC_DRAW)

    def update_from_pinned(self, host_transforms_np, count, colors=None, materials=None):
        """Upload pre-computed mat44 transforms from pinned host memory to GL.

        Args:
            host_transforms_np: Numpy array slice of mat44 transforms.
            count: Number of active instances.
            colors: Optional wp.array of per-instance colors.
            materials: Optional wp.array of per-instance materials.
        """
        gl = RendererGL.gl
        if count > self.num_instances:
            raise ValueError(f"Active instance count ({count}) exceeds allocated capacity ({self.num_instances}).")
        self.active_instances = count
        if count > 0:
            nbytes = count * self.transform_byte_size
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_buffer)
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, nbytes, host_transforms_np.ctypes.data)
        if colors is not None:
            host_colors = colors.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_colors.nbytes, host_colors.ctypes.data, gl.GL_STATIC_DRAW)
        if materials is not None:
            host_materials = materials.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_material_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_materials.nbytes, host_materials.ctypes.data, gl.GL_STATIC_DRAW)

    def render(self):
        gl = RendererGL.gl

        if self.hidden:
            return

        if self.mesh.backface_culling:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        if self.mesh.texture_id is not None:
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.mesh.texture_id)
        else:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        gl.glBindVertexArray(self.vao)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES, self.mesh.num_indices, gl.GL_UNSIGNED_INT, None, self.active_instances
        )
        gl.glBindVertexArray(0)


class RendererGL:
    gl = None  # Class-level variable to hold the imported module

    @classmethod
    def initialize_gl(cls):
        if cls.gl is None:  # Only import if not already imported
            from pyglet import gl

            cls.gl = gl

    def __init__(self, title="Newton", screen_width=1920, screen_height=1080, vsync=True, headless=None, device=None):
        self.draw_sky = True
        self.draw_fps = True
        self.draw_shadows = True
        self.draw_wireframe = False

        self.background_color = (68.0 / 255.0, 161.0 / 255.0, 255.0 / 255.0)

        self.sky_upper = self.background_color
        self.sky_lower = (40.0 / 255.0, 44.0 / 255.0, 55.0 / 255.0)

        try:
            import pyglet

            # disable error checking for performance
            pyglet.options["debug_gl"] = False

            # try imports
            from pyglet.graphics.shader import Shader, ShaderProgram  # noqa: F401
            from pyglet.math import Vec3 as PyVec3  # noqa: F401

            RendererGL.initialize_gl()
            gl = RendererGL.gl
        except ImportError as e:
            raise Exception("OpenGLRenderer requires pyglet (version >= 2.0) to be installed.") from e

        self._title = title

        try:
            # try to enable MSAA
            config = pyglet.gl.Config(sample_buffers=1, samples=8, double_buffer=True)
            self.window = pyglet.window.Window(
                width=screen_width,
                height=screen_height,
                caption=title,
                resizable=True,
                vsync=vsync,
                visible=not headless,
                config=config,
            )
            gl.glEnable(gl.GL_MULTISAMPLE)
            # remember sample count for later (e.g., resolving FBO)
            self.msaa_samples = 4
        except pyglet.window.NoSuchConfigException:
            print("Warning: Could not get MSAA config, falling back to non-AA.")
            self.window = pyglet.window.Window(
                width=screen_width,
                height=screen_height,
                caption=title,
                resizable=True,
                vsync=vsync,
                visible=not headless,
            )
            self.msaa_samples = 0

        self._set_icon()

        if headless is None:
            self.headless = pyglet.options.get("headless", False)
        else:
            self.headless = headless
        self.app = pyglet.app

        # making window current opengl rendering context
        self._make_current()

        self._screen_width, self._screen_height = self.window.get_framebuffer_size()

        self._camera_speed = 0.04
        self._last_x, self._last_y = self._screen_width // 2, self._screen_height // 2
        self._key_callbacks = []
        self._key_release_callbacks = []

        self._env_texture = None
        self._env_intensity = 1.0
        self._env_path = None
        self._env_texture_obj = None

        default_env = os.path.join(os.path.dirname(__file__), "newton_envmap.jpg")
        if os.path.exists(default_env):
            self._env_path = default_env
        self._mouse_drag_callbacks = []
        self._mouse_press_callbacks = []
        self._mouse_release_callbacks = []
        self._mouse_motion_callbacks = []
        self._mouse_scroll_callbacks = []
        self._resize_callbacks = []

        # Initialize device and shape lookup
        self._device = device if device is not None else wp.get_device()
        self._shape_lookup = {}

        self._shadow_fbo = None
        self._shadow_texture = None
        self._shadow_shader = None
        self._shadow_width = 4096
        self._shadow_height = 4096

        self._frame_texture = None
        self._frame_depth_texture = None
        self._frame_fbo = None
        self._frame_pbo = None

        self._sun_direction = np.array((0.2, -0.3, 0.8))
        self._sun_direction /= np.linalg.norm(self._sun_direction)

        self._light_color = (1.0, 1.0, 1.0)

        check_gl_error()

        if not headless:
            # set up our own event handling so we can synchronously render frames
            # by calling update() in a loop
            from pyglet.window import Window

            Window._enable_event_queue = False

            self.window.dispatch_pending_events()

            platform_event_loop = self.app.platform_event_loop
            platform_event_loop.start()

            # start event loop
            # self.app.event_loop.dispatch_event("on_enter")

        # create frame buffer for rendering to a texture
        self._setup_shadow_buffer()
        self._setup_frame_buffer()
        self._setup_sky_mesh()
        self._setup_frame_mesh()

        self._shadow_shader = ShadowShader(gl)
        self._shape_shader = ShaderShape(gl)
        self._frame_shader = FrameShader(gl)
        self._sky_shader = ShaderSky(gl)
        self._line_shader = ShaderLine(gl)

        if not headless:
            self._setup_window_callbacks()

    def update(self):
        self._make_current()

        if not self.headless:
            import pyglet

            pyglet.clock.tick()

            self.app.platform_event_loop.step(0.001)  # 1ms app polling latency
            try:
                self.window.dispatch_events()
            except (ctypes.ArgumentError, TypeError):
                # Handle known issue with pyglet xlib backend on some Linux configurations
                # where window handle can have wrong type in XCheckWindowEvent
                # This is a non-fatal error that can be safely ignored
                pass

    def render(self, camera, objects, lines=None):
        gl = RendererGL.gl
        self._make_current()

        gl.glClearColor(*self.sky_upper, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(True)
        gl.glDepthRange(0.0, 1.0)

        self.camera = camera

        # Store matrices for other methods
        self._view_matrix = self.camera.get_view_matrix()
        self._projection_matrix = self.camera.get_projection_matrix()

        # Lazy-load environment map after a valid GL context is active
        if self._env_path is not None and self._env_texture is None:
            try:
                self.set_environment_map(self._env_path)
            except Exception:
                pass
            self._env_path = None

        # 1. render depth of scene to texture (from light's perspective)
        gl.glViewport(0, 0, self._shadow_width, self._shadow_height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._shadow_fbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        if self.draw_shadows:
            # Note: lines are skipped during shadow pass since they don't cast shadows
            self._render_shadow_map(objects)

        # reset viewport
        gl.glViewport(0, 0, self._screen_width, self._screen_height)

        # select target framebuffer (MSAA or regular) for scene rendering
        target_fbo = self._frame_msaa_fbo if getattr(self, "msaa_samples", 0) > 0 else self._frame_fbo

        # ---------------------------------------
        # Set texture as render target for MSAA resolve

        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, target_fbo)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)

        gl.glClearColor(*self.sky_upper, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(0)

        self._render_scene(objects)

        # Render lines after main scene but before MSAA resolve
        if lines:
            self._render_lines(lines)

        # ------------------------------------------------------------------
        # If MSAA is enabled, resolve the multi-sample buffer into texture FBO
        # ------------------------------------------------------------------
        if getattr(self, "msaa_samples", 0) > 0 and self._frame_msaa_fbo is not None:
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._frame_msaa_fbo)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)

            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._frame_fbo)
            gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)

            gl.glBlitFramebuffer(
                0,
                0,
                self._screen_width,
                self._screen_height,
                0,
                0,
                self._screen_width,
                self._screen_height,
                gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT,
                gl.GL_NEAREST,
            )

        # ------------------------------------------------------------------
        # Draw resolved texture to the screen
        # ------------------------------------------------------------------
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self._screen_width, self._screen_height)

        # render frame buffer texture to screen
        if self._frame_fbo is not None:
            with self._frame_shader:
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
                self._frame_shader.update(0)

                gl.glBindVertexArray(self._frame_vao)
                gl.glDrawElements(gl.GL_TRIANGLES, len(self._frame_indices), gl.GL_UNSIGNED_INT, None)
                gl.glBindVertexArray(0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        if self.draw_fps:
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)

        err = gl.glGetError()
        assert err == gl.GL_NO_ERROR, hex(err)

    def present(self):
        if not self.headless:
            self.window.flip()
        #    self.app.event_loop._redraw_windows(1.0 / 60.0)

    def resize(self, width, height):
        self._screen_width, self._screen_height = self.window.get_framebuffer_size()
        self._setup_frame_buffer()

    def set_title(self, title):
        self.window.set_caption(title)

    def set_vsync(self, enabled: bool):
        """Enable or disable vertical synchronization (vsync).

        Args:
            enabled: If True, enable vsync; if False, disable vsync.
        """
        self.window.set_vsync(enabled)

    def get_vsync(self) -> bool:
        """Get the current vsync state.

        Returns:
            True if vsync is enabled, False otherwise.
        """
        return self.window.vsync

    def has_exit(self):
        return self.app.event_loop.has_exit

    def close(self):
        self._make_current()

        if not self.headless:
            self.app.event_loop.dispatch_event("on_exit")
            self.app.platform_event_loop.stop()

        self.window.close()

    def _setup_window_callbacks(self):
        """Set up the basic window event handlers."""
        import pyglet

        self.window.push_handlers(on_draw=self._on_draw)
        self.window.push_handlers(on_resize=self._on_window_resize)
        self.window.push_handlers(on_key_press=self._on_key_press)
        self.window.push_handlers(on_key_release=self._on_key_release)
        self.window.push_handlers(on_close=self._on_close)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self._key_handler)

        self.window.push_handlers(on_mouse_press=self._on_mouse_press)
        self.window.push_handlers(on_mouse_release=self._on_mouse_release)

        self.window.on_mouse_scroll = self._on_scroll
        self.window.on_mouse_drag = self._on_mouse_drag
        self.window.on_mouse_motion = self._on_mouse_motion

    def register_key_press(self, callback):
        """Register a callback for key press events.

        Args:
            callback: Function that takes (symbol, modifiers) parameters
        """
        self._key_callbacks.append(callback)

    def register_key_release(self, callback):
        """Register a callback for key release events.

        Args:
            callback: Function that takes (symbol, modifiers) parameters
        """
        self._key_release_callbacks.append(callback)

    def register_mouse_press(self, callback):
        """Register a callback for mouse press events.

        Args:
            callback: Function that takes (x, y, button, modifiers) parameters
        """
        self._mouse_press_callbacks.append(callback)

    def register_mouse_release(self, callback):
        """Register a callback for mouse release events.

        Args:
            callback: Function that takes (x, y, button, modifiers) parameters
        """
        self._mouse_release_callbacks.append(callback)

    def register_mouse_drag(self, callback):
        """Register a callback for mouse drag events.

        Args:
            callback: Function that takes (x, y, dx, dy, buttons, modifiers) parameters
        """
        self._mouse_drag_callbacks.append(callback)

    def register_mouse_motion(self, callback):
        """Register a callback for mouse motion events.

        Args:
            callback: Function that takes (x, y, dx, dy) parameters
        """
        self._mouse_motion_callbacks.append(callback)

    def register_mouse_scroll(self, callback):
        """Register a callback for mouse scroll events.

        Args:
            callback: Function that takes (x, y, scroll_x, scroll_y) parameters
        """
        self._mouse_scroll_callbacks.append(callback)

    def register_resize(self, callback):
        """Register a callback for window resize events.

        Args:
            callback: Function that takes (width, height) parameters
        """
        self._resize_callbacks.append(callback)

    def register_update(self, callback):
        """Register a per-frame update callback receiving dt (seconds)."""
        self._update_callbacks.append(callback)

    def _on_key_press(self, symbol, modifiers):
        # update key state
        for callback in self._key_callbacks:
            callback(symbol, modifiers)

    def _on_key_release(self, symbol, modifiers):
        # update key state
        for callback in self._key_release_callbacks:
            callback(symbol, modifiers)

    def _on_mouse_press(self, x, y, button, modifiers):
        """Handle mouse button press events."""
        for callback in self._mouse_press_callbacks:
            callback(x, y, button, modifiers)

    def _on_mouse_release(self, x, y, button, modifiers):
        """Handle mouse button release events."""
        for callback in self._mouse_release_callbacks:
            callback(x, y, button, modifiers)

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # Then call registered callbacks
        for callback in self._mouse_drag_callbacks:
            callback(x, y, dx, dy, buttons, modifiers)

    def _on_mouse_motion(self, x, y, dx, dy):
        """Handle mouse motion events."""
        for callback in self._mouse_motion_callbacks:
            callback(x, y, dx, dy)

    def _on_scroll(self, x, y, scroll_x, scroll_y):
        for callback in self._mouse_scroll_callbacks:
            callback(x, y, scroll_x, scroll_y)

    def _on_window_resize(self, width, height):
        self.resize(width, height)

        for callback in self._resize_callbacks:
            callback(width, height)

    def _on_close(self):
        self.close()

    def _on_draw(self):
        pass

    # public query for key state
    def is_key_down(self, symbol: int) -> bool:
        if self.headless:
            return False

        return bool(self._key_handler[symbol])

    def _setup_sky_mesh(self):
        gl = RendererGL.gl

        # create VAO, VBO, and EBO
        self._sky_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._sky_vao)
        gl.glBindVertexArray(self._sky_vao)

        vertices, indices = create_sphere_mesh(1.0, 32, 32, reverse_winding=True)
        self._sky_tri_count = len(indices)

        self._sky_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._sky_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._sky_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

        self._sky_ebo = gl.GLuint()
        gl.glGenBuffers(1, self._sky_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._sky_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)

        # set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        # unbind the VBO and VAO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        check_gl_error()

    def _setup_frame_buffer(self):
        gl = RendererGL.gl

        # Ensure MSAA member variables exist even on first call
        if not hasattr(self, "_frame_msaa_color_rb"):
            self._frame_msaa_color_rb = None
        if not hasattr(self, "_frame_msaa_depth_rb"):
            self._frame_msaa_depth_rb = None
        if not hasattr(self, "_frame_msaa_fbo"):
            self._frame_msaa_fbo = None

        self._make_current()

        if self._frame_texture is None:
            self._frame_texture = gl.GLuint()
            gl.glGenTextures(1, self._frame_texture)
        if self._frame_depth_texture is None:
            self._frame_depth_texture = gl.GLuint()
            gl.glGenTextures(1, self._frame_depth_texture)

        # set up RGB texture
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            self._screen_width,
            self._screen_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # set up depth texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_depth_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT32,
            self._screen_width,
            self._screen_height,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # create a framebuffer object (FBO)
        if self._frame_fbo is None:
            self._frame_fbo = gl.GLuint()
            gl.glGenFramebuffers(1, self._frame_fbo)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)

            # attach the texture to the FBO as its color attachment
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._frame_texture, 0
            )
            # attach the depth texture to the FBO as its depth attachment
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._frame_depth_texture, 0
            )

            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                print("Framebuffer is not complete!", flush=True)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                sys.exit(1)

        # unbind the FBO (switch back to the default framebuffer)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        if self._frame_pbo is None:
            self._frame_pbo = gl.GLuint()
            gl.glGenBuffers(1, self._frame_pbo)  # generate 1 buffer reference
        # binding to this buffer
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._frame_pbo)

        # allocate memory for PBO
        rgb_bytes_per_pixel = 3
        depth_bytes_per_pixel = 4
        pixels = np.zeros(
            (self._screen_height, self._screen_width, rgb_bytes_per_pixel + depth_bytes_per_pixel), dtype=np.uint8
        )
        gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, pixels.nbytes, pixels.ctypes.data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

        # ---------------------------------------------------------------------
        # Additional: create MSAA framebuffer if multi-sampling is enabled
        # ---------------------------------------------------------------------
        if getattr(self, "msaa_samples", 0) > 0:
            # color renderbuffer
            if self._frame_msaa_color_rb is None:
                self._frame_msaa_color_rb = gl.GLuint()
                gl.glGenRenderbuffers(1, self._frame_msaa_color_rb)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._frame_msaa_color_rb)
            gl.glRenderbufferStorageMultisample(
                gl.GL_RENDERBUFFER, self.msaa_samples, gl.GL_RGB8, self._screen_width, self._screen_height
            )

            # depth renderbuffer
            if self._frame_msaa_depth_rb is None:
                self._frame_msaa_depth_rb = gl.GLuint()
                gl.glGenRenderbuffers(1, self._frame_msaa_depth_rb)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._frame_msaa_depth_rb)
            gl.glRenderbufferStorageMultisample(
                gl.GL_RENDERBUFFER, self.msaa_samples, gl.GL_DEPTH_COMPONENT32, self._screen_width, self._screen_height
            )

            # FBO
            if self._frame_msaa_fbo is None:
                self._frame_msaa_fbo = gl.GLuint()
                gl.glGenFramebuffers(1, self._frame_msaa_fbo)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_msaa_fbo)
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, self._frame_msaa_color_rb
            )
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self._frame_msaa_depth_rb
            )

            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                print("Warning: MSAA framebuffer incomplete, disabling MSAA.")
                self.msaa_samples = 0
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        check_gl_error()

    def _setup_frame_mesh(self):
        gl = RendererGL.gl

        # fmt: off
        # set up VBO for the quad that is rendered to the user window with the texture
        self._frame_vertices = np.array([
            # Positions  TexCoords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ], dtype=np.float32)
        # fmt: on

        self._frame_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self._frame_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._frame_vao)
        gl.glBindVertexArray(self._frame_vao)

        self._frame_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._frame_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, self._frame_vertices.nbytes, self._frame_vertices.ctypes.data, gl.GL_STATIC_DRAW
        )

        self._frame_ebo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._frame_ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, self._frame_indices.nbytes, self._frame_indices.ctypes.data, gl.GL_STATIC_DRAW
        )

        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * self._frame_vertices.itemsize, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            1,
            2,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            4 * self._frame_vertices.itemsize,
            ctypes.c_void_p(2 * self._frame_vertices.itemsize),
        )
        gl.glEnableVertexAttribArray(1)

        check_gl_error()

    def _setup_shadow_buffer(self):
        gl = RendererGL.gl

        self._make_current()

        # create depth texture FBO
        self._shadow_fbo = gl.GLuint()
        gl.glGenFramebuffers(1, self._shadow_fbo)

        self._shadow_texture = gl.GLuint()
        gl.glGenTextures(1, self._shadow_texture)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._shadow_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT,
            self._shadow_width,
            self._shadow_height,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, (gl.GLfloat * 4)(*border_color))

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._shadow_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._shadow_texture, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        check_gl_error()

    def _render_shadow_map(self, objects):
        gl = RendererGL.gl
        from pyglet.math import Mat4, Vec3

        self._make_current()

        extents = 10.0

        light_near = 1.0
        light_far = 1000.0
        camera_pos = np.array(self.camera.pos, dtype=np.float32)
        light_pos = camera_pos + self._sun_direction * extents
        light_proj = Mat4.orthogonal_projection(-extents, extents, -extents, extents, light_near, light_far)

        light_view = Mat4.look_at(Vec3(*light_pos), Vec3(*camera_pos), Vec3(*self.camera.get_up()))
        self._light_space_matrix = np.array(light_proj @ light_view, dtype=np.float32)

        self._shadow_shader.update(self._light_space_matrix)

        # render from light's point of view
        with self._shadow_shader:
            self._draw_objects(objects)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        check_gl_error()

    def _render_scene(self, objects):
        gl = RendererGL.gl

        if self.draw_sky:
            self._draw_sky()

        if self.draw_wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        self._shape_shader.update(
            view_matrix=self._view_matrix,
            projection_matrix=self._projection_matrix,
            view_pos=self.camera.pos,
            fog_color=self.sky_lower,
            up_axis=self.camera.up_axis,
            sun_direction=self._sun_direction,
            enable_shadows=self.draw_shadows,
            shadow_texture=self._shadow_texture,
            light_space_matrix=self._light_space_matrix,
            light_color=self._light_color,
            sky_color=self.sky_upper,
            ground_color=self.sky_lower,
            env_texture=self._env_texture,
            env_intensity=self._env_intensity,
        )

        with self._shape_shader:
            self._draw_objects(objects)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        check_gl_error()

    def _render_lines(self, lines):
        """Render all line objects using the line shader."""
        # Set up line shader once for all line objects
        self._line_shader.update(self._view_matrix, self._projection_matrix)

        with self._line_shader:
            for line_obj in lines.values():
                if hasattr(line_obj, "render"):
                    line_obj.render()

        check_gl_error()

    def _draw_objects(self, objects):
        for o in objects.values():
            if hasattr(o, "render"):
                o.render()

        check_gl_error()

    def _draw_sky(self):
        gl = RendererGL.gl

        self._make_current()

        self._sky_shader.update(
            view_matrix=self._view_matrix,
            projection_matrix=self._projection_matrix,
            camera_pos=self.camera.pos,
            camera_far=self.camera.far,
            sky_upper=self.sky_upper,
            sky_lower=self.sky_lower,
            sun_direction=self._sun_direction,
        )

        gl.glBindVertexArray(self._sky_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._sky_tri_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        check_gl_error()

    def set_environment_map(self, path: str, intensity: float = 1.0) -> None:
        gl = RendererGL.gl
        from ...utils.texture import load_texture_from_file  # noqa: PLC0415

        image = load_texture_from_file(path)
        if image is None:
            return
        if self._env_texture is not None:
            try:
                gl.glDeleteTextures(1, self._env_texture)
            except Exception:
                pass
            self._env_texture = None
        self._env_texture = _upload_texture_from_file(gl, image)
        self._env_texture_obj = None
        self._env_intensity = float(intensity)

    def _make_current(self):
        try:
            self.window.switch_to()
        except AttributeError:
            # The window could be in the process of being closed, in which case
            # its corresponding context might have been destroyed and set to `None`.
            pass

    def _set_icon(self):
        import pyglet

        def load_icon(filename):
            filename = os.path.join(os.path.dirname(__file__), filename)

            if not os.path.exists(filename):
                raise FileNotFoundError(
                    f"Error: Icon file '{filename}' not found. Please run the 'generate_icons.py' script first."
                )

            with open(filename, "rb") as f:
                icon_bytes = f.read()

            icon_stream = io.BytesIO(icon_bytes)
            icon = pyglet.image.load(filename=filename, file=icon_stream)

            return icon

        icons = [load_icon("icon_16.png"), load_icon("icon_32.png"), load_icon("icon_64.png")]

        # 5. Create the window and set the icon
        self.window.set_icon(*icons)
