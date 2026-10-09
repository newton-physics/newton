# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Owned image storage independent of renderer output and logged textures."""

import ctypes

import numpy as np


class FrameCache:
    """Keep a GPU copy of a displayed texture, including across window resize."""

    def __init__(self):
        self.texture = 0
        self.width = 0
        self.height = 0
        self._fbo = 0

    def store(self, texture: int, width: int, height: int) -> None:
        """Copy a texture without changing its row orientation."""
        from pyglet import gl

        if not self.texture:
            handle = gl.GLuint()
            gl.glGenTextures(1, handle)
            self.texture = handle.value
            gl.glGenFramebuffers(1, handle)
            self._fbo = handle.value

        previous = gl.GLint()
        gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING, previous)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._fbo)
        gl.glFramebufferTexture2D(gl.GL_READ_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        if (width, height) != (self.width, self.height):
            gl.glCopyTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, 0, 0, width, height, 0)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            self.width, self.height = width, height
        else:
            gl.glCopyTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, previous.value)

    def pixels(self) -> np.ndarray:
        """Read the stored RGBA image with its original row orientation."""
        from pyglet import gl

        if not self.texture:
            raise RuntimeError("Frame capture requires at least one displayed frame")
        pixels = np.empty((self.height, self.width, 4), dtype=np.uint8)
        previous = gl.GLint()
        gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING, previous)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._fbo)
        gl.glFramebufferTexture2D(gl.GL_READ_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.texture, 0)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        gl.glReadPixels(
            0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels.ctypes.data_as(ctypes.c_void_p)
        )
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, previous.value)
        return pixels

    def clear(self) -> None:
        """Release the cached image while its GL context is current."""
        if self.texture:
            from pyglet import gl

            gl.glDeleteTextures(1, gl.GLuint(self.texture))
            gl.glDeleteFramebuffers(1, gl.GLuint(self._fbo))
        self.texture = self._fbo = self.width = self.height = 0
