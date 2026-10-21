# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ImGui presentation and interaction for the viewer.

Keep this package lightweight: panels and the UI backend load on demand.
Shared simulation metadata and history remain in the parent viewer package;
OpenGL rendering stays in viewer.gl.
"""
