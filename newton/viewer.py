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

from ._src.utils.gizmo import GizmoSystem
from ._src.utils.recorder_gui import RecorderImGuiManager
from ._src.utils.render import RendererOpenGL, RendererUsd

# Import all viewer classes (they handle missing dependencies at instantiation time)
from ._src.viewer import ViewerGL, ViewerNull, ViewerRerun, ViewerUSD

__all__ = [
    "GizmoSystem",
    "RecorderImGuiManager",
    "RendererOpenGL",  # deprecated
    "RendererUsd",  # deprecated
    "ViewerGL",
    "ViewerNull",
    "ViewerRerun",
    "ViewerUSD",
]
