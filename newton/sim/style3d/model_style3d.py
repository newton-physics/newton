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

"""Style3D model class derived from the Newton model class."""

from __future__ import annotations

from ...core.types import Devicelike
from ..model import Model


class Style3DModel(Model):
    """Style3DModel derived from newton model, contains extended simulation attributes.

    Note:
        Use the Style3DModelBuilder to construct a
        simulation rather than creating your own Model object directly,
        however it is possible to do so if desired.
    """

    def __init__(self, device: Devicelike | None = None):
        """
        Initializes the Model object.

        Args:
            device (wp.Device): Device on which the Model's data will be allocated.
        """
        super().__init__(device=device)
        self.tri_aniso_ke = None
        """Triangle element aniso stretch stiffness(weft, warp, shear), shape [tri_count, 3], float."""
        self.edge_rest_area = None
        """Bending edge area, sum area of adjacent two triangles, shape [edge_count], float."""
        self.edge_bending_cot = None
        """Bending edge cotangents, shape [edge_count, 4], float."""

    @classmethod
    def from_model(cls, model: Model):
        style3d_model = cls.__new__(cls)
        style3d_model.__dict__.update(model.__dict__)
        style3d_model.tri_aniso_ke = None
        style3d_model.edge_rest_area = None
        style3d_model.edge_bending_cot = None
        return style3d_model
