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

"""Custom attributes for solver properties."""

from typing import Any


class SolverProperty:
    def __init__(self, name: str, type: type, frequency: str, default: Any):
        self.name = name
        self.type = type
        self.frequency = frequency
        self.default = default

class SolverProperties:

    @property
    def get_model_properties(self) -> dict[str, SolverProperty]:
        return {}

    @property
    def get_shape_properties(self) -> dict[str, SolverProperty]:
        return {}

    @property
    def get_body_properties(self) -> dict[str, SolverProperty]:
        return {}

    @property
    def get_joint_properties(self) -> dict[str, SolverProperty]:
        return {}


    def parse_urdf_begin(self, builder, root_element):
        pass

    def parse_urdf_shape(self, builder, geom_element):
        pass

    def parse_urdf_body(self, builder, body_element):
        pass

    def parse_urdf_joint(self, builder, joint_element):
        pass

    def parse_urdf_end(self, builder, root_element):
        pass


    def parse_mjcf_begin(self, builder, root_element):
        pass

    def parse_mjcf_shape(self, builder, geom_element):
        pass

    def parse_mjcf_body(self, builder, body_element):
        pass

    def parse_mjcf_joint(self, builder, joint_element):
        pass

    def parse_mjcf_end(self, builder, root_element):
        pass


    def parse_usd_begin(self, builder, stage):
        pass

    def parse_usd_shape(self, builder, prim):
        pass

    def parse_usd_body(self, builder, prim):
        pass

    def parse_usd_joint(self, builder, prim):
        pass

    def parse_usd_end(self, builder, stage):
        pass


    def finalize(self, builder, model):
        pass
