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

"""
KAMINO: Kinematics Module
"""

from .constraints import make_unilateral_constraints_info, update_constraints_info
from .jacobians import (
    DenseSystemJacobians,
    DenseSystemJacobiansData,
    build_contact_jacobians,
    build_jacobians,
    build_joint_jacobians,
    build_limit_jacobians,
)
from .joints import compute_joints_state
from .limits import Limits, LimitsData

###
# Module interface
###

__all__ = [
    "DenseSystemJacobians",
    "DenseSystemJacobiansData",
    "Limits",
    "LimitsData",
    "build_contact_jacobians",
    "build_jacobians",
    "build_joint_jacobians",
    "build_limit_jacobians",
    "compute_joints_state",
    "make_unilateral_constraints_info",
    "update_constraints_info",
]
