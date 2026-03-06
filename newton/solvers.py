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
Solvers are used to integrate the dynamics of a Newton model.
The typical workflow is to construct a :class:`~newton.Model` and a :class:`~newton.State` object, then use a solver to advance the state forward in time
via the :meth:`~newton.solvers.SolverBase.step` method:

.. mermaid::
  :config: {"theme": "forest", "themeVariables": {"lineColor": "#76b900"}}

  flowchart LR
      subgraph Input["Input Data"]
          M[newton.Model]
          S[newton.State]
          C[newton.Control]
          K[newton.Contacts]
          DT[Time step dt]
      end

      STEP["solver.step()"]

      subgraph Output["Output Data"]
          SO["newton.State (updated)"]
      end

      %% Connections
      M --> STEP
      S --> STEP
      C --> STEP
      K --> STEP
      DT --> STEP
      STEP --> SO

Supported Features
------------------

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 0

   * - Solver
     - :abbr:`Integration (Available methods for integrating the dynamics)`
     - Rigid bodies
     - :ref:`Articulations <Articulations>`
     - Particles
     - Cloth
     - Soft bodies
   * - :class:`~newton.solvers.SolverFeatherstone`
     - Explicit
     - ✅
     - ✅ generalized coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
   * - :class:`~newton.solvers.SolverImplicitMPM`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverMuJoCo`
     - Explicit, Semi-implicit, Implicit
     - ✅ (uses its own collision pipeline from MuJoCo/mujoco_warp by default, unless ``use_mujoco_contacts`` is set to False)
     - ✅ generalized coordinates
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverSemiImplicit`
     - Semi-implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
   * - :class:`~newton.solvers.SolverStyle3D`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
   * - :class:`~newton.solvers.SolverVBD`
     - Implicit
     - ✅
     - 🟨 :ref:`limited joint support <Joint feature support>`
     - ✅
     - ✅
     - ❌
   * - :class:`~newton.solvers.SolverXPBD`
     - Implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - 🟨 experimental

.. _Joint feature support:

Joint Feature Support
---------------------

Not every solver supports every joint type or joint property.
The tables below document which joint features each solver handles.

Only :class:`~newton.solvers.SolverFeatherstone` and :class:`~newton.solvers.SolverMuJoCo`
operate on :ref:`articulations <Articulations>` (generalized/reduced coordinates).
The maximal-coordinate solvers (:class:`~newton.solvers.SolverSemiImplicit`,
:class:`~newton.solvers.SolverXPBD`) enforce joints as pairwise body constraints
but do not use the articulation kinematic-tree structure.
:class:`~newton.solvers.SolverVBD` supports a subset of joint types via soft
constraints (AVBD). :class:`~newton.solvers.SolverStyle3D` and
:class:`~newton.solvers.SolverImplicitMPM` do not support joints.

**Joint types**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Joint type
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
   * - PRISMATIC
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |no|
   * - REVOLUTE
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |no|
   * - BALL
     - |yes|
     - |yes|
     - |no|
     - |yes|
     - |yes|
   * - FIXED
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - FREE
     - |yes|
     - |yes|
     - |no|
     - |yes|
     - |yes| :sup:`1`
   * - DISTANCE
     - |yes| :sup:`2`
     - |yes| :sup:`2`
     - |yes|
     - |yes| :sup:`2`
     - |no|
   * - D6
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |no|
   * - CABLE
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes|

| :sup:`1` FREE joints in VBD apply no constraints (unconstrained motion).
| :sup:`2` DISTANCE joints are treated as FREE (no distance constraint enforcement).

**Joint properties**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Property
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
   * - :attr:`~newton.Model.joint_enabled`
     - |no|
     - |yes|
     - |yes|
     - |no|
     - |no|
   * - :attr:`~newton.Model.joint_armature`
     - |yes|
     - |no|
     - |no|
     - |yes|
     - |no|
   * - :attr:`~newton.Model.joint_friction`
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
   * - :attr:`~newton.Model.joint_limit_lower` / :attr:`~newton.Model.joint_limit_upper`
     - |yes|
     - |yes| :sup:`3`
     - |yes|
     - |yes|
     - |no|
   * - :attr:`~newton.Model.joint_limit_ke` / :attr:`~newton.Model.joint_limit_kd`
     - |yes|
     - |yes| :sup:`3`
     - |no|
     - |yes|
     - |no|
   * - :attr:`~newton.Model.joint_effort_limit`
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
   * - :attr:`~newton.Model.joint_velocity_limit`
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|

| :sup:`3` Not enforced for BALL joints in SemiImplicit.

**Actuation and control**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Feature
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
   * - :attr:`~newton.Model.joint_target_ke` / :attr:`~newton.Model.joint_target_kd`
     - |yes|
     - |yes| :sup:`3`
     - |yes|
     - |yes|
     - 🟨 :sup:`5`
   * - :attr:`~newton.Model.joint_target_mode`
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
   * - :attr:`~newton.Control.joint_f` (feedforward forces)
     - |yes|
     - |yes|
     - |yes|
     - |no|
     - |no|

**Constraints**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Feature
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
   * - Equality constraints (CONNECT, WELD, JOINT)
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
   * - Mimic constraints
     - |no|
     - |no|
     - |no|
     - |yes| :sup:`4`
     - |no|

| :sup:`4` Mimic constraints in MuJoCo are supported for REVOLUTE and PRISMATIC joints only.
| :sup:`5` Used for CABLE joints only (as stretch/bend stiffness and damping).

.. |yes| unicode:: U+2705
.. |no| unicode:: U+274C
"""

# solver types
from ._src.solvers import (
    SolverBase,
    SolverFeatherstone,
    SolverImplicitMPM,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverStyle3D,
    SolverVBD,
    SolverXPBD,
    style3d,
)

# solver flags
from ._src.solvers.flags import SolverNotifyFlags

__all__ = [
    "SolverBase",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverMuJoCo",
    "SolverNotifyFlags",
    "SolverSemiImplicit",
    "SolverStyle3D",
    "SolverVBD",
    "SolverXPBD",
    "style3d",
]
