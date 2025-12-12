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
Proximal-ADMM Solver

Provides a forward dynamics solver for constrained rigid multi-body systems using
the Alternating Direction Method of Multipliers (ADMM). This implementation realizes
the Proximal-ADMM algorithm described in [1] and is based on the work of J. Carpentier
et al in [2]. It solves the Lagrange dual of the constrained forward dynamics problem
in constraint reactions (i.e. impulses) and post-event constraint-space velocities. The
diagonal preconditioner strategy described in [3] is also implemented to improve
numerical conditioning. This version also incorporates Nesterov-style gradient
acceleration with adaptive restarts based on the work of O'Donoghue and Candes in [4].

Notes
----
- ADMM is based on the Augmented Lagrangian Method (ALM) for dealing with set-inclusion
  (i.e. inequality) constraints, but introduces an alternating primal-dual descent/ascent scheme.
- Proximal-ADMM introduces an additional proximal regularization term to the optimization objective.
- Uses (optional) Nesterov-style gradient acceleration with adaptive restarts.
- Uses (optional) adaptive penalty updates based on primal-dual residual balancing.

References
----
- [1] https://arxiv.org/abs/2504.19771
- [2] https://arxiv.org/pdf/2405.17020
- [3] https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6693
- [4] https://epubs.siam.org/doi/abs/10.1137/120896219

Usage
----
A typical example for using this module is:

    # Import all relevant types from Kamino
    from newton._src.solvers.kamino.core import ModelBuilder
    from newton._src.solvers.kamino.geometry import Contacts
    from newton._src.solvers.kamino.kinematics import Limits
    from newton._src.solvers.kamino.kinematics import DenseSystemJacobians
    from newton._src.solvers.kamino.dynamics import DualProblem
    from newton._src.solvers.kamino.solvers import PADMMSolver

    # Create a model builder and add bodies, joints, geoms, etc.
    builder = ModelBuilder()
    ...

    # Create a model from the builder and construct additional
    # containers to hold joint-limits, contacts, Jacobians
    model = builder.finalize()
    data = model.data()
    limits = Limits(builder)
    contacts = Contacts(builder)
    jacobians = DenseSystemJacobians(model, limits, contacts)

    # Build the Jacobians for the model and active limits and contacts
    jacobians.build(model, data, limits, contacts)
    ...

    # Create a forward-dynamics DualProblem to be solved
    dual = DualProblem(model, limits, contacts)
    dual.build(model, data, limits, contacts, jacobians)

    # Create a forward-dynamics PADMM solver
    solver = PADMMSolver(model, limits, contacts)

    # Solve the dual forward dynamics problem
    solver.coldstart()
    solver.solve(problem=dual)

    # Extract the resulting constraint reactions
    multipliers = solver.solution.lambdas
"""

from .solver import PADMMSolver
from .types import PADMMPenaltyUpdate, PADMMSettings, PADMMWarmStartMode

###
# Module interface
###

__all__ = [
    "PADMMPenaltyUpdate",
    "PADMMSettings",
    "PADMMSolver",
    "PADMMWarmStartMode",
]
