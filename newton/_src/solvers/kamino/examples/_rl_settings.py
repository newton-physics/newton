# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from newton._src.solvers.kamino._src.utils.sim import Simulator


def create_rl_settings(sim_dt: float = 0.01) -> Simulator.Config:
    """Create the default Kamino settings used by the RL examples."""
    settings = Simulator.Config()
    settings.dt = sim_dt
    settings.solver.sparse_jacobian = True
    settings.solver.use_fk_solver = False
    settings.solver.use_collision_detector = True
    settings.collision_detector.pipeline = "unified"
    settings.collision_detector.max_contacts_per_pair = 8
    settings.solver.integrator = "moreau"
    settings.solver.constraints.alpha = 0.1
    settings.solver.padmm.primal_tolerance = 1e-4
    settings.solver.padmm.dual_tolerance = 1e-4
    settings.solver.padmm.compl_tolerance = 1e-4
    settings.solver.padmm.max_iterations = 200
    settings.solver.padmm.eta = 1e-5
    settings.solver.padmm.rho_0 = 0.05
    settings.solver.padmm.use_acceleration = True
    settings.solver.padmm.warmstart_mode = "containers"
    settings.solver.padmm.contact_warmstart_method = "geom_pair_net_force"
    settings.solver.collect_solver_info = False
    settings.solver.compute_solution_metrics = False
    settings.solver.padmm.use_graph_conditionals = False
    return settings
