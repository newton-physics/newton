Fix `SolverMuJoCo(use_mujoco_cpu=True)` culling penetrating contacts on multi-shape bodies, whose midphase bounds were offset from the physical center of mass by 1 mm.
