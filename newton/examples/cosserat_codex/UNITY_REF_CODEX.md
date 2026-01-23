## DefKit Cosserat Rod (Iterative) - Codex Notes

This example runs the iterative Position and Orientation Based Cosserat Rod
solver from `DefKitAdv.dll` and uses Newton as the viewer only.

### Key DLL Entry Points
- `PredictPositions_native`
- `Integrate_native`
- `PredictRotationsPBD`
- `IntegrateRotationsPBD`
- `ProjectElasticRodConstraints`

### Per-Substep Call Order
1. `PredictPositions_native(dt, damping, ...)`
2. `PredictRotationsPBD(dt, rot_damping, ...)`
3. Repeat `ProjectElasticRodConstraints(...)` for `constraints_iterations`
4. `Integrate_native(dt, ...)`
5. `IntegrateRotationsPBD(dt, ...)`

### Data Layout Notes
- Use `float32` arrays shaped `(N, 4)` to match Bullet `btVector3` / `btQuaternion`
  (16-byte aligned with unused `w` component).
- Quaternions are in `(x, y, z, w)` order.
- `rest_darboux` uses identity quaternions `(0, 0, 0, 1)` for a straight rod.
- `bend_twist_ks` uses `(bend_x, bend_y, twist_z, 0)`.

### Viewer Integration
Newton `Model` is only for visualization (particles + optional lines/directors).
Positions are copied from the DLL buffers into `state.particle_q` each frame.
