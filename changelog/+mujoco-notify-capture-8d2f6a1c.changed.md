Validate authored `mujoco.solreflimit` values only when constructing `SolverMuJoCo` with the MuJoCo Warp backend so later `notify_model_changed()` calls remain graph-capturable. Recreate the solver or use `use_mujoco_cpu=True` to have reassigned values re-validated.

Skip host-side cone-scale validation during CUDA graph capture and replay. Cone resizing remains unsupported; keep cone scales fixed during replay and use an eager `SHAPE_PROPERTIES` notification to validate edits outside capture.
