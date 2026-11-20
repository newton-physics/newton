Add a docked ViewerGL workspace with a scene explorer, typed State/Control/Model
inspectors, a recorded-state timeline, logging console, and integrated images
and plots. Add selection and camera navigation tools, collision and BVH
inspection, and X-ray rendering with adaptive surface and wireframe detail.
Fit replicated deformables using displayed world offsets and visibility while
retaining selected-object, contact, and follow-camera navigation.
Handle empty and multi-shape visual batches when loading examples or switching body variants.
Keep viewport picking and selection outlines synchronized with body variants and SDF collision surfaces.
Refresh geometry properties and BVH references when shape variants change, respect
particle visibility during picking, and apply viewport display modes across layers.
Inspect and edit custom joint-frequency attributes using their own namespace and row width.
Record history from the application's `log_state()`, `log_contacts()`, `log_points()` and `log_lines()` calls and replay it through those calls, pausing stepping through `ViewerBase.should_step()` until playback reaches the newest frame.
Frame scenes and selections from displayed geometry when rendering uses different units from the collision model.
Filter contact glyphs to selected shapes and bodies by default, with a toggle to show all contacts.
Show logged per-body and per-world solver observables, such as `solver_residual`, read-only in the State tab, and list a solver's supported observable kinds in the Solver tab.
Add experimental `ViewerBase.set_control()`, `set_solver()`, and
`set_collision_pipeline()` methods; with the existing frame logging methods,
they are the inspector's only inputs, for examples and custom loops alike.
