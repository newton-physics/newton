Add a docked ViewerGL workspace with a scene explorer, typed State/Control/Model
inspectors, a recorded-state timeline, logging console, and integrated images
and plots. Add selection and camera navigation tools, collision and BVH
inspection, and X-ray rendering with adaptive surface and wireframe detail.
Fit replicated deformables using displayed world offsets and visibility while
retaining selected-object, contact, and follow-camera navigation.
Handle empty and multi-shape visual batches when loading examples or switching body variants.
Add experimental `ViewerBase.set_control()`, `set_solver()`, and
`set_collision_pipeline()` methods for inspecting custom simulation loops,
with State and Contacts supplied through the existing frame logging methods.
