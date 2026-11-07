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
Preserve complete auxiliary render states during history replay, including mixed rigid/deformable scenes.
Frame scenes and selections from displayed geometry when rendering uses different units from the collision model.
Add experimental `ViewerBase.set_control()`, `set_solver()`, and
`set_collision_pipeline()` methods for inspecting custom simulation loops,
with State and Contacts supplied through the existing frame logging methods.
