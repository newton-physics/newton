# Shared QA requirements

Apply these shared launch, viewer, and shutdown checks alongside the
example-specific criteria; do not copy them into each example.

## Every case

- Install the requested artifact in an isolated environment and record the
  actual version and dependencies.
- The example should launch without unexpected errors. Observe stdout/stderr
  throughout execution and on exit. Record warnings and assess them; do not
  silently accept historical known issues as permanent exemptions.
- Exercise all requested CLI variants and relevant supported example controls.
  Keep variant-specific expectations with the example.
- Interactive windows should close cleanly and their processes should terminate
  without errors. Noninteractive jobs should finish and produce their specified
  outputs. Viewers that run as services may require terminal interruption;
  distinguish that expected shutdown from a crash.
- Perform the skill's short exploratory interaction pass and applicable
  late-run smoothness/settling checks. Establish expected rest versus continuing
  motion from the example; no blanket assumption that every scene must settle.

## Viewer checks

For each requested viewer/platform combination, exercise its supported controls
on a suitable scene. Check basic input responsiveness while testing each example;
the exhaustive shared viewer checklist need not be repeated for every scene.
Compare behavior to the target version rather than hard-coded historical labels
or counts. Scene-specific counts, layouts and visuals belong to example criteria.

- Statistics and model panels show plausible live FPS, world/body/shape/joint
  counts, up axis and gravity consistent with the scene. Compare exact counts
  only where the example's criteria prescribe them.
- Camera translation, looking, orbiting, dolly and field-of-view controls change
  the view; displayed position/orientation/FOV values reflect those changes.
- Picking selects and manipulates supported movable objects. Pause/resume,
  single-step, reset, UI visibility and frame-model controls have their expected
  effects. Physics pause prevents force-driven motion until a step or resume.
- Visualization switches show/hide visual meshes, collision shapes, joint
  frames, contacts, centers of mass, inertia boxes, and supported gap/margin
  overlays. Use actual contacts when testing contact visualization. Line-width
  controls should affect the relevant overlays.
- Sky, shadows, wireframe, light/sky/ground colors, and supported rendering
  controls visibly affect the intended scene properties. Restore settings before
  baseline comparison; do not confuse an intentionally hidden shape with a bug.
- Closing the viewer or its documented service shutdown ends the case cleanly.

For GL, use [interaction.md](interaction.md). Rerun, Viser, and RTX require their
own installed dependencies and verified input/capture paths. Validate browser
startup/connection and camera interaction where appropriate; Rerun additionally
has object-hover inspection. Use addresses actually reported by the target
viewer rather than assuming a fixed port. RTX can compile shaders for minutes
on first launch. Historical RTX overlay limitations are leads to recheck, not
current waivers. Do not install or claim coverage of unrequested backends.

## Coverage limits

Record the platforms, devices, and viewer backends actually tested, plus gaps
in requested coverage. Visual example QA does not imply unit-test, sanitizer,
benchmark, CPU-only, multi-GPU, or other-platform coverage. Run broader release
checks only when requested.
