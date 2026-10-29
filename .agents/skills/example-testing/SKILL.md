---
name: example-testing
description: Run Newton examples for visual QA of a release wheel or requested checkout, exercising viewer controls and checking motion, picking, and end-of-run settling against example-specific criteria. Use for example testing and release QA, not as a substitute for unit tests or release approval.
---

# Newton Example Testing

Produce a reproducible local visual QA report with commands, observations, and
image evidence. Preserve the requested artifact and scope: a wheel test must
import the installed wheel, not this checkout. Do not modify examples to make
them pass or infer acceptance from a clean exit. QA does not authorize changes
to simulation or example implementation code. Record failures and proposed CI
checks as text for follow-up work. Evidence-backed example-spec corrections are
allowed when within the user's scope; keep the tested artifact unchanged.

## Establish the run

1. Use the version/wheel, examples, device, and criteria supplied by the user.
   If the release artifact is unspecified, ask for it; do not silently test main
   or a moving latest release. Prepare the environment while resolving inputs.
2. Read [setup.md](references/setup.md) for isolated uv installation, provenance,
   device/display checks, and launch commands. Inventory examples from the
   installed artifact. For a requested full suite, retain every inventory entry
   in the report, including blocked and untested examples and requested variants.
3. Read [criteria.md](references/criteria.md). Extract each example's description,
   common criteria, and required run variants with `--describe`; expectations live
   beside the example implementation. Apply explicit user overrides and record
   both artifact and criteria revisions. Follow the pinned-source fallback for
   older wheels without QA docstrings. Missing or contradictory criteria leave the
   affected acceptance checks incomplete; continue independent checks. Never
   infer a release pass or reuse old observations as current results.
4. Establish a working capture/input path before a large campaign. The proven
   path is the native GL viewer on Linux X11, screenshots inspected by the agent,
   and real mouse/keyboard events. If using another environment/backend, verify
   its equivalent interactions and record the difference.

## Coordinate a full campaign

Use subagents for independent examples when delegation and isolated visual
sessions are available. Start with up to three workers, adjusting concurrency
to CPU/GPU memory and capture cadence. Read [parallel.md](references/parallel.md)
for display isolation, ownership, handoffs, and coordinator review. If isolation
or delegation is unavailable, record the limitation and continue sequentially.

The coordinator owns the complete checklist. Wait for workers, review their
actual evidence, then assign the next untested examples; spawning workers is not
a stopping point. Continue until every requested example and variant has an
honest assessed status, honoring user stop instructions and reporting unresolved
coverage. A completed assessment may contain failures or blocked criteria.

## Execute and inspect

- Apply the shared checks in [shared-qa.md](references/shared-qa.md), including
  relevant viewer controls. These checks are defined once here rather than
  repeated in every example's educational source. Record unsupported controls
  and environment limits instead of assuming every backend behaves identically.
- Read [interaction.md](references/interaction.md) for capture, pan, pick,
  stepping, and cleanup commands. Use the bundled `scripts/desktop.py` helper
  for screenshots and smooth drags on X11. Launch one example per process,
  initially paused where supported; retain logs and a fresh initial screenshot.
- Exercise the requested scenario and inspect actual images after each action.
  An input command returning success is not proof the GUI responded. Preserve
  action order, camera views, and simulation time/steps when known.
- For smoothness, jitter, or settling criteria, read
  [jitter.md](references/jitter.md). Observe consecutive late-run frames with
  physics advancing and a fixed camera; a paused final image is not evidence
  of stability. Distinguish wall time, simulation time, and render cadence.
- After the prescribed checks, spend a short exploratory interval on each
  interactive example (roughly 30–60 seconds of interaction, adjusted to its
  complexity). Vary a few plausible actions: pull different movable objects,
  release during motion, pause and inspect contacts up close, step and resume,
  change camera angles, toggle relevant overlays, or adjust an example control
  within its supported range. Choose actions that make sense for the scene,
  rather than blindly clicking random screen coordinates. For noninteractive
  examples, explore a relevant supported input/variant instead and record why
  desktop interaction does not apply.
- Keep exploration separate from the baseline and settling observations. Record
  the initial state, targets, action order, parameters, timing/step counts when
  known, and screenshots around anomalies. If actions are randomly generated,
  retain the seed and generated action list; a seed alone cannot reproduce GUI
  timing. Release all held inputs and reset/relaunch before rechecking a
  suspected problem against the baseline. Do not excuse explosions or crashes
  just because exploration found them, or label expected extreme-input motion
  as a defect without investigating.
- Reproduce suspected defects from reset or a fresh process. Separate physics
  defects, rendering/capture artifacts, missing assets/dependencies, and unmet
  hardware requirements. Do not silently substitute CPU for requested CUDA.
- Close the viewer and confirm process termination before the next example.
  Release held buttons/modifiers even after failed input or capture. Continue
  independent cases when one is blocked; honor a user's stop/checkpoint scope.

## Report and hand off

Write the report and evidence outside the checkout unless the user specifies
otherwise. Record artifact version, wheel origin/hash or source commit, resolved
dependencies, Python/Warp versions, device/driver, renderer, display, and command
lines. Include a row for each requested example/variant with criteria source,
expected/observed behavior, status, and relative log/image/sequence paths.
Work example-by-example and update a persistent progress document after each
case/variant with completed checks, exploratory actions, evidence, unresolved
questions, and the next case. Keep criteria migration, static validation, and
executed visual QA as separate statuses so none is mistaken for the others.

Use **pass** only for criteria actually evaluated and satisfied; **fail** for an
observed criterion violation; **blocked** for an environment/input obstacle;
**not tested** for remaining coverage or undefined criteria. Report smoke-check
success separately from acceptance. Give coverage totals, sampling limitations,
reproduction steps for failures, and the next work needed. Resume from this
record without treating evidence from another artifact as a pass for this one.
The report supports QA/release decisions; invoking this skill does not publish
results, file issues, or approve a release.
