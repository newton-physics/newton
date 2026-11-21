# Parallel visual QA

Use independent subagents to cover separate examples while a coordinator owns
the full inventory, checklist, findings, and any authorized spec corrections.
Start with up to three workers, each handling one example and all its required
variants before taking another. Keep dependent cases together: for example,
recording must produce the artifact subsequently inspected by replay_viewer.

## Isolation and capacity

Each worker needs its own display/input session, viewer process, output subtree,
and case report. Never let workers drive the same desktop or reuse window IDs
across sessions. A read-only pinned checkout/environment may be shared; do not
install or update dependencies while other workers use it. Keep the artifact
and criteria revisions explicit when reporting spec corrections separately.

For Linux with no spare desktops, Xvfb is an optional system dependency
(`sudo apt-get install xvfb` on Ubuntu). Retain each server as a managed process
and use a free display number, not one belonging to an existing desktop:

```bash
NEWTON_DISPLAY=:100
Xvfb "$NEWTON_DISPLAY" -screen 0 1280x900x24 -nolisten tcp
```

Run this in its own retained tool session. Give each worker its assigned display
explicitly in every launch, capture, and input environment. Verify GL and the
requested simulation device on that display. For software GL, `LP_NUM_THREADS=1`
and `--render-fps 20` are useful starting limits, not requirements for other
renderers. Shared GPU memory and CPU capacity still limit parallelism. Reduce
concurrency if contention prevents useful advancing captures; do not attribute
contended rendering performance to a simulation defect.

Bare Xvfb has no window manager. The bundled drag helper uses `windowactivate`,
which may fail there. Use direct X11 focus/input instead, after inspecting the
current window and image:

```bash
export DISPLAY="$NEWTON_DISPLAY"
xdotool search --onlyvisible --name 'Newton Viewer'
# Use the discovered ID on this display.
NEWTON_WINDOW=2097162
xdotool windowfocus "$NEWTON_WINDOW"
xdotool windowsize "$NEWTON_WINDOW" 1280 900
```

Follow the button/modifier sequences in [interaction.md](interaction.md), using
incremental mouse movements and cleanup that releases held inputs. Verify every
action in fresh images. The viewer can initially exceed the Xvfb screen size;
resize before interpreting clipping or choosing coordinates. Use the assigned
display for the screenshot helper too. Close only the worker's own viewer and
server processes; never use global process-name kills.

## Worker assignment and handoff

Give each worker the pinned artifact/environment, skill, common checks, exact
example and required variants, exclusive display, and permitted output paths.
Workers write only their own report/evidence; they do not edit the shared
checklist, simulation, example implementation, or CI assertions. Proposed spec
corrections, defects, and CI improvements are text findings for coordinator
review. CI proposals must avoid extra device/host transfers or synchronization,
significant runtime increases, and test boilerplate in educational examples;
prefer existing snapshots, predicates, or shared test helpers.

Require an observed result for every criterion/variant, commands and action
order, actual inspected images, advancing late-motion timing, exploratory
interaction/picking results, limits, defect reproduction, CI recommendations,
and clean process termination. A worker's final message is a handoff, not
sufficient evidence of a pass. Missing or blocked criteria remain explicit.

## Coordinator loop

Review each handoff against the requested variants. Inspect representative
images and consecutive advancing frames yourself, especially for claimed
failures, picking responses, and jitter; distinguish capture cadence from
render cadence and simulation time. Evaluate CI proposals against the source
and the no-extra-transfer constraint. Request missing evidence or reproduction
before accepting an unsupported conclusion.

Update the master checklist and findings after review, retaining failures and
coverage limitations. Make only authorized, evidence-backed spec corrections,
without changing the pinned test artifact. Reuse or close finished workers and
assign remaining unowned cases, maintaining the concurrency limit. Persist
worker IDs, assignments, displays, and the next work so context changes cannot
lose ownership. Continue waiting/reviewing/reassigning until the requested
campaign is assessed; do not end merely because workers are still running.
