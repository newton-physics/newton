# Native viewer interaction

Use the run workspace and `NEWTON_SKILL` path from [setup.md](setup.md).
Capture actual pixels and inspect them with the agent's image tool before
choosing input coordinates. The helper uses `mss.MSS(backend="xlib")`: MSS 10.2's
default XCB backend failed on the demonstrated Kasm desktop with a drawable
visual error, while Xlib worked.

```bash
xdotool search --onlyvisible --name 'Newton Viewer'
# Set this to the window belonging to the current example; IDs are not stable.
NEWTON_WINDOW=123456
xdotool windowactivate --sync "$NEWTON_WINDOW"
xdotool getwindowgeometry --shell "$NEWTON_WINDOW"
uv run --frozen python "$NEWTON_SKILL/scripts/desktop.py" shot artifacts/before.png
```

Screenshots use top-left desktop coordinates; `mousemove --window` uses
window-relative coordinates. Decorations and scaling can change their offset.
Use fresh screenshots, not remembered coordinates. Activate the window rather
than clicking the scene to focus. ImGui can capture keyboard/mouse events over
widgets; move into an unobstructed viewport region before camera or key input.

```bash
# Illustrative coordinates only: adapt them to the current screenshot/window.
uv run --frozen python "$NEWTON_SKILL/scripts/desktop.py" drag \
  "$NEWTON_WINDOW" 2 1000 450 1130 510 --modifier Shift_L
uv run --frozen python "$NEWTON_SKILL/scripts/desktop.py" shot artifacts/after-pan.png
# Click a control at desktop coordinates obtained from the current image.
xdotool mousemove 35 156 click 1
xdotool mousemove --window "$NEWTON_WINDOW" 1100 450 key space
xdotool key period
```

The drag helper sends 20 intermediate motion events and releases the held
button/modifier in cleanup. Wait for fresh rendered frames after actions before
capturing evidence. Immediate screenshots can retain pre-action pixels.

## Controls and verification scope

These are the GL controls inspected in 1.7.0rc1; check the target version's
Controls panel/source if they differ. Each run must verify its own interactions.

| Action | Input | Initial procedure evidence |
| --- | --- | --- |
| Run/pause physics | Pause checkbox or Space | Checkbox demonstrated |
| Step paused physics | Step button or period | Period demonstrated |
| Reset physics | Reset button, when available | Demonstrated; camera stayed panned |
| Pan | Shift + middle drag | Demonstrated |
| Orbit | Middle drag | Source inspected |
| Dolly | Scroll or Ctrl + middle drag | Source inspected |
| Look | Left drag in viewport | Source inspected |
| Frame model | F | Source inspected |
| Toggle UI | H | Source inspected |
| Pick/manipulate | Right drag | Demonstrated on CUDA XPBD pendulum |
| Close | Escape in viewport | Demonstrated, exit status 0 |

**Pause** stops physics; **Pause Rendering** stops image updates. Do not confuse
them. Camera comparisons should use paused physics; settling checks should not.

## Picking

Use X11 button **3**, targeting a visible movable shape. Right press selects a
point; drag moves its target. The demonstrated viewer draws a cyan pick line.
Physics must advance for the applied force to move the object: while paused,
the target line changes but the body stays still. For a brief live drag:

```bash
uv run --frozen python "$NEWTON_SKILL/scripts/desktop.py" drag \
  "$NEWTON_WINDOW" 3 850 140 850 340
```

For a paused held-pick check, adapt these desktop coordinates and keep cleanup
in the same shell process. Do not run them blindly against another layout:

```bash
(
  trap 'xdotool mouseup 3' EXIT
  xdotool mousemove 850 190 mousedown 3 mousemove 850 390
  sleep 1
  uv run --frozen python "$NEWTON_SKILL/scripts/desktop.py" shot artifacts/pick-held.png
  xdotool key --repeat 40 --delay 70 period
  sleep 1
  uv run --frozen python "$NEWTON_SKILL/scripts/desktop.py" shot artifacts/pick-stepped.png
)
sleep 1
uv run --frozen python "$NEWTON_SKILL/scripts/desktop.py" shot artifacts/pick-released.png
```

Verify a hit/line, physical response, and removal of the line after release in
a fresh render. Reset and apply the same step inputs without picking to separate
drag response from natural motion. Counted key inputs are not guaranteed accepted
simulation steps: rapid events can coalesce. Use instrumented timing for exact
trajectory comparisons.

The initial CUDA XPBD pendulum check showed a target line while paused, a changed
pose after 40 period inputs with the pick held, and a visibly different pose
after reset with the same inputs and no pick. An immediate post-release screenshot
still showed the previous line: it did not validate visual release behavior.
This establishes manipulation for that configuration only, not all examples,
solvers, particles, or cloth. If manipulation fails, inspect hit location, body
mobility, picking enablement, and whether the example applies viewer forces
before classifying a picker defect.

## Cleanup

Release all held buttons and modifiers on failures. Close the active example:

```bash
xdotool windowactivate --sync "$NEWTON_WINDOW"
xdotool mousemove --window "$NEWTON_WINDOW" 1100 450 key Escape
```

Wait for process completion and record its exit status/log tail. If it fails to
close, terminate only the process launched for this case; do not kill unrelated
desktop sessions or examples.

The helper attempts mouse-button and modifier release independently; a failed
release still reports an error. Its failure paths can be checked without a
display, capture dependencies, or real input:

```bash
uv run --frozen python "$NEWTON_SKILL/tests/test_desktop.py"
```

This uses mocked input commands and is not a picking or visual QA test.
