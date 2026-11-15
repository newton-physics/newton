# Environment and artifact setup

## Software

The reference path needs uv, a Python version supported by the target release,
Newton's `examples` extra, and a working desktop/OpenGL implementation. The
bundled desktop helper additionally needs Python `mss` and the `xdotool` binary.
`mss` is an MIT-licensed QA tooling dependency, not a Newton runtime dependency.
Use `mss==10.2.0` for the helper's Xlib backend. No browser automation,
Pillow, or video encoder is required for single screenshots and input.

On Linux X11, diagnostic tools are `glxinfo` (Ubuntu package `mesa-utils`) and
`nvidia-smi` supplied by the NVIDIA driver. `xdotool` is a separate system package.
Use the existing desktop/session and its credentials. Do not replace a working
display server. On a host missing tools, install only the needed packages using
its normal package manager and applicable permissions. Example Ubuntu command:

```bash
sudo apt-get install xdotool mesa-utils
```

If uv is absent, use its official installation instructions:
https://docs.astral.sh/uv/getting-started/installation/
An agent needs an image inspection capability in addition to executing commands.
Wayland, macOS, and Windows need a separately verified capture/input path; this
helper is X11-specific.

## Fresh wheel workspace

Choose the release from the user's instruction and replace the version below.
Run outside the source checkout to avoid importing local `newton` instead of the
wheel. Use a fresh
directory per artifact/run, not a reused environment from another release.

```bash
NEWTON_VERSION="X.Y.Z"
NEWTON_REPO=/absolute/path/to/newton
NEWTON_SKILL="$NEWTON_REPO/.agents/skills/example-testing"
NEWTON_RUN=$(mktemp -d -t newton-qa-XXXXXXXX)
cd "$NEWTON_RUN"
uv init --bare --python 3.12
uv python pin 3.12
uv add "newton[examples]==$NEWTON_VERSION" 'mss==10.2.0'
mkdir artifacts
uv run --frozen python -c 'import sys, importlib.metadata as m, newton; print(sys.version); print(m.version("newton")); print(newton.__file__)' > artifacts/provenance.txt
uv pip freeze > artifacts/dependencies.txt
uv run --frozen python -m newton.examples --list > artifacts/examples.txt
```

Keep `pyproject.toml`, `.python-version` and `uv.lock`; the lock records the wheel
URL and hash. Verify the requested version and import path in provenance and
that resolution selected a wheel. For a supplied local wheel, install that exact
file and record its SHA256 instead of resolving a PyPI version. Do not copy a
historic lock into a new release test without recording that dependency choice.
To reproduce this environment use `uv sync --frozen` from its workspace.

If explicitly asked to test source instead, use a separate checkout/environment,
record commit and dirty state, and label it a source run. Do not call it wheel QA.

## Device and launch checks

```bash
nvidia-smi
glxinfo -B
xdotool getdisplaygeometry
uv run --frozen python -c 'import warp as wp; wp.init(); print(wp.get_devices())'
uv run --frozen python -m newton.examples basic_pendulum --help > artifacts/help.txt
uv run --frozen python -u -m newton.examples basic_pendulum \
  --device cuda:0 --viewer gl --paused --render-fps 30 \
  > artifacts/example.log 2>&1
```

Launch the last command as a retained process/tool session so the agent can
interact while it runs and collect its exit status. Read each target example's
help and source for additional dependencies/assets and solver options. Do not
assume `examples` installs every optional viewer/backend. Check actual Warp
device selection in the log, not merely the launch argument.

Record both the simulation device and OpenGL renderer: CUDA simulation can run
with software rendering, whose FPS is not GPU rendering performance. Follow the
environment's permission mechanism if device or display access is denied; a
sandbox denial is not proof of absent hardware. First-run kernel compilation
can delay startup; inspect logs before declaring a hang. Verify the target
viewer's frame-limit and shutdown behavior rather than assuming a windowed
viewer will stop after `--num-frames`.
