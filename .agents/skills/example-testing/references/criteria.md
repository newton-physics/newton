# Example acceptance criteria

Keep scene expectations beside the implementation: the module docstring supplies
its description and literal `_EXAMPLE_SPEC` supplies common criteria and run
variants. Do not maintain a second list here. The schema and JSON contract are
documented in `docs/guide/development.rst` (Embedded success criteria).

## Extract and plan

From the isolated artifact workspace prepared in [setup.md](setup.md):

```bash
uv run python -m newton.examples --describe basic_pendulum
uv run python -m newton.examples --describe basic_pendulum --format json > basic_pendulum-spec.json
```

Repeat for each name in the artifact's registry. Extraction parses source with
AST; it does not import or run the selected example. For schema version 1:

- `description` explains the scene; `success_criteria` applies to every listed run.
- Each `runs` entry adds ordered `args`, a rendered `command`, and any additional
  `success_criteria`. Schedule every listed variant within the requested scope.
- Combine common and run-specific criteria. Do not apply one variant's special
  expectations to another. Additional supported CLI options are not necessarily
  mandatory QA variants.
- Keep checks for a filtered scene or plot in the runs that display it. A common
  criterion must be observable in every listed run; hidden comparisons cannot
  be silently counted as passed.
- Use the artifact environment's Python to execute commands. When invoking a
  subprocess, pass the example name and `args` as an argument list; the rendered
  command is for human readability.

Save the extracted JSON with the results. Record metadata revision and explicit
user overrides. Check prerequisites described by the example (assets, recording
inputs, optional dependencies). Respect requested viewer/device coverage and
apply [shared-qa.md](shared-qa.md) on top of scene-specific checks.

## Older wheels without metadata

An older release may lack `--describe` or some current examples. Do not replace
the artifact import with the checkout to
obtain metadata. Use a separately pinned metadata checkout and read its files
statically. For example, run this from the wheel workspace, adjusting paths:

```bash
uv run python - <<'PY'
import importlib.util
from pathlib import Path
import sys

repo = Path('/path/to/pinned/newton')
name = 'basic_pendulum'
source = repo / 'newton/examples/basic/example_basic_pendulum.py'
module_spec = importlib.util.spec_from_file_location('qa_spec', repo / 'newton/examples/_spec.py')
module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = module
module_spec.loader.exec_module(module)
spec = module.load_example_spec(source)
if spec is None:
    raise SystemExit('No metadata: acceptance coverage is incomplete')
Path(f'{name}-spec.json').write_text(
    module.render_example_spec(name, spec, output_format='json'), encoding='utf-8'
)
PY
```

This loads the metadata reader only, not the checkout's simulation package or
example. Record the checkout commit (`git -C /path/to/pinned/newton rev-parse
HEAD`) and any local changes separately from wheel provenance. Compare names,
options, defaults, and expected scenes with the installed release. A renamed
example or changed scene needs an explicit mapping; never silently use current
main's expectations as historical truth. Keep examples absent from the wheel
out of its tested count, and retain wheel examples with missing metadata as
uncovered. Resolve incompatible criteria before judging the affected check.

## Judge and maintain

Criteria describe intended behavior, not previous observations or guaranteed
automated assertions. A clean exit or passing `--test` is separate evidence.
Check every stated phase, especially settling after the final action. Do not
require every scene to stop: driven systems may legitimately keep moving.
Use [jitter.md](jitter.md) for consecutive-frame evidence; missing numeric jitter
tolerances are an uncertainty to report, not permission to invent a threshold.

When extending an example's metadata, specify the initial state, required
inputs/actions, expected motion/contact/UI response, relevant terminal event or
simulation time, and rest versus smooth ongoing motion. Add tolerances with
units and observation intervals when established. Keep generic closure, camera,
exploration, and reporting procedures in this skill. Preserve important teaching
explanations in the module docstring. Validate with:

```bash
uv run --extra examples python -m unittest newton.tests.test_example_spec
```

Source-derived expectations for new examples still need visual review. Historical
issues and open questions are not blanket waivers; report observed violations
and leave unresolved acceptance decisions explicit.
