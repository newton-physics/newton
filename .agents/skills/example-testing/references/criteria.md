# Example acceptance criteria

The module docstring contains the teaching description, common success criteria,
and run commands, with additional observations below the relevant command.

## Extract and plan

From the isolated artifact workspace prepared in [setup.md](setup.md):

```bash
uv run python -m newton.examples --describe basic_pendulum > basic_pendulum-spec.md
```

Repeat for each requested example. Extraction reads source with AST; it does not
import or run the example. Schedule the listed commands within the requested
scope, combining common criteria with each command's additional observations.
Do not apply a variant's special expectations to another. A common criterion
must be observable in every listed run. Additional supported CLI options are not
necessarily mandatory QA variants.

Save the extracted prose with results and record the criteria revision and user
overrides. Use the artifact environment's Python when running commands, and
check the example's prerequisites (assets, recording inputs, optional
dependencies). Apply [shared-qa.md](shared-qa.md) alongside scene-specific checks.

## Judge and maintain

Criteria describe intended behavior, not prior observations or guaranteed
automated assertions. A clean exit or passing `--test` is separate evidence.
Check every stated phase, especially settling after the final action. Driven
systems may legitimately keep moving. Use [jitter.md](jitter.md) for
consecutive-frame evidence; missing numeric tolerances are an uncertainty to
report, not permission to invent a threshold.

When updating a docstring, preserve teaching explanations, all documented
commands, and their applicable observations. Keep generic camera, closure,
exploration, and reporting procedures in this skill. Validate with:

```bash
uv run --extra dev -m newton.tests -k test_example_spec
```

Source-derived expectations still need visual review. Historical issues and
open questions are not blanket waivers; report observed violations and leave
unresolved acceptance decisions explicit.
