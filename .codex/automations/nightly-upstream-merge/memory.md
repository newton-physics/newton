# Nightly Upstream Merge Memory

Append one short dated entry after each run with:
- whether `origin/main` was updated
- which branches merged cleanly
- whether `research/pressure-field` validation passed
- any conflicts or skipped pushes that need follow-up

## 2026-04-25

- Ran `uv run --script scripts/nightly_upstream_merge.py --push --skip-main-sync`; `origin/main` was intentionally left unchanged.
- `protomotions` merged `upstream/main` cleanly and pushed `7a81c59ddf85` -> `3f417e4479ba`.
- `research/pressure-field` merged `upstream/main` cleanly, passed pressure-field validation, and pushed `8752d78deaae` -> `474515cb092d`.
- Validation commands: `uv run --extra dev -m newton.tests -k pressure`; `uv run --extra examples -m newton.examples.contacts.example_hydro_pressure_slice --test --viewer null --num-frames 1 --shape box`.
- No merge conflicts, validation failures, push failures, or queued repair items.

## 2026-04-26

- Ran `uv run --script scripts/nightly_upstream_merge.py --push --skip-main-sync`; `origin/main` was intentionally left unchanged.
- `protomotions` already contained `upstream/main`; no merge or push was needed (`3f417e4479ba`).
- `research/pressure-field` already contained `upstream/main`; no merge or push was needed (`474515cb092d`), and pressure-field validation passed.
- Validation commands: `uv run --extra dev -m newton.tests -k pressure`; `uv run --extra examples -m newton.examples.contacts.example_hydro_pressure_slice --test --viewer null --num-frames 1 --shape box`.
- No merge conflicts, validation failures, push failures, or queued repair items.

## 2026-04-27

- Ran `uv run --script scripts/nightly_upstream_merge.py --push --skip-main-sync`; `origin/main` was intentionally left unchanged.
- `protomotions` merged `upstream/main` cleanly and pushed `3f417e4479ba` -> `e5b18b4a3466`.
- `research/pressure-field` merged `upstream/main` cleanly, passed pressure-field validation, and pushed `474515cb092d` -> `7707b7f9bf74`.
- Validation commands: `uv run --extra dev -m newton.tests -k pressure`; `uv run --extra examples -m newton.examples.contacts.example_hydro_pressure_slice --test --viewer null --num-frames 1 --shape box`.
- No merge conflicts, validation failures, push failures, or queued repair items.
