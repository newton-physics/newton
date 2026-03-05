# CLAUDE.md

Guidance for cleanup and merge work in `newton/examples/cosserat_codex`.

@AGENTS.md

## Merge Strategy

Use a two-phase merge plan:

1. **Phase 1 (current):** Warp-only core solver + runnable minimum example.
2. **Phase 2 (later):** Catheter/guidewire (concentric) demo and advanced features.

Do not mix both phases in one PR.

## Phase 1 Requirements (Warp-Only)

### Required outcomes

- Keep a **runnable** example entrypoint in this folder.
- Support **Warp solver only**.
- Remove NumPy and DLL paths from runtime behavior.
- Keep code surface as small as possible while preserving correctness.

### Canonical runtime path (Phase 1)

`xcath.py` -> `simulation/example.py` -> Warp rod state + required kernels

If `xcath.py` is replaced, do it in the same PR with a working equivalent
entrypoint and updated run command.

### Scope to keep (initially)

- `xcath.py` (or direct replacement entrypoint)
- `cli.py` (simplified to Warp-only arguments)
- `constants.py`
- `math_utils.py` (only if still used)
- `rod/base.py`, `rod/config.py`, `rod/warp_rod.py`, `rod/__init__.py`
- `solver/xpbd.py`
- `kernels/*` that are directly used by Warp runtime path
- `simulation/example.py`

### Scope to remove or defer (Phase 1)

- `rod/dll_rod.py`
- `rod/numpy_rod.py`
- CLI args and parsing for `dll`/`numpy` solvers
- Multi-backend compare logic in `simulation/example.py`
- Unused helper modules/surfaces after Warp-only simplification

Only remove modules after import-site verification (`rg`) confirms no runtime
references.

## Phase 2 Scope (Deferred)

Treat these as out-of-scope for Phase 1 unless needed for minimum runnability:

- catheter/guidewire concentric demo behavior
- track insertion UX complexity
- advanced visualization/meshing extras
- mesh collision extras coupled to cosserat2

## Implementation Rules

- Do not import from `newton._src`.
- No new dependencies.
- Prefer removal over refactor when behavior is not required for Phase 1.
- Keep public behavior stable inside the chosen Phase 1 scope.
- If behavior changes for users, update `CHANGELOG.md` under `[Unreleased]`.

## Validation Gates (Phase 1 PR must pass)

- Minimal example runs end-to-end in headless/null mode.
- `Example.test_final()` passes.
- No remaining runtime references to NumPy/DLL rod implementations.
- `uvx pre-commit run -a` passes.

## Suggested Checks

```bash
# find remaining non-Warp solver references
rg -n "NumpyDirectRodState|DefKitDirect|SolverType\.NUMPY|SolverType\.DLL|--dll-path|--calling-convention"

# run minimal example
uv run -m newton.examples.cosserat_codex.xcath --viewer null --num-frames 32

# lint/format
uvx pre-commit run -a
```
