# Example acceptance criteria

No example-specific release criteria are defined yet. This file is the extension
point for QA and release maintainers; procedure demonstrations are not criteria.
Until entries are added, ask for expectations and report acceptance coverage as
incomplete. Do not require every scene to settle: driven systems can legitimately
continue moving.

Add a section keyed by the exact example CLI name and applicable solver/backend
variant. Each entry should specify:

- Applicable releases, command-line options, device and asset requirements.
- Initial visible state and the scenario/action sequence to exercise.
- Expected motion, contacts, UI changes, and picking behavior where relevant.
- Terminal event or simulation time, settling interval, observation duration,
  and whether the expected end behavior is rest or smooth ongoing motion.
- Required camera views, inspected objects, sampling method and numerical or
  visual tolerances with units. Explain acceptable intentional oscillations.
- Pass/fail conditions and required evidence; known limitations with scope and
  issue references, rather than blanket exclusions.

Record the criteria revision (repository commit plus any local changes) and
user overrides in each run. Resolve contradictory expectations before judging
the affected check. Keep unknown or unobserved behavior explicit.
