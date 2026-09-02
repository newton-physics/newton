.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _review-guidelines:

Review Guidelines
=================

This document provides suggested criteria for reviewing Newton changes,
particularly for AI-assisted reviews. Human reviewers may use their own
process and judgment. The criteria organize substantial concerns into three
independent axes: **Fit**, **Requirements**, and **Standards**.

Review axes
-----------

- **Fit:** Should Newton own this outcome, is the proposed design the right
  approach, and is its durable cost proportionate?
- **Requirements:** Does the change faithfully and correctly implement the
  agreed behavior, without omissions or unrelated scope, and is that behavior
  supported by suitable evidence?
- **Standards:** Does the change follow the
  :ref:`source-code-guidelines` and other repository requirements, including
  compatibility and release-facing obligations?

Passing one axis does not compensate for failing another. In particular, an
implementation can follow every coding standard while solving the wrong
problem, or implement the specification while introducing an unsuitable
public contract.

Review feedback
---------------

Separate discovery order from reporting order. When practical, first inspect
the exact change and perform a correctness-first Requirements pass before
evaluating Fit or reading existing review comments. Then use the existing
discussion to refine or deduplicate the independently discovered concerns.

In the final review, raise project-Fit and architectural concerns before
detailed implementation findings. When the overall direction is in question,
avoid obscuring that discussion with minor observations. Still report every
supported material behavioral, correctness, compatibility, and performance
finding even when the change does not Fit.

For each concern, state the problem, its impact, and the supporting evidence.
Distinguish a documented requirement from a judgment call or open question.
Prefer opening a design discussion to prescribing a particular correction;
the author may identify a better resolution. Do not repeat a concern already
raised in the pull-request discussion or another review thread.

Use three priorities:

- **P0:** A merge-blocking concern, such as incorrect behavior, an unsupported
  breaking change, or a fundamental Fit or requirements failure.
- **P1:** A material concern that should normally be addressed before merge.
- **P2:** A non-blocking observation or question worth considering.

Fit
---

Judge whether the outcome belongs in Newton, has the right owner, and is
proportionate. Keep Fit separate from implementation correctness, coding
standards, and requirements fidelity.

For a large, cross-cutting, or API-expanding proposal, ask whether the outcome
is valuable enough to justify its durable cost, whether this is the right
component or API, and whether a materially different approach could achieve
the outcome more simply. Consider existing interfaces, composition, private
or experimental seams, documentation, tooling, simpler workflows, and
upstream or downstream ownership. Do not assume the implementation named by a
proposal or specification is the only solution.

Apply four gates:

- **Product value, standalone surface, and ownership:** Determine whether the
  change advances robot learning, a validated robotics workflow, or Newton's
  independent discovery, installation, and use. Identify whether the core
  library or a downstream integration, application, importer, renderer, or
  dependency should own it. Downstream demand is evidence, but does not
  automatically make core Newton the right owner.
- **Coherent Newton capability:** Prefer deepening a canonical Newton concept
  over parallel machinery or new vocabulary. Follow Newton's existing design
  and terminology. Where Newton has no established concept, compare durable
  designs in other physics engines, robotics libraries, or relevant open
  standards. For a large change, compare at least one materially different
  design.
- **Public contract and total cost:** Seek the smallest coherent slice. Count
  API, new defaults, compatibility, documentation, tests, CI, dependencies,
  migration, support, review cost, and blast radius. Prefer an existing API,
  composition, private implementation, or bounded experiment before adding a
  public commitment. User-facing behavior should work through the public
  package in a normal checkout unless an integration boundary is explicit.
  Make the normal path difficult to use incorrectly and avoid APIs whose
  obvious use is slow, brittle, or unmaintainable.
- **Executable evidence and sustained capacity:** Require a named user or
  workload with measurable failure and success criteria plus executable
  validation. Ask whether Newton would accept the change if the contributor
  left, who can maintain and validate it, and which existing priority will
  lose capacity.

For new or changed user-authorable configuration, require an explicit USD
schema disposition: reuse an existing USD schema; add or track a USD schema
with lifecycle ownership, importer coverage, and runtime coverage; or explain
why the concept is intentionally runtime-only or non-authorable.

Return one explicit verdict:

- **Fits:** The value, owner, evidence, scope, and approach are proportionate.
- **Fits if narrowed:** Constrain the ownership, surface, scope, cost, or
  design before accepting the change.
- **Does not fit:** The justification is weak, the owner or layer is wrong, a
  clearly better approach exists, the cost is disproportionate, or validation
  and maintenance are unsustainable.

Ask for missing evidence rather than inventing it.

Requirements
------------

Treat the linked issue, PRD, design document, or agreed pull-request scope as
the intended outcome, not as proof that the proposed implementation is
correct or belongs in Newton. When no separate specification exists, use the
pull-request description as the stated intent.

Check for:

- requirements that are missing, partial, or implemented incorrectly;
- behavior or machinery that was not requested and adds risk or maintenance
  scope;
- undocumented changes to defaults, semantics, errors, ordering, supported
  inputs, or outputs; and
- inconsistencies between the implementation, documentation, tests, and
  stated behavior.

Review the relevant invariants, boundary conditions, failure paths, state and
resource ownership, and supported execution modes. Pay particular attention
to solver, integrator, collision, and math semantics; device and backend
differences; asynchronous execution and graph capture; batching and world
isolation; determinism and ordering; serialization and importers; and
performance-sensitive allocation, synchronization, and data transfer.

Require evidence in proportion to the claim and risk. Tests should demonstrate
observable behavior and meaningful regression boundaries rather than merely
repeat the implementation. Performance, scalability, determinism, numerical
accuracy, and compatibility claims need representative measurements or
primary-source evidence with enough context to interpret them. Missing
evidence should prompt a question. Treat its absence as a finding when the
change cannot be reviewed safely without it.

Missing tests, documentation, or measurements do not substitute for identifying
an underlying behavioral defect. Distinguish demonstrated defects from
questions and requests for evidence.

Do not silently reinterpret the intended behavior to match the implementation.
A better but different design can pass Fit while still requiring agreement to
change the requirements.

Standards
---------

Apply the :ref:`source-code-guidelines` to every code and public-API change.
They are authoritative for public exports, namespaces, naming, signatures,
types, documentation, testing, compatibility, and repository conventions.

For public API changes, check the canonical public path, stability status,
signature and defaults, behavior and return contract, documentation, and
generated API reference. Enforce the deprecation policy and migration guidance
for incompatible changes.

Check the repository obligations relevant to the change, including:

- focused regression coverage and documentation;
- accurate, correctly categorized Towncrier fragments for user-facing changes;
- dependency and lockfile changes, licenses, and required notices;
- packaging, supported platforms, workflows, and downstream compatibility;
  and
- release-visible semantic changes, including numerical behavior that changes
  without a Python signature change.

Treat recognized code and design smells as prompts for closer inspection, not
automatic violations. Explain the concrete Newton-specific cost rather than
reporting a smell label alone. A documented Newton design takes precedence over
a generic heuristic.

The release audit remains the final cross-release reconciliation. Standards
review should catch obligations visible in an individual change early, without
trying to reproduce release-level stabilization or retrospective analysis.
