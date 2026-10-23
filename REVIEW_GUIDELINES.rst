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

.. _review-questions:

Questions for every change
--------------------------

Ask these of every change, including small bug fixes, in proportion to its
size. The Fit gates below add depth for large proposals.

#. **Simplest:** Is this the smallest change that solves the stated problem?
   Question each new parameter, mode, alias, public symbol, and file, and
   prefer an existing mechanism. Check whether ``main`` or another open pull
   request already covers the change, and whether unrelated parts should be
   split out.
#. **Principled:** Does it fix the cause at the right layer rather than a
   symptom, proxy, or workaround? Prefer designs that are correct by
   construction over detecting and reporting bad states, and upstream fixes
   when Warp or another dependency owns the behavior. Treat loosened
   thresholds, suppressed warnings, and special cases as prompts to look
   deeper.
#. **Honest contract:** Does every accepted input either take effect or get
   rejected? Look for silent fallbacks, partially honored options, and
   documentation, changelog entries, or tests that describe behavior the code
   does not implement.
#. **Siblings:** When the change fixes a defect or you find one, check
   sibling code paths, such as other joint types, entity kinds, importers,
   solvers, and duplicated helpers, for the same pattern.
#. **Tests:** Would each new test fail without the change, for the failure
   mode it claims? Could an existing test, class, or fixture be extended
   instead? Are the model, step count, and device matrix as small as
   possible?
#. **Cost:** What does the change add to the default path? Setup, import,
   memory, and steady-state costs need evidence even when the change is not
   about performance.

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

For authorable simulation features, apply the schema and importer policy in
:ref:`source-code-guidelines`: schema design should inform the API before merge.
Require implemented support, an explicitly accepted and tracked deferral, or a
reason this does not apply. A follow-up issue alone is not acceptance.

Confirm that schemas the change newly reads, authors, or models come from
sources the :ref:`source-code-guidelines` permit. Pre-existing sources are out
of scope for this policy review unless their removal or migration is an
explicit part of the task. Treat new support for a proprietary shading system,
such as MDL or OmniPBR, as a Fit failure about ownership and durable cost
rather than as an implementation detail.

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

For authorable features, trace schema parsing through the builder to runtime
behavior and check tests against equivalent builder-created models. Verify
schema semantics and the documented supported subset.

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

For test changes, explicitly assess :ref:`test-organization-and-runtime`.
Confirm that tests exercise the actual implementation and verify meaningful
technical results. Flag tests that replace the functionality under test with
mocks or only check fake interfaces, expected calls, or canned responses. Apply
the ``newton/tests/test_api.py`` exception only to targeted checks of actual API
contracts; it does not excuse mock-only tests or missing functional coverage.
Inspect related existing coverage and check whether new assertions or scenarios
could extend it, whether new solver-specific modules are placed correctly, and
whether related tests reuse an appropriate class, module, and fixture. Assess
redundant simulation work and added CI cost; new or expanded expensive scenarios
need before/after measurements and justification. Ground concerns in the
specific duplicated coverage, avoidable setup, or misplaced tests. Preserve
meaningful coverage, failure diagnosis, and test independence, and weigh setup
reuse against lost class-level parallelism when recommending consolidation.

For public API changes, check the canonical public path, stability status,
signature and defaults, behavior and return contract, documentation, and
generated API reference. Enforce the deprecation policy and migration guidance
for incompatible changes.

Check the repository obligations relevant to the change, including:

- focused regression coverage and documentation;
- schema/importer coverage, compatible schema-package versions, and deferrals;
- diagnostics that follow the error, warning, and logging rules, including
  Newton warning categories and module loggers instead of ``print()``;
- accurate, correctly categorized Towncrier fragments for user-facing changes;
- dependency and lockfile changes, licenses, and required notices;
- USD schemas the change newly reads, authors, or models, including whether
  they stay within the supported OpenUSD, Newton, and bounded solver-fallback
  tiers. Existing proposal schemas are grandfathered, as are legacy
  compatibility paths during their deprecation windows, including the
  ``omniphysics:`` and ``physxDeformableBody:`` deformable fallbacks on the
  default import path. Do not treat these as policy violations;
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
