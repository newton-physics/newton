---
name: code-review-newton
description: Use when reviewing a Newton pull request, branch, commit range, work-in-progress change, or design proposal for project fit, requirements fidelity, and coding and repository standards.
---

# Review Newton Changes

Read `REVIEW_GUIDELINES.rst` and `CODING_GUIDELINES.rst` in full. Treat the
former as the explicit definitions of the Fit, Requirements, and Standards
axes, and enforce the latter during Standards review.

1. Resolve the exact merge base and head. Read the commit messages, the
   pull-request description, linked issue or specification, and relevant
   primary sources. If no separate specification exists, use the description,
   or the commit messages when there is none, as the stated intent. Do not
   read the diff or existing review comments yet.
2. Sketch from the stated problem the smallest change that would fix it at
   its cause: the layer it belongs in, the existing mechanisms it would reuse,
   and the code, public surface, and tests it would need. Freeze this baseline
   for step 7.
3. Read the complete `merge-base..head` diff and `CODING_GUIDELINES.rst`.
   Check the changes on `main` since the merge base for semantic conflicts,
   overlapping fixes, and changed assumptions, even when Git reports a clean
   merge.
4. Perform an adversarial correctness-first Requirements pass. Trace changed
   inputs, state, counts, offsets, ownership, and outputs through every relevant
   supported execution mode. Check applicable zero/one/many and capacity
   boundaries, heterogeneous inputs, toggle/reset/reuse and partial failures,
   CPU/CUDA and backend differences, autodiff, determinism, graph capture, and
   numerical invariants. For performance changes, examine setup and steady-state
   complexity, allocations, synchronization, transfers, and whether benchmarks
   exercise the changed path. Answer the Honest contract, Siblings, Tests, and
   Cost questions in `REVIEW_GUIDELINES.rst`. Run a minimal adversarial probe
   when feasible, focused on failure modes specific to the changed domain.
   Apply the guides' schema and importer checks to authorable
   features, including behavioral equivalence and explicit acceptance of
   deferrals.
5. Freeze candidate findings before reading existing reviews. Record the
   location, triggering input or condition, mechanism, observable impact,
   evidence, and confidence for each candidate.
6. Read the top-level discussion and review threads. Refine or deduplicate the
   independently discovered concerns; acknowledge existing concerns when they
   affect the verdict.
7. Review Fit and Standards independently using their definitions in
   `REVIEW_GUIDELINES.rst`. Answer Simplest and Principled by comparing the
   change with the step 2 baseline: list each parameter, mode, public symbol,
   special case, file, and test the baseline does not need, and judge whether
   the change justifies it in its description, commits, or code. If the diff
   or discussion shows the baseline would not work, amend it and state why.
   When independent agent contexts are available, use separate passes so one
   conclusion does not bias another.
8. Report Fit and architectural concerns first, but retain every supported
   material behavioral, correctness, compatibility, and performance finding
   even when Fit fails. Do not let missing tests, documentation, or measurements
   replace the search for an underlying behavioral defect.
9. For each remaining concern, state its priority, location, problem, impact,
   and evidence. Distinguish demonstrated defects from questions, evidence
   requests, requirements, and judgment calls. Do not prescribe a correction
   unless the user asks for one, except that a Simplest or Principled concern
   should describe the smaller or deeper alternative, phrased as a question.

Aggregate the three axes without suppressing a finding merely because another
axis passes. Give the explicit Fit verdict defined by the review guide, then
answer each question for every change in `REVIEW_GUIDELINES.rst` in one line
with its evidence, or "n/a" when it does not apply. Answering "yes" to
Simplest requires naming the alternative you compared against. An output
format set by the caller takes precedence.

A re-review repeats every step on the complete current diff; step 6 is where
it checks whether earlier findings were addressed. Follow the caller when it
limits the scope, as the automated follow-up review does.
