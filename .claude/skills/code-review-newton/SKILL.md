---
name: code-review-newton
description: Use when reviewing a Newton pull request, branch, commit range, work-in-progress change, or design proposal for project fit, requirements fidelity, and coding and repository standards.
---

# Review Newton Changes

Read `REVIEW_GUIDELINES.rst` and `CODING_GUIDELINES.rst` in full. Treat the
former as the explicit definitions of the Fit, Requirements, and Standards
axes, and enforce the latter during Standards review.

1. Resolve the exact merge base and head. Read the complete
   `merge-base..head` diff and commit list.
2. Read the pull-request description, linked issue or specification, relevant
   primary sources, and `CODING_GUIDELINES.rst`. If no separate specification
   exists, use the pull-request description as the stated intent. Do not read
   existing review comments yet.
3. Perform an adversarial correctness-first Requirements pass. Trace changed
   inputs, state, counts, offsets, ownership, and outputs through every relevant
   supported execution mode. Check applicable zero/one/many and capacity
   boundaries, heterogeneous inputs, toggle/reset/reuse and partial failures,
   CPU/CUDA and backend differences, autodiff, determinism, graph capture, and
   numerical invariants. For performance changes, examine setup and steady-state
   complexity, allocations, synchronization, transfers, and whether benchmarks
   exercise the changed path. Run a minimal adversarial probe when feasible.
4. Freeze candidate findings before reading existing reviews. Record the
   location, triggering input or condition, mechanism, observable impact,
   evidence, and confidence for each candidate.
5. Read the top-level discussion and review threads. Refine or deduplicate the
   independently discovered concerns; acknowledge existing concerns when they
   affect the verdict.
6. Review Fit and Standards independently using their definitions in
   `REVIEW_GUIDELINES.rst`. When independent agent contexts are available, use
   separate passes so one conclusion does not bias another.
7. Report Fit and architectural concerns first, but retain every supported
   material behavioral, correctness, compatibility, and performance finding
   even when Fit fails. Do not let missing tests, documentation, or measurements
   replace the search for an underlying behavioral defect.
8. For each remaining concern, state its priority, location, problem, impact,
   and evidence. Distinguish demonstrated defects from questions, evidence
   requests, requirements, and judgment calls. Do not prescribe a correction
   unless the user asks for one.

Aggregate the three axes without suppressing a finding merely because another
axis passes. Give the explicit Fit verdict defined by the review guide.
