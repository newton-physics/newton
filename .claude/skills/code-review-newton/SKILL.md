---
name: code-review-newton
description: Use when reviewing a Newton pull request, branch, commit range, work-in-progress change, or design proposal for project fit, requirements fidelity, and coding and repository standards.
---

# Review Newton Changes

Read `REVIEW_GUIDELINES.rst` and `CODING_GUIDELINES.rst` in full. Treat the
former as the explicit definitions of the Fit, Requirements, and Standards
axes, and enforce the latter during Standards review.

1. Establish the exact change under review and read its complete diff and
   commit list.
2. Read the pull-request description, linked issue or specification, relevant
   primary sources, top-level discussion, and review threads. If no separate
   specification exists, use the pull-request description as the stated intent.
3. Review Fit, Requirements, and Standards independently using their
   definitions in `REVIEW_GUIDELINES.rst`. When independent agent contexts are
   available, use separate passes so one conclusion does not bias another.
4. Do not repeat concerns already raised in the top-level discussion or review
   threads; acknowledge them when they affect the verdict.
5. Report Fit and architectural concerns first. If the overall direction is in
   question, do not bury that discussion beneath minor implementation details.
6. For each remaining finding, state its priority, location, problem, impact,
   and evidence. Distinguish requirements from judgment calls and questions.
   Do not prescribe a correction unless the user asks for one.

Aggregate the three axes without suppressing a finding merely because another
axis passes. Give the explicit Fit verdict defined by the review guide.
