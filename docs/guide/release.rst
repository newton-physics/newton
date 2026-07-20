.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Release Process
===============

This document describes how to prepare, publish, and follow up on a Newton
release.  It is intended for release engineers and maintainers.

Overview
--------

Newton follows PEP 440 versioning; see :ref:`versioning` in the
compatibility guide for details.

Releases are published to `PyPI <https://pypi.org/p/newton>`__ and
documentation is deployed to
`GitHub Pages <https://newton-physics.github.io/newton/>`__.

Version source of truth
^^^^^^^^^^^^^^^^^^^^^^^

The version string lives in the ``[project]`` table of ``pyproject.toml``.
All other version references (PyPI metadata, documentation) are derived from
this file.  At runtime, ``newton/_version.py`` reads the version from
installed package metadata via ``importlib.metadata``.

Dependency versioning strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pyproject.toml`` generally specifies **minimum** compatible versions
(e.g. ``warp-lang>=1.12.0``).  ``uv.lock`` pins the **latest known-good**
versions for reproducible installs.

``mujoco`` and ``mujoco-warp`` instead use **compatible-release** pins on
both ``main`` and release branches (e.g. ``mujoco~=3.5.0``) to allow
``3.5.x`` updates while excluding ``3.6.0`` and later.  MuJoCo follows
`custom versioning from 3.5.0 onward`_; its third component is
``MINOR_OR_PATCH`` and guarantees API backward compatibility.

.. _custom versioning from 3.5.0 onward:
   https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md#from-350--semantic-versioning


Deprecation and removal timeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The user-facing deprecation and removal policy lives in
:ref:`deprecation-policy`.  Release engineers should ensure that every
deprecation, removal, or other breaking change in a minor release is
reflected in ``CHANGELOG.md`` and the API documentation, and that
deprecations emit a runtime ``DeprecationWarning`` where applicable.


Release workspace and progress record
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use dedicated Git worktrees once the release branch is cut: keep the primary
worktree on ``main`` and use separate worktrees for release-branch preparation,
stabilization, and GA changes.  This avoids switching a dirty worktree between
``main`` and ``release-X.Y`` and makes the source branch of every operation
explicit.

At the start of the release, create a top-level working checklist such as
``RELEASE_X_Y_PROGRESS.md``.  Keep it uncommitted unless the maintainers want
the operational record in the repository.  Update it after every branch, PR,
merge, workflow, tag, approval, and publication transition.  At minimum record:

- the target version, previous release, canonical remote, branch cut commit,
  release branch, and preparation PRs;
- selected and excluded backports, including source commits and PR numbers;
- validation commands, workflow run IDs, results, and approved exceptions;
- each tag's exact target commit and the PyPI approval state; and
- the GitHub Release and documentation URLs plus remaining owner actions.

Suggested sections are Release information, Pre-release planning, Branch
creation, one section per RC, Stabilization/backports, GA, and Post-release.
The checklist is the handoff record when release work spans multiple sessions
or maintainers; do not rely on chat history as the only state record.


Pre-release planning
--------------------

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - Determine target version (``X.Y.Z``).
   * - ☐
     - Read this guide and inspect the previous release branch, preparation
       PRs, and lightweight tags so the new branch/tag sequence follows the
       repository's established convention.  Record the chosen refs in the
       release progress checklist.
   * - ☐
     - Select release dependency versions and confirm their availability on
       public PyPI: warp-lang, mujoco, mujoco-warp, newton-usd-schemas.  Land
       release-ready versions on ``main`` before the branch cut when possible.
   * - ☐
     - Set timeline: code freeze → RC1 → testing window → GA.
   * - ☐
     - Conduct public API audit:

       - Review all new/changed symbols since the last release for unintended
         breaking changes.
       - Confirm intended public API changes and breaking changes have
         maintainer approval.
       - Verify deprecated symbols carry proper deprecation warnings and
         migration guidance (see :ref:`deprecation-policy`).
       - Verify experimental public API is explicitly marked with
         ``.. experimental::`` in API or concept documentation, following the
         :ref:`developer guidance <experimental-features>`.
       - Confirm new public API has complete docstrings and is included in
         Sphinx docs (run ``uv run docs/generate_api.py``).

       Run the ``release-audit`` Claude Code skill
       (``.claude/skills/release-audit``) in **pre-release mode** to automate
       this audit.


Code freeze and release branch creation
---------------------------------------

Fetch the canonical ``upstream`` remote immediately before cutting the branch.
Create the release branch from the verified ``upstream/main`` commit, record
that commit as the branch point, and push the branch to the canonical
repository.  Use feature branches and pull requests for subsequent release
changes; never commit directly to ``main`` or ``release-X.Y``.

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - Create ``release-X.Y`` from the verified ``upstream/main`` branch point
       in a dedicated worktree, record the exact commit, and push it.
   * - ☐
     - On **main**: bump the version in ``pyproject.toml`` to ``X.(Y+1).0.dev0`` and run
       ``uv run docs/generate_api.py``, then regenerate ``uv.lock`` (``uv lock``).
   * - ☐
     - On **release-X.Y**: bump the version in ``pyproject.toml`` to ``X.Y.ZrcN`` and
       run ``uv run docs/generate_api.py``, then regenerate ``uv.lock`` (``uv lock``).
   * - ☐
     - On **release-X.Y**: update dependencies in ``pyproject.toml`` from dev
       to RC or stable versions where applicable and remove the NVIDIA package
       index (``[[tool.uv.index]]`` entry for ``nvidia`` **and** the
       ``warp-lang`` entry in ``[tool.uv.sources]`` that references it) so the
       release wheel installs purely from PyPI.  Update the Warp install command
       in ``asv.conf.json`` to the same stable release from public PyPI, without
       ``--pre`` or the NVIDIA index.  Then regenerate ``uv.lock`` (``uv lock``)
       and commit.
   * - ☐
     - Run the ``release-audit`` skill in **release-candidate mode** against
       ``release-X.Y``; address or acknowledge flagged entries before
       tagging.
   * - ☐
     - Manually trigger the **minimum-dependency** and **multi-GPU** CI
       workflows on the ``release-X.Y`` branch (the nightly orchestrator
       only runs on ``main``).  Verify both pass before tagging.

       .. code-block:: bash

          # Minimum-dependency tests (lowest compatible PyPI versions)
          gh workflow run minimum_deps_tests.yml --ref release-X.Y

          # Multi-GPU tests (g7e.12xlarge = 4× L40S GPUs)
          gh workflow run aws_gpu_tests.yml --ref release-X.Y \
              -f instance-type=g7e.12xlarge
   * - ☐
     - Push tag ``vX.Y.Zrc1``.  This triggers the ``release.yml`` workflow
       (build wheel → PyPI publish with manual approval).
   * - ☐
     - RC1 published to PyPI (approve in GitHub environment).


Release candidate stabilization
-------------------------------

Bug fixes merge to ``main`` first, then are cherry-picked to
``release-X.Y``.  Cherry-pick relevant commits from ``main`` onto a feature
branch and open a pull request targeting ``release-X.Y`` — never push
directly to the release branch.

Before selecting backports, fetch the latest ``upstream/main`` and enumerate
all commits since the recorded branch point.  PRs assigned to the release
milestone are mandatory unless maintainers explicitly decide otherwise.
Review unmarked, later-release, dependency, and revert commits with the
maintainers, then record the approved set and explicit exclusions in the
release progress checklist before changing the release branch.

Apply approved commits sequentially in their order on ``main``.  Keep one
source commit per cherry-pick commit and preserve the original subject, PR
number, and source SHA (``git cherry-pick -x`` provides the source trailer).
Do not squash the backport PR: the one-to-one history is the audit trail.  When
an approved range contains a temporary change and its later revert, either
include both in order or exclude both; never include only one side.  Prefer
cherry-picks over a bulk merge because ``main`` normally already contains the
next development version and unrelated post-cut work.  Use a bulk merge only
when every intervening commit has been explicitly approved.

The final changelog is an exception to the main-first flow.  Prepare it from
the current ``release-X.Y`` branch and merge its dedicated pull request
directly into ``release-X.Y``.  Reconcile the dated release section back to
``main`` after the release, as described in :ref:`post-release`.

For each new RC (``rc2``, ``rc3``, …) bump the version in
``pyproject.toml``, run ``uv run docs/generate_api.py``, and regenerate
``uv.lock`` (``uv lock``).  Run pre-commit, focused tests, and a clean package
build before opening the RC preparation PR.  After it merges, run the required
release-branch workflows on the merge commit.  Create and push the next
lightweight RC tag only after the required validations pass, and record the tag
target and workflow run IDs.  Resolve any cherry-pick conflicts or missing
dependent cherry-picks that cause CI failures before tagging.

ASV comparison jobs execute the benchmark definition against both the base and
head revisions.  A newly added benchmark can therefore fail on the base when
it calls an API that exists only on the head.  Do not ignore the job result:
inspect the per-revision logs, confirm that the head benchmark completed, and
document the base incompatibility in the progress checklist and PR.  Treat any
head failure or unexplained comparison failure as a real release failure.

.. _testing-criteria:

Testing criteria
^^^^^^^^^^^^^^^^

The release engineer and maintainers decide which issues must be fixed
before GA and which can ship as known issues documented in the release
materials.  Features marked experimental have a lower bar —
regressions in experimental APIs do not necessarily block a release.

As a guideline, an RC is typically ready for GA when:

- All examples run without crashes, excessive warnings, or visual
  artifacts (``uv run -m newton.examples <name>``).
- Testing covers **Windows and Linux**, **all supported Python versions**,
  and both **latest and minimum-spec CUDA drivers** (see
  :ref:`system requirements <system-requirements>` in the installation guide).
- PyPI installation of the RC works in a clean, isolated ``uv`` environment:
  dependency resolution succeeds, ``import newton`` works, and package
  metadata plus ``newton.__version__`` both report ``X.Y.ZrcN``.
- No unexpected regressions compared to the previous release have been
  identified.

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - All release-targeted fixes cherry-picked from ``main``.
   * - ☐
     - Re-run the ``release-audit`` skill after final cherry-picks; confirm
       no new flags since the last RC.
   * - ☐
     - Prepare draft GitHub Release notes: summary, a few highlights, link
       to ``CHANGELOG.md``, acknowledgments.
   * - ☐
     - :ref:`Testing criteria <testing-criteria>` satisfied.
   * - ☐
     - No outstanding release-blocking issues.


.. _final-release:

Final GA release
----------------

Before proceeding, obtain explicit go/no-go approval from the
maintainers.  Do not start the final release steps until sign-off is
confirmed.

All steps below are performed on the **release-X.Y** branch unless noted
otherwise.

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - Go/no-go approval obtained from maintainers.
   * - ☐
     - Finalize ``CHANGELOG.md`` in a fresh worktree based on the current
       ``release-X.Y`` branch.  Run the
       ``.claude/skills/release-changelog`` skill with the previous GA or
       micro-release tag, the current release branch, and the latest
       ``release-audit`` report as inputs.  Rename ``[Unreleased]`` to
       ``[X.Y.Z] - YYYY-MM-DD`` using the actual GA date, updating the date if
       the schedule changes.  Open a dedicated pull request targeting
       ``release-X.Y`` and merge it before creating the final version and tag;
       do not merge it to ``main`` first and cherry-pick it back.
   * - ☐
     - Update ``README.md`` documentation links to point to versioned URLs
       (e.g. ``/X.Y.Z/guide.html`` instead of ``/latest/``).
   * - ☐
     - Verify all dependency pins in ``pyproject.toml`` use stable
       (non-pre-release) versions.
   * - ☐
     - Bump the version in ``pyproject.toml`` to ``X.Y.Z`` (remove the RC suffix) and
       run ``uv run docs/generate_api.py``.
   * - ☐
     - Regenerate ``uv.lock`` (``uv lock``) after all ``pyproject.toml``
       changes and verify that no pre-release dependencies remain in the lock
       file.
   * - ☐
     - Run pre-commit, focused release tests, and a clean wheel/source build.
       Verify the built metadata reports exactly ``X.Y.Z``.
   * - ☐
     - Confirm programmatically that ``X.Y.Z`` is unused on PyPI and that
       ``vX.Y.Z`` does not exist locally or on the canonical remote.
   * - ☐
     - Merge the GA preparation PR, then create lightweight tag ``vX.Y.Z`` at
       that exact merge commit.  Verify the tag target before pushing it to the
       canonical repository.  Automated workflows trigger:

       - ``release.yml``: builds wheel, publishes to PyPI (requires manual
         approval), creates a draft GitHub Release.
       - ``docs-release.yml``: deploys docs to ``/X.Y.Z/`` and ``/stable/``
         on gh-pages, updates ``switcher.json``.
   * - ☐
     - In the tag-triggered Release workflow, open the waiting **Publish Python
       distribution to PyPI** job, choose **Review deployments**, select the
       ``pypi`` environment, and approve it.  Verify publication with a clean,
       isolated ``uv`` install.
   * - ☐
     - Review the draft GitHub Release notes before publishing.  Keep them
       concise: summary, a few highlights, link to ``CHANGELOG.md``,
       acknowledgments.
   * - ☐
     - GitHub Release un-drafted and published.
   * - ☐
     - Docs live at ``/X.Y.Z/`` and ``/stable/``: verify links and version
       switcher.
   * - ☐
     - Release announcement posted.

List the versions known to PyPI before tagging with its JSON API rather than
relying on the package web page:

.. code-block:: bash

   uv run --no-project --isolated --python 3.12 python -c \
     "import json, urllib.request; print(sorted(json.load(urllib.request.urlopen('https://pypi.org/pypi/newton/json'))['releases']))"

After approval, verify that the published artifact is actually imported from a
clean environment rather than from the release worktree:

.. code-block:: bash

   uv run --no-project --isolated --python 3.12 --with newton==X.Y.Z \
     python -c "import importlib.metadata as m, newton; print(m.version('newton')); print(newton.__version__); print(newton.__file__)"

The documentation workflow may finish before the GitHub Pages CDN and browser
caches update.  If the workflow passed but ``/stable/`` still shows the prior
release, wait a few minutes and retry the public versioned page, stable page,
and ``switcher.json``.  If needed, inspect the canonical ``gh-pages`` branch to
distinguish a deployment failure from propagation delay.


.. _post-release:

Post-release
------------

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - Reconcile the released changelog back to ``main`` in a dedicated
       changelog-only pull request.  Start a dedicated feature branch from the
       latest ``upstream/main`` and run the
       ``.claude/skills/release-changelog`` skill using the dated
       ``[X.Y.Z] - YYYY-MM-DD`` section from ``release-X.Y`` and the current
       ``main`` changelog as inputs.  Preserve a fresh ``[Unreleased]`` section
       and every post-cut entry not shipped in ``X.Y.Z``; do not replace
       ``main``'s changelog wholesale.  Target this pull request to ``main``.
   * - ☐
     - Verify PyPI installation works in a clean environment.
   * - ☐
     - Verify published docs render correctly.


Micro releases
--------------

Micro releases continue cherry-picking fixes to the existing
``release-X.Y`` branch.  For example, ``1.0.1`` follows ``1.0.0``.
Follow the same :ref:`final-release` flow — bump version, update changelog,
tag, and push.  There is no need to create a new branch or bump ``main``.
