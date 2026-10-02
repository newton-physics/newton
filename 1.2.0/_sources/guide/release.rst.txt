.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Release Process
===============

This document describes how to prepare, publish, and follow up on a Newton
release.  It is intended for release engineers and maintainers.

Overview
--------

Newton follows PEP 440 versioning as described in the
:ref:`versioning <versioning>` section of the installation guide.

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

``pyproject.toml`` specifies **minimum** compatible versions
(e.g. ``warp-lang>=1.12.0``).  ``uv.lock`` pins the **latest known-good**
versions for reproducible installs.

Exception: on the **release branch**, ``mujoco`` and ``mujoco-warp`` use
**compatible-release** pins (e.g. ``mujoco~=3.5.0``) to allow patch
updates while locking the minor version.  MuJoCo follows
`semantic versioning from 3.5.0 onward <https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md#from-350--semantic-versioning>`__,
so patch releases are safe to pick up automatically.  ``main`` uses a
version floor like other dependencies.


.. _deprecation-timeline:

Deprecation timeline
^^^^^^^^^^^^^^^^^^^^

Following `Warp's deprecation policy
<https://nvidia.github.io/warp/user_guide/compatibility.html#deprecation-timeline>`__,
a deprecated feature is maintained for **two full minor release cycles**
after deprecation (e.g. deprecated in 1.2.0 → removed in 1.4.0).
Deprecations and removals only happen in minor releases, never in patch
releases.


Pre-release planning
--------------------

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - Determine target version (``X.Y.Z``).
   * - ☐
     - Confirm dependency versions and availability: warp-lang, mujoco,
       mujoco-warp, newton-usd-schemas.
   * - ☐
     - Set timeline: code freeze → RC1 → testing window → GA.
   * - ☐
     - Conduct public API audit:

       - Review all new/changed symbols since the last release for unintended
         breaking changes.
       - Verify deprecated symbols carry proper deprecation warnings and
         migration guidance (see :ref:`deprecation-timeline`).
       - Confirm new public API has complete docstrings and is included in
         Sphinx docs (run ``uv run docs/generate_api.py``).

       Run the ``release-audit`` Claude Code skill
       (``.claude/skills/release-audit``) in **pre-release mode** to automate
       this audit.
   * - ☐
     - Communicate the timeline to the community.


Code freeze and release branch creation
---------------------------------------

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - Create ``release-X.Y`` branch from ``main`` and push it.
   * - ☐
     - On **main**: bump the version in ``pyproject.toml`` to ``X.(Y+1).0.dev0`` and run
       ``uv run docs/generate_api.py``.
   * - ☐
     - On **release-X.Y**: bump the version in ``pyproject.toml`` to ``X.Y.ZrcN`` and
       run ``uv run docs/generate_api.py``.
   * - ☐
     - On **release-X.Y**: update dependencies in ``pyproject.toml`` from dev
       to RC versions where applicable and remove the NVIDIA package index
       (``[[tool.uv.index]]`` entry for ``nvidia`` **and** the
       ``warp-lang`` entry in ``[tool.uv.sources]`` that references it) so
       the release wheel installs purely from PyPI, then regenerate
       ``uv.lock`` (``uv lock``) and commit.
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

For each new RC (``rc2``, ``rc3``, …) bump the version in
``pyproject.toml`` and run ``uv run docs/generate_api.py``, then tag and push.
Resolve any cherry-pick conflicts or missing dependent cherry-picks that
cause CI failures before tagging.

.. _testing-criteria:

Testing criteria
^^^^^^^^^^^^^^^^

The release engineer and maintainers decide which issues must be fixed
before GA and which can ship as known issues documented in the release
notes.  Features explicitly marked **experimental** have a lower bar —
regressions in experimental APIs do not necessarily block a release.

As a guideline, an RC is typically ready for GA when:

- All examples run without crashes, excessive warnings, or visual
  artifacts (``uv run -m newton.examples <name>``).
- Testing covers **Windows and Linux**, **all supported Python versions**,
  and both **latest and minimum-spec CUDA drivers** (see
  :ref:`system requirements <system-requirements>` in the installation guide).
- PyPI installation of the RC works in a clean environment: ``pip install``
  succeeds, ``import newton`` works, and examples and tests can be run from
  the installed wheel (``pip install newton==X.Y.ZrcN``).
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
     - Finalize ``CHANGELOG.md``: rename ``[Unreleased]`` →
       ``[X.Y.Z] - YYYY-MM-DD``.  Review the entries for:

       - **Missing entries** — cross-check merged PRs since the last GA
         release (or patch) to catch changes that were not recorded in the
         changelog.
       - **Redundant entries** — consolidate or remove duplicates for changes
         within the same release period (e.g. a bug fix for a feature added
         in the same cycle should not appear as both an "Added" and a "Fixed"
         entry).

       The ``release-audit`` skill's CHANGELOG language review is a useful
       first pass before this manual sweep.
   * - ☐
     - Update ``README.md`` documentation links to point to versioned URLs
       (e.g. ``/X.Y.Z/guide.html`` instead of ``/latest/``).
   * - ☐
     - Verify all dependency pins in ``pyproject.toml`` use stable
       (non-pre-release) versions.
   * - ☐
     - Regenerate ``uv.lock`` (``uv lock``) and verify that no pre-release
       dependencies remain in the lock file.
   * - ☐
     - Bump the version in ``pyproject.toml`` to ``X.Y.Z`` (remove the RC suffix) and
       run ``uv run docs/generate_api.py``.
   * - ☐
     - Commit and push tag ``vX.Y.Z``.  Automated workflows trigger:

       - ``release.yml``: builds wheel, publishes to PyPI (requires manual
         approval), creates a draft GitHub Release.
       - ``docs-release.yml``: deploys docs to ``/X.Y.Z/`` and ``/stable/``
         on gh-pages, updates ``switcher.json``.
   * - ☐
     - PyPI publish approved and verified: ``pip install newton==X.Y.Z``.
   * - ☐
     - GitHub Release un-drafted and published.
   * - ☐
     - Docs live at ``/X.Y.Z/`` and ``/stable/``: verify links and version
       switcher.
   * - ☐
     - Release announcement posted.


Post-release
------------

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - Merge back the changelog from ``release-X.Y`` to ``main``: move
       entries included in the release from ``[Unreleased]`` to a new
       ``[X.Y.Z]`` section.  (``[Unreleased]`` is a permanent header in
       the changelog that always exists on ``main``.)
   * - ☐
     - Verify PyPI installation works in a clean environment.
   * - ☐
     - Verify published docs render correctly.


Patch releases
--------------

Patch releases continue cherry-picking fixes to the existing
``release-X.Y`` branch.  For example, ``1.0.1`` follows ``1.0.0``.
Follow the same :ref:`final-release` flow — bump version, update changelog,
tag, and push.  There is no need to create a new branch or bump ``main``.
