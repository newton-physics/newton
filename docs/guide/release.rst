.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Release Process
===============

This document describes how to prepare, publish, and follow up on a Newton
release.  It is intended for release engineers and maintainers.

Overview
--------

Newton uses `PEP 440 <https://peps.python.org/pep-0440/>`__ versioning
(``Major.Minor.Micro``), consistent with warp-lang:

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Kind
     - Example
     - When
   * - Stable
     - ``1.0.0``
     - Tagged GA releases published to PyPI
   * - Release candidate
     - ``1.0.0rc1``
     - Pre-release builds for QA validation
   * - Development
     - ``1.1.0.dev0``
     - ``main`` between releases

Releases are published to `PyPI <https://pypi.org/p/newton>`__ and
documentation is deployed to
`GitHub Pages <https://newton-physics.github.io/newton/>`__.

Version source of truth
^^^^^^^^^^^^^^^^^^^^^^^

The version string lives in ``newton/_version.py``.  All other version
references (PyPI metadata, documentation) are derived from this file.

Dependency versioning strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pyproject.toml`` specifies **minimum** compatible versions
(e.g. ``warp-lang>=1.12.0``).  ``uv.lock`` pins the **latest known-good**
versions for reproducible installs.

Exception: on the **release branch**, ``mujoco`` and ``mujoco-warp`` are
pinned to **exact** versions (e.g. ``mujoco==3.5.0``) because Newton is
tightly coupled to the MuJoCo API surface.  ``main`` uses a version floor
like other dependencies.


Pre-release planning
--------------------

.. rubric:: Checklist

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - Determine target version (``X.Y.Z``).
   * - ☐
     - Confirm dependency versions and availability: warp-lang, mujoco,
       mujoco-warp, newton-usd-schemas, newton-actuators.
   * - ☐
     - Set timeline: code freeze → RC1 → testing window → GA.
   * - ☐
     - Conduct public API audit:

       - Review all new/changed symbols since the last release for unintended
         breaking changes.
       - Verify deprecated symbols carry proper deprecation warnings and
         migration guidance.
       - Confirm new public API has complete docstrings and is included in
         Sphinx docs (run ``docs/generate_api.py``).
   * - ☐
     - Communicate the timeline to contributors.


Code freeze and release branch creation
---------------------------------------

1. Create the ``release-X.Y`` branch from ``main``.
2. On **main**: bump ``newton/_version.py`` to ``X.(Y+1).0.dev0`` and run
   ``docs/generate_api.py``.
3. On **release-X.Y**: bump ``newton/_version.py`` to ``X.Y.ZrcN`` and run
   ``docs/generate_api.py``.
4. On **release-X.Y**: update dependencies in ``pyproject.toml`` from dev
   to RC versions where applicable, then regenerate ``uv.lock``
   (``uv lock``) and commit it.
5. Push tag ``vX.Y.ZrcN``.  This triggers the ``release.yml`` workflow
   (build wheel → PyPI publish with manual approval).

.. rubric:: Checklist

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - ``release-X.Y`` branch created and pushed.
   * - ☐
     - ``main`` bumped to next dev version; ``generate_api.py`` run.
   * - ☐
     - RC1 version set on release branch; ``generate_api.py`` run.
   * - ☐
     - Dependencies updated from dev to RC versions; ``uv.lock`` regenerated.
   * - ☐
     - Tag ``vX.Y.Zrc1`` pushed.
   * - ☐
     - RC1 published to PyPI (approve in GitHub environment).


Release candidate stabilization
-------------------------------

Bug fixes merge to ``main`` first, then are cherry-picked to
``release-X.Y``.  Cherry-pick relevant commits from ``main`` onto a feature
branch and open a pull request targeting ``release-X.Y`` — never push
directly to the release branch.

For each new RC (``rc2``, ``rc3``, …) bump the version in
``newton/_version.py`` and run ``docs/generate_api.py``, then tag and push.
Iterate until CI is green and QA signs off.

Testing criteria
^^^^^^^^^^^^^^^^

An RC is considered ready for GA when all of the following are met:

- The full test suite passes on CI (``uv run --extra dev -m newton.tests``).
- All examples run successfully with the viewer
  (``uv run -m newton.examples <name>``).
- Testing covers **Windows and Linux**, **all supported Python versions**,
  and both **latest CUDA drivers** and **minimum-spec drivers**.
- PyPI installation of the RC works in a clean environment
  (``pip install newton==X.Y.ZrcN``).
- No regressions compared to the previous release have been identified.

.. rubric:: Checklist

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - All release-targeted fixes cherry-picked from ``main``.
   * - ☐
     - CI passing on release branch.
   * - ☐
     - Testing criteria above satisfied.
   * - ☐
     - No outstanding release-blocking issues.


.. _final-release:

Final release
-------------

Before proceeding, obtain explicit go/no-go approval from testing and
stakeholders.  Do not start the final release steps until sign-off is
confirmed.

1. Finalize ``CHANGELOG.md``: rename ``[Unreleased]`` →
   ``[X.Y.Z] - YYYY-MM-DD``.
2. Update documentation links to point to versioned URLs where appropriate.
3. Verify all dependency pins in ``pyproject.toml`` use stable
   (non-pre-release) versions.
4. Regenerate ``uv.lock`` (``uv lock``) and verify that no pre-release
   dependencies remain in the lock file.
5. Bump ``newton/_version.py`` to ``X.Y.Z`` (remove the RC suffix) and run
   ``docs/generate_api.py``.
6. Commit and push tag ``vX.Y.Z``.
7. Automated workflows trigger:

   - ``release.yml``: builds wheel, publishes to PyPI (requires manual
     approval), creates a draft GitHub Release.
   - ``docs-release.yml``: deploys docs to ``/X.Y.Z/`` and ``/stable/``
     on gh-pages, updates ``switcher.json``.

8. Review and publish (un-draft) the GitHub Release.

.. rubric:: Checklist

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - Go/no-go approval obtained from testing and stakeholders.
   * - ☐
     - Changelog finalized with release date.
   * - ☐
     - All dependency pins in ``pyproject.toml`` use stable (non-pre-release)
       versions.
   * - ☐
     - ``uv.lock`` regenerated and verified free of pre-release dependencies.
   * - ☐
     - Version bumped to ``X.Y.Z``; ``generate_api.py`` run.
   * - ☐
     - Tag ``vX.Y.Z`` pushed.
   * - ☐
     - PyPI publish approved and verified: ``pip install newton==X.Y.Z``.
   * - ☐
     - GitHub Release un-drafted and published.
   * - ☐
     - Docs live at ``/X.Y.Z/`` and ``/stable/``: verify links and version
       switcher.


Post-release
------------

1. Compare ``CHANGELOG.md`` between ``release-X.Y`` and ``main``.  The
   release branch has the finalized ``[X.Y.Z]`` section, while ``main``
   still has those entries under ``[Unreleased]``.
2. On **main**: merge back the changelog from the release branch so that
   all entries included in the release are moved from ``[Unreleased]`` to
   the ``[X.Y.Z]`` section.  (The ``[Unreleased]`` header already exists
   on ``main`` from the post-branch-creation bump.)
3. Verify PyPI installation works in a clean environment.
4. Verify published docs render correctly.

.. rubric:: Checklist

.. list-table::
   :widths: 5 95
   :header-rows: 0

   * - ☐
     - ``CHANGELOG.md`` on ``main`` updated: released entries moved from
       ``[Unreleased]`` to ``[X.Y.Z]`` section.
   * - ☐
     - PyPI install verified.
   * - ☐
     - Published docs verified.


Patch releases
--------------

Patch releases (``X.Y.Z+1``) continue cherry-picking fixes to the existing
``release-X.Y`` branch.  Follow the same :ref:`final-release` flow — bump
version, update changelog, tag, and push.  There is no need to create a new
branch or bump ``main``.
