.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Compatibility and Support
=========================

This page describes which platforms and configurations Newton supports,
Newton's versioning scheme, and the policy that governs deprecations
and removals of public API.

Tested Configurations
---------------------

Newton releases are tested on the following configurations:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Component
     - Configuration
   * - OS
     - Ubuntu 22.04/24.04 (x86-64 + ARM64), Windows, macOS (CPU only)
   * - GPU
     - NVIDIA Ada Lovelace, Blackwell
   * - Python
     - 3.10+; full tests use ``.python-version``; CI imports 3.10-3.14 and resolves deps on 3.10-3.13
   * - CUDA
     - 12, 13

For the minimum requirements to install Newton, see
:ref:`system-requirements` in the installation guide.

.. _cpu-limitations:

CPU-Only Limitations
--------------------

Newton can run on CPU (including macOS), but the following features require an
NVIDIA GPU and are unavailable in CPU-only mode:

- **Tiled camera sensor** — GPU-accelerated raytraced rendering.
- **Implicit MPM solver** — designed for GPU execution with CUDA graph support.
- **Tile-based VBD solve** — uses GPU tile API; gracefully disabled on CPU.

Inherited Platform Support
--------------------------

Newton's baseline operating system, CUDA toolkit, NVIDIA driver, and
GPU architecture compatibility is inherited from `NVIDIA Warp
<https://nvidia.github.io/warp/stable/user_guide/compatibility.html>`__.
Warp's compatibility page is the source of truth for:

* Supported operating systems and their runtime requirements (e.g.,
  GLIBC versions on Linux).
* Supported CUDA toolkit versions and the corresponding NVIDIA driver
  requirements.
* Minimum GPU compute capability and forward-compatibility via PTX.

Newton may apply additional constraints on top of Warp's baseline; see
:ref:`cuda-compatibility` below.  For the install-relevant minimums,
see :ref:`system-requirements` in the installation guide.

.. _cuda-compatibility:

CUDA Compatibility
------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - CUDA Version
     - Notes
   * - 12.3+
     - Required for reliable CUDA graph capture
   * - 12.4+
     - Recommended for best performance
   * - 13
     - Supported

.. _versioning:

Versioning
----------

Newton currently uses the following versioning scheme. This may evolve
depending on the needs of the project and its users.

Newton uses a **major.minor.micro** versioning scheme, similar to
`Python itself <https://devguide.python.org/developer-workflow/development-cycle/#devcycle>`__:

* New **major** versions are reserved for major reworks of Newton causing
  disruptive incompatibility (or reaching the 1.0 milestone).
* New **minor** versions are feature releases with a new set of features.
  May contain deprecations, breaking changes, and removals.
* New **micro** versions are bug-fix releases. In principle, there are no
  new features. The first release of a new minor version always includes
  the micro version (e.g., ``1.1.0``), though informal references may
  shorten it (e.g., "Newton 1.1").

Prerelease Versions
^^^^^^^^^^^^^^^^^^^

In addition to stable releases, Newton uses the following prerelease
version formats:

* **Development builds** (``major.minor.micro.dev0``): The version string
  used in the source code on the main branch between stable releases
  (e.g., ``1.1.0.dev0``).
* **Release candidates** (``major.minor.microrcN``): Pre-release versions
  for QA testing before a stable release, starting with ``rc1`` and
  incrementing (e.g., ``1.1.0rc1``). Usually not published to PyPI.

Prerelease versions should be considered unstable and are not subject
to the same compatibility guarantees as stable releases.

Component States
----------------

Components of Newton — public API symbols, supported Python versions,
supported GPU architectures, and so on — exist in one of the following
states:

* **Experimental**: A feature still under active development.  May
  change without notice, including in micro releases.  Experimental
  features are available for early adopters who can tolerate breakage.
* **Stable**: The default state for most features.  Changes follow the
  :ref:`deprecation-policy`.
* **Deprecated**: A feature scheduled for removal in a future release.
  Remains fully functional during the deprecation window.
* **Removed**: A feature no longer in the library.  Attempting to use
  it raises an error.

.. _deprecation-policy:

Deprecation Policy
------------------

A deprecated feature is maintained for **two full minor release cycles**
after deprecation (e.g. deprecated in 1.2.0 → removed in 1.4.0).
Deprecations and removals only happen in minor releases, never in micro
releases.

Example timeline
^^^^^^^^^^^^^^^^

Assuming a feature is deprecated in release ``1.2.0``:

* ``1.2.0``: feature is deprecated.  It still works; using it emits a
  ``DeprecationWarning`` and the deprecation is noted in
  ``CHANGELOG.md``.
* ``1.2.x`` (micro releases): deprecated feature remains fully
  functional.
* ``1.3.0``: deprecated feature remains fully functional.  The release
  notes flag the upcoming removal.
* ``1.4.0``: feature is removed.  Using it raises an error.

How deprecations are communicated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **CHANGELOG.md**: every deprecation and removal is recorded in the
  ``Deprecated`` and ``Removed`` sections of ``CHANGELOG.md``, with
  migration guidance.
* **Runtime warnings**: using a deprecated feature emits a
  ``DeprecationWarning`` pointing at a replacement when one exists.
* **API documentation**: deprecated symbols are marked with
  ``.. deprecated:: X.Y`` directives indicating the version in which
  deprecation was introduced.

What to do when you see a DeprecationWarning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Read the warning message for the suggested replacement.
#. Check the ``Deprecated`` section of ``CHANGELOG.md`` to confirm the
   release in which the deprecation was first announced.
#. Migrate to the replacement.  The deprecated feature remains
   functional for two minor releases (see the example above).
#. If migration is blocked by a gap in the replacement, open a
   `GitHub issue <https://github.com/newton-physics/newton/issues>`__.

Your installed Newton version is available via ``newton.__version__``.

Release Support Policy
----------------------

Only the most recent minor release line is actively maintained:

* **Active support**: Only the latest minor release line (e.g.,
  ``1.4.x``) is eligible to receive micro releases.
* **No backporting**: Fixes are not backported to earlier minor release
  lines by default.
* **Upgrade path**: Users who encounter bugs or need fixes should
  upgrade to the latest minor release.

Public vs. Private API
----------------------

Newton's **public API** consists of symbols accessible from public
modules (``newton``, ``newton.geometry``, ``newton.solvers``,
``newton.utils``, and so on) without underscore prefixes.  The public
API is covered by the :ref:`deprecation-policy`.

Symbols under ``newton._src.*`` and any symbol with an underscore
prefix are **private**.  Private APIs may change or be removed in any
release without going through the deprecation policy.  Examples and
user code must not import from ``newton._src``.

Python Version Support
----------------------

Newton supports Python versions that are in "bugfix" or "security"
status according to the `Python release cycle
<https://devguide.python.org/versions/>`__.  Support for newly released
Python versions is added in the next Newton minor release after the
Python version reaches stable status.

When a Python version reaches end-of-life, support is dropped following
the :ref:`deprecation-policy`.
