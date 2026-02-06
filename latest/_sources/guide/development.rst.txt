Development
===========

This document is a guide for developers who want to contribute to the project or understand its internal workings in more detail.

Please refer to `CONTRIBUTING.md <https://github.com/newton-physics/governance/blob/main/CONTRIBUTING.md>`_ for how to best contribute to Newton and relevant legal information (CLA).

Installation
------------

To install Newton, see the :doc:`installation` guide.

Python Dependency Management
----------------------------

uv lockfile management
^^^^^^^^^^^^^^^^^^^^^^

When using uv, the `lockfile <https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile>`__
(``uv.lock``) is used to resolve project dependencies into exact versions for reproducibility among different machines.

We maintain a lockfile in the root of the repository that pins exact versions of all dependencies and their transitive dependencies.

Sometimes, a dependency in the lockfile needs to be updated to a newer version.
This can be done by running ``uv lock -P <package-name>``:

.. code-block:: console

    uv lock -P warp-lang --prerelease allow

    uv lock -P mujoco-warp --prerelease allow

The ``--prerelease allow`` flag is used to allow updating to pre-release versions of dependencies.

uv also provides a command to update all dependencies in the lockfile:

.. code-block:: console

    uv lock -U

Remember to commit ``uv.lock`` after running a command that updates the lockfile.

Running the tests
-----------------

The Newton test suite supports both ``uv`` and standard ``venv`` workflows,
and by default runs in up to eight parallel processes. On some systems, the
tests must be run in a serial manner with ``--serial-fallback`` due to an
outstanding bug.

Pass ``--help`` to either run method below to see all available flags.

.. tab-set::
    :sync-group: env

    .. tab-item:: uv
        :sync: uv
        
        .. code-block:: console

            # install development extras and run tests
            uv run --extra dev -m newton.tests

    .. tab-item:: venv
        :sync: venv

        .. code-block:: console

            # install dev extras (including testing & coverage deps)
            python -m pip install -e .[dev]
            # run tests
            python -m newton.tests
            
Most tests run when the ``dev`` extras are installed. The tests that run examples that use PyTorch to inference an RL policy are skipped if the ``torch`` dependency is not installed. In order to run these tests, include the ``torch-cu12`` extras:

.. tab-set::
    :sync-group: env

    .. tab-item:: uv
        :sync: uv

        .. code-block:: console

            # install development extras and run tests
            uv run --extra dev --extra torch-cu12 -m newton.tests

    .. tab-item:: venv
        :sync: venv

        .. code-block:: console

            # install both dev and torch-cu12 extras (need to pull from PyTorch CUDA 12.8 wheel index)
            python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .[dev,torch-cu12]
            # run tests
            python -m newton.tests

Specific Newton examples can be tested in isolation via the ``-k`` argument:

.. tab-set::
    :sync-group: env

    .. tab-item:: uv
        :sync: uv
        
        .. code-block:: console

            # test the basic_shapes example
            uv run --extra dev -m newton.tests.test_examples -k test_basic.example_basic_shapes

    .. tab-item:: venv
        :sync: venv

        .. code-block:: console

            # test the basic_shapes example
            python -m newton.tests.test_examples -k test_basic.example_basic_shapes


To generate a coverage report:

.. tab-set::
    :sync-group: env

    .. tab-item:: uv
        :sync: uv

        .. code-block:: console
            
            # append the coverage flags:
            uv run --extra dev -m newton.tests --coverage --coverage-html htmlcov

    .. tab-item:: venv
        :sync: venv

        .. code-block:: console

            # append the coverage flags and make sure `coverage[toml]` is installed (it comes in `[dev]`)
            python -m newton.tests --coverage --coverage-html htmlcov

The file ``htmlcov/index.html`` can be opened with a web browser to view the coverage report.

Code formatting and linting
---------------------------

`Ruff <https://docs.astral.sh/ruff/>`_ is used for Python linting and code formatting.
`pre-commit <https://pre-commit.com/>`_ can be used to ensure that local code complies with Newton's checks.
From the top of the repository, run:

.. tab-set::
    :sync-group: env

    .. tab-item:: uv
        :sync: uv

        .. code-block:: console

            uvx pre-commit run -a

    .. tab-item:: venv
        :sync: venv

        .. code:: console

            python -m pip install pre-commit
            pre-commit run -a

To automatically run pre-commit hooks with ``git commit``:

.. tab-set::
    :sync-group: env

    .. tab-item:: uv
        :sync: uv

        .. code-block:: console

            uvx pre-commit install

    .. tab-item:: venv
        :sync: venv

        .. code:: console

            pre-commit install

The hooks can be uninstalled with ``pre-commit uninstall``.

Typos
-----

To proactively catch spelling mistakes, Newton uses the `typos <https://github.com/crate-ci/typos>`_ tool. Typos scans source files for common misspellings and is integrated into our pre-commit hooks, so spelling errors in both code and documentation are flagged when you run or install pre-commit (see above). You can also run ``typos`` manually if needed. Refer to the `typos documentation <https://github.com/crate-ci/typos>`_ for more details on usage and configuration options.

Dealing with false positives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typos may occasionally flag legitimate project-specific terminology, domain terms, or variable names as misspellings (false positives). To handle these, the Newton codebase configures typos in ``pyproject.toml`` at the repository root.

False positives are managed as follows:

- **File exclusions:** The ``[tool.typos]`` section includes ``files.extend-exclude`` to ignore matching files and directories, such as ``examples/assets`` and specific model or asset file types (e.g., ``*.urdf``, ``*.usd``).
- **Word allowlist:** Words or acronyms that would otherwise be flagged can be listed in ``[tool.typos.default.extend-words]`` (e.g., ``ba``, ``HAA``).
- **Identifier allowlist:** Specific identifiers, such as variable or constant names, can be declared in ``[tool.typos.default.extend-identifiers]`` (e.g., ``PNGs``).

When typos reports a word that is valid within the Newton codebase, you can add it to the appropriate section in ``pyproject.toml`` to suppress future warnings. After updating, re-run typos (or pre-commit) to confirm that the word is ignored. Use these options to keep the codebase clean while ensuring needed flexibility for accepted project-specific words and identifiers.





Using a local Warp installation with uv
---------------------------------------

Use the following steps to run Newton with a local build of Warp:

.. code-block:: console

    uv venv
    source .venv/bin/activate
    uv sync --extra dev
    uv pip install -e "warp-lang @ ../warp"

The Warp initialization message should then properly reflect the local Warp installation instead of the locked version,
e.g. when running ``python -m newton.examples basic_pendulum``.

Building the documentation
--------------------------

To build the documentation locally, ensure you have the documentation dependencies installed.

.. tab-set::
    :sync-group: env

    .. tab-item:: uv
        :sync: uv

        .. code-block:: console

            rm -rf docs/_build
            uv run --extra docs --extra sim sphinx-build -W -b html docs docs/_build/html

    .. tab-item:: venv
        :sync: venv

        .. code:: console

            python -m pip install -e .[docs]
            cd path/to/newton/docs && make html

The built documentation will be available in ``docs/_build/html``.

.. note::

    The documentation build requires `pandoc <https://pandoc.org/>`_ for converting Jupyter notebooks.
    While ``pypandoc_binary`` is included in the ``[docs]`` dependencies, some systems may require
    pandoc to be installed separately:

    - **Ubuntu/Debian:** ``sudo apt-get install pandoc``
    - **macOS:** ``brew install pandoc``
    - **Windows:** Download from https://pandoc.org/installing.html or ``choco install pandoc``

Serving the documentation locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After building the documentation, you can serve it locally using the ``docs/serve.py`` script.
This is particularly useful for testing interactive features like the Viser 3D visualizations
in the tutorial notebooks, which require proper MIME types for WebAssembly and JavaScript modules.

.. tab-set::
    :sync-group: env

    .. tab-item:: uv
        :sync: uv

        .. code-block:: console

            uv run docs/serve.py

    .. tab-item:: venv
        :sync: venv

        .. code:: console

            python docs/serve.py

Then open http://localhost:8000 in your browser. You can specify a custom port with ``--port``:

.. code-block:: console

    python docs/serve.py --port 8080

.. note::

    Using Python's built-in ``http.server`` or simply opening the HTML files directly
    will not work correctly for the interactive Viser visualizations, as they require
    specific CORS headers and MIME types that ``serve.py`` provides.

Documentation Versioning
------------------------

Newton's documentation is versioned and hosted on GitHub Pages. Multiple versions
are available simultaneously, with a version switcher dropdown in the navigation bar.

**Published documentation URLs:**

- **Latest stable release**: https://newton-physics.github.io/newton/stable/
- **Development (main branch)**: https://newton-physics.github.io/newton/latest/
- **Specific versions**: https://newton-physics.github.io/newton/1.0.0/ (etc.)

How It Works
^^^^^^^^^^^^

The ``gh-pages`` branch contains versioned documentation in subdirectories:

.. code-block:: text

    /
    ├── index.html      # Redirects to /stable/
    ├── switcher.json   # Version manifest for dropdown
    ├── stable/         # Copy of latest release
    ├── latest/         # Dev docs from main branch
    ├── 1.1.0/          # Release versions
    └── 1.0.0/

Two GitHub Actions workflows manage deployment:

- **docs-dev.yml**: Deploys to ``/latest/`` on every push to ``main``
- **docs-release.yml**: Deploys to ``/X.Y.Z/`` and updates ``/stable/`` on version tags

Deploying Documentation
^^^^^^^^^^^^^^^^^^^^^^^

**Dev docs** are deployed automatically when changes are pushed to ``main``.

**Release docs** are deployed when a version tag is pushed:

.. code-block:: bash

    git tag v1.0.0
    git push origin v1.0.0

Only strict semver tags (``vX.Y.Z``) trigger release deployments. Pre-release tags
like ``v1.0.0-rc.1`` are ignored.

Manual Operations
^^^^^^^^^^^^^^^^^

**Removing a version** (rare):

1. Check out the ``gh-pages`` branch
2. Delete the version directory (e.g., ``rm -rf 1.0.0``)
3. Edit ``switcher.json`` to remove the entry
4. Commit and push

**Rebuilding all docs** (disaster recovery): Check out each version tag, build its
docs with Sphinx, and deploy to the corresponding directory on ``gh-pages``. Update
``switcher.json`` after each version using ``scripts/ci/update_docs_switcher.py``.

Testing documentation code snippets
-----------------------------------

The ``doctest`` Sphinx builder is used to ensure that code snippets in the documentation remain up-to-date.

The doctests can be run with:

.. tab-set::
    :sync-group: env

    .. tab-item:: uv
        :sync: uv

        .. code-block:: console

            uv run --extra docs --extra sim sphinx-build -W -b doctest docs docs/_build/doctest

    .. tab-item:: venv
        :sync: venv

        .. code:: console

            python -m sphinx -W -b doctest docs docs/_build/doctest

For more information, see the `sphinx.ext.doctest <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`__
documentation.

Style Guide
-----------

- Follow PEP 8 for Python code.
- Use Google-style docstrings (compatible with Napoleon extension).
- Write clear, concise commit messages.
- Keep pull requests focused on a single feature or bug fix.
- Use kebab-case instead of snake_case for command line arguments, e.g. ``--use-cuda-graph`` instead of ``--use_cuda_graph``.

Roadmap and Future Work
-----------------------

(Placeholder for future roadmap and planned features)

- Advanced solver coupling
- More comprehensive sensor models
- Expanded robotics examples

See the `GitHub Discussions <https://github.com/newton-physics/newton/discussions>`__ for ongoing feature planning.

Benchmarking with airspeed velocity
-----------------------------------

The Newton repository contains a benchmarking suite implemented using the `airspeed velocity <https://asv.readthedocs.io/en/latest/>`__ framework.
The full set of benchmarks are intended to be run on a machine with a CUDA-capable GPU.

To get started, install airspeed velocity from PyPI:

.. code-block:: console

    python -m pip install asv

If airspeed velocity has not been previously run on the machine, it will need to be initialized with:

.. code-block:: console

    asv machine --yes

To run the benchmarks, run the following command from the root of the repository:

.. tab-set::
    :sync-group: shell

    .. tab-item:: Unix
        :sync: unix

        .. code-block:: console

            asv run --launch-method spawn main^!

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: console

            asv run --launch-method spawn main^^!

.. note::

    On Windows CMD, the ``^`` character is an escape character, so it must be doubled (``^^``) to be interpreted literally.

The benchmarks discovered by airspeed velocity are in the ``asv/benchmarks`` directory. This command runs the
benchmark code from the ``asv/benchmarks`` directory against the code state of the ``main`` branch. Note that
the benchmark definitions themselves are not checked out from different branches—only the code being
benchmarked is.

Tips for writing benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rather than running the entire benchmark suite, use the ``--bench BENCH, -b BENCH`` flag to filter the benchmarks
to just the ones under development:

.. tab-set::
    :sync-group: shell

    .. tab-item:: Unix
        :sync: unix

        .. code-block:: console

            asv run --launch-method spawn main^! --bench example_anymal.PretrainedSimulate

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: console

            asv run --launch-method spawn main^^! --bench example_anymal.PretrainedSimulate

The most time-consuming benchmarks are those that measure the time it takes to load and run one frame of the example
starting from an empty kernel cache.
These benchmarks have names ending with ``time_load``. It is sometimes convenient to exclude these benchmarks
from running by using the following command:

.. tab-set::
    :sync-group: shell

    .. tab-item:: Unix
        :sync: unix

        .. code-block:: console

            asv run --launch-method spawn main^! -b '^(?!.*time_load$).*'

    .. tab-item:: Windows
        :sync: windows

        .. code-block:: console

            asv run --launch-method spawn main^^! -b "^^(?!.*time_load$).*"

While airspeed velocity has built-in mechanisms to determine automatically how to collect measurements,
it is often useful to manually specify benchmark attributes like ``repeat`` and ``number`` to control the
number of times a benchmark is run and the number of times a benchmark is repeated.

.. code-block:: python

    class PretrainedSimulate:
        repeat = 3
        number = 1

As the airspeed documentation on `benchmark attributes <https://asv.readthedocs.io/en/stable/writing_benchmarks.html#benchmark-attributes>`__ notes,
the ``setup`` and ``teardown`` methods are not run between the ``number`` iterations that make up a sample.

These benchmark attributes should be tuned to ensure that the benchmark runs in a reasonable amount of time while
also ensuring that the benchmark is run a sufficient number of times to get a statistically meaningful result.

The ``--durations all`` flag can be passed to the ``asv run`` command to show the durations of all benchmarks,
which is helpful for ensuring that a single benchmark is not requiring an abnormally long amount of time compared
to the other benchmarks.
