.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _diagnostics:

Logging and Warnings
====================

Newton reports diagnostics through two standard Python mechanisms:

- **Log records** from the :mod:`logging` module, under the ``newton``
  logger. These carry ``verbose=True`` output, status messages, and problems
  Newton worked around, such as asset content an importer skipped.
- **Warnings** from the :mod:`warnings` module, using the categories in
  :mod:`newton.exceptions`. These ask you to change how you call Newton, for
  example to stop using a deprecated feature.

Each mechanism has its own configuration: logger levels do not affect
warnings, and warning filters do not affect log records.

Default output
--------------

Until your application configures logging, Newton prints its log records
itself:

- ``INFO`` records go to stdout, unchanged.
- ``WARNING`` and higher go to stderr, prefixed with ``Warning:`` or
  ``Error:``.
- ``DEBUG`` records are hidden.

As soon as a handler is attached to the ``newton`` logger or to any of its
ancestors, including the root logger, Newton stops printing and its records
go only to the configured handlers. This happens when your application calls
:func:`logging.basicConfig`, under pytest's log capture, and in host
applications and frameworks that configure logging themselves.

Rich console output
^^^^^^^^^^^^^^^^^^^

:func:`newton.utils.enable_rich_console` switches this default output to a
format that prefixes each record with its time, source file and line, and
level name, colored by level when stdout is a terminal:

.. code-block:: text

    [2026-11-02 10:15:42,108][import_mjcf.py:412][INFO]: Parsing MJCF file

Scripts log through the ``__main__`` logger, which is outside the ``newton``
logger tree, so their ``INFO`` records are dropped unless logging is
configured. Call :func:`newton.utils.enable_script_logging` from the script's
``if __name__ == "__main__":`` block to print them the same way as ``newton``
records. :func:`newton.utils.setup_console_logging` calls both helpers. Newton
examples run through ``newton.examples.init()`` and the Newton test runner
already call it.

Controlling log output
----------------------

Configure the ``newton`` logger with the standard :mod:`logging` API:

.. code-block:: python

    import logging

    # Hide INFO output, such as verbose=True diagnostics.
    logging.getLogger("newton").setLevel(logging.WARNING)

    # Or show DEBUG details as well.
    logging.getLogger("newton").setLevel(logging.DEBUG)

To change the format or destination, configure handlers as usual. Once you do,
Newton no longer prints on its own, so add a console handler if you still want
console output:

.. code-block:: python

    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("newton").addHandler(logging.FileHandler("newton.log"))

Configure the ``newton`` logger itself rather than its children. Child logger
names follow Newton's internal module layout and may change between releases.

``verbose`` arguments
^^^^^^^^^^^^^^^^^^^^^

Functions that accept ``verbose=True``, such as
:meth:`~newton.ModelBuilder.add_mjcf` and :meth:`~newton.ModelBuilder.add_usd`,
log their output at ``INFO``; some also report skipped asset content at
``WARNING`` only when ``verbose`` is set. The ``newton`` logger defaults to
``INFO``, so this output is visible unless you raise the logger's level.

Controlling warnings
--------------------

Newton emits deprecations as
:class:`~newton.exceptions.NewtonDeprecationWarning`, a subclass of
:class:`DeprecationWarning`, and all other warnings as
:class:`~newton.exceptions.NewtonWarning`, a subclass of :class:`UserWarning`.
Existing filters for the builtin categories therefore keep matching the same
warnings as before. Filter them by category instead of by message:

.. code-block:: python

    import warnings

    import newton

    # Silence Newton warnings other than deprecations.
    warnings.filterwarnings("ignore", category=newton.exceptions.NewtonWarning)

    # Fail on Newton deprecations, for example in your test suite.
    warnings.filterwarnings("error", category=newton.exceptions.NewtonDeprecationWarning)

Warnings are printed to stderr by Python, not through the ``newton`` logger.
To send them to your logging handlers as well, call
:func:`logging.captureWarnings`; Python then logs them under the
``py.warnings`` logger.

Testing
-------

Assert Newton log output with :meth:`~unittest.TestCase.assertLogs` on the
``"newton"`` logger or with pytest's ``caplog`` fixture, rather than by
capturing stdout. Assert warnings with :meth:`~unittest.TestCase.assertWarns`
or ``pytest.warns``.

Limitations
-----------

Some diagnostics are printed from inside Warp kernels while they run on the
device. This output bypasses both logging and warnings, so it cannot be
filtered, redirected, or silenced through either mechanism.
