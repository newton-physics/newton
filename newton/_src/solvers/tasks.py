# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tasks for solvers (optional, through ``cuda.stf``'s Warp adapter).

Under a ``cuda.stf`` recording every Warp launch belongs to a task, and the
order between tasks is given by tokens, as in ``cuda.stf``. A solver chooses:

* It does nothing. Its caller wraps the whole step in one task.
* It sets ``records_tasks = True`` and puts all of its launches in tasks. Its
  caller then calls it from a scope, not from inside a task::

      seq = scope_token()  # the sequence of the current scope
      with warp_task(seq.rw()):  # ordered against every task that takes seq
          ...
      with warp_task(seq.read(), symbol="mass"):  # these two may overlap
          ...
      with warp_task(seq.read(), symbol="rows"):
          ...
      with warp_task(seq.rw()):  # waits for both
          ...

Nothing is recorded unless the application opened a ``cuda.stf`` scope; every
construct is then a no-op and the step is the sequential step. The same holds
inside a task: a solver with ``records_tasks`` that is called from one runs as
part of it. Tasks that read the same token must not depend on each other;
nothing checks it.
"""

try:
    from cuda.stf._experimental.interop.warp import scope, scope_token, token, warp_task
except ImportError:  # cuda-stf not installed (or an older adapter): the constructs do nothing
    import contextlib

    class _NullToken:
        def rw(self):
            return None

        def read(self):
            return None

        def write(self):
            return None

    def token(*args, **kwargs):
        return _NullToken()

    def scope_token():
        return _NullToken()

    def scope(*args, **kwargs):
        return contextlib.nullcontext()

    def warp_task(*deps, **kwargs):
        return contextlib.nullcontext()


def solver_step(solver, *deps, symbol=None):
    """The construct a caller wraps one solver step in: a scope for a solver that
    records its own tasks (``records_tasks``), one task otherwise. ``deps`` are the
    dependencies of the step as a whole, in the caller's scope."""
    if getattr(solver, "records_tasks", False):
        return scope(
            reads=[d.ld for d in deps if d is not None and not _writes(d)],
            writes=[d.ld for d in deps if d is not None and _writes(d)],
            symbol=symbol,
        )
    return warp_task(*deps, symbol=symbol)


def _writes(dep):
    return int(dep.mode) & 2 != 0


__all__ = ["scope", "scope_token", "solver_step", "token", "warp_task"]
