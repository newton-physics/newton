# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share helpers for USD importer tests without adding discovered cases."""

import functools
import warnings

_INVALID_ARTICULATION_DESC = "Warning: Invalid ArticulationDesc descriptor"


def _expect_jointless_articulation_warning(test):
    """Require the benign jointless-articulation warning on OpenUSD < 26.0.

    ``UsdPhysics``'s native physics parser in OpenUSD < 26.0 (e.g. the
    ``usd-exchange`` build resolved on ``aarch64``) reports an articulation root
    that has no joints as an invalid ``ArticulationDesc``, which
    :func:`~newton.utils.parse_usd` surfaces as a ``UserWarning``; usd-core
    >= 26.0 treats it as valid. The fixtures wrapped here intentionally import
    single-body (jointless) articulations -- a shape Newton parses identically
    either way. On the USD versions that emit it, assert exactly that warning
    while leaving every other warning subject to the ambient policy, so an
    unexpected ``newton.*`` warning here still fails under ``--strict-warnings``.
    """

    @functools.wraps(test)
    def wrapper(self, *args, **kwargs):
        from pxr import Usd

        if Usd.GetVersion() >= (0, 26, 0):
            return test(self, *args, **kwargs)
        with warnings.catch_warnings(record=True) as caught:
            # Record (do not escalate) only the expected warning; the inherited
            # "error" filter still applies to everything else under strict mode.
            warnings.filterwarnings("always", message=_INVALID_ARTICULATION_DESC, category=UserWarning)
            result = test(self, *args, **kwargs)
        self.assertTrue(
            any(
                issubclass(w.category, UserWarning) and str(w.message).startswith(_INVALID_ARTICULATION_DESC)
                for w in caught
            ),
            f"expected a {_INVALID_ARTICULATION_DESC!r} warning on OpenUSD < 26.0",
        )
        return result

    return wrapper
