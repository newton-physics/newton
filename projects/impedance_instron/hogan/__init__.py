# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hogan-style reference-tracking impedance for a pelvis and one stance leg.

The controller is ``tau = tau_ff(phi) + K(phi) (q_ref(phi) - q) + D(phi) (v_ref(phi) - v)``.
``q_ref`` is the measured motion, ``tau_ff`` comes from inverse dynamics, and
only the impedance schedule ``K(phi)``, ``D(phi)`` is meant to be learned.
"""
