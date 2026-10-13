# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share the anchored rod fixture between model and selection tests."""

import newton


def _add_test_anchored_cable(builder, label="anchored_curve"):
    """Add a two-segment cable fixed to the world for collapse tests."""
    bodies, joints = builder.add_rod(
        rod=newton.Rod([(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0)], radius=0.02),
        label=label,
        wrap_in_articulation=False,
        body_frame_origin="com",
    )
    anchor = builder.add_joint_fixed(-1, bodies[0], label="anchor")
    builder.add_articulation([*joints, anchor])
