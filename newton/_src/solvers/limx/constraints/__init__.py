# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static particle constraint batches for the LIMX solver."""

from .affine_static_plane_contact import ConstraintAffineStaticPlaneContact
from .anchor import ConstraintAnchor
from .dihedral_bending import ConstraintDihedralBending
from .distance import ConstraintDistance
from .dynamic_group import ConstraintGroupDynamic
from .self_collision import ConstraintSelfCollision
from .static_plane_contact import ConstraintStaticPlaneContact
from .tetrahedron_arap import ConstraintTetrahedronARAP
from .triangle_elastic import ConstraintTriangleElastic

__all__ = [
    "ConstraintAffineStaticPlaneContact",
    "ConstraintAnchor",
    "ConstraintDihedralBending",
    "ConstraintDistance",
    "ConstraintGroupDynamic",
    "ConstraintSelfCollision",
    "ConstraintStaticPlaneContact",
    "ConstraintTetrahedronARAP",
    "ConstraintTriangleElastic",
]
