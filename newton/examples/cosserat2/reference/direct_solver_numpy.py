# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy-based Direct Position Based Solver for Stiff Rods.

Reference implementation of "Direct Position-Based Solver for Stiff Rods"
(Deul et al., 2017)
https://animation.rwth-aachen.de/publication/0557/

This implementation follows the pbd_rods C++ reference code closely,
providing a non-GPU, sequential solver for validation and testing.

Key differences from iterative PBD:
- Formulates the entire system as a single linear system
- Uses LDLT factorization for direct solving
- Better handling of stiff constraints
- Single pass per solve (vs. multiple Gauss-Seidel iterations)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from newton.examples.cosserat2.reference.cosserat_rod_numpy import CosseratRodNumpy
from newton.examples.cosserat2.reference.quaternion_ops import (
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
)

if TYPE_CHECKING:
    pass

# Constants
EPS = 1.0e-6


@dataclass
class RodSegment:
    """Segment (rigid body) in the stiff rod model.

    Represents one segment of the rod with its position, rotation, and mass properties.

    Attributes:
        index: Index in the segments array.
        position: Center of mass position [3].
        rotation: Quaternion rotation [x, y, z, w].
        velocity: Linear velocity [3].
        angular_velocity: Angular velocity [3].
        mass: Mass of the segment (0 for static).
        inertia_tensor: Diagonal inertia tensor in local space [3].
    """

    index: int
    position: NDArray
    rotation: NDArray
    velocity: NDArray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: NDArray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    inertia_tensor: NDArray = field(default_factory=lambda: np.ones(3))

    def is_dynamic(self) -> bool:
        """Check if segment is dynamic (not static)."""
        return self.mass > 0.0

    @property
    def inv_mass(self) -> float:
        """Inverse mass (0 for static segments)."""
        return 1.0 / self.mass if self.mass > 0 else 0.0


@dataclass
class RodConstraint:
    """Combined stretch-bending-torsion constraint between two segments.

    Attributes:
        index: Index in the constraints array.
        segment_indices: Indices of the two connected segments [2].
        constraint_info: 3x4 matrix containing:
            col 0: connector in segment 0 (local)
            col 1: connector in segment 1 (local)
            col 2: connector in segment 0 (global)
            col 3: connector in segment 1 (global)
        stiffness_coefficient_k: Diagonal stiffness matrix [bend_d1, torsion, bend_d2].
        rest_darboux_vector: Rest Darboux vector [3].
        average_segment_length: Average length of connected segments.
        stretch_compliance: Compliance for stretch [3].
        bending_torsion_compliance: Compliance for bending/torsion [3].
    """

    index: int
    segment_indices: NDArray  # [2]
    constraint_info: NDArray = field(default_factory=lambda: np.zeros((3, 4)))
    stiffness_coefficient_k: NDArray = field(default_factory=lambda: np.ones(3))
    rest_darboux_vector: NDArray = field(default_factory=lambda: np.zeros(3))
    average_segment_length: float = 1.0
    stretch_compliance: NDArray = field(default_factory=lambda: np.zeros(3))
    bending_torsion_compliance: NDArray = field(default_factory=lambda: np.zeros(3))

    def segment_index(self, i: int) -> int:
        """Get segment index (0 or 1)."""
        return int(self.segment_indices[i])


@dataclass
class Node:
    """Node in the simulated tree structure for direct solving.

    The tree alternates between segment nodes and constraint nodes.

    Attributes:
        is_constraint: True if this node represents a constraint.
        index: Index in the segment or constraint array.
        D: 6x6 diagonal block matrix.
        Dinv: Inverse of D.
        J: 6x6 Jacobian matrix.
        parent: Parent node (None for root).
        children: List of child nodes.
        soln: Solution vector [6].
    """

    is_constraint: bool
    index: int
    D: NDArray = field(default_factory=lambda: np.zeros((6, 6)))
    Dinv: NDArray = field(default_factory=lambda: np.zeros((6, 6)))
    J: NDArray = field(default_factory=lambda: np.zeros((6, 6)))
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    soln: NDArray = field(default_factory=lambda: np.zeros(6))

    # Reference to actual segment or constraint object
    segment: RodSegment | None = None
    constraint: RodConstraint | None = None


class DirectPositionBasedSolverForStiffRods:
    """Direct solver for stiff rod constraints.

    Implements the direct position-based solver from the 2017 paper by Deul et al.
    This solver formulates the entire constraint system as a single linear system
    and solves it using LDLT factorization.
    """

    def __init__(self):
        """Initialize the direct solver."""
        self.segments: list[RodSegment] = []
        self.constraints: list[RodConstraint] = []
        self.root: Node | None = None
        self.forward_list: list[Node] = []
        self.backward_list: list[Node] = []
        self.rhs: list[NDArray] = []
        self.lambda_sums: list[NDArray] = []
        self.bending_torsion_jacobians: list[list[NDArray]] = []
        self.corr_x: list[NDArray] = []
        self.corr_q: list[NDArray] = []

    @staticmethod
    def compute_darboux_vector(q0: NDArray, q1: NDArray, average_segment_length: float) -> NDArray:
        """Compute the discrete Darboux vector (Equation 7).

        Args:
            q0: First quaternion [x, y, z, w].
            q1: Second quaternion [x, y, z, w].
            average_segment_length: Average length of connected segments.

        Returns:
            Darboux vector [3].
        """
        # omega = 2/L * Im(conjugate(q0) * q1)
        q0_conj = quat_conjugate(q0)
        product = quat_multiply(q0_conj, q1)
        return (2.0 / average_segment_length) * product[:3]

    @staticmethod
    def compute_bending_torsion_jacobians(
        q0: NDArray, q1: NDArray, average_segment_length: float
    ) -> tuple[NDArray, NDArray]:
        """Compute bending and torsion Jacobians (Equations 10 and 11).

        Args:
            q0: First quaternion [x, y, z, w].
            q1: Second quaternion [x, y, z, w].
            average_segment_length: Average length of connected segments.

        Returns:
            Tuple of (jOmega0, jOmega1), each 3x4 matrices.
        """
        x0, y0, z0, w0 = q0
        x1, y1, z1, w1 = q1
        scale = 2.0 / average_segment_length

        # jOmega0 (derivative of omega w.r.t. q0)
        jOmega0 = (
            np.array(
                [
                    [-w1, -z1, y1, x1],
                    [z1, -w1, -x1, y1],
                    [-y1, x1, -w1, z1],
                ]
            )
            * scale
        )

        # jOmega1 (derivative of omega w.r.t. q1)
        jOmega1 = (
            np.array(
                [
                    [w0, z0, -y0, -x0],
                    [-z0, w0, x0, -y0],
                    [y0, -x0, w0, -z0],
                ]
            )
            * scale
        )

        return jOmega0, jOmega1

    @staticmethod
    def compute_matrix_G(q: NDArray) -> NDArray:
        """Compute the G matrix for quaternion-angular velocity relationship (Equation 27).

        G maps angular velocity to quaternion derivative: q_dot = G * omega

        Args:
            q: Quaternion [x, y, z, w].

        Returns:
            4x3 G matrix.
        """
        x, y, z, w = q
        return 0.5 * np.array(
            [
                [w, z, -y],
                [-z, w, x],
                [y, -x, w],
                [-x, -y, -z],
            ]
        )

    @staticmethod
    def cross_product_matrix(v: NDArray) -> NDArray:
        """Compute the skew-symmetric cross product matrix.

        Args:
            v: Vector [3].

        Returns:
            3x3 skew-symmetric matrix.
        """
        return np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ]
        )

    @staticmethod
    def get_mass_matrix(segment: RodSegment) -> NDArray:
        """Get the 6x6 mass matrix for a segment.

        Args:
            segment: Rod segment.

        Returns:
            6x6 mass matrix (upper 3x3 for translation, lower 3x3 for rotation).
        """
        if not segment.is_dynamic():
            return np.eye(6)

        M = np.zeros((6, 6))

        # Upper 3x3: mass * identity
        M[:3, :3] = segment.mass * np.eye(3)

        # Lower 3x3: world-space inertia tensor
        R = quat_to_rotation_matrix(segment.rotation)
        inertia_local = np.diag(segment.inertia_tensor)
        inertia_world = R @ inertia_local @ R.T
        M[3:, 3:] = inertia_world

        return M

    def init_constraint(
        self,
        segment0: RodSegment,
        segment1: RodSegment,
        constraint_position: NDArray,
        average_radius: float,
        average_segment_length: float,
        youngs_modulus: float,
        torsion_modulus: float,
    ) -> RodConstraint:
        """Initialize a stretch-bending-twisting constraint.

        Args:
            segment0: First segment.
            segment1: Second segment.
            constraint_position: Position of constraint in world space.
            average_radius: Average radius at constraint position.
            average_segment_length: Average length of connected segments.
            youngs_modulus: Young's modulus (Pa).
            torsion_modulus: Torsion/shear modulus (Pa).

        Returns:
            Initialized RodConstraint.
        """
        constraint = RodConstraint(
            index=len(self.constraints),
            segment_indices=np.array([segment0.index, segment1.index]),
            average_segment_length=average_segment_length,
        )

        # Transform constraint position to local coordinates
        R0 = quat_to_rotation_matrix(segment0.rotation)
        R1 = quat_to_rotation_matrix(segment1.rotation)

        constraint.constraint_info[:, 0] = R0.T @ (constraint_position - segment0.position)
        constraint.constraint_info[:, 1] = R1.T @ (constraint_position - segment1.position)
        constraint.constraint_info[:, 2] = constraint_position.copy()
        constraint.constraint_info[:, 3] = constraint_position.copy()

        # Compute stiffness coefficients (Equation 5)
        # Second moment of area for circular cross-section
        I = (math.pi / 4.0) * (average_radius**4)
        bending_stiffness = youngs_modulus * I
        torsion_stiffness = 2.0 * torsion_modulus * I

        # K = diag(bend, torsion, bend) following Blender's y-axis convention
        constraint.stiffness_coefficient_k = np.array([bending_stiffness, torsion_stiffness, bending_stiffness])

        # Compute rest Darboux vector
        constraint.rest_darboux_vector = self.compute_darboux_vector(
            segment0.rotation, segment1.rotation, average_segment_length
        )

        return constraint

    def update_constraint(self, constraint: RodConstraint) -> None:
        """Update constraint info (world-space connectors) after state change.

        Args:
            constraint: Constraint to update.
        """
        seg0 = self.segments[constraint.segment_index(0)]
        seg1 = self.segments[constraint.segment_index(1)]

        R0 = quat_to_rotation_matrix(seg0.rotation)
        R1 = quat_to_rotation_matrix(seg1.rotation)

        # Update world-space connector positions
        constraint.constraint_info[:, 2] = R0 @ constraint.constraint_info[:, 0] + seg0.position
        constraint.constraint_info[:, 3] = R1 @ constraint.constraint_info[:, 1] + seg1.position

    def init_before_projection(self, inverse_time_step_size: float) -> None:
        """Initialize solver state before projection iterations.

        Computes compliance parameters and resets lambda sums.

        Args:
            inverse_time_step_size: 1/dt.
        """
        inv_dt_sq = inverse_time_step_size * inverse_time_step_size

        for i, constraint in enumerate(self.constraints):
            # Stretch compliance (very small regularization)
            stretch_reg = 1.0e-10
            constraint.stretch_compliance = np.full(3, stretch_reg * inv_dt_sq)

            # Bending/torsion compliance (Equation 24)
            constraint.bending_torsion_compliance = (
                inv_dt_sq / constraint.stiffness_coefficient_k
            ) / constraint.average_segment_length

            # Reset lambda sum
            self.lambda_sums[i] = np.zeros(6)

    def init_tree(self) -> None:
        """Initialize the tree structure for direct solving.

        Builds the alternating segment-constraint-segment tree starting from
        a static (fixed) segment as root.
        """
        # Find root: prefer static segment, otherwise first segment
        root_segment = None
        for segment in self.segments:
            # Check if segment is connected to any constraint
            is_connected = any(segment.index in (c.segment_index(0), c.segment_index(1)) for c in self.constraints)
            if not is_connected:
                continue

            if root_segment is None:
                root_segment = segment
            if not segment.is_dynamic():
                root_segment = segment
                break

        if root_segment is None:
            raise ValueError("No valid root segment found")

        # Create root node
        self.root = Node(is_constraint=False, index=root_segment.index, segment=root_segment)

        # Build tree recursively
        marked_constraints: set[int] = set()
        self._init_segment_node(self.root, marked_constraints)

        # Build forward and backward lists
        self.forward_list = []
        self.backward_list = []
        self._order_matrix(self.root)

    def _init_segment_node(self, node: Node, marked_constraints: set[int]) -> None:
        """Recursively initialize segment node and its children.

        Args:
            node: Current segment node.
            marked_constraints: Set of already visited constraint indices.
        """
        segment = node.segment

        # Find all constraints connected to this segment
        for constraint in self.constraints:
            if constraint.index in marked_constraints:
                continue

            if segment.index not in (
                constraint.segment_index(0),
                constraint.segment_index(1),
            ):
                continue

            # Create constraint node
            constraint_node = Node(
                is_constraint=True,
                index=constraint.index,
                constraint=constraint,
                parent=node,
            )
            node.children.append(constraint_node)
            marked_constraints.add(constraint.index)

            # Get the other segment
            if self.segments[constraint.segment_index(0)] == segment:
                other_segment = self.segments[constraint.segment_index(1)]
            else:
                other_segment = self.segments[constraint.segment_index(0)]

            # Create segment node for the other segment
            segment_node = Node(
                is_constraint=False,
                index=other_segment.index,
                segment=other_segment,
                parent=constraint_node,
            )
            constraint_node.children.append(segment_node)

            # Recursively process the other segment
            self._init_segment_node(segment_node, marked_constraints)

    def _order_matrix(self, node: Node) -> None:
        """Build forward and backward lists for tree traversal.

        Args:
            node: Current node.
        """
        # Process children first (leaves before parents)
        for child in node.children:
            self._order_matrix(child)

        # Add to lists
        self.forward_list.append(node)
        self.backward_list.insert(0, node)

    def factor(self) -> float:
        """Factor the system matrix using LDLT decomposition.

        Computes:
        1. Right-hand side (constraint violations)
        2. Jacobians
        3. Diagonal blocks with Schur complement

        Returns:
            Maximum constraint error.
        """
        max_error = 0.0

        # Update constraint info (world-space connectors)
        for constraint in self.constraints:
            self.update_constraint(constraint)

        # Compute RHS and Jacobians for each constraint
        for i, constraint in enumerate(self.constraints):
            seg0 = self.segments[constraint.segment_index(0)]
            seg1 = self.segments[constraint.segment_index(1)]

            # Get world-space connector positions
            connector0 = constraint.constraint_info[:, 2]
            connector1 = constraint.constraint_info[:, 3]

            # Stretch violation (should be zero)
            stretch_violation = connector0 - connector1

            # Darboux vector (bending/torsion)
            omega = self.compute_darboux_vector(seg0.rotation, seg1.rotation, constraint.average_segment_length)
            bending_torsion_violation = omega - constraint.rest_darboux_vector

            # Fill RHS (Equation 22)
            lambda_sum = self.lambda_sums[i]
            rhs = np.zeros(6)
            rhs[:3] = -stretch_violation - constraint.stretch_compliance * lambda_sum[:3]
            rhs[3:] = -bending_torsion_violation - constraint.bending_torsion_compliance * lambda_sum[3:]
            self.rhs[i] = rhs

            # Track max error
            max_error = max(max_error, np.max(np.abs(rhs)))

            # Compute bending/torsion Jacobians
            G0 = self.compute_matrix_G(seg0.rotation)
            G1 = self.compute_matrix_G(seg1.rotation)

            jOmega0, jOmega1 = self.compute_bending_torsion_jacobians(
                seg0.rotation, seg1.rotation, constraint.average_segment_length
            )

            self.bending_torsion_jacobians[i][0] = jOmega0 @ G0
            self.bending_torsion_jacobians[i][1] = jOmega1 @ G1

        # Build system matrix diagonal and Jacobians
        for node in self.forward_list:
            if node.is_constraint:
                # Constraint node: D = -compliance (diagonal)
                constraint = node.constraint
                node.D = np.zeros((6, 6))
                node.D[:3, :3] = -np.diag(constraint.stretch_compliance)
                node.D[3:, 3:] = -np.diag(constraint.bending_torsion_compliance)
            else:
                # Segment node: D = mass matrix
                node.D = self.get_mass_matrix(node.segment)

            # Compute Jacobian for non-root nodes
            if node.parent is not None:
                if node.is_constraint:
                    # Constraint -> segment Jacobian (J)
                    self._compute_constraint_jacobian(node)
                else:
                    # Segment -> constraint Jacobian (J^T transposed layout)
                    self._compute_segment_jacobian(node)

        # Schur complement factorization
        for node in self.forward_list:
            # Accumulate children's contributions
            for child in node.children:
                JT = child.J.T
                JTDJ = JT @ child.D @ child.J
                node.D = node.D - JTDJ

            # LDLT factorization
            no_zero_dinv = True
            if not node.is_constraint:
                if not node.segment.is_dynamic():
                    node.Dinv = np.zeros((6, 6))
                    no_zero_dinv = False

            if no_zero_dinv:
                try:
                    # Use scipy's LDLT factorization for symmetric indefinite matrices
                    # This is what Eigen's LDLT uses in the C++ code
                    node._lu, node._piv = linalg.lu_factor(node.D)
                    node._factorization_ok = True
                except linalg.LinAlgError:
                    # Fall back to pseudo-inverse if singular
                    node._factorization_ok = False
                    node.Dinv = np.linalg.pinv(node.D)

            # Solve D * J_new = J for non-root nodes
            if node.parent is not None:
                if no_zero_dinv and node._factorization_ok:
                    node.J = linalg.lu_solve((node._lu, node._piv), node.J)
                elif no_zero_dinv:
                    node.J = node.Dinv @ node.J
                else:
                    node.J = np.zeros((6, 6))

        return max_error

    def _compute_constraint_jacobian(self, node: Node) -> None:
        """Compute Jacobian for constraint node (constraint -> parent segment).

        Args:
            node: Constraint node.
        """
        constraint = node.constraint
        parent_segment = node.parent.segment

        # Determine which segment is the parent
        if parent_segment.index == constraint.segment_index(0):
            segment_idx = 0
            sign = 1.0
        else:
            segment_idx = 1
            sign = -1.0

        # Connector position relative to segment center
        connector = constraint.constraint_info[:, 2 + segment_idx]
        r = connector - parent_segment.position

        # Cross product matrix
        r_cross = self.cross_product_matrix(-sign * r)

        # Build 6x6 Jacobian
        J = np.zeros((6, 6))

        # Upper left 3x3: identity * sign
        J[:3, :3] = sign * np.eye(3)

        # Upper right 3x3: r_cross
        J[:3, 3:] = r_cross

        # Lower left 3x3: zero (already initialized)

        # Lower right 3x3: bending/torsion Jacobian
        J[3:, 3:] = self.bending_torsion_jacobians[constraint.index][segment_idx]

        node.J = J

    def _compute_segment_jacobian(self, node: Node) -> None:
        """Compute Jacobian for segment node (segment -> parent constraint).

        Args:
            node: Segment node.
        """
        segment = node.segment
        parent_constraint = node.parent.constraint

        # Determine which segment this is
        if segment.index == parent_constraint.segment_index(0):
            segment_idx = 0
            sign = 1.0
        else:
            segment_idx = 1
            sign = -1.0

        # Connector position
        connector = parent_constraint.constraint_info[:, 2 + segment_idx]
        r = connector - segment.position

        # Build 6x6 Jacobian (transpose layout)
        J = np.zeros((6, 6))

        # Upper left 3x3: identity * sign
        J[:3, :3] = sign * np.eye(3)

        # Lower left 3x3: r_cross (transposed relative to constraint Jacobian)
        J[3:, :3] = self.cross_product_matrix(sign * r)

        # Upper right 3x3: zero

        # Lower right 3x3: transposed bending/torsion Jacobian
        J[3:, 3:] = self.bending_torsion_jacobians[parent_constraint.index][segment_idx].T

        node.J = J

    def solve(self) -> None:
        """Solve the factored system using back-substitution.

        Updates position and rotation corrections in corr_x and corr_q.
        """
        # Forward pass: compute RHS propagation
        for node in self.forward_list:
            if node.is_constraint:
                node.soln = -self.rhs[node.index].copy()
            else:
                node.soln = np.zeros(6)

            # Subtract children's contributions
            for child in node.children:
                node.soln = node.soln - child.J.T @ child.soln

        # Backward pass: solve and back-substitute
        for node in self.backward_list:
            no_zero_dinv = True
            if not node.is_constraint:
                no_zero_dinv = node.segment.is_dynamic()

            if no_zero_dinv:
                # Solve D * x = soln
                if hasattr(node, "_factorization_ok") and node._factorization_ok:
                    node.soln = linalg.lu_solve((node._lu, node._piv), node.soln)
                else:
                    node.soln = node.Dinv @ node.soln

                # Back-substitute from parent
                if node.parent is not None:
                    node.soln = node.soln - node.J @ node.parent.soln
            else:
                node.soln = np.zeros(6)

            # Update lambda sums for constraints
            if node.is_constraint:
                self.lambda_sums[node.index] = self.lambda_sums[node.index] + node.soln

        # Extract position and rotation corrections
        for node in self.forward_list:
            if not node.is_constraint:
                segment = node.segment
                if not segment.is_dynamic():
                    continue

                # Position correction
                self.corr_x[segment.index] = -node.soln[:3]

                # Rotation correction via G matrix
                G = self.compute_matrix_G(segment.rotation)
                delta_q = G @ (-node.soln[3:])
                self.corr_q[segment.index] = delta_q


@dataclass
class SolverDirectConfig:
    """Configuration for the Direct PBD Stiff Rods solver.

    Attributes:
        dt: Time step size.
        substeps: Number of substeps per frame.
        iterations: Number of solver iterations per substep.
        gravity: Gravity acceleration vector [3].
        particle_damping: Particle velocity damping (0 to 1, 1 = no damping).
        quaternion_damping: Quaternion angular velocity damping.
        youngs_modulus: Young's modulus (Pa) for bending stiffness. Default 10 GPa.
        torsion_modulus: Torsion/shear modulus (Pa). Default 5 GPa.
        radius_fraction: Fraction of segment length to use as rod radius. Default 0.25.
    """

    dt: float = 1.0 / 60.0
    substeps: int = 4
    iterations: int = 2
    gravity: NDArray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    particle_damping: float = 0.99
    quaternion_damping: float = 0.99
    youngs_modulus: float = 1.0e10  # 10 GPa
    torsion_modulus: float = 0.5e10  # 5 GPa
    radius_fraction: float = 0.25


class SolverDirectStiffRodsNumpy:
    """Main solver class for Direct PBD Stiff Rods simulation.

    This solver uses the Direct Position Based Solver for Stiff Rods algorithm,
    which formulates constraints as a single linear system solved via LDLT.
    """

    def __init__(self, rod: CosseratRodNumpy, config: SolverDirectConfig | None = None):
        """Initialize the solver.

        Args:
            rod: Cosserat rod data structure.
            config: Solver configuration.
        """
        self.rod = rod
        self.config = config or SolverDirectConfig()

        # Create internal solver
        self._direct_solver = DirectPositionBasedSolverForStiffRods()

        # Initialize segments from rod particles
        self._init_segments()

        # Initialize constraints
        self._init_constraints()

        # Initialize tree structure
        self._direct_solver.init_tree()

        # Allocate working arrays
        n_constraints = len(self._direct_solver.constraints)
        n_segments = len(self._direct_solver.segments)

        self._direct_solver.rhs = [np.zeros(6) for _ in range(n_constraints)]
        self._direct_solver.lambda_sums = [np.zeros(6) for _ in range(n_constraints)]
        self._direct_solver.bending_torsion_jacobians = [
            [np.zeros((3, 3)), np.zeros((3, 3))] for _ in range(n_constraints)
        ]
        self._direct_solver.corr_x = [np.zeros(3) for _ in range(n_segments)]
        self._direct_solver.corr_q = [np.zeros(4) for _ in range(n_segments)]

        # Temporary arrays
        self._positions_old = np.zeros_like(rod.particle_positions)
        self._quaternions_old = np.zeros_like(rod.edge_quaternions)

    def _init_segments(self) -> None:
        """Initialize segments from rod particles and edges."""
        rod = self.rod

        # For stiff rods, each "segment" is centered between two particles
        # and has a quaternion from the edge
        for i in range(rod.num_edges):
            # Segment center is at the edge center
            p0 = rod.particle_positions[i]
            p1 = rod.particle_positions[i + 1]
            center = 0.5 * (p0 + p1)

            # Use edge quaternion
            rotation = rod.edge_quaternions[i].copy()

            # Compute effective mass and inertia
            # A segment is static if its edge quaternion is fixed (edge_inv_mass = 0)
            # This ensures proper tree rooting in the direct solver
            if rod.edge_inv_mass[i] <= 0:
                mass = 0.0  # Static segment (fixed quaternion)
            else:
                m0 = 1.0 / rod.particle_inv_mass[i] if rod.particle_inv_mass[i] > 0 else 0.0
                m1 = 1.0 / rod.particle_inv_mass[i + 1] if rod.particle_inv_mass[i + 1] > 0 else 0.0
                mass = m0 + m1

            # Simple inertia estimate (cylinder approximation)
            # Radius is computed as a fraction of segment length
            length = rod.rest_lengths[i]
            radius = self.config.radius_fraction * length
            Ix = (1.0 / 12.0) * mass * (3 * radius**2 + length**2)
            Iz = 0.5 * mass * radius**2
            inertia = np.array([Ix, Iz, Ix])

            segment = RodSegment(
                index=i,
                position=center,
                rotation=rotation,
                mass=mass,
                inertia_tensor=inertia,
            )
            self._direct_solver.segments.append(segment)

    def _init_constraints(self) -> None:
        """Initialize constraints between adjacent segments."""
        rod = self.rod
        config = self.config

        for i in range(rod.num_bend):
            # Constraint connects segment i and segment i+1
            # Position is at particle i+1 (shared between the two segments)
            constraint_pos = rod.particle_positions[i + 1].copy()

            seg0 = self._direct_solver.segments[i]
            seg1 = self._direct_solver.segments[i + 1]

            # Average length and radius (radius computed from segment length)
            avg_length = 0.5 * (rod.rest_lengths[i] + rod.rest_lengths[i + 1])
            avg_radius = self.config.radius_fraction * avg_length

            constraint = self._direct_solver.init_constraint(
                segment0=seg0,
                segment1=seg1,
                constraint_position=constraint_pos,
                average_radius=avg_radius,
                average_segment_length=avg_length,
                youngs_modulus=config.youngs_modulus,
                torsion_modulus=config.torsion_modulus,
            )

            # Override rest Darboux with rod's stored value
            if i < len(rod.rest_darboux):
                # Convert quaternion rest_darboux to vector
                rest_q = rod.rest_darboux[i]
                # Extract imaginary part scaled
                constraint.rest_darboux_vector = (2.0 / avg_length) * rest_q[:3] / (np.abs(rest_q[3]) + 1e-10)

            self._direct_solver.constraints.append(constraint)

    def step(self) -> None:
        """Advance the simulation by one frame (dt seconds)."""
        sub_dt = self.config.dt / self.config.substeps

        for _ in range(self.config.substeps):
            self._substep(sub_dt)

    def _substep(self, dt: float) -> None:
        """Perform one substep of the simulation.

        Args:
            dt: Substep time increment.
        """
        rod = self.rod
        config = self.config

        # Store old states
        np.copyto(self._positions_old, rod.particle_positions)
        np.copyto(self._quaternions_old, rod.edge_quaternions)

        # Phase 1: Integration (semi-implicit Euler for particles)
        for i in range(rod.num_particles):
            if rod.particle_inv_mass[i] > 0:
                rod.particle_velocities[i] += config.gravity * dt
                rod.particle_positions[i] += rod.particle_velocities[i] * dt

        # Sync segments with rod state
        self._sync_segments_from_rod()

        # Phase 2: Initialize solver for this time step
        self._direct_solver.init_before_projection(1.0 / dt)

        # Phase 3: Projection iterations
        for _ in range(config.iterations):
            # Factor and solve
            self._direct_solver.factor()
            self._direct_solver.solve()

            # Apply corrections to segments
            self._apply_segment_corrections()

        # Sync rod state from segments
        self._sync_rod_from_segments()

        # Normalize quaternions
        rod.normalize_quaternions()

        # Phase 4: Velocity update
        for i in range(rod.num_particles):
            if rod.particle_inv_mass[i] > 0:
                rod.particle_velocities[i] = (rod.particle_positions[i] - self._positions_old[i]) / dt

        # Angular velocity update
        for i in range(rod.num_edges):
            if rod.edge_inv_mass[i] > 0:
                q_old = self._quaternions_old[i]
                q_new = rod.edge_quaternions[i]
                q_delta = quat_multiply(q_new, quat_conjugate(q_old))
                rod.edge_angular_velocities[i] = 2.0 * q_delta[:3] / dt

        # Phase 5: Damping
        rod.particle_velocities *= config.particle_damping
        rod.edge_angular_velocities *= config.quaternion_damping

    def _sync_segments_from_rod(self) -> None:
        """Update segment positions and rotations from rod state.

        Static segments (those with mass=0) are NOT updated - they serve as
        fixed anchors in the direct solver's tree structure.

        For dynamic segments, we compute segment rotations from particle positions
        to ensure proper bending constraint evaluation. This uses a minimal rotation
        approach to update the orientation while preserving twist as much as possible.
        """
        rod = self.rod

        for i, segment in enumerate(self._direct_solver.segments):
            # Only update dynamic segments - static segments stay fixed
            if segment.is_dynamic():
                p0 = rod.particle_positions[i]
                p1 = rod.particle_positions[i + 1]
                segment.position = 0.5 * (p0 + p1)

                # Compute segment direction from particle positions
                edge_dir = p1 - p0
                edge_len = np.linalg.norm(edge_dir)
                if edge_len > 1e-10:
                    edge_dir = edge_dir / edge_len
                else:
                    edge_dir = np.array([0.0, 0.0, 1.0])

                # Get the current d3 (local Z-axis) from stored quaternion
                q_stored = rod.edge_quaternions[i]
                R_stored = quat_to_rotation_matrix(q_stored)
                d3_stored = R_stored[:, 2]

                # Compute minimal rotation from d3_stored to edge_dir
                # This preserves twist around the axis
                segment.rotation = self._compute_minimal_rotation(
                    q_stored, d3_stored, edge_dir
                )

    def _compute_minimal_rotation(
        self, q_base: NDArray, from_dir: NDArray, to_dir: NDArray
    ) -> NDArray:
        """Compute minimal rotation quaternion to align directions.

        Finds the rotation that transforms from_dir to to_dir with minimal
        additional rotation, preserving twist around the axis.

        Args:
            q_base: Base quaternion to start from.
            from_dir: Current direction (should be unit vector).
            to_dir: Target direction (should be unit vector).

        Returns:
            Updated quaternion.
        """
        # Compute rotation axis and angle
        dot = np.clip(np.dot(from_dir, to_dir), -1.0, 1.0)

        if dot > 0.9999:
            # Already aligned
            return q_base.copy()
        elif dot < -0.9999:
            # Anti-parallel: rotate 180 degrees around any perpendicular axis
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(from_dir, perp)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            axis = np.cross(from_dir, perp)
            axis = axis / np.linalg.norm(axis)
            q_rot = np.array([axis[0], axis[1], axis[2], 0.0])
        else:
            # General case
            axis = np.cross(from_dir, to_dir)
            axis_len = np.linalg.norm(axis)
            if axis_len > 1e-10:
                axis = axis / axis_len
            angle = np.arccos(dot)
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)
            q_rot = np.array([
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                cos_half,
            ])

        # Apply rotation to base quaternion
        return quat_normalize(quat_multiply(q_rot, q_base))

    def _sync_rod_from_segments(self) -> None:
        """Update rod state from segment positions and rotations."""
        rod = self.rod
        solver = self._direct_solver

        # Update edge quaternions directly
        for i, segment in enumerate(solver.segments):
            rod.edge_quaternions[i] = quat_normalize(segment.rotation)

        # Update constraint world-space connectors
        for constraint in solver.constraints:
            solver.update_constraint(constraint)

        # Extract particle positions from constraint connectors
        # Particle 0 stays fixed (it's the root)
        # For each constraint i connecting segment i to segment i+1,
        # the connectors give us particle i+1's position
        for i, constraint in enumerate(solver.constraints):
            # constraint_info[:, 2] is the world-space connector from segment constraint.segment_index(0)
            # This should be at particle i+1
            particle_idx = i + 1  # Constraint i gives us particle i+1

            # Use average of both connectors (they should be very close after solving)
            connector0 = constraint.constraint_info[:, 2]
            connector1 = constraint.constraint_info[:, 3]
            rod.particle_positions[particle_idx] = 0.5 * (connector0 + connector1)

        # The last particle (tip) is not covered by constraints
        # Compute it from the last segment using the midpoint relationship
        if rod.num_edges > 0:
            last_seg_idx = rod.num_edges - 1
            last_segment = solver.segments[last_seg_idx]
            last_particle_idx = rod.num_particles - 1
            prev_particle = rod.particle_positions[last_particle_idx - 1]

            # segment.position = midpoint of (p_prev, p_last)
            # => p_last = 2 * segment.position - p_prev
            rod.particle_positions[last_particle_idx] = 2 * last_segment.position - prev_particle

    def _apply_segment_corrections(self) -> None:
        """Apply position and rotation corrections to segments."""
        for i, segment in enumerate(self._direct_solver.segments):
            if segment.is_dynamic():
                segment.position = segment.position + self._direct_solver.corr_x[i]

                # Apply quaternion correction
                delta_q = self._direct_solver.corr_q[i]
                new_q = segment.rotation + delta_q
                segment.rotation = quat_normalize(new_q)

    def get_particle_positions(self) -> NDArray:
        """Get current particle positions."""
        return self.rod.particle_positions.copy()

    def get_edge_quaternions(self) -> NDArray:
        """Get current edge quaternions."""
        return self.rod.edge_quaternions.copy()
