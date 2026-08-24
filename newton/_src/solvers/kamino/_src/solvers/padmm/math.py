# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Defines mathematical operations used by the Proximal-ADMM solver."""

from __future__ import annotations

import warp as wp

###
# Module interface
###

__all__ = [
    "compute_box_complementarity_residual",
    "compute_cwise_vec_div",
    "compute_cwise_vec_mul",
    "compute_desaxce_corrections",
    "compute_dot_product",
    "compute_double_dot_product",
    "compute_gemv",
    "compute_inf_norm",
    "compute_inverse_preconditioned_iterate_residual",
    "compute_l1_norm",
    "compute_l2_norm",
    "compute_ncp_complementarity_residual",
    "compute_ncp_dual_residual",
    "compute_ncp_natural_map_residual",
    "compute_ncp_primal_residual",
    "compute_preconditioned_iterate_residual",
    "compute_vector_sum",
    "project_to_coulomb_cone",
    "project_to_coulomb_dual_cone",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func
def project_to_coulomb_cone(x: wp.vec3f, mu: wp.float32) -> wp.vec3f:
    """
    Projects a 3D vector `x` onto an isotropic Coulomb friction cone defined by the friction coefficient `mu`.

    Args:
        x: The input vector to be projected.
        mu: The friction coefficient defining the aperture of the cone.

    Returns:
        The vector projected onto the Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    # The polar-cone test must precede the membership test: at mu = 0 the primal
    # cone degenerates to the ray { xt = 0, xn >= 0 }, testing xt_norm <= mu * xn
    # first would leave the infeasible point (0, 0, -xn) unchanged.
    if mu * xt_norm > -xn:
        if xt_norm <= mu * xn:
            y = x
        else:
            ys = (mu * xt_norm + xn) / (mu * mu + 1.0)
            yts = mu * ys / xt_norm
            y[0] = yts * x[0]
            y[1] = yts * x[1]
            y[2] = ys
    return y


@wp.func
def project_to_coulomb_dual_cone(x: wp.vec3f, mu: wp.float32) -> wp.vec3f:
    """
    Projects a 3D vector `x` onto the dual of an isotropic Coulomb
    friction cone defined by the friction coefficient `mu`.

    Args:
        x: The input vector to be projected.
        mu: The friction coefficient defining the aperture of the cone.

    Returns:
        The vector projected onto the dual Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    # The membership test must precede the polar-cone test: at mu = 0 the dual
    # cone degenerates to the half-space xn >= 0, testing xt_norm > -mu * xn first
    # would map the feasible point (0, 0, +xn) to zero.
    if mu * xt_norm <= xn:
        y = x
    elif xt_norm > -mu * xn:
        ys = (xt_norm + mu * xn) / (mu * mu + 1.0)
        yts = ys / xt_norm
        y[0] = yts * x[0]
        y[1] = yts * x[1]
        y[2] = mu * ys
    return y


@wp.func
def compute_box_complementarity_residual(
    lambda_value: wp.float32,
    velocity: wp.float32,
    lower: wp.float32,
    upper: wp.float32,
) -> wp.float32:
    """Compute directional complementarity for a box-constrained multiplier.

    Positive velocity complements the lower face and negative velocity
    complements the upper face.
    """
    return (lambda_value - lower) * wp.max(velocity, 0.0) + (upper - lambda_value) * wp.max(-velocity, 0.0)


###
# BLAS-like Utility Functions
#
# TODO: All of these should be re-implemented to exploit parallelism
###


@wp.func
def compute_inf_norm(
    dim: wp.int32,
    vio: wp.int32,
    x: wp.array[wp.float32],
) -> wp.float32:
    norm = float(0.0)
    for i in range(dim):
        norm = wp.max(norm, wp.abs(x[vio + i]))
    return norm


@wp.func
def compute_l1_norm(
    dim: wp.int32,
    vio: wp.int32,
    x: wp.array[wp.float32],
) -> wp.float32:
    sum = float(0.0)
    for i in range(dim):
        sum += wp.abs(x[vio + i])
    return sum


@wp.func
def compute_l2_norm(
    dim: wp.int32,
    vio: wp.int32,
    x: wp.array[wp.float32],
) -> wp.float32:
    sum = float(0.0)
    for i in range(dim):
        x_i = x[vio + i]
        sum += x_i * x_i
    return wp.sqrt(sum)


@wp.func
def compute_dot_product(
    dim: wp.int32,
    vio: wp.int32,
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
) -> wp.float32:
    """
    Computes the dot (i.e. inner) product between two vectors `x` and `y` stored in flat arrays.
    Both vectors are of dimension `dim`, starting from the vector index offset `vio`.

    Args:
        dim: The dimension (i.e. size) of the vectors.
        vio: The vector index offset (i.e. start index).
        x: The first vector.
        y: The second vector.

    Returns:
        The dot product of the two vectors.
    """
    product = float(0.0)
    for i in range(dim):
        v_i = vio + i
        product += x[v_i] * y[v_i]
    return product


@wp.func
def compute_double_dot_product(
    dim: wp.int32,
    vio: wp.int32,
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    z: wp.array[wp.float32],
) -> wp.float32:
    """
    Computes the the inner product `x.T @ (y + z)` between a vector `x` and the sum of two vectors `y` and `z`.
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim: The dimension (i.e. size) of the vectors.
        vio: The vector index offset (i.e. start index).
        x: The first vector.
        y: The second vector.
        z: The third vector.

    Returns:
        The inner product of `x` with the sum of `y` and `z`.
    """
    product = float(0.0)
    for i in range(dim):
        v_i = vio + i
        product += x[v_i] * (y[v_i] + z[v_i])
    return product


@wp.func
def compute_vector_sum(
    dim: wp.int32, vio: wp.int32, x: wp.array[wp.float32], y: wp.array[wp.float32], z: wp.array[wp.float32]
):
    """
    Computes the sum of two vectors `x` and `y` and stores the result in vector `z`.
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim: The dimension (i.e. size) of the vectors.
        vio: The vector index offset (i.e. start index).
        x: The first vector.
        y: The second vector.
        z: The output vector where the sum is stored.

    Returns:
        The result is stored in the output vector `z`.
    """
    for i in range(dim):
        v_i = vio + i
        z[v_i] = x[v_i] + y[v_i]


@wp.func
def compute_cwise_vec_mul(
    dim: wp.int32,
    vio: wp.int32,
    a: wp.array[wp.float32],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
):
    """
    Computes the coefficient-wise vector-vector product `y =  a * x`.

    Args:
        dim: The dimension (i.e. size) of the vectors.
        vio: The vector index offset (i.e. start index).
        a: Input array containing the first set of vectors.
        x: Input array containing the second set of vectors.
        y: Output array where the result is stored.
    """
    for i in range(dim):
        v_i = vio + i
        y[v_i] = a[v_i] * x[v_i]


@wp.func
def compute_cwise_vec_div(
    dim: wp.int32,
    vio: wp.int32,
    a: wp.array[wp.float32],
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
):
    """
    Computes the coefficient-wise vector-vector division `y =  a / x`.

    Args:
        dim: The dimension (i.e. size) of the vectors.
        vio: The vector index offset (i.e. start index).
        a: Input array containing the first set of vectors.
        x: Input array containing the second set of vectors.
        y: Output array where the result is stored.
    """
    for i in range(dim):
        v_i = vio + i
        y[v_i] = a[v_i] / x[v_i]


@wp.func
def compute_gemv(
    dim: wp.int32,
    vio: wp.int32,
    mio: wp.int32,
    sigma: wp.float32,
    P: wp.array[wp.float32],
    A: wp.array[wp.float32],
    x: wp.array[wp.float32],
    b: wp.array[wp.float32],
    c: wp.array[wp.float32],
):
    """
    Computes the generalized matrix-vector product `c =  b + (A - sigma * I_n)@ x`.

    The matrix `A` is stored using row-major order in flat array with allocation size `maxdim x maxdim`,
    starting from the matrix index offset `mio`. The active dimensions of the matrix are `dim x dim`,
    where `dim` is the number of rows and columns. The vectors `x, b, c` are stored in flat arrays with
    dimensions `dim`, starting from the vector index offset `vio`.

    Args:
        maxdim: The maximum dimension of the matrix `A`.
        dim: The active dimension of the matrix `A` and the vectors `x, b, c`.
        vio: The vector index offset (i.e. start index) for the vectors `x, b, c`.
        mio: The matrix index offset (i.e. start index) for the matrix `A`.
        A: Input matrix `A` stored in row-major order.
        x: Input array `x` to be multiplied with the matrix `A`.
        b: Input array `b` to be added to the product `A @ x`.
        c: Output array `c` where the result of the operation is stored.
    """
    b_i = float(0.0)
    x_j = float(0.0)
    for i in range(dim):
        v_i = vio + i
        m_i = mio + dim * i
        b_i = b[v_i]
        for j in range(dim):
            x_j = x[vio + j]
            b_i += A[m_i + j] * x_j
        b_i -= sigma * x[v_i]
        c[v_i] = (1.0 / P[v_i]) * b_i


@wp.func
def compute_desaxce_corrections(
    nc: wp.int32,
    cio: wp.int32,
    vio: wp.int32,
    ccgo: wp.int32,
    mu: wp.array[wp.float32],
    v_plus: wp.array[wp.float32],
    s: wp.array[wp.float32],
):
    """
    Computes the De Saxce correction for each active contact.

    `s = G(v) := [ 0, 0 , mu * || vt ||_2,]^T, where v := [vtx, vty, vn]^T, vt := [vtx, vty]^T`

    Args:
        nc: The number of active contact constraints.
        cio: The contact index offset (i.e. start index) for the contacts.
        vio: The vector index offset (i.e. start index)
        ccgo: The contact constraint group offset (i.e. start index)
        mu: The array of friction coefficients for each contact constraint.
        v_plus: The post-event constraint-space velocities array, which contains the tangential velocities `vtx`
            and `vty` for each contact constraint.
        s: The output array where the De Saxce corrections are stored.
            The size of this array should be at least `vio + ccgo + 3 * nc`, where `vio` is the vector index offset,
            `ccgo` is the contact constraint group offset, and `nc` is the number of active contact constraints.

    Returns:
        The De Saxce corrections are stored in the output array `s`.
    """
    # Iterate over each active contact
    for k in range(nc):
        # Compute the contact index
        c_k = cio + k

        # Compute the constraint vector index
        v_k = vio + ccgo + 3 * k

        # Retrieve the friction coefficient for this contact
        mu_k = mu[c_k]

        # Compute the 2D norm of the tangential velocity
        vtx_k = v_plus[v_k]
        vty_k = v_plus[v_k + 1]
        vt_norm_k = wp.sqrt(vtx_k * vtx_k + vty_k * vty_k)

        # Store De Saxce correction for this block
        s[v_k] = 0.0
        s[v_k + 1] = 0.0
        s[v_k + 2] = mu_k * vt_norm_k


@wp.func
def compute_ncp_primal_residual(
    nbc: wp.int32,
    nl: wp.int32,
    nc: wp.int32,
    vio: wp.int32,
    bcio: wp.int32,
    bcgo: wp.int32,
    lcgo: wp.int32,
    ccgo: wp.int32,
    cio: wp.int32,
    mu: wp.array[wp.float32],
    bound_lower: wp.array[wp.float32],
    bound_upper: wp.array[wp.float32],
    P: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
) -> tuple[wp.float32, wp.int32]:
    """
    Computes the NCP primal residual as: `r_p := || lambda - proj_C(lambda) ||_inf`, where:
    - `lambda` is the vector of constraint reactions (i.e. impulses)
    - `proj_C()` is the projection operator onto the feasible set `C`
    - `C` is the product of bounded-multiplier boxes, unilateral limit cones, and Coulomb friction cones
    - `|| . ||_inf` is the infinity norm (i.e. maximum absolute value of the vector components)

    Notes:
    - The cone for bilateral joint constraints is all of `R^njc`, so projection is a no-op.
    - For bounded-multiplier constraints, the set is the box `C_b := { lambda | lower <= lambda <= upper }`
    - For limit constraints, the cone is defined as `K_l := { lambda | lambda >= 0 }`
    - For contact constraints, the cone is defined as `K_c := { lambda | || lambda ||_2 <= mu * || vn ||_2 }`

    Args:
        nbc: The number of active bounded-multiplier constraints.
        nl: The number of active limit constraints.
        nc: The number of active contact constraints.
        vio: The vector index offset (i.e. start index) for the constraints.
        bcio: The bounded-multiplier index offset (i.e. start index).
        bcgo: The bounded-multiplier constraint group offset (i.e. start index).
        lcgo: The limit constraint group offset (i.e. start index).
        ccgo: The contact constraint group offset (i.e. start index).
        cio: The contact index offset (i.e. start index) for the contacts.
        mu: The array of friction coefficients for each contact.
        bound_lower: The lower bounds in preconditioned coordinates.
        bound_upper: The upper bounds in preconditioned coordinates.
        P: The dual-problem diagonal preconditioner.
        lambdas: The array of constraint reactions (i.e. impulses).

    Returns:
        The maximum primal residual across all constraints, computed as the infinity norm,
        and the index of the constraint with the maximum primal residual.
    """
    # Initialize the primal residual
    r_ncp_p = float(0.0)
    r_ncp_p_argmax = wp.int32(-1)

    # NOTE: We skip the bilateral joint constraint, as their reactions are not bounded.

    for bid in range(nbc):
        # Bounds are stored in preconditioned coordinates, while the residual uses physical reactions.
        bcio_b = bcio + bid
        bcio_v = vio + bcgo + bid
        lower = P[bcio_v] * bound_lower[bcio_b]
        upper = P[bcio_v] * bound_upper[bcio_b]
        # Compute the primal residual for the bounded-multiplier constraints
        lambda_b = lambdas[bcio_v]
        r_b = wp.abs(lambda_b - wp.clamp(lambda_b, lower, upper))
        r_ncp_p = wp.max(r_ncp_p, r_b)
        if r_ncp_p == r_b:
            r_ncp_p_argmax = bid

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio = vio + lcgo + lid
        # Compute the primal residual for the limit constraints
        lambda_l = lambdas[lcio]
        lambda_l -= wp.max(0.0, lambda_l)
        r_l = wp.abs(lambda_l)
        r_ncp_p = wp.max(r_ncp_p, r_l)
        if r_ncp_p == r_l:
            r_ncp_p_argmax = lid

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio = vio + ccgo + 3 * cid
        # Retrieve the friction coefficient for this contact
        mu_c = mu[cio + cid]
        # Compute the primal residual for the contact constraints
        lambda_c = wp.vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        lambda_c -= project_to_coulomb_cone(lambda_c, mu_c)
        r_c = wp.max(wp.abs(lambda_c))
        r_ncp_p = wp.max(r_ncp_p, r_c)
        if r_ncp_p == r_c:
            r_ncp_p_argmax = cid

    # Return the maximum primal residual
    return r_ncp_p, r_ncp_p_argmax


@wp.func
def compute_ncp_dual_residual(
    njc: wp.int32,
    nl: wp.int32,
    nc: wp.int32,
    vio: wp.int32,
    lcgo: wp.int32,
    ccgo: wp.int32,
    cio: wp.int32,
    mu: wp.array[wp.float32],
    v_aug: wp.array[wp.float32],
) -> tuple[wp.float32, wp.int32]:
    """
    Computes the NCP dual residual as: `r_d := || v_aug - proj_K^*(v_aug) ||_inf`, where:
    - `v_aug` is the vector of augmented constraint velocities: v_aug := v_plus + s
    - `v_plus` is the post-event constraint-space velocities
    - `s` is the De Saxce correction vector
    - `proj_K^*()` is the projection operator onto the dual cone `K^*`
    - `K^*` is the dual of the total cone defined by unilateral limit and contact constraints
    - `|| . ||_inf` is the infinity norm (i.e. maximum absolute value of the vector components)

    Notes:
    - The dual cone for joint constraints is the origin point x=0.
    - For limit constraints, the cone is defined as `K_l := { lambda | lambda >= 0 }`
    - For contact constraints, the cone is defined as `K_c := { lambda | || lambda ||_2 <= mu * || vn ||_2 }`
    - Bounded-multiplier constraints contribute no term. Their box is bounded on both sides, so its
      recession cone is the origin and the admissible set for `v_aug` is all of `R`. Box optimality is
      certified instead by the primal and complementarity residuals, which together are equivalent to
      the box KKT conditions: see :func:`compute_ncp_primal_residual` and
      :func:`compute_box_complementarity_residual`.

    Args:
        njc: The number of bilateral joint constraints.
        nl: The number of active limit constraints.
        nc: The number of active contact constraints.
        vio: The vector index offset (i.e. start index) for the constraints.
        lcgo: The limit constraint group offset (i.e. start index).
        ccgo: The contact constraint group offset (i.e. start index).
        cio: The contact index offset (i.e. start index) for the contacts.
        mu: The array of friction coefficients for each contact constraint.
        v_aug: The array of augmented constraint velocities.

    Returns:
        The maximum dual residual across all constraints, computed as the infinity norm,
        and the index of the constraint with the maximum dual residual.
    """
    # Initialize the dual residual
    r_ncp_d = float(0.0)
    r_ncp_d_argmax = wp.int32(-1)

    for jid in range(njc):
        # Compute the joint constraint index offset
        jcio_j = vio + jid
        # Compute the dual residual for the joint constraints
        # NOTE #1: Each constraint-space velocity for joint should be zero
        # NOTE #2: the dual of R^njc is the origin zero vector
        v_j = v_aug[jcio_j]
        r_j = wp.abs(v_j)
        r_ncp_d = wp.max(r_ncp_d, r_j)
        if r_ncp_d == r_j:
            r_ncp_d_argmax = jid

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio_l = vio + lcgo + lid
        # Compute the dual residual for the limit constraints
        # NOTE: Each constraint-space velocity should be non-negative
        v_l = v_aug[lcio_l]
        v_l -= wp.max(0.0, v_l)
        r_l = wp.abs(v_l)
        r_ncp_d = wp.max(r_ncp_d, r_l)
        if r_ncp_d == r_l:
            r_ncp_d_argmax = lid

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio_c = vio + ccgo + 3 * cid
        # Retrieve the friction coefficient for this contact
        mu_c = mu[cio + cid]
        # Compute the dual residual for the contact constraints
        # NOTE: Each constraint-space velocity should be lie in the dual of the Coulomb friction cone
        v_c = wp.vec3f(v_aug[ccio_c], v_aug[ccio_c + 1], v_aug[ccio_c + 2])
        v_c -= project_to_coulomb_dual_cone(v_c, mu_c)
        r_c = wp.max(wp.abs(v_c))
        r_ncp_d = wp.max(r_ncp_d, r_c)
        if r_ncp_d == r_c:
            r_ncp_d_argmax = cid

    # Return the maximum dual residual
    return r_ncp_d, r_ncp_d_argmax


@wp.func
def compute_ncp_complementarity_residual(
    nbc: wp.int32,
    nl: wp.int32,
    nc: wp.int32,
    vio: wp.int32,
    bcio: wp.int32,
    bcgo: wp.int32,
    lcgo: wp.int32,
    ccgo: wp.int32,
    bound_lower: wp.array[wp.float32],
    bound_upper: wp.array[wp.float32],
    P: wp.array[wp.float32],
    v_aug: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
) -> tuple[wp.float32, wp.int32]:
    """
    Computes the generalized NCP complementarity residual.

    For each bounded multiplier, directional products measure complementarity
    with the lower and upper box faces. For each limit, the residual is
    `v_k * lambda_k`, and for each contact it is `v_k.dot(lambda_k)`.

    Args:
        nbc: The number of active bounded-multiplier constraints.
        nl: The number of active limit constraints.
        nc: The number of active contact constraints.
        vio: The vector index offset (i.e. start index) for the constraints.
        bcio: The bounded-multiplier index offset (i.e. start index).
        bcgo: The bounded-multiplier constraint group offset (i.e. start index).
        lcgo: The limit constraint group offset (i.e. start index).
        ccgo: The contact constraint group offset (i.e. start index).
        bound_lower: The lower bounds in preconditioned coordinates.
        bound_upper: The upper bounds in preconditioned coordinates.
        P: The dual-problem diagonal preconditioner.
        v_aug: The array of augmented constraint velocities.
        lambdas: The array of constraint reactions (i.e. impulses).

    Returns:
        The maximum complementarity residual across all constraints, computed as the infinity norm,
        and the index of the constraint with the maximum complementarity residual.
    """
    # Initialize the complementarity residual
    r_ncp_c = float(0.0)
    r_ncp_c_argmax = wp.int32(-1)

    for bid in range(nbc):
        bcio_b = bcio + bid
        bcio_v = vio + bcgo + bid
        lower = P[bcio_v] * bound_lower[bcio_b]
        upper = P[bcio_v] * bound_upper[bcio_b]
        r_b = wp.abs(compute_box_complementarity_residual(lambdas[bcio_v], v_aug[bcio_v], lower, upper))
        r_ncp_c = wp.max(r_ncp_c, r_b)
        if r_ncp_c == r_b:
            r_ncp_c_argmax = bid

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio = vio + lcgo + lid
        # Compute the complementarity residual for the limit constraints
        v_l = v_aug[lcio]
        lambda_l = lambdas[lcio]
        r_l = wp.abs(v_l * lambda_l)
        r_ncp_c = wp.max(r_ncp_c, r_l)
        if r_ncp_c == r_l:
            r_ncp_c_argmax = lid

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio = vio + ccgo + 3 * cid
        # Compute the complementarity residual for the contact constraints
        v_c = wp.vec3f(v_aug[ccio], v_aug[ccio + 1], v_aug[ccio + 2])
        lambda_c = wp.vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        r_c = wp.abs(wp.dot(v_c, lambda_c))
        r_ncp_c = wp.max(r_ncp_c, r_c)
        if r_ncp_c == r_c:
            r_ncp_c_argmax = cid

    # Return the maximum complementarity residual
    return r_ncp_c, r_ncp_c_argmax


@wp.func
def compute_ncp_natural_map_residual(
    njc: wp.int32,
    nbc: wp.int32,
    nl: wp.int32,
    nc: wp.int32,
    vio: wp.int32,
    bcio: wp.int32,
    bcgo: wp.int32,
    lcgo: wp.int32,
    ccgo: wp.int32,
    cio: wp.int32,
    mu: wp.array[wp.float32],
    bound_lower: wp.array[wp.float32],
    bound_upper: wp.array[wp.float32],
    P: wp.array[wp.float32],
    v_aug: wp.array[wp.float32],
    lambdas: wp.array[wp.float32],
) -> tuple[wp.float32, wp.int32]:
    """
    Computes the natural-map residual as: `r_natmap = || lambda - proj_C(lambda - (v + s)) ||_inf`,
    where `C` is the product of bounded-multiplier box constraints, unilateral limit cones, and
    Coulomb friction cones.

    Notes:
    - For bilateral joint constraints, the cone is all of `R^njc`, so the natural-map residual is `abs(v_aug)`.
    - For bounded-multiplier constraints, the feasible set is the box
      `C_b := { lambda | lower <= lambda <= upper }`, so the natural-map residual is
      `abs(lambda - clamp(lambda - v_aug, lower, upper))`.
    - For limit constraints, the cone is defined as `K_l := { lambda | lambda >= 0 }`.
    - For contact constraints, the cone is defined as `K_c := { lambda | || lambda ||_2 <= mu * || vn ||_2 }`.

    Args:
        njc: The number of bilateral joint constraints.
        nbc: The number of active bounded-multiplier constraints.
        nl: The number of active limit constraints.
        nc: The number of active contact constraints.
        vio: The vector index offset (i.e. start index) for the constraints.
        bcio: The bounded-multiplier constraint index offset (i.e. start index).
        bcgo: The bounded-multiplier constraint group offset (i.e. start index).
        lcgo: The limit constraint group offset (i.e. start index).
        ccgo: The contact constraint group offset (i.e. start index).
        cio: The contact index offset (i.e. start index) for the contacts.
        mu: The array of friction coefficients for each contact.
        bound_lower: The lower bounds in preconditioned coordinates.
        bound_upper: The upper bounds in preconditioned coordinates.
        P: The dual-problem diagonal preconditioner.
        v_aug: The array of augmented constraint velocities.
        lambdas: The array of constraint reactions (i.e. impulses).

    Returns:
        The maximum natural-map residual across all constraints, computed as the infinity norm,
        and the index of the constraint with the maximum natural-map residual.
    """

    # Initialize the natural-map residual
    r_ncp_natmap = float(0.0)
    r_ncp_natmap_argmax = wp.int32(-1)

    for jid in range(njc):
        jcio_j = vio + jid
        v_j = v_aug[jcio_j]
        r_j = wp.abs(v_j)
        r_ncp_natmap = wp.max(r_ncp_natmap, r_j)
        if r_ncp_natmap == r_j:
            r_ncp_natmap_argmax = jid

    for bid in range(nbc):
        # Bounds are stored in preconditioned coordinates, while the residual uses physical reactions.
        bcio_b = bcio + bid
        bcio_v = vio + bcgo + bid
        lower = P[bcio_v] * bound_lower[bcio_b]
        upper = P[bcio_v] * bound_upper[bcio_b]
        lambda_b = lambdas[bcio_v]
        r_b = wp.abs(lambda_b - wp.clamp(lambda_b - v_aug[bcio_v], lower, upper))
        r_ncp_natmap = wp.max(r_ncp_natmap, r_b)
        if r_ncp_natmap == r_b:
            r_ncp_natmap_argmax = bid

    for lid in range(nl):
        # Compute the limit constraint index offset
        lcio = vio + lcgo + lid
        # Compute the natural-map residual for the limit constraints
        v_l = v_aug[lcio]
        lambda_l = lambdas[lcio]
        lambda_l -= wp.max(0.0, lambda_l - v_l)
        lambda_l = wp.abs(lambda_l)
        r_ncp_natmap = wp.max(r_ncp_natmap, lambda_l)
        if r_ncp_natmap == lambda_l:
            r_ncp_natmap_argmax = lid

    for cid in range(nc):
        # Compute the contact constraint index offset
        ccio = vio + ccgo + 3 * cid
        # Retrieve the friction coefficient for this contact
        mu_c = mu[cio + cid]
        # Compute the natural-map residual for the contact constraints
        v_c = wp.vec3f(v_aug[ccio], v_aug[ccio + 1], v_aug[ccio + 2])
        lambda_c = wp.vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        lambda_c -= project_to_coulomb_cone(lambda_c - v_c, mu_c)
        lambda_c = wp.abs(lambda_c)
        lambda_c_max = wp.max(lambda_c)
        r_ncp_natmap = wp.max(r_ncp_natmap, lambda_c_max)
        if r_ncp_natmap == lambda_c_max:
            r_ncp_natmap_argmax = cid

    # Return the maximum natural-map residual
    return r_ncp_natmap, r_ncp_natmap_argmax


@wp.func
def compute_preconditioned_iterate_residual(
    ncts: wp.int32, vio: wp.int32, P: wp.array[wp.float32], x: wp.array[wp.float32], x_p: wp.array[wp.float32]
) -> wp.float32:
    """
    Computes the iterate residual as: `r_dx := || P @ (x - x_p) ||_2`

    Args:
        ncts: The number of active constraints in the world.
        vio: The vector index offset (i.e. start index) for the constraints.
        x: The current solution vector.
        x_p: The previous solution vector.

    Returns:
        The iterate residual across all active constraints, computed as the L2 norm.
    """
    # Initialize the iterate residual
    r_dx_squared = float(0.0)
    for i in range(ncts):
        # Compute the index offset of the vector block of the world
        v_i = vio + i
        # Update the iterate and proximal-point residuals
        delta = P[v_i] * (x[v_i] - x_p[v_i])
        r_dx_squared += delta * delta
    # Return the iterate residual
    return wp.sqrt(r_dx_squared)


@wp.func
def compute_inverse_preconditioned_iterate_residual(
    ncts: wp.int32, vio: wp.int32, P: wp.array[wp.float32], x: wp.array[wp.float32], x_p: wp.array[wp.float32]
) -> wp.float32:
    """
    Computes the iterate residual as: `r_dx := || P^{-1} @ (x - x_p) ||_2`

    Args:
        ncts: The number of active constraints in the world.
        vio: The vector index offset (i.e. start index) for the constraints.
        x: The current solution vector.
        x_p: The previous solution vector.

    Returns:
        The iterate residual across all active constraints, computed as the L2 norm.
    """
    # Initialize the iterate residual
    r_dx_squared = float(0.0)
    for i in range(ncts):
        # Compute the index offset of the vector block of the world
        v_i = vio + i
        # Update the iterate and proximal-point residuals
        delta = (1.0 / P[v_i]) * (x[v_i] - x_p[v_i])
        r_dx_squared += delta * delta
    # Return the iterate residual
    return wp.sqrt(r_dx_squared)
