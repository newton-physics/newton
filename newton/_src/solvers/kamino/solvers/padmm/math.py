# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines mathematical operations used by the Proximal-ADMM solver."""

import warp as wp

from ...core.types import float32, int32, vec3f

###
# Module interface
###

__all__ = [
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
def project_to_coulomb_cone(x: vec3f, mu: float32, epsilon: float32 = 0.0) -> vec3f:
    """
    Projects a 3D vector `x` onto an isotropic Coulomb friction cone defined by the friction coefficient `mu`.

    Args:
        x (vec3f): The input vector to be projected.
        mu (float32): The friction coefficient defining the aperture of the cone.
        epsilon (float32, optional): A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        vec3f: The vector projected onto the Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if mu * xt_norm > -xn + epsilon:
        if xt_norm <= mu * xn + epsilon:
            y = x
        else:
            ys = (mu * xt_norm + xn) / (mu * mu + 1.0)
            yts = mu * ys / xt_norm
            y[0] = yts * x[0]
            y[1] = yts * x[1]
            y[2] = ys
    return y


@wp.func
def project_to_coulomb_dual_cone(x: vec3f, mu: float32, epsilon: float32 = 0.0) -> vec3f:
    """
    Projects a 3D vector `x` onto the dual of an isotropic Coulomb
    friction cone defined by the friction coefficient `mu`.

    Args:
        x (vec3f): The input vector to be projected.
        mu (float32): The friction coefficient defining the aperture of the cone.
        epsilon (float32, optional): A numerical tolerance applied to the cone boundary. Defaults to 0.0.

    Returns:
        vec3f: The vector projected onto the dual Coulomb cone.
    """
    xn = x[2]
    xt_norm = wp.sqrt(x[0] * x[0] + x[1] * x[1])
    y = wp.vec3f(0.0)
    if xt_norm > -mu * xn + epsilon:
        if mu * xt_norm <= xn + epsilon:
            y = x
        else:
            ys = (xt_norm + mu * xn) / (mu * mu + 1.0)
            yts = ys / xt_norm
            y[0] = yts * x[0]
            y[1] = yts * x[1]
            y[2] = mu * ys
    return y


###
# BLAS-like Utility Functions
#
# TODO: All of these should be re-implemented to exploit parallelism
###


@wp.func
def compute_inf_norm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
) -> float32:
    norm = float(0.0)
    for i in range(dim):
        norm = wp.max(norm, wp.abs(x[vio + i]))
    return norm


@wp.func
def compute_l1_norm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
) -> float32:
    sum = float(0.0)
    for i in range(dim):
        sum += wp.abs(x[vio + i])
    return sum


@wp.func
def compute_l2_norm(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
) -> float32:
    sum = float(0.0)
    for i in range(dim):
        x_i = x[vio + i]
        sum += x_i * x_i
    return wp.sqrt(sum)


@wp.func
def compute_dot_product(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
) -> float32:
    """
    Computes the dot (i.e. inner) product between two vectors `x` and `y` stored in flat arrays.\n
    Both vectors are of dimension `dim`, starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.

    Returns:
        float32: The dot product of the two vectors.
    """
    product = float(0.0)
    for i in range(dim):
        v_i = vio + i
        product += x[v_i] * y[v_i]
    return product


@wp.func
def compute_double_dot_product(
    dim: int32,
    vio: int32,
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
    z: wp.array(dtype=float32),
) -> float32:
    """
    Computes the the inner product `x.T @ (y + z)` between a vector `x` and the sum of two vectors `y` and `z`.\n
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.
        z (wp.array(dtype=float32)): The third vector.

    Returns:
        float32: The inner product of `x` with the sum of `y` and `z`.
    """
    product = float(0.0)
    for i in range(dim):
        v_i = vio + i
        product += x[v_i] * (y[v_i] + z[v_i])
    return product


@wp.func
def compute_vector_sum(
    dim: int32, vio: int32, x: wp.array(dtype=float32), y: wp.array(dtype=float32), z: wp.array(dtype=float32)
):
    """
    Computes the sum of two vectors `x` and `y` and stores the result in vector `z`.\n
    All vectors are stored in flat arrays, with dimension `dim` and starting from the vector index offset `vio`.

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        x (wp.array(dtype=float32)): The first vector.
        y (wp.array(dtype=float32)): The second vector.
        z (wp.array(dtype=float32)): The output vector where the sum is stored.

    Returns:
        None: The result is stored in the output vector `z`.
    """
    for i in range(dim):
        v_i = vio + i
        z[v_i] = x[v_i] + y[v_i]


@wp.func
def compute_cwise_vec_mul(
    dim: int32,
    vio: int32,
    a: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    y: wp.array(dtype=float32),
):
    """
    Computes the coefficient-wise vector-vector product `y =  a * x`.\n

    Args:
        dim (int32): The dimension (i.e. size) of the vectors.
        vio (int32): The vector index offset (i.e. start index).
        a (wp.array(dtype=float32)): Input array containing the first set of vectors.
        x (wp.array(dtype=float32)): Input array containing the second set of vectors.
        y (wp.array(dtype=float32)): Output array where the result is stored.
    """
    for i in range(dim):
        v_i = vio + i
        y[v_i] = a[v_i] * x[v_i]


@wp.func
def compute_gemv(
    dim: int32,
    vio: int32,
    mio: int32,
    sigma: float32,
    P: wp.array(dtype=float32),
    A: wp.array(dtype=float32),
    x: wp.array(dtype=float32),
    b: wp.array(dtype=float32),
    c: wp.array(dtype=float32),
):
    """
    Computes the generalized matrix-vector product `c =  b + (A - sigma * I_n)@ x`.\n

    The matrix `A` is stored using row-major order in flat array with allocation size `maxdim x maxdim`,
    starting from the matrix index offset `mio`. The active dimensions of the matrix are `dim x dim`,
    where `dim` is the number of rows and columns. The vectors `x, b, c` are stored in flat arrays with
    dimensions `dim`, starting from the vector index offset `vio`.

    Args:
        maxdim (int32): The maximum dimension of the matrix `A`.
        dim (int32): The active dimension of the matrix `A` and the vectors `x, b, c`.
        vio (int32): The vector index offset (i.e. start index) for the vectors `x, b, c`.
        mio (int32): The matrix index offset (i.e. start index) for the matrix `A`.
        A (wp.array(dtype=float32)):
            Input matrix `A` stored in row-major order.
        x (wp.array(dtype=float32)):
            Input array `x` to be multiplied with the matrix `A`.
        b (wp.array(dtype=float32)):
            Input array `b` to be added to the product `A @ x`.
        c (wp.array(dtype=float32)):
            Output array `c` where the result of the operation is stored.
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
    nc: int32,
    cio: int32,
    vio: int32,
    ccgo: int32,
    mu: wp.array(dtype=float32),
    v_plus: wp.array(dtype=float32),
    s: wp.array(dtype=float32),
):
    """
    Computes the De Saxce correction for each active contact.

    `s = G(v) := [ 0, 0 , mu * || vt ||_2,]^T, where v := [vtx, vty, vn]^T, vt := [vtx, vty]^T`

    Args:
        nc (int32): The number of active contact constraints.
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        vio (int32): The vector index offset (i.e. start index)
        ccgo (int32): The contact constraint group offset (i.e. start index)
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact constraint.
        v_plus (wp.array(dtype=float32)):
            The post-event constraint-space velocities array, which contains the tangential velocities `vtx`
            and `vty` for each contact constraint.
        s (wp.array(dtype=float32)):
            The output array where the De Saxce corrections are stored.\n
            The size of this array should be at least `vio + ccgo + 3 * nc`, where `vio` is the vector index offset,
            `ccgo` is the contact constraint group offset, and `nc` is the number of active contact constraints.

    Returns:
        None: The De Saxce corrections are stored in the output array `s`.
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
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    mu: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
) -> tuple[float32, int32]:
    """
    Computes the NCP primal residual as: `r_p := || lambda - proj_K(lambda) ||_inf`, where:
    - `lambda` is the vector of constraint reactions (i.e. impulses)
    - `proj_K()` is the projection operator onto the cone `K`
    - `K` is the total cone defined by the unilateral constraints such as limits and contacts
    - `|| . ||_inf` is the infinity norm (i.e. maximum absolute value of the vector components)

    Notes:
    - The cone for joint constraints is all of `R^njc`, so projection is a no-op.
    - For limit constraints, the cone is defined as `K_l := { lambda | lambda >= 0 }`
    - For contact constraints, the cone is defined as `K_c := { lambda | || lambda ||_2 <= mu * || vn ||_2 }`

    Args:
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact.
        lambdas (wp.array(dtype=float32)):
            The array of constraint reactions (i.e. impulses).

    Returns:
        float32: The maximum primal residual across all constraints, computed as the infinity norm.
        int32: The index of the constraint with the maximum primal residual.
    """
    # Initialize the primal residual
    r_ncp_p = float(0.0)
    r_ncp_p_argmax = int32(-1)

    # NOTE: We skip the joint constraint reactions are not bounded, the cone is all of R^njc

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
        lambda_c = vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        lambda_c -= project_to_coulomb_cone(lambda_c, mu_c)
        r_c = wp.max(wp.abs(lambda_c))
        r_ncp_p = wp.max(r_ncp_p, r_c)
        if r_ncp_p == r_c:
            r_ncp_p_argmax = cid

    # Return the maximum primal residual
    return r_ncp_p, r_ncp_p_argmax


@wp.func
def compute_ncp_dual_residual(
    njc: int32,
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    mu: wp.array(dtype=float32),
    v_aug: wp.array(dtype=float32),
) -> tuple[float32, int32]:
    """
    Computes the NCP dual residual as: `r_d := || v_aug - proj_K^*(v_aug) ||_inf`, where:
    - `v_aug` is the vector of augmented constraint velocities: v_aug := v_plus + s
    - `v_plus` is the post-event constraint-space velocities
    - `s` is the De Saxce correction vector
    - `proj_K^*()` is the projection operator onto the dual cone `K^*`
    - `K^*` is the dual of the total cone defined by the unilateral constraints such as limits and contacts
    - `|| . ||_inf` is the infinity norm (i.e. maximum absolute value of the vector components)

    Notes:
    - The dual cone for joint constraints is the origin point x=0.
    - For limit constraints, the cone is defined as `K_l := { lambda | lambda >= 0 }`
    - For contact constraints, the cone is defined as `K_c := { lambda | || lambda ||_2 <= mu * || vn ||_2 }`

    Args:
        njc (int32): The number of joint constraints.
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact constraint.
        v_aug (wp.array(dtype=float32)):
            The array of augmented constraint velocities.

    Returns:
        float32: The maximum dual residual across all constraints, computed as the infinity norm.
        int32: The index of the constraint with the maximum dual residual.
    """
    # Initialize the dual residual
    r_ncp_d = float(0.0)
    r_ncp_d_argmax = int32(-1)

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
        v_c = vec3f(v_aug[ccio_c], v_aug[ccio_c + 1], v_aug[ccio_c + 2])
        v_c -= project_to_coulomb_dual_cone(v_c, mu_c)
        r_c = wp.max(wp.abs(v_c))
        r_ncp_d = wp.max(r_ncp_d, r_c)
        if r_ncp_d == r_c:
            r_ncp_d_argmax = cid

    # Return the maximum dual residual
    return r_ncp_d, r_ncp_d_argmax


@wp.func
def compute_ncp_complementarity_residual(
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    v_aug: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
) -> tuple[float32, int32]:
    """
    Computes the NCP complementarity residual as `r_c := || lambda.dot(v_plus + s) ||_inf`

    Satisfaction of the complementarity condition `lambda _|_ (v_plus + s))` is measured
    using the per-constraint entity inner product, i.e. per limit and per contact. Thus,
    for each limit constraint `k`, we compute `v_k * lambda_k` and for each contact
    constraint `k`, we compute `v_k.dot(lambda_k)`.

    Args:
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        v_aug (wp.array(dtype=float32)):
            The array of augmented constraint velocities.
        lambdas (wp.array(dtype=float32)):
            The array of constraint reactions (i.e. impulses).

    Returns:
        float32: The maximum complementarity residual across all constraints, computed as the infinity norm.
        int32: The index of the constraint with the maximum complementarity residual.
    """
    # Initialize the complementarity residual
    r_ncp_c = float(0.0)
    r_ncp_c_argmax = int32(-1)

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
        v_c = vec3f(v_aug[ccio], v_aug[ccio + 1], v_aug[ccio + 2])
        lambda_c = vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
        r_c = wp.abs(wp.dot(v_c, lambda_c))
        r_ncp_c = wp.max(r_ncp_c, r_c)
        if r_ncp_c == r_c:
            r_ncp_c_argmax = cid

    # Return the maximum complementarity residual
    return r_ncp_c, r_ncp_c_argmax


@wp.func
def compute_ncp_natural_map_residual(
    nl: int32,
    nc: int32,
    vio: int32,
    lcgo: int32,
    ccgo: int32,
    cio: int32,
    mu: wp.array(dtype=float32),
    v_aug: wp.array(dtype=float32),
    lambdas: wp.array(dtype=float32),
) -> tuple[float32, int32]:
    """
    Computes the natural-map residuals as: `r_natmap = || lambda - proj_K(lambda - (v + s)) ||_inf`

    Args:
        nl (int32): The number of active limit constraints.
        nc (int32): The number of active contact constraints.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        lcgo (int32): The limit constraint group offset (i.e. start index).
        ccgo (int32): The contact constraint group offset (i.e. start index).
        cio (int32): The contact index offset (i.e. start index) for the contacts.
        mu (wp.array(dtype=float32)):
            The array of friction coefficients for each contact.
        v_aug (wp.array(dtype=float32)):
            The array of augmented constraint velocities.
        lambdas (wp.array(dtype=float32)):
            The array of constraint reactions (i.e. impulses).

    Returns:
        float32: The maximum natural-map residual across all constraints, computed as the infinity norm.
        int32: The index of the constraint with the maximum natural-map residual.
    """

    # Initialize the natural-map residual
    r_ncp_natmap = float(0.0)
    r_ncp_natmap_argmax = int32(-1)

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
        v_c = vec3f(v_aug[ccio], v_aug[ccio + 1], v_aug[ccio + 2])
        lambda_c = vec3f(lambdas[ccio], lambdas[ccio + 1], lambdas[ccio + 2])
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
    ncts: int32, vio: int32, P: wp.array(dtype=float32), x: wp.array(dtype=float32), x_p: wp.array(dtype=float32)
) -> float32:
    """
    Computes the iterate residual as: `r_dx := || P @ (x - x_p) ||_inf`

    Args:
        ncts (int32): The number of active constraints in the world.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        x (wp.array(dtype=float32)):
            The current solution vector.
        x_p (wp.array(dtype=float32)):
            The previous solution vector.

    Returns:
        float32: The maximum iterate residual across all active constraints, computed as the infinity norm.
    """
    # Initialize the iterate residual
    r_dx = float(0.0)
    for i in range(ncts):
        # Compute the index offset of the vector block of the world
        v_i = vio + i
        # Update the iterate and proximal-point residuals
        r_dx = wp.max(r_dx, P[v_i] * wp.abs(x[v_i] - x_p[v_i]))
    # Return the maximum iterate residual
    return r_dx


@wp.func
def compute_inverse_preconditioned_iterate_residual(
    ncts: int32, vio: int32, P: wp.array(dtype=float32), x: wp.array(dtype=float32), x_p: wp.array(dtype=float32)
) -> float32:
    """
    Computes the iterate residual as: `r_dx := || P^{-1} @ (x - x_p) ||_inf`

    Args:
        ncts (int32): The number of active constraints in the world.
        vio (int32): The vector index offset (i.e. start index) for the constraints.
        x (wp.array(dtype=float32)):
            The current solution vector.
        x_p (wp.array(dtype=float32)):
            The previous solution vector.

    Returns:
        float32: The maximum iterate residual across all active constraints, computed as the infinity norm.
    """
    # Initialize the iterate residual
    r_dx = float(0.0)
    for i in range(ncts):
        # Compute the index offset of the vector block of the world
        v_i = vio + i
        # Update the iterate and proximal-point residuals
        r_dx = wp.max(r_dx, (1.0 / P[v_i]) * wp.abs(x[v_i] - x_p[v_i]))
    # Return the maximum iterate residual
    return r_dx
