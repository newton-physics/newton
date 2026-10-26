# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import warp.fem as fem
import warp.sparse as sp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def compute_collider_inv_mass(
    J_mat_offsets: wp.array[int],
    J_mat_columns: wp.array[int],
    J_mat_values: wp.array[wp.mat33],
    IJtm_mat_offsets: wp.array[int],
    IJtm_mat_columns: wp.array[int],
    IJtm_mat_values: wp.array[wp.mat33],
    collider_inv_mass: wp.array[float],
):
    i = wp.tid()

    block_beg = J_mat_offsets[i]
    block_end = J_mat_offsets[i + 1]

    w_mat = wp.mat33(0.0)

    for b in range(block_beg, block_end):
        col = J_mat_columns[b]
        transposed_block = sp.bsr_block_index(col, i, IJtm_mat_offsets, IJtm_mat_columns)
        if transposed_block == -1:
            continue

        # Mass-splitting: divide by number of nodes overlapping with this body
        multiplicity = float(IJtm_mat_offsets[col + 1] - IJtm_mat_offsets[col])

        w_mat += (J_mat_values[b] @ IJtm_mat_values[transposed_block]) * multiplicity

    _eigvecs, eigvals = wp.eig3(w_mat)
    collider_inv_mass[i] = wp.max(0.0, wp.max(eigvals))


@wp.func
def project_on_friction_cone(
    mu: float,
    nor: wp.vec3,
    r: wp.vec3,
):
    """Projects a stress vector ``r`` onto the Coulomb friction cone (non-orthogonally)."""

    r_n = wp.dot(r, nor)
    r_t = r - r_n * nor

    r_n = wp.max(0.0, r_n)
    mu_rn = mu * r_n

    r_t_n2 = wp.length_sq(r_t)
    if r_t_n2 > mu_rn * mu_rn:
        r_t *= mu_rn / wp.sqrt(r_t_n2)

    return r_n * nor + r_t


@wp.func
def solve_coulomb_isotropic(
    mu: float,
    nor: wp.vec3,
    u: wp.vec3,
):
    """Solves for the relative velocity in the Coulomb friction model,
    assuming an isotropic velocity-impulse relationship, u = r + b
    """

    u_n = wp.dot(u, nor)
    if u_n < 0.0:
        u -= u_n * nor
        tau = wp.length_sq(u)
        alpha = mu * u_n
        if tau <= alpha * alpha:
            u = wp.vec3(0.0)
        else:
            u *= 1.0 + mu * u_n / wp.sqrt(tau)

    return u


@wp.func
def filter_collider_impulse_warmstart(
    friction: float,
    nor: wp.vec3,
    adhesion: float,
    impulse: wp.vec3,
):
    """Filters the collider impulse to be within the friction cone"""

    if friction < 0.0:
        return wp.vec3(0.0)

    return project_on_friction_cone(friction, nor, impulse + adhesion * nor) - adhesion * nor


@wp.kernel
def apply_nodal_impulse_warmstart(
    collider_impulse: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_adhesion: wp.array[float],
    inv_mass: wp.array[float],
    velocities: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    """
    Applies pre-computed impulses to particles and colliders.
    """
    i = wp.tid()

    impulse = filter_collider_impulse_warmstart(
        collider_friction[i], collider_normals[i], collider_adhesion[i], collider_impulse[i]
    )

    collider_impulse[i] = impulse
    delta_impulse[i] = impulse
    velocities[i] += inv_mass[i] * impulse


@wp.kernel
def solve_nodal_friction(
    inv_mass: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_inv_mass: wp.array[float],
    velocities: wp.array[wp.vec3],
    collider_velocities: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    """
    Solves for frictional impulses at nodes interacting with colliders.

    For each node (i) potentially in contact:
    1. Skips if friction coefficient is negative (no friction).
    2. Calculates the relative velocity `u0` between the particle and collider,
       accounting for the existing impulse and adhesion.
    3. Computes the effective inverse mass `w` for the interaction.
    4. Calls `solve_coulomb_isotropic` to determine the change in relative
       velocity `u` due to friction.
    5. Calculates the change in impulse `delta_impulse` required to achieve this
       change in relative velocity.
    6. Updates the total impulse, particle velocity, and collider velocity.
    """
    i = wp.tid()

    friction_coeff = collider_friction[i]
    if friction_coeff < 0.0:
        return

    n = collider_normals[i]
    u0 = velocities[i] - collider_velocities[i]

    w = inv_mass[i] + collider_inv_mass[i]

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - (impulse[i] + collider_adhesion[i] * n) * w)

    delta_u = u - u0
    delta_lambda = delta_u / w

    delta_impulse[i] = delta_lambda
    impulse[i] += delta_lambda
    velocities[i] += inv_mass[i] * delta_lambda


@wp.kernel
def apply_subgrid_impulse(
    tr_collider_mat_offsets: wp.array[int],
    tr_collider_mat_columns: wp.array[int],
    tr_collider_mat_values: wp.array[float],
    inv_mass: wp.array[float],
    impulses: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
):
    """
    Applies pre-computed impulses to particles and colliders.
    """

    u_i = wp.tid()
    block_beg = tr_collider_mat_offsets[u_i]
    block_end = tr_collider_mat_offsets[u_i + 1]
    if block_beg == block_end:
        return

    delta_f = wp.vec3(0.0)
    for b in range(block_beg, block_end):
        delta_f += tr_collider_mat_values[b] * impulses[tr_collider_mat_columns[b]]

    velocities[u_i] += inv_mass[u_i] * delta_f


@wp.kernel
def apply_subgrid_impulse_warmstart(
    collider_friction: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_adhesion: wp.array[float],
    collider_impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    i = wp.tid()

    impulse = filter_collider_impulse_warmstart(
        collider_friction[i], collider_normals[i], collider_adhesion[i], collider_impulse[i]
    )

    collider_impulse[i] = impulse
    delta_impulse[i] = impulse


@wp.kernel
def compute_collider_delassus_diagonal(
    collider_mat_offsets: wp.array[int],
    collider_mat_columns: wp.array[int],
    collider_mat_values: wp.array[float],
    collider_inv_mass: wp.array[float],
    transposed_collider_mat_offsets: wp.array[int],
    inv_volume: wp.array[float],
    delassus_diagonal: wp.array[float],
):
    i = wp.tid()

    block_beg = collider_mat_offsets[i]
    block_end = collider_mat_offsets[i + 1]

    inv_mass = collider_inv_mass[i]
    w = inv_mass

    for b in range(block_beg, block_end):
        u_i = collider_mat_columns[b]
        weight = collider_mat_values[b]

        multiplicity = transposed_collider_mat_offsets[u_i + 1] - transposed_collider_mat_offsets[u_i]

        w += weight * weight * inv_volume[u_i] * float(multiplicity)

    delassus_diagonal[i] = w


@wp.kernel
def solve_subgrid_friction(
    velocity: wp.array[wp.vec3],
    collider_mat_offsets: wp.array[int],
    collider_mat_columns: wp.array[int],
    collider_mat_values: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_delassus_diagonal: wp.array[float],
    collider_velocities: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    i = wp.tid()

    w = collider_delassus_diagonal[i]
    friction_coeff = collider_friction[i]
    if w <= 0.0 or friction_coeff < 0.0:
        return

    beg = collider_mat_offsets[i]
    end = collider_mat_offsets[i + 1]

    u0 = -collider_velocities[i]
    for b in range(beg, end):
        u_i = collider_mat_columns[b]
        u0 += collider_mat_values[b] * velocity[u_i]

    n = collider_normals[i]

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - (impulse[i] + collider_adhesion[i] * n) * w)

    delta_u = u - u0
    delta_lambda = delta_u / w

    impulse[i] += delta_lambda
    delta_impulse[i] = delta_lambda


@wp.kernel
def compute_collider_unsplit_diagonal(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    inv_volume: wp.array[float],
    collider_inv_mass: wp.array[float],
    diagonal: wp.array[float],
):
    i = wp.tid()
    # Grid nodes use GS; rigid-body feedback remains mass-split between sweeps.
    w = collider_inv_mass[i]
    for b in range(offsets[i], offsets[i + 1]):
        w += values[b] * values[b] * inv_volume[columns[b]]
    diagonal[i] = w


@wp.func
def _solve_cell_contacts_gs_generic(
    cell: int,
    cell_offsets: wp.array[int],
    cell_rows: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    friction: wp.array[float],
    adhesion: wp.array[float],
    normal: wp.array[wp.vec3],
    diagonal: wp.array[float],
    boundary_velocity: wp.array[wp.vec3],
    inv_volume: wp.array[float],
    velocity: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    # Same-cell contacts overlap, so update them sequentially with the true diagonal.
    for entry in range(cell_offsets[cell], cell_offsets[cell + 1]):
        i = cell_rows[entry]
        w = diagonal[i]
        if w > 0.0 and friction[i] >= 0.0:
            u0 = -boundary_velocity[i]
            for b in range(offsets[i], offsets[i + 1]):
                u0 += values[b] * velocity[columns[b]]
            n = normal[i]
            u = solve_coulomb_isotropic(friction[i], n, u0 - (impulse[i] + adhesion[i] * n) * w)
            delta = (u - u0) / w
            impulse[i] += delta
            delta_impulse[i] = delta
            for b in range(offsets[i], offsets[i + 1]):
                node = columns[b]
                velocity[node] += inv_volume[node] * values[b] * delta


_ContactScalar8 = wp.types.vector(length=8, dtype=float)
_ContactIndex8 = wp.types.vector(length=8, dtype=int)


@wp.func
def solve_cell_contacts_gs(
    cell: int,
    cell_offsets: wp.array[int],
    cell_rows: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    friction: wp.array[float],
    adhesion: wp.array[float],
    normal: wp.array[wp.vec3],
    diagonal: wp.array[float],
    boundary_velocity: wp.array[wp.vec3],
    inv_volume: wp.array[float],
    velocity: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    beg = cell_offsets[cell]
    end = cell_offsets[cell + 1]
    if beg == end:
        return

    first = cell_rows[beg]
    full_rows = bool(True)
    for entry in range(beg, end):
        row = cell_rows[entry]
        if offsets[row + 1] - offsets[row] != 8:
            full_rows = False
    if not full_rows:
        _solve_cell_contacts_gs_generic(
            cell,
            cell_offsets,
            cell_rows,
            offsets,
            columns,
            values,
            friction,
            adhesion,
            normal,
            diagonal,
            boundary_velocity,
            inv_volume,
            velocity,
            impulse,
            delta_impulse,
        )
        return

    # Full eight-node rows in one cell share its sorted velocity nodes.
    # Cache them across the sequential local contacts, scattering only once.
    nodes = _ContactIndex8()
    mass = _ContactScalar8()
    vx = _ContactScalar8()
    vy = _ContactScalar8()
    vz = _ContactScalar8()
    for k in range(8):
        node = columns[offsets[first] + k]
        nodes[k] = node
        mass[k] = inv_volume[node]
        v = velocity[node]
        vx[k] = v[0]
        vy[k] = v[1]
        vz[k] = v[2]

    for entry in range(beg, end):
        i = cell_rows[entry]
        w = diagonal[i]
        if w > 0.0 and friction[i] >= 0.0:
            weights = _ContactScalar8()
            for k in range(8):
                weights[k] = values[offsets[i] + k]
            u0 = wp.vec3(wp.dot(weights, vx), wp.dot(weights, vy), wp.dot(weights, vz)) - boundary_velocity[i]
            n = normal[i]
            u = solve_coulomb_isotropic(friction[i], n, u0 - (impulse[i] + adhesion[i] * n) * w)
            delta = (u - u0) / w
            impulse[i] += delta
            delta_impulse[i] = delta
            mw = wp.cw_mul(mass, weights)
            vx += mw * delta[0]
            vy += mw * delta[1]
            vz += mw * delta[2]

    for k in range(8):
        velocity[nodes[k]] = wp.vec3(vx[k], vy[k], vz[k])


@wp.struct
class ColoredContacts:
    """Particle contact rows grouped by strain cell, solved within the colored strain sweep."""

    cell_offsets: wp.array[int]
    cell_rows: wp.array[int]
    offsets: wp.array[int]
    columns: wp.array[int]
    values: wp.array[float]
    friction: wp.array[float]
    adhesion: wp.array[float]
    normal: wp.array[wp.vec3]
    diagonal: wp.array[float]
    boundary_velocity: wp.array[wp.vec3]
    impulse: wp.array[wp.vec3]
    delta_impulse: wp.array[wp.vec3]


def make_solve_colored_cell_contacts(nodes_per_cell: int):
    """Solve one strain cell's contact rows, caching eight-node cells in registers."""
    solve_cell = solve_cell_contacts_gs if nodes_per_cell == 8 else _solve_cell_contacts_gs_generic

    @fem.cache.dynamic_func(suffix=nodes_per_cell)
    def solve_colored_cell_contacts(
        cell: int,
        contacts: ColoredContacts,
        inv_volume: wp.array[float],
        velocity: wp.array[wp.vec3],
    ):
        wp.static(solve_cell)(
            cell,
            contacts.cell_offsets,
            contacts.cell_rows,
            contacts.offsets,
            contacts.columns,
            contacts.values,
            contacts.friction,
            contacts.adhesion,
            contacts.normal,
            contacts.diagonal,
            contacts.boundary_velocity,
            inv_volume,
            velocity,
            contacts.impulse,
            contacts.delta_impulse,
        )

    return solve_colored_cell_contacts
