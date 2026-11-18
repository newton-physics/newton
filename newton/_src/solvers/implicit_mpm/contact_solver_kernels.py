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
    check_end = end - (end - beg) % 8
    for check_entry in range(beg, check_end, 8):
        check_row0 = cell_rows[check_entry + 0]
        check_row1 = cell_rows[check_entry + 1]
        check_row2 = cell_rows[check_entry + 2]
        check_row3 = cell_rows[check_entry + 3]
        check_row4 = cell_rows[check_entry + 4]
        check_row5 = cell_rows[check_entry + 5]
        check_row6 = cell_rows[check_entry + 6]
        check_row7 = cell_rows[check_entry + 7]
        check_count0 = offsets[check_row0 + 1] - offsets[check_row0]
        check_count1 = offsets[check_row1 + 1] - offsets[check_row1]
        check_count2 = offsets[check_row2 + 1] - offsets[check_row2]
        check_count3 = offsets[check_row3 + 1] - offsets[check_row3]
        check_count4 = offsets[check_row4 + 1] - offsets[check_row4]
        check_count5 = offsets[check_row5 + 1] - offsets[check_row5]
        check_count6 = offsets[check_row6 + 1] - offsets[check_row6]
        check_count7 = offsets[check_row7 + 1] - offsets[check_row7]
        if (
            check_count0 != 8
            or check_count1 != 8
            or check_count2 != 8
            or check_count3 != 8
            or check_count4 != 8
            or check_count5 != 8
            or check_count6 != 8
            or check_count7 != 8
        ):
            full_rows = False
    for entry in range(check_end, end):
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

    contact_quad_end = end - (end - beg) % 4
    for contact_entry in range(beg, contact_quad_end, 4):
        contact_row0 = cell_rows[contact_entry + 0]
        contact_row1 = cell_rows[contact_entry + 1]
        contact_row2 = cell_rows[contact_entry + 2]
        contact_row3 = cell_rows[contact_entry + 3]
        distinct_rows = (
            contact_row0 != contact_row1
            and contact_row0 != contact_row2
            and contact_row0 != contact_row3
            and contact_row1 != contact_row2
            and contact_row1 != contact_row3
            and contact_row2 != contact_row3
        )
        contact_diagonal0 = diagonal[contact_row0]
        contact_friction0 = friction[contact_row0]
        contact_diagonal1 = diagonal[contact_row1]
        contact_friction1 = friction[contact_row1]
        contact_diagonal2 = diagonal[contact_row2]
        contact_friction2 = friction[contact_row2]
        contact_diagonal3 = diagonal[contact_row3]
        contact_friction3 = friction[contact_row3]
        contact_weights0 = _ContactScalar8()
        contact_normal0 = wp.vec3(0.0)
        contact_boundary0 = wp.vec3(0.0)
        contact_adhesion0 = float(0.0)
        contact_impulse0 = wp.vec3(0.0)
        if contact_diagonal0 > 0.0 and contact_friction0 >= 0.0:
            contact_impulse0 = impulse[contact_row0]
            contact_normal0 = normal[contact_row0]
            contact_boundary0 = boundary_velocity[contact_row0]
            contact_adhesion0 = adhesion[contact_row0]
            for contact_lane in range(8):
                contact_weights0[contact_lane] = values[offsets[contact_row0] + contact_lane]
        contact_weights1 = _ContactScalar8()
        contact_normal1 = wp.vec3(0.0)
        contact_boundary1 = wp.vec3(0.0)
        contact_adhesion1 = float(0.0)
        contact_impulse1 = wp.vec3(0.0)
        if contact_diagonal1 > 0.0 and contact_friction1 >= 0.0:
            contact_impulse1 = impulse[contact_row1]
            contact_normal1 = normal[contact_row1]
            contact_boundary1 = boundary_velocity[contact_row1]
            contact_adhesion1 = adhesion[contact_row1]
            for contact_lane in range(8):
                contact_weights1[contact_lane] = values[offsets[contact_row1] + contact_lane]
        contact_weights2 = _ContactScalar8()
        contact_normal2 = wp.vec3(0.0)
        contact_boundary2 = wp.vec3(0.0)
        contact_adhesion2 = float(0.0)
        contact_impulse2 = wp.vec3(0.0)
        if contact_diagonal2 > 0.0 and contact_friction2 >= 0.0:
            contact_impulse2 = impulse[contact_row2]
            contact_normal2 = normal[contact_row2]
            contact_boundary2 = boundary_velocity[contact_row2]
            contact_adhesion2 = adhesion[contact_row2]
            for contact_lane in range(8):
                contact_weights2[contact_lane] = values[offsets[contact_row2] + contact_lane]
        contact_weights3 = _ContactScalar8()
        contact_normal3 = wp.vec3(0.0)
        contact_boundary3 = wp.vec3(0.0)
        contact_adhesion3 = float(0.0)
        contact_impulse3 = wp.vec3(0.0)
        if contact_diagonal3 > 0.0 and contact_friction3 >= 0.0:
            contact_impulse3 = impulse[contact_row3]
            contact_normal3 = normal[contact_row3]
            contact_boundary3 = boundary_velocity[contact_row3]
            contact_adhesion3 = adhesion[contact_row3]
            for contact_lane in range(8):
                contact_weights3[contact_lane] = values[offsets[contact_row3] + contact_lane]
        for contact_slot in range(4):
            contact_row = contact_row0
            contact_diagonal = contact_diagonal0
            contact_friction = contact_friction0
            contact_weights = contact_weights0
            contact_normal = contact_normal0
            contact_boundary = contact_boundary0
            contact_adhesion = contact_adhesion0
            contact_previous_impulse = contact_impulse0
            if wp.static(contact_slot == 1):
                contact_row = contact_row1
                contact_diagonal = contact_diagonal1
                contact_friction = contact_friction1
                contact_weights = contact_weights1
                contact_normal = contact_normal1
                contact_boundary = contact_boundary1
                contact_adhesion = contact_adhesion1
                contact_previous_impulse = contact_impulse1
            if wp.static(contact_slot == 2):
                contact_row = contact_row2
                contact_diagonal = contact_diagonal2
                contact_friction = contact_friction2
                contact_weights = contact_weights2
                contact_normal = contact_normal2
                contact_boundary = contact_boundary2
                contact_adhesion = contact_adhesion2
                contact_previous_impulse = contact_impulse2
            if wp.static(contact_slot == 3):
                contact_row = contact_row3
                contact_diagonal = contact_diagonal3
                contact_friction = contact_friction3
                contact_weights = contact_weights3
                contact_normal = contact_normal3
                contact_boundary = contact_boundary3
                contact_adhesion = contact_adhesion3
                contact_previous_impulse = contact_impulse3
            if contact_diagonal > 0.0 and contact_friction >= 0.0:
                if not distinct_rows:
                    contact_previous_impulse = impulse[contact_row]
                contact_velocity = (
                    wp.vec3(wp.dot(contact_weights, vx), wp.dot(contact_weights, vy), wp.dot(contact_weights, vz))
                    - contact_boundary
                )
                normal_vector = contact_normal
                projected_velocity = solve_coulomb_isotropic(
                    contact_friction,
                    normal_vector,
                    contact_velocity - (contact_previous_impulse + contact_adhesion * normal_vector) * contact_diagonal,
                )
                delta_velocity = (projected_velocity - contact_velocity) / contact_diagonal
                impulse[contact_row] = contact_previous_impulse + delta_velocity
                delta_impulse[contact_row] = delta_velocity
                weighted_mass = wp.cw_mul(mass, contact_weights)
                vx += weighted_mass * delta_velocity[0]
                vy += weighted_mass * delta_velocity[1]
                vz += weighted_mass * delta_velocity[2]
    for contact_entry in range(contact_quad_end, end):
        contact_row = cell_rows[contact_entry]
        contact_diagonal = diagonal[contact_row]
        contact_friction = friction[contact_row]
        contact_weights = _ContactScalar8()
        if contact_diagonal > 0.0 and contact_friction >= 0.0:
            for contact_lane in range(8):
                contact_weights[contact_lane] = values[offsets[contact_row] + contact_lane]
        if contact_diagonal > 0.0 and contact_friction >= 0.0:
            contact_velocity = (
                wp.vec3(wp.dot(contact_weights, vx), wp.dot(contact_weights, vy), wp.dot(contact_weights, vz))
                - boundary_velocity[contact_row]
            )
            normal_vector = normal[contact_row]
            projected_velocity = solve_coulomb_isotropic(
                contact_friction,
                normal_vector,
                contact_velocity - (impulse[contact_row] + adhesion[contact_row] * normal_vector) * contact_diagonal,
            )
            delta_velocity = (projected_velocity - contact_velocity) / contact_diagonal
            impulse[contact_row] += delta_velocity
            delta_impulse[contact_row] = delta_velocity
            weighted_mass = wp.cw_mul(mass, contact_weights)
            vx += weighted_mass * delta_velocity[0]
            vy += weighted_mass * delta_velocity[1]
            vz += weighted_mass * delta_velocity[2]

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
