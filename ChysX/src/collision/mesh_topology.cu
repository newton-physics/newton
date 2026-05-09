// SPDX-License-Identifier: Apache-2.0

#include "mesh_topology.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chysx {
namespace collision {

namespace {

// 64-bit packed key for an undirected edge (lo32 = min vid, hi32 = max vid).
// Lets us use std::unordered_map without a custom hasher.
inline std::uint64_t pack_edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
            static_cast<std::uint32_t>(b);
}

}  // namespace

void MeshTopology::build(const std::vector<math::Vec3i>& tris,
                         int n_verts) {
    n_verts_ = n_verts;
    n_faces_ = static_cast<int>(tris.size());

    // ---- 1. Dedup edges + record the 1 or 2 faces touching each one.
    //
    // Algorithm and on-disk layout match cuda-cloth's
    // `ClothBW98::PreprocessConnectivity`: edges are unordered (min,
    // max), and edge2face[e] = (f0, f1) with -1 in the second slot for
    // boundary edges.

    std::unordered_map<std::uint64_t, int> edge_id;
    edge_id.reserve(static_cast<std::size_t>(n_faces_) * 3);

    std::vector<math::Vec2i> edges_h;
    edges_h.reserve(static_cast<std::size_t>(n_faces_) * 3 / 2);

    std::vector<math::Vec2i> edge2face_h;  // (face0, face1), -1 == empty
    edge2face_h.reserve(edges_h.capacity());

    // face2edge[f] = (e0, e1, e2) ordered by encounter, no semantics
    // beyond "the 3 edges of face f".  Used as scratch for v2e and
    // edge_in_face.
    std::vector<math::Vec3i> face2edge(n_faces_, math::Vec3i(-1, -1, -1));

    auto get_or_make_edge = [&](int a, int b) -> int {
        const std::uint64_t key = pack_edge_key(a, b);
        const auto it = edge_id.find(key);
        if (it != edge_id.end()) return it->second;
        const int eid = static_cast<int>(edges_h.size());
        edge_id.emplace(key, eid);
        edges_h.push_back(math::Vec2i(std::min(a, b), std::max(a, b)));
        edge2face_h.push_back(math::Vec2i(-1, -1));
        return eid;
    };

    for (int fid = 0; fid < n_faces_; ++fid) {
        const math::Vec3i f = tris[static_cast<std::size_t>(fid)];
        const int e0 = get_or_make_edge(f.x, f.y);
        const int e1 = get_or_make_edge(f.y, f.z);
        const int e2 = get_or_make_edge(f.z, f.x);
        face2edge[static_cast<std::size_t>(fid)] = math::Vec3i(e0, e1, e2);
        for (int e : {e0, e1, e2}) {
            math::Vec2i& slot = edge2face_h[static_cast<std::size_t>(e)];
            if (slot.x < 0)      slot.x = fid;
            else if (slot.y < 0) slot.y = fid;
            // 3+ triangles sharing an edge => non-manifold, silently drop.
        }
    }

    n_edges_ = static_cast<int>(edges_h.size());

    // ---- 2. vert_in_edge: cuda-cloth's "v2e[v] = first incident edge
    // matching v -> v_next in face-traversal order".  Each global
    // vertex ends up owned by exactly one edge.  Different from a
    // "lowest-indexed incident edge" assignment, but the ownership
    // property is the same -- and matching cuda-cloth byte-for-byte
    // makes it easier to cross-check the BVH-driven VF emissions.

    std::vector<int> v2e(n_verts_, -1);
    for (int fid = 0; fid < n_faces_; ++fid) {
        const math::Vec3i tri = tris[static_cast<std::size_t>(fid)];
        const math::Vec3i f2e = face2edge[static_cast<std::size_t>(fid)];
        for (int j = 0; j < 3; ++j) {
            const int v      = tri.data[j];
            const int v_next = tri.data[(j + 1) % 3];
            if (v2e[static_cast<std::size_t>(v)] != -1) continue;
            for (int k = 0; k < 3; ++k) {
                const int eid = f2e.data[k];
                if (eid < 0) continue;
                const math::Vec2i e = edges_h[static_cast<std::size_t>(eid)];
                if ((e.x == v && e.y == v_next) ||
                    (e.x == v_next && e.y == v)) {
                    v2e[static_cast<std::size_t>(v)] = eid;
                    break;
                }
            }
        }
    }

    std::vector<int> vert_in_edge_h(n_edges_, -1);
    for (int v = 0; v < n_verts_; ++v) {
        const int e = v2e[static_cast<std::size_t>(v)];
        if (e >= 0) vert_in_edge_h[static_cast<std::size_t>(e)] = v;
    }

    // ---- 3. edge_in_face with cuda-cloth's load-balanced ownership.
    //
    // For boundary edges (only one incident face) the unique face
    // claims them.  For shared edges the face whose `num[f]` slot
    // counter is smaller takes the edge; that way the per-face
    // edge_in_face table stays sparse (3 entries max) AND each edge
    // is owned by exactly ONE face -- which is what makes the EE
    // narrow-phase dedup automatically (edge eid only appears in
    // the candidate list of its owner face, so no two distinct EF
    // candidates ever spawn the same EE pair).

    std::vector<int> num(n_faces_, 0);
    std::vector<math::Vec3i> edge_in_face_h(n_faces_, math::Vec3i(-1, -1, -1));
    for (int eid = 0; eid < n_edges_; ++eid) {
        const math::Vec2i ef = edge2face_h[static_cast<std::size_t>(eid)];
        const int f0 = ef.x;
        const int f1 = ef.y;
        if (f1 < 0) {
            int& n0 = num[static_cast<std::size_t>(f0)];
            edge_in_face_h[static_cast<std::size_t>(f0)].data[n0] = eid;
            ++n0;
        } else {
            int& n0 = num[static_cast<std::size_t>(f0)];
            int& n1 = num[static_cast<std::size_t>(f1)];
            if (n0 <= n1) {
                edge_in_face_h[static_cast<std::size_t>(f0)].data[n0] = eid;
                ++n0;
            } else {
                edge_in_face_h[static_cast<std::size_t>(f1)].data[n1] = eid;
                ++n1;
            }
        }
    }

    // ---- 4. Adjacent EE pairs (cuda-cloth's `m_prt_ee`).
    //
    // For each triangle f and each edge eid of f:
    //   other_point = the vertex of f not on eid.
    //   For every edge `nbr` incident to other_point that is NOT one
    //   of the 3 edges of f and has `nbr > eid`, emit (eid, nbr).
    //
    // This is the "diagonal across an adjacent face" pattern: e0 in
    // face f vs. an edge incident to f's opposite vertex.  Crucially,
    // it AVOIDS the trivial "e0 and e1 share a vertex" pairs that
    // collapse to zero distance and produce NaN normals.

    std::vector<std::vector<int>> vert_to_edges(n_verts_);
    for (int eid = 0; eid < n_edges_; ++eid) {
        const math::Vec2i e = edges_h[static_cast<std::size_t>(eid)];
        vert_to_edges[static_cast<std::size_t>(e.x)].push_back(eid);
        vert_to_edges[static_cast<std::size_t>(e.y)].push_back(eid);
    }

    // Per-face all-three edges (NOT the load-balanced edge_in_face_h).
    // The cuda-cloth `Prt_ee` constructor uses `face2edge` here so it
    // sees every face's full set of 3 edges, regardless of which face
    // "owns" each shared edge for the EF narrow-phase.

    std::vector<math::Vec4i> adj_h;
    adj_h.reserve(static_cast<std::size_t>(n_faces_) * 3);
    for (int fid = 0; fid < n_faces_; ++fid) {
        const math::Vec3i tri = tris[static_cast<std::size_t>(fid)];
        const math::Vec3i f2e = face2edge[static_cast<std::size_t>(fid)];
        for (int j = 0; j < 3; ++j) {
            const int eid = f2e.data[j];
            if (eid < 0) continue;
            const math::Vec2i e = edges_h[static_cast<std::size_t>(eid)];

            // other_point = the vertex of `tri` not on edge `eid`.
            int other_point = -1;
            for (int k = 0; k < 3; ++k) {
                const int v = tri.data[k];
                if (v != e.x && v != e.y) { other_point = v; break; }
            }
            if (other_point < 0) continue;

            for (int nbr : vert_to_edges[static_cast<std::size_t>(other_point)]) {
                if (nbr == f2e.x || nbr == f2e.y || nbr == f2e.z) continue;
                if (nbr <= eid) continue;
                const math::Vec2i en = edges_h[static_cast<std::size_t>(nbr)];
                adj_h.push_back(math::Vec4i(e.x, e.y, en.x, en.y));
            }
        }
    }
    n_adj_ee_ = static_cast<int>(adj_h.size());

    // ---- 4. Ship to device.

    faces_.resize(static_cast<std::size_t>(n_faces_));
    edges_.resize(static_cast<std::size_t>(n_edges_));
    edge2face_.resize(static_cast<std::size_t>(n_edges_));
    vert_in_edge_.resize(static_cast<std::size_t>(n_edges_));
    edge_in_face_.resize(static_cast<std::size_t>(n_faces_));
    adj_ee_pairs_.resize(static_cast<std::size_t>(std::max(n_adj_ee_, 1)));

    std::copy(tris.begin(),            tris.end(),            faces_.cpu_data());
    std::copy(edges_h.begin(),         edges_h.end(),         edges_.cpu_data());
    std::copy(edge2face_h.begin(),     edge2face_h.end(),     edge2face_.cpu_data());
    std::copy(vert_in_edge_h.begin(),  vert_in_edge_h.end(),  vert_in_edge_.cpu_data());
    std::copy(edge_in_face_h.begin(),  edge_in_face_h.end(),  edge_in_face_.cpu_data());
    if (n_adj_ee_ > 0) {
        std::copy(adj_h.begin(), adj_h.end(), adj_ee_pairs_.cpu_data());
    }

    faces_.copy_to_device();
    edges_.copy_to_device();
    edge2face_.copy_to_device();
    vert_in_edge_.copy_to_device();
    edge_in_face_.copy_to_device();
    if (n_adj_ee_ > 0) {
        adj_ee_pairs_.copy_to_device();
    }
}

}  // namespace collision
}  // namespace chysx
