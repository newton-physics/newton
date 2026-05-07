// SPDX-License-Identifier: Apache-2.0
//
// chysx::geometry::TriangleMesh<T>
//
// Minimal triangle mesh container.  Stores three CUDA-aware buffers:
//
//   * vertices   : per-vertex positions (Vec3<T>)
//   * triangles  : per-face vertex indices (Vec3i)
//   * edges      : per-edge vertex indices (Vec2i), populated lazily
//                  by build_edges()
//
// Each buffer is a chysx::CudaArray<...>, so callers can mutate the host
// side, then call `copy_to_device()` to push the data to the GPU and
// hand `gpu_data() / gpu_ptr()` to a CUDA kernel — same zero-copy
// pattern the rest of ChysX uses.
//
// build_edges() runs on the host (mesh edges are typically built once
// at load time, not per step).  It produces a deduplicated, sorted
// list of undirected edges; the device side is left untouched, so call
// `mesh.edges().copy_to_device()` afterwards if a kernel needs them.

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../memory/cuda_array.h"
#include "../math/vec.cuh"

namespace chysx {
namespace geometry {

template <typename T = float>
class TriangleMesh {
public:
    using Scalar = T;
    using Vertex = math::Vec3<T>;
    using Triangle = math::Vec3i;  // (i, j, k) — indices into vertices
    using Edge = math::Vec2i;      // (i, j) with i < j

    TriangleMesh() = default;

    TriangleMesh(std::size_t vertex_count, std::size_t triangle_count) {
        resize_vertices(vertex_count);
        resize_triangles(triangle_count);
    }

    // Move-only, like the underlying CudaArray.
    TriangleMesh(const TriangleMesh&) = delete;
    TriangleMesh& operator=(const TriangleMesh&) = delete;
    TriangleMesh(TriangleMesh&&) noexcept = default;
    TriangleMesh& operator=(TriangleMesh&&) noexcept = default;

    // ---- sizing --------------------------------------------------------

    // Allocate `n` vertices on both host and device.
    void resize_vertices(std::size_t n) { vertices_.resize(n); }
    // Allocate `n` triangles on both host and device.
    void resize_triangles(std::size_t n) { triangles_.resize(n); }

    // Free everything (host + device).
    void clear() noexcept {
        vertices_.clear();
        triangles_.clear();
        edges_.clear();
    }

    // ---- edge extraction ----------------------------------------------

    // Build the deduplicated undirected edge list from `triangles_`'s
    // host side and store it in `edges_`'s host side.
    //
    // Each edge is stored as `(min, max)` of its two vertex indices, and
    // the list is sorted lexicographically — useful both for
    // deterministic output and for downstream binary-search lookups.
    //
    // Triangles must already be present on the host (call
    // `triangles().copy_to_host()` first if they only live on the GPU).
    // The result is *not* automatically uploaded; do
    // `mesh.edges().copy_to_device()` if a kernel needs them.
    void build_edges() {
        const std::size_t n_tri = triangles_.cpu_size();
        if (n_tri == 0) {
            edges_.clear();
            return;
        }
        if (triangles_.cpu_data() == nullptr) {
            throw std::runtime_error(
                "chysx::geometry::TriangleMesh::build_edges: triangles must "
                "be allocated on the host (call triangles().copy_to_host() "
                "first if they only live on the GPU).");
        }

        // Collect every directed edge as a sorted (lo, hi) pair, then
        // dedupe.  We use std::vector<std::pair<int,int>> because pair
        // already gives lexicographic ordering and equality.
        std::vector<std::pair<int, int>> tmp;
        tmp.reserve(n_tri * 3);

        const Triangle* tris = triangles_.cpu_data();
        for (std::size_t t = 0; t < n_tri; ++t) {
            const Triangle& tri = tris[t];
            const int a = tri.x;
            const int b = tri.y;
            const int c = tri.z;
            tmp.emplace_back(std::min(a, b), std::max(a, b));
            tmp.emplace_back(std::min(b, c), std::max(b, c));
            tmp.emplace_back(std::min(c, a), std::max(c, a));
        }

        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());

        // Materialise into CudaArray.  We keep the device side at zero
        // for now; the caller decides whether to upload.
        edges_.allocate_host(tmp.size());
        edges_.allocate_device(tmp.size());

        Edge* out = edges_.cpu_data();
        for (std::size_t i = 0; i < tmp.size(); ++i) {
            out[i] = Edge(tmp[i].first, tmp[i].second);
        }
    }

    // ---- buffer accessors ---------------------------------------------

    CudaArray<Vertex>&         vertices()        noexcept { return vertices_; }
    const CudaArray<Vertex>&   vertices() const  noexcept { return vertices_; }
    CudaArray<Triangle>&       triangles()       noexcept { return triangles_; }
    const CudaArray<Triangle>& triangles() const noexcept { return triangles_; }
    CudaArray<Edge>&           edges()           noexcept { return edges_; }
    const CudaArray<Edge>&     edges() const     noexcept { return edges_; }

    // ---- size helpers --------------------------------------------------

    std::size_t num_vertices()  const noexcept { return vertices_.cpu_size(); }
    std::size_t num_triangles() const noexcept { return triangles_.cpu_size(); }
    std::size_t num_edges()     const noexcept { return edges_.cpu_size(); }

private:
    CudaArray<Vertex> vertices_;
    CudaArray<Triangle> triangles_;
    CudaArray<Edge> edges_;
};

// Convenience aliases.
using TriangleMeshf = TriangleMesh<float>;
using TriangleMeshd = TriangleMesh<double>;

}  // namespace geometry
}  // namespace chysx
