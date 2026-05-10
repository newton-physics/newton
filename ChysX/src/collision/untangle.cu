// SPDX-License-Identifier: Apache-2.0
//
// CUDA implementation of chysx::collision::UntangleDetector.
//
// Algorithm: ICM (Intersection Contour Minimization), Volino &
// Magnenat-Thalmann, "Resolving Surface Collisions through
// Intersection Contour Minimization", SIGGRAPH 2006.
//
// We follow the reference implementation in `newton/_src/solvers/
// style3d/collision/kernels.py::solve_untangling_kernel` rather than
// cuda-cloth's `kernel_cull_EF_pairs`.  Style3D's version differs
// from cuda-cloth's in three significant ways, all of which we adopt:
//
//   1. **Signed-distance ray-tri test**.  Instead of three Möller
//      cross-product sign tests, check that the edge endpoints lie
//      on opposite sides of the face plane (`d1 * d2 < 0`) and
//      compute the hit point by linear interpolation along the
//      face normal: `hit = (v0 * |d2| + v1 * |d1|) / (|d1| + |d2|)`.
//      Numerically more stable for nearly-parallel edges, and lets
//      us reuse the unsigned distances `(|d1|, |d2|)` directly as
//      the edge-side barycentric weights.
//
//   2. **Adjacent face normals computed per-edge in-kernel from the
//      opposite vertex**, instead of from a precomputed
//      face-normal cache.  Saves one full kernel launch per
//      detection pass and removes a per-face memory buffer.
//
//   3. **G accumulated then normalised**, instead of averaged.
//      cuda-cloth's `G = (G_part1 + G_part2) / 2` keeps the
//      magnitude of a single-face contribution, so when the two
//      adjacent face contributions partially cancel (which happens
//      on dense pre-tangled meshes) the magnitude shrinks below
//      the per-tangle restoring scale and the contact never opens
//      up.  Style3D's `G = normalize(G_part1 + G_part2)` keeps
//      direction information from both faces but always commits a
//      full `disp = 2 * thickness` displacement target -- this is
//      what the apply kernel multiplies through in the force.
//
// In addition we keep cuda-cloth's per-tangle barycentric tolerance
// of `1e-2` (style3d's value) to reject grazing hits at face
// boundaries -- the singular `R = N x N_adj -> 0` case otherwise
// produces NaN gradients that the BVH happily catches every step.

#include "untangle.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace chysx {
namespace collision {

namespace {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("chysx::collision::UntangleDetector: ") + what +
            " failed: " + cudaGetErrorString(err));
    }
}

constexpr int kBlockDim = 128;
inline int grid_for(int n) { return (n + kBlockDim - 1) / kBlockDim; }

// One-element zero-write.  Used to clear the int counter on a
// stream so the launch captures into a CUDA graph cleanly.
__global__ void clear_int_kernel(int* p) { *p = 0; }

// Triangle normal of (a, b, c), oriented by right-hand rule on the
// (b - a) x (c - a) cross.  Returns (0, 0, 0) for degenerate
// triangles so downstream `length(R) < eps` checks short-circuit.
__device__ __forceinline__ math::Vec3f tri_normal(
    const math::Vec3f& a, const math::Vec3f& b, const math::Vec3f& c) {
    const math::Vec3f n = cross(b - a, c - a);
    const float l2 = dot(n, n);
    return (l2 > 1.0e-24f) ? n * (1.0f / sqrtf(l2))
                            : math::Vec3f(0.0f, 0.0f, 0.0f);
}

// Barycentric coordinates of `p` w.r.t. triangle (a, b, c), with
// `(u, v, w)` summing to 1 such that `p = u*a + v*b + w*c`.  Uses
// the "two-vector projection" form (Ericson §3.4) which is the
// same as `triangle_barycentric` in style3d's kernels.py.
__device__ __forceinline__ math::Vec3f tri_barycentric(
    const math::Vec3f& a, const math::Vec3f& b, const math::Vec3f& c,
    const math::Vec3f& p) {
    const math::Vec3f v0 = a - c;
    const math::Vec3f v1 = b - c;
    const math::Vec3f v2 = p - c;
    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d02 = dot(v0, v2);
    const float d11 = dot(v1, v1);
    const float d12 = dot(v1, v2);
    const float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) < 1.0e-24f) return math::Vec3f(0.0f, 0.0f, 0.0f);
    const float inv = 1.0f / denom;
    const float u = (d11 * d02 - d01 * d12) * inv;
    const float v = (d00 * d12 - d01 * d02) * inv;
    return math::Vec3f(u, v, 1.0f - u - v);
}

// Volino's intersection-gradient vector:
//
//     G(R, E, N)  =  R - 2 * N * (E . R) / (E . N)
//
// Direction along which moving the edge endpoints (and oppositely
// moving the face vertices) reduces the intersection contour
// length.  Singular when E is parallel to N (edge in face plane);
// in that degenerate case we just return R unchanged so the
// downstream `length(G)` check rejects the contact.
__device__ __forceinline__ math::Vec3f intersection_gradient(
    const math::Vec3f& R, const math::Vec3f& E, const math::Vec3f& N) {
    const float dot_EN = dot(E, N);
    if (fabsf(dot_EN) > 1.0e-6f) {
        return R - (2.0f * dot(E, R) / dot_EN) * N;
    }
    return R;
}

// Find the vertex of triangle `f` that is NOT one of (e0, e1).
// Returns -1 when the adjacency table is malformed (would mean
// `f` does not actually share the edge (e0, e1) -- guard against
// data corruption rather than crash).
__device__ __forceinline__ int opposite_vertex(const math::Vec3i& f,
                                                int e0, int e1) {
    if (f.x != e0 && f.x != e1) return f.x;
    if (f.y != e0 && f.y != e1) return f.y;
    if (f.z != e0 && f.z != e1) return f.z;
    return -1;
}

// One thread per EF candidate.  Reads (eid, fid), tests whether the
// edge actually pierces the face, and emits a 5-vertex untangle
// contact when it does.
//
// Ordering inside the output stream (matches `UntangleConstraint`'s
// edge-positive / face-negative gradient sign convention):
//
//     pairs5  [5*c + 0..4] = (e0, e1, f0, f1, f2)
//     weights5[5*c + 0..4] = (edge_w0, edge_w1, face_w0, face_w1, face_w2)
//
// All five weights are NON-negative; the opposite force directions
// for the edge and face vertices are applied in the constraint
// kernel by hard-coding the sign flip on indices [2..4].
__global__ void cull_ef_pairs_kernel(
    const math::Vec3f* __restrict__ positions,
    const math::Vec2i* __restrict__ edges,
    const math::Vec3i* __restrict__ faces,
    const math::Vec2i* __restrict__ edge2face,
    float                            thickness,
    const math::Vec2i* __restrict__ ef_pairs,
    const int*         __restrict__ ef_count_dev,
    int                              ef_max,
    int                              max_contacts,
    int*               __restrict__ out_count,
    int*               __restrict__ out_pairs5,
    float*             __restrict__ out_weights5,
    math::Vec3f*       __restrict__ out_normals,
    float*             __restrict__ out_depths) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_raw = *ef_count_dev;
    const int n = (n_raw < ef_max) ? n_raw : ef_max;
    if (idx >= n) return;

    const math::Vec2i ef = ef_pairs[idx];
    const int eid = ef.x;
    const int fid = ef.y;

    // ---- edge data + adjacent-face normals (per-edge, computed once) -
    const math::Vec2i e = edges[eid];
    const math::Vec3f v0 = positions[e.x];
    const math::Vec3f v1 = positions[e.y];

    const float E_len2 = dot(v0 - v1, v0 - v1);
    if (!(E_len2 > 2.5e-7f)) return;  // |E| < 5e-4: degenerate edge
    const math::Vec3f E = (v0 - v1) * (1.0f / sqrtf(E_len2));

    const math::Vec2i adj = edge2face[eid];
    int opp_a = -1;
    int opp_b = -1;
    math::Vec3f Na(0.0f, 0.0f, 0.0f);
    math::Vec3f Nb(0.0f, 0.0f, 0.0f);
    if (adj.x >= 0) {
        opp_a = opposite_vertex(faces[adj.x], e.x, e.y);
        if (opp_a >= 0) Na = tri_normal(v0, v1, positions[opp_a]);
    }
    if (adj.y >= 0) {
        opp_b = opposite_vertex(faces[adj.y], e.x, e.y);
        if (opp_b >= 0) Nb = tri_normal(v0, v1, positions[opp_b]);
    }

    // ---- intersected face --------------------------------------------
    const math::Vec3i f = faces[fid];

    // Covertex filter: any face vertex shared with the edge -> skip.
    // (BVH query already does this for the *sorted* leaf payload but
    // cheap to repeat against the raw face id we got here.)
    if (f.x == e.x || f.x == e.y) return;
    if (f.y == e.x || f.y == e.y) return;
    if (f.z == e.x || f.z == e.y) return;

    const math::Vec3f a = positions[f.x];
    const math::Vec3f b = positions[f.y];
    const math::Vec3f c = positions[f.z];

    const math::Vec3f Nf_raw = cross(b - a, c - a);
    const float Nf_len2 = dot(Nf_raw, Nf_raw);
    if (!(Nf_len2 > 1.0e-16f)) return;  // degenerate face
    const math::Vec3f Nf = Nf_raw * (1.0f / sqrtf(Nf_len2));

    // Signed-distance ray-tri intersection: the edge actually
    // pierces the face iff its two endpoints sit on opposite
    // sides of the face plane.
    const float d1_signed = dot(Nf, v0 - a);
    const float d2_signed = dot(Nf, v1 - a);
    if (d1_signed * d2_signed >= 0.0f) return;

    const float ad1 = fabsf(d1_signed);
    const float ad2 = fabsf(d2_signed);
    const float dsum = ad1 + ad2;
    if (!(dsum > 1.0e-12f)) return;

    const math::Vec3f hit = (v0 * ad2 + v1 * ad1) * (1.0f / dsum);

    // Face barycentric of the hit point.  Reject grazing hits
    // (any bary < 1e-2) to avoid the singular `R = N_f x N_adj -> 0`
    // case at face boundaries.  Same threshold as style3d.
    const math::Vec3f bary = tri_barycentric(a, b, c, hit);
    constexpr float kBaryEps = 1.0e-2f;
    if (bary.x < kBaryEps || bary.y < kBaryEps || bary.z < kBaryEps) return;

    // ---- intersection-gradient direction G ---------------------------
    //
    // Sum the contributions from both adjacent faces of the edge
    // (skipping boundary sides where opp_* < 0), then normalise.
    // Using `+=` here (rather than cuda-cloth's average) keeps the
    // direction information from both sides without letting the
    // magnitude shrink to zero when the two contributions partially
    // cancel.  The actual displacement target lives in the apply
    // kernel as a `2 * thickness` constant -- the magnitude of G
    // itself never reaches the RHS.
    math::Vec3f G(0.0f, 0.0f, 0.0f);

    auto accumulate_g = [&](int opp_id, const math::Vec3f& N_adj) {
        if (opp_id < 0) return;
        const math::Vec3f R_raw = cross(Nf, N_adj);
        const float R_l2 = dot(R_raw, R_raw);
        if (!(R_l2 > 1.0e-12f)) return;  // edge ~~ face plane intersection ill-defined
        math::Vec3f R = R_raw * (1.0f / sqrtf(R_l2));

        // Sign-correct R using the cross-product test from style3d:
        // `cross(E, R)` and `cross(E, opp - hit)` should point the
        // same way iff R is on the "body of the adjacent face"
        // side of the intersection contour.
        const math::Vec3f to_opp = positions[opp_id] - hit;
        if (dot(cross(E, R), cross(E, to_opp)) < 0.0f) {
            R = -R;
        }

        G += intersection_gradient(R, E, Nf);
    };

    accumulate_g(opp_a, Na);
    accumulate_g(opp_b, Nb);

    const float G_l2 = dot(G, G);
    if (!(G_l2 > 1.0e-24f)) return;
    G = G * (1.0f / sqrtf(G_l2));

    // ---- emit contact -----------------------------------------------
    //
    // Edge-side weights: signed-distance ratio `(d2, d1) / (d1+d2)`.
    // Equivalent to "fraction of the edge between the hit point and
    // the opposite endpoint" -- when the hit lands on v0 we get
    // (1, 0), on v1 we get (0, 1).  Numerically nicer than the
    // length-ratio form because it shares its denominator with the
    // hit-point computation above.
    const float inv_dsum = 1.0f / dsum;
    const float ew0 = ad2 * inv_dsum;
    const float ew1 = ad1 * inv_dsum;

    const int slot = atomicAdd(out_count, 1);
    if (slot >= max_contacts) return;

    int*   pdst = out_pairs5   + 5 * slot;
    float* wdst = out_weights5 + 5 * slot;
    pdst[0] = e.x;
    pdst[1] = e.y;
    pdst[2] = f.x;
    pdst[3] = f.y;
    pdst[4] = f.z;
    wdst[0] = ew0;
    wdst[1] = ew1;
    wdst[2] = bary.x;
    wdst[3] = bary.y;
    wdst[4] = bary.z;
    out_normals[slot] = G;
    out_depths[slot]  = thickness;
}

}  // namespace

void UntangleDetector::reserve(int max_contacts) {
    if (max_contacts < 1) max_contacts = 1;
    max_contacts_ = max_contacts;
    pairs_.resize(static_cast<std::size_t>(5 * max_contacts));
    weights_.resize(static_cast<std::size_t>(5 * max_contacts));
    normals_.resize(static_cast<std::size_t>(max_contacts));
    depths_.resize(static_cast<std::size_t>(max_contacts));
    if (count_.gpu_size() == 0) count_.resize(1);
}

void UntangleDetector::bind_topology(const MeshTopology* topology) {
    topology_ = topology;
    // No per-face cache to allocate -- adjacent-face normals are
    // computed on the fly inside `cull_ef_pairs_kernel` from the
    // edge's two opposite vertices (see `triangle_normal` call
    // sites there).
}

void UntangleDetector::detect(DeviceSpan<math::Vec3f> positions,
                              const math::Vec2i*      ef_pairs_dev,
                              const int*              ef_count_dev,
                              int                     ef_max,
                              float                   thickness,
                              std::uintptr_t          cuda_stream) {
    if (topology_ == nullptr || !topology_->valid()) return;
    if (max_contacts_ <= 0) return;
    if (ef_pairs_dev == nullptr || ef_count_dev == nullptr || ef_max <= 0)
        return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

    // 1. Zero the contact counter on stream.
    clear_int_kernel<<<1, 1, 0, stream>>>(count_.gpu_data());
    check_cuda(cudaGetLastError(), "clear_int_kernel launch");

    // 2. Cull EF candidates -> 5-vertex tangle contacts.
    //
    // Adjacent-face normals are computed inside the kernel from
    // `edge2face` + `faces` + the current `positions` -- no
    // per-face normal-cache prepass.
    cull_ef_pairs_kernel<<<grid_for(ef_max), kBlockDim, 0, stream>>>(
        positions.data(),
        topology_->edges().gpu_data(),
        topology_->faces().gpu_data(),
        topology_->edge2face().gpu_data(),
        thickness,
        ef_pairs_dev,
        ef_count_dev,
        ef_max,
        max_contacts_,
        count_.gpu_data(),
        pairs_.gpu_data(),
        weights_.gpu_data(),
        normals_.gpu_data(),
        depths_.gpu_data());
    check_cuda(cudaGetLastError(), "cull_ef_pairs_kernel launch");
}

int UntangleDetector::count(std::uintptr_t cuda_stream) {
    if (count_.gpu_size() == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    int host_count = 0;
    check_cuda(cudaMemcpyAsync(&host_count, count_.gpu_data(), sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               "count() memcpy");
    check_cuda(cudaStreamSynchronize(stream), "count() sync");
    if (host_count > max_contacts_) host_count = max_contacts_;
    if (host_count < 0) host_count = 0;
    return host_count;
}

}  // namespace collision
}  // namespace chysx
