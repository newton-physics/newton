// SPDX-License-Identifier: Apache-2.0
//
// 64-bit 3D Morton encoding used to seed the LBVH build.  The encoder
// takes coordinates already mapped into the unit cube [0, 1]^3 and
// returns a 63-bit interleaved integer (21 bits per axis), packed into
// the high half of a uint64.  The low 32 bits get the primitive id so
// that the radix-sort produces a stable ordering even when many
// primitives share the same coarse Morton code.

#pragma once

#include <cstdint>

#include "../math/common.cuh"

namespace chysx {
namespace collision {

// Spread the lower 21 bits of `v` so each input bit lands in slots
// 0, 3, 6, ...  (Three-fold "magic-bits" interleave.)
CHYSX_HDI std::uint64_t morton_expand21(std::uint64_t v) {
    v &= 0x1fffffull;                              // 21 bits
    v = (v | (v << 32)) & 0x001f00000000ffffull;
    v = (v | (v << 16)) & 0x001f0000ff0000ffull;
    v = (v | (v << 8))  & 0x100f00f00f00f00full;
    v = (v | (v << 4))  & 0x10c30c30c30c30c3ull;
    v = (v | (v << 2))  & 0x1249249249249249ull;
    return v;
}

// 3D Morton code from coordinates in [0, 1].  Out-of-range inputs are
// clamped: the BVH still works for primitives that briefly leave the
// scene bbox, just at lower spatial-hash quality.
CHYSX_HDI std::uint64_t morton3d(float fx, float fy, float fz) {
    const float scale = 2097152.0f;  // 2^21
    fx = fx < 0.0f ? 0.0f : (fx > 1.0f ? 1.0f : fx);
    fy = fy < 0.0f ? 0.0f : (fy > 1.0f ? 1.0f : fy);
    fz = fz < 0.0f ? 0.0f : (fz > 1.0f ? 1.0f : fz);
    const std::uint64_t ix = static_cast<std::uint64_t>(fx * scale);
    const std::uint64_t iy = static_cast<std::uint64_t>(fy * scale);
    const std::uint64_t iz = static_cast<std::uint64_t>(fz * scale);
    return (morton_expand21(ix) << 2) |
           (morton_expand21(iy) << 1) |
            morton_expand21(iz);
}

}  // namespace collision
}  // namespace chysx
