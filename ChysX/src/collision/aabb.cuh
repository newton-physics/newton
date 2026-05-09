// SPDX-License-Identifier: Apache-2.0
//
// Axis-aligned bounding box used by the chysx broad-phase BVH and by
// the per-primitive AABB kernels that feed it.  Header-only POD.
//
// Layout matches cuda-cloth's `aabb` so the BVH refit / overlap logic
// is a 1:1 port.  Methods are all `__host__ __device__` so the same
// type can be used to seed the build on the host side.

#pragma once

#include "../math/common.cuh"
#include "../math/vec.cuh"

namespace chysx {
namespace collision {

struct alignas(16) Aabb {
    math::Vec3f mn;
    math::Vec3f mx;

    CHYSX_HD Aabb()
        : mn(1e30f, 1e30f, 1e30f), mx(-1e30f, -1e30f, -1e30f) {}

    CHYSX_HD Aabb(const math::Vec3f& a, const math::Vec3f& b) {
        set(a, b);
    }

    CHYSX_HDI void set(const math::Vec3f& a, const math::Vec3f& b) {
        mn = math::Vec3f(math::min(a.x, b.x), math::min(a.y, b.y),
                         math::min(a.z, b.z));
        mx = math::Vec3f(math::max(a.x, b.x), math::max(a.y, b.y),
                         math::max(a.z, b.z));
    }

    CHYSX_HDI void add(const math::Vec3f& p) {
        mn = math::Vec3f(math::min(mn.x, p.x), math::min(mn.y, p.y),
                         math::min(mn.z, p.z));
        mx = math::Vec3f(math::max(mx.x, p.x), math::max(mx.y, p.y),
                         math::max(mx.z, p.z));
    }

    CHYSX_HDI void add(const Aabb& b) {
        mn = math::Vec3f(math::min(mn.x, b.mn.x), math::min(mn.y, b.mn.y),
                         math::min(mn.z, b.mn.z));
        mx = math::Vec3f(math::max(mx.x, b.mx.x), math::max(mx.y, b.mx.y),
                         math::max(mx.z, b.mx.z));
    }

    // Inflate isotropically by `eps` along every axis.  Used so the
    // BVH overlap test handles a contact margin without the consumer
    // having to redo it per leaf.
    CHYSX_HDI void enlarge(float eps) {
        mn.x -= eps; mn.y -= eps; mn.z -= eps;
        mx.x += eps; mx.y += eps; mx.z += eps;
    }

    CHYSX_HDI math::Vec3f center() const {
        return math::Vec3f(0.5f * (mn.x + mx.x),
                           0.5f * (mn.y + mx.y),
                           0.5f * (mn.z + mx.z));
    }

    CHYSX_HDI bool overlaps(const Aabb& b) const {
        return !(mx.x < b.mn.x || mn.x > b.mx.x ||
                 mx.y < b.mn.y || mn.y > b.mx.y ||
                 mx.z < b.mn.z || mn.z > b.mx.z);
    }

    CHYSX_HDI bool overlaps(const math::Vec3f& p) const {
        return !(p.x < mn.x || p.x > mx.x ||
                 p.y < mn.y || p.y > mx.y ||
                 p.z < mn.z || p.z > mx.z);
    }
};

}  // namespace collision
}  // namespace chysx
