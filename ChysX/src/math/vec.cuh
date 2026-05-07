// SPDX-License-Identifier: Apache-2.0
//
// Tiny fixed-size numeric vectors usable from both host and device code.
//
// Conventions
// -----------
// * Each vector type is a templated POD with a `union { struct { T x, y, ...; };
//   T data[N]; };` so callers can pick named fields or array indexing.
// * Mutating operators (`+=`, `-=`, `*=`, `/=`, unary `-`) live on the
//   class.  Pure binary operators and queries (`+`, `-`, `*`, `dot`,
//   `cross`, `length`, ...) are non-members so that conversions on the
//   left-hand side work the same as on the right.
// * `operator*(Vec, Vec)` is the Hadamard (component-wise) product, to
//   match common GPU math libraries.  Inner products go through `dot`.
// * No operator returns a reference to the source: every binary op
//   produces a fresh value, which keeps semantics predictable in
//   templated contexts.

#pragma once

#include "common.cuh"

namespace chysx {
namespace math {

// ============================================================================
// Vec2
// ============================================================================

template <typename T>
struct Vec2 {
    union {
        struct {
            T x, y;
        };
        T data[2];
    };

    CHYSX_HD Vec2() : x(T{}), y(T{}) {}
    CHYSX_HD Vec2(T x_, T y_) : x(x_), y(y_) {}
    CHYSX_HD explicit Vec2(T s) : x(s), y(s) {}

    CHYSX_HD T& operator[](int i) { return data[i]; }
    CHYSX_HD const T& operator[](int i) const { return data[i]; }

    CHYSX_HD Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    CHYSX_HD Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    CHYSX_HD Vec2& operator*=(const Vec2& o) { x *= o.x; y *= o.y; return *this; }
    CHYSX_HD Vec2& operator*=(T s)           { x *= s;   y *= s;   return *this; }
    CHYSX_HD Vec2& operator/=(const Vec2& o) { x /= o.x; y /= o.y; return *this; }
    CHYSX_HD Vec2& operator/=(T s)           { x /= s;   y /= s;   return *this; }

    CHYSX_HD Vec2 operator-() const { return Vec2(-x, -y); }
};

template <typename T>
CHYSX_HDI Vec2<T> operator+(Vec2<T> a, const Vec2<T>& b) { return a += b; }
template <typename T>
CHYSX_HDI Vec2<T> operator-(Vec2<T> a, const Vec2<T>& b) { return a -= b; }
template <typename T>
CHYSX_HDI Vec2<T> operator*(Vec2<T> a, const Vec2<T>& b) { return a *= b; }
template <typename T>
CHYSX_HDI Vec2<T> operator*(Vec2<T> a, T s)              { return a *= s; }
template <typename T>
CHYSX_HDI Vec2<T> operator*(T s, Vec2<T> a)              { return a *= s; }
template <typename T>
CHYSX_HDI Vec2<T> operator/(Vec2<T> a, const Vec2<T>& b) { return a /= b; }
template <typename T>
CHYSX_HDI Vec2<T> operator/(Vec2<T> a, T s)              { return a /= s; }

template <typename T>
CHYSX_HDI bool operator==(const Vec2<T>& a, const Vec2<T>& b) {
    return a.x == b.x && a.y == b.y;
}
template <typename T>
CHYSX_HDI bool operator!=(const Vec2<T>& a, const Vec2<T>& b) { return !(a == b); }

template <typename T>
CHYSX_HDI T dot(const Vec2<T>& a, const Vec2<T>& b) {
    return a.x * b.x + a.y * b.y;
}
template <typename T>
CHYSX_HDI T length_sqr(const Vec2<T>& a) { return dot(a, a); }
template <typename T>
CHYSX_HDI T length(const Vec2<T>& a) {
    using std::sqrt;
    return sqrt(length_sqr(a));
}
template <typename T>
CHYSX_HDI Vec2<T> normalize(const Vec2<T>& a) { return a * (T{1} / length(a)); }

template <typename T>
CHYSX_HDI Vec2<T> min(const Vec2<T>& a, const Vec2<T>& b) {
    return Vec2<T>(min(a.x, b.x), min(a.y, b.y));
}
template <typename T>
CHYSX_HDI Vec2<T> max(const Vec2<T>& a, const Vec2<T>& b) {
    return Vec2<T>(max(a.x, b.x), max(a.y, b.y));
}

// ============================================================================
// Vec3
// ============================================================================

template <typename T>
struct Vec3 {
    union {
        struct {
            T x, y, z;
        };
        T data[3];
    };

    CHYSX_HD Vec3() : x(T{}), y(T{}), z(T{}) {}
    CHYSX_HD Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    CHYSX_HD explicit Vec3(T s) : x(s), y(s), z(s) {}

    CHYSX_HD T& operator[](int i) { return data[i]; }
    CHYSX_HD const T& operator[](int i) const { return data[i]; }

    CHYSX_HD Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    CHYSX_HD Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    CHYSX_HD Vec3& operator*=(const Vec3& o) { x *= o.x; y *= o.y; z *= o.z; return *this; }
    CHYSX_HD Vec3& operator*=(T s)           { x *= s;   y *= s;   z *= s;   return *this; }
    CHYSX_HD Vec3& operator/=(const Vec3& o) { x /= o.x; y /= o.y; z /= o.z; return *this; }
    CHYSX_HD Vec3& operator/=(T s)           { x /= s;   y /= s;   z /= s;   return *this; }

    CHYSX_HD Vec3 operator-() const { return Vec3(-x, -y, -z); }
};

template <typename T>
CHYSX_HDI Vec3<T> operator+(Vec3<T> a, const Vec3<T>& b) { return a += b; }
template <typename T>
CHYSX_HDI Vec3<T> operator-(Vec3<T> a, const Vec3<T>& b) { return a -= b; }
template <typename T>
CHYSX_HDI Vec3<T> operator*(Vec3<T> a, const Vec3<T>& b) { return a *= b; }
template <typename T>
CHYSX_HDI Vec3<T> operator*(Vec3<T> a, T s)              { return a *= s; }
template <typename T>
CHYSX_HDI Vec3<T> operator*(T s, Vec3<T> a)              { return a *= s; }
template <typename T>
CHYSX_HDI Vec3<T> operator/(Vec3<T> a, const Vec3<T>& b) { return a /= b; }
template <typename T>
CHYSX_HDI Vec3<T> operator/(Vec3<T> a, T s)              { return a /= s; }

template <typename T>
CHYSX_HDI bool operator==(const Vec3<T>& a, const Vec3<T>& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
template <typename T>
CHYSX_HDI bool operator!=(const Vec3<T>& a, const Vec3<T>& b) { return !(a == b); }

template <typename T>
CHYSX_HDI T dot(const Vec3<T>& a, const Vec3<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <typename T>
CHYSX_HDI Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) {
    return Vec3<T>(a.y * b.z - a.z * b.y,
                   a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x);
}
template <typename T>
CHYSX_HDI T length_sqr(const Vec3<T>& a) { return dot(a, a); }
template <typename T>
CHYSX_HDI T length(const Vec3<T>& a) {
    using std::sqrt;
    return sqrt(length_sqr(a));
}
template <typename T>
CHYSX_HDI Vec3<T> normalize(const Vec3<T>& a) { return a * (T{1} / length(a)); }

template <typename T>
CHYSX_HDI Vec3<T> min(const Vec3<T>& a, const Vec3<T>& b) {
    return Vec3<T>(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
template <typename T>
CHYSX_HDI Vec3<T> max(const Vec3<T>& a, const Vec3<T>& b) {
    return Vec3<T>(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

// ============================================================================
// Vec4
// ============================================================================

template <typename T>
struct Vec4 {
    union {
        struct {
            T x, y, z, w;
        };
        T data[4];
    };

    CHYSX_HD Vec4() : x(T{}), y(T{}), z(T{}), w(T{}) {}
    CHYSX_HD Vec4(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}
    CHYSX_HD explicit Vec4(T s) : x(s), y(s), z(s), w(s) {}

    CHYSX_HD T& operator[](int i) { return data[i]; }
    CHYSX_HD const T& operator[](int i) const { return data[i]; }

    CHYSX_HD Vec4& operator+=(const Vec4& o) {
        x += o.x; y += o.y; z += o.z; w += o.w;
        return *this;
    }
    CHYSX_HD Vec4& operator-=(const Vec4& o) {
        x -= o.x; y -= o.y; z -= o.z; w -= o.w;
        return *this;
    }
    CHYSX_HD Vec4& operator*=(const Vec4& o) {
        x *= o.x; y *= o.y; z *= o.z; w *= o.w;
        return *this;
    }
    CHYSX_HD Vec4& operator*=(T s) {
        x *= s; y *= s; z *= s; w *= s;
        return *this;
    }
    CHYSX_HD Vec4& operator/=(const Vec4& o) {
        x /= o.x; y /= o.y; z /= o.z; w /= o.w;
        return *this;
    }
    CHYSX_HD Vec4& operator/=(T s) {
        x /= s; y /= s; z /= s; w /= s;
        return *this;
    }

    CHYSX_HD Vec4 operator-() const { return Vec4(-x, -y, -z, -w); }
};

template <typename T>
CHYSX_HDI Vec4<T> operator+(Vec4<T> a, const Vec4<T>& b) { return a += b; }
template <typename T>
CHYSX_HDI Vec4<T> operator-(Vec4<T> a, const Vec4<T>& b) { return a -= b; }
template <typename T>
CHYSX_HDI Vec4<T> operator*(Vec4<T> a, const Vec4<T>& b) { return a *= b; }
template <typename T>
CHYSX_HDI Vec4<T> operator*(Vec4<T> a, T s)              { return a *= s; }
template <typename T>
CHYSX_HDI Vec4<T> operator*(T s, Vec4<T> a)              { return a *= s; }
template <typename T>
CHYSX_HDI Vec4<T> operator/(Vec4<T> a, const Vec4<T>& b) { return a /= b; }
template <typename T>
CHYSX_HDI Vec4<T> operator/(Vec4<T> a, T s)              { return a /= s; }

template <typename T>
CHYSX_HDI bool operator==(const Vec4<T>& a, const Vec4<T>& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
template <typename T>
CHYSX_HDI bool operator!=(const Vec4<T>& a, const Vec4<T>& b) { return !(a == b); }

template <typename T>
CHYSX_HDI T dot(const Vec4<T>& a, const Vec4<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
template <typename T>
CHYSX_HDI T length_sqr(const Vec4<T>& a) { return dot(a, a); }
template <typename T>
CHYSX_HDI T length(const Vec4<T>& a) {
    using std::sqrt;
    return sqrt(length_sqr(a));
}
template <typename T>
CHYSX_HDI Vec4<T> normalize(const Vec4<T>& a) { return a * (T{1} / length(a)); }

// ============================================================================
// Type aliases
// ============================================================================

using Vec2f = Vec2<float>;
using Vec3f = Vec3<float>;
using Vec4f = Vec4<float>;

using Vec2d = Vec2<double>;
using Vec3d = Vec3<double>;
using Vec4d = Vec4<double>;

using Vec2i = Vec2<int>;
using Vec3i = Vec3<int>;
using Vec4i = Vec4<int>;

}  // namespace math
}  // namespace chysx
