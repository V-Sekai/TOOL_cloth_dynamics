/*
 * MIT License
 *
 * Copyright (c) 2024-present K. S. Ernest (Fire) Lee
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef OMEGAENGINE_VEC3F_H
#define OMEGAENGINE_VEC3F_H

#include "Macros.h"
#include <cmath>

// CHI-11 subtask 2: in-house Vec3f, replaces Eigen::Vector3d in
// Particle for the per-vertex state that doesn't need fp64 precision.
// Implicit conversions from Eigen::Vector3d (and 3-row segments of
// VecXd) keep the existing call sites compiling during the migration.
struct Vec3f {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;

	Vec3f() = default;
	Vec3f(float a, float b, float c) : x(a), y(b), z(c) {}

	// Eigen interop. Implicit on the read side so `Vec3f v = some_segment`
	// just works; on the write side use `eigen_segment = v.toVec3d()`.
	Vec3f(const Eigen::Vector3d &v)
	    : x(float(v[0])), y(float(v[1])), z(float(v[2])) {}
	Vec3f(const Eigen::Vector3f &v) : x(v[0]), y(v[1]), z(v[2]) {}

	template <typename Derived>
	Vec3f(const Eigen::MatrixBase<Derived> &v)
	    : x(float(v[0])), y(float(v[1])), z(float(v[2])) {
		static_assert(int(Derived::SizeAtCompileTime) == 3 ||
		                int(Derived::SizeAtCompileTime) == Eigen::Dynamic,
		    "Vec3f source must be a 3-vector");
	}

	Eigen::Vector3d toVec3d() const {
		return Eigen::Vector3d(double(x), double(y), double(z));
	}
	Eigen::Vector3f toVec3f() const { return Eigen::Vector3f(x, y, z); }
	// Implicit conversion to Eigen::Vector3d: enables `segment(...) = p.pos`
	// and `Vec3d v = p.pos - other` to compile during the transition. Cost
	// is a float -> double promotion, which is exact.
	operator Eigen::Vector3d() const { return toVec3d(); }

	float &operator[](int i) { return (&x)[i]; }
	const float &operator[](int i) const { return (&x)[i]; }

	Vec3f &operator+=(const Vec3f &o) { x += o.x; y += o.y; z += o.z; return *this; }
	Vec3f &operator-=(const Vec3f &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
	Vec3f &operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
	Vec3f &operator/=(float s) { x /= s; y /= s; z /= s; return *this; }

	Vec3f operator+(const Vec3f &o) const { return {x + o.x, y + o.y, z + o.z}; }
	Vec3f operator-(const Vec3f &o) const { return {x - o.x, y - o.y, z - o.z}; }
	Vec3f operator-() const { return {-x, -y, -z}; }
	Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
	Vec3f operator/(float s) const { return {x / s, y / s, z / s}; }

	float norm() const { return std::sqrt(x * x + y * y + z * z); }
	float squaredNorm() const { return x * x + y * y + z * z; }
	float dot(const Vec3f &o) const { return x * o.x + y * o.y + z * o.z; }
	Vec3f cross(const Vec3f &o) const {
		return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
	}
	Vec3f normalized() const {
		const float n = norm();
		return n > 0.0f ? Vec3f{x / n, y / n, z / n} : Vec3f{};
	}
	// In-place normalize, matching Eigen::Vector3d::normalize().
	void normalize() {
		const float n = norm();
		if (n > 0.0f) { x /= n; y /= n; z /= n; }
	}
	void setZero() { x = 0.0f; y = 0.0f; z = 0.0f; }
	bool hasNaN() const {
		return std::isnan(x) || std::isnan(y) || std::isnan(z);
	}
};

inline Vec3f operator*(float s, const Vec3f &v) { return v * s; }

#endif // OMEGAENGINE_VEC3F_H
