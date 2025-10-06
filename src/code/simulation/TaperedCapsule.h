#pragma once

#include "../engine/Macros.h"
#include "Primitive.h"
#include <vector>

namespace tool_cloth_dynamics {

/**
 * @brief Tapered capsule primitive - cylinder with hemispherical caps of different radii
 *
 * Useful for approximating human body parts (torso, limbs) where width varies.
 * Consists of:
 * - Top hemisphere with radius_top
 * - Tapered cylinder in the middle
 * - Bottom hemisphere with radius_bottom
 */
class TaperedCapsule : public Primitive {
public:
	double radius_top; // Radius of top hemisphere
	double radius_bottom; // Radius of bottom hemisphere
	double mid_height; // Height of middle cylinder section
	int radial_segments; // Mesh resolution (circumference)
	int rings; // Vertical resolution

	Vec3d axis; // Orientation axis (default: Y-up)

	// Mesh data for visualization (generated without particles)
	std::vector<Vec3d> vertices; // Mesh vertices in world space
	std::vector<Vec3i> faces; // Triangle faces (vertex indices)
	bool mesh_generated = false; // Flag indicating mesh is ready

	/**
	 * @brief Construct a new Tapered Capsule
	 *
	 * @param center Center point of the capsule
	 * @param r_top Radius at top (wider end, e.g., pelvis)
	 * @param r_bottom Radius at bottom (narrower end, e.g., ankles)
	 * @param height Total height of middle cylinder section
	 * @param orientation Axis direction (normalized)
	 * @param color RGB color for rendering
	 * @param segments Radial segments for mesh (default: 16)
	 * @param rings_count Vertical rings for mesh (default: 8)
	 */
	TaperedCapsule(
			Vec3d center,
			double r_top,
			double r_bottom,
			double height,
			Vec3d orientation = Vec3d(0, 1, 0),
			Vec3d color = Vec3d(0.8, 0.8, 0.8),
			int segments = 16,
			int rings_count = 8);

	~TaperedCapsule() override = default;

	/**
	 * @brief Check if a point is in contact with the tapered capsule
	 *
	 * @param center_prim Primitive center position
	 * @param pos Point position to test
	 * @param velocity Point velocity
	 * @param normal Output: contact normal (points away from surface)
	 * @param dist Output: signed distance (negative = penetration)
	 * @param v_out Output: resolved position (on surface if penetrating)
	 * @return true if in contact (within threshold)
	 */
	bool isInContact(
			const Vec3d &center_prim,
			const Vec3d &pos,
			const Vec3d &velocity,
			Vec3d &normal,
			double &dist,
			Vec3d &v_out) override;

	/**
	 * @brief Get total height including caps
	 */
	double getTotalHeight() const {
		return mid_height + radius_top + radius_bottom;
	}

	/**
	 * @brief Find closest point on tapered capsule surface to given point
	 *
	 * @param point Point in world space
	 * @param closest_point Output: closest point on surface
	 * @param surface_normal Output: normal at closest point
	 * @return Signed distance (negative if inside)
	 */
	double closestPointOnSurface(
			const Vec3d &point,
			Vec3d &closest_point,
			Vec3d &surface_normal) const;

public:
	/**
	 * @brief Generate triangle mesh for visualization
	 * Adapted from Godot's TaperedCapsuleMesh
	 */
	void generateMesh();

	/**
	 * @brief Septic smoothstep function for smooth tapering
	 *
	 * Provides C³ continuous interpolation for natural limb shapes.
	 * septic(t) = 6t⁷ - 35t⁶ + 84t⁵ - 70t⁴ + 20t³
	 *
	 * @param t Parameter in [0,1]
	 * @return Smooth interpolation value
	 */
	static double septicSmoothstep(double t) {
		// Clamp t to [0,1] for safety
		t = std::max(0.0, std::min(1.0, t));

		double t2 = t * t;
		double t3 = t2 * t;
		double t4 = t3 * t;
		double t5 = t4 * t;
		double t6 = t5 * t;
		double t7 = t6 * t;

		return 6.0 * t7 - 35.0 * t6 + 84.0 * t5 - 70.0 * t4 + 20.0 * t3;
	}

	/**
	 * @brief Compute tapered radius at parameter t along capsule axis
	 *
	 * Uses septic smoothstep for C³ continuous tapering.
	 *
	 * @param t Parameter along capsule axis [0,1] (0=top, 1=bottom)
	 * @param radius_start Radius at start (t=0)
	 * @param radius_end Radius at end (t=1)
	 * @return Interpolated radius
	 */
	static double computeTaperedRadius(double t, double radius_start, double radius_end) {
		double smooth_t = septicSmoothstep(t);
		return radius_start * (1.0 - smooth_t) + radius_end * smooth_t;
	}

	/**
	 * @brief Build rotation matrix to transform from source to target direction
	 *
	 * @param from Source direction vector (normalized)
	 * @param to Target direction vector (normalized)
	 * @return 3x3 rotation matrix
	 */
	Mat3x3d buildRotationMatrix(const Vec3d &from, const Vec3d &to) const;

	/**
	 * @brief Export tapered capsule mesh to OBJ file
	 *
	 * Generates mesh if not already generated, then writes to OBJ format.
	 *
	 * @param filename Output OBJ file path
	 * @return true if export successful
	 */
	bool exportToOBJ(const std::string &filename) const;
};

} // namespace tool_cloth_dynamics
