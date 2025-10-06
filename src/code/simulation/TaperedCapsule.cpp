#include "TaperedCapsule.h"
#include "../engine/MeshFileHandler.h"
#include <algorithm>
#include <cmath>

namespace tool_cloth_dynamics {

TaperedCapsule::TaperedCapsule(
		Vec3d center,
		double r_top,
		double r_bottom,
		double height,
		Vec3d orientation,
		Vec3d color,
		int segments,
		int rings_count) :
		Primitive(PrimitiveType::CAPSULE, center, false, color),
		radius_top(r_top),
		radius_bottom(r_bottom),
		mid_height(height),
		radial_segments(std::max(4, segments)),
		rings(std::max(0, rings_count)),
		axis(orientation.normalized()) {
	generateMesh();
}

void TaperedCapsule::generateMesh() {
	// Generate mesh vertices and faces for visualization
	// This creates a tapered capsule mesh without requiring Particle objects

	vertices.clear();
	faces.clear();

	const int segments = radial_segments;
	const int rings_per_cap = std::max(1, rings / 2);

	// Generate vertices
	std::vector<Vec3d> mesh_vertices;
	std::vector<Vec3i> mesh_faces;

	// Bottom hemisphere vertices
	for (int ring = 0; ring <= rings_per_cap; ++ring) {
		double v = static_cast<double>(ring) / rings_per_cap;
		double phi = v * M_PI * 0.5; // 0 to π/2

		double y = -mid_height * 0.5 - radius_bottom * std::cos(phi);
		double ring_radius = radius_bottom * std::sin(phi);

		for (int segment = 0; segment <= segments; ++segment) {
			double u = static_cast<double>(segment) / segments;
			double theta = u * 2.0 * M_PI;

			double x = ring_radius * std::cos(theta);
			double z = ring_radius * std::sin(theta);

			mesh_vertices.emplace_back(x, y, z);
		}
	}

	// Cylinder vertices (tapered)
	int cylinder_rings = std::max(1, rings - 2 * rings_per_cap);
	for (int ring = 0; ring <= cylinder_rings; ++ring) {
		double v = static_cast<double>(ring) / cylinder_rings;
		double y = -mid_height * 0.5 + v * mid_height;

		// Use septic smoothstep for C³ continuous tapering
		double current_radius = computeTaperedRadius(v, radius_bottom, radius_top);

		for (int segment = 0; segment <= segments; ++segment) {
			double u = static_cast<double>(segment) / segments;
			double theta = u * 2.0 * M_PI;

			double x = current_radius * std::cos(theta);
			double z = current_radius * std::sin(theta);

			mesh_vertices.emplace_back(x, y, z);
		}
	}

	// Top hemisphere vertices
	for (int ring = 0; ring <= rings_per_cap; ++ring) {
		double v = static_cast<double>(ring) / rings_per_cap;
		double phi = v * M_PI * 0.5; // 0 to π/2

		double y = mid_height * 0.5 + radius_top * std::sin(phi);
		double ring_radius = radius_top * std::cos(phi);

		for (int segment = 0; segment <= segments; ++segment) {
			double u = static_cast<double>(segment) / segments;
			double theta = u * 2.0 * M_PI;

			double x = ring_radius * std::cos(theta);
			double z = ring_radius * std::sin(theta);

			mesh_vertices.emplace_back(x, y, z);
		}
	}

	// Transform vertices to world space with full 3D rotation support
	vertices.reserve(mesh_vertices.size());

	// Build rotation matrix to transform from Y-up local space to bone orientation
	Mat3x3d rotation_matrix = buildRotationMatrix(Vec3d(0, 1, 0), axis);

	for (const Vec3d &local_vertex : mesh_vertices) {
		// Apply rotation and translation
		Vec3d rotated_vertex = rotation_matrix * local_vertex;
		Vec3d world_vertex = center + rotated_vertex;
		vertices.push_back(world_vertex);
	}

	// Generate faces (simplified - basic quad tessellation)
	int vertices_per_ring = segments + 1;
	int total_rings = (rings_per_cap + 1) + (cylinder_rings + 1) + (rings_per_cap + 1);

	for (int ring = 0; ring < total_rings - 1; ++ring) {
		for (int segment = 0; segment < segments; ++segment) {
			int i0 = ring * vertices_per_ring + segment;
			int i1 = ring * vertices_per_ring + segment + 1;
			int i2 = (ring + 1) * vertices_per_ring + segment;
			int i3 = (ring + 1) * vertices_per_ring + segment + 1;

			// Create two triangles per quad
			if (i0 < vertices.size() && i1 < vertices.size() &&
					i2 < vertices.size() && i3 < vertices.size()) {
				mesh_faces.emplace_back(i0, i1, i2);
				mesh_faces.emplace_back(i1, i3, i2);
			}
		}
	}

	faces = mesh_faces;

	// Mark mesh as generated
	mesh_generated = true;
}

bool TaperedCapsule::isInContact(
		const Vec3d &center_prim,
		const Vec3d &pos,
		const Vec3d &velocity,
		Vec3d &normal,
		double &dist,
		Vec3d &v_out) {
	// Use center_prim as the effective center for collision detection
	Vec3d closest_point;
	Vec3d surface_normal;

	dist = closestPointOnSurface(pos, closest_point, surface_normal);

	// Contact threshold (can be tuned)
	const double contact_threshold = 0.01;

	if (dist < contact_threshold) {
		normal = surface_normal;
		v_out = closest_point;
		return true;
	}

	return false;
}

double TaperedCapsule::closestPointOnSurface(
		const Vec3d &point,
		Vec3d &closest_point,
		Vec3d &surface_normal) const {
	// Transform point to capsule local space (relative to center)
	Vec3d local_point = point - center;

	// Assume axis is Y-up for now (full rotation support can be added later)
	double total_height = mid_height + radius_top + radius_bottom;
	double half_height = total_height * 0.5;

	double y = local_point[1];
	double radial_dist_xz = std::sqrt(local_point[0] * local_point[0] +
			local_point[2] * local_point[2]);

	// Determine which section of the tapered capsule the point is closest to

	// TOP HEMISPHERE (y > half_height - radius_top)
	if (y > half_height - radius_top) {
		Vec3d hemisphere_center(0, half_height - radius_top, 0);
		Vec3d to_point = local_point - hemisphere_center;
		double dist_from_center = to_point.norm();

		if (dist_from_center < 1e-10) {
			// Point at hemisphere center
			surface_normal = Vec3d(0, 1, 0);
			closest_point = center + hemisphere_center + surface_normal * radius_top;
		} else {
			surface_normal = to_point / dist_from_center;
			closest_point = center + hemisphere_center + surface_normal * radius_top;
		}

		return (point - closest_point).norm();
	}

	// BOTTOM HEMISPHERE (y < -half_height + radius_bottom)
	if (y < -half_height + radius_bottom) {
		Vec3d hemisphere_center(0, -half_height + radius_bottom, 0);
		Vec3d to_point = local_point - hemisphere_center;
		double dist_from_center = to_point.norm();

		if (dist_from_center < 1e-10) {
			surface_normal = Vec3d(0, -1, 0);
			closest_point = center + hemisphere_center + surface_normal * radius_bottom;
		} else {
			surface_normal = to_point / dist_from_center;
			closest_point = center + hemisphere_center + surface_normal * radius_bottom;
		}

		return (point - closest_point).norm();
	}

	// TAPERED CYLINDER SECTION
	// Interpolate radius based on y position using septic smoothstep
	double v = (half_height - radius_top - y) / mid_height;
	v = std::max(0.0, std::min(1.0, v)); // Clamp to [0, 1]
	double current_radius = computeTaperedRadius(v, radius_top, radius_bottom);

	// Find closest point on the circle at this height
	if (radial_dist_xz < 1e-10) {
		// Point on axis - pick arbitrary radial direction
		surface_normal = Vec3d(1, 0, 0);
		closest_point = center + Vec3d(current_radius, y, 0);
	} else {
		// Project to cylinder surface
		double scale = current_radius / radial_dist_xz;
		Vec3d radial_dir(local_point[0] / radial_dist_xz, 0, local_point[2] / radial_dist_xz);
		surface_normal = radial_dir;
		closest_point = center + Vec3d(local_point[0] * scale, y, local_point[2] * scale);
	}

	return (point - closest_point).norm();
}

Mat3x3d TaperedCapsule::buildRotationMatrix(const Vec3d &from, const Vec3d &to) const {
	// Build rotation matrix from source direction to target direction
	// Using Rodrigues' rotation formula for robust 3D rotation

	Vec3d from_normalized = from.normalized();
	Vec3d to_normalized = to.normalized();

	// Check if vectors are already aligned (or opposite)
	double dot_product = from_normalized.dot(to_normalized);

	if (std::abs(dot_product - 1.0) < 1e-9) {
		// Vectors are already aligned - return identity matrix
		return Mat3x3d::Identity();
	}

	if (std::abs(dot_product + 1.0) < 1e-9) {
		// Vectors are opposite - rotate 180 degrees around any perpendicular axis
		Vec3d perpendicular;
		if (std::abs(from_normalized.x()) < 0.9) {
			perpendicular = Vec3d(1, 0, 0);
		} else {
			perpendicular = Vec3d(0, 1, 0);
		}

		// Make perpendicular axis orthogonal to from vector
		perpendicular = (perpendicular - from_normalized * from_normalized.dot(perpendicular)).normalized();

		// 180-degree rotation matrix
		Mat3x3d rotation = 2.0 * perpendicular * perpendicular.transpose() - Mat3x3d::Identity();
		return rotation;
	}

	// General case: use Rodrigues' rotation formula
	Vec3d rotation_axis = from_normalized.cross(to_normalized).normalized();
	double rotation_angle = std::acos(std::max(-1.0, std::min(1.0, dot_product)));

	// Rodrigues' formula: R = I + sin(θ)[k]× + (1-cos(θ))[k]×²
	// where [k]× is the skew-symmetric cross-product matrix

	Mat3x3d K; // Skew-symmetric matrix for cross product
	K << 0, -rotation_axis.z(), rotation_axis.y(),
			rotation_axis.z(), 0, -rotation_axis.x(),
			-rotation_axis.y(), rotation_axis.x(), 0;

	Mat3x3d rotation = Mat3x3d::Identity() +
			std::sin(rotation_angle) * K +
			(1.0 - std::cos(rotation_angle)) * K * K;

	return rotation;
}

bool TaperedCapsule::exportToOBJ(const std::string &filename) const {
	if (!mesh_generated) {
		// Generate mesh if not already done
		const_cast<TaperedCapsule *>(this)->generateMesh();
	}

	if (vertices.empty() || faces.empty()) {
		std::cerr << "Error: No mesh data available for OBJ export" << std::endl;
		return false;
	}

	// Use the new writeOBJFile method from MeshFileHandler
	MeshFileHandler::writeOBJFile(filename.c_str(), vertices, faces);
	return true;
}

} // namespace tool_cloth_dynamics
