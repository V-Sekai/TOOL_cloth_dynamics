#pragma once

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector3d;

namespace tool_cloth_dynamics {
namespace collision {

enum class ClosestPointOnTriangleType {
	AtA,
	AtB,
	AtC,
	AtAB,
	AtBC,
	AtAC,
	AtInterior,
	NotFound
};

struct Contact {
	Vector3d point;
	Vector3d normal;
	double depth;
	int particle_id;
	Vector3d bary; // Barycentrics for edge/vertex averaging if needed
};

Vector3d closestPointTriangle(const Vector3d &p, const Vector3d &a, const Vector3d &b, const Vector3d &c, Vector3d &bary, ClosestPointOnTriangleType &type);
double getClosestDistanceBetweenSegments(const Vector3d &p0, const Vector3d &p1, const Vector3d &q0, const Vector3d &q1);
std::vector<Contact> detectPointTriCollisions(const MatrixXd &cloth_V, const MatrixXi &cloth_F,
		const MatrixXd &obstacle_V, const MatrixXi &obstacle_F, double max_dist = 0.1);

} // namespace collision
} // namespace tool_cloth_dynamics
