#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector3i;

namespace mesh_collision {

struct Mesh {
	MatrixXd V;
	MatrixXi F;
};

struct Contact {
	Vector3d point;
	Vector3d normal;
	double depth;
	int particle_id;
	Vector3d bary;
	int mesh_id; // triangle id in mesh.F
};

inline void loadOBJ(const std::string &filepath, MatrixXd &V, MatrixXi &F) {
	V = MatrixXd::Zero(3, 0);
	F = MatrixXi::Zero(3, 0);
	std::ifstream file(filepath);
	std::string line;
	while (std::getline(file, line)) {
		if (line.empty() || line[0] == '#')
			continue;
		std::istringstream iss(line);
		std::string prefix;
		iss >> prefix;
		if (prefix == "v ") {
			Vector3d vert;
			iss >> vert.x() >> vert.y() >> vert.z();
			V.conservativeResize(Eigen::NoChange, V.cols() + 1);
			V.col(V.cols() - 1) = vert;
		} else if (prefix == "f ") {
			Vector3i face;
			std::string token;
			int k = 0;
			while (iss >> token) {
				size_t slash1 = token.find('/');
				std::string idx_str = token.substr(0, slash1);
				if (!idx_str.empty())
					face(k) = std::stoi(idx_str) - 1; // 0-based
				k++;
				if (k == 3)
					break;
			}
			F.conservativeResize(Eigen::NoChange, F.cols() + 1);
			F.col(F.cols() - 1) = face;
		}
	}
}

inline std::vector<Contact> detectPointTriCollisions(const MatrixXd &cloth_V, const MatrixXi &cloth_F,
		const MatrixXd &obstacle_V, const MatrixXi &obstacle_F, double max_dist) {
	std::vector<Contact> contacts;
	for (int pv = 0; pv < cloth_V.cols(); ++pv) {
		const Vector3d p = cloth_V.col(pv);
		for (int tf = 0; tf < obstacle_F.cols(); ++tf) {
			const Vector3d &v1 = obstacle_V.col(obstacle_F(0, tf));
			const Vector3d &v2 = obstacle_V.col(obstacle_F(1, tf));
			const Vector3d &v3 = obstacle_V.col(obstacle_F(2, tf));
			Vector3d edge1 = v2 - v1;
			Vector3d edge2 = v3 - v1;
			Vector3d p_vec = p - v1;
			Vector3d cross_edge1_edge2 = edge1.cross(edge2);
			double denom = cross_edge1_edge2.squaredNorm();
			if (denom == 0)
				continue;
			double u = edge2.cross(p_vec).dot(cross_edge1_edge2) / denom;
			double v = p_vec.cross(edge2).dot(cross_edge1_edge2) / denom;
			if (u < 0 || v < 0 || u + v > 1)
				continue;
			Vector3d closest = v1 + u * edge1 + v * edge2;
			double dist = (p - closest).norm();
			if (dist <= max_dist) {
				double signed_dist = p.dot(cross_edge1_edge2) / std::sqrt(denom);
				Vector3d normal = cross_edge1_edge2.normalized();
				if (signed_dist < 0)
					normal = -normal;
				contacts.push_back({ closest, normal, dist, pv, Vector3d(u, v, 1 - u - v), tf });
			}
		}
	}
	return contacts;
}

} // namespace mesh_collision
