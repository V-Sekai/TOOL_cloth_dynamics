#include "MeshLoader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mesh_collision {

Mesh loadOBJ(const std::string &filepath) {
	Mesh mesh;
	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open OBJ file: " + filepath);
	}

	std::vector<Eigen::Vector3d> verts;
	std::vector<Eigen::Vector3i> faces;

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string prefix;
		iss >> prefix;

		if (prefix == "v") {
			Eigen::Vector3d vert;
			iss >> vert.x() >> vert.y() >> vert.z();
			verts.push_back(vert);
		} else if (prefix == "f") {
			Eigen::Vector3i face;
			std::string token;
			int idx = 0;
			while (iss >> token) {
				size_t pos1 = token.find('/');
				size_t pos2 = token.rfind('/');
				std::istringstream token_iss(token.substr(0, pos1));
				token_iss >> face[idx];
				--face[idx]; // 1-based to 0-based
				++idx;
				if (idx == 3)
					break;
			}
			faces.push_back(face);
		}
		// Ignore vt, vn, etc. for proto
	}

	if (verts.empty() || faces.empty()) {
		throw std::runtime_error("No verts or faces found in OBJ: " + filepath);
	}

	mesh.V.resize(3, verts.size());
	for (size_t i = 0; i < verts.size(); ++i) {
		mesh.V.col(i) = verts[i];
	}

	mesh.F.resize(3, faces.size());
	for (size_t i = 0; i < faces.size(); ++i) {
		mesh.F.col(i) = faces[i];
	}

	return mesh;
}

} // namespace mesh_collision
