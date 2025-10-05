#include "SkeletonLoader.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tool_cloth_dynamics {

Skeleton Skeleton::fromOBJ(const std::string &filepath) {
	return SkeletonLoader::loadFromOBJ(filepath);
}

Skeleton SkeletonLoader::loadFromOBJ(const std::string &filepath) {
	if (!validateFile(filepath)) {
		throw std::runtime_error("Cannot read skeleton file: " + filepath);
	}

	// Parse vertices (joints) and line segments (bones)
	std::vector<Vec3d> vertices = parseVertices(filepath);
	std::vector<std::pair<int, int>> edges = parseLines(filepath);

	if (vertices.empty()) {
		throw std::runtime_error("No vertices found in skeleton file: " + filepath);
	}

	if (edges.empty()) {
		throw std::runtime_error("No line segments found in skeleton file: " + filepath);
	}

	// Validate edge indices
	for (const auto &edge : edges) {
		if (edge.first < 0 || edge.first >= static_cast<int>(vertices.size()) ||
				edge.second < 0 || edge.second >= static_cast<int>(vertices.size())) {
			throw std::runtime_error("Invalid edge indices in skeleton file: " + filepath);
		}
	}

	// Convert edges to bones
	std::vector<Bone> bones;
	bones.reserve(edges.size());

	for (const auto &edge : edges) {
		Bone bone;
		bone.start = vertices[edge.first];
		bone.end = vertices[edge.second];
		bone.parent_id = -1; // Hierarchy determination postponed
		bone.name = ""; // Names can be added later if needed
		bones.push_back(bone);
	}

	Skeleton skeleton;
	skeleton.bones = std::move(bones);
	skeleton.joints = std::move(vertices);

	std::cout << "Loaded skeleton with " << skeleton.getBoneCount() << " bones and " << skeleton.getJointCount() << " joints" << std::endl;

	return skeleton;
}

std::vector<std::pair<Vec3d, Vec3d>> SkeletonLoader::loadBones(const std::string &filepath) {
	if (!validateFile(filepath)) {
		throw std::runtime_error("Cannot read skeleton file: " + filepath);
	}

	std::vector<Vec3d> vertices = parseVertices(filepath);
	std::vector<std::pair<int, int>> edges = parseLines(filepath);

	std::vector<std::pair<Vec3d, Vec3d>> bones;
	bones.reserve(edges.size());

	for (const auto &edge : edges) {
		if (edge.first >= 0 && edge.first < static_cast<int>(vertices.size()) &&
				edge.second >= 0 && edge.second < static_cast<int>(vertices.size())) {
			bones.emplace_back(vertices[edge.first], vertices[edge.second]);
		}
	}

	return bones;
}

bool SkeletonLoader::validateEdgeCompatibility(const Skeleton &skeleton_a, const Skeleton &skeleton_b) {
	// Full implementation of cloth-fit's are_same_edges() constraint
	if (skeleton_a.getBoneCount() != skeleton_b.getBoneCount()) {
		std::cout << "Edge compatibility failed: Bone count mismatch ("
				  << skeleton_a.getBoneCount() << " vs " << skeleton_b.getBoneCount() << ")" << std::endl;
		return false;
	}

	if (skeleton_a.getJointCount() != skeleton_b.getJointCount()) {
		std::cout << "Edge compatibility failed: Joint count mismatch ("
				  << skeleton_a.getJointCount() << " vs " << skeleton_b.getJointCount() << ")" << std::endl;
		return false;
	}

	// Build edge connectivity matrices for both skeletons
	std::set<std::pair<int, int>> edges_a, edges_b;

	// For skeleton A: Map vertex positions to canonical indices
	std::map<Vec3d, int, std::function<bool(const Vec3d &, const Vec3d &)>> vertex_map_a(
			[](const Vec3d &a, const Vec3d &b) {
				const double epsilon = 1e-9;
				if (std::abs(a.x() - b.x()) > epsilon)
					return a.x() < b.x();
				if (std::abs(a.y() - b.y()) > epsilon)
					return a.y() < b.y();
				return a.z() < b.z();
			});

	// Build canonical vertex indices for skeleton A
	for (size_t i = 0; i < skeleton_a.getJointCount(); ++i) {
		vertex_map_a[skeleton_a.joints[i]] = static_cast<int>(i);
	}

	// Convert bones to canonical edge indices for skeleton A
	for (const Bone &bone : skeleton_a.bones) {
		auto start_it = vertex_map_a.find(bone.start);
		auto end_it = vertex_map_a.find(bone.end);

		if (start_it != vertex_map_a.end() && end_it != vertex_map_a.end()) {
			int idx_start = start_it->second;
			int idx_end = end_it->second;
			// Store edges in canonical order (smaller index first)
			if (idx_start > idx_end)
				std::swap(idx_start, idx_end);
			edges_a.insert({ idx_start, idx_end });
		}
	}

	// For skeleton B: Same process
	std::map<Vec3d, int, std::function<bool(const Vec3d &, const Vec3d &)>> vertex_map_b(
			[](const Vec3d &a, const Vec3d &b) {
				const double epsilon = 1e-9;
				if (std::abs(a.x() - b.x()) > epsilon)
					return a.x() < b.x();
				if (std::abs(a.y() - b.y()) > epsilon)
					return a.y() < b.y();
				return a.z() < b.z();
			});

	for (size_t i = 0; i < skeleton_b.getJointCount(); ++i) {
		vertex_map_b[skeleton_b.joints[i]] = static_cast<int>(i);
	}

	for (const Bone &bone : skeleton_b.bones) {
		auto start_it = vertex_map_b.find(bone.start);
		auto end_it = vertex_map_b.find(bone.end);

		if (start_it != vertex_map_b.end() && end_it != vertex_map_b.end()) {
			int idx_start = start_it->second;
			int idx_end = end_it->second;
			if (idx_start > idx_end)
				std::swap(idx_start, idx_end);
			edges_b.insert({ idx_start, idx_end });
		}
	}

	// Compare edge sets
	if (edges_a.size() != edges_b.size()) {
		std::cout << "Edge compatibility failed: Edge set size mismatch ("
				  << edges_a.size() << " vs " << edges_b.size() << ")" << std::endl;
		return false;
	}

	// Check if all edges match
	for (const auto &edge : edges_a) {
		if (edges_b.find(edge) == edges_b.end()) {
			std::cout << "Edge compatibility failed: Edge (" << edge.first
					  << "," << edge.second << ") not found in skeleton B" << std::endl;
			return false;
		}
	}

	std::cout << "Edge compatibility validation: SUCCESS - " << edges_a.size()
			  << " edges match perfectly" << std::endl;

	return true;
}

std::vector<Vec3d> SkeletonLoader::parseVertices(const std::string &filepath) {
	std::ifstream file(filepath);
	std::string line;
	std::vector<Vec3d> vertices;

	while (std::getline(file, line)) {
		// Trim whitespace
		line.erase(0, line.find_first_not_of(" \t"));
		line.erase(line.find_last_not_of(" \t") + 1);

		// Skip empty lines and comments
		if (line.empty() || line[0] == '#') {
			continue;
		}

		// Parse vertex lines: "v x y z"
		if (line.substr(0, 2) == "v ") {
			std::istringstream iss(line.substr(2));
			double x, y, z;
			if (iss >> x >> y >> z) {
				vertices.emplace_back(x, y, z);
			} else {
				std::cout << "Warning: Failed to parse vertex: " << line << std::endl;
			}
		}
	}

	return vertices;
}

std::vector<std::pair<int, int>> SkeletonLoader::parseLines(const std::string &filepath) {
	std::ifstream file(filepath);
	std::string line;
	std::vector<std::pair<int, int>> edges;

	while (std::getline(file, line)) {
		// Trim whitespace
		line.erase(0, line.find_first_not_of(" \t"));
		line.erase(line.find_last_not_of(" \t") + 1);

		// Skip empty lines and comments
		if (line.empty() || line[0] == '#') {
			continue;
		}

		// Parse line segments: "l v1 v2"
		if (line.substr(0, 2) == "l ") {
			std::istringstream iss(line.substr(2));
			int v1, v2;
			if (iss >> v1 >> v2) {
				// OBJ indices are 1-based, convert to 0-based
				edges.emplace_back(v1 - 1, v2 - 1);
			} else {
				std::cout << "Warning: Failed to parse line segment: " << line << std::endl;
			}
		}
	}

	return edges;
}

bool SkeletonLoader::validateFile(const std::string &filepath) {
	try {
		return std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath);
	} catch (const std::filesystem::filesystem_error &e) {
		std::cerr << "Error: Filesystem error: " << e.what() << std::endl;
		return false;
	}
}

} // namespace tool_cloth_dynamics
