#include "SparseOctree.h"
#include "../supports/Logging.h"
#include "SkeletonLoader.h"
#include <algorithm>
#include <limits>
#include <numeric>

namespace tool_cloth_dynamics {

void SparseOctree::build(const MatXd &mesh_V) {
	mesh_vertices = &mesh_V;

	if (mesh_V.cols() == 0) {
		Logging::logWarning("SparseOctree::build - Empty mesh provided");
		return;
	}

	// Compute bounding box
	Vec3d min_corner = mesh_V.rowwise().minCoeff();
	Vec3d max_corner = mesh_V.rowwise().maxCoeff();
	Vec3d center = (min_corner + max_corner) * 0.5;
	double half_width = (max_corner - min_corner).maxCoeff() * 0.5;

	// Add small padding to ensure all points are inside
	half_width *= 1.1;

	// Build root with all vertices
	std::vector<int> all_indices(mesh_V.cols());
	std::iota(all_indices.begin(), all_indices.end(), 0);

	root = buildRecursive(center, half_width, all_indices);

	// Log statistics
	int total_nodes, leaf_nodes, max_depth;
	getStatistics(total_nodes, leaf_nodes, max_depth);

	std::cout << "SparseOctree built: " << std::to_string(total_nodes) << " nodes, " << std::to_string(leaf_nodes) << " leaves, max depth " << std::to_string(max_depth) << std::endl;
}

double SparseOctree::nearestDistance(const Vec3d &query_point) const {
	Vec3d nearest_vertex;
	return nearestVertex(query_point, nearest_vertex);
}

double SparseOctree::nearestVertex(const Vec3d &query_point, Vec3d &nearest_vertex) const {
	if (!root || !mesh_vertices) {
		std::cerr << "ERROR: SparseOctree not built - call build() first" << std::endl;
		return std::numeric_limits<double>::infinity();
	}

	double best_dist = std::numeric_limits<double>::infinity();
	nearest_vertex = Vec3d::Zero();

	searchNearest(root.get(), query_point, best_dist, nearest_vertex);

	return best_dist;
}

void SparseOctree::getStatistics(int &total_nodes, int &leaf_nodes, int &max_depth) const {
	total_nodes = leaf_nodes = max_depth = 0;
	if (root) {
		calculateStatistics(root.get(), total_nodes, leaf_nodes, max_depth);
	}
}

std::unique_ptr<OctreeNode> SparseOctree::buildRecursive(
		const Vec3d &center,
		double half_width,
		const std::vector<int> &vertex_indices) {
	auto node = std::make_unique<OctreeNode>(center, half_width);

	// Leaf conditions: small enough or few vertices
	if (half_width < min_node_size || vertex_indices.size() <= static_cast<size_t>(max_vertices_per_leaf)) {
		node->vertex_indices = vertex_indices;
		return node;
	}

	// Subdivide into 8 octants using LCRS structure
	std::vector<std::vector<int>> octant_vertices(8);

	for (int idx : vertex_indices) {
		Vec3d vertex = mesh_vertices->col(idx);
		int octant = 0;
		if (vertex.x() > center.x())
			octant |= 1;
		if (vertex.y() > center.y())
			octant |= 2;
		if (vertex.z() > center.z())
			octant |= 4;
		octant_vertices[octant].push_back(idx);
	}

	// Build non-empty children in LCRS format
	OctreeNode *prev_sibling = nullptr;
	double child_half_width = half_width * 0.5;

	for (int i = 0; i < 8; i++) {
		if (octant_vertices[i].empty())
			continue; // Sparse: skip empty octants

		Vec3d child_center = center;
		child_center.x() += (i & 1) ? child_half_width : -child_half_width;
		child_center.y() += (i & 2) ? child_half_width : -child_half_width;
		child_center.z() += (i & 4) ? child_half_width : -child_half_width;

		auto child = buildRecursive(child_center, child_half_width, octant_vertices[i]);

		// LCRS linking
		if (node->first_child == nullptr) {
			node->first_child = std::move(child); // First child
			prev_sibling = node->first_child.get();
		} else {
			prev_sibling->next_sibling = std::move(child); // Link as sibling
			prev_sibling = prev_sibling->next_sibling.get();
		}
	}

	return node;
}

void SparseOctree::searchNearest(
		const OctreeNode *node,
		const Vec3d &query,
		double &best_dist,
		Vec3d &best_vertex) const {
	if (!node)
		return;

	// Check if node can contain closer point
	double node_dist = (query - node->center).norm() - node->half_width * std::sqrt(3.0);
	if (node_dist > best_dist)
		return; // Prune

	if (node->isLeaf()) {
		// Check all vertices in leaf
		for (int idx : node->vertex_indices) {
			Vec3d vertex = mesh_vertices->col(idx);
			double dist = (query - vertex).norm();
			if (dist < best_dist) {
				best_dist = dist;
				best_vertex = vertex;
			}
		}
	} else {
		// Recursively search children (LCRS traversal)
		for (const OctreeNode *child = node->first_child.get();
				child;
				child = child->next_sibling.get()) {
			searchNearest(child, query, best_dist, best_vertex);
		}
	}
}

void SparseOctree::calculateStatistics(
		const OctreeNode *node,
		int &total_nodes,
		int &leaf_nodes,
		int &max_depth,
		int current_depth) const {
	if (!node)
		return;

	total_nodes++;
	max_depth = std::max(max_depth, current_depth);

	if (node->isLeaf()) {
		leaf_nodes++;
	} else {
		// Traverse children
		for (const OctreeNode *child = node->first_child.get();
				child;
				child = child->next_sibling.get()) {
			calculateStatistics(child, total_nodes, leaf_nodes, max_depth, current_depth + 1);
		}
	}
}

// RadiusEstimator implementation

double RadiusEstimator::estimateRadius(
		const Bone &bone,
		const SparseOctree &octree,
		int num_samples) {
	std::vector<double> radii;
	radii.reserve(num_samples);

	// Sample points along bone
	for (int i = 0; i < num_samples; i++) {
		double t = (num_samples > 1) ? static_cast<double>(i) / (num_samples - 1) : 0.5;
		Vec3d sample = bone.start + t * (bone.end - bone.start);

		// O(log N) nearest distance query
		double dist = octree.nearestDistance(sample);
		radii.push_back(dist);
	}

	// Return median radius (robust to outliers)
	std::sort(radii.begin(), radii.end());
	size_t median_idx = radii.size() / 2;

	if (radii.size() % 2 == 0 && radii.size() > 1) {
		return (radii[median_idx - 1] + radii[median_idx]) * 0.5;
	} else {
		return radii[median_idx];
	}
}

std::pair<double, double> RadiusEstimator::estimateTaper(
		const Bone &bone,
		const SparseOctree &octree) {
	// Sample directly at bone endpoints for clean taper estimation
	Vec3d sample_start = bone.start; // t = 0
	Vec3d sample_end = bone.end; // t = 1

	double r_start = octree.nearestDistance(sample_start);
	double r_end = octree.nearestDistance(sample_end);

	return { r_start, r_end };
}

std::vector<double> RadiusEstimator::sampleRadiusProfile(
		const Bone &bone,
		const SparseOctree &octree,
		int num_samples) {
	if (num_samples <= 2) {
		num_samples = 20; // Default for continuous profile sampling
	}

	std::vector<double> radius_profile;
	radius_profile.reserve(num_samples);

	// Sample radius at points along the full bone length
	for (int i = 0; i < num_samples; ++i) {
		double t = static_cast<double>(i) / (num_samples - 1);
		Vec3d sample_point = bone.start + t * (bone.end - bone.start);

		// Get distance to nearest mesh vertex (this is our radius estimate)
		double radius = octree.nearestDistance(sample_point);
		radius_profile.push_back(radius);
	}

	return radius_profile;
}

std::vector<double> RadiusEstimator::estimateAllRadii(
		const Skeleton &skeleton,
		const MatXd &mesh_V) {
	// Build octree once for all bones
	SparseOctree octree;
	octree.build(mesh_V);

	std::vector<double> radii;
	radii.reserve(skeleton.getBoneCount());

	for (const Bone &bone : skeleton.bones) {
		double radius = estimateRadius(bone, octree);
		radii.push_back(radius);
	}

	std::cout << "Estimated radii for " << std::to_string(skeleton.getBoneCount()) << " bones" << std::endl;

	return radii;
}

} // namespace tool_cloth_dynamics
