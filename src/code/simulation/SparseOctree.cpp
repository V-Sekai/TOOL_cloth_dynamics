#include "SparseOctree.h"
#include "../supports/Logging.h"
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

} // namespace tool_cloth_dynamics
