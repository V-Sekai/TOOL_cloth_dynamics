#pragma once

#include "../engine/UtilityFunctions.h"
#include <memory>
#include <vector>

namespace tool_cloth_dynamics {

/**
 * @brief Sparse octree node using Left-Child Right-Sibling optimization
 *
 * Uses LCRS structure to reduce memory usage from 8 pointers per node to 2.
 * Only non-empty octants are stored, making it truly sparse.
 */
struct OctreeNode {
	Vec3d center;
	double half_width;
	std::vector<int> vertex_indices; // Leaf nodes only

	// LCRS optimization: only 2 pointers instead of 8
	std::unique_ptr<OctreeNode> first_child; // First child in LCRS tree
	std::unique_ptr<OctreeNode> next_sibling; // Next sibling in LCRS tree

	/**
	 * @brief Check if this is a leaf node
	 */
	bool isLeaf() const { return first_child == nullptr; }

	/**
	 * @brief Constructor
	 */
	OctreeNode(const Vec3d &c, double hw) :
			center(c), half_width(hw) {}
};

/**
 * @brief Sparse octree for fast nearest-neighbor queries on mesh vertices
 *
 * Optimized for skeleton-to-mesh radius estimation with O(log N) performance.
 * Uses LCRS (Left-Child Right-Sibling) structure to minimize memory usage.
 */
class SparseOctree {
public:
	/**
	 * @brief Build octree from mesh vertices
	 *
	 * @param mesh_V Matrix of vertex positions (3 x N)
	 */
	void build(const MatXd &mesh_V);

	/**
	 * @brief Find nearest distance to any mesh vertex
	 *
	 * @param query_point Point to query
	 * @return Distance to nearest vertex
	 */
	double nearestDistance(const Vec3d &query_point) const;

	/**
	 * @brief Find nearest vertex to query point
	 *
	 * @param query_point Point to query
	 * @param nearest_vertex Output: nearest vertex position
	 * @return Distance to nearest vertex
	 */
	double nearestVertex(const Vec3d &query_point, Vec3d &nearest_vertex) const;

	/**
	 * @brief Get tree statistics for debugging
	 */
	void getStatistics(int &total_nodes, int &leaf_nodes, int &max_depth) const;

private:
	std::unique_ptr<OctreeNode> root;
	const MatXd *mesh_vertices = nullptr;

	// Build parameters
	double min_node_size = 0.01; // Prevent over-subdivision
	int max_vertices_per_leaf = 16; // Balance between depth and leaf size

	/**
	 * @brief Recursively build octree
	 */
	std::unique_ptr<OctreeNode> buildRecursive(
			const Vec3d &center,
			double half_width,
			const std::vector<int> &vertex_indices);

	/**
	 * @brief Recursively search for nearest point
	 */
	void searchNearest(
			const OctreeNode *node,
			const Vec3d &query,
			double &best_dist,
			Vec3d &best_vertex) const;

	/**
	 * @brief Calculate statistics recursively
	 */
	void calculateStatistics(
			const OctreeNode *node,
			int &total_nodes,
			int &leaf_nodes,
			int &max_depth,
			int current_depth = 0) const;
};

/**
 * @brief Radius estimation utilities using sparse octree
 */
class RadiusEstimator {
public:
	/**
	 * @brief Estimate radius for a bone using mesh proximity
	 *
	 * @param bone Bone structure
	 * @param octree Pre-built octree for mesh
	 * @param num_samples Number of samples along bone (default: 10)
	 * @return Estimated radius (median of samples)
	 */
	static double estimateRadius(
			const struct Bone &bone,
			const SparseOctree &octree,
			int num_samples = 10);

	/**
	 * @brief Estimate taper for a bone (different radii at start/end)
	 *
	 * @param bone Bone structure
	 * @param octree Pre-built octree for mesh
	 * @return Pair of (radius_start, radius_end)
	 */
	static std::pair<double, double> estimateTaper(
			const struct Bone &bone,
			const SparseOctree &octree);

	/**
	 * @brief Estimate radii for all bones in skeleton
	 *
	 * @param skeleton Input skeleton
	 * @param mesh_V Mesh vertices for proximity estimation
	 * @return Vector of estimated radii per bone
	 */
	static std::vector<double> estimateAllRadii(
			const struct Skeleton &skeleton,
			const MatXd &mesh_V);
};

} // namespace tool_cloth_dynamics
