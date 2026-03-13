/*
 * Collision stencil adapter: build EfSet / EeSet / VfSet from the cloth
 * simulator's collision data for the Schwarz preconditioner.
 * See docs/PRECONDITIONER_INVESTIGATION.md and the integration plan.
 */
#ifndef OMEGAENGINE_COLLISIONSTENCILADAPTER_H
#define OMEGAENGINE_COLLISIONSTENCILADAPTER_H

#include "Macros.h"
#include <vector>

namespace OmegaEngine {

/** POD compatible with preconditioner VfSet (vertex–face). */
struct VfSetPOD {
	int m_vId = -1;
	int m_fId = -1;
	float stiff = 1.f;
	float m_bary[2] = {0.f, 0.f};  /* (x, y) with 1-x-y implicit */
	float m_normal[3] = {0.f, 0.f, 0.f};
};

/** POD compatible with preconditioner EfSet (edge–face). */
struct EfSetPOD {
	int m_eId = -1;
	int m_fId = -1;
	float stiff = 1.f;
	float m_bary[3] = {0.f, 0.f, 0.f};
	float m_normal[3] = {0.f, 0.f, 0.f};
};

/** POD compatible with preconditioner EeSet (edge–edge). */
struct EeSetPOD {
	int m_eId0 = -1;
	int m_eId1 = -1;
	float stiff = 1.f;
	float m_bary[2] = {0.f, 0.f};
	float m_normal[3] = {0.f, 0.f, 0.f};
};

/** Edge as (v0, v1) with v0 < v1. */
struct EdgeVerts {
	int v0 = 0;
	int v1 = 0;
	bool operator<(const EdgeVerts& o) const {
		if (v0 != o.v0) return v0 < o.v0;
		return v1 < o.v1;
	}
};

/**
 * Mesh topology for collision stencils: unique edges and face vertex indices.
 * Build with buildMeshTopology().
 */
struct CollisionStencilMeshTopology {
	int numVerts = 0;
	int numEdges = 0;
	int numFaces = 0;
	std::vector<EdgeVerts> edges;           /* length numEdges */
	std::vector<int> faceVerts;             /* 3 * numFaces: tri i = [3*i, 3*i+1, 3*i+2] */
	std::vector<std::vector<int>> vertFaces; /* vertFaces[v] = face indices containing v */
};

/** Minimal self-collision info for the adapter (avoids including Simulation.h). */
struct SelfCollisionInfo {
	int particleId1 = -1;
	int particleId2 = -1;
	double normal[3] = {0., 0., 0.};
};

/**
 * Build mesh topology (edges, face list, vert->faces) from triangle mesh.
 * Each triangle is (p0_idx, p1_idx, p2_idx). Edges are deduplicated.
 *
 * Example from Simulation::mesh:
 *   std::vector<int> triP0, triP1, triP2;
 *   for (const auto& t : mesh) { triP0.push_back(t.p0_idx); triP1.push_back(t.p1_idx); triP2.push_back(t.p2_idx); }
 *   buildMeshTopology(static_cast<int>(particles.size()), triP0, triP1, triP2, topology);
 */
void buildMeshTopology(
	int numVerts,
	const std::vector<int>& triP0,
	const std::vector<int>& triP1,
	const std::vector<int>& triP2,
	CollisionStencilMeshTopology& out);

/**
 * Fill VfSet (and optionally EfSet/EeSet) and count arrays from the simulator's
 * self-collision list. Uses mesh topology to map particle pairs to vertex–face
 * pairs (one particle as vertex, other's incident triangle as face).
 *
 * Input:
 *   - topology: from buildMeshTopology()
 *   - selfCollisions: list of (particleId1, particleId2, normal, ...)
 *   - defaultStiff: stiffness for stencils
 *
 * Output:
 *   - vfSets, efSets, eeSets: filled only for types we have data for (VF from
 *     self-collision; EF/EE left empty for now).
 *   - efCounts: length topology.numEdges+1; efCounts[topology.numEdges] = efSets.size()
 *   - eeCounts: length topology.numEdges+1; eeCounts[topology.numEdges] = eeSets.size()
 *   - vfCounts: length topology.numVerts+1; vfCounts[topology.numVerts] = vfSets.size()
 */
void fillCollisionStencilsFromSelfCollisions(
	const CollisionStencilMeshTopology& topology,
	const std::vector<SelfCollisionInfo>& selfCollisions,
	float defaultStiff,
	std::vector<EfSetPOD>& efSets,
	std::vector<EeSetPOD>& eeSets,
	std::vector<VfSetPOD>& vfSets,
	std::vector<unsigned int>& efCounts,
	std::vector<unsigned int>& eeCounts,
	std::vector<unsigned int>& vfCounts);

} // namespace OmegaEngine

#endif
