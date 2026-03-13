/*
 * Collision stencil adapter: build EfSet / EeSet / VfSet from the cloth
 * simulator's collision data. See CollisionStencilAdapter.h.
 */
#include "CollisionStencilAdapter.h"
#include <algorithm>
#include <set>

namespace OmegaEngine {

void buildMeshTopology(
	int numVerts,
	const std::vector<int>& triP0,
	const std::vector<int>& triP1,
	const std::vector<int>& triP2,
	CollisionStencilMeshTopology& out)
{
	out.numVerts = numVerts;
	out.numFaces = static_cast<int>(triP0.size());
	out.edges.clear();
	out.faceVerts.clear();
	out.vertFaces.assign(static_cast<size_t>(numVerts), std::vector<int>());

	std::set<EdgeVerts> edgeSet;
	for (size_t i = 0; i < triP0.size(); i++) {
		int a = triP0[i], b = triP1[i], c = triP2[i];
		out.faceVerts.push_back(a);
		out.faceVerts.push_back(b);
		out.faceVerts.push_back(c);
		out.vertFaces[static_cast<size_t>(a)].push_back(static_cast<int>(i));
		out.vertFaces[static_cast<size_t>(b)].push_back(static_cast<int>(i));
		out.vertFaces[static_cast<size_t>(c)].push_back(static_cast<int>(i));
		EdgeVerts e1{ std::min(a, b), std::max(a, b) };
		EdgeVerts e2{ std::min(b, c), std::max(b, c) };
		EdgeVerts e3{ std::min(a, c), std::max(a, c) };
		edgeSet.insert(e1);
		edgeSet.insert(e2);
		edgeSet.insert(e3);
	}
	out.edges.assign(edgeSet.begin(), edgeSet.end());
	out.numEdges = static_cast<int>(out.edges.size());
}

static void setBaryForVertexInFace(int vertexId, const int* faceVerts, float* bary)
{
	/* Face has 3 vertices; which one is vertexId? bary = (x, y) with 1-x-y for third. */
	if (faceVerts[0] == vertexId) {
		bary[0] = 1.f;
		bary[1] = 0.f;
	} else if (faceVerts[1] == vertexId) {
		bary[0] = 0.f;
		bary[1] = 1.f;
	} else {
		bary[0] = 0.f;
		bary[1] = 0.f;
	}
}

void fillCollisionStencilsFromSelfCollisions(
	const CollisionStencilMeshTopology& topology,
	const std::vector<SelfCollisionInfo>& selfCollisions,
	float defaultStiff,
	std::vector<EfSetPOD>& efSets,
	std::vector<EeSetPOD>& eeSets,
	std::vector<VfSetPOD>& vfSets,
	std::vector<unsigned int>& efCounts,
	std::vector<unsigned int>& eeCounts,
	std::vector<unsigned int>& vfCounts)
{
	efSets.clear();
	eeSets.clear();
	vfSets.clear();

	for (const auto& sc : selfCollisions) {
		if (sc.particleId1 < 0 || sc.particleId2 < 0) continue;
		if (static_cast<size_t>(sc.particleId2) >= topology.vertFaces.size()) continue;
		const std::vector<int>& facesOfP2 = topology.vertFaces[static_cast<size_t>(sc.particleId2)];
		if (facesOfP2.empty()) continue;

		/* Pick first incident face as the "face" for this vertex–face pair. */
		int fId = facesOfP2[0];
		size_t off = static_cast<size_t>(fId) * 3;
		if (off + 2 >= topology.faceVerts.size()) continue;

		VfSetPOD vf;
		vf.m_vId = sc.particleId1;
		vf.m_fId = fId;
		vf.stiff = defaultStiff;
		setBaryForVertexInFace(sc.particleId2, &topology.faceVerts[off], vf.m_bary);
		vf.m_normal[0] = static_cast<float>(sc.normal[0]);
		vf.m_normal[1] = static_cast<float>(sc.normal[1]);
		vf.m_normal[2] = static_cast<float>(sc.normal[2]);
		vfSets.push_back(vf);
	}

	/* Count arrays: preconditioner expects length numEdges+1 and numVerts+1, with total at last index. */
	efCounts.assign(static_cast<size_t>(topology.numEdges + 1), 0u);
	eeCounts.assign(static_cast<size_t>(topology.numEdges + 1), 0u);
	vfCounts.assign(static_cast<size_t>(topology.numVerts + 1), 0u);
	efCounts[static_cast<size_t>(topology.numEdges)] = static_cast<unsigned int>(efSets.size());
	eeCounts[static_cast<size_t>(topology.numEdges)] = static_cast<unsigned int>(eeSets.size());
	vfCounts[static_cast<size_t>(topology.numVerts)] = static_cast<unsigned int>(vfSets.size());
}

} // namespace OmegaEngine
