/*
 * MIT License
 *
 * Copyright (c) 2024-present K. S. Ernest (Fire) Lee
 * Copyright (c) 2022-2024 Yifei Li
 *
 * FCL-based collision detection: cloth BVH, primitives, layering via contactSorting.
 * Only compiled when USE_FCL is defined.
 */

#ifdef USE_FCL

#include "CollisionFCL.h"
#include "Simulation.h"
#include "Primitive.h"
#include "Triangle.h"
#include "Particle.h"
#include "../engine/Constants.h"
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/geometry/shape/sphere.h>
#include <fcl/geometry/shape/capsule.h>
#include <fcl/geometry/shape/halfspace.h>
#include <fcl/narrowphase/collision.h>
#include <fcl/narrowphase/collision_request.h>
#include <fcl/narrowphase/collision_result.h>
#include <fcl/narrowphase/collision_object.h>
#include <Eigen/Core>
#include <map>
#include <set>
#include <vector>

using S = double;
using BV = fcl::OBBRSS<S>;
using Vec3 = fcl::Vector3<S>;
using Transform3 = fcl::Transform3<S>;

namespace {

// Build FCL BVH for the cloth from mesh and vertex positions x (3n).
std::shared_ptr<fcl::BVHModel<BV>> buildClothBVH(
    const std::vector<OmegaEngine::Triangle>& mesh,
    const Eigen::VectorXd& x) {
  auto model = std::make_shared<fcl::BVHModel<BV>>();
  const int numVertices = static_cast<int>(x.size() / 3);
  const int numTris = static_cast<int>(mesh.size());
  model->beginModel(numTris, numVertices);
  for (int i = 0; i < numVertices; ++i) {
    model->addVertex(Vec3(x(3 * i), x(3 * i + 1), x(3 * i + 2)));
  }
  for (const auto& t : mesh) {
    const Vec3 p0(x(3 * t.p0_idx), x(3 * t.p0_idx + 1), x(3 * t.p0_idx + 2));
    const Vec3 p1(x(3 * t.p1_idx), x(3 * t.p1_idx + 1), x(3 * t.p1_idx + 2));
    const Vec3 p2(x(3 * t.p2_idx), x(3 * t.p2_idx + 1), x(3 * t.p2_idx + 2));
    model->addTriangle(p0, p1, p2);
  }
  model->endModel();
  return model;
}

// Create FCL collision object for a primitive. Returns null if type not supported for direct shape.
std::shared_ptr<fcl::CollisionGeometry<S>> createPrimitiveShape(
    OmegaEngine::Primitive* prim, const Vec3d& center) {
  using namespace OmegaEngine;
  if (!prim->isEnabled) return nullptr;
  if (auto* s = dynamic_cast<Sphere*>(prim)) {
    return std::make_shared<fcl::Sphere<S>>(s->radius);
  }
  if (auto* p = dynamic_cast<Plane*>(prim)) {
    // FCL Halfspace: n·x = d, interior is n·x < d. Plane: planeNormal·x + d = 0 => n·x = -d/|n|.
    OmegaEngine::Vec3d n = p->planeNormal / p->planeNormalNorm;
    S d_val = static_cast<S>(-p->d / p->planeNormalNorm);
    return std::make_shared<fcl::Halfspace<S>>(
        Vec3(n(0), n(1), n(2)), d_val);
  }
  if (auto* c = dynamic_cast<Capsule*>(prim)) {
    return std::make_shared<fcl::Capsule<S>>(c->radius, c->length);
  }
  // Bowl, Foot, LowerLeg, etc.: build BVH from primitive mesh (mesh/convex from existing geometry)
  if (prim->mesh.empty()) return nullptr;
  std::vector<Vec3> pts;
  for (const auto& pt : prim->points) {
    OmegaEngine::Vec3d pos = pt.pos + prim->center + center;
    pts.push_back(Vec3(pos(0), pos(1), pos(2)));
  }
  std::vector<fcl::Triangle> tris;
  for (const auto& t : prim->mesh) {
    tris.push_back(fcl::Triangle(static_cast<std::size_t>(t.p0_idx),
                                 static_cast<std::size_t>(t.p1_idx),
                                 static_cast<std::size_t>(t.p2_idx)));
  }
  auto model = std::make_shared<fcl::BVHModel<BV>>();
  model->beginModel(static_cast<int>(tris.size()), static_cast<int>(pts.size()));
  model->addSubModel(pts, tris);
  model->endModel();
  return model;
}

Transform3 getPrimitiveTransform(OmegaEngine::Primitive* prim, const Vec3d& center) {
  Transform3 tf = Transform3::Identity();
  tf.translation() = Vec3(center(0), center(1), center(2));
  if (auto* c = dynamic_cast<OmegaEngine::Capsule*>(prim)) {
    // Capsule axis: use globalAxis or rotation
    Eigen::Matrix3d R = c->globalRotation.linear();
    tf.linear() = R;
  }
  return tf;
}

// Pick particle id from cloth triangle contact: vertex closest to contact position.
int triangleToParticleId(const OmegaEngine::Triangle& t, const Vec3d& contactPos,
                         const Eigen::VectorXd& x) {
  auto dist = [&](int idx) {
    return (contactPos - x.segment<3>(3 * idx)).norm();
  };
  int p0 = t.p0_idx, p1 = t.p1_idx, p2 = t.p2_idx;
  double d0 = dist(p0), d1 = dist(p1), d2 = dist(p2);
  if (d0 <= d1 && d0 <= d2) return p0;
  if (d1 <= d0 && d1 <= d2) return p1;
  return p2;
}

}  // namespace

namespace OmegaEngine {

std::pair<Simulation::collisionInfoPair,
          std::vector<std::vector<Simulation::SelfCollisionInformation>>>
Simulation::collisionDetectionFCL(const VecXd& x_n, const VecXd& v,
                                  const VecXd& x_prim, const VecXd& v_prim) {
  std::vector<PrimitiveCollisionInformation> infos;
  std::vector<SelfCollisionInformation> selfinfos;
  std::vector<std::vector<SelfCollisionInformation>> layers;

  if (!contactEnabled) {
    return std::make_pair(std::make_pair(infos, selfinfos), layers);
  }

  const int n = static_cast<int>(particles.size());
  const int numPrims = static_cast<int>(primitives.size());

  // Build cloth BVH
  std::shared_ptr<fcl::BVHModel<BV>> clothModel = buildClothBVH(mesh, x_n);
  Transform3 clothTf = Transform3::Identity();
  auto clothGeom = std::shared_ptr<fcl::CollisionGeometry<S>>(clothModel);
  fcl::CollisionObject<S> clothObj(clothGeom, clothTf);

  fcl::CollisionRequest<S> request;
  request.num_max_contacts = 10000;
  request.enable_contact = true;

  // Cloth vs primitives
  for (int i = 0; i < numPrims; ++i) {
    Primitive* prim = primitives[i];
    Vec3d center = x_prim.segment<3>(3 * i);
    auto geom = createPrimitiveShape(prim, center);
    if (!geom) continue;
    Transform3 primTf = getPrimitiveTransform(prim, center);
    fcl::CollisionObject<S> primObj(geom, primTf);
    fcl::CollisionResult<S> result;
    fcl::collide(&clothObj, &primObj, request, result);
    for (size_t k = 0; k < result.numContacts(); ++k) {
      const auto& c = result.getContact(static_cast<int>(k));
      int triId = static_cast<int>(c.b1);
      if (triId < 0 || triId >= static_cast<int>(mesh.size())) continue;
      const Triangle& t = mesh[triId];
      Vec3d pos(c.pos(0), c.pos(1), c.pos(2));
      int particleId = triangleToParticleId(t, pos, x_n);
      PrimitiveCollisionInformation info{};
      info.primitiveId = i;
      info.particleId = particleId;
      info.normal = Vec3d(c.normal(0), c.normal(1), c.normal(2));
      info.v_out = Vec3d(0, 0, 0);
      info.collides = true;
      info.dist = static_cast<double>(c.penetration_depth);
      info.primTotalCollision = 1;
      info.r = Vec3d(0, 0, 0);
      info.d = info.normal * info.dist;
      info.type = STICK;
      infos.push_back(info);
    }
  }

  // Self-collision: cloth vs cloth (same BVH, two objects)
  if (selfcollisionEnabled && mesh.size() > 1) {
    auto clothGeom2 = std::shared_ptr<fcl::CollisionGeometry<S>>(clothModel);
    fcl::CollisionObject<S> clothObj2(clothGeom2, clothTf);
    fcl::CollisionResult<S> selfResult;
    fcl::collide(&clothObj, &clothObj2, request, selfResult);
    for (size_t k = 0; k < selfResult.numContacts(); ++k) {
      const auto& c = selfResult.getContact(static_cast<int>(k));
      int tri1 = static_cast<int>(c.b1);
      int tri2 = static_cast<int>(c.b2);
      if (tri1 == tri2) continue;
      if (tri1 < 0 || tri1 >= static_cast<int>(mesh.size()) ||
          tri2 < 0 || tri2 >= static_cast<int>(mesh.size()))
        continue;
      const Triangle& t1 = mesh[tri1];
      const Triangle& t2 = mesh[tri2];
      Vec3d pos(c.pos(0), c.pos(1), c.pos(2));
      int p1 = triangleToParticleId(t1, pos, x_n);
      int p2 = triangleToParticleId(t2, pos, x_n);
      if (p1 == p2) continue;
      SelfCollisionInformation info{};
      info.particleId1 = std::min(p1, p2);
      info.particleId2 = std::max(p1, p2);
      info.normal = Vec3d(c.normal(0), c.normal(1), c.normal(2));
      info.d = info.normal * static_cast<double>(c.penetration_depth);
      info.r = Vec3d(0, 0, 0);
      info.collides = true;
      info.dist = static_cast<double>(c.penetration_depth);
      info.layerId = 0;
      info.type = STICK;
      selfinfos.push_back(info);
    }
  }

  // Build selfCollisionMap and selfCollisionTable for contactSorting
  std::map<int, std::set<int>> selfCollisionMap;
  MatXi selfCollisionTable(n, n);
  selfCollisionTable.setZero();
  for (size_t i = 0; i < selfinfos.size(); ++i) {
    int a = selfinfos[i].particleId1;
    int b = selfinfos[i].particleId2;
    selfCollisionMap[a].insert(b);
    selfCollisionMap[b].insert(a);
    int idx = static_cast<int>(i);
    selfCollisionTable(a, b) = idx;
    selfCollisionTable(b, a) = idx;
  }

  Simulation::collisionInfoPair detections = std::make_pair(infos, selfinfos);
  if (contactEnabled && selfcollisionEnabled && !selfinfos.empty()) {
    layers = contactSorting(detections, selfCollisionMap, selfCollisionTable);
  }

  return std::make_pair(detections, layers);
}

}  // namespace OmegaEngine

#endif  // USE_FCL
