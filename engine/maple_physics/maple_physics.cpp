#include "maple_physics.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "../maple_logging/log_macros.h"
#include "box3d/base.h"
#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/id.h"
#include "box3d/math_functions.h"
#include "box3d/types.h"

namespace maple {

b3Vec3 toB3Vec3(const glm::vec3& v) { return b3Vec3{.x = v.x, .y = v.y, .z = v.z}; }
b3Pos toB3Vec3(const glm::dvec3& v) { return b3Pos{.x = v.x, .y = v.y, .z = v.z}; }
glm::vec3 vec3FromB3(const b3Vec3& v) { return glm::vec3(v.x, v.y, v.z); }
glm::dvec3 vec3FromB3(const b3Pos& v) { return glm::dvec3(v.x, v.y, v.z); }
b3Quat toB3Quat(const glm::quat& v) { return b3Quat{.v = {.x = v.x, .y = v.y, .z = v.z}, .s = v.w}; }
glm::quat quatFromB3(const b3Quat& v) { return glm::quat(v.s, v.v.x, v.v.y, v.v.z); }
b3BodyId toB3BodyId(Physics::BodyID id) { return std::bit_cast<b3BodyId>(id.id); }
Physics::BodyID BodyIdFromB3(b3BodyId id) { return std::bit_cast<Physics::BodyID>(id); }
void* toB3UserData(Physics::EntityID v) { return reinterpret_cast<void*>(v); }
Physics::EntityID userDataFromB3(void* v) { return reinterpret_cast<Physics::EntityID>(v); }

struct Physics::Impl {
  b3WorldId mWorldId = b3_nullWorldId;
  std::unordered_map<uint64_t, std::vector<std::variant<b3HullData*, b3MeshData*, b3HeightFieldData*>>> mBodyShapeAllocs;
};

Physics::Physics() = default;
Physics::~Physics() { Destroy(); }

Physics::Physics(Physics&&) noexcept = default;
Physics& Physics::operator=(Physics&&) noexcept = default;

Physics::Physics(const CreateInfo& info) : impl(std::make_unique<Impl>()) {
  b3WorldDef worldDef = b3DefaultWorldDef();
  worldDef.gravity = std::bit_cast<b3Vec3>(info.gravity);
  worldDef.hitEventThreshold = info.hitEventThreshold;
  worldDef.workerCount = std::thread::hardware_concurrency();

  impl->mWorldId = b3CreateWorld(&worldDef);
}

void Physics::Destroy() {
  if (!impl) return;
  if (B3_IS_NULL(impl->mWorldId)) return;
  b3DestroyWorld(impl->mWorldId);
  impl->mWorldId = b3_nullWorldId;
}

void Physics::Update(float timeStep, uint32_t subStepCount) {
  MAPLE_ASSERT(B3_IS_NON_NULL(impl->mWorldId), "physics world id was null");
  b3World_Step(impl->mWorldId, timeStep, subStepCount);
}

Physics::BodyID Physics::CreateBody(const BodyInfo& info) {
  MAPLE_ASSERT(B3_IS_NON_NULL(impl->mWorldId), "physics world id was null");
  if (!info.Validate()) MAPLE_FATAL("invalid physics body info");

  auto bodyDef = b3DefaultBodyDef();
  bodyDef.userData = toB3UserData(info.entityID);
  bodyDef.position = b3Pos{.x = static_cast<typeof(b3Pos::x)>(info.position.x),
                           .y = static_cast<typeof(b3Pos::y)>(info.position.y),
                           .z = static_cast<typeof(b3Pos::z)>(info.position.z)};
  bodyDef.rotation = toB3Quat(info.orientation);
  bodyDef.linearDamping = info.linearDamping;
  bodyDef.angularDamping = info.angularDamping;
  bodyDef.type =
    info.motionType == MotionType::Static ? b3_staticBody : (info.motionType == MotionType::Kinematic ? b3_kinematicBody : b3_dynamicBody);
  bodyDef.isBullet = info.motionQuality == MotionQuality::Continuous;

  auto bodyID = b3CreateBody(impl->mWorldId, &bodyDef);

  for (auto& v : info.shapes) {
    b3ShapeDef shapeDef = b3DefaultShapeDef();
    shapeDef.baseMaterial.friction = v.friction;
    shapeDef.baseMaterial.restitution = v.restitution;
    shapeDef.enableHitEvents = v.enableHitEvents;
    shapeDef.enableSensorEvents = v.enableSensorEvents;
    shapeDef.isSensor = v.isSensor;
    if (v.isSensor && !v.enableSensorEvents) MAPLE_WARN("physics shape created as sensor but doesn't have sensor events enabled");
    shapeDef.filter.categoryBits = v.categoryBits;
    shapeDef.filter.maskBits = v.maskBits;

    auto& shape = v.shape;

    if (auto* v = std::get_if<Box>(&shape)) {
      auto hull = b3MakeBoxHull(v->halfExtent.x, v->halfExtent.y, v->halfExtent.z);
      b3CreateHullShape(bodyID, &shapeDef, &hull.base);
      break;
    }

    if (auto* v = std::get_if<Sphere>(&shape)) {
      b3Sphere sphere{b3Vec3{0.0f, 0.0f, 0.0f}, v->radius};
      b3CreateSphereShape(bodyID, &shapeDef, &sphere);
      break;
    }

    if (auto v = std::get_if<Cylinder>(&shape)) {
      MAPLE_FATAL("unimplmeneted physics shape");
      auto cylinder = b3CreateCylinder(v->height, v->radius, v->yOffset, v->sides);
      MAPLE_ASSERT(cylinder, "invalid cylinder hull shape");
      b3CreateHullShape(bodyID, &shapeDef, cylinder);

      auto it = impl->mBodyShapeAllocs.find(std::bit_cast<uint64_t>(bodyID));
      if (it == impl->mBodyShapeAllocs.end()) {
        impl->mBodyShapeAllocs[std::bit_cast<uint64_t>(bodyID)] = {cylinder};
      } else {
        it->second.push_back(cylinder);
      }

      break;
    }

    if (auto v = std::get_if<ConvexHull>(&shape)) {
      auto hull = b3CreateHull(reinterpret_cast<const b3Vec3*>(v->points.data()), v->points.size(), v->points.size());
      MAPLE_ASSERT(hull, "invalid convex hull shape");
      b3CreateHullShape(bodyID, &shapeDef, hull);

      auto it = impl->mBodyShapeAllocs.find(std::bit_cast<uint64_t>(bodyID));
      if (it == impl->mBodyShapeAllocs.end()) {
        impl->mBodyShapeAllocs[std::bit_cast<uint64_t>(bodyID)] = {hull};
      } else {
        it->second.push_back(hull);
      }

      break;
    }

    if (auto v = std::get_if<Mesh>(&shape)) {
      b3MeshDef def = {};
      // doing this copy because the box3d pointers are non-const
      auto vertices = std::make_unique<b3Vec3[]>(v->vertices.size());
      std::memcpy(vertices.get(), v->vertices.data(), v->vertices.size() * sizeof(decltype(v->vertices)::value_type));
      auto indices = std::make_unique<int32_t[]>(v->indices.size());
      std::memcpy(indices.get(), v->indices.data(), v->indices.size() * sizeof(decltype(v->indices)::value_type));

      def.vertices = vertices.get();
      def.vertexCount = v->vertices.size();
      def.indices = indices.get();  // 3 per triangle
      def.triangleCount = v->indices.size() / 3;
      def.weldVertices = true;
      def.identifyEdges = true;  // adjacency info for smooth inter-triangle normals

      b3MeshData* mesh = b3CreateMesh(&def, NULL, 0);
      MAPLE_ASSERT(mesh, "invalid mesh shape");
      b3ShapeId meshID = b3CreateMeshShape(bodyID, &shapeDef, mesh, {1.0f, 1.0f, 1.0f});

      auto it = impl->mBodyShapeAllocs.find(std::bit_cast<uint64_t>(bodyID));
      if (it == impl->mBodyShapeAllocs.end()) {
        impl->mBodyShapeAllocs[std::bit_cast<uint64_t>(bodyID)] = {mesh};
      } else {
        it->second.push_back(mesh);
      }

      break;
    }

    if (auto v = std::get_if<HeightField>(&shape)) {
      b3HeightFieldDef def = {0};
      def.heights = v->heights.get();
      def.countX = v->sizeX;
      def.countZ = v->sizeZ;
      def.scale = (b3Vec3){1.0f, 1.0f, 1.0f};
      def.globalMinimumHeight = -10.0f;
      def.globalMaximumHeight = 50.0f;

      b3HeightFieldData* hf = b3CreateHeightField(&def);
      b3ShapeId id = b3CreateHeightFieldShape(bodyID, &shapeDef, hf);

      auto it = impl->mBodyShapeAllocs.find(std::bit_cast<uint64_t>(bodyID));
      if (it == impl->mBodyShapeAllocs.end()) {
        impl->mBodyShapeAllocs[std::bit_cast<uint64_t>(bodyID)] = {hf};
      } else {
        it->second.push_back(hf);
      }

      break;
    }
  }

  return {.id = std::bit_cast<uint64_t>(bodyID)};
}

void Physics::DestroyBody(BodyID id) {
  MAPLE_ASSERT(B3_IS_NON_NULL(impl->mWorldId), "physics world id was null");

  b3DestroyBody(toB3BodyId(id));

  auto meshIt = impl->mBodyShapeAllocs.find(id.id);
  if (meshIt != impl->mBodyShapeAllocs.end()) {
    for (auto ptr : meshIt->second) {
      if (auto v = std::get_if<b3HullData*>(&ptr)) {
        b3DestroyHull(*v);
        continue;
      }

      if (auto v = std::get_if<b3MeshData*>(&ptr)) {
        b3DestroyMesh(*v);
        continue;
      }

      if (auto v = std::get_if<b3HeightFieldData*>(&ptr)) {
        b3DestroyHeightField(*v);
        continue;
      }
    }

    impl->mBodyShapeAllocs.erase(meshIt);
  }
}

Physics::EntityID Physics::GetBodyEntity(BodyID id) { return userDataFromB3(b3Body_GetUserData(toB3BodyId(id))); }
std::pair<glm::dvec3, glm::quat> Physics::GetBodyTransform(BodyID id) const {
  auto transform = b3Body_GetTransform(toB3BodyId(id));
  return std::make_pair(vec3FromB3(transform.p), quatFromB3(transform.q));
}
void Physics::SetBodyTransform(BodyID id, const glm::dvec3& pos, const glm::quat& quat) {
  b3Body_SetTransform(toB3BodyId(id), toB3Vec3(pos), toB3Quat(quat));
}

void Physics::ApplyForce(BodyID id, const glm::vec3& force) { b3Body_ApplyForceToCenter(toB3BodyId(id), toB3Vec3(force), true); }
void Physics::ApplyForceAtPosition(BodyID id, const glm::vec3& force, const glm::dvec3& worldPoint) {
  b3Body_ApplyForce(toB3BodyId(id), toB3Vec3(force), toB3Vec3(worldPoint), true);
}

std::optional<Physics::CastResult> Physics::CastRay(const CastInfo& info) const {
  MAPLE_ASSERT(B3_IS_NON_NULL(impl->mWorldId), "physics world id was null");
  auto result = b3World_CastRayClosest(
    impl->mWorldId, toB3Vec3(info.origin), toB3Vec3(info.translation), b3QueryFilter{.categoryBits = info.categoryBits, .maskBits = info.maskBits});
  if (!result.hit) return std::nullopt;

  return CastResult{
    .bodyID = BodyIdFromB3(b3Shape_GetBody(result.shapeId)),
    .position = vec3FromB3(result.point),
    .normal = vec3FromB3(result.normal),
  };
}

struct CastContext {
  b3ShapeId shapeId;
  b3Pos point;
  b3Vec3 normal;
  float fraction;
};

float ShapeCastCallback(
  b3ShapeId shapeId, b3Pos point, b3Vec3 normal, float fraction, uint64_t userMaterialId, int triangleIndex, int childIndex, void* context) {
  CastContext* castCtx = reinterpret_cast<CastContext*>(context);
  castCtx->shapeId = shapeId;
  castCtx->point = point;
  castCtx->normal = normal;
  castCtx->fraction = fraction;
  return fraction;
}

std::optional<Physics::CastResult> Physics::CastShape(const ShapeCastInfo& info) const {
  b3ShapeProxy proxy;
  proxy.radius = info.pointsRadii;
  proxy.count = info.points.size();
  proxy.points = reinterpret_cast<const b3Vec3*>(info.points.data());

  CastContext ctx{0};

  b3World_CastShape(impl->mWorldId,
                    toB3Vec3(info.origin),
                    &proxy,
                    toB3Vec3(info.translation),
                    b3QueryFilter{.categoryBits = info.categoryBits, .maskBits = info.maskBits},
                    ShapeCastCallback,
                    &ctx);

  if (B3_IS_NULL(ctx.shapeId)) return std::nullopt;

  return CastResult{
    .bodyID = BodyIdFromB3(b3Shape_GetBody(ctx.shapeId)),
    .position = vec3FromB3(ctx.point),
    .normal = vec3FromB3(ctx.normal),
  };
}

bool OverlapCallback(b3ShapeId shapeId, void* context) {
  b3BodyId bodyId = b3Shape_GetBody(shapeId);
  b3Body_SetAwake(bodyId, true);

  std::vector<Physics::BodyID>* entities = reinterpret_cast<std::vector<Physics::BodyID>*>(context);

  entities->push_back(BodyIdFromB3(bodyId));

  // Return true to continue the query.
  return true;
}

std::vector<Physics::BodyID> Physics::OverlapShape(const Physics::OverlapShapeInfo& info) const {
  std::vector<Physics::BodyID> entities;

  b3ShapeProxy proxy;
  proxy.radius = info.pointsRadii;
  proxy.count = info.points.size();
  proxy.points = reinterpret_cast<const b3Vec3*>(info.points.data());

  b3World_OverlapShape(impl->mWorldId,
                       toB3Vec3(info.origin),
                       &proxy,
                       b3QueryFilter{.categoryBits = info.categoryBits, .maskBits = info.maskBits},
                       OverlapCallback,
                       &entities);

  return entities;
}

std::vector<Physics::BodyID> Physics::OverlapSphere(const OverlapInfo& info, float radius) const {
  std::array<glm::vec3, 1> point = {glm::vec3(0)};
  OverlapShapeInfo shapeInfo{};

  shapeInfo.origin = info.origin;
  shapeInfo.categoryBits = info.categoryBits;
  shapeInfo.maskBits = info.maskBits;
  shapeInfo.points = point;
  shapeInfo.pointsRadii = radius;

  return OverlapShape(shapeInfo);
}

std::vector<Physics::ContactHitEvent> Physics::GetGlobalContactHitEvents() const {
  b3ContactEvents contactEvents = b3World_GetContactEvents(impl->mWorldId);

  std::vector<ContactHitEvent> events(contactEvents.hitCount);

  for (uint32_t i = 0; i < contactEvents.hitCount; i++) {
    events[i] = {
      .a = userDataFromB3(b3Body_GetUserData(b3Shape_GetBody(contactEvents.hitEvents[i].shapeIdA))),
      .b = userDataFromB3(b3Body_GetUserData(b3Shape_GetBody(contactEvents.hitEvents[i].shapeIdB))),
      .worldPoint = vec3FromB3(contactEvents.hitEvents[i].point),
      .normal = vec3FromB3(contactEvents.hitEvents[i].normal),
      .approachSpeed = contactEvents.hitEvents[i].approachSpeed,
    };
  }

  return events;
}

}  // namespace maple