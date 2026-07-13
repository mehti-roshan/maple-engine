#include "maple_physics.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

#include "../maple_logging/log_macros.h"
#include "Jolt/Jolt.h"
#include "Jolt/Core/Factory.h"
#include "Jolt/Core/JobSystemThreadPool.h"
#include "Jolt/Core/Reference.h"
#include "Jolt/Geometry/Plane.h"
#include "Jolt/Math/DMat44.h"
#include "Jolt/Math/MathTypes.h"
#include "Jolt/Math/Real.h"
#include "Jolt/Math/Vec3.h"
#include "Jolt/Physics/Body/Body.h"
#include "Jolt/Physics/Body/BodyCreationSettings.h"
#include "Jolt/Physics/Body/BodyInterface.h"
#include "Jolt/Physics/Body/BodyLock.h"
#include "Jolt/Physics/Collision/CastResult.h"
#include "Jolt/Physics/Collision/CollideShape.h"
#include "Jolt/Physics/Collision/RayCast.h"
#include "Jolt/Physics/Collision/Shape/BoxShape.h"
#include "Jolt/Physics/Collision/Shape/CapsuleShape.h"
#include "Jolt/Physics/Collision/Shape/ConvexHullShape.h"
#include "Jolt/Physics/Collision/Shape/CylinderShape.h"
#include "Jolt/Physics/Collision/Shape/HeightFieldShape.h"
#include "Jolt/Physics/Collision/Shape/MeshShape.h"
#include "Jolt/Physics/Collision/Shape/PlaneShape.h"
#include "Jolt/Physics/Collision/Shape/Shape.h"
#include "Jolt/Physics/Collision/Shape/SphereShape.h"
#include "Jolt/Physics/Collision/Shape/StaticCompoundShape.h"
#include "Jolt/Physics/Collision/Shape/TaperedCapsuleShape.h"
#include "Jolt/Physics/Collision/Shape/TaperedCylinderShape.h"
#include "Jolt/Physics/Collision/Shape/TriangleShape.h"
#include "Jolt/Physics/PhysicsSystem.h"
#include "Jolt/RegisterTypes.h"
#include "helpers.h"

namespace maple {

namespace Layers {
static constexpr JPH::ObjectLayer NON_MOVING = 0;
static constexpr JPH::ObjectLayer MOVING = 1;
static constexpr uint32_t NUM_LAYERS = 2;

}  // namespace Layers

class MapleBroadPhaseLayerInterface final : public JPH::BroadPhaseLayerInterface {
 public:
  MapleBroadPhaseLayerInterface() {
    mObjectToBroadPhase[Layers::NON_MOVING] = JPH::BroadPhaseLayer(0);

    mObjectToBroadPhase[Layers::MOVING] = JPH::BroadPhaseLayer(1);
  }

  uint GetNumBroadPhaseLayers() const override { return 2; }

  JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override { return mObjectToBroadPhase[layer]; }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)

  const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
    switch (static_cast<uint8_t>(layer)) {
      case 0:
        return "STATIC";

      case 1:
        return "DYNAMIC";

      default:
        return "UNKNOWN";
    }
  }

#endif

 private:
  JPH::BroadPhaseLayer mObjectToBroadPhase[Layers::NUM_LAYERS];
};

struct MaplePhysics::Impl {
  JPH::PhysicsSystem physicsSystem;

  JPH::BodyInterface* bodyInterface = nullptr;

  std::unique_ptr<JPH::ObjectVsBroadPhaseLayerFilter> objectVsBroadPhaseLayerFilter;
  std::unique_ptr<JPH::ObjectLayerPairFilter> objectLayerPairFilter;

  std::unique_ptr<MapleBroadPhaseLayerInterface> broadPhaseLayerInterface;

  std::unique_ptr<JPH::TempAllocatorImpl> tempAllocator;
  std::unique_ptr<JPH::JobSystemThreadPool> jobSystem;

  bool initialized = false;
};

MaplePhysics::MaplePhysics() : impl(std::make_unique<Impl>()) {}

MaplePhysics::~MaplePhysics() { Shutdown(); }

bool MaplePhysics::Initialize(const glm::vec3& gravity) {
  // Jolt initialization
  JPH::RegisterDefaultAllocator();
  JPH::Factory::sInstance = new JPH::Factory();
  JPH::RegisterTypes();

  impl->broadPhaseLayerInterface = std::make_unique<MapleBroadPhaseLayerInterface>();

  impl->objectVsBroadPhaseLayerFilter = std::make_unique<JPH::ObjectVsBroadPhaseLayerFilter>();

  impl->objectLayerPairFilter = std::make_unique<JPH::ObjectLayerPairFilter>();

  // Temp values for now
  // TODO: make values available as initialization struct
  constexpr uint cMaxBodies = 1024 * 1024;
  constexpr uint cNumBodyMutexes = 0;
  constexpr uint cMaxBodyPairs = 64 * 1024;
  constexpr uint cMaxContactConstraints = 64 * 1024;
  constexpr uint cTempAllocatorSize = 100 * 1024 * 1024;

  impl->physicsSystem.Init(cMaxBodies,
                           cNumBodyMutexes,
                           cMaxBodyPairs,
                           cMaxContactConstraints,
                           *impl->broadPhaseLayerInterface,
                           *impl->objectVsBroadPhaseLayerFilter,
                           *impl->objectLayerPairFilter);

  impl->physicsSystem.SetGravity(hlp::ToJolt(gravity));

  impl->bodyInterface = &impl->physicsSystem.GetBodyInterface();

  impl->tempAllocator = std::make_unique<JPH::TempAllocatorImpl>(cTempAllocatorSize);

  const uint maxJobs = 1024;
  const uint maxBarriers = 1024;

  impl->jobSystem = std::make_unique<JPH::JobSystemThreadPool>(maxJobs, maxBarriers, std::thread::hardware_concurrency() - 1);

  impl->initialized = true;

  return true;
}

void MaplePhysics::Shutdown() {
  if (!impl || !impl->initialized) return;

  impl->initialized = false;

  JPH::UnregisterTypes();

  delete JPH::Factory::sInstance;
  JPH::Factory::sInstance = nullptr;
}

void MaplePhysics::Update(float deltaTime) {
  if (!impl->initialized) return;

  constexpr int collisionSteps = 1;

  impl->physicsSystem.Update(deltaTime, collisionSteps, impl->tempAllocator.get(), impl->jobSystem.get());
}

JPH::Ref<JPH::Shape> constructShape(MaplePhysics::CollisionShape& shape) {
  if (auto v = std::get_if<std::unique_ptr<MaplePhysics::CompoundShape>>(&shape)) {
    auto settings = JPH::StaticCompoundShapeSettings();
    for (auto& child : v->get()->children) {
      settings.AddShape(hlp::ToJolt(child.position), hlp::ToJolt(child.orientation), constructShape(child.shape));
    }

    auto result = settings.Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (auto v = std::get_if<MaplePhysics::Sphere>(&shape)) {
    auto result = JPH::SphereShapeSettings(v->radius).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (auto v = std::get_if<MaplePhysics::Box>(&shape)) {
    auto result = JPH::BoxShapeSettings(hlp::ToJolt(v->halfExtent)).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (auto v = std::get_if<MaplePhysics::Capsule>(&shape)) {
    auto result = JPH::CapsuleShapeSettings(v->halfHeight, v->radius).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (auto v = std::get_if<MaplePhysics::TaperedCapsule>(&shape)) {
    auto result = JPH::TaperedCapsuleShapeSettings(v->halfHeight, v->topRadius, v->bottomRadius).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (auto v = std::get_if<MaplePhysics::Cylinder>(&shape)) {
    auto result = JPH::CylinderShapeSettings(v->halfHeight, v->radius).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (auto v = std::get_if<MaplePhysics::TaperedCylinder>(&shape)) {
    auto result = JPH::TaperedCylinderShapeSettings(v->halfHeight, v->topRadius, v->bottomRadius).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (auto v = std::get_if<MaplePhysics::ConvexHull>(&shape)) {
    auto result = JPH::ConvexHullShapeSettings(reinterpret_cast<JPH::Vec3*>(v->points.data()), v->points.size()).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (auto v = std::get_if<MaplePhysics::Triangle>(&shape)) {
    auto result = JPH::TriangleShapeSettings(hlp::ToJolt(v->a), hlp::ToJolt(v->b), hlp::ToJolt(v->c)).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (auto v = std::get_if<MaplePhysics::Plane>(&shape)) {
    auto result = JPH::PlaneShapeSettings(JPH::Plane(hlp::ToJolt(v->normal), v->distance)).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (MaplePhysics::HeightField* v = std::get_if<MaplePhysics::HeightField>(&shape)) {
    auto result = JPH::HeightFieldShapeSettings(v->heights.get(), hlp::ToJolt(glm::vec3(0)), hlp::ToJolt(glm::vec3(1)), v->N).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  if (MaplePhysics::Mesh* v = std::get_if<MaplePhysics::Mesh>(&shape)) {
    JPH::VertexList vertices;
    vertices.reserve(v->vertices.size());

    for (auto& v : v->vertices) vertices.emplace_back(v.x, v.y, v.z);

    JPH::IndexedTriangleList triangles;
    triangles.reserve(v->indices.size() / 3);

    for (size_t i = 0; i < v->indices.size(); i += 3) triangles.emplace_back(v->indices[i], v->indices[i + 1], v->indices[i + 2]);

    auto result = JPH::MeshShapeSettings(vertices, triangles).Create();
    if (result.HasError()) MAPLE_FATAL("failed to create physics shape: '{}'", result.GetError());
    return JPH::Ref<JPH::Shape>(result.Get());
  }

  MAPLE_FATAL("unknown Collision Shape type");
}

PhysicsBodyID MaplePhysics::CreateRigidBody(BodyInfo& info) {
  if (!impl->initialized) return 0;
  if (!info.Validate()) MAPLE_FATAL("invalid physics body info");
  bool isStaticShape = std::holds_alternative<MaplePhysics::Mesh>(info.shape) || std::holds_alternative<MaplePhysics::HeightField>(info.shape);
  if (isStaticShape && info.motionType != MotionType::Static) MAPLE_FATAL("mesh shape cannot be used on non-static geometry");

  auto shape = constructShape(info.shape);

  JPH::BodyCreationSettings settings(shape, JPH::RVec3(0, 0, 0), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);

  settings.mUserData = info.entityID;
  settings.mMotionType = hlp::ToJolt(info.motionType);
  settings.mMotionQuality = hlp::ToJolt(info.motionQuality);
  settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
  settings.mMassPropertiesOverride.mMass = info.mass;
  settings.mInertiaMultiplier = info.intertiaMultiplier;
  settings.mFriction = info.friction;
  settings.mRestitution = info.restitution;
  settings.mLinearDamping = info.linearDamping;
  settings.mAngularDamping = info.angularDamping;

  settings.mPosition = hlp::ToJolt(info.position);
  settings.mRotation = hlp::ToJolt(info.orientation);

  JPH::Body* body = impl->bodyInterface->CreateBody(settings);

  if (!body) return 0;

  impl->bodyInterface->AddBody(body->GetID(), JPH::EActivation::Activate);

  return body->GetID().GetIndexAndSequenceNumber();
}

void MaplePhysics::DestroyRigidBody(PhysicsBodyID id) {
  if (!impl->initialized) return;

  JPH::BodyID bodyID(id);

  impl->bodyInterface->RemoveBody(bodyID);
  impl->bodyInterface->DestroyBody(bodyID);
}

uint64_t MaplePhysics::GetBodyEntity(PhysicsBodyID id) {
  return impl->bodyInterface->GetUserData(static_cast<JPH::BodyID>(id));
}

glm::vec3 MaplePhysics::GetBodyPosition(PhysicsBodyID id) const {
  JPH::RVec3 v = impl->bodyInterface->GetPosition(static_cast<JPH::BodyID>(id));
  return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
}

glm::quat MaplePhysics::GetBodyRotation(PhysicsBodyID id) const {
  JPH::Quat v = impl->bodyInterface->GetRotation(static_cast<JPH::BodyID>(id));
  return glm::quat(v.GetW(), v.GetX(), v.GetY(), v.GetZ());
}

void MaplePhysics::SetBodyPosition(PhysicsBodyID id, const glm::vec3& pos) {
  JPH::BodyID bodyID(id);

  impl->bodyInterface->SetPosition(bodyID, JPH::RVec3(pos.x, pos.y, pos.z), JPH::EActivation::Activate);
}

void MaplePhysics::SetBodyRotation(PhysicsBodyID id, const glm::quat& quat) {
  JPH::BodyID bodyID(id);

  impl->bodyInterface->SetRotation(bodyID, JPH::Quat(quat.x, quat.y, quat.z, quat.w), JPH::EActivation::Activate);
}

void MaplePhysics::ApplyForce(PhysicsBodyID id, const glm::vec3& force) {
  JPH::BodyID bodyID(id);

  impl->bodyInterface->ActivateBody(bodyID);
  impl->bodyInterface->AddForce(bodyID, JPH::Vec3(force.x, force.y, force.z));
}

std::optional<MaplePhysics::RayCastResult> MaplePhysics::Raycast(const glm::vec3& origin, const glm::vec3& dir, float distance) {
  JPH::RayCastResult result;
  JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z), JPH::Vec3(dir.x, dir.y, dir.z) * distance);
  auto& query = impl->physicsSystem.GetNarrowPhaseQuery();
  bool didHit = query.CastRay(ray, result);  // TODO: add filters for which things will be collided with

  if (!didHit) return std::nullopt;

  JPH::RVec3 hitPositionJolt = ray.GetPointOnRay(result.mFraction);

  return MaplePhysics::RayCastResult{
    .bodyID = result.mBodyID.GetIndexAndSequenceNumber(),
    .position = glm::vec3(hitPositionJolt.GetX(), hitPositionJolt.GetY(), hitPositionJolt.GetZ()),
  };
}

std::optional<MaplePhysics::RayCastResultWithNormal> MaplePhysics::RaycastWNormal(const glm::vec3& origin, const glm::vec3& dir, float distance) {
  JPH::RayCastResult result;
  JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z), JPH::Vec3(dir.x, dir.y, dir.z) * distance);
  auto& query = impl->physicsSystem.GetNarrowPhaseQuery();
  bool didHit = query.CastRay(ray, result);  // TODO: add filters for which things will be collided with

  if (!didHit) return std::nullopt;

  JPH::RVec3 hitPositionJolt = ray.GetPointOnRay(result.mFraction);

  JPH::BodyLockRead lock(impl->physicsSystem.GetBodyLockInterface(), result.mBodyID);

  if (!lock.Succeeded()) MAPLE_FATAL("failed to get lock on body");
  const JPH::Body& body = lock.GetBody();

  JPH::Vec3 normal = body.GetWorldSpaceSurfaceNormal(result.mSubShapeID2, hitPositionJolt);

  return MaplePhysics::RayCastResultWithNormal{
    .bodyID = result.mBodyID.GetIndexAndSequenceNumber(),
    .position = glm::vec3(hitPositionJolt.GetX(), hitPositionJolt.GetY(), hitPositionJolt.GetZ()),
    .normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ()),
  };
}

std::vector<PhysicsBodyID> MaplePhysics::OverlapSphere(Sphere shape, const glm::vec3& origin) {
  std::vector<PhysicsBodyID> result;

  MaplePhysics::CollisionShape sh = shape;
  auto joltShape = constructShape(sh);

  class Collector : public JPH::CollideShapeCollector {
   public:
    std::vector<PhysicsBodyID>& results;

    Collector(std::vector<PhysicsBodyID>& results) : results(results) {}

    void AddHit(const JPH::CollideShapeResult& result) override { results.push_back(result.mBodyID2.GetIndexAndSequenceNumber()); }
  };

  Collector collector(result);

  JPH::RMat44 CoMTrans = JPH::RMat44::sIdentity();
  CoMTrans.SetTranslation(JPH::RVec3Arg(origin.x, origin.y, origin.z)); // TODO: fix
  JPH::CollideShapeSettings settings;
  
  auto& query = impl->physicsSystem.GetNarrowPhaseQuery();
  query.CollideShape(joltShape, JPH::Vec3::sReplicate(1.0f), CoMTrans, settings, JPH::RVec3::sZero(), collector);

  return result;
}

}  // namespace maple