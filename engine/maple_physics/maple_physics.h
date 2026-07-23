#pragma once

#include <cstdint>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>
#include <vector>

namespace maple {

class Physics {
 public:
  using BodyID = uint32_t;
  Physics();
  ~Physics();

  Physics(const Physics&) = delete;
  Physics& operator=(const Physics&) = delete;

  bool Initialize(const glm::vec3& gravity);
  void Shutdown();

  void Update(float deltaTime);

  struct Sphere {
    float radius = 0.5f;
  };

  struct Box {
    glm::vec3 halfExtent = glm::vec3(0.5f);
  };

  struct Capsule {
    float halfHeight = 0.5f;
    float radius = 0.5f;
  };

  struct TaperedCapsule {
    float halfHeight = 0.5f;
    float topRadius = 0.5f;
    float bottomRadius = 0.5f;
  };

  struct Cylinder {
    float halfHeight = 0.5f;
    float radius = 0.5f;
  };

  struct TaperedCylinder {
    float halfHeight = 0.5f;
    float topRadius = 0.5f;
    float bottomRadius = 0.5f;
  };

  struct ConvexHull {
    std::vector<glm::vec3> points;
  };

  struct Triangle {
    glm::vec3 a;
    glm::vec3 b;
    glm::vec3 c;
  };

  struct Plane {
    glm::vec3 normal = glm::vec3(0, 1, 0);
    float distance = 0.0f;
  };

  // Only usable with static body type
  struct Mesh {
    std::vector<glm::vec3> vertices;
    std::vector<uint32_t> indices;
  };

  // Only usable with static body type
  // Grid is NxN sized
  struct HeightField {
    uint32_t N = 0;
    std::unique_ptr<float> heights;
  };

  struct CompoundShape;

  using CollisionShape = std::variant<Sphere,
                                      Box,
                                      Capsule,
                                      TaperedCapsule,
                                      Cylinder,
                                      TaperedCylinder,
                                      ConvexHull,
                                      Triangle,
                                      Plane,
                                      Mesh,
                                      HeightField,
                                      std::unique_ptr<CompoundShape>>;

  struct CompoundShape {
    struct Child {
      glm::vec3 position{};
      glm::quat orientation = glm::identity<glm::quat>();
      CollisionShape shape;
    };
    std::vector<Child> children;
  };

  enum MotionType { Static, Dynamic, Kinematic };
  enum MotionQuality { Discrete, Continuous };

  struct BodyInfo {
    uint32_t entityID = 0;
    CollisionShape shape;
    MotionType motionType = Static;
    MotionQuality motionQuality = Discrete;

    glm::dvec3 position = glm::vec3(0);
    glm::quat orientation = glm::identity<glm::quat>();

    float mass = 1.0f;
    float intertiaMultiplier = 1.0f;  // > 0, larger than 1 makes the body harder to rotate, less than 1 vice versa

    float friction = 0.5f;     // 0 -> 1
    float restitution = 0.5f;  // 0 -> 1, how bouncy the collision is, 0: full bounce, 1: no bounce

    float linearDamping = 0.0f;   // > 0, how quickly velocity slows over time
    float angularDamping = 0.0f;  // > 0, how quickly angular velocity (rotation) slows over time

    bool Validate() const {
      if (mass <= 0.0f) return false;
      if (intertiaMultiplier <= 0.0f) return false;
      if (friction > 1.0f) return false;
      if (friction < 0.0f) return false;
      if (restitution > 1.0f) return false;
      if (restitution < 0.0f) return false;
      if (linearDamping < 0.0f) return false;
      if (angularDamping < 0.0f) return false;
      return true;
    }
  };

  [[nodiscard]]
  BodyID CreateRigidBody(BodyInfo& info);
  void DestroyRigidBody(BodyID id);

  uint64_t GetBodyEntity(BodyID id);

  glm::vec3 GetBodyPosition(BodyID id) const;

  glm::quat GetBodyRotation(BodyID id) const;

  void SetBodyPosition(BodyID id, const glm::vec3& pos);

  void SetBodyRotation(BodyID id, const glm::quat& quat);

  void ApplyForce(BodyID id, const glm::vec3& force);

  struct RayCastResult {
    BodyID bodyID = 0;
    glm::vec3 position{};
  };

  struct RayCastResultWithNormal {
    BodyID bodyID = 0;
    glm::vec3 position{};
    glm::vec3 normal{};
  };

  std::optional<RayCastResult> Raycast(const glm::vec3& origin, const glm::vec3& dir, float distance);
  std::optional<RayCastResultWithNormal> RaycastWNormal(const glm::vec3& origin, const glm::vec3& dir, float distance);

  std::vector<BodyID> OverlapSphere(Sphere sphere, const glm::vec3& origin);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

}  // namespace maple