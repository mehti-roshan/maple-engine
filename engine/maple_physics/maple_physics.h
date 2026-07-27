#pragma once

#include <cstdint>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace maple {

class Physics {
 public:
  struct BodyID {
    static constexpr auto INVALID_ID = 0;
    uint64_t id = INVALID_ID;
  };

  using EntityID = uint64_t;

  struct CreateInfo {
    glm::vec3 gravity = glm::vec3(0, -9.81, 0);
    float hitEventThreshold = 1.0f;  // speed (m/s) above which colliding shapes (that have enableHitEvents) will generate hit events
  };

  struct Sphere {
    float radius = 0.5f;
  };

  struct Box {
    glm::vec3 halfExtent = glm::vec3(0.5f);
  };

  struct Cylinder {
    float height;
    float radius;
    uint32_t sides;
    float yOffset = 0.0f;
  };

  struct ConvexHull {
    std::vector<glm::vec3> points;
  };

  // Only usable with static body type
  struct Mesh {
    std::vector<glm::vec3> vertices;
    std::vector<uint32_t> indices;
  };

  // Only usable with static body type
  // Grid is NxN sized
  struct HeightField {
    uint32_t sizeX = 0;
    uint32_t sizeZ = 0;
    float globalMinimumHeight;
    float globalMaximumHeight;
    std::unique_ptr<float> heights;
  };

  using CategoryBits = uint64_t;
  using MaskBits = uint64_t;
  static constexpr CategoryBits DefaultCategoryBits = std::numeric_limits<uint64_t>::max();
  static constexpr MaskBits DefaultMaskBits = std::numeric_limits<uint64_t>::max();

  struct CollisionShape {
    std::variant<Sphere, Box, Cylinder, ConvexHull, Mesh, HeightField> shape;
    float density = 1.0f;
    float friction = 0.5f;                            // 0 -> 1
    float restitution = 0.5f;                         // 0 -> 1, how bouncy the collision is, 0: full bounce, 1: no bounce
    bool enableHitEvents = false;                     // required if shape is to generate contact hit events
    bool enableSensorEvents = false;                  // required on sensor shapes and shapes that trigger sensors in order to generate sensor events
    bool isSensor = false;                            // detects overlap but provides no collision response
    CategoryBits categoryBits = DefaultCategoryBits;  // collision layer categories of this shape
    MaskBits maskBits = DefaultMaskBits;              // mask of collision layers this shape will accept for collision
  };

  enum MotionType : uint8_t { Static, Dynamic, Kinematic };
  enum MotionQuality : uint8_t { Discrete, Continuous };

#ifdef MAPLE_PHYSICS_DOUBLE_PRECISION
  using WorldPos = glm::dvec3;
#else
  using WorldPos = glm::vec3;
#endif

  struct BodyInfo {
    EntityID entityID;
    std::span<const CollisionShape> shapes;
    MotionType motionType = Static;
    MotionQuality motionQuality = Discrete;

    WorldPos position = WorldPos(0);
    glm::quat orientation = glm::identity<glm::quat>();

    float linearDamping = 0.0f;   // > 0, how quickly velocity slows over time
    float angularDamping = 0.0f;  // > 0, how quickly angular velocity (rotation) slows over time

    bool Validate() const {
      if (linearDamping < 0.0f) return false;
      if (angularDamping < 0.0f) return false;
      if (motionType == Dynamic) {
        bool hasOneShapeWNonZeroDensity = false;
        for (auto& v : shapes) {
          if (v.friction < 0.0f || v.friction > 1.0f) return false;
          if (v.restitution < 0.0f || v.restitution > 1.0f) return false;
          if (motionType != Static && (std::holds_alternative<Mesh>(v.shape) || std::holds_alternative<HeightField>(v.shape))) return false;
          if (v.density > 0.0f) hasOneShapeWNonZeroDensity = true;
        }
        if (!hasOneShapeWNonZeroDensity) return false;
      }
      return true;
    }
  };

  struct CastResult {
    BodyID bodyID{};
    WorldPos position{};
    glm::vec3 normal{};
  };

  struct ContactHitEvent {
    BodyID a{};
    BodyID b{};
    WorldPos worldPoint;
    glm::vec3 normal;     /// Normal vector pointing from A to B
    float approachSpeed;  // The speed the shapes are approaching. Always positive. Typically in meters per second.
  };

  struct OverlapInfo {
    WorldPos origin;
    CategoryBits categoryBits = DefaultCategoryBits;
    MaskBits maskBits = DefaultMaskBits;
  };

  struct CastInfo : public OverlapInfo {
    glm::vec3 translation;  // distance and direction the cast will travel from the origin, aka endpoint: origin + translation
  };

  struct ShapeCastInfo : public CastInfo {
    std::span<const glm::vec3> points;
    float pointsRadii;
  };

  struct OverlapShapeInfo : public OverlapInfo {
    std::span<const glm::vec3> points;
    float pointsRadii;
  };

 public:
  Physics();
  ~Physics();
  Physics(Physics&&) noexcept;
  Physics& operator=(Physics&&) noexcept;

  Physics(const CreateInfo&);

  void Update(float timeStep = 1.0f / 60.0f, uint32_t subStepCount = 4);

  [[nodiscard]]
  BodyID CreateBody(const BodyInfo& info);
  void DestroyBody(BodyID id);

  EntityID GetBodyEntity(BodyID id);

  std::pair<WorldPos, glm::quat> GetBodyTransform(BodyID id) const;
  void SetBodyTransform(BodyID id, const WorldPos& pos, const glm::quat& quat);

  void ApplyForce(BodyID id, const glm::vec3& force);
  void ApplyForceAtPosition(BodyID id, const glm::vec3& force, const WorldPos& worldPoint);

  std::vector<ContactHitEvent> GetGlobalContactHitEvents() const;

  std::optional<CastResult> CastRay(const CastInfo&) const;
  std::optional<CastResult> CastShape(const ShapeCastInfo&) const;

  std::vector<BodyID> OverlapShape(const OverlapShapeInfo&) const;
  std::vector<BodyID> OverlapSphere(const OverlapInfo& info, float radius) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;

  void Destroy();
};

}  // namespace maple