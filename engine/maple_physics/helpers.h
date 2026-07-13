#pragma once

#include <Jolt/Jolt.h>

#include <glm/glm.hpp>

#include <Jolt/Math/Quat.h>
#include <Jolt/Physics/Body/MotionQuality.h>
#include <Jolt/Physics/Body/MotionType.h>
#include <Jolt/Math/Real.h>
#include "maple_physics.h"

namespace maple {

class hlp {
 public:
  static JPH::Vec3 ToJolt(const glm::vec3& v) { return JPH::Vec3(v.x, v.y, v.z); }
  static JPH::RVec3 ToJolt(const glm::dvec3& v) { return JPH::RVec3(v.x, v.y, v.z); }
  static JPH::Quat ToJolt(const glm::quat& v) { return JPH::Quat(v.x, v.y, v.z, v.w); }

  static JPH::EMotionQuality ToJolt(MaplePhysics::MotionQuality motionQuality) {
    return motionQuality == MaplePhysics::MotionQuality::Discrete ? JPH::EMotionQuality::Discrete : JPH::EMotionQuality::LinearCast;
  }
  static JPH::EMotionType ToJolt(MaplePhysics::MotionType motionType) {
    return motionType == MaplePhysics::MotionType::Static
      ? JPH::EMotionType::Static
      : (motionType == MaplePhysics::MotionType::Dynamic ? JPH::EMotionType::Dynamic : JPH::EMotionType::Kinematic);
  }
};

}  // namespace maple