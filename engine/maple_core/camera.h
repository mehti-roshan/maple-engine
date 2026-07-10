#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace maple {
class Camera {
 public:
  glm::vec3 GetPosition() const;
  void SetPosition(const glm::vec3& position);

  void Pitch(float angle);

  void Yaw(float angle);

  void Roll(float angle);

  glm::mat4 GetView() const;

  glm::mat4 GetProjection(float aspectRatio, float fov, float nearPlane, float farPlane) const;

  glm::vec3 Forward() const;
  glm::vec3 Right() const;
  glm::vec3 Up() const;

 private:
  glm::vec3 mPosition;
  glm::quat mOrientation;
};
}  // namespace maple