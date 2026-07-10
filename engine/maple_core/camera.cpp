#include "camera.h"

namespace maple {
glm::vec3 Camera::GetPosition() const { return mPosition; }
void Camera::SetPosition(const glm::vec3& position) { mPosition = position; }

void Camera::Pitch(float angle) { mOrientation = glm::normalize(glm::angleAxis(angle, Right()) * mOrientation); }

void Camera::Yaw(float angle) { mOrientation = glm::normalize(glm::angleAxis(angle, Up()) * mOrientation); }

void Camera::Roll(float angle) { mOrientation = glm::normalize(glm::angleAxis(angle, Forward()) * mOrientation); }

glm::mat4 Camera::GetView() const {
  glm::mat4 rotation = glm::mat4_cast(glm::conjugate(mOrientation));
  glm::mat4 translation = glm::translate(glm::mat4(1.0f), -mPosition);

  return rotation * translation;
}

glm::mat4 Camera::GetProjection(float aspectRatio, float fov, float nearPlane, float farPlane) const {
  glm::mat4 proj = glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);

  proj[1][1] *= -1.0f;  // Vulkan flip y coordinates

  return proj;
}

glm::vec3 Camera::Up() const { return mOrientation * glm::vec3(0, 1, 0); }
glm::vec3 Camera::Right() const { return mOrientation * glm::vec3(1, 0, 0); }
glm::vec3 Camera::Forward() const { return mOrientation * glm::vec3(0, 0, -1); }
}  // namespace maple