#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>

namespace maple {
class Audio {
 public:
  struct CreateInfo {
    // std::optional<Device> playbackDevice = std::nullopt;
    // std::optional<Device> captureDevice = std::nullopt;
  };

  using ClipHndl = uint32_t;
  using SourceHndl = uint32_t;

  struct ClipCreateInfo {
    std::unique_ptr<uint8_t[]> data;
    uint64_t sampleCount = 0;
    uint32_t sampleRate = 0;
    uint8_t bitsPerSample = 0;
    bool formatIsFloatingPoint = false;
    bool isStereo = false;
  };

  struct SourceInfo {
    glm::vec3 position = glm::vec3(0);
    glm::vec3 velocity = glm::vec3(0);
    float pitch = 1.0f;
    float gain = 1.0f;
    bool loop = false;
    bool posRelativeToListener = false;
  };

  Audio();
  ~Audio();
  Audio(const CreateInfo&);
  Audio(Audio&&) noexcept;
  Audio& operator=(Audio&&) noexcept;

  void UpdateListener(const glm::vec3& pos, const glm::vec3& velocity, const glm::quat& orientation);
  ClipHndl CreateClip(const ClipCreateInfo&);
  void DestroyClip(ClipHndl);
  
  SourceHndl PlayClip(ClipHndl, const SourceInfo&);
  void UpdateSource(SourceHndl, const SourceInfo&);
  bool IsPlaying(SourceHndl);
  void StopSource(SourceHndl);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;

  void Destroy();
};
}  // namespace maple