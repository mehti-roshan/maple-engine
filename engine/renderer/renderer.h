#pragma once
#include <memory>

typedef struct VkInstance_T* VkInstance;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;

namespace maple {

class Renderer {
 public:
  Renderer();
  ~Renderer();

  void Init(const uint32_t requiredExtensionsCount, const char* const* requiredExtensions);
  void Destroy();

  VkInstance GetInstance() const;
  void SetSurface(VkSurfaceKHR surface);

 private:
  struct Impl;
  std::unique_ptr<Impl> mPimpl;
};
}  // namespace maple