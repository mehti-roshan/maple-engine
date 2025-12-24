#pragma once
#include <memory>
#include <functional>

typedef struct VkInstance_T* VkInstance;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;

// creates a VkSurfaceKHR using the provided VkInstance
using SurfaceCreateCallback = std::function<VkSurfaceKHR(VkInstance)>;

namespace maple {

class Renderer {
 public:
  Renderer();
  ~Renderer();

  void Init(const uint32_t requiredExtensionsCount, const char* const* requiredExtensions, SurfaceCreateCallback surfaceCreateCallback);
  void Destroy();

 private:
  struct Impl;
  std::unique_ptr<Impl> mPimpl;
};
}  // namespace maple