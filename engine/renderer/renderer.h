#pragma once
#include <functional>
#include <memory>
#include <vector>

typedef struct VkInstance_T* VkInstance;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;

// creates a VkSurfaceKHR using the provided VkInstance
using SurfaceCreateCallback = std::function<VkSurfaceKHR(VkInstance)>;
// query the framebuffer size from the window library
using GetFramebufferSizeCallback = std::function<void(uint32_t&, uint32_t&)>;

namespace maple {

class Renderer {
 public:
  Renderer();
  ~Renderer();

  void Init(const std::vector<const char*>& requiredExtensions, SurfaceCreateCallback surfaceCreateCallback,
            GetFramebufferSizeCallback getFramebufferSizeCallback);
  void Destroy();

 private:
  struct Impl;
  std::unique_ptr<Impl> mPimpl;
};
}  // namespace maple