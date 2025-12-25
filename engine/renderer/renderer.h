#pragma once
#include <memory>
#include <functional>
#include <vector>

typedef struct VkInstance_T* VkInstance;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;

// creates a VkSurfaceKHR using the provided VkInstance
using SurfaceCreateCallback = std::function<VkSurfaceKHR(VkInstance)>;

namespace maple {

class Renderer {
 public:
  Renderer();
  ~Renderer();

  void Init(const std::vector<const char*>& requiredExtensions, SurfaceCreateCallback surfaceCreateCallback);
  void Destroy();

 private:
  struct Impl;
  std::unique_ptr<Impl> mPimpl;
};
}  // namespace maple