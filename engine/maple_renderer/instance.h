#pragma once

#include <vulkan/vulkan_core.h>

#include <string>
#include <vector>

#include "device.h"
#include "renderer_callbacks.h"

namespace maple {
class Instance {
 public:
  bool Init(const std::vector<const char*>& requiredExtensions,
            SurfaceCreateCallback surfaceFunc,
            FrameBufferSizeCallback frameBufferSizeFunc,
            bool debug,
            std::string& err);
  void Destroy();

  VkInstance mInstance = nullptr;
  VkDebugUtilsMessengerEXT mDebugMessenger = nullptr;
  VkSurfaceKHR mSurface = nullptr;
  Device mDevice;
};
}  // namespace maple