#pragma once
#include <functional>
#include <vector>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

typedef struct VkInstance_T* VkInstance;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;

// creates a VkSurfaceKHR using the provided VkInstance
using SurfaceCreateCallback = std::function<VkSurfaceKHR(VkInstance)>;
// query the framebuffer size from the window library
using FrameBufferSizeCallback = std::function<void(uint32_t&, uint32_t&)>;

namespace maple {

class Renderer {
 public:
  void Init(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback, FrameBufferSizeCallback);
  void SetFrameBufferResized() { mFrameBufferResized = true; }

 private:
  FrameBufferSizeCallback mFrameBufferSizeCallback;
  bool mFrameBufferResized = false;

  vk::raii::Context mContext;
  vk::raii::Instance mInstance = nullptr;
  vk::raii::DebugUtilsMessengerEXT mDebugMessenger = nullptr;
  vk::raii::PhysicalDevice mPhysicalDevice = nullptr;

  void createInstance(const std::vector<const char*>& glfwExtensions);
  void setupDebugMessenger();
  void pickPhysicalDevice();
  void createLogicalDevice();

  uint32_t findQueueFamilies(vk::raii::PhysicalDevice physicalDevice);
};
}  // namespace maple