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

struct SwapChainDetails {
  vk::SurfaceFormatKHR format;
  vk::Extent2D extent;
};

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
  vk::raii::Device mDevice = nullptr;
  vk::raii::Queue mGraphicsQueue = nullptr;
  vk::raii::Queue mPresentQueue = nullptr;
  vk::raii::SurfaceKHR mSurface = nullptr;
  vk::raii::SwapchainKHR mSwapChain = nullptr;
  SwapChainDetails mSwapChainDetails;
  std::vector<vk::Image> mSwapChainImages;
  std::vector<vk::ImageView> mSwapChainImageViews;
  vk::raii::PipelineLayout mPipelineLayout = nullptr;
  vk::raii::Pipeline mGraphicsPipeline = nullptr;

  void createInstance(const std::vector<const char*>& glfwExtensions);
  void setupDebugMessenger();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSwapChain();
  void createImageViews();
  void createGraphicsPipeline();
};
}  // namespace maple