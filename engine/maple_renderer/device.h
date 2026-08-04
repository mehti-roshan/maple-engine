#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_core.h>

#include <string>
#include <vector>

#include "renderer_callbacks.h"

namespace maple {
class Device {
 public:
  bool Init(VkInstance instance, VkSurfaceKHR surface, FrameBufferSizeCallback frameBufferSizeFunc, std::string& err);
  void Destroy();

  bool UpdateSwapChain(FrameBufferSizeCallback frameBufferSizeFunc, std::string& err);

  VkPhysicalDevice mPhysicalDevice = nullptr;

  VkDevice mDevice = nullptr;
  uint32_t mGraphicsQFamilyIdx = 0;
  uint32_t mPresentQFamilyIdx = 0;
  VkQueue mGraphicsQueue = nullptr;
  VkQueue mPresentQueue = nullptr;

  VkSwapchainKHR mSwapchain = nullptr;
  struct SwapChainData {
    VkImage img;
    VkImageView view;
    VkSemaphore renderCompleteSemaphore;
  };
  std::vector<SwapChainData> mSwapchainData;

  VmaAllocator mAllocator = nullptr;

 private:
  VkSurfaceKHR mSurface = nullptr;  // reference, non owning
  VkExtent2D mSwapchainExtent;

  bool CreateSwapChain(FrameBufferSizeCallback frameBufferSizeFunc, std::string& err);
};
}  // namespace maple