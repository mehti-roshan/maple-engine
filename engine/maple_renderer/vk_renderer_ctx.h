#pragma once

#include <glm/fwd.hpp>
#include <glm/glm.hpp>

#include "vk_logical_device.h"
#include "vk_physical_device.h"
#include "vk_swapchain.h"
#include "vkm/vkm_allocator.h"
#define GLM_ENABLE_EXPERIMENTAL

#include <vector>
#include <vulkan/vulkan_raii.hpp>

#include "vk_instance_ctx.h"

namespace maple {
class VkRendererCtx {
 public:
  VkRendererCtx() = default;
  VkRendererCtx(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback, FrameBufferSizeCallback);
  void Destroy() {
    if (mDevice.device != nullptr) {
      mDevice.device.waitIdle();
    }
  }
  VkRendererCtx(VkRendererCtx&&) noexcept;
  VkRendererCtx& operator=(VkRendererCtx&&) noexcept;

  VkRendererCtx(VkRendererCtx&) noexcept = delete;
  VkRendererCtx& operator=(VkRendererCtx&) noexcept = delete;

  void Init(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback, FrameBufferSizeCallback);

  [[nodiscard]]
  vk::raii::CommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer);

  FrameBufferSizeCallback mFrameBufferSizeCallback;

  VulkanInstanceContext mInstanceCtx;
  vk::raii::SurfaceKHR mSurface = nullptr;
  VulkanPhysicalDevice mPhysicalDevice;
  VulkanLogicalDevice mDevice;
  VulkanSwapChain mSwapChain;
  vkm::Allocator mAllocator;

  vk::raii::CommandPool mGraphicsCommandPool = nullptr;
  vk::raii::CommandPool mTransferCommandPool = nullptr;

  // TOOD: tranfer all this per frame data into a stack array size of MAX_FRAMES_IN_FLIGHT
  struct FrameData {
    vk::raii::CommandBuffer cmd = nullptr;
    vk::raii::Semaphore presentCompleteSem = nullptr;
    vk::raii::Fence drawFence = nullptr;
  };
  std::vector<FrameData> mFrameData;
  std::vector<vk::raii::Semaphore> mRenderCompleteSems;  // length: length of swapchain images

  static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

 private:
  void createCommandPools();
  void createFrameData();
};
}  // namespace maple