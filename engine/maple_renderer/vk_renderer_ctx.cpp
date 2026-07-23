#include "vk_renderer_ctx.h"

#include <engine/maple_logging/log_macros.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "vk_device_features.h"

#define GLM_FORCE_RADIANS
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define TINYOBJLOADER_IMPLEMENTATION

#include "vk_physical_device.h"

#ifdef NDEBUG
constexpr bool debug = false;
#else
constexpr bool debug = true;
#endif

namespace maple {

static std::vector<const char*> requiredDeviceExtensions = {vk::KHRSwapchainExtensionName, vk::KHRBufferDeviceAddressExtensionName};
auto requiredFeatures = DeviceFeature::SamplerAnisotropy | DeviceFeature::ShaderDrawParameters | DeviceFeature::Synchronization2 |
  DeviceFeature::DynamicRendering | DeviceFeature::ExtendedDynamicState | DeviceFeature::BufferDeviceAddress | DeviceFeature::DescriptorIndexing |
  DeviceFeature::ShaderInt64 | DeviceFeature::ScalarBlockLayout;

void VkRendererCtx::Init(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback surfaceCallback, FrameBufferSizeCallback fbCallback) {
  mFrameBufferSizeCallback = fbCallback;
  mInstanceCtx = std::move(VulkanInstanceContext(glfwExtensions, debug));
  mSurface = vk::raii::SurfaceKHR(mInstanceCtx.mInstance, (VkSurfaceKHR)surfaceCallback(*mInstanceCtx.mInstance));

  mPhysicalDevice = VulkanPhysicalDevice(VulkanPhysicalDevice::CreateInfo{
    .surface = mSurface,
    .availableDevices = mInstanceCtx.mInstance.enumeratePhysicalDevices(),
    .requiredDeviceExtensions = requiredDeviceExtensions,
    .requiredFeatureMask = requiredFeatures,
  });

  mDevice = VulkanLogicalDevice(VulkanLogicalDevice::CreateInfo{
    .physicalDevice = mPhysicalDevice,
    .requiredDeviceExtensions = requiredDeviceExtensions,
    .requiredFeatures = requiredFeatures,
  });

  mSwapChain = VulkanSwapChain({.physicalDevice = mPhysicalDevice,
                                .device = mDevice,
                                .surface = mSurface,
                                .allocator = mAllocator,
                                .framebufferSizeCb = mFrameBufferSizeCallback});
  mAllocator = vkm::Allocator(mDevice.device, mPhysicalDevice.device);

  createCommandPools();
  createFrameData();
}

void VkRendererCtx::createCommandPools() {
  auto qIndices = mPhysicalDevice.queueFamilyIndices;

  vk::CommandPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = qIndices.graphics;

  mGraphicsCommandPool = vk::raii::CommandPool(mDevice.device, poolInfo);

  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = qIndices.transfer;
  mTransferCommandPool = vk::raii::CommandPool(mDevice.device, poolInfo);
}

void VkRendererCtx::createFrameData() {
  mFrameData.clear();
  mFrameData.resize(MAX_FRAMES_IN_FLIGHT);

  vk::CommandBufferAllocateInfo allocInfo{
    .commandPool = mGraphicsCommandPool,
    .level = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
  };

  vk::raii::CommandBuffers cmdBuffers(mDevice.device, allocInfo);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    mFrameData[i].cmd = std::move(cmdBuffers[i]);
    mFrameData[i].presentCompleteSem = vk::raii::Semaphore(mDevice.device, vk::SemaphoreCreateInfo{});
    mFrameData[i].drawFence = vk::raii::Fence(mDevice.device, {.flags = vk::FenceCreateFlagBits::eSignaled});
  }

  mRenderCompleteSems.clear();
  mRenderCompleteSems.reserve(mSwapChain.images.size());
  for (size_t i = 0; i < mSwapChain.images.size(); i++) mRenderCompleteSems.emplace_back(mDevice.device, vk::SemaphoreCreateInfo{});
}

vk::raii::CommandBuffer VkRendererCtx::beginSingleTimeCommands() {
  vk::CommandBufferAllocateInfo allocInfo{.commandPool = mGraphicsCommandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
  vk::raii::CommandBuffer commandBuffer = std::move(mDevice.device.allocateCommandBuffers(allocInfo).front());
  vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
  commandBuffer.begin(beginInfo);
  return commandBuffer;
}

void VkRendererCtx::endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) {
  commandBuffer.end();
  vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
  mDevice.queues.graphics.submit(submitInfo, nullptr);
  mDevice.queues.graphics.waitIdle();
}

}  // namespace maple