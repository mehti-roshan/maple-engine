#pragma once

#include "engine/renderer/vk_memory_manager.h"
#include "engine/renderer/vk_texture.h"
#include "renderer_callbacks.h"
#include "vk_logical_device.h"
#include "vk_physical_device.h"

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

namespace maple {
struct VulkanSwapChain {
 public:
  struct ImageAndImageView {
    vk::Image img;
    vk::raii::ImageView view;
  };

  vk::raii::SwapchainKHR swapchain;
  std::vector<ImageAndImageView> images;
  vk::SurfaceFormatKHR format;
  vk::Extent2D extent;
  VulkanTexture depthTexture;
  vk::Format depthFormat;

  struct CreateInfo {
    const VulkanPhysicalDevice& physicalDevice;
    const VulkanLogicalDevice& device;
    const vk::raii::SurfaceKHR& surface;
    VulkanMemoryManager& memoryManager;
    FrameBufferSizeCallback framebufferSizeCb;
  };

  VulkanSwapChain() : swapchain(nullptr) {}
  VulkanSwapChain(const CreateInfo& info) : swapchain(nullptr) { create(info); }

  void ReCreate(const CreateInfo& info) {
    uint32_t width, height;
    info.framebufferSizeCb(width, height);
    while (width == 0 || height == 0) {
      MAPLE_DEBUG("minimized...");
      info.framebufferSizeCb(width, height);
    }
    info.device.device.waitIdle();

    cleanupSwapChain();
    create(info);
  }

 private:
  void create(const CreateInfo& info) {
    auto surfaceCapabilities = info.physicalDevice.SurfaceCapabilities();
    format = chooseSwapSurfaceFormat(info.physicalDevice.SurfaceFormats());
    extent = chooseSwapExtent(surfaceCapabilities, info.framebufferSizeCb);

    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount
                                                                                                                 : minImageCount;

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
      .flags = vk::SwapchainCreateFlagsKHR(),
      .surface = info.surface,
      .minImageCount = minImageCount,
      .imageFormat = format.format,
      .imageColorSpace = format.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .preTransform = surfaceCapabilities.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = chooseSwapPresentMode(info.physicalDevice.device.getSurfacePresentModesKHR(info.surface)),
      .clipped = true,
      .oldSwapchain = nullptr,
    };

    auto qIndices = info.physicalDevice.queueFamilyIndices;
    uint32_t queueFamilyIndices[] = {qIndices.graphics, qIndices.present};

    if (qIndices.graphics != qIndices.present) {
      swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
      swapChainCreateInfo.queueFamilyIndexCount = 2;
      swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
      swapChainCreateInfo.queueFamilyIndexCount = 0;      // Optional
      swapChainCreateInfo.pQueueFamilyIndices = nullptr;  // Optional
    }

    swapchain = vk::raii::SwapchainKHR(info.device.device, swapChainCreateInfo);
    auto swapchainImages = swapchain.getImages();

    // swapchainImages.clear();

    vk::ImageViewCreateInfo imageViewCreateInfo{
      .viewType = vk::ImageViewType::e2D,
      .format = format.format,
      .components =
        {
          vk::ComponentSwizzle::eIdentity,
          vk::ComponentSwizzle::eIdentity,
          vk::ComponentSwizzle::eIdentity,
          vk::ComponentSwizzle::eIdentity,
        },
      .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };

    for (auto& swapchainImg : swapchainImages) {
      imageViewCreateInfo.image = swapchainImg;
      images.emplace_back(std::move(ImageAndImageView{swapchainImg, vk::raii::ImageView(info.device.device, imageViewCreateInfo)}));
    }

    depthFormat = findDepthFormat(info.physicalDevice);

    depthTexture = info.memoryManager.createTexture(
      {extent.width, extent.height, 1}, depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth);
  }

  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats)
      if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        return availableFormat;

    return availableFormats[0];
  }

  static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
      if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
        return availablePresentMode;
      }
    }
    return vk::PresentModeKHR::eFifo;
  }

  static vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, FrameBufferSizeCallback fbCallback) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    uint32_t width, height;
    fbCallback(width, height);

    return {std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
  }

  static vk::Format findSupportedFormat(const VulkanPhysicalDevice& physicalDevice,
                                        const std::vector<vk::Format>& candidates,
                                        vk::ImageTiling tiling,
                                        vk::FormatFeatureFlags features) {
    for (vk::Format format : candidates) {
      vk::FormatProperties props = physicalDevice.device.getFormatProperties(format);

      if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
        return format;
      else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
        return format;
    }

    MAPLE_FATAL("Failed to find supported format");
  }

  static vk::Format findDepthFormat(const VulkanPhysicalDevice& physicalDevice) {
    return findSupportedFormat(physicalDevice,
                               {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                               vk::ImageTiling::eOptimal,
                               vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  void cleanupSwapChain() {
    images.clear();
    swapchain = nullptr;
  }
};
}  // namespace maple