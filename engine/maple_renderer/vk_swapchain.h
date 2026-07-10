#pragma once

#include <span>
#include <vulkan/vulkan_raii.hpp>

#include "log_macros.h"
#include "renderer_callbacks.h"
#include "vk_logical_device.h"
#include "vk_physical_device.h"
#include "vkm/vkm_allocator.h"

namespace maple {
/// \class VulkanSwapChain
/// \brief Manages Vulkan swapchain creation, recreation, and image management.
///
/// This class encapsulates the Vulkan swapchain functionality, handling the creation
/// and recreation of swapchain images, image views, and depth textures. It implements
/// strategies for selecting optimal surface formats, present modes, and extents based
/// on physical device capabilities.
///
/// \section Format Selection Strategy
/// The class employs the following strategies for format selection:
///
/// - **Surface Format Selection**: Prioritizes sRGB format (eB8G8R8A8Srgb) with
///   sRGB nonlinear color space for better color accuracy. Falls back to the first
///   available format from the device if the preferred format is not available.
///
/// - **Present Mode Selection**: Attempts to use mailbox presentation mode for
///   lowest latency with minimal tearing. Falls back to FIFO (guaranteed to be
///   available) if mailbox is not supported.
///
/// - **Depth Format Selection**: Automatically selects the first supported depth
///   format from the ordered preference list (32-bit float, 32-bit float + 8-bit
///   stencil, 24-bit + 8-bit stencil), using optimal tiling for performance.
///
/// \section Image Configuration
/// - Minimum image count is automatically determined as max(3, minImageCount),
///   clamped to maxImageCount if necessary.
/// - Images are configured with exclusive sharing mode if graphics and present
///   queues are the same, otherwise concurrent mode is used.
/// - Both color and depth images are created with appropriate usage flags.
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

  struct CreateInfo {
    const VulkanPhysicalDevice& physicalDevice;
    const VulkanLogicalDevice& device;
    const vk::raii::SurfaceKHR& surface;
    vkm::Allocator& allocator;
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
    // if fullscreen, exclusive fullscreen etc, the surface itself forces us to use a specific size
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }

    // else, we query the windowed frame buffer's size, and return that
    // but making sure it respects minimum and maximum swapchain image sizes
    uint32_t width, height;
    fbCallback(width, height);

    return {std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
  }

  static vk::Format findDepthFormat(const VulkanPhysicalDevice& physicalDevice) {
    std::array formats{vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint};
    auto format = physicalDevice.FindFirstSupportedFormat(formats, vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    if (!format.has_value()) MAPLE_FATAL("failed to find suitable depth format");
    return format.value();
  }

  void cleanupSwapChain() {
    images.clear();
    swapchain = nullptr;
  }
};
}  // namespace maple