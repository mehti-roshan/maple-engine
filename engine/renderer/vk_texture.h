#pragma once

#include "vk_image.h"

struct VulkanTexture {
  VulkanImage image;                   // owns VkImage + allocation
  VkImageView view;                    // created from image
  VkDevice device;

  VulkanTexture() = default;

  VulkanTexture(VkDevice device,
                VmaAllocator allocator,
                vk::Extent3D extent,
                vk::Format format,
                vk::ImageUsageFlags usage,
                vk::ImageAspectFlags aspect)
      : device(device) {
    // 1. Allocate image using VMA
    image = VulkanImage(allocator, extent, format, usage, VMA_MEMORY_USAGE_AUTO);

    // 2. Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = static_cast<VkFormat>(format);
    viewInfo.subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(aspect);
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &viewInfo, nullptr, &view);
  }

  ~VulkanTexture() { destroy(); }

  VulkanTexture(const VulkanTexture&) = delete;
  VulkanTexture& operator=(const VulkanTexture&) = delete;

  VulkanTexture(VulkanTexture&& other) noexcept { *this = std::move(other); }

  VulkanTexture& operator=(VulkanTexture&& other) noexcept {
    if (this != &other) {
      destroy();
      device = other.device;
      image = std::move(other.image);
      view = other.view;

      other.view = VK_NULL_HANDLE;
      other.device = VK_NULL_HANDLE;
    }
    return *this;
  }

 private:
  void destroy() {
    if (!device) return;
    if (view) {
      vkDestroyImageView(device, view, nullptr);
      view = VK_NULL_HANDLE;
    }
    // VulkanImage destructor handles vmaDestroyImage
  }
};