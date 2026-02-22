#pragma once

#include "engine/renderer/vk_buffer.h"
#include "log_macros.h"
#include "vk_header.h" // IWYU pragma: export

#include <engine/third_party/vma/vk_mem_alloc.h>

// RAII wrapper for a VMA-managed Vulkan image
struct VulkanImage {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VmaAllocator allocator = nullptr;

  vk::Format format{};
  vk::Extent3D extent{};
  VkImageUsageFlags usage{};

  VulkanImage() {}

  // Constructor
  VulkanImage(VmaAllocator allocator, vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage, VmaMemoryUsage memoryUsage)
      : allocator(allocator), extent(extent), format(format), usage(usage) {
    MAPLE_ASSERT(allocator, "VMA allocator must be valid");

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = extent;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = static_cast<VkFormat>(format);
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = static_cast<VkImageUsageFlags>(usage);
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryUsage;

    VkResult result = vmaCreateImage(allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);
    if (result != VK_SUCCESS) MAPLE_FATAL("Failed to create VMA image");
  }

  // Disable copy, allow move
  VulkanImage(const VulkanImage&) = delete;
  VulkanImage& operator=(const VulkanImage&) = delete;

  VulkanImage(VulkanImage&& other) noexcept
      : image(other.image), allocation(other.allocation), allocator(other.allocator), format(other.format), extent(other.extent), usage(other.usage) {
    other.image = VK_NULL_HANDLE;
    other.allocation = VK_NULL_HANDLE;
  }

  VulkanImage& operator=(VulkanImage&& other) noexcept {
    if (this != &other) {
      destroy();

      image = other.image;
      allocation = other.allocation;
      allocator = other.allocator;
      format = other.format;
      extent = other.extent;
      usage = other.usage;

      other.image = VK_NULL_HANDLE;
      other.allocation = VK_NULL_HANDLE;
    }
    return *this;
  }

  // Destructor
  ~VulkanImage() { destroy(); }

  // Helper: transition image layout (requires command buffer)
  void TransitionLayout(const vk::raii::CommandBuffer& cmd,
                        vk::ImageLayout oldLayout,
                        vk::ImageLayout newLayout,
                        vk::PipelineStageFlags srcStage = vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlags dstStage = vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::ImageAspectFlagBits aspectMask = vk::ImageAspectFlagBits::eColor) {
    vk::ImageMemoryBarrier barrier{.oldLayout = oldLayout, .newLayout = newLayout, .image = image, .subresourceRange = {aspectMask, 0, 1, 0, 1}};

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      MAPLE_FATAL("Unsupported layout transition");
    }

    cmd.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
  }

  // Helper: upload a buffer to image (requires command buffer)
  void UploadBuffer(const vk::raii::CommandBuffer& cmd, const VulkanBuffer& buffer, uint32_t width, uint32_t height) {
    vk::BufferImageCopy region{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {width, height, 1},
    };
    cmd.copyBufferToImage(buffer.buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
  }

 private:
  void destroy() {
    if (image && allocator) {
      vmaDestroyImage(allocator, image, allocation);
      image = VK_NULL_HANDLE;
      allocation = VK_NULL_HANDLE;
      allocator = nullptr;
    }
  }
};
