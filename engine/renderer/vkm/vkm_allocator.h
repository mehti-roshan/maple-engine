#pragma once

#include <cstdint>
#include <utility>
#include <vulkan/vulkan_raii.hpp>

#include "engine/renderer/vkm/vkm_image.h"
#include "log_macros.h"

namespace vkm {

struct Buffer {
  vk::raii::Buffer buffer = nullptr;
  vk::raii::DeviceMemory memory = nullptr;
  VkDeviceSize size = 0;

  // Convenience: copy data into buffer
  void Upload(const void* src, VkDeviceSize bytes, VkDeviceSize offset = 0) {
    MAPLE_ASSERT(offset + bytes <= size, "Upload size exceeds buffer");
    void* dst = memory.mapMemory(0, size);
    std::memcpy(static_cast<char*>(dst) + offset, src, static_cast<size_t>(bytes));
    memory.unmapMemory();
  }

  struct CopyRegion {
    VkDeviceSize size;
    VkDeviceSize srcOffset = 0;
    VkDeviceSize dstOffset = 0;
  };
  // Convenience: copy buffer data into another buffer, required command buffer
  void CopyToBuffer(const vk::raii::CommandBuffer& commandBuffer, const vk::raii::Buffer& dstBuffer, const CopyRegion& offsets) const {
    vk::BufferCopy copyRegion{.srcOffset = offsets.srcOffset, .dstOffset = offsets.dstOffset, .size = offsets.size};
    commandBuffer.copyBuffer(buffer, dstBuffer, copyRegion);
  }
};

class Allocator {
 public:
  Allocator() : device(nullptr), physicalDevice(nullptr) {}
  Allocator(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice) : device(&device), physicalDevice(&physicalDevice) {}

  enum BufType {
    Mesh,
    UBO,
    SSBO,
    Stage,
  };

  Buffer CreateBuffer(uint32_t size, BufType type, vk::SharingMode sharingMode = vk::SharingMode::eExclusive) {
    auto [bufferUsageFlags, memoryPropertyFlags] = vkBufferUsageFlagBitsFromBufType(type);

    vk::raii::Buffer buffer(*device,
                            vk::BufferCreateInfo{
                              .size = size,
                              .usage = bufferUsageFlags,
                              .sharingMode = sharingMode,
                            });

    auto memTypeIdx = findProperties(physicalDevice->getMemoryProperties(), buffer.getMemoryRequirements().memoryTypeBits, memoryPropertyFlags);
    if (memTypeIdx == -1) MAPLE_FATAL("failed to find required memory type idx");
    vk::raii::DeviceMemory memory(
      *device, vk::MemoryAllocateInfo{.allocationSize = buffer.getMemoryRequirements().size, .memoryTypeIndex = static_cast<uint32_t>(memTypeIdx)});
    buffer.bindMemory(memory, 0);
    return {.buffer = std::move(buffer), .memory = std::move(memory), .size = size};
  }

  struct ImageCreateInfo {
    vk::ImageType imageType = vk::ImageType::e2D;
    vk::Format format = vk::Format::eR8G8B8A8Srgb;
    vk::Extent3D extent;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
    vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
    vk::ImageUsageFlags usage;
    vk::SharingMode sharingMode = vk::SharingMode::eExclusive;
    vk::ImageLayout initialLayout = vk::ImageLayout::eUndefined;

    vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor;
  };

  Image CreateImage(const ImageCreateInfo& info) {
    vk::raii::Image img(*device,
                        vk::ImageCreateInfo{
                          .imageType = info.imageType,
                          .format = info.format,
                          .extent = info.extent,
                          .mipLevels = info.mipLevels,
                          .arrayLayers = info.arrayLayers,
                          .samples = info.samples,
                          .tiling = info.tiling,
                          .usage = info.usage,
                          .sharingMode = info.sharingMode,
                          .initialLayout = info.initialLayout,
                        });
    // TODO: add vkGetPhysicalDeviceFormatProperties format checks
    auto memRequirements = img.getMemoryRequirements();
    auto memoryTypeIdx =
      findProperties(physicalDevice->getMemoryProperties(), memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (memoryTypeIdx == -1) MAPLE_FATAL("failed to find required memory type idx");

    vk::raii::DeviceMemory memory(
      *device, vk::MemoryAllocateInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = static_cast<uint32_t>(memoryTypeIdx)});
    img.bindMemory(memory, 0);

    vk::ImageSubresourceRange subresourceRange{
      .aspectMask = info.aspectMask,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
    };
    vk::raii::ImageView view(
      *device,
      vk::ImageViewCreateInfo{.image = img, .viewType = vk::ImageViewType::e2D, .format = info.format, .subresourceRange = subresourceRange});
    return {.img = std::move(img), .memory = std::move(memory), .view = std::move(view)};
  }

 private:
  const vk::raii::Device* device = nullptr;
  const vk::raii::PhysicalDevice* physicalDevice = nullptr;

  static std::pair<vk::BufferUsageFlags, vk::MemoryPropertyFlags> vkBufferUsageFlagBitsFromBufType(BufType t) {
    vk::MemoryPropertyFlags mappableMemFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    vk::MemoryPropertyFlags deviceLocalMemFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    switch (t) {
      case Mesh:
        return {vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                deviceLocalMemFlags};
      case UBO:
        return {vk::BufferUsageFlagBits::eUniformBuffer, mappableMemFlags};
      case SSBO:
        return {vk::BufferUsageFlagBits::eStorageBuffer, mappableMemFlags};
      case Stage:
        return {vk::BufferUsageFlagBits::eTransferSrc, mappableMemFlags};
      default:
        MAPLE_FATAL("unknown mvk allocator BufType");
    }
  }

  // Find a memory in `memoryTypeBitsRequirement` that includes all of `requiredProperties`
  static int32_t findProperties(const vk::PhysicalDeviceMemoryProperties& memoryProperties,
                                uint32_t memoryTypeBitsRequirement,
                                vk::MemoryPropertyFlags requiredProperties) {
    const uint32_t memoryCount = memoryProperties.memoryTypeCount;

    for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
      const uint32_t memoryTypeBits = (1 << memoryIndex);
      const bool isRequiredMemoryType = memoryTypeBitsRequirement & memoryTypeBits;

      const vk::MemoryPropertyFlags properties = memoryProperties.memoryTypes[memoryIndex].propertyFlags;
      const bool hasRequiredProperties = (properties & requiredProperties) == requiredProperties;

      if (isRequiredMemoryType && hasRequiredProperties) {
        return static_cast<int32_t>(memoryIndex);
      }
    }

    // failed to find memory type
    return -1;
  }
};

}  // namespace vkm