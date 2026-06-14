#pragma once
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>

#include "log_macros.h"

namespace vkm {
class Allocator {
 public:
  Allocator(const vk::raii::Device& device) : device(&device) {}

  enum BufType {
    Mesh,
    UBO,
    SSBO,
  };

  vk::raii::Buffer CreateBuffer(uint32_t size, BufType type, vk::SharingMode sharingMode = vk::SharingMode::eExclusive) {
    return vk::raii::Buffer(*device,
                            vk::BufferCreateInfo{
                              .size = size,
                              .usage = vkBufferUsageFlagBitsFromBufType(type),
                              .sharingMode = sharingMode,
                            });
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
  };

  vk::raii::Image CreateImage(const ImageCreateInfo& info) {
    return vk::raii::Image(*device,
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
  }

 private:
  const vk::raii::Device* const device = nullptr;

  static vk::BufferUsageFlags vkBufferUsageFlagBitsFromBufType(BufType t) {
    switch (t) {
      case Mesh:
        return vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer;
      case UBO:
        return vk::BufferUsageFlagBits::eUniformBuffer;
      case SSBO:
        return vk::BufferUsageFlagBits::eStorageBuffer;
      default:
        MAPLE_FATAL("unknown mvk allocator BufType");
    }
  }

  // Find a memory in `memoryTypeBitsRequirement` that includes all of `requiredProperties`
  static int32_t findProperties(const VkPhysicalDeviceMemoryProperties* pMemoryProperties,
                                uint32_t memoryTypeBitsRequirement,
                                VkMemoryPropertyFlags requiredProperties) {
    const uint32_t memoryCount = pMemoryProperties->memoryTypeCount;

    for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
      const uint32_t memoryTypeBits = (1 << memoryIndex);
      const bool isRequiredMemoryType = memoryTypeBitsRequirement & memoryTypeBits;

      const VkMemoryPropertyFlags properties = pMemoryProperties->memoryTypes[memoryIndex].propertyFlags;
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