#pragma once
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>

namespace mvk {
class Allocator {
 public:
  enum MemType {
    Mesh,
    Image,
    Texture,
    UBO,
    SSBO,
  };

  void AllocBuffer();
  void AllocImageMemory();

 private:
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

}  // namespace mvk