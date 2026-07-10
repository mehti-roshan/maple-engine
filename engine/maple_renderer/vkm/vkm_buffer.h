#pragma once
#include <vulkan/vulkan_raii.hpp>

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

};  // namespace vkm