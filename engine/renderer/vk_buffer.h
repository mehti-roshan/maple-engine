#pragma once

#include "log_macros.h"
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <engine/third_party/vma/vk_mem_alloc.h>

// Simple RAII wrapper for VMA-managed Vulkan buffers
struct VulkanBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  VmaAllocator allocator = nullptr;
  VmaAllocation allocation = VK_NULL_HANDLE;

  VulkanBuffer() {};

  // Construction
  VulkanBuffer(VmaAllocator allocator, VkDeviceSize size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags flags) : allocator(allocator), size(size) {
    MAPLE_ASSERT(allocator, "VMA allocator must be valid");

    vk::BufferCreateInfo bufferInfo{
      .size = size,
      .usage = usage,
      .sharingMode = vk::SharingMode::eExclusive,
    };

    VmaAllocationCreateInfo allocInfo{.flags = flags, .usage = memoryUsage};

    VkResult result = vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfo), &allocInfo, &buffer, &allocation, nullptr);
    if (result != VK_SUCCESS) MAPLE_FATAL("Failed to create VMA buffer");
  }

  // Disable copying, allow moving
  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;

  VulkanBuffer(VulkanBuffer&& other) noexcept : buffer(other.buffer), size(other.size), allocator(other.allocator), allocation(other.allocation) {
    other.buffer = VK_NULL_HANDLE;
    other.allocation = VK_NULL_HANDLE;
  }

  VulkanBuffer& operator=(VulkanBuffer&& other) noexcept {
    if (this != &other) {
      destroy();
      buffer = other.buffer;
      size = other.size;
      allocator = other.allocator;
      allocation = other.allocation;

      other.buffer = VK_NULL_HANDLE;
      other.allocation = VK_NULL_HANDLE;
    }
    return *this;
  }

  // Destruction
  ~VulkanBuffer() { destroy(); }

  // Map memory for CPU access
  void* Map() {
    void* data = nullptr;
    vmaMapMemory(allocator, allocation, &data);
    return data;
  }

  void UnMap() { vmaUnmapMemory(allocator, allocation); }

  // Convenience: copy data into buffer
  void Upload(const void* src, VkDeviceSize bytes, VkDeviceSize offset = 0) {
    MAPLE_ASSERT(offset + bytes <= size, "Upload size exceeds buffer");
    void* dst = Map();
    std::memcpy(static_cast<char*>(dst) + offset, src, static_cast<size_t>(bytes));
    UnMap();
  }

 private:
  void destroy() {
    if (buffer) {
      vmaDestroyBuffer(allocator, buffer, allocation);
      buffer = VK_NULL_HANDLE;
      allocation = VK_NULL_HANDLE;
    }
  }
};
