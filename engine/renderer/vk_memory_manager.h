#pragma once

#include <engine/logging/log_macros.h>

#include "vk_buffer.h"
#include "vk_image.h"
#include "vk_texture.h"

struct VulkanMemoryManager {
 private:
  VmaAllocator allocator = nullptr;
  vk::raii::Device* device = nullptr;

 public:
  // Default constructor
  VulkanMemoryManager() {}

  // Constructor: creates the VMA allocator
  VulkanMemoryManager(vk::raii::Instance& instance, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device* device) : device(device) {
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = *physicalDevice;
    allocatorInfo.device = **device;
    allocatorInfo.instance = *instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_4;

    VkResult result = vmaCreateAllocator(&allocatorInfo, &allocator);
    if (result != VK_SUCCESS) MAPLE_FATAL("Failed to create VMA allocator");

    MAPLE_INFO("VulkanMemoryManager: Allocator created");
  }

  // Destructor: destroys the allocator
  ~VulkanMemoryManager() {
    if (allocator) {
      vmaDestroyAllocator(allocator);
      allocator = nullptr;
      MAPLE_INFO("VulkanMemoryManager: Allocator destroyed");
    }
  }

  VulkanMemoryManager(const VulkanMemoryManager&) = delete;
  VulkanMemoryManager& operator=(const VulkanMemoryManager&) = delete;

  VulkanMemoryManager(VulkanMemoryManager&& other) noexcept : allocator(other.allocator) { other.allocator = nullptr; }

  VulkanMemoryManager& operator=(VulkanMemoryManager&& other) noexcept {
    if (this != &other) {
      if (allocator) vmaDestroyAllocator(allocator);
      allocator = other.allocator;
      other.allocator = nullptr;
    }
    return *this;
  }

  // Accessor for raw allocator
  VmaAllocator get() const { return allocator; }

  // Create a VulkanBuffer
  VulkanBuffer createBuffer(VkDeviceSize size, vk::BufferUsageFlags usage, VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags flags) {
    return VulkanBuffer(allocator, size, usage, memoryUsage, flags);
  }

  // helper for creating VulkanImage
  VulkanImage createImage(vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage, VmaMemoryUsage memoryUsage) {
    return VulkanImage(allocator, extent, format, usage, memoryUsage);
  }

  // // helper for creating VulkanTexture
  VulkanTexture createTexture(vk::Extent3D extent, vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect) {
    return VulkanTexture(device, allocator, extent, format, usage, aspect);
  }
};
