#pragma once
#include <cassert>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <engine/logging/log_macros.h>

#include <vulkan/vulkan_raii.hpp>

struct BufferCreateInfo {
  const vk::raii::Device& device;
  const vk::raii::PhysicalDevice& physicalDevice;
  vk::DeviceSize size;
  vk::BufferUsageFlags usage;
  vk::MemoryPropertyFlags properties;
};

struct Buffer {
  vk::raii::Buffer buffer;
  vk::raii::DeviceMemory bufferMemory;
  vk::DeviceSize size;

  Buffer(const BufferCreateInfo& info)
      : size(info.size),
        buffer(info.device, {.size = info.size, .usage = info.usage, .sharingMode = vk::SharingMode::eExclusive}),
        bufferMemory(info.device,
                     {.allocationSize = buffer.getMemoryRequirements().size,
                      .memoryTypeIndex = findMemoryType(info.physicalDevice, buffer.getMemoryRequirements().memoryTypeBits, info.properties)}) {
    buffer.bindMemory(*bufferMemory, 0);
  }

  [[nodiscard]]
  void* MapMemory(vk::DeviceSize offset, vk::DeviceSize size) {
    return bufferMemory.mapMemory(offset, size);
  }
  void UnMapMemory() { bufferMemory.unmapMemory(); }

 private:
  static uint32_t findMemoryType(vk::raii::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
      if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;

    MAPLE_FATAL("Failed to find suitable memory type");
  }
};