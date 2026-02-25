#pragma once

#include <cstdint>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

struct QueueFamilyIndices {
  uint32_t graphics = VK_QUEUE_FAMILY_IGNORED;
  uint32_t present = VK_QUEUE_FAMILY_IGNORED;
  uint32_t compute = VK_QUEUE_FAMILY_IGNORED;
  uint32_t transfer = VK_QUEUE_FAMILY_IGNORED;
  bool hasGraphics() const { return graphics != VK_QUEUE_FAMILY_IGNORED; }
  bool hasPresent() const { return present != VK_QUEUE_FAMILY_IGNORED; }
  bool hasCompute() const { return compute != VK_QUEUE_FAMILY_IGNORED; }
  bool hasTransfer() const { return transfer != VK_QUEUE_FAMILY_IGNORED; }
  bool complete() const { return hasGraphics() && hasPresent() && hasCompute() && hasTransfer(); }
};