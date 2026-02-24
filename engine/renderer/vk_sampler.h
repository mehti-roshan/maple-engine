#pragma once

#include <engine/third_party/vma/vk_mem_alloc.h>

#include "log_macros.h"

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

struct VulkanSampler {
  VkSampler sampler = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;

  VulkanSampler() = default;

  VulkanSampler(VkDevice device, const vk::SamplerCreateInfo& info) : device(device) {
    const VkSamplerCreateInfo& ci = reinterpret_cast<const VkSamplerCreateInfo&>(info);

    if (vkCreateSampler(device, &ci, nullptr, &sampler) != VK_SUCCESS) {
      sampler = VK_NULL_HANDLE;
      MAPLE_FATAL("Failed to create Vulkan sampler");
    }
  }

  ~VulkanSampler() { destroy(); }

  // no copy
  VulkanSampler(const VulkanSampler&) = delete;
  VulkanSampler& operator=(const VulkanSampler&) = delete;

  // move ctor
  VulkanSampler(VulkanSampler&& other) noexcept { *this = std::move(other); }

  // move assign
  VulkanSampler& operator=(VulkanSampler&& other) noexcept {
    if (this != &other) {
      destroy();
      sampler = other.sampler;
      device = other.device;

      other.sampler = VK_NULL_HANDLE;
      other.device = VK_NULL_HANDLE;
    }
    return *this;
  }

 private:
  void destroy() {
    if (sampler && device) {
      vkDestroySampler(device, sampler, nullptr);
      sampler = VK_NULL_HANDLE;
    }
  }
};
