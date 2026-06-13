#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace mvk {
struct DescriptorPool {
  vk::raii::DescriptorPool pool;

  struct CreateInfo {
    vk::raii::Device& device;
    uint32_t maxSets;
    const std::vector<std::pair<vk::DescriptorType, uint32_t>>& resourceSizes;
    bool freeDescriptorSet = false;
  };

  DescriptorPool() : pool(nullptr) {}
  DescriptorPool(const CreateInfo& info) : pool(nullptr) {
    std::vector<vk::DescriptorPoolSize> poolSizes;
    poolSizes.reserve(info.resourceSizes.size());
    for (const auto& v : info.resourceSizes) {
      poolSizes.push_back(vk::DescriptorPoolSize(v.first, v.second));
    }
    pool = vk::raii::DescriptorPool(
      info.device,
      vk::DescriptorPoolCreateInfo{
        .flags = info.freeDescriptorSet ? vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet : (vk::DescriptorPoolCreateFlagBits)0,
        .maxSets = info.maxSets,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
      });
  }
};

}  // namespace mvk