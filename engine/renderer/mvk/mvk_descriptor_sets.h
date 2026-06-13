#pragma once
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

#include "log_macros.h"

namespace mvk {
struct DescriptorSets {
  enum Type {
    Uniform,
    SSBO,
    CombinedImageSampler,
  };

  vk::raii::DescriptorSetLayout layout;
  std::vector<vk::raii::DescriptorSet> sets;
  std::unordered_map<uint32_t, Type> bindingTypes;

  struct Layout {
    uint32_t bindingSlot;
    Type type;
    uint32_t arrayCount;
    vk::ShaderStageFlagBits usedStages;
  };

  struct CreateInfo {
    const vk::raii::Device& device;
    vk::DescriptorPool pool;
    uint32_t count;
    const std::vector<Layout>& description;
  };

  DescriptorSets() : layout(nullptr) {}
  DescriptorSets(const CreateInfo& info) : layout(nullptr) {
    auto vkFormat = layoutToVk(info.description);
    vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(vkFormat.size()), .pBindings = vkFormat.data()};

    layout = vk::raii::DescriptorSetLayout(info.device, layoutInfo);

    std::vector<vk::DescriptorSetLayout> layouts(info.count, layout);
    sets = vk::raii::DescriptorSets(info.device, {.descriptorPool = info.pool, .descriptorSetCount = info.count, .pSetLayouts = layouts.data()});

    for (const auto& v : info.description) {
      bindingTypes[v.bindingSlot] = v.type;
    }
  }

  struct UpdateData {
    uint32_t descriptorIdx, binding, arrayIdxOffset, numArrayToUpdate;
    vk::DescriptorBufferInfo* buffer = nullptr;
    vk::DescriptorImageInfo* image = nullptr;
  };

  void UpdateDescriptorSets(const vk::raii::Device& device, const std::vector<UpdateData>& data) {
    std::vector<vk::WriteDescriptorSet> updateData;
    updateData.reserve(data.size());
    for (const auto& v : data) {
      updateData.push_back(vk::WriteDescriptorSet{
        .dstSet = sets[v.descriptorIdx],
        .dstBinding = v.binding,
        .dstArrayElement = v.arrayIdxOffset,
        .descriptorCount = v.numArrayToUpdate,
        .descriptorType = typeToVkDescriptorType(bindingTypes.at(v.binding)),
        .pImageInfo = v.image,
        .pBufferInfo = v.buffer,
      });
    }
    device.updateDescriptorSets(updateData, {});
  }

 private:
  vk::DescriptorType typeToVkDescriptorType(Type t) {
    switch (t) {
      case Type::Uniform:
        return vk::DescriptorType::eUniformBuffer;
      case Type::SSBO:
        return vk::DescriptorType::eStorageBuffer;
      case Type::CombinedImageSampler:
        return vk::DescriptorType::eCombinedImageSampler;
      default:
        MAPLE_FATAL("unknown vk::DescriptorType for descriptor set");
    }
  }

  std::vector<vk::DescriptorSetLayoutBinding> layoutToVk(const std::vector<Layout>& description) {
    std::vector<vk::DescriptorSetLayoutBinding> vkFormat;

    vkFormat.reserve(description.size());
    for (const auto& v : description) {
      vkFormat.push_back({
        .binding = v.bindingSlot,
        .descriptorType = typeToVkDescriptorType(v.type),
        .descriptorCount = v.arrayCount,
        .stageFlags = v.usedStages,
      });
    }

    return vkFormat;
  }
};

}  // namespace mvk