#pragma once

#include <vulkan/vulkan_raii.hpp>

#include "enums.h"
#include "vk_enum_translation.h"
namespace vkm {
class PipelineLayout {
 public:
  struct PushConstantInfo {
    maple::ShaderStage stage;
    uint32_t size;
    uint32_t offset = 0;
  };

  struct Info {
    const vk::raii::Device& device;
    std::optional<PushConstantInfo> pushConstantInfo = std::nullopt;
    const vk::raii::DescriptorSetLayout& descriptorSetLayout;
  };

  PipelineLayout() = default;
  PipelineLayout(const Info& info) {
    vk::PushConstantRange pushRange{};
    uint32_t pushConstantRangeCount = 0;
    if (info.pushConstantInfo.has_value()) {
      pushConstantRangeCount = 1;
      pushRange.stageFlags = maple::ToVulkan(info.pushConstantInfo->stage);
      pushRange.size = info.pushConstantInfo->size;
      pushRange.offset = info.pushConstantInfo->offset;
    }

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
      .setLayoutCount = 1,
      .pSetLayouts = &*info.descriptorSetLayout,
      .pushConstantRangeCount = pushConstantRangeCount,
      .pPushConstantRanges = pushConstantRangeCount == 0 ? nullptr : &pushRange,
    };
    layout = vk::raii::PipelineLayout(info.device, pipelineLayoutInfo);
  };

  const vk::raii::PipelineLayout& GetLayout() const { return layout; };
  vk::raii::PipelineLayout& GetLayout() { return layout; };

 private:
  vk::raii::PipelineLayout layout = nullptr;
};
};  // namespace vkm