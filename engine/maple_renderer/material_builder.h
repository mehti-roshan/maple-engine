#pragma once

#include "material.h"
#include "material_builder_data.h"
#include "vkm/vkm_pipeline.h"

namespace maple {
struct MaterialBuilder {
 public:
  static Material Build(const vk::raii::Device& device,
                        uint32_t pushConstantSize,
                        const vk::raii::DescriptorSetLayout& descriptorSetLayout,
                        std::span<const vk::Format> colorFormats,
                        std::optional<vk::Format> depthFormat,
                        const MaterialBuilderData& data) {
    Material result;

    result.pipeline = vkm::Pipeline(vkm::Pipeline::CreateInfo{
      .device = device,
      .pushConstantInfo =
        vkm::Pipeline::PushConstantInfo{
          .stage = ShaderStage::AllGraphicsAndCompute,
          .size = pushConstantSize,
        },
      .formats =
        vkm::Pipeline::AttachmentFormats{
          .colorFormats = colorFormats,
          .depthFormat = depthFormat,
        },
      .descriptorSetLayout = descriptorSetLayout,
      .shaderCode = data.shaderCodeSpirv,
    });

    return result;
  }
};
}  // namespace maple