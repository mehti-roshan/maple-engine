#pragma once

#include <cstdint>
#include <optional>
#include <vulkan/vulkan_raii.hpp>

#include "material_builder_data.h"
#include "vk_enum_translation.h"
#include "vkm/vkm_pipeline_layout.h"

namespace vkm {
class Pipeline {
 public:
  struct VertexLayoutDescription {
    vk::VertexInputBindingDescription binding;
    std::vector<vk::VertexInputAttributeDescription> attributes;
  };

  struct AttachmentFormats {
    std::span<const vk::Format> colorFormats;
    std::optional<vk::Format> depthFormat;
  };

  struct CreateInfo {
    const vk::raii::Device& device;
    std::optional<VertexLayoutDescription> vertexLayoutDescription = std::nullopt;
    const vkm::PipelineLayout& layout;
    AttachmentFormats formats;

    const maple::MaterialBuilderData& materialData;
  };

  Pipeline(const CreateInfo& info) {
    auto shaderModule = createShaderModule(info.device, info.materialData.shaderCode);

    std::array shaderStages = {
      vk::PipelineShaderStageCreateInfo{
        .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = info.materialData.vertEntryFuncName.c_str()},
      vk::PipelineShaderStageCreateInfo{
        .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = info.materialData.fragEntryFuncName.c_str()},
    };

    auto& vertDesc = info.vertexLayoutDescription;
    bool hasVertDesc = vertDesc.has_value();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
      .vertexBindingDescriptionCount = hasVertDesc ? 1u : 0u,
      .pVertexBindingDescriptions = hasVertDesc ? &vertDesc->binding : nullptr,
      .vertexAttributeDescriptionCount = hasVertDesc ? static_cast<uint32_t>(vertDesc->attributes.size()) : 0,
      .pVertexAttributeDescriptions = hasVertDesc ? vertDesc->attributes.data() : nullptr,
    };

    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
      .colorAttachmentCount = static_cast<uint32_t>(info.formats.colorFormats.size()),
      .pColorAttachmentFormats = info.formats.colorFormats.empty() ? nullptr : info.formats.colorFormats.data(),
      .depthAttachmentFormat = info.formats.depthFormat.value_or(vk::Format{}),
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};

    vk::PipelineViewportStateCreateInfo viewportState{
      .viewportCount = 1,
      .scissorCount = 1,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = info.materialData.rasterizer.depthClampEnable,
      .rasterizerDiscardEnable = info.materialData.rasterizer.rasterizerDiscardEnable,
      .polygonMode = maple::ToVulkan(info.materialData.rasterizer.polygonMode),
      .cullMode = maple::ToVulkan(info.materialData.rasterizer.cullMode),
      .frontFace = maple::ToVulkan(info.materialData.rasterizer.frontFace),
      .depthBiasEnable = info.materialData.rasterizer.depthBiasEnable,
      .depthBiasSlopeFactor = 1.0f,
      .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable = info.formats.depthFormat.has_value() && info.materialData.depthStencil.depthTest ? vk::True : vk::False,
      .depthWriteEnable = info.formats.depthFormat.has_value() && info.materialData.depthStencil.depthWrite ? vk::True : vk::False,
      .depthCompareOp = maple::ToVulkan(info.materialData.depthStencil.depthCompareOp),
      .depthBoundsTestEnable = info.materialData.depthStencil.depthBoundsTestEnable,
      .stencilTestEnable = info.materialData.depthStencil.stencilTestEnable,
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable = info.materialData.blendingState.blendEnable,
      .colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{
      .logicOpEnable = vk::False,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment,
    };

    std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data(),
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo{
      .pNext = &pipelineRenderingCreateInfo,
      .stageCount = shaderStages.size(),
      .pStages = shaderStages.data(),
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = info.layout.GetLayout(),
      .renderPass = nullptr,
    };

    pipeline = vk::raii::Pipeline(info.device, nullptr, pipelineInfo);
  };

  const vk::raii::Pipeline& GetPipeline() const { return pipeline; }
  vk::raii::Pipeline& GetPipeline() { return pipeline; }

 private:
  vk::raii::Pipeline pipeline = nullptr;

  [[nodiscard]]
  static vk::raii::ShaderModule createShaderModule(const vk::raii::Device& device, std::span<const uint8_t> code) {
    return vk::raii::ShaderModule(
      device,
      vk::ShaderModuleCreateInfo{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),  // std::vector guarantees a 4 byte boundary, so this is safe
      });
  }
};

}  // namespace vkm