#pragma once
#include <engine/file/file.h>

#include <cstdint>
#include <vulkan/vulkan_raii.hpp>

namespace mvk {
class Pipeline {
 public:
  struct VertexBindingLayout {
    vk::VertexInputBindingDescription binding;
  };

  struct VertexAttributeLayout {
    std::vector<vk::VertexInputAttributeDescription> attributes;
  };

  struct VertexLayoutDescription {
    VertexBindingLayout binding;
    VertexAttributeLayout attributes;
  };

  struct SwapChainData {
    vk::raii::SwapchainKHR& swapchain;
    vk::SurfaceFormatKHR format;
    vk::Format depthFormat;
  };

  struct CreateInfo {
    const std::string& shaderFile;
    const vk::raii::Device& device;
    const SwapChainData& swapChainData;

    const vk::raii::DescriptorSetLayout& descriptorSetLayout;
    const VertexLayoutDescription& vertexLayoutDescription;
  };

  Pipeline() : layout(nullptr), pipeline(nullptr) {};

  Pipeline(const CreateInfo& info) : layout(nullptr), pipeline(nullptr) {
    auto shaderModule = createShaderModule(info.device, maple::file::ReadFile(info.shaderFile));

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data(),
    };

    auto bindingDescription = info.vertexLayoutDescription.binding.binding;
    auto attributeDescriptions = info.vertexLayoutDescription.attributes.attributes;

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bindingDescription,
      .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
      .pVertexAttributeDescriptions = attributeDescriptions.data(),
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};

    vk::PipelineViewportStateCreateInfo viewportState{
      .viewportCount = 1,
      .scissorCount = 1,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = vk::False,
      .depthBiasSlopeFactor = 1.0f,
      .lineWidth = 1.0f,
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
      .depthTestEnable = vk::True,
      .depthWriteEnable = vk::True,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False,
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
      .blendEnable = vk::False,
      .colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{
      .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount = 1, .pSetLayouts = &*info.descriptorSetLayout, .pushConstantRangeCount = 0};
    layout = vk::raii::PipelineLayout(info.device, pipelineLayoutInfo);

    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &info.swapChainData.format.format, .depthAttachmentFormat = info.swapChainData.depthFormat};
    vk::GraphicsPipelineCreateInfo pipelineInfo{
      .pNext = &pipelineRenderingCreateInfo,
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizer,
      .pMultisampleState = &multisampling,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicState,
      .layout = layout,
      .renderPass = nullptr,
    };

    pipeline = vk::raii::Pipeline(info.device, nullptr, pipelineInfo);
  };

  auto& GetPipeline() { return pipeline; }
  auto& GetLayout() { return layout; }

 private:
  vk::raii::Pipeline pipeline;
  vk::raii::PipelineLayout layout;

  [[nodiscard]]
  static vk::raii::ShaderModule createShaderModule(const vk::raii::Device& device, const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createInfo{
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };

    return vk::raii::ShaderModule(device, createInfo);
  }
};

}  // namespace mvk