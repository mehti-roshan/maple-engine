#pragma once

#include <engine/file/file.h>

#include <cstdint>

#include "engine/renderer/vk_swapchain.h"
#include "vk_logical_device.h"

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

namespace maple {
struct VulkanGraphicsPipeline {
  vk::raii::PipelineLayout layout;
  vk::raii::Pipeline pipeline;

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

  struct CreateInfo {
    const std::string& shaderFile;
    const VulkanLogicalDevice& device;
    const VulkanSwapChain& swapchain;
    const vk::raii::DescriptorSetLayout& descriptorSetLayout;
    const VertexLayoutDescription& vertexLayoutDescription;
  };

  VulkanGraphicsPipeline() : layout(nullptr), pipeline(nullptr) {};

  VulkanGraphicsPipeline(const CreateInfo& info) : layout(nullptr), pipeline(nullptr) {
    auto shaderModule = createShaderModule(info.device, file::ReadFile(info.shaderFile));

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicState{
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data(),
    };

    // auto bindingDescription = mMesh.GetBindingDescription();
    auto bindingDescription = info.vertexLayoutDescription.binding.binding;
    // auto attributeDescriptions = mMesh.GetAttributeDescriptions();
    auto attributeDescriptions = info.vertexLayoutDescription.attributes.attributes;
    
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bindingDescription,
      .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
      .pVertexAttributeDescriptions = attributeDescriptions.data(),
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};

    vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(info.swapchain.extent.width), static_cast<float>(info.swapchain.extent.height), 0.0f, 1.0f};
    vk::Rect2D scissor{{0, 0}, info.swapchain.extent};
    vk::PipelineViewportStateCreateInfo viewportState{
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
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
    layout = vk::raii::PipelineLayout(info.device.device, pipelineLayoutInfo);

    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &info.swapchain.format.format, .depthAttachmentFormat = info.swapchain.depthFormat};
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

    pipeline = vk::raii::Pipeline(info.device.device, nullptr, pipelineInfo);
  };

 private:
  [[nodiscard]]
  static vk::raii::ShaderModule createShaderModule(const VulkanLogicalDevice& device, const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createInfo{
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };

    return vk::raii::ShaderModule(device.device, createInfo);
  }
};
};  // namespace maple