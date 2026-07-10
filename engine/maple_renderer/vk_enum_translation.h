#pragma once

#include <vulkan/vulkan_raii.hpp>

#include "enums.h"
#include "material_builder_data.h"
#include "sampler_enums.h"
#include "log_macros.h"

namespace maple {
vk::ImageLayout ToVulkan(ImageLayout layout) {
  switch (layout) {
    case ImageLayout::Undefined:
      return vk::ImageLayout::eUndefined;
    case ImageLayout::AttachmentOptimal:
      return vk::ImageLayout::eAttachmentOptimal;
    case ImageLayout::ShaderReadOnlyOptimal:
      return vk::ImageLayout::eShaderReadOnlyOptimal;
    case ImageLayout::PresentSrc:
      return vk::ImageLayout::ePresentSrcKHR;
  }
  MAPLE_FATAL("Unknown ImageLayout");
}

vk::AccessFlags2 ToVulkan(AccessMask access) {
  switch (access) {
    case AccessMask::None:
      return vk::AccessFlagBits2::eNone;
    case AccessMask::ColorAttachmentWrite:
      return vk::AccessFlagBits2::eColorAttachmentWrite;
    case AccessMask::DepthStencilAttachmentWrite:
      return vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    case AccessMask::ShaderRead:
      return vk::AccessFlagBits2::eShaderSampledRead;  // sampled image read
  }
  MAPLE_FATAL("Unknown AccessMask");
}

vk::ShaderStageFlags ToVulkan(ShaderStage stage) {
  switch (stage) {
    case ShaderStage::Vertex:
      return vk::ShaderStageFlagBits::eVertex;
    case ShaderStage::Fragment:
      return vk::ShaderStageFlagBits::eFragment;
    case ShaderStage::AllGraphics:
      return vk::ShaderStageFlagBits::eAllGraphics;
    case ShaderStage::Compute:
      return vk::ShaderStageFlagBits::eCompute;
    case ShaderStage::AllGraphicsAndCompute:
      return vk::ShaderStageFlagBits::eAllGraphics | vk::ShaderStageFlagBits::eCompute;
  }
  MAPLE_FATAL("Unknown ShaderStage");
}

vk::PipelineStageFlags2 ToVulkan(PipelineStage stage) {
  switch (stage) {
    case PipelineStage::TopOfPipe:
      return vk::PipelineStageFlagBits2::eTopOfPipe;
    case PipelineStage::BottomOfPipe:
      return vk::PipelineStageFlagBits2::eBottomOfPipe;
    case PipelineStage::ColorAttachmentOutput:
      return vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    case PipelineStage::EarlyFragmentTests:
      return vk::PipelineStageFlagBits2::eEarlyFragmentTests;
    case PipelineStage::LateFragmentTests:
      return vk::PipelineStageFlagBits2::eLateFragmentTests;
    case PipelineStage::EarlyAndLateFragmentTests:
      return vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
    case PipelineStage::VertexShader:
      return vk::PipelineStageFlagBits2::eVertexShader;
    case PipelineStage::FragmentShader:
      return vk::PipelineStageFlagBits2::eFragmentShader;
    case PipelineStage::ComputeShader:
      return vk::PipelineStageFlagBits2::eComputeShader;
    case PipelineStage::AllGraphics:
      return vk::PipelineStageFlagBits2::eAllGraphics;
    case PipelineStage::AllGraphicsAndCompute:
      return vk::PipelineStageFlagBits2::eAllCommands;
  }
  MAPLE_FATAL("Unknown PipelineStage");
}

vk::Format ToVulkan(Format format) {
  switch (format) {
    case Format::Undefined:
      return vk::Format::eUndefined;
    case Format::R8G8B8A8_UNORM:
      return vk::Format::eR8G8B8A8Unorm;
    case Format::B8G8R8A8_UNORM:
      return vk::Format::eB8G8R8A8Unorm;
    case Format::R8G8B8A8_SRGB:
      return vk::Format::eR8G8B8A8Srgb;
    case Format::B8G8R8A8_SRGB:
      return vk::Format::eB8G8R8A8Srgb;
    case Format::R16G16B16A16_SFLOAT:
      return vk::Format::eR16G16B16A16Sfloat;
    case Format::B10G11R11_UFLOAT:
      return vk::Format::eB10G11R11UfloatPack32;
    case Format::R10G10B10A2_UNORM:
      return vk::Format::eA2R10G10B10UnormPack32;  // A2R10G10B10 = A2 R10 G10 B10
    case Format::R32G32B32A32_SFLOAT:
      return vk::Format::eR32G32B32A32Sfloat;
    case Format::R8_UNORM:
      return vk::Format::eR8Unorm;
    case Format::R16_SFLOAT:
      return vk::Format::eR16Sfloat;
    case Format::R32_SFLOAT:
      return vk::Format::eR32Sfloat;
    case Format::D16_UNORM:
      return vk::Format::eD16Unorm;
    case Format::D16_UNORM_S8:
      return vk::Format::eD16UnormS8Uint;
    case Format::D24_UNORM_S8:
      return vk::Format::eD24UnormS8Uint;
    case Format::D32_SFLOAT:
      return vk::Format::eD32Sfloat;
    case Format::D32_SFLOAT_S8:
      return vk::Format::eD32SfloatS8Uint;
  }
  MAPLE_FATAL("Unknown Format");
}

vk::ImageAspectFlags GetImageAspectFlags(Format format) {
  if (FormatIsColor(format)) return vk::ImageAspectFlagBits::eColor;
  if (FormatIsDepth(format) && FormatHasStencil(format)) return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
  if (FormatIsDepth(format)) return vk::ImageAspectFlagBits::eDepth;
  MAPLE_FATAL("failed to find image aspect flags");
}

vk::PolygonMode ToVulkan(MaterialBuilderData::PolygonMode polygon) {
  switch (polygon) {
    case MaterialBuilderData::PolygonMode::Fill:
      return vk::PolygonMode::eFill;
    case MaterialBuilderData::PolygonMode::Line:
      return vk::PolygonMode::eLine;
    case MaterialBuilderData::PolygonMode::Point:
      return vk::PolygonMode::ePoint;
  }
  MAPLE_FATAL("failed to find PolygonMode");
}

vk::CullModeFlagBits ToVulkan(MaterialBuilderData::CullModeFlagBits cullMode) {
  switch (cullMode) {
    case MaterialBuilderData::CullModeFlagBits::None:
      return vk::CullModeFlagBits::eNone;
    case MaterialBuilderData::CullModeFlagBits::Front:
      return vk::CullModeFlagBits::eFront;
    case MaterialBuilderData::CullModeFlagBits::Back:
      return vk::CullModeFlagBits::eBack;
    case MaterialBuilderData::CullModeFlagBits::FrontAndBack:
      return vk::CullModeFlagBits::eFrontAndBack;
  }
  MAPLE_FATAL("failed to find CullModeFlagBits");
}

vk::FrontFace ToVulkan(MaterialBuilderData::FrontFace frontFace) {
  switch (frontFace) {
    case MaterialBuilderData::FrontFace::CounterClockwise:
      return vk::FrontFace::eCounterClockwise;
    case MaterialBuilderData::FrontFace::Clockwise:
      return vk::FrontFace::eClockwise;
  }
  MAPLE_FATAL("failed to find FrontFace");
}

vk::CompareOp ToVulkan(MaterialBuilderData::CompareOp depthCompareOp) {
  switch (depthCompareOp) {
    case MaterialBuilderData::CompareOp::Never:
      return vk::CompareOp::eNever;
    case MaterialBuilderData::CompareOp::Less:
      return vk::CompareOp::eLess;
    case MaterialBuilderData::CompareOp::Equal:
      return vk::CompareOp::eEqual;
    case MaterialBuilderData::CompareOp::LessOrEqual:
      return vk::CompareOp::eLessOrEqual;
    case MaterialBuilderData::CompareOp::Greater:
      return vk::CompareOp::eGreater;
    case MaterialBuilderData::CompareOp::NotEqual:
      return vk::CompareOp::eNotEqual;
    case MaterialBuilderData::CompareOp::GreaterOrEqual:
      return vk::CompareOp::eGreaterOrEqual;
    case MaterialBuilderData::CompareOp::Always:
      return vk::CompareOp::eAlways;
  }
  MAPLE_FATAL("failed to find CompareOp");
}

vk::Filter ToVulkan(SamplerFilter filter) {
  switch (filter) {
    case maple::SamplerFilter::Nearest:
      return vk::Filter::eNearest;
    case maple::SamplerFilter::Linear:
      return vk::Filter::eLinear;
  }
  MAPLE_FATAL("failed to find SamplerFilter");
}

vk::SamplerMipmapMode ToVulkan(SamplerMipmapMode mode) {
  switch (mode) {
    case SamplerMipmapMode::Nearest:
      return vk::SamplerMipmapMode::eNearest;
    case SamplerMipmapMode::Linear:
      return vk::SamplerMipmapMode::eLinear;
  }
  MAPLE_FATAL("failed to find SamplerMipmapMode");
}

vk::SamplerAddressMode ToVulkan(SamplerAddressMode mode) {
  switch (mode) {
    case SamplerAddressMode::Repeat:
      return vk::SamplerAddressMode::eRepeat;
    case SamplerAddressMode::MirroredRepeat:
      return vk::SamplerAddressMode::eMirroredRepeat;
    case SamplerAddressMode::ClampToEdge:
      return vk::SamplerAddressMode::eClampToEdge;
    case SamplerAddressMode::ClampToBorder:
      return vk::SamplerAddressMode::eClampToBorder;
    case SamplerAddressMode::MirrorClampToEdge:
      return vk::SamplerAddressMode::eMirrorClampToEdge;
  }
  MAPLE_FATAL("failed to find SamplerAddressMode");
}

vk::BorderColor ToVulkan(SamplerBorderColor color) {
  switch (color) {
    case SamplerBorderColor::FloatTransparentBlack:
      return vk::BorderColor::eFloatTransparentBlack;
    case SamplerBorderColor::IntTransparentBlack:
      return vk::BorderColor::eIntTransparentBlack;
    case SamplerBorderColor::FloatOpaqueBlack:
      return vk::BorderColor::eFloatOpaqueBlack;
    case SamplerBorderColor::IntOpaqueBlack:
      return vk::BorderColor::eIntOpaqueBlack;
    case SamplerBorderColor::FloatOpaqueWhite:
      return vk::BorderColor::eFloatOpaqueWhite;
    case SamplerBorderColor::IntOpaqueWhite:
      return vk::BorderColor::eIntOpaqueWhite;
  }
  MAPLE_FATAL("failed to find SamplerBorderColor");
}

}  // namespace maple