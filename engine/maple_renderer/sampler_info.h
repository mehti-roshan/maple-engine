#pragma once

#include "engine/renderer/sampler_enums.h"
namespace maple {
struct SamplerInfo {
  SamplerFilter magFilter = SamplerFilter::Linear;
  SamplerFilter minFilter = SamplerFilter::Linear;
  SamplerMipmapMode mipmapMode = SamplerMipmapMode::Linear;
  SamplerAddressMode addressModeU = SamplerAddressMode::Repeat;
  SamplerAddressMode addressModeV = SamplerAddressMode::Repeat;
  SamplerAddressMode addressModeW = SamplerAddressMode::Repeat;
  float mipLodBias = 0.0f;
  bool anisotropyEnable = true;
  float maxAnisotropy = 1.0f;  // filled from device limits
  bool compareEnable = false;
  CompareOp compareOp = CompareOp::Always;
  float minLod = 0.0f;
  float maxLod = 1.0f;
  SamplerBorderColor borderColor = SamplerBorderColor::IntOpaqueBlack;
  bool unnormalizedCoordinates = false;
};
}  // namespace maple