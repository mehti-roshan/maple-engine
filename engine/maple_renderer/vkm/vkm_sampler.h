#pragma once
#include <vulkan/vulkan_raii.hpp>

namespace vkm {
class Sampler {
 public:
  vk::raii::Sampler sampler = nullptr;

  struct Info {
    vk::Filter magFilter = vk::Filter::eLinear;
    vk::Filter minFilter = vk::Filter::eLinear;
    vk::SamplerMipmapMode mipmapMode = vk::SamplerMipmapMode::eLinear;
    vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat;
    vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat;
    vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat;
    float mipLodBias = 0.0f;
    vk::Bool32 anisotropyEnable = vk::True;
    float maxAnisotropy;  // Can be gathered from device properties.limits.maxSamplerAnisotropy
    vk::Bool32 compareEnable = vk::False;
    vk::CompareOp compareOp = vk::CompareOp::eAlways;
    float minLod = 0.0f;
    float maxLod = 1.0f;
    vk::BorderColor borderColor = vk::BorderColor::eIntOpaqueBlack;
    vk::Bool32 unnormalizedCoordinates = vk::False;
  };

  // static uint64_t HashInfo(const Info& info) {
  //   // TODO: fill
  // };

  Sampler() = default;
  Sampler(const vk::raii::Device& device, const Info& info) {
    sampler = vk::raii::Sampler(device,
                                vk::SamplerCreateInfo{
                                  .magFilter = info.magFilter,
                                  .minFilter = info.minFilter,
                                  .mipmapMode = info.mipmapMode,
                                  .addressModeU = info.addressModeU,
                                  .addressModeV = info.addressModeV,
                                  .addressModeW = info.addressModeW,
                                  .mipLodBias = info.mipLodBias,
                                  .anisotropyEnable = info.anisotropyEnable,
                                  .maxAnisotropy = info.maxAnisotropy,
                                  .compareEnable = info.compareEnable,
                                  .compareOp = info.compareOp,
                                  .minLod = info.minLod,
                                  .maxLod = info.maxLod,
                                  .borderColor = info.borderColor,
                                  .unnormalizedCoordinates = info.unnormalizedCoordinates,
                                });
  }
};
}  // namespace vkm