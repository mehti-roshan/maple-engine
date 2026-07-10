#pragma once

#include <vulkan/vulkan_raii.hpp>

#include "log_macros.h"

namespace vkm {
struct Image {
  vk::raii::Image img = nullptr;
  vk::raii::DeviceMemory memory = nullptr;
  vk::raii::ImageView view = nullptr;
  vk::Extent3D extent;

  // Helper: transition image layout (requires command buffer)
  void TransitionLayout(const vk::raii::CommandBuffer& cmd,
                        vk::ImageLayout oldLayout,
                        vk::ImageLayout newLayout,
                        vk::PipelineStageFlags srcStage = vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlags dstStage = vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::ImageAspectFlagBits aspectMask = vk::ImageAspectFlagBits::eColor) {
    vk::ImageMemoryBarrier barrier{.oldLayout = oldLayout, .newLayout = newLayout, .image = img, .subresourceRange = {aspectMask, 0, 1, 0, 1}};

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      MAPLE_FATAL("Unsupported layout transition");
    }

    cmd.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
  }

  // Helper: upload a buffer to image (requires command buffer)
  void UploadBuffer(const vk::raii::CommandBuffer& cmd, const vk::raii::Buffer& buffer, uint32_t width, uint32_t height) {
    vk::BufferImageCopy region{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {width, height, 1},
    };
    cmd.copyBufferToImage(buffer, img, vk::ImageLayout::eTransferDstOptimal, region);
  }
};
}  // namespace vkm