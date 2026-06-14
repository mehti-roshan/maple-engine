#pragma once
#include <glm/glm.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "log_macros.h"

namespace vkh {
// upload a buffer to image (requires command buffer)
void ImageUploadBuffer(const vk::raii::Image& image, const vk::raii::Buffer& buffer, glm::uvec2 dimensions, const vk::raii::CommandBuffer& cmd) {
  vk::BufferImageCopy region{
    .bufferOffset = 0,
    .bufferRowLength = 0,
    .bufferImageHeight = 0,
    .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
    .imageOffset = {0, 0, 0},
    .imageExtent = {dimensions.x, dimensions.y, 1},
  };
  cmd.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
}

// transition image layout (requires command buffer)
void ImageTransitionLayout(const vk::raii::Image& image,
                           vk::ImageLayout oldLayout,
                           vk::ImageLayout newLayout,
                           const vk::raii::CommandBuffer& cmd,
                           vk::PipelineStageFlags srcStage = vk::PipelineStageFlagBits::eTopOfPipe,
                           vk::PipelineStageFlags dstStage = vk::PipelineStageFlagBits::eTopOfPipe,
                           vk::ImageAspectFlagBits aspectMask = vk::ImageAspectFlagBits::eColor) {
  vk::ImageMemoryBarrier barrier{
    .oldLayout = oldLayout,
    .newLayout = newLayout,
    .image = image,
    .subresourceRange = {aspectMask, 0, 1, 0, 1},
  };

  vk::PipelineStageFlags sourceStage, destinationStage;

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

}  // namespace vkh