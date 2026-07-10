#include "maple_renderer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <glm/common.hpp>
#include <glm/fwd.hpp>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vulkan/vulkan_raii.hpp>

#include "enums.h"
#include "log_macros.h"
#include "material.h"
#include "material_builder_data.h"
#include "mesh_data.h"
#include "pool.h"
#include "render_graph.h"
#include "render_target.h"
#include "shader_compilation.h"
#include "vk_enum_translation.h"
#include "vk_renderer_ctx.h"
#include "vkm/vkm_allocator.h"
#include "vkm/vkm_buffer.h"
#include "vkm/vkm_descriptor_pool.h"
#include "vkm/vkm_descriptor_sets.h"
#include "vkm/vkm_mesh.h"
#include "vkm/vkm_pipeline.h"
#include "vkm/vkm_pipeline_layout.h"
#include "vkm/vkm_sampler.h"

namespace maple {

static constexpr uint32_t NUM_INSTANCES = 1024 * 1024;
// TODO: to save material memory introduce a "material buffer cache"
// each pipeline that uses the exact same combination of input shaders will get the same byte offset into the material buffer
// no more unnecessary writing to the material buffer
static constexpr uint32_t NUM_MATERIALS = 1024 * 1024;
static constexpr uint32_t MAX_BINDLESS_TEXTURES = 4096;

struct DrawPush {
  uint64_t vertexBufferAddress;
  uint32_t indexBufferOffset;     // offset of start of index data from start of vertex buffer in bytes
  uint32_t materialBufferOffset;  // byte offset into global material buffer
  uint32_t instanceBufferIndex;   // byte offset into global instance buffer
};

struct MapleRenderer::Impl {
  VkRendererCtx mCtx;
  Pool<vkm::Mesh> mMeshPool;
  Pool<Material> mMaterialPool;

  // TODO: merge render target with general textures
  Pool<RenderTarget> mRenderTargets;  // slots directly map to texture slot
  std::unordered_map<std::string, RenderTargetHndl> mRenderTargetMap;

  vkm::PipelineLayout mGlobalPipelineLayout;

  vkm::DescriptorPool mGlobalDescriptorPool;
  vkm::DescriptorSets mGlobalDescriptorSets;  // view & proj matrices, instance buffer SSBO, etc. for each frame_in_flight

  vkm::Buffer mInstanceSSBO[VkRendererCtx::MAX_FRAMES_IN_FLIGHT];
  vkm::Buffer mGlobalsUniform[VkRendererCtx::MAX_FRAMES_IN_FLIGHT];
  vkm::Buffer mMaterialBuffers[VkRendererCtx::MAX_FRAMES_IN_FLIGHT];

  vkm::Sampler mDefaultSampler;  // TODO: replace with a map of samplers indexed by their settings
};

std::optional<Format> FindFirstSupportedFormat(std::span<const Format> formats, const VkRendererCtx& ctx, vk::FormatFeatureFlags formatFeatures) {
  std::vector<vk::Format> vkFormats(formats.size());
  for (auto [i, format] : std::views::enumerate(formats)) vkFormats[i] = ToVulkan(format);

  auto vkResult = ctx.mPhysicalDevice.FindFirstSupportedFormat(vkFormats, vk::ImageTiling::eOptimal, formatFeatures);
  if (!vkResult.has_value()) return std::nullopt;

  auto it = std::find(vkFormats.begin(), vkFormats.end(), vkResult.value());
  if (it != vkFormats.end()) return formats[std::distance(vkFormats.begin(), it)];

  return std::nullopt;
}

std::optional<Format> MapleRenderer::FindFirstSupportedTextureFormat(std::span<const Format> formats) const {
  return FindFirstSupportedFormat(formats, impl->mCtx, vk::FormatFeatureFlagBits::eSampledImage);
}

std::optional<Format> MapleRenderer::FindFirstSupportedDepthAttachmentFormat(std::span<const Format> formats) const {
  return FindFirstSupportedFormat(formats, impl->mCtx, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

MapleRenderer::MeshHndl MapleRenderer::CreateMesh(const MeshData& data) {
  auto& ctx = impl->mCtx;
  auto mesh = vkm::Mesh(impl->mCtx.mAllocator, data);
  auto cmd = ctx.beginSingleTimeCommands();

  auto stage = ctx.mAllocator.CreateBuffer(data.GetTotalSize(), vkm::Allocator::Stage);
  stage.Upload(data.verts.data(), data.verts.size());
  stage.Upload(data.indices.data(), data.indices.size() * sizeof(decltype(data.indices)::value_type), data.verts.size());
  stage.CopyToBuffer(cmd, mesh.meshBuffer.buffer, {.size = stage.size});
  ctx.endSingleTimeCommands(cmd);

  auto val = impl->mMeshPool.Add(std::move(mesh));
  return val;
}

void MapleRenderer::DestroyMesh(MeshHndl handle) { impl->mMeshPool.Remove(handle); }

MapleRenderer::MaterialHndl MapleRenderer::CreateMaterial(const std::string& shaderCode,
                                                          const std::string& shaderFileName,
                                                          const MaterialBuilderData& data) {
  MaterialBuilderData compiledData = data;
  compiledData.shaderCode = compileSlangToSpirv(shaderCode, shaderFileName, data.vertEntryFuncName, data.fragEntryFuncName);
  return impl->mMaterialPool.Add({.data = compiledData});
}

void MapleRenderer::DestroyMaterial(MaterialHndl handle) { impl->mMaterialPool.Remove(handle); }

void MapleRenderer::CreateTexture(const std::string& name, glm::uvec2 dimensions, std::span<const uint8_t> bytes, uint8_t pixelSize, Format format) {
  auto& ctx = impl->mCtx;

  vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
  auto initialLayout = vk::ImageLayout::eUndefined;

  auto img = ctx.mAllocator.CreateImage({
    .extent = {dimensions.x, dimensions.y, 1},
    .usage = usage,
    .initialLayout = initialLayout,
  });

  auto stageBuf = ctx.mAllocator.CreateBuffer(bytes.size(), vkm::Allocator::Stage);
  stageBuf.Upload(bytes.data(), bytes.size());

  auto cmd = ctx.beginSingleTimeCommands();
  img.TransitionLayout(cmd, initialLayout, vk::ImageLayout::eTransferDstOptimal);
  img.UploadBuffer(cmd, stageBuf.buffer, dimensions.x, dimensions.y);
  img.TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal, ToVulkan(ImageLayout::ShaderReadOnlyOptimal));
  ctx.endSingleTimeCommands(cmd);

  auto hndl = impl->mRenderTargets.Add({
    .info =
      {
        .sizeType = SizeType::Absolute,
        .size = dimensions,
        .format = format,
      },
    .target = std::move(img),
  });

  impl->mRenderTargetMap[name] = hndl;
}

void MapleRenderer::DestroyTexture(const std::string& name) {
  auto it = impl->mRenderTargetMap.find(name);

  MAPLE_ASSERT(it != impl->mRenderTargetMap.end(), "failed to find texture '{}' in DestroyTexture", name);
  auto hndl = it->second;
  impl->mRenderTargetMap.erase(it);

  MAPLE_ASSERT(impl->mRenderTargets.IsValid(hndl), "texture handle for '{}' was not valid", name);
  impl->mRenderTargets.Remove(hndl);
}

// FrameIdx, SwapChainIdx
std::optional<std::pair<uint8_t, uint32_t>> acquireFrameIdxAndSwapChainIdx(const VkRendererCtx& ctx) {
  static uint8_t frameNumber = 0;
  uint8_t frameIdx = frameNumber % ctx.MAX_FRAMES_IN_FLIGHT;

  auto& frameData = ctx.mFrameData[frameIdx];

  auto fenceResult = ctx.mDevice.device.waitForFences(*frameData.drawFence, vk::True, 0);
  if (fenceResult == vk::Result::eTimeout) return std::nullopt;

  auto [swapChainResult, swapChainImageIdx] = ctx.mSwapChain.swapchain.acquireNextImage(0, frameData.presentCompleteSem, nullptr);
  if (swapChainResult == vk::Result::eNotReady) return std::nullopt;

  ctx.mDevice.device.resetFences(*frameData.drawFence);
  if (swapChainResult != vk::Result::eSuccess && swapChainResult != vk::Result::eSuboptimalKHR) MAPLE_FATAL("Failed to acquire swapchain image");

  frameNumber++;
  return std::make_pair(frameIdx, swapChainImageIdx);
};

void CreateMissingAttachments(const std::vector<RenderGraph::NameAndAttachment>& requiredAttachments,
                              Pool<RenderTarget>& renderTargets,
                              std::unordered_map<std::string, MapleRenderer::RenderTargetHndl>& map,
                              VkRendererCtx& ctx) {
  glm::uvec2 swapChainSize(ctx.mSwapChain.extent.width, ctx.mSwapChain.extent.height);

  for (auto& v : requiredAttachments) {
    // if (!attachmentExists(v)) continue;
    if (v.name == RenderGraph::SWAPCHAIN_TARGET_NAME) continue;  // Exclude attachment creation for the swapchain
    auto it = map.find(v.name);
    if (it != map.end()) continue;

    vk::ImageUsageFlags usage =
      FormatIsColor(v.info.format) ? vk::ImageUsageFlagBits::eColorAttachment : vk::ImageUsageFlagBits::eDepthStencilAttachment;
    usage |= vk::ImageUsageFlagBits::eSampled;  // all render targets assumed to be sampleable cus of bindless

    auto size = v.info.GetAbsoluteSize(swapChainSize);
    auto hndl = renderTargets.Add(RenderTarget{
      .info = v.info,
      .target = ctx.mAllocator.CreateImage({
        .format = ToVulkan(v.info.format),
        .extent = {size.x, size.y, 1},
        .usage = usage,
        .aspectMask = GetImageAspectFlags(v.info.format),
      }),
    });

    map[v.name] = hndl;
  }
}

void MapleRenderer::DrawFrame(const UBO& frameUBO, const RenderGraph::CompileResult& compiledRenderGraph, std::span<const PassDraw> passDraws) {
  auto& ctx = impl->mCtx;
  // TODO: manage and remove unused attachments
  CreateMissingAttachments(compiledRenderGraph.attachments, impl->mRenderTargets, impl->mRenderTargetMap, ctx);

  auto result = acquireFrameIdxAndSwapChainIdx(ctx);
  if (!result.has_value()) return;  // timeout
  auto [frameIdx, swapChainImageIdx] = result.value();
  auto& frameData = ctx.mFrameData[frameIdx];
  auto& cmd = frameData.cmd;

  impl->mGlobalsUniform[frameIdx].Upload(&frameUBO, sizeof(frameUBO));

  // TODO: optimize
  uint32_t i = 0;
  uint32_t updated = 0;
  uint32_t activeCount = impl->mRenderTargets.ActiveCount();
  while (updated != activeCount) {
    if (!impl->mRenderTargets.IsValid(i)) continue;

    vk::DescriptorImageInfo imgInfo{};
    imgInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imgInfo.imageView = impl->mRenderTargets.Get(i).target.view;
    imgInfo.sampler = impl->mDefaultSampler.sampler;

    vk::WriteDescriptorSet write{};
    write.dstSet = *impl->mGlobalDescriptorSets.sets[frameIdx], write.dstBinding = 3, write.dstArrayElement = i, write.descriptorCount = 1,
    write.descriptorType = vk::DescriptorType::eCombinedImageSampler, write.pImageInfo = &imgInfo,

    ctx.mDevice.device.updateDescriptorSets(write, {});
    updated++;
    i++;
  }

  cmd.reset();
  cmd.begin({});
  cmd.bindDescriptorSets2({
    .sType = vk::StructureType::eBindDescriptorSetsInfo,
    .pNext = nullptr,
    .stageFlags = ToVulkan(ShaderStage::AllGraphics),
    .layout = impl->mGlobalPipelineLayout.GetLayout(),
    .firstSet = 0,
    .descriptorSetCount = 1,
    .pDescriptorSets = &(*impl->mGlobalDescriptorSets.sets[frameIdx]),
    .dynamicOffsetCount = 0,
    .pDynamicOffsets = nullptr,
  });

  std::vector<glm::mat4> instanceData;
  instanceData.reserve(NUM_INSTANCES);

  std::vector<uint32_t> materialBuffer;
  materialBuffer.reserve(NUM_MATERIALS);

  for (auto& pass : compiledRenderGraph.passes) {
    std::vector<vk::ImageMemoryBarrier2> barriers(pass.preTransitions.size());
    for (auto [i, transition] : std::views::enumerate(pass.preTransitions)) {
      auto getImgAndAspect = [&](const std::string& name) -> std::pair<vk::Image, vk::ImageAspectFlags> {
        if (name == RenderGraph::SWAPCHAIN_TARGET_NAME)
          return std::make_pair(ctx.mSwapChain.images[swapChainImageIdx].img, vk::ImageAspectFlagBits::eColor);
        auto& v = impl->mRenderTargets.Get(impl->mRenderTargetMap.at(transition.resource));
        return std::make_pair(*v.target.img, GetImageAspectFlags(v.info.format));
      };
      auto imgAndAspectFlags = getImgAndAspect(transition.resource);

      barriers[i] = vk::ImageMemoryBarrier2{
        .srcStageMask = ToVulkan(transition.oldState.stage),
        .srcAccessMask = ToVulkan(transition.oldState.access),
        .dstStageMask = ToVulkan(transition.newState.stage),
        .dstAccessMask = ToVulkan(transition.newState.access),
        .oldLayout = ToVulkan(transition.oldState.layout),
        .newLayout = ToVulkan(transition.newState.layout),
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = imgAndAspectFlags.first,
        .subresourceRange = {.aspectMask = imgAndAspectFlags.second, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
      };
    }
    vk::DependencyInfo dependencyInfo{
      .dependencyFlags = {}, .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()), .pImageMemoryBarriers = barriers.data()};
    cmd.pipelineBarrier2(dependencyInfo);

    if (pass.outputs.empty()) {
      continue;  // it's a barrier-only pass (e.g present transition of swapchain), continue
    }

    std::vector<vk::RenderingAttachmentInfo> colorAttachments;
    colorAttachments.reserve(pass.outputs.size());
    std::optional<vk::RenderingAttachmentInfo> depthAttachment = std::nullopt;

    auto getImageView = [&](const std::string& name) -> vk::ImageView {
      if (name == RenderGraph::SWAPCHAIN_TARGET_NAME) return ctx.mSwapChain.images[swapChainImageIdx].view;
      return impl->mRenderTargets.Get(impl->mRenderTargetMap.at(name)).target.view;
    };

    std::optional<glm::uvec2> renderingArea = std::nullopt;

    for (auto& out : pass.outputs) {
      auto attachmentSize = out.info.GetAbsoluteSize(glm::uvec2(ctx.mSwapChain.extent.width, ctx.mSwapChain.extent.height));

      if (!renderingArea.has_value()) renderingArea = attachmentSize;
      MAPLE_ASSERT(renderingArea.value() == attachmentSize, "attachments in pass with unequal sizes, attachment '{}'", out.name);

      if (FormatIsDepth(out.info.format)) {
        MAPLE_ASSERT(!depthAttachment.has_value(), "attempted to use multiple depth outputs in a single pass, attachment '{}'", out.name);

        depthAttachment = vk::RenderingAttachmentInfo{
          .imageView = getImageView(out.name),
          .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,            // TODO: make this configurable
          .storeOp = vk::AttachmentStoreOp::eStore,          // TODO: make this configurable
          .clearValue = vk::ClearDepthStencilValue{1.0f, 0}  // TODO: make this configurable
        };
      } else {
        colorAttachments.push_back({
          .imageView = getImageView(out.name),
          .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .loadOp = vk::AttachmentLoadOp::eClear,                    // TODO: make this configurable
          .storeOp = vk::AttachmentStoreOp::eStore,                  // TODO: make this configurable
          .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}  // TODO: make this configurable
        });
      }
    }

    cmd.beginRendering({
      .renderArea = {.offset = {0, 0}, .extent = {renderingArea->x, renderingArea->y}},
      .layerCount = 1,
      .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
      .pColorAttachments = colorAttachments.data(),
      .pDepthAttachment = depthAttachment.has_value() ? &depthAttachment.value() : nullptr,
    });

    auto& passName = pass.name;
    const std::span<const MaterialDraw>* materialDraws = nullptr;
    for (auto& passDraw : passDraws) {
      if (passDraw.passName == passName) {
        MAPLE_ASSERT(materialDraws == nullptr, "duplicate material draw for pass '{}'", passName);
        materialDraws = &passDraw.materialDraws;
      }
    }

    if (materialDraws) {
      std::vector<vk::Format> outputColorFormats;
      std::optional<vk::Format> outputDepthFormat;
      outputColorFormats.reserve(pass.outputs.size());
      for (auto& output : pass.outputs) {
        if (FormatIsDepth(output.info.format)) {
          outputDepthFormat = ToVulkan(output.info.format);  // above we already make sure the pass only contains 1 depth attachment
        } else {
          // TODO: later add string output names to material info, to make sure we define the attachments in the right order
          // index 0 in the array will be location 0 of the output attachment

          // if the target is the swapchain, dynamically fetch the actual format
          auto format = output.name == RenderGraph::SWAPCHAIN_TARGET_NAME ? ctx.mSwapChain.format.format : ToVulkan(output.info.format);
          outputColorFormats.push_back(format);
        }
      }

      for (auto& materialDraw : *materialDraws) {
        auto& mat = impl->mMaterialPool.Get(materialDraw.material);
        // TODO: assert pass.pipelineType == materialDraw.material's pipeline type

        if (!mat.pipeline.has_value()) {
          mat.pipeline = vkm::Pipeline(vkm::Pipeline::CreateInfo{
            .device = ctx.mDevice.device,
            .layout = impl->mGlobalPipelineLayout,
            .formats =
              vkm::Pipeline::AttachmentFormats{
                .colorFormats = outputColorFormats,
                .depthFormat = outputDepthFormat,
              },
            .materialData = mat.data,
          });
        }

        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, mat.pipeline.value().GetPipeline());
        cmd.setViewport(0, vk::Viewport{0.0f, 0.0f, static_cast<float>(renderingArea->x), static_cast<float>(renderingArea->y), 0.0f, 1.0f});
        vk::Rect2D scissor{vk::Offset2D{0, 0}, vk::Extent2D{.width = renderingArea->x, .height = renderingArea->y}};
        cmd.setScissor(0, {scissor});

        for (auto& meshDraw : materialDraw.meshes) {
          auto& mesh = impl->mMeshPool.Get(meshDraw.mesh);

          uint32_t materialBufferOffset = materialBuffer.size();
          for (auto& usedResource : meshDraw.usedResources) {
            MAPLE_ASSERT(usedResource != RenderGraph::SWAPCHAIN_TARGET_NAME, "cannot use swapchain as sampled attachment");
            auto it = impl->mRenderTargetMap.find(usedResource);
            MAPLE_ASSERT(it != impl->mRenderTargetMap.end(), "failed to find meshDraw resource '{}'", usedResource);
            RenderTargetHndl hndl = it->second;
            MAPLE_ASSERT(impl->mRenderTargets.IsValid(hndl), "invalid meshDraw resource '{}'", usedResource);

            auto hndlNumBytes = sizeof(hndl);
            size_t offset = materialBuffer.size() * sizeof(decltype(materialBuffer)::value_type);
            materialBuffer.resize(offset + hndlNumBytes);
            std::memcpy(materialBuffer.data() + offset, &hndl, hndlNumBytes);
          }

          VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR};
          info.buffer = *mesh.meshBuffer.buffer;

          DrawPush push{
            .vertexBufferAddress = vkGetBufferDeviceAddress(*ctx.mDevice.device, &info),
            .indexBufferOffset = mesh.GetIndexBufferOffset(),
            .materialBufferOffset = materialBufferOffset,
            .instanceBufferIndex = static_cast<uint32_t>(instanceData.size()),
          };

          for (auto& instance : meshDraw.instanceData) instanceData.push_back(instance);

          // TODO: optimize this depending on if its a graphics pipeline or a compute pipeline
          auto stageFlags = ToVulkan(ShaderStage::AllGraphicsAndCompute);
          cmd.pushConstants<DrawPush>(impl->mGlobalPipelineLayout.GetLayout(), stageFlags, 0, push);
          cmd.draw(mesh.GetNumIndices(), meshDraw.instanceData.size(), 0, 0);  // non-indexed, emulated indexed drawing
        }
      }
    }

    cmd.endRendering();
  }

  cmd.end();

  impl->mInstanceSSBO[frameIdx].Upload(instanceData.data(), instanceData.size() * sizeof(decltype(instanceData)::value_type));
  impl->mMaterialBuffers[frameIdx].Upload(materialBuffer.data(), materialBuffer.size());

  vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  vk::SubmitInfo submitInfo{
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &*frameData.presentCompleteSem,
    .pWaitDstStageMask = &waitDestinationStageMask,
    .commandBufferCount = 1,
    .pCommandBuffers = &*frameData.cmd,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &*ctx.mRenderCompleteSems[swapChainImageIdx],
  };
  ctx.mDevice.queues.graphics.submit(submitInfo, frameData.drawFence);

  vk::PresentInfoKHR presentInfo{
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &*ctx.mRenderCompleteSems[swapChainImageIdx],
    .swapchainCount = 1,
    .pSwapchains = &*ctx.mSwapChain.swapchain,
    .pImageIndices = &swapChainImageIdx,
  };

  auto presentResult = ctx.mDevice.queues.present.presentKHR(presentInfo);
  if (mFrameBufferResized || presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR) {
    mFrameBufferResized = false;
    ctx.mSwapChain.ReCreate({
      .physicalDevice = ctx.mPhysicalDevice,
      .device = ctx.mDevice,
      .surface = ctx.mSurface,
      .allocator = ctx.mAllocator,
      .framebufferSizeCb = ctx.mFrameBufferSizeCallback,
    });

    // swapchain recreation above already calls device.waitIdle() so these attachments won't be use-after-freed
    glm::uvec2 swapChainSize(ctx.mSwapChain.extent.width, ctx.mSwapChain.extent.height);
    auto multiplySwapChain = [&](glm::vec2 v) { return glm::uvec2(v.x * swapChainSize.x, v.y * swapChainSize.y); };

    auto& renderTargets = impl->mRenderTargets;
    // TODO: optimize
    for (uint32_t i = 0; i < renderTargets.Capacity(); i++) {
      if (!renderTargets.IsValid(i)) continue;
      auto& rt = renderTargets.Get(i);
      if (rt.info.sizeType != SwapChainRelative) continue;

      glm::uvec2 size = multiplySwapChain(rt.info.size);
      vk::ImageUsageFlags usage =
        FormatIsColor(rt.info.format) ? vk::ImageUsageFlagBits::eColorAttachment : vk::ImageUsageFlagBits::eDepthStencilAttachment;
      usage |= vk::ImageUsageFlagBits::eSampled;  // all render targets assumed to be sampleable cus of bindless

      rt.target = ctx.mAllocator.CreateImage({
        .format = ToVulkan(rt.info.format),
        .extent = {size.x, size.y, 1},
        .usage = usage,
        .aspectMask = GetImageAspectFlags(rt.info.format),
      });
    }

  } else if (presentResult != vk::Result::eSuccess) {
    MAPLE_FATAL("Failed to present swap chain image");
  }
}

MapleRenderer::MapleRenderer() : impl(std::make_unique<Impl>()) {}
MapleRenderer::MapleRenderer(const std::vector<const char*>& glfwExtensions,
                             SurfaceCreateCallback surfaceCb,
                             FrameBufferSizeCallback frameBufferSizeCb)
    : impl(std::make_unique<Impl>()) {
  auto& ctx = impl->mCtx;
  ctx.Init(glfwExtensions, surfaceCb, frameBufferSizeCb);

  impl->mGlobalDescriptorPool = vkm::DescriptorPool(vkm::DescriptorPool::CreateInfo{
    .device = ctx.mDevice.device,
    .maxSets = ctx.MAX_FRAMES_IN_FLIGHT,
    .resourceSizes =
      {
        std::make_pair(vk::DescriptorType::eUniformBuffer, ctx.MAX_FRAMES_IN_FLIGHT),
        std::make_pair(vk::DescriptorType::eStorageBuffer, ctx.MAX_FRAMES_IN_FLIGHT * 2),
        std::make_pair(vk::DescriptorType::eCombinedImageSampler, ctx.MAX_FRAMES_IN_FLIGHT * MAX_BINDLESS_TEXTURES),
      },
  });

  std::array description = {
    vkm::DescriptorSets::Layout{
      .bindingSlot = 0, .type = vkm::DescriptorSets::Type::Uniform, .usedStages = ShaderStage::AllGraphicsAndCompute},  // UBO buffer
    vkm::DescriptorSets::Layout{
      .bindingSlot = 1, .type = vkm::DescriptorSets::Type::SSBO, .usedStages = ShaderStage::AllGraphicsAndCompute},  // Instance buffer
    vkm::DescriptorSets::Layout{
      .bindingSlot = 2, .type = vkm::DescriptorSets::Type::SSBO, .usedStages = ShaderStage::AllGraphicsAndCompute},  // Material buffer
    // Texture array
    vkm::DescriptorSets::Layout{
      .bindingSlot = 3,
      .type = vkm::DescriptorSets::Type::CombinedImageSampler,
      .usedStages = ShaderStage::AllGraphicsAndCompute,
      .arrayCount = MAX_BINDLESS_TEXTURES,
    },
  };
  impl->mGlobalDescriptorSets = vkm::DescriptorSets(vkm::DescriptorSets::CreateInfo{
    .device = ctx.mDevice.device,
    .pool = impl->mGlobalDescriptorPool.pool,
    .count = ctx.MAX_FRAMES_IN_FLIGHT,
    .description = description,
  });

  for (size_t i = 0; i < ctx.MAX_FRAMES_IN_FLIGHT; i++) {
    impl->mInstanceSSBO[i] = ctx.mAllocator.CreateBuffer(sizeof(glm::mat4) * NUM_INSTANCES, vkm::Allocator::SSBO);
    impl->mGlobalsUniform[i] = ctx.mAllocator.CreateBuffer(sizeof(UBO), vkm::Allocator::UBO);
    impl->mMaterialBuffers[i] = ctx.mAllocator.CreateBuffer(NUM_MATERIALS, vkm::Allocator::SSBO);
  }

  impl->mGlobalPipelineLayout = vkm::PipelineLayout(vkm::PipelineLayout::Info{
    .device = ctx.mDevice.device,
    .pushConstantInfo =
      vkm::PipelineLayout::PushConstantInfo{
        .stage = ShaderStage::AllGraphicsAndCompute,
        .size = sizeof(DrawPush),
      },
    .descriptorSetLayout = impl->mGlobalDescriptorSets.layout,
  });

  impl->mDefaultSampler = vkm::Sampler(ctx.mDevice.device, {.maxAnisotropy = ctx.mPhysicalDevice.GetProperties().limits.maxSamplerAnisotropy});

  // Writing descriptor sets
  for (size_t i = 0; i < ctx.MAX_FRAMES_IN_FLIGHT; i++) {
    // UBO (binding 0)
    vk::DescriptorBufferInfo uboInfo{
      .buffer = *impl->mGlobalsUniform[i].buffer,
      .offset = 0,
      .range = VK_WHOLE_SIZE  // or sizeof(UBO) if you want explicit size
    };
    vk::WriteDescriptorSet writeUbo{.dstSet = *impl->mGlobalDescriptorSets.sets[i],
                                    .dstBinding = 0,
                                    .dstArrayElement = 0,
                                    .descriptorCount = 1,
                                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                                    .pBufferInfo = &uboInfo};

    // Instance SSBO (binding 1)
    vk::DescriptorBufferInfo instanceInfo{.buffer = *impl->mInstanceSSBO[i].buffer, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::WriteDescriptorSet writeInstance{.dstSet = *impl->mGlobalDescriptorSets.sets[i],
                                         .dstBinding = 1,
                                         .dstArrayElement = 0,
                                         .descriptorCount = 1,
                                         .descriptorType = vk::DescriptorType::eStorageBuffer,
                                         .pBufferInfo = &instanceInfo};

    // Material SSBO (binding 2)
    vk::DescriptorBufferInfo materialInfo{.buffer = *impl->mMaterialBuffers[i].buffer, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::WriteDescriptorSet writeMaterial{.dstSet = *impl->mGlobalDescriptorSets.sets[i],
                                         .dstBinding = 2,
                                         .dstArrayElement = 0,
                                         .descriptorCount = 1,
                                         .descriptorType = vk::DescriptorType::eStorageBuffer,
                                         .pBufferInfo = &materialInfo};

    std::array writes = {writeUbo, writeInstance, writeMaterial};
    ctx.mDevice.device.updateDescriptorSets(writes, {});
  }
}
MapleRenderer::~MapleRenderer() {
  if (impl) {
    impl->mCtx.Destroy();
  }
};

MapleRenderer::MapleRenderer(MapleRenderer&&) noexcept = default;
MapleRenderer& MapleRenderer::operator=(MapleRenderer&&) noexcept = default;
}  // namespace maple