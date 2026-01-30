#include <engine/file/file.h>
#include <engine/logging/log_macros.h>
#include <engine/renderer/renderer.h>
#include <vulkan/vulkan.h>

#include <map>
#include <ranges>
#include <set>

#include "vk_utils.h"

const std::vector<const char*> validationLayers = {
#ifndef NDEBUG
    "VK_LAYER_KHRONOS_validation"
#endif
};

const std::vector<const char*> REQUIRED_DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                          VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                          const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
  MAPLE_DEBUG("Vulkan validation layer: {}", pCallbackData->pMessage);

  return VK_FALSE;
}

const int MAX_FRAMES_IN_FLIGHT = 2;

namespace maple {

struct Renderer::Impl {
  VkInstance mInstance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;
  VkSurfaceKHR mSurface = VK_NULL_HANDLE;
  size_t mSelectedDeviceIdx = 0;
  VkDevice mDevice = VK_NULL_HANDLE;
  VkSwapchainKHR mSwapChain = VK_NULL_HANDLE;
  std::vector<VkImage> mSwapChainImages;
  uint32_t mPresentModeIdx, mSurfaceFormatIdx;
  VkExtent2D mExtent;

  std::vector<VkExtensionProperties> mAvailableInstanceExtensions;
  std::vector<VkLayerProperties> mAvailableInstanceLayers;

  std::vector<PhysicalDevice> mPhysicalDevices;

  struct QueueIndices {
    uint32_t Graphics, Present;
  };
  QueueIndices mQueueIndices;
  struct QueueHandles {
    VkQueue Graphics, Present;
  };
  QueueHandles mQueueHandles;

  std::vector<VkImageView> mSwapChainImageViews;

  VkRenderPass mRenderPass;
  VkShaderModule mVertShader, mFragShader;
  VkPipelineLayout mPipelineLayout;
  VkPipeline mPipeline;

  std::vector<VkFramebuffer> mSwapChainFramebuffers;

  VkCommandPool mCommandPool;
  std::vector<VkCommandBuffer> mCommandBuffers;

  std::vector<VkSemaphore> mImageAvailableSems, mRenderFinishedSems;
  std::vector<VkFence> mInFlightFences;

  uint32_t mCurrentFrame = 0;

  FramebufferSizeCallback mFramebufferSizeCallback;

  bool framebufferResized = false;

  void Init(const std::vector<const char*>& requiredExtensions, SurfaceCreateCallback surfaceCreateCallback,
            FramebufferSizeCallback framebufferSizeCallback) {
    MAPLE_INFO("Initializing Renderer...");
    mFramebufferSizeCallback = framebufferSizeCallback;

    probeInstanceExtensions();
    probeInstanceLayers();
    createVulkanInstance(requiredExtensions);
    setupDebugCallback();

    mSurface = surfaceCreateCallback(mInstance);

    probePhysicalDevices();
    selectPhysicalDevice();
    createLogicalDevice();

    // there may be a slight overhead if the graphics and present queues are different families (synchronization issues, memory transfer
    // weirdness on some devices). currently will only use the graphics queue to check for present capabilities. correct approach would be
    // to check that, and if the graphics queue doesn't have present, probe other queue families
    if (mQueueIndices.Graphics != mQueueIndices.Present)
      MAPLE_FATAL("Different Graphics and Present queue families, separate families not implemented");

    vkGetDeviceQueue(mDevice, mQueueIndices.Graphics, 0, &mQueueHandles.Graphics);
    vkGetDeviceQueue(mDevice, mQueueIndices.Present, 0, &mQueueHandles.Present);

    createSwapChain();
    createImageViews();

    createRenderPass();
    createGraphicsPipeline();

    createFramebuffers();

    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
  }

  void DrawFrame() {
    vkWaitForFences(mDevice, 1, &mInFlightFences[mCurrentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
    vkResetFences(mDevice, 1, &mInFlightFences[mCurrentFrame]);

    uint32_t imageIdx;
    vkAcquireNextImageKHR(mDevice, mSwapChain, std::numeric_limits<uint64_t>::max(), mImageAvailableSems[mCurrentFrame], nullptr, &imageIdx);

    recordCommandBuffer(mCommandBuffers[mCurrentFrame], imageIdx);

    VkSemaphore waitSemaphores[] = {mImageAvailableSems[mCurrentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signalSemaphores[] = {mRenderFinishedSems[mCurrentFrame]};
    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = waitSemaphores,
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = &mCommandBuffers[mCurrentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = signalSemaphores,
    };

    if (vkQueueSubmit(mQueueHandles.Graphics, 1, &submitInfo, mInFlightFences[mCurrentFrame]) != VK_SUCCESS)
      MAPLE_FATAL("Failed to submit draw command buffer");

    VkSwapchainKHR swapchains[] = {mSwapChain};
    VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = signalSemaphores,
        .swapchainCount = 1,
        .pSwapchains = swapchains,
        .pImageIndices = &imageIdx,
    };

    VkResult result = vkQueuePresentKHR(mQueueHandles.Present, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
      if (framebufferResized)
        MAPLE_DEBUG("framebufferResized was set, recreating swap chain");
      else
        MAPLE_DEBUG("vkQueuePresentKHR returned {}, recreating swap chain", result == VK_ERROR_OUT_OF_DATE_KHR ? "VK_ERROR_OUT_OF_DATE_KHR" : "VK_SUBOPTIMAL_KHR");

      framebufferResized = false;
      recreateSwapChain();
    } else if (result != VK_SUCCESS) {
      MAPLE_FATAL("Failed to present swapchain image");
    }

    mCurrentFrame = (mCurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void SetFrameBufferResized() { framebufferResized = true; }

  void Destroy() {
    MAPLE_INFO("Cleaning Renderer...");

    vkDeviceWaitIdle(mDevice);  // Wait for device to become idle so semaphores and fences used while being destroyed

    cleanupSwapChain();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(mDevice, mImageAvailableSems[i], nullptr);
      vkDestroySemaphore(mDevice, mRenderFinishedSems[i], nullptr);
      vkDestroyFence(mDevice, mInFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(mDevice, mCommandPool, nullptr);

    vkDestroyPipeline(mDevice, mPipeline, nullptr);
    vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
    vkDestroyRenderPass(mDevice, mRenderPass, nullptr);

    vkDestroyDevice(mDevice, nullptr);

    vkDestroySurfaceKHR(mInstance, mSurface, nullptr);

    if (!validationLayers.empty()) DestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
    vkDestroyInstance(mInstance, nullptr);
  }

  void recreateSwapChain() {
    vkDeviceWaitIdle(mDevice);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createFramebuffers();
  }

  void cleanupSwapChain() {
    for (auto fb : mSwapChainFramebuffers) vkDestroyFramebuffer(mDevice, fb, nullptr);
    for (auto imgView : mSwapChainImageViews) vkDestroyImageView(mDevice, imgView, nullptr);
    vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);
  }

  void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIdx) {
    vkResetCommandBuffer(commandBuffer, 0);

    VkCommandBufferBeginInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

    if (vkBeginCommandBuffer(commandBuffer, &info) != VK_SUCCESS) MAPLE_FATAL("Failed to begin recording command buffer");

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    VkRenderPassBeginInfo renderPassInfo{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                         .renderPass = mRenderPass,
                                         .framebuffer = mSwapChainFramebuffers[imageIdx],
                                         .renderArea =
                                             {
                                                 .offset = {0, 0},
                                                 .extent = mExtent,
                                             },
                                         .clearValueCount = 1,
                                         .pClearValues = &clearColor};

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline);

    VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(mExtent.width),
        .height = static_cast<float>(mExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{
        .offset = {0, 0},
        .extent = mExtent,
    };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) MAPLE_FATAL("Failed to end recording of command buffer");
  }

  void createSyncObjects() {
    VkSemaphoreCreateInfo semInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fenceInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,  // Create it in a signaled state initially, so the first frame's rendering doesn't block indefinitely
    };

    mImageAvailableSems.resize(MAX_FRAMES_IN_FLIGHT);
    mRenderFinishedSems.resize(MAX_FRAMES_IN_FLIGHT);
    mInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkCreateSemaphore(mDevice, &semInfo, nullptr, &mImageAvailableSems[i]);
      vkCreateSemaphore(mDevice, &semInfo, nullptr, &mRenderFinishedSems[i]);
      vkCreateFence(mDevice, &fenceInfo, nullptr, &mInFlightFences[i]);
    }
  }

  void createCommandBuffers() {
    mCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = mCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = static_cast<uint32_t>(mCommandBuffers.size()),
    };

    if (vkAllocateCommandBuffers(mDevice, &allocInfo, mCommandBuffers.data()) != VK_SUCCESS) MAPLE_FATAL("Failed to create command buffers");
  }

  void createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = mQueueIndices.Graphics,
    };

    if (vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool) != VK_SUCCESS) MAPLE_FATAL("failed to create command pool");
  }

  void createFramebuffers() {
    mSwapChainFramebuffers.resize(mSwapChainImageViews.size());

    for (auto [i, v] : std::views::enumerate(mSwapChainImageViews)) {
      VkImageView attachments[] = {v};
      VkFramebufferCreateInfo framebufferInfo{
          .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
          .renderPass = mRenderPass,
          .attachmentCount = 1,
          .pAttachments = attachments,
          .width = mExtent.width,
          .height = mExtent.height,
          .layers = 1,
      };
      if (vkCreateFramebuffer(mDevice, &framebufferInfo, nullptr, &mSwapChainFramebuffers[i]) != VK_SUCCESS)
        MAPLE_FATAL("Failed to create swapchain framebuffer");
    }
  }

  void createGraphicsPipeline() {
    auto vCode = file::ReadFile("assets/shaders/vert.spv");
    auto fCode = file::ReadFile("assets/shaders/frag.spv");

    auto vertShaderModule = CreateShaderModule(mDevice, vCode);
    auto fragShaderModule = CreateShaderModule(mDevice, fCode);

    VkPipelineShaderStageCreateInfo vCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertShaderModule,
        .pName = "main",  // Configurable entry point function name
    };

    VkPipelineShaderStageCreateInfo fCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragShaderModule,
        .pName = "main",  // Configurable entry point function name
    };

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .vertexAttributeDescriptionCount = 0,
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(mExtent.width),
        .height = static_cast<float>(mExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    VkRect2D scissor{
        .offset = {0, 0},
        .extent = mExtent,
    };

    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = false,
    };

    VkPipelineColorBlendAttachmentState colorBlendAttach{
        .blendEnable = false,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo colorBlending{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttach,
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };

    if (vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayout) != VK_SUCCESS)
      MAPLE_FATAL("Failed to create pipeline layout");

    VkPipelineShaderStageCreateInfo stages[] = {vCreateInfo, fCreateInfo};

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicStateCreateInfo,
        .layout = mPipelineLayout,
        .renderPass = mRenderPass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(mDevice, nullptr, 1, &pipelineInfo, nullptr, &mPipeline) != VK_SUCCESS) MAPLE_FATAL("Failed to create pipeline");

    vkDestroyShaderModule(mDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(mDevice, fragShaderModule, nullptr);
  }

  void createRenderPass() {
    VkAttachmentDescription colorAttach{
        .format = mPhysicalDevices[mSelectedDeviceIdx].surfaceFormats[mSurfaceFormatIdx].format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    VkAttachmentReference colorAttachRef{
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass{
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachRef,
    };

    VkSubpassDependency dependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    VkRenderPassCreateInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorAttach,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    if (vkCreateRenderPass(mDevice, &renderPassInfo, nullptr, &mRenderPass) != VK_SUCCESS) MAPLE_FATAL("Failed to create render pass");
  }

  void createImageViews() {
    uint32_t imageCount = 0;
    vkGetSwapchainImagesKHR(mDevice, mSwapChain, &imageCount, nullptr);
    mSwapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(mDevice, mSwapChain, &imageCount, mSwapChainImages.data());

    mSwapChainImageViews.resize(imageCount);

    for (auto [i, v] : std::views::enumerate(mSwapChainImageViews)) {
      VkImageViewCreateInfo createInfo{
          .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
          .image = mSwapChainImages[i],
          .viewType = VK_IMAGE_VIEW_TYPE_2D,
          .format = mPhysicalDevices[mSelectedDeviceIdx].surfaceFormats[mSurfaceFormatIdx].format,
          .components =
              {
                  .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                  .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                  .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                  .a = VK_COMPONENT_SWIZZLE_IDENTITY,
              },
          .subresourceRange =
              {
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .baseMipLevel = 0,
                  .levelCount = 1,
                  .baseArrayLayer = 0,
                  .layerCount = 1,
              },
      };

      if (vkCreateImageView(mDevice, &createInfo, nullptr, &v) != VK_SUCCESS) MAPLE_FATAL("Failed to create image view");
    }
  }

  void createSwapChain() {
    uint32_t framebufferWidth, framebufferHeight;
    mFramebufferSizeCallback(framebufferWidth, framebufferHeight);
    MAPLE_DEBUG("Creating swapchain with framebuffer size {}x{}", framebufferWidth, framebufferHeight);

    const auto& dev = mPhysicalDevices[mSelectedDeviceIdx];

    mPresentModeIdx = ChooseOptimalPresentMode(dev.presentModes);
    mSurfaceFormatIdx = ChooseOptimalSurfaceFormat(dev.surfaceFormats);
    mExtent = ChooseOptimalSwapExtent(dev.surfaceCapabilities, framebufferWidth, framebufferHeight);

    uint32_t imageCount = dev.surfaceCapabilities.minImageCount + 1;
    if (dev.surfaceCapabilities.maxImageCount > 0) imageCount = std::min(imageCount, dev.surfaceCapabilities.maxImageCount);

    VkSwapchainCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = mSurface,
        .minImageCount = imageCount,
        .imageFormat = dev.surfaceFormats[mSurfaceFormatIdx].format,
        .imageColorSpace = dev.surfaceFormats[mSurfaceFormatIdx].colorSpace,
        .imageExtent = mExtent,
        .imageArrayLayers = 1,  // always one, unless we're developing 3D applications where each image consists of multiple layers
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,  // for now will render directly to swapchain images but later on we'll render to
                                                            // another image for post processing. in that case,
                                                            // VK_IMAGE_USAGE_TRANSFER_DST_BIT is the more ideal option.
                                                            // specifically need the value of those pixels, so enable it for better performance
        .preTransform = dev.surfaceCapabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = dev.presentModes[mPresentModeIdx],
        .clipped = VK_TRUE,  // means we don't care about pixels that are obscured (for example behind another window). we don't
        .oldSwapchain = VK_NULL_HANDLE,
    };

    uint32_t queueFamilyIndices[] = {mQueueIndices.Graphics, mQueueIndices.Present};
    if (mQueueIndices.Graphics != mQueueIndices.Present) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    if (vkCreateSwapchainKHR(mDevice, &createInfo, nullptr, &mSwapChain) != VK_SUCCESS) MAPLE_FATAL("Failed to create swapchain");
  }

  void createLogicalDevice() {
    const auto& caps = mPhysicalDevices[mSelectedDeviceIdx].queueFamiliesCapabilities;
    std::vector<GraphicsQueueCapabilityType> requiredCaps = {GraphicsQueueCapabilityType::GRAPHICS, GraphicsQueueCapabilityType::PRESENT};

    const auto graphicsFamilyIdx = GetQueueFamilyIdxWithCapability(caps, requiredCaps[0]);
    if (!graphicsFamilyIdx.has_value()) MAPLE_FATAL("Failed to find graphics queue family for device");
    mQueueIndices.Graphics = graphicsFamilyIdx.value();

    const auto presentFamilyIdx = GetQueueFamilyIdxWithCapability(caps, requiredCaps[1]);
    if (!presentFamilyIdx.has_value()) MAPLE_FATAL("Failed to find present queue family for device");
    mQueueIndices.Present = presentFamilyIdx.value();

    // if selecting multiple queues from the queue family, we need to provide a float array to indicate each one's priority (0.0
    // to 1.0). currently only using one queue, so this is fine
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo devQueueInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                         .queueFamilyIndex = static_cast<uint32_t>(graphicsFamilyIdx.value()),
                                         .queueCount = 1,
                                         .pQueuePriorities = &queuePriority};

    // don't need any advanced or specific graphics features right now (like a geometry shader) so we leave everything empty
    VkPhysicalDeviceFeatures devFeatsInfo{};

    VkDeviceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &devQueueInfo,
        .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames = validationLayers.size() > 0 ? validationLayers.data() : nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(REQUIRED_DEVICE_EXTENSIONS.size()),
        .ppEnabledExtensionNames = REQUIRED_DEVICE_EXTENSIONS.data(),
        .pEnabledFeatures = &devFeatsInfo,
    };

    if (vkCreateDevice(mPhysicalDevices[mSelectedDeviceIdx].dev, &createInfo, nullptr, &mDevice) != VK_SUCCESS)
      MAPLE_FATAL("Failed to create logical device");
  }

  void createVulkanInstance(const std::vector<const char*>& requiredExtensions) {
    for (const char* layerName : validationLayers) {
      bool found = false;

      for (const auto& l : mAvailableInstanceLayers) {
        if (strcmp(layerName, l.layerName) == 0) {
          found = true;
          break;
        }
      }

      if (!found) MAPLE_FATAL("Failed to find required Vulkan instance layer \"{}\"", layerName);
    }

    std::vector<const char*> enabledExtensions = requiredExtensions;
    if (validationLayers.size() > 0) enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Application name";                 // TODO: replace
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);  // TODO: replace
    appInfo.pEngineName = "Maple Engine";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);  // TODO: replace
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    populateDebugMessengerCreateInfo(debugCreateInfo, vulkanDebugCallback);

    VkInstanceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = validationLayers.size() > 0 ? &debugCreateInfo : nullptr,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames = validationLayers.size() > 0 ? validationLayers.data() : nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size()),
        .ppEnabledExtensionNames = enabledExtensions.data(),
    };

    if (vkCreateInstance(&createInfo, nullptr, &mInstance) != VK_SUCCESS) MAPLE_FATAL("Failed to create vulkan instance");
  }

  void setupDebugCallback() {
    if (validationLayers.size() < 1) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo, vulkanDebugCallback);

    if (CreateDebugUtilsMessengerEXT(mInstance, &createInfo, nullptr, &mDebugMessenger) != VK_SUCCESS)
      MAPLE_FATAL("Failed to create a Vulkan debug messenger");
  }

  void selectPhysicalDevice() {
    std::multimap<int32_t, size_t> candidates;

    for (auto [i, device] : std::views::enumerate(mPhysicalDevices)) {
      std::set<std::string> requiredExtensions(REQUIRED_DEVICE_EXTENSIONS.begin(), REQUIRED_DEVICE_EXTENSIONS.end());
      for (const auto& e : device.extensions) requiredExtensions.erase(e.extensionName);
      if (!requiredExtensions.empty()) {
        MAPLE_INFO("Device {} doesn't have required device extensions", device.properties.deviceName);
        continue;
      }

      if (device.presentModes.empty() || device.surfaceFormats.empty()) {
        MAPLE_INFO("Device {} doesn't have required surface features", device.properties.deviceName);
        continue;
      }

      int32_t score = 0;

      switch (device.properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
          score += 2000;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
          score += 500;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
          score += 50;
          break;
      }

      score += device.properties.limits.maxImageDimension2D / 100;

      candidates.insert(std::make_pair(score, i));
    }

    // TODO: disqualify devices with no graphics queue (maybe compute queue aswell)
    // TODO: weigh memory heaps, push constants, and maxFramebuffer dimensions

    if (candidates.size() < 1) MAPLE_FATAL("Failed to find any appropriate device");

    for (const auto& e : candidates) MAPLE_INFO("\tScore {}: {}", mPhysicalDevices[e.second].properties.deviceName, e.first);

    mSelectedDeviceIdx = candidates.rbegin()->second;
    MAPLE_INFO("Selected Graphics Device {}", mPhysicalDevices[mSelectedDeviceIdx].properties.deviceName);
  }

  void probePhysicalDevices() {
    mPhysicalDevices = GetPhysicalDevices(mInstance, mSurface);

    MAPLE_INFO("Available Vulkan devices ({}):", mPhysicalDevices.size());
    if (mPhysicalDevices.empty()) MAPLE_FATAL("Failed to find graphics device with Vulkan support");

    for (const auto& dev : mPhysicalDevices) {
      MAPLE_INFO("\t{}: {}", dev.properties.deviceName, vkPhysicalDeviceTypeToString(dev.properties.deviceType));
      MAPLE_INFO("\tQueue Families ({}):", dev.queueFamiliesCapabilities.size());

      for (const auto& caps : dev.queueFamiliesCapabilities) {
        MAPLE_INFO(
            "\t\tQueue count: {} Compute: {} Graphics: {} Optical_flow: {} Protected: {} Sparse_binding: {} "
            "Transfer: {} Video_decode: {} Video_encode: {}",
            caps.QueueCount, caps.Compute, caps.Graphics, caps.Optical_flow, caps.Protected, caps.Sparse_binding, caps.Transfer, caps.Video_decode,
            caps.Video_encode);
      }
    }
  }

  void probeInstanceExtensions() {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    mAvailableInstanceExtensions.resize(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, mAvailableInstanceExtensions.data());

    MAPLE_INFO("Available Vulkan instance extensions ({}):", count);
    for (const auto& e : mAvailableInstanceExtensions) MAPLE_INFO("\t{}: {}", e.extensionName, e.specVersion);
  }

  void probeInstanceLayers() {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    mAvailableInstanceLayers.resize(count);
    vkEnumerateInstanceLayerProperties(&count, mAvailableInstanceLayers.data());

    MAPLE_INFO("Available Vulkan instance layers ({}):", count);
    for (const auto& l : mAvailableInstanceLayers)
      MAPLE_INFO("\t{}: {}, {}, ({})", l.layerName, l.specVersion, l.implementationVersion, l.description);
  }
};  // namespace maple

Renderer::Renderer() : mPimpl(std::make_unique<Impl>()) {}
Renderer::~Renderer() {}
void Renderer::Init(const std::vector<const char*>& requiredExtensions, SurfaceCreateCallback surfaceCreateCallback,
                    FramebufferSizeCallback framebufferSizeCallback) {
  mPimpl->Init(requiredExtensions, surfaceCreateCallback, framebufferSizeCallback);
}
void Renderer::DrawFrame() { mPimpl->DrawFrame(); }
void Renderer::SetFrameBufferResized() { mPimpl->SetFrameBufferResized(); }
void Renderer::Destroy() { mPimpl->Destroy(); }

}  // namespace maple
