#pragma once

#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "enums.h"
#include "log_macros.h"

namespace maple {
class RenderGraph {
 public:
  struct AttachmentInfo {
    SizeType sizeType = SizeType::SwapChainRelative;
    glm::vec2 size = glm::vec2(1.f, 1.f);
    Format format = Undefined;

    glm::uvec2 GetAbsoluteSize(glm::uvec2 swapChainSize) const {
      if (sizeType == Absolute) return size;
      return glm::uvec2(size.x * swapChainSize.x, size.y * swapChainSize.y);
    }

    bool operator==(const AttachmentInfo& rhs) const { return sizeType == rhs.sizeType && size == rhs.size && format == rhs.format; }
  };

  enum PipelineType { Graphics, Compute };

  struct NameAndAttachment {
    std::string name;
    AttachmentInfo info;
  };

  struct Pass {
   public:
    Pass& AddInput(const std::string& inputName) {
      inputs.push_back(inputName);
      return *this;
    }
    Pass& AddOutput(const std::string& outputName, const AttachmentInfo& info) {
      outputs.push_back({outputName, info});
      return *this;
    }

   private:
    Pass() = default;
    Pass(const std::string& name, PipelineType pipelineType) : name(name), pipelineType(pipelineType) {}

    std::string name;
    PipelineType pipelineType;
    std::vector<std::string> inputs;
    std::vector<NameAndAttachment> outputs;

    friend class RenderGraph;
  };

  Pass& AddPass(const std::string& name, PipelineType pipelineType) {
    passes.push_back(Pass(name, pipelineType));
    return passes.back();
  }

  struct ResourceState {
    ImageLayout layout = ImageLayout::Undefined;
    AccessMask access = AccessMask::None;
    PipelineStage stage = PipelineStage::TopOfPipe;
  };

  struct Transition {
    std::string resource;
    ResourceState oldState, newState;
  };

  struct ExecutablePass {
    std::string name;
    PipelineType pipelineType;
    std::vector<Transition> preTransitions;  // barriers to apply before the pass
    std::vector<NameAndAttachment> outputs;
  };

  struct CompileResult {
    std::vector<ExecutablePass> passes;
    std::vector<NameAndAttachment> attachments;
  };

  CompileResult Compile() const {
    // 1. Build output -> pass map
    std::unordered_map<std::string, const Pass*> outputToPass;
    for (const auto& pass : passes) {
      for (const auto& out : pass.outputs) {
        MAPLE_ASSERT(outputToPass.find(out.name) == outputToPass.end(), "Duplicate output name '{}' in render graph", out.name);
        outputToPass[out.name] = &pass;
      }
    }

    std::vector<ExecutablePass> execOrder;
    std::unordered_set<const Pass*> visited;
    std::unordered_set<const Pass*> inStack;

    // Tracked resource state: resource name -> its current layout/access/stage
    std::unordered_map<std::string, ResourceState> resourceStates;

    std::function<void(const std::string&)> walk = [&](const std::string& resource) {
      auto it = outputToPass.find(resource);
      if (it == outputToPass.end()) return;  // external, skip

      const Pass* pass = it->second;
      if (visited.count(pass)) return;
      if (inStack.count(pass)) {
        MAPLE_FATAL("Render graph cycle detected involving pass: {}", pass->name);
      }

      inStack.insert(pass);

      // Process dependencies (inputs)
      for (const auto& inputName : pass->inputs) {
        walk(inputName);
      }

      inStack.erase(pass);
      visited.insert(pass);

      // ---- Build the executable pass ----
      ExecutablePass exec;
      exec.name = pass->name;
      exec.pipelineType = pass->pipelineType;

      // For each input, ensure it is in ShaderReadOnlyOptimal before this pass
      for (const auto& inputName : pass->inputs) {
        ResourceState required = GetReadState();
        InsertTransitionIfNeeded(inputName, required, resourceStates, exec.preTransitions);
      }

      // For each output, ensure it is in the appropriate attachment layout before the pass writes it
      for (const auto& out : pass->outputs) {
        ResourceState required = GetWriteState(out.info);
        InsertTransitionIfNeeded(out.name, required, resourceStates, exec.preTransitions);
      }

      exec.outputs = pass->outputs;

      // Special handling: if a resource appears as both input and output (read‑write, e.g., depth),
      // we use the write state (attachment state) and skip the read transition.
      // Since we process inputs first and then outputs, the write state may override.
      // To keep it simple, we can just process outputs after inputs, allowing outputs to override.

      // Actually, for read‑write resources (like depth with depth test), the required state
      // should be DepthStencilAttachmentOptimal with both read and write access.
      // Our current GetWriteState gives write access only; we may want to include read access later.
      // But it's fine as a first pass.

      execOrder.push_back(std::move(exec));
    };

    // Start from the final target
    walk(SWAPCHAIN_TARGET_NAME);

    std::vector swapChainTransition = {Transition{
      .resource = SWAPCHAIN_TARGET_NAME,
      .oldState = resourceStates[SWAPCHAIN_TARGET_NAME],
      .newState =
        {
          .layout = ImageLayout::PresentSrc,
          .access = AccessMask::None,
          .stage = PipelineStage::BottomOfPipe,
        },
    }};

    execOrder.push_back({
      .name = "SWAPCHAIN_TRANSITION",
      .pipelineType = PipelineType::Graphics,
      .preTransitions = std::move(swapChainTransition),
    });

    std::vector<NameAndAttachment> attachments;
    for (const auto& pass : passes) {
      for (const auto& v : pass.outputs) {
        if (v.name != SWAPCHAIN_TARGET_NAME) attachments.push_back(v);
      }
    }

    return {std::move(execOrder), std::move(attachments)};
  }

  static constexpr std::string SWAPCHAIN_TARGET_NAME = "SWAPCHAIN";

 private:
  std::vector<Pass> passes;

  // Get the state required for reading a resource at a given shader stage
  static ResourceState GetReadState() {
    ResourceState s;
    s.layout = ImageLayout::ShaderReadOnlyOptimal;
    s.access = AccessMask::ShaderRead;
    s.stage = PipelineStage::AllGraphicsAndCompute;
    return s;
  }

  // Get the state required for writing to an attachment
  static ResourceState GetWriteState(const AttachmentInfo& info) {
    ResourceState s{.layout = ImageLayout::AttachmentOptimal};
    if (FormatIsDepth(info.format)) {
      s.access = AccessMask::DepthStencilAttachmentWrite;
      s.stage = PipelineStage::EarlyAndLateFragmentTests;  // safe conservative
    } else {
      // Color or ColorHDR
      s.access = AccessMask::ColorAttachmentWrite;
      s.stage = PipelineStage::ColorAttachmentOutput;
    }
    return s;
  }

  // If the current state of 'resource' differs from 'required', add a transition and update the tracked state
  static void InsertTransitionIfNeeded(const std::string& resource,
                                       const ResourceState& required,
                                       std::unordered_map<std::string, ResourceState>& states,
                                       std::vector<Transition>& transitions) {
    auto& current = states[resource];
    if (current.layout != required.layout || current.access != required.access || current.stage != required.stage) {
      Transition t;
      t.resource = resource;

      t.newState.layout = required.layout;
      t.newState.access = required.access;
      t.newState.stage = required.stage;

      t.oldState = current;

      transitions.push_back(t);

      current = required;
    }
  }
};
}  // namespace maple