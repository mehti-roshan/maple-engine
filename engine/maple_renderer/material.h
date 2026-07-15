#pragma once

#include <cstdint>
#include <optional>

#include "material_builder_data.h"
#include "vkm/vkm_pipeline.h"

namespace maple {
struct Material {
 public:
  Material() {};
  Material(const MaterialBuilderData& data) : data(data) {}

  std::optional<vkm::Pipeline>& Pipeline() { return pipeline; }
  MaterialBuilderData& Data() { return data; }

  uint32_t AddRef() { return ++numRefs; }
  uint32_t RemoveRef() { return --numRefs; }
  uint32_t GetRefs() const { return numRefs; }

 private:
  // since we don't know the output attachment formats of the pipeline
  // we only store the build data and compile lazily when needed
  std::optional<vkm::Pipeline> pipeline = std::nullopt;
  MaterialBuilderData data;
  uint32_t numRefs = 0;
};
}  // namespace maple