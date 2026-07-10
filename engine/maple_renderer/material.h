#pragma once

#include <optional>

#include "material_builder_data.h"
#include "vkm/vkm_pipeline.h"

namespace maple {
struct Material {
  // TODO: store a map of pipelines instead, for when a pipeline is used with different format of attachments
  // since we don't know the output attachment formats of the pipeline
  // we only store the build data and compile lazily when needed
  std::optional<vkm::Pipeline> pipeline = std::nullopt;
  MaterialBuilderData data;
};
}  // namespace maple