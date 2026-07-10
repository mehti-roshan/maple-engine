#pragma once

#include "render_graph.h"
#include "vkm/vkm_image.h"

namespace maple {

struct RenderTarget {  
  RenderGraph::AttachmentInfo info;
  vkm::Image target;
};
}  // namespace maple