#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace maple {
struct MaterialBuilderData {
 public:
  enum class PolygonMode : uint8_t {
    Fill,
    Line,
    Point,
  };

  enum class CullModeFlagBits : uint8_t {
    None,
    Front,
    Back,
    FrontAndBack,  // Front | Back
  };

  enum class FrontFace : uint8_t {
    CounterClockwise,
    Clockwise,
  };

  enum class CompareOp : uint8_t {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
  };

  struct RasterizerState {
    bool depthClampEnable = false;
    bool rasterizerDiscardEnable = false;
    PolygonMode polygonMode = PolygonMode::Fill;
    CullModeFlagBits cullMode = CullModeFlagBits::Back;
    FrontFace frontFace = FrontFace::CounterClockwise;
    bool depthBiasEnable = false;
  };

  struct DepthStencilState {
    bool depthTest = true;
    bool depthWrite = true;
    CompareOp depthCompareOp = CompareOp::Less;
    bool depthBoundsTestEnable = false;
    bool stencilTestEnable = false;
  };

  enum class BlendColorWriteMask : uint8_t {};
  struct BlendingState {
    bool blendEnable = false;
  };

  BlendingState blendingState;
  RasterizerState rasterizer;
  DepthStencilState depthStencil;

  std::string vertEntryFuncName = "vertMain";
  std::string fragEntryFuncName = "fragMain";
  // Must be slang shader code, compiled internally to spirv
  std::vector<uint8_t> shaderCode;
};
}  // namespace maple