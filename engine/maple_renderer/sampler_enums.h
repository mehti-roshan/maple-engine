#pragma once

#include <cstdint>
namespace maple {
enum class SamplerFilter : uint8_t {
  Nearest,
  Linear,
};

enum class SamplerMipmapMode : uint8_t {
  Nearest,
  Linear,
};

enum class SamplerAddressMode : uint8_t {
  Repeat,
  MirroredRepeat,
  ClampToEdge,
  ClampToBorder,
  MirrorClampToEdge,
};

enum class SamplerBorderColor : uint8_t {
  FloatTransparentBlack,
  IntTransparentBlack,
  FloatOpaqueBlack,
  IntOpaqueBlack,
  FloatOpaqueWhite,
  IntOpaqueWhite,
};

// CompareOp already defined earlier; if not, reuse the same enum as before:
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

}  // namespace maple