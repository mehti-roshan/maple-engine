#pragma once

namespace maple {
enum class ImageLayout {
  Undefined,
  AttachmentOptimal,
  ShaderReadOnlyOptimal,
  PresentSrc,
};

enum class AccessMask {
  None,
  ColorAttachmentWrite,
  DepthStencilAttachmentWrite,
  ShaderRead,
};

enum ShaderStage { Vertex, Fragment, AllGraphics, Compute, AllGraphicsAndCompute };

enum class PipelineStage {
  TopOfPipe,
  BottomOfPipe,
  ColorAttachmentOutput,
  EarlyFragmentTests,
  LateFragmentTests,
  EarlyAndLateFragmentTests,
  VertexShader,
  FragmentShader,
  ComputeShader,
  AllGraphics,
  AllGraphicsAndCompute,
};

enum SizeType { Absolute, SwapChainRelative };
enum Format {
  Undefined,
  // 8‑bit unsigned normalized (standard SDR)
  R8G8B8A8_UNORM,
  B8G8R8A8_UNORM,  // common swapchain format
  R8G8B8A8_SRGB,
  B8G8R8A8_SRGB,  // sRGB swapchain

  // 16‑bit floating point (HDR intermediates)
  R16G16B16A16_SFLOAT,

  // Packed HDR / high precision
  B10G11R11_UFLOAT,   // no alpha, 3‑channel HDR
  R10G10B10A2_UNORM,  // 10‑bit RGB, 2‑bit alpha

  // 32‑bit floating point (reference / compute)
  R32G32B32A32_SFLOAT,

  // Single‑component formats (for masks, lookup tables, etc.)
  R8_UNORM,
  R16_SFLOAT,
  R32_SFLOAT,

  // Depth & Stencil formats
  D16_UNORM,
  D16_UNORM_S8,
  D24_UNORM_S8,
  D32_SFLOAT,
  D32_SFLOAT_S8,
};

bool FormatIsDepth(Format format);

bool FormatHasStencil(Format format);

bool FormatIsColor(Format format);
}  // namespace maple