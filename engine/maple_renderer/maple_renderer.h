#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "enums.h"
#include "material_builder_data.h"
#include "mesh_data.h"
#include "render_graph.h"
#include "renderer_callbacks.h"

namespace maple {

class MapleRenderer {
 public:
  MapleRenderer();
  MapleRenderer(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback, FrameBufferSizeCallback);
  ~MapleRenderer();
  MapleRenderer(MapleRenderer&&) noexcept;
  MapleRenderer& operator=(MapleRenderer&&) noexcept;

  void SetFrameBufferResized() { mFrameBufferResized = true; };

  using MeshHndl = uint32_t;
  using MaterialHndl = uint32_t;
  using TextureHndl = uint32_t;

  [[nodiscard]]
  MeshHndl CreateMesh(const MeshData& data);
  void AddMeshRef(MeshHndl);
  void RemoveMeshRef(MeshHndl);

  [[nodiscard]]
  MaterialHndl CreateMaterial(const std::string& shaderCode, const std::string& shaderFileName, const MaterialBuilderData& data);
  void AddMaterialRef(MaterialHndl);
  void RemoveMaterialRef(MaterialHndl);

  [[nodiscard]]
  TextureHndl CreateTexture(glm::uvec2 dimensions, std::span<const uint8_t> bytes, Format format);
  void AddTextureRef(TextureHndl hndl);
  void RemoveTextureRef(TextureHndl hndl);

  std::optional<Format> FindFirstSupportedTextureFormat(std::span<const Format> formats) const;
  std::optional<Format> FindFirstSupportedDepthAttachmentFormat(std::span<const Format> formats) const;

  struct UBO {
    glm::mat4 view;
    glm::mat4 proj;
    float time;
  };

  struct MeshDraw {
    MeshHndl mesh;
    std::span<const glm::mat4> instanceData;
    std::span<std::variant<const std::string, TextureHndl>> usedResources; // string means sampling a render target, otherwise a regular texture
  };

  struct MaterialDraw {
    MaterialHndl material;
    std::span<const MeshDraw> meshes;
  };

  struct PassDraw {
    std::string passName;
    std::span<const MaterialDraw> materialDraws;
  };

  void DrawFrame(const UBO& frameUBO, const RenderGraph::CompileResult& compiledRenderGraph, std::span<const PassDraw> passDraws);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
  bool mFrameBufferResized = false;
};
}  // namespace maple