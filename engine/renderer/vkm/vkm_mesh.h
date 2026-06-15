#pragma once

#include <cstddef>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace vkm {

struct Vertex {
  glm::vec3 pos;
  glm::vec2 uv;

  bool operator==(const Vertex& other) const { return pos == other.pos && uv == other.uv; }

  static vk::VertexInputBindingDescription GetVertInputBindDesc() {
    return vk::VertexInputBindingDescription{.binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};
  }

  static std::vector<vk::VertexInputAttributeDescription> GetVertAttribDescriptions() {
    return {
      {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)},
      {1, 0, vk::Format::eR32G32Sfloat,  offsetof(Vertex, uv)},
    };
  }
};

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  size_t GetVerticesSizeBytes() const { return sizeof(vertices[0]) * vertices.size(); }
  size_t GetIndicesSizeBytes() const { return sizeof(indices[0]) * indices.size(); }
  size_t GetTotalSizeBytes() const { return GetVerticesSizeBytes() + GetIndicesSizeBytes(); }

  static vk::IndexType GetVkIndexType() { return vk::IndexType::eUint32; }
};
}  // namespace vkm

namespace std {
template <>
struct hash<vkm::Vertex> {
  size_t operator()(vkm::Vertex const& vertex) const { return (hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec2>()(vertex.uv) << 1)); }
};
}  // namespace std