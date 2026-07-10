#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace maple {
class MeshData {
 public:
  std::span<const std::byte> verts;
  std::span<const uint32_t> indices;
  uint32_t numVerts;

  uint32_t GetStride() const { return verts.size() / numVerts; }
  uint32_t GetTotalSize() const { return indices.size() * sizeof(decltype(indices)::value_type) + verts.size(); }
};
}  // namespace maple