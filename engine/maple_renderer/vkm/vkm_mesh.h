#pragma once

#include <cstdint>
#include "mesh_data.h"
#include "vkm/vkm_allocator.h"

namespace vkm {
class Mesh {
 public:
  vkm::Buffer meshBuffer;

  Mesh() = default;
  Mesh(Allocator& allocator, const maple::MeshData& mesh) {
    meshBuffer =
      allocator.CreateBuffer(mesh.verts.size() + mesh.indices.size() * sizeof(decltype(mesh.indices)::value_type), Allocator::BufType::Mesh);
    numVerts = mesh.numVerts;
    numIndices = mesh.indices.size();
    indexBufferOffset = mesh.GetStride() * numVerts;
  }

  uint32_t GetNumVertices() const { return numVerts; }
  uint32_t GetNumIndices() const { return numIndices; }
  uint32_t GetIndexBufferOffset() const { return indexBufferOffset; }

  static vk::IndexType GetVkIndexType() { return vk::IndexType::eUint32; }

  uint32_t AddRef() { return ++numRefs; }
  uint32_t RemoveRef() { return --numRefs; }
  uint32_t GetRefs() const { return numRefs; }

 private:
  uint32_t numVerts = 0;
  uint32_t numIndices = 0;
  uint32_t indexBufferOffset = 0;
  uint32_t numRefs = 0;
};
}  // namespace vkm