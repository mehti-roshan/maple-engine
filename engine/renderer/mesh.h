#pragma once
#include <type_traits>
#include <vector>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

template <typename T>
concept VertexType = requires {
  { T::getBindingDescription() } -> std::same_as<vk::VertexInputBindingDescription>;

  { T::getAttributeDescriptions() } -> std::same_as<std::vector<vk::VertexInputAttributeDescription>>;
};

template <typename T>
concept IndexType = std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>;

template <VertexType VertexT, IndexType IndexT>
struct Mesh {
  std::vector<VertexT> vertices;
  std::vector<IndexT> indices;

  static vk::IndexType GetVkIndexType() { return std::is_same_v<IndexT, uint16_t> ? vk::IndexType::eUint16 : vk::IndexType::eUint32; }
  static vk::VertexInputBindingDescription GetBindingDescription() { return VertexT::getBindingDescription(); }
  static std::vector<vk::VertexInputAttributeDescription> GetAttributeDescriptions() { return VertexT::getAttributeDescriptions(); }

  size_t GetVerticesSizeBytes() const {return sizeof(vertices[0]) * vertices.size(); }
  size_t GetIndicesSizeBytes() const { return sizeof(indices[0]) * indices.size(); }
  size_t GetTotalSizeBytes() const { return GetVerticesSizeBytes() + GetIndicesSizeBytes(); }
};