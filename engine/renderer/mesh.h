#pragma once
#include <type_traits>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

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

  static constexpr vk::IndexType vkIndexType = std::is_same_v<IndexT, uint16_t> ? vk::IndexType::eUint16 : vk::IndexType::eUint32;

  static vk::VertexInputBindingDescription getBindingDescription() { return VertexT::getBindingDescription(); }

  static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions() { return VertexT::getAttributeDescriptions(); }
};