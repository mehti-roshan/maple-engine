#pragma once

#include <entt/entt.hpp>
#include <utility>

namespace maple {

using Entity = entt::entity;

class MapleScene {
 public:
  Entity CreateEntity() { return mRegistry.create(); }

  void DestroyEntity(Entity entity) { mRegistry.destroy(entity); }

  template <typename T, typename... Args>
  T& Add(Entity entity, Args&&... args) {
    return mRegistry.emplace<T>(entity, std::forward<Args>(args)...);
  }

  template <typename T>
  T& Get(Entity entity) {
    return mRegistry.get<T>(entity);
  }

  template <typename T>
  const T& Get(Entity entity) const {
    return mRegistry.get<T>(entity);
  }

  template <typename T>
  bool Has(Entity entity) const {
    return mRegistry.all_of<T>(entity);
  }

  template <typename T>
  void Remove(Entity entity) {
    mRegistry.remove<T>(entity);
  }

  template <typename... Ts>
  auto View() {
    return mRegistry.view<Ts...>();
  }

  template <typename... Ts>
  auto View() const {
    return mRegistry.view<Ts...>();
  }

  void Clear() { mRegistry.clear(); }

  template<typename T>
  auto OnCreate() {
    return mRegistry.on_construct<T>();
  }

  template<typename T>
  auto OnDestroy() {
    return mRegistry.on_destroy<T>();
  }

 private:
  entt::registry mRegistry;
};

}  // namespace maple