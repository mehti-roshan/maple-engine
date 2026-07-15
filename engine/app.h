#pragma once
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

#include "maple_core/camera.h"
#include "maple_core/input.h"
#include "maple_core/time.h"
#include "maple_physics.h"
#include "maple_renderer/maple_renderer.h"
#include "maple_scene.h"
#include "maple_window/maple_window.h"

namespace maple {

struct PosQuat {
  glm::vec3 position{};
  glm::quat rotation = glm::identity<glm::quat>();
};

struct Scale {
  glm::vec3 scale;
};

struct PhysicsBody {
  PhysicsBodyID id;
};

using MeshHndl = MapleRenderer::MeshHndl;
using MaterialHndl = MapleRenderer::MaterialHndl;
using TextureHndl = MapleRenderer::TextureHndl;
struct Renderable {
  MeshHndl mesh;
  MaterialHndl material;
  std::vector<TextureHndl>& textures;
};

class App {
 public:
  ~App();

  void Run();

 private:
  MapleWindow mWindow;
  MapleRenderer mRenderer;
  MaplePhysics mPhysics;
  MapleScene mScene;
  Time mTime;
  Input mInput;
  Camera mCam;
  RenderGraph::CompileResult mCompiledRenderGraph;

  MapleRenderer::MeshHndl mMesh;
  MapleRenderer::MaterialHndl mMaterial;
  TextureHndl mTex1, mTex2;

  void Init();

  // Entity component creation & destruction callbacks
  static void OnDestroyPhysicsBody(entt::registry& registry, entt::entity entity);
  static void OnCreateRenderable(entt::registry& registry, entt::entity entity);
  static void OnDestroyRenderable(entt::registry& registry, entt::entity entity);
};
}  // namespace maple