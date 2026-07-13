#pragma once
#include "maple_core/camera.h"
#include "maple_core/input.h"
#include "maple_core/time.h"
#include "maple_physics.h"
#include "maple_renderer/maple_renderer.h"
#include "maple_scene.h"
#include "maple_window/maple_window.h"

namespace maple {
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
  std::string mTex1 = "texture.jpg";
  std::string mTex2 = "viking_room.png";

  void Init();
};
}  // namespace maple