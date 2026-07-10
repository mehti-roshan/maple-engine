#pragma once
#include <engine/maple_renderer/maple_renderer.h>
#include <engine/maple_window/maple_window.h>

#include "camera.h"
#include "time.h"
#include "input.h"

namespace maple {
class Engine {
 public:
  ~Engine();

  void Run();

 private:
  MapleWindow mWindow;
  MapleRenderer mRenderer;
  Time mTime;
  RenderGraph::CompileResult mCompiledRenderGraph;
  Input mInput;
  Camera mCam;

  MapleRenderer::MeshHndl mMesh;
  MapleRenderer::MaterialHndl mMaterial;
  std::string mTex1 = "texture.jpg";
  std::string mTex2 = "viking_room.png";

  void Init();
};
}  // namespace maple