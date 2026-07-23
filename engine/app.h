#pragma once

#include "engine/maple_core/camera.h"
#include "engine/maple_core/input.h"
#include "engine/maple_core/time.h"
#include "engine/maple_window/maple_window.h"
#include "maple_audio.h"
#include "maple_physics.h"
#include "maple_renderer.h"

namespace maple {
using MeshHndl = Renderer::MeshHndl;
using MaterialHndl = Renderer::MaterialHndl;
using TextureHndl = Renderer::TextureHndl;
using PhysicsBodyID = Physics::BodyID;
using AudioClipHndl = Audio::ClipHndl;

class App {
 public:
  ~App();

  void Run();

 private:
  Window mWindow;
  Audio mAudio;
  Renderer mRenderer;
  Physics mPhysics;
  Time mTime;
  Input mInput;
  Camera mCam;
  RenderGraph::CompileResult mCompiledRenderGraph;

  void Init();

  Renderer::MeshHndl mMesh;
  Renderer::MaterialHndl mMaterial;
  TextureHndl mTex1, mTex2;
};
}  // namespace maple