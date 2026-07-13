#include <bit>
#include <cstdlib>
#include <ctime>
#include <glm/ext/quaternion_transform.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/gtc/constants.hpp>
#include <utility>
#include <vector>

#include "enums.h"
#include "maple_asset/maple_asset.h"
#include "maple_core/input_enums.h"
#include "maple_core/noise.h"
#include "maple_core/prng.h"
#include "maple_logging/log_macros.h"
#include "maple_physics.h"
#include "maple_renderer.h"
#include "maple_renderer/material_builder_data.h"
#include "maple_renderer/render_graph.h"
#include "maple_window/maple_window.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/glm.hpp>

#include "app.h"

namespace maple {

void App::Init() {
  logging::Log::init();
  MAPLE_INFO("Initializing...");

  mWindow = MapleWindow({.title = "Maple"});

  if (mWindow.RawMouseMotionSupported()) mWindow.SetRawMouseMotion(true);
  mWindow.LockCursor();
  mWindow.AddFramebufferSizeCallback([&](int32_t width, int32_t height) { mRenderer.SetFrameBufferResized(); });

  mWindow.AddCursorPosCallback([&](auto a, auto b) { mInput.OnCursorPos(a, b); });
  mWindow.AddScrollCallback([&](auto a, auto b) { mInput.OnMouseScroll(a, b); });
  mWindow.AddKeyCallback([&](auto a, auto b, auto c, auto d) { mInput.OnKey(a, b, c, d); });
  mWindow.AddMouseButtonCallback([&](auto a, auto b, auto c) { mInput.OnMouseButtons(a, b, c); });
  mWindow.AddJoySticksCallback([&](const auto& joyStates) {
    std::vector<std::pair<int32_t, Input::JoystickState>> cpy;
    cpy.reserve(joyStates.size());
    for (const auto& v : joyStates) {
      cpy.push_back(std::make_pair(v.first, std::bit_cast<Input::JoystickState>(v.second)));
    }
    mInput.OnJoySticks(cpy);
  });

  mCam.SetPosition(glm::vec3(0));

  mPhysics.Initialize(glm::vec3(0, -9.81, 0));

  mRenderer = MapleRenderer(
    mWindow.RequiredVkInstanceExtensions(),
    [&](void* pVkInstance) { return mWindow.CreateWindowSurface(pVkInstance); },
    [&](uint32_t& width, uint32_t& height) {
      auto [x, y] = mWindow.GetFrameBufferSize();
      width = x;
      height = y;
    });

  auto formats = {Format::D32_SFLOAT, Format::D32_SFLOAT_S8, Format::D24_UNORM_S8};
  auto depthFormat = mRenderer.FindFirstSupportedDepthAttachmentFormat(formats);
  if (!depthFormat.has_value()) MAPLE_FATAL("failed to find supported depth format");

  auto graph = RenderGraph();
  graph.AddPass("draw", RenderGraph::Graphics)
    .AddOutput("depth", {.sizeType = SwapChainRelative, .size = glm::vec2(1), .format = *depthFormat})
    .AddOutput(RenderGraph::SWAPCHAIN_TARGET_NAME, {});

  mCompiledRenderGraph = graph.Compile();

  struct Vertex {
    glm::vec3 pos;
    glm::vec2 uv;
  };

  std::array frontVerts = {
    glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f),
    glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f),
    glm::vec4(0.5f, -0.5f, 0.5f, 1.0f),
    glm::vec4(0.5f, 0.5f, 0.5f, 1.0f),
  };
  std::array verts = {
    // Front Face
    Vertex{glm::vec3(frontVerts[0]), glm::vec2(0.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[1]), glm::vec2(0.0f, 1.0f)},
    Vertex{glm::vec3(frontVerts[2]), glm::vec2(1.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[3]), glm::vec2(1.0f, 1.0f)},

    // Right Face
    Vertex{glm::vec3(frontVerts[0] * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 1, 0))), glm::vec2(0.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[1] * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 1, 0))), glm::vec2(0.0f, 1.0f)},
    Vertex{glm::vec3(frontVerts[2] * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 1, 0))), glm::vec2(1.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[3] * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 1, 0))), glm::vec2(1.0f, 1.0f)},

    // Back Face
    Vertex{glm::vec3(frontVerts[0] * glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0, 1, 0))), glm::vec2(0.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[1] * glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0, 1, 0))), glm::vec2(0.0f, 1.0f)},
    Vertex{glm::vec3(frontVerts[2] * glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0, 1, 0))), glm::vec2(1.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[3] * glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0, 1, 0))), glm::vec2(1.0f, 1.0f)},

    // Left Face
    Vertex{glm::vec3(frontVerts[0] * glm::rotate(glm::mat4(1.0f), glm::radians(270.0f), glm::vec3(0, 1, 0))), glm::vec2(0.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[1] * glm::rotate(glm::mat4(1.0f), glm::radians(270.0f), glm::vec3(0, 1, 0))), glm::vec2(0.0f, 1.0f)},
    Vertex{glm::vec3(frontVerts[2] * glm::rotate(glm::mat4(1.0f), glm::radians(270.0f), glm::vec3(0, 1, 0))), glm::vec2(1.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[3] * glm::rotate(glm::mat4(1.0f), glm::radians(270.0f), glm::vec3(0, 1, 0))), glm::vec2(1.0f, 1.0f)},

    // Up Face
    Vertex{glm::vec3(frontVerts[0] * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0))), glm::vec2(0.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[1] * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0))), glm::vec2(0.0f, 1.0f)},
    Vertex{glm::vec3(frontVerts[2] * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0))), glm::vec2(1.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[3] * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0))), glm::vec2(1.0f, 1.0f)},

    // Down Face
    Vertex{glm::vec3(frontVerts[0] * glm::rotate(glm::mat4(1.0f), glm::radians(270.0f), glm::vec3(1, 0, 0))), glm::vec2(0.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[1] * glm::rotate(glm::mat4(1.0f), glm::radians(270.0f), glm::vec3(1, 0, 0))), glm::vec2(0.0f, 1.0f)},
    Vertex{glm::vec3(frontVerts[2] * glm::rotate(glm::mat4(1.0f), glm::radians(270.0f), glm::vec3(1, 0, 0))), glm::vec2(1.0f, 0.0f)},
    Vertex{glm::vec3(frontVerts[3] * glm::rotate(glm::mat4(1.0f), glm::radians(270.0f), glm::vec3(1, 0, 0))), glm::vec2(1.0f, 1.0f)},
  };

  std::vector<uint32_t> indices = {
    0,
    2,
    1,

    1,
    2,
    3,
  };

  uint32_t numVerts = 4;
  for (size_t face = 0; face < 5; face++) {
    indices.push_back(indices[0] + numVerts);
    indices.push_back(indices[1] + numVerts);
    indices.push_back(indices[2] + numVerts);
    indices.push_back(indices[3] + numVerts);
    indices.push_back(indices[4] + numVerts);
    indices.push_back(indices[5] + numVerts);
    numVerts += 4;
  }

  mMesh = mRenderer.CreateMesh({
    .verts = std::as_bytes(std::span<const Vertex>(verts)),
    .indices = indices,
    .numVerts = static_cast<uint32_t>(verts.size()),
  });

  mMaterial = mRenderer.CreateMaterial(MapleAsset::ReadFileStr("assets/shaders/shader.slang"), "shader", {.rasterizer = {}});

  std::array colorFormats = {Format::R8G8B8A8_SRGB, Format::B8G8R8A8_SRGB};
  auto format = mRenderer.FindFirstSupportedTextureFormat(colorFormats);
  if (!format.has_value()) MAPLE_FATAL("failed to find suitable color format");

  {
    auto img = MapleAsset::LoadImage("assets/textures/" + mTex1);
    mRenderer.CreateTexture(mTex1, img.size, img.bytes, *format);
  }

  {
    auto img = MapleAsset::LoadImage("assets/textures/" + mTex2);
    mRenderer.CreateTexture(mTex2, img.size, img.bytes, *format);
  }

  mInput.Bind("exit", {Key::Escape});
  mInput.Bind("exit", {GamepadButton::Start});

  mInput.Bind("forward", {Key::W});
  mInput.Bind("forward", {Key::S, false});
  mInput.Bind("sideways", {Key::D});
  mInput.Bind("sideways", {Key::A, false});
  mInput.Bind("upward", {Key::Space});
  mInput.Bind("upward", {Key::LeftControl, false});

  mInput.Bind("roll", {Key::E});
  mInput.Bind("roll", {Key::Q, false});

  mInput.Bind("forward", {GamepadAxis::LeftY, false});
  mInput.Bind("sideways", {GamepadAxis::LeftX});
  mInput.Bind("upward", {GamepadButton::A});
  mInput.Bind("upward", {GamepadButton::B, false});

  mInput.Bind("look_vertical", {GamepadAxis::RightY, false});
  mInput.Bind("look_horizontal", {GamepadAxis::RightX, false});
  mInput.Bind("roll", {GamepadButton::RightBumper});
  mInput.Bind("roll", {GamepadButton::LeftBumper, false});

  mInput.Bind("click", {MouseButton::Left});
  mInput.Bind("click", {GamepadAxis::RightTrigger});
  mInput.Bind("delete", {MouseButton::Right});
  mInput.Bind("delete", {GamepadButton::Y});

  mTime.Initialize();
}

void App::Run() {
  Init();

  // Noise noise(rng.NextUInt64(), Noise::Type::Perlin);
  // noise.SetFrequency(0.1f).SetFractalType(Noise::FractalType::FBm).SetFractalOctaves(2);
  PRNG rng(time(0));

  struct Transform {
    glm::vec3 pos{};
  };

  struct Velocity {
    glm::vec3 velocity{};
  };

  struct RigidBody {
    PhysicsBodyID id;
  };

  struct Renderable {
    MapleRenderer::MeshHndl mesh;
    MapleRenderer::MaterialHndl material;
  };

  std::vector<PhysicsBodyID> physicsBodies;

  auto shape = MaplePhysics::Box{};
  for (size_t i = 0; i < 10000; i++) {
    auto ent = mScene.CreateEntity();
    mScene.Add<Transform>(ent, Transform{glm::vec3(0)});
    auto dir = glm::normalize(glm::vec3(rng.NextFloat(-1), rng.NextFloat(), rng.NextFloat(-1)));
    auto speed = rng.NextFloat() * 500.0f + 50.0f;
    auto pos = dir * speed;

    auto angle = rng.NextFloat(0, glm::two_pi<float>());
    auto axis = glm::vec3(rng.NextFloat(-1.0f, 1.0f), rng.NextFloat(-1.0f, 1.0f), rng.NextFloat(-1.0f, 1.0f));

    auto data = MaplePhysics::BodyInfo{
      .entityID = static_cast<uint32_t>(ent),
      .shape = shape,
      .motionType = MaplePhysics::MotionType::Dynamic,
      .position = pos,
      .orientation = glm::normalize(glm::angleAxis(angle, axis)),
      .restitution = 0.2f,
    };
    auto rb = mPhysics.CreateRigidBody(data);

    physicsBodies.push_back(rb);
  }

  auto data = MaplePhysics::BodyInfo{0, MaplePhysics::Plane{}, MaplePhysics::MotionType::Static};
  auto floorRB = mPhysics.CreateRigidBody(data);
  // physicsBodies.push_back(rb);

  std::vector<glm::mat4> instances;

  float physicsDeltaTime = 1.0f / 60.0f;
  float remainingPhysicsTime = 0.0f;

  while (!mWindow.ShouldClose()) {
    mTime.BeginFrame();
    mInput.BeginFrame();
    mWindow.PollEvents();

    if (mInput.Released("exit")) mWindow.SetShouldClose(true);

    float movementSpeed = 15.0f;
    float rollSpeed = 1.5f;
    float mouseSens = 0.01f;
    float gamePadSens = 1.0f;

    auto movement = mCam.Forward() * mInput.Value("forward") + mCam.Right() * mInput.Value("sideways") + mCam.Up() * mInput.Value("upward");
    auto movementLength = glm::length(movement);
    if (movementLength > 1.0f * mTime.DeltaTime()) movement /= movementLength;
    movement *= movementSpeed * mTime.DeltaTime();
    mCam.SetPosition(mCam.GetPosition() + movement);

    glm::vec2 look(mInput.Value("look_horizontal"), mInput.Value("look_vertical"));
    look *= gamePadSens * mTime.DeltaTime();
    look += -mInput.GetMouseDelta() * mouseSens;
    mCam.Yaw(look.x);
    mCam.Pitch(look.y);
    mCam.Roll(mInput.Value("roll") * rollSpeed * mTime.DeltaTime());

    remainingPhysicsTime += mTime.DeltaTime();
    while (remainingPhysicsTime >= physicsDeltaTime) {
      remainingPhysicsTime -= physicsDeltaTime;
      mPhysics.Update(physicsDeltaTime);
    }

    instances.clear();
    for (auto id : physicsBodies) {
      glm::mat4 rotation = glm::mat4_cast(mPhysics.GetBodyRotation(id));
      glm::mat4 translation = glm::translate(glm::mat4(1.0f), mPhysics.GetBodyPosition(id));
      instances.push_back(translation * rotation);
    }

    instances.push_back(glm::translate(glm::scale(glm::mat4(1.0f), glm::vec3(1000)), glm::vec3(0, -0.5, 0)));

    if (mInput.Value("click") > 0.5) {
      auto rayResult = mPhysics.Raycast(mCam.GetPosition(), mCam.Forward(), 1000.0f);
      if (rayResult != std::nullopt) {
        instances.push_back(glm::scale(glm::translate(glm::mat4(1.0f), rayResult->position), glm::vec3(0.1f)));
        auto overlaps = mPhysics.OverlapSphere({.radius = 5}, rayResult->position);
        for (auto body : overlaps) {
          instances.push_back(glm::scale(glm::translate(glm::mat4(1.0f), mPhysics.GetBodyPosition(body)), glm::vec3(2.0f)));
          if (mInput.Value("delete") > 0.5) {
            if (body == floorRB) continue;
            mPhysics.DestroyRigidBody(body);
            // mScene.DestroyEntity(static_cast<Entity>(mPhysics.GetBodyEntity(body)));
          }
        }
      }
    }

    auto [frameBufferX, frameBufferY] = mWindow.GetFrameBufferSize();
    MapleRenderer::UBO ubo{
      .view = mCam.GetView(),
      .proj = mCam.GetProjection(float(frameBufferX) / frameBufferY, 60.0f, 0.1f, 1000.0f),
      .time = static_cast<float>(mTime.TimeSinceStart()),
    };

    std::vector<std::string> usedResources = {mTex1};
    std::array meshDraws = {MapleRenderer::MeshDraw{mMesh, instances, usedResources}};

    std::array materialDraws = {MapleRenderer::MaterialDraw{.material = mMaterial, .meshes = meshDraws}};
    std::array passDraws = {MapleRenderer::PassDraw{
      .passName = "draw",
      .materialDraws = materialDraws,
    }};

    mRenderer.DrawFrame(ubo, mCompiledRenderGraph, passDraws);
  }
}

App::~App() { MAPLE_INFO("Shutting down..."); }

}  // namespace maple