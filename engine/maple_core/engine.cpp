#include <engine/maple_asset/maple_asset.h>
#include <engine/maple_logging/log_macros.h>
#include <engine/maple_renderer/material_builder_data.h>
#include <engine/maple_renderer/render_graph.h>
#include <engine/third_party/stb_image.h>

#include <bit>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <vector>

#include "engine/maple_core/input_enums.h"
#include "engine/maple_window/maple_window.h"
#include "enums.h"
#include "noise.h"
#include "prng.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/glm.hpp>

#include "engine.h"

namespace maple {

void Engine::Run() {
  Init();

  mInput.Bind("exit", {Key::Escape});
  mInput.Bind("exit", {GamepadButton::Start});

  mInput.Bind("forward", {Key::W});
  mInput.Bind("forward", {Key::S, false});
  mInput.Bind("sideways", {Key::D});
  mInput.Bind("sideways", {Key::A, false});
  mInput.Bind("upwards", {Key::Space});
  mInput.Bind("upwards", {Key::LeftControl, false});

  mInput.Bind("roll", {Key::E});
  mInput.Bind("roll", {Key::Q, false});

  mInput.Bind("forward", {GamepadAxis::LeftY, false});
  mInput.Bind("sideways", {GamepadAxis::LeftX});
  mInput.Bind("upwards", {GamepadButton::A});
  mInput.Bind("upwards", {GamepadButton::B, false});

  mInput.Bind("look_vertical", {GamepadAxis::RightY, false});
  mInput.Bind("look_horizontal", {GamepadAxis::RightX, false});
  mInput.Bind("roll", {GamepadButton::RightBumper});
  mInput.Bind("roll", {GamepadButton::LeftBumper, false});

  std::vector<glm::mat4> InstanceTransforms;
  PRNG rng(time(0));
  Noise noise(rng.NextUInt64(), Noise::Type::Perlin);
  noise.SetFrequency(0.1f).SetFractalType(Noise::FractalType::FBm).SetFractalOctaves(2);

  uint32_t dimensions = 100;
  for (size_t x = 0; x < dimensions; x++) {
    for (int64_t z = 0; z < dimensions; z++) {
      float n = noise.GetNoisef(x, z) * 5.0f;

      InstanceTransforms.push_back(glm::translate(glm::mat4(1.0f), glm::vec3(x, 0, -z) + glm::vec3(0, n, 0)));
    }
  }

  while (!mWindow.ShouldClose()) {
    mTime.BeginFrame();
    mInput.BeginFrame();
    mWindow.PollEvents();

    MAPLE_DEBUG(mTime.DeltaTime());

    if (mInput.Released("exit")) mWindow.SetShouldClose(true);

    float movementSpeed = 15.0f;
    float rollSpeed = 1.5f;
    float mouseSens = 0.01f;
    float gamePadSens = 1.0f;

    auto movement = mCam.Forward() * mInput.Value("forward") + mCam.Right() * mInput.Value("sideways") + mCam.Up() * mInput.Value("upwards");
    if (glm::length(movement) > 1.0f * mTime.DeltaTime()) movement = glm::normalize(movement);
    movement *= movementSpeed * mTime.DeltaTime();
    mCam.SetPosition(mCam.GetPosition() + movement);

    glm::vec2 look(mInput.Value("look_horizontal"), mInput.Value("look_vertical"));
    look *= gamePadSens * mTime.DeltaTime();
    look += -mInput.GetMouseDelta() * mouseSens;
    mCam.Yaw(look.x);
    mCam.Pitch(look.y);
    mCam.Roll(mInput.Value("roll") * rollSpeed * mTime.DeltaTime());

    auto [frameBufferX, frameBufferY] = mWindow.GetFrameBufferSize();

    MapleRenderer::UBO ubo{
      .view = mCam.GetView(),
      .proj = mCam.GetProjection(float(frameBufferX) / frameBufferY, 60.0f, 0.1f, 1000.0f),
      .time = static_cast<float>(mTime.TimeSinceStart()),
    };

    std::vector<std::string> usedResources = {mTex1};
    std::array meshDraws = {MapleRenderer::MeshDraw{mMesh, InstanceTransforms, usedResources}};

    std::array materialDraws = {MapleRenderer::MaterialDraw{.material = mMaterial, .meshes = meshDraws}};
    std::array passDraws = {MapleRenderer::PassDraw{
      .passName = "draw",
      .materialDraws = materialDraws,
    }};

    mRenderer.DrawFrame(ubo, mCompiledRenderGraph, passDraws);
  }
}

void Engine::Init() {
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
  std::array verts = {
    Vertex{glm::vec3(-0.5f, -0.5f, 0.0f), glm::vec2(0.0f, 0.0f)},
    Vertex{glm::vec3(-0.5f, 0.5f, 0.0f), glm::vec2(0.0f, 1.0f)},
    Vertex{glm::vec3(0.5f, -0.5f, 0.0f), glm::vec2(1.0f, 0.0f)},
    Vertex{glm::vec3(0.5f, 0.5f, 0.0f), glm::vec2(1.0f, 1.0f)},
  };

  std::vector<uint32_t> indices = {
    0,
    2,
    1,

    1,
    2,
    3,
  };

  mMesh = mRenderer.CreateMesh({
    .verts = std::as_bytes(std::span<const Vertex>(verts)),
    .indices = indices,
    .numVerts = static_cast<uint32_t>(verts.size()),
  });

  mMaterial = mRenderer.CreateMaterial(
    MapleAsset::ReadFileStr("assets/shaders/shader.slang"), "shader", {.rasterizer = {.cullMode = MaterialBuilderData::CullModeFlagBits::None}});

  std::array colorFormats = {Format::R8G8B8A8_SRGB, Format::B8G8R8A8_SRGB};
  auto format = mRenderer.FindFirstSupportedTextureFormat(colorFormats);
  if (!format.has_value()) MAPLE_FATAL("failed to find suitable color format");

  glm::ivec2 dimensions;
  int32_t texChannels;
  stbi_set_flip_vertically_on_load(1);

  auto ptr = stbi_load(std::string("assets/textures/" + mTex1).c_str(), &dimensions.x, &dimensions.y, &texChannels, STBI_rgb_alpha);
  if (!ptr) MAPLE_FATAL("failed to load texture");
  mRenderer.CreateTexture(mTex1, dimensions, std::span<uint8_t>(ptr, dimensions.x * dimensions.y * texChannels * 4), 4 * 8, *format);
  stbi_image_free(ptr);

  ptr = stbi_load(std::string("assets/textures/" + mTex2).c_str(), &dimensions.x, &dimensions.y, &texChannels, STBI_rgb_alpha);
  if (!ptr) MAPLE_FATAL("failed to load texture");
  mRenderer.CreateTexture(mTex2, dimensions, std::span<uint8_t>(ptr, dimensions.x * dimensions.y * texChannels * 4), 4 * 8, *format);
  stbi_image_free(ptr);

  mTime.Initialize();
}

Engine::~Engine() { MAPLE_INFO("Shutting down..."); }

}  // namespace maple