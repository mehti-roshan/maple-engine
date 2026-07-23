#include "maple_asset_loader.h"

#include <engine/maple_logging/log_macros.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "third_party/dr_wav.h"
#include "third_party/stb_image.h"

namespace maple {

std::vector<uint8_t> AssetLoader::LoadFileBytes(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) MAPLE_FATAL("failed to open file {}", filename);

  size_t fileSize = file.tellg();
  std::vector<uint8_t> buffer(fileSize);

  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

  file.close();

  return buffer;
}

std::string AssetLoader::LoadFileStr(const std::string& filename) {
  std::vector<uint8_t> v = LoadFileBytes(filename);

  char* ptr = static_cast<char*>(malloc(v.size() + 1));
  ptr[v.size()] = '\0';

  std::memcpy(ptr, v.data(), v.size());

  return ptr;
}

AssetLoader::Image AssetLoader::LoadImage(const std::string& filename) {
  stbi_set_flip_vertically_on_load(1);

  AssetLoader::Image img{};

  auto ptr = stbi_load(filename.c_str(), &img.size.x, &img.size.y, &img.numChannels, STBI_rgb_alpha);
  if (!ptr) MAPLE_FATAL("failed to load texture");
  static auto constexpr NumChannels = 4;  // 4 is the num of channels we forced stb to load with
  img.bytes.resize(img.size.x * img.size.y * NumChannels);
  std::memcpy(img.bytes.data(), ptr, img.bytes.size());

  stbi_image_free(ptr);

  return img;
}

AssetLoader::Audio AssetLoader::LoadAudio(const std::string& filename) {
  AssetLoader::Audio audio{};

  auto bytes = LoadFileBytes(filename);
  auto ptr = drwav_open_memory_and_read_pcm_frames_s16(
    bytes.data(), bytes.size(), &audio.channels, &audio.sampleRate, reinterpret_cast<drwav_uint64*>(&audio.sampleCount), nullptr);
  if (!ptr) MAPLE_FATAL("failed to load audio '{}'", filename);

  audio.bitsPerSample = 16;  // if changing this or the s16 load above, update this
  audio.floatingPointFormat = false;

  auto bufSize = audio.bitsPerSample / 8 * audio.channels * audio.sampleCount;
  audio.data = std::make_unique<uint8_t[]>(bufSize);
  std::memcpy(audio.data.get(), ptr, bufSize);

  drwav_free(ptr, nullptr);

  return audio;
}

}  // namespace maple