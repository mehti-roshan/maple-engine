#include "maple_asset.h"

#include <engine/maple_logging/log_macros.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "third_party/stb_image.h"

namespace maple {

std::vector<uint8_t> MapleAsset::ReadFileBytes(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) MAPLE_FATAL("failed to open file {}", filename);

  size_t fileSize = file.tellg();
  std::vector<uint8_t> buffer(fileSize);

  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

  file.close();

  return buffer;
}

std::string MapleAsset::ReadFileStr(const std::string& filename) {
  std::vector<uint8_t> v = ReadFileBytes(filename);

  char* ptr = static_cast<char*>(malloc(v.size() + 1));
  ptr[v.size()] = '\0';

  std::memcpy(ptr, v.data(), v.size());

  return ptr;
}

MapleAsset::Image MapleAsset::LoadImage(const std::string& filename) {
  stbi_set_flip_vertically_on_load(1);

  MapleAsset::Image img{};

  auto ptr = stbi_load(filename.c_str(), &img.size.x, &img.size.y, &img.numChannels, STBI_rgb_alpha);
  if (!ptr) MAPLE_FATAL("failed to load texture");
  static auto constexpr NumChannels = 4;  // 4 is the num of channels we forced stb to load with
  img.bytes.resize(img.size.x * img.size.y * NumChannels);
  std::memcpy(img.bytes.data(), ptr, img.bytes.size());

  stbi_image_free(ptr);

  return img;
}

}  // namespace maple