#include "maple_asset.h"

#include <log_macros.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

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
}  // namespace maple