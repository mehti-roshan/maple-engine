#include "file.h"
#include <fstream>
#include <log_macros.h>

std::vector<char> maple::file::ReadFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open()) MAPLE_FATAL("failed to open file {}", filename);

  size_t fileSize = file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}