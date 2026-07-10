#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace maple {
class MapleAsset {
 public:
  struct Image {
    std::vector<uint8_t> bytes;
    glm::ivec2 size{};
    int32_t numChannels = 0;
  };

  static std::vector<uint8_t> ReadFileBytes(const std::string& filename);
  static std::string ReadFileStr(const std::string& filename);
  static Image LoadImage(const std::string& filename);
};

}  // namespace maple
