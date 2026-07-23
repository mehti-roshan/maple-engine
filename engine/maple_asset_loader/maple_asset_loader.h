#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace maple {
class AssetLoader {
 public:
  struct Image {
    std::vector<uint8_t> bytes;
    glm::ivec2 size{};
    int32_t numChannels = 0;
  };

  struct Audio {
    bool floatingPointFormat = false;
    uint8_t bitsPerSample = 0;
    uint32_t channels = 0;
    uint32_t sampleRate = 0;
    uint64_t sampleCount = 0;
    std::unique_ptr<uint8_t[]> data;
  };

  static std::vector<uint8_t> LoadFileBytes(const std::string& filename);
  static std::string LoadFileStr(const std::string& filename);
  static Image LoadImage(const std::string& filename);
  static Audio LoadAudio(const std::string& filename);
};

}  // namespace maple
