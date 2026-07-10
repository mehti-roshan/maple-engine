#include <cstdint>
#include <string>
#include <vector>

namespace maple {
class MapleAsset {
 public:
  static std::vector<uint8_t> ReadFileBytes(const std::string& filename);
  static std::string ReadFileStr(const std::string& filename);
};

}  // namespace maple
