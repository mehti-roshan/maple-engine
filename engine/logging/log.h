#pragma once
#include <memory>

namespace spdlog {
class logger;
}

namespace maple::logging {

class Log {
 public:
  static void init();

  static std::shared_ptr<spdlog::logger>& get();
};

}  // namespace maple::logging