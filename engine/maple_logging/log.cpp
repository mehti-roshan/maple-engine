#include "log.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace maple::logging {

static std::shared_ptr<spdlog::logger> s_logger;

void Log::init() {
  spdlog::set_pattern("[%T] [%^%l%$] %v");
  s_logger = spdlog::stdout_color_mt("MAPLE");
  s_logger->set_level(spdlog::level::trace);
}

std::shared_ptr<spdlog::logger>& Log::get() { return s_logger; }

}  // namespace maple::logging