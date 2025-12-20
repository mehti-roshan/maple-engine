#pragma once
#include "log.h"
#include <spdlog/spdlog.h>

#define MAPLE_INFO(...) ::maple::logging::Log::get()->info(__VA_ARGS__)
#define MAPLE_WARN(...) ::maple::logging::Log::get()->warn(__VA_ARGS__)
#define MAPLE_ERROR(...) ::maple::logging::Log::get()->error(__VA_ARGS__)
#define MAPLE_DEBUG(...) ::maple::logging::Log::get()->debug(__VA_ARGS__)
#define MAPLE_FATAL(...)                                 \
  do {                                                   \
    ::maple::logging::Log::get()->critical(__VA_ARGS__); \
    std::terminate();                                    \
  } while (0)
