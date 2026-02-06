#pragma once
#include <spdlog/spdlog.h>

#include "log.h"

#define MAPLE_INFO(...) ::maple::logging::Log::get()->info(__VA_ARGS__)
#define MAPLE_WARN(...) ::maple::logging::Log::get()->warn(__VA_ARGS__)
#define MAPLE_ERROR(...) ::maple::logging::Log::get()->error(__VA_ARGS__)
#define MAPLE_DEBUG(...) ::maple::logging::Log::get()->debug(__VA_ARGS__)
#define MAPLE_FATAL(...)                                 \
  do {                                                   \
    ::maple::logging::Log::get()->critical(__VA_ARGS__); \
    std::terminate();                                    \
  } while (0)

// Assert macro
#if defined(NDEBUG)
#define MAPLE_ASSERT(cond, ...) ((void)0)
#else
#define MAPLE_ASSERT(cond, ...)                                                                             \
  do {                                                                                                      \
    if (!(cond)) {                                                                                          \
      ::maple::logging::Log::get()->critical("Assertion failed: {} | {}", #cond, fmt::format(__VA_ARGS__)); \
      std::abort();                                                                                         \
    }                                                                                                       \
  } while (0)
#endif