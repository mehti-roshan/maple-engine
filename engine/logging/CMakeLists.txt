include(FetchContent)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.16.0
)

FetchContent_MakeAvailable(spdlog)

add_library(maple_logging STATIC log.cpp)

target_include_directories(maple_logging PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(maple_logging PUBLIC spdlog::spdlog)