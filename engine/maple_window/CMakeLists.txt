include(FetchContent)

FetchContent_Declare(
  SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG release-3.2.0
)

FetchContent_MakeAvailable(SDL3)

add_library(maple_window STATIC maple_window.cpp)

target_include_directories(maple_window PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(maple_window SDL3::SDL3)