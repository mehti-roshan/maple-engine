include(FetchContent)
FetchContent_Declare(
  fetch_vk_bootstrap
  GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
  GIT_TAG        v1.4.342
)

FetchContent_MakeAvailable(fetch_vk_bootstrap)

add_library(maple_renderer STATIC renderer.cpp renderer_vma.cpp)
target_include_directories(maple_renderer PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(maple_renderer vulkan maple_logging maple_file vk-bootstrap::vk-bootstrap)