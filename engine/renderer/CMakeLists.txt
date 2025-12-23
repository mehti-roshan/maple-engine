add_library(maple_renderer STATIC renderer.cpp)
target_include_directories(maple_renderer PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(maple_renderer vulkan maple_logging)