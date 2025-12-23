add_library(maple_engine STATIC core/engine.cpp)

add_subdirectory(logging)
add_subdirectory(renderer)

target_include_directories(maple_engine PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(maple_engine maple_logging maple_renderer glfw dl pthread X11 Xxf86vm Xrandr Xi)