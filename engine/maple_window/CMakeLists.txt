add_library(maple_window STATIC maple_window.cpp)

target_include_directories(maple_window PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(maple_window glfw dl pthread X11 Xxf86vm Xrandr Xi)