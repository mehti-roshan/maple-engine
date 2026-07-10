add_library(maple_app STATIC app.cpp)

add_subdirectory(maple_asset)
add_subdirectory(maple_core)
add_subdirectory(maple_scene)
add_subdirectory(maple_logging)
add_subdirectory(maple_renderer)
add_subdirectory(maple_window)

target_include_directories(maple_app PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(maple_app maple_asset maple_core maple_scene maple_logging maple_renderer maple_window)