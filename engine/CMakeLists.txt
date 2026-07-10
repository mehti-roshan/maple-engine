add_library(maple_engine STATIC third_party/stb_image.cpp)

add_subdirectory(maple_core)
add_subdirectory(maple_asset)
add_subdirectory(maple_logging)
add_subdirectory(maple_renderer)
add_subdirectory(maple_window)

target_include_directories(maple_engine PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(maple_engine maple_core maple_logging maple_renderer)