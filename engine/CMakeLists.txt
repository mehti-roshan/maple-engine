add_library(maple_app STATIC app.cpp)

add_subdirectory(maple_asset_loader)
add_subdirectory(maple_audio)
add_subdirectory(maple_core)
add_subdirectory(maple_physics)
add_subdirectory(maple_logging)
add_subdirectory(maple_renderer)
add_subdirectory(maple_window)

target_include_directories(maple_app PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(maple_app maple_asset_loader maple_audio maple_core maple_physics maple_logging maple_renderer maple_window)