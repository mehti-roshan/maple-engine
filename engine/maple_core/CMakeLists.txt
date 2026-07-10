add_library(maple_core STATIC engine.cpp camera.cpp input.cpp noise.cpp prng.cpp time.cpp ../third_party/stb_image.cpp)

target_include_directories(maple_core PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(maple_core maple_window maple_renderer maple_logging)