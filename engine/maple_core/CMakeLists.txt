add_library(maple_core STATIC camera.cpp input.cpp noise.cpp prng.cpp time.cpp)

target_include_directories(maple_core PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(maple_core maple_window maple_renderer maple_logging)