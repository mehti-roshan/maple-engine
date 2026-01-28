add_library(maple_file STATIC file.cpp)
target_include_directories(maple_file PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(maple_file maple_logging)