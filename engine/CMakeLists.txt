include(FetchContent)

FetchContent_Declare(
  SDL2
  URL https://github.com/libsdl-org/SDL/archive/refs/tags/release-2.30.0.tar.gz
)

FetchContent_MakeAvailable(SDL2)

add_library(maple_engine STATIC core/application.cpp)

add_subdirectory(logging)

target_include_directories(maple_engine PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(maple_engine maple_logging SDL2::SDL2-static)