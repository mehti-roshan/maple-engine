FetchContent_Declare(
  JoltPhysics
  GIT_REPOSITORY "https://github.com/jrouwe/JoltPhysics.git"
  GIT_TAG "v5.6.0"
  SOURCE_SUBDIR Build
)

FetchContent_MakeAvailable(JoltPhysics)

target_compile_definitions(Jolt PUBLIC JPH_DOUBLE_PRECISION)

add_library(maple_physics STATIC maple_physics.cpp)

target_include_directories(maple_physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(maple_physics PRIVATE Jolt maple_logging)