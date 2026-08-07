option(MAPLE_PHYSICS_DOUBLE_PRECISION "Build Physics engine with double precision" OFF)

include(FetchContent)
FetchContent_Declare(
  box3d
  GIT_REPOSITORY https://github.com/erincatto/box3d.git
  GIT_TAG v0.1.0
)

set(BOX3D_DOUBLE_PRECISION ${MAPLE_PHYSICS_DOUBLE_PRECISION})

FetchContent_MakeAvailable(box3d)

add_library(maple_physics STATIC maple_physics.cpp)

target_compile_definitions(maple_physics PUBLIC $<$<BOOL:${MAPLE_PHYSICS_DOUBLE_PRECISION}>:MAPLE_PHYSICS_DOUBLE_PRECISION>)

target_include_directories(maple_physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(maple_physics PRIVATE box3d::box3d maple_logging)