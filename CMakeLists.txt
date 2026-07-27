cmake_minimum_required(VERSION 3.21)
project(maple LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MAPLE_PHYSICS_DOUBLE_PRECISION ON CACHE BOOL "" FORCE)
add_subdirectory(engine)
add_subdirectory(runtime)