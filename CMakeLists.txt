cmake_minimum_required(VERSION 3.21)
project(maple LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_subdirectory(external)
add_subdirectory(engine)
add_subdirectory(runtime)