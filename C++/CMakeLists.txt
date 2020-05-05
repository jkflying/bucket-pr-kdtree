cmake_minimum_required(VERSION 3.7)
project(kdtree_h)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(kdtree_test test/main.cpp)
target_include_directories(kdtree_test PUBLIC "include")
