cmake_minimum_required(VERSION 3.14)
project(flinn VERSION 0.7 LANGUAGES CXX)

add_library(flinn INTERFACE)
target_include_directories(flinn INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(flinn INTERFACE cxx_std_11)

set(CMAKE_CXX_FLAGS_COVERAGE "--coverage -fprofile-arcs -ftest-coverage -Og -g -fprofile-abs-path -fprofile-update=atomic")
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage -ftest-coverage -lgcov")
set(CMAKE_CXX_FLAGS_ASAN "-fsanitize=address -g -O2 -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_ASAN "-fsanitize=address -g")

include(GNUInstallDirs)
install(TARGETS flinn EXPORT flinnTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT flinnTargets
    FILE flinnConfig.cmake
    NAMESPACE flinn::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/flinn
)

option(FLINN_BUILD_TESTS "Build tests" OFF)
option(FLINN_BUILD_BENCH "Build benchmarks" OFF)

if(MSVC)
    set(FLINN_WARNING_FLAGS /W4 /WX)
else()
    set(FLINN_WARNING_FLAGS -Wall -Wextra -pedantic -Werror)
endif()

if(FLINN_BUILD_TESTS)
    add_executable(flinn_test test/main.cpp)
    target_link_libraries(flinn_test PRIVATE flinn)
    target_compile_options(flinn_test PRIVATE ${FLINN_WARNING_FLAGS})
endif()

if(FLINN_BUILD_BENCH)
    add_executable(flinn_benchmark bench/benchmark.cpp)
    target_link_libraries(flinn_benchmark PRIVATE flinn)
    target_compile_features(flinn_benchmark PRIVATE cxx_std_14)
    target_compile_options(flinn_benchmark PRIVATE -O3 -DNDEBUG -march=native)
endif()
