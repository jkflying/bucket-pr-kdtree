#!/bin/bash

set -e
base_dir=`pwd`
rm -rf build_coverage || true
mkdir build_coverage
pushd build_coverage
cmake .. -DCMAKE_BUILD_TYPE=Coverage -DCMAKE_CXX_FLAGS_COVERAGE="--coverage -fprofile-arcs -ftest-coverage -Og -g -fno-default-inline -fno-inline -fno-inline-small-functions -fno-elide-constructors"  -DCMAKE_EXE_LINKER_FLAGS_COVERAGE="--coverage -ftest-coverage -lgcov"
cmake --build .
./kdtree_test
# lcov -z --output-file coverage.info
lcov --capture --directory ${base_dir} --no-external --output-file coverage.info --rc lcov_branch_coverage=1
lcov --remove coverage.info \*/main.cpp  --rc lcov_branch_coverage=1 --output-file coverage.info
lcov --rc lcov_branch_coverage=1 --summary coverage.info
genhtml coverage.info --output-directory out --branch-coverage
