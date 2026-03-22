#!/usr/bin/env bash
set -euo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v clang-format &>/dev/null; then
    sudo apt-get install -y clang-format libnanoflann-dev
fi

# clang-format check
FORMAT_ERRORS=0
for f in $(find "$CPP_DIR/include" "$CPP_DIR/test" "$CPP_DIR/bench" -not -path '*/build/*' -type f \( -name '*.h' -o -name '*.cpp' \) 2>/dev/null); do
    DIFF="$(clang-format --style=file:"$CPP_DIR/.clang-format" "$f" | diff -u "$f" - || true)"
    if [ -n "$DIFF" ]; then
        echo "$DIFF"
        FORMAT_ERRORS=1
    fi
done
if [ "$FORMAT_ERRORS" -ne 0 ]; then
    exit 1
fi

# release build
rm -rf "$CPP_DIR/build"
cmake -B "$CPP_DIR/build" -S "$CPP_DIR" -DCMAKE_BUILD_TYPE=Release -DFLINN_BUILD_TESTS=ON -DFLINN_BUILD_BENCH=ON
cmake --build "$CPP_DIR/build"
"$CPP_DIR/build/flinn_test"

# install test
INSTALL_DIR="$(mktemp -d)"
trap "rm -rf '$INSTALL_DIR'" EXIT
cmake --install "$CPP_DIR/build" --prefix "$INSTALL_DIR"

cat > "$INSTALL_DIR/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.14)
project(flinn_install_test LANGUAGES CXX)
find_package(flinn REQUIRED)
add_executable(flinn_install_test test.cpp)
target_link_libraries(flinn_install_test PRIVATE flinn::flinn)
EOF

cat > "$INSTALL_DIR/test.cpp" <<'EOF'
#include <flinn.h>
int main() {
    flinn::FlinnIndex<int, 2> tree;
    tree.addPoint({{0.0, 0.0}}, 1);
    return tree.size() == 1 ? 0 : 1;
}
EOF

cmake -B "$INSTALL_DIR/build" -S "$INSTALL_DIR" -DCMAKE_PREFIX_PATH="$INSTALL_DIR"
cmake --build "$INSTALL_DIR/build"
"$INSTALL_DIR/build/flinn_install_test"

# asan build
rm -rf "$CPP_DIR/build"
cmake -B "$CPP_DIR/build" -S "$CPP_DIR" -DCMAKE_BUILD_TYPE=asan -DFLINN_BUILD_TESTS=ON
cmake --build "$CPP_DIR/build"
ASAN_OPTIONS=detect_leaks=0 "$CPP_DIR/build/flinn_test"

# coverage build
rm -rf "$CPP_DIR/build"
cmake -B "$CPP_DIR/build" -S "$CPP_DIR" -DCMAKE_BUILD_TYPE=Coverage -DFLINN_BUILD_TESTS=ON
cmake --build "$CPP_DIR/build"

VENV_DIR="$CPP_DIR/build/venv"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -q fastcov

"$VENV_DIR/bin/fastcov" -z -d "$CPP_DIR/build"
"$CPP_DIR/build/flinn_test"

"$VENV_DIR/bin/fastcov" -d "$CPP_DIR/build" -i "$(realpath "$CPP_DIR")" -e test/ bench/ -l -o "$CPP_DIR/build/coverage.info"
lcov --summary "$CPP_DIR/build/coverage.info" --ignore-errors mismatch,inconsistent
genhtml "$CPP_DIR/build/coverage.info" --output-directory "$CPP_DIR/build/coverage_html" --branch-coverage --ignore-errors mismatch,inconsistent
