#!/bin/bash
set -e
BUILD_DIR="build"
BUILD_WITH_CUDA="OFF"
CLEAN_BUILD=0
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --cuda) BUILD_WITH_CUDA="ON"; shift ;;
    --clean) CLEAN_BUILD=1; shift ;;
    -h|--help)
      echo "Usage: ./build.sh [--cuda] [--clean]"
      echo "  --cuda : Enable CUDA compilation."
      echo "  --clean: Remove the build directory before starting."
      # exit 0 # Removed
      ;;
    *) echo "Unknown option: $1"; # exit 1 # Removed
      ;;
  esac
done
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "Project Root: ${PROJECT_ROOT}"
BUILD_PATH="${PROJECT_ROOT}/${BUILD_DIR}"
if [ "${CLEAN_BUILD}" -eq 1 ]; then
  if [ -d "${BUILD_PATH}" ]; then
    echo "Cleaning build directory: ${BUILD_PATH}"
    rm -rf "${BUILD_PATH}"
  else
    echo "Build directory ${BUILD_PATH} does not exist. No need to clean."
  fi
fi
mkdir -p "${BUILD_PATH}"
echo "Configuring project with CMake..."
echo "Build with CUDA: ${BUILD_WITH_CUDA}"
cd "${BUILD_PATH}"
cmake "${PROJECT_ROOT}" -DBUILD_WITH_CUDA=${BUILD_WITH_CUDA}
echo "Building project..."
cmake --build . --config Release
echo "Build complete. Executable should be in ${BUILD_PATH}/cudabsgs"
cd "${PROJECT_ROOT}"
# exit 0 # Removed
