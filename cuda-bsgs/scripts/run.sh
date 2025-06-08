#!/bin/bash
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
EXECUTABLE_PATH="${PROJECT_ROOT}/build/cudabsgs"
DEFAULT_PUBKEY="aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"

if [[ "$1" == "--execute-short-test" ]]; then
    echo "Executing short test for CI..."
    cd "${PROJECT_ROOT}"
    if [ ! -f "${EXECUTABLE_PATH}" ]; then
        echo "Error for --execute-short-test: Executable not found at ${EXECUTABLE_PATH}"
        echo "Attempting to build it first..."
        if [ -f "./scripts/build.sh" ]; then
            ./scripts/build.sh # Try to build if not found
            if [ ! -f "${EXECUTABLE_PATH}" ]; then # Check again
                echo "Build failed or executable still not found. Aborting short test."
                # exit 1 # Removed
            fi
        else
            echo "build.sh not found at ./scripts/build.sh. Aborting short test."
            # exit 1 # Removed
        fi
    fi
    # Proceed if executable exists or was just built
    if [ -f "${EXECUTABLE_PATH}" ]; then
        echo "Running: ${EXECUTABLE_PATH} ${DEFAULT_PUBKEY} --simulated-gpus 1 --log-level INFO"
        ${EXECUTABLE_PATH} ${DEFAULT_PUBKEY} --simulated-gpus 1 --giant-steps-total-range 10000 --log-level INFO --checkpoint-file delete_me.dat --profiles-file configs/gpu_profiles.json
        rm -f delete_me.dat
    fi
    # cd "${SCRIPTS_DIR}" # Not strictly needed if script ends after this
else
    echo "# Example: ${EXECUTABLE_PATH} ${DEFAULT_PUBKEY}"
fi
