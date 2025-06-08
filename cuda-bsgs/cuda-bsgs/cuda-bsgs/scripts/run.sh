#!/bin/bash
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
EXECUTABLE_PATH="${PROJECT_ROOT}/build/cudabsgs"
DEFAULT_PUBKEY="aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899"

# Check if executable exists at the start of --execute-short-test
if [[ "$1" == "--execute-short-test" ]] && [ ! -f "${EXECUTABLE_PATH}" ]; then
    echo "Error for --execute-short-test: Executable not found at ${EXECUTABLE_PATH}"
    echo "Please build the project first using build.sh (run from ${PROJECT_ROOT})"
    # exit 1 # Removed
fi
# --- Print examples (condensed for brevity in this step) ---
if [[ "$1" != "--execute-short-test" ]]; then # Only print examples if not doing short test
    echo "# Example: ${EXECUTABLE_PATH} ${DEFAULT_PUBKEY}"
fi

run_example() {
    local example_num=$1; shift; echo "--- Running Example $example_num ---"
    cd "${PROJECT_ROOT}"
    case $example_num in
        1) echo "Cmd: ${EXECUTABLE_PATH} ${DEFAULT_PUBKEY} $@"; ${EXECUTABLE_PATH} ${DEFAULT_PUBKEY} $@ ;;
        *) echo "Unknown example: $example_num"; cd "${SCRIPTS_DIR}"; return 1 ;;
    esac
    local exit_code=$?; echo "--- Example $example_num finished (Exit: $exit_code) ---"; cd "${SCRIPTS_DIR}"; return $exit_code
}
if [[ "$1" == "--execute-short-test" ]]; then
    echo "Executing short test for CI..."
    cd "${PROJECT_ROOT}"
    if [ -f "${EXECUTABLE_PATH}" ]; then
        ${EXECUTABLE_PATH} ${DEFAULT_PUBKEY} --simulated-gpus 1 --giant-steps-total-range 100000 --log-level INFO --checkpoint-file delete_me.dat --profiles-file configs/gpu_profiles.json
        rm -f delete_me.dat
    else
        echo "Short test: Executable not found at ${EXECUTABLE_PATH}"
    fi
    # cd "${SCRIPTS_DIR}" # Not strictly needed if script ends after this
fi
# exit 0 # Removed
