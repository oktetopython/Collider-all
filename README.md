# CUDA BSGS Collider

A CUDA-based implementation of the Baby-Step Giant-Step (BSGS) algorithm, designed for solving instances of the discrete logarithm problem, typically in the context of elliptic curve cryptography. This project leverages NVIDIA GPUs to accelerate the computationally intensive parts of the algorithm.

**Note:** This README describes the `cuda-bsgs` C++/CUDA project. Information regarding a Windows `Collider.exe` (potentially built with PureBasic) found in older READMEs pertains to a separate wrapper or version and is not covered by the build and run instructions below.

## Disclaimer

ALL THE CODES, PROGRAMS, AND INFORMATION ARE FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. USE IT AT YOUR OWN RISK. THE DEVELOPER WILL NOT BE RESPONSIBLE FOR ANY LOSS, DAMAGE, OR CLAIM ARISING FROM USING THIS PROGRAM. USERS ARE RESPONSIBLE FOR ENSURING COMPLIANCE WITH ALL APPLICABLE LAWS AND REGULATIONS.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **NVIDIA GPU**: A CUDA-enabled NVIDIA GPU. Compute Capability 3.5 or higher is generally recommended (as per the original `cuda-bsgs/README.md`).
*   **CUDA Toolkit**: Version 11.0 or later is recommended. This provides the CUDA compiler (`nvcc`) and runtime libraries.
*   **CMake**: Version 3.15 or later. Used for building the project.
*   **C++ Compiler**: A C++17 compatible compiler (e.g., GCC 7.0 or later).
*   **Git**: For cloning the repository.
*   **NVIDIA Drivers**: Ensure you have appropriate NVIDIA drivers installed for your GPU and CUDA toolkit version.

## Building the Project

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Navigate to the CUDA BSGS directory:**
    ```bash
    cd cuda-bsgs
    ```

3.  **Run the Build Script:**
    The provided `build.sh` script simplifies the CMake configuration and build process. To build with CUDA support (recommended for GPU acceleration), use the `--cuda` flag:
    ```bash
    ./scripts/build.sh --cuda
    ```
    If you want to do a clean build, you can add the `--clean` flag:
    ```bash
    ./scripts/build.sh --clean --cuda
    ```
    The script will create a `build` directory inside `cuda-bsgs` and compile the project.

4.  **Locate the Executable:**
    The compiled executable will be located at `cuda-bsgs/build/cudabsgs`.

**Manual CMake Build (Alternative):**

If you prefer to run CMake manually:

```bash
cd cuda-bsgs
mkdir build
cd build
cmake .. -DBUILD_WITH_CUDA=ON # Set to OFF for CPU-only (simulation) mode
make -j$(nproc) # Or simply 'make'
```

## Running the Application

The `cudabsgs` executable accepts several command-line arguments to configure its operation.

**Synopsis:**

```bash
./cuda-bsgs/build/cudabsgs <target_pubkey_hex> [options]
```

**Required Argument:**

*   `<target_pubkey_hex>`: A 64-character hexadecimal string representing the target public key.

**Common Options:**

*   `--gpu-list <ids>`: Comma-separated list of GPU IDs to use (e.g., "0,1"). If not specified, the application attempts to use all available GPUs or defaults to simulated ones if `--simulated-gpus` is set.
*   `--simulated-gpus <count>`: Number of simulated GPUs to use (default: 1). Useful for testing logic without a physical CUDA-enabled GPU. Ensure `BUILD_WITH_CUDA` is OFF or that no real GPUs are detected for this to be the primary mode of operation.
*   `--baby-steps-count <N>`: Specifies the exact number of baby steps to compute (e.g., `1048576` for 2^20).
*   `--w <val>`: Sets the `w_param`. This parameter is often used as an exponent for the number of baby steps (2<sup>w</sup>), especially by profiles. Consult `gpu_profiles.json` or experiment. The default in code is 64 (likely too high for direct exponent use without a profile override). Using `--baby-steps-count` might be more straightforward.
*   `--htsz <MB>`: Size of the hash table in Megabytes (e.g., `2048`). This is the `htsz_param_mb`.
*   `--giant-steps-total-range <N>`: The total range of giant steps to cover.
*   `--steps-per-kernel-launch <N>`: The number of giant steps processed per single CUDA kernel launch.
*   `--checkpoint-file <path>`: Path to the checkpoint file (default: `bsgs_checkpoint.dat`).
*   `--checkpoint-interval <N>`: Save checkpoint every N steps/operations (default: `1 << 15`).
*   `--log-level <level>`: Set logging verbosity. Options: `DEBUG`, `INFO`, `WARN`, `ERROR` (default: `INFO`).
*   `--profiles-file <path>`: Path to the GPU profiles JSON configuration file (default: `configs/gpu_profiles.json`).
*   `--gpu-profile <name>`: Name of a specific GPU profile to load and apply from the `profiles-file`. This overrides automatic profile selection.

**Example Usage:**

```bash
# Run with a specific public key, using GPU 0, and specifying baby steps count and hash table size
./cuda-bsgs/build/cudabsgs aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899     --gpu-list 0     --baby-steps-count 1048576     --htsz 1024     --log-level DEBUG
```

## Configuration

The application can utilize GPU-specific profiles defined in a JSON file (default: `cuda-bsgs/configs/gpu_profiles.json`).

*   **Profiles**: The JSON file contains an array named `"profiles"`. Each element is a profile object.
*   **Profile Matching**:
    *   If `--gpu-profile <name>` is used, the application will try to load the profile with that exact name.
    *   Otherwise, it attempts to auto-match a profile based on the detected GPU's SM (Streaming Multiprocessor) major version (e.g., 7 for Volta, 8 for Ampere).
*   **Profile Parameters**: A profile can define:
    *   `name` (string): Unique name for the profile.
    *   `matches_sm_major` (int): The SM major version this profile is intended for.
    *   `params` (object):
        *   `w_param` (int): Value for the 'w' parameter (often 2<sup>w</sup> baby steps).
        *   `htsz_param_mb` (int): Hash table size in MB.
        *   `conceptual_threads_per_block` (int): Threads per CUDA block.
        *   `conceptual_blocks_factor` (int): Factor for calculating grid size.

If no specific profile is matched or forced, the application will use default parameters or those provided via command-line arguments. Command-line arguments generally override profile settings.

## Checkpointing

The application supports checkpointing to save and resume progress:

*   **Saving**: Progress is automatically saved to the file specified by `--checkpoint-file` (default: `bsgs_checkpoint.dat`) at intervals defined by `--checkpoint-interval`.
*   **Resuming**: If a valid checkpoint file is found at startup, the application will attempt to resume from the saved state. Ensure that BSGS parameters (like public key, baby step count, etc.) are consistent with the checkpoint.

## Output

*   **Logging**: The application logs its progress and findings to standard output, based on the selected `--log-level`.
*   **Found Keys**: When a private key is successfully found (a "collision" in BSGS terms), the application will log this information. The exact format or method of saving (e.g., to a specific file like `Found.txt`) is not explicitly managed by the `cuda-bsgs/src/host.cpp` source and might require manual redirection of output or be part of an external wrapper script if persistent file output is desired.

## Elliptic Curve Operations

The provided CUDA kernel (`cuda-bsgs/src/kernel.cu`) contains an abstracted framework for the BSGS algorithm. The specific elliptic curve point arithmetic (e.g., point addition, scalar multiplication for a curve like secp256k1) is represented by placeholder operations in the publicly visible code. A functional collider for a specific cryptocurrency would require these operations to be fully and correctly implemented according to the target curve's mathematics.

## Development and Simulation

The codebase includes CUDA stubs that allow compiling and running the host logic in a simulated GPU environment. This is useful for development and testing on machines without NVIDIA hardware or for debugging CPU-side logic.

*   To build for simulation: Ensure `BUILD_WITH_CUDA=OFF` during CMake configuration (e.g., run `./scripts/build.sh` without `--cuda`).
*   Use `--simulated-gpus <count>` to specify the number of simulated GPUs.

## Contributing

Contributions are welcome. Please fork the repository, make your changes on a separate branch, and submit a pull request for review. Ensure your code adheres to the existing style and includes appropriate documentation or tests where applicable.

## License

This project is licensed under the MIT License. See the `LICENSE` file in the repository for details (assuming one exists, standard practice).
