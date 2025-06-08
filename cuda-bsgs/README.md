# CUDA Implementation of Baby-Step Giant-Step Algorithm

**IMPORTANT NOTE:** For the most up-to-date and comprehensive instructions on building, running, and configuring this project, please refer to the main `README.md` file in the root directory of this repository. This current file provides a general overview but might not reflect all specific command-line options or the recommended `build.sh` script usage.

This repository contains a CUDA-based implementation of the Baby-Step Giant-Step (BSGS) algorithm, an efficient method for solving the discrete logarithm problem.

## Table of Contents

- [Introduction](#introduction)
- [Algorithm Details](#algorithm-details)
- [Prerequisites](#prerequisites)
- [Building the Project](#building-the-project)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Baby-Step Giant-Step algorithm is a meet-in-the-middle algorithm used to solve the discrete logarithm problem (DLP) in a finite cyclic group. This implementation leverages CUDA to accelerate the computation, particularly the "baby steps" and "giant steps" phases.

## Algorithm Details

Given a cyclic group $G$, a generator $g$, and an element $h \in G$, the DLP is to find an integer $x$ such that $g^x = h$. The BSGS algorithm works as follows:

1.  **Parameter Selection**: Choose an integer $m \ge \sqrt{N}$, where $N$ is the order of the group.
2.  **Baby Steps**: Compute $g^j \pmod{p}$ for $0 \le j < m$. Store these pairs $(j, g^j \pmod{p})$ in a hash table or a sorted list.
3.  **Giant Steps**: Compute $h(g^{-m})^k \pmod{p}$ for $0 \le k < m$.
4.  **Collision Detection**: For each giant step, check if the computed value exists in the stored baby steps. If a match $g^j = h(g^{-m})^k$ is found, then $x = j + mk$ is the solution.

This CUDA implementation parallelizes the computation of baby steps and giant steps, significantly speeding up the process for large groups.

## Prerequisites

-   NVIDIA GPU with CUDA support (Compute Capability 3.5 or higher)
-   CUDA Toolkit (version 11.0 or later recommended)
-   CMake (version 3.15 or later)
-   A C++ compiler with C++17 support (e.g., GCC 7.0 or later)

## Building the Project

Instructions for building the project using CMake.

```bash
mkdir build
cd build
cmake ..
make
```

## Running the Application

Detailed instructions on how to run the compiled application, including command-line arguments.

```bash
./bsgs <g> <h> <p> [options]
```

Refer to `src/host.cpp` or run with `--help` for more details on options.

## Configuration

The application can be configured using a JSON file (`configs/gpu_profiles.json`). This file allows specifying GPU-specific parameters like block size, grid size, etc., to optimize performance for different NVIDIA GPU architectures.

Example `gpu_profiles.json`:

```json
{
  "profiles": [
    {
      "gpu_arch": "sm_70", // Volta
      "block_size": 256,
      "grid_size_factor": 64
    },
    {
      "gpu_arch": "sm_80", // Ampere
      "block_size": 512,
      "grid_size_factor": 128
    }
  ],
  "default_profile": {
    "block_size": 256,
    "grid_size_factor": 32
  }
}
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure your code adheres to the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
