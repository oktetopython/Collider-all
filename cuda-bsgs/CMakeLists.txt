cmake_minimum_required(VERSION 3.18)
project(CudaBSGS CXX CUDA) # Added CUDA language back

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to build with CUDA
option(BUILD_WITH_CUDA "Enable CUDA compilation" OFF)
option(WITH_NVML_MONITORING "Enable NVML for GPU temperature monitoring" OFF)

if(BUILD_WITH_CUDA)
    find_package(CUDA REQUIRED) # Find CUDA only if we intend to use it
    if(NOT CUDA_FOUND)
        message(WARNING "BUILD_WITH_CUDA is ON but CUDA toolkit not found. Disabling CUDA.")
        set(BUILD_WITH_CUDA OFF) # Force off if not found
    else()
        # NVML Setup (only if CUDA is found and NVML monitoring is requested)
        if(WITH_NVML_MONITORING)
            message(STATUS "WITH_NVML_MONITORING is ON. Attempting to find NVML.")
            if(WIN32)
                set(NVML_LIBRARY_NAME "nvml")
            else()
                set(NVML_LIBRARY_NAME "nvidia-ml")
            endif()

            find_library(NVML_LIBRARY NAMES ${NVML_LIBRARY_NAME})
            find_path(NVML_INCLUDE_DIR NAMES nvml.h)

            if(NVML_LIBRARY AND NVML_INCLUDE_DIR)
                message(STATUS "NVML found: Library=${NVML_LIBRARY}, Include=${NVML_INCLUDE_DIR}")
                # NVML components will be added to the target further down if BUILD_WITH_CUDA is confirmed.
            else()
                message(WARNING "NVML library or headers not found. Disabling NVML monitoring.")
                set(WITH_NVML_MONITORING OFF CACHE BOOL "NVML components not found, force disable" FORCE)
            endif()
        else()
            message(STATUS "NVML monitoring disabled by option.")
        endif()
    endif()
endif()


if(BUILD_WITH_CUDA)
    message(STATUS "CUDA compilation ENABLED.")
    add_executable(cudabsgs
        src/host.cpp
        src/checkpoint.cpp
        src/kernel.cu
    )
    # For .cu files, LANGUAGE CUDA is implicit if project has CUDA enabled.
    # If host.cpp contains CUDA calls (like kernel launches), it also needs to be compiled with nvcc.
    set_source_files_properties(src/host.cpp PROPERTIES LANGUAGE CUDA) # If host.cpp makes CUDA RT calls or launches kernels
    # set_source_files_properties(src/kernel.cu PROPERTIES LANGUAGE CUDA) # Implicit for .cu

    set(CUDA_ARCHITECTURES "70;75;80;86;90")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
    message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # Using modern CMake targets for CUDA
    target_link_libraries(cudabsgs PRIVATE CUDA::cudart)

    # Add NVML if found and enabled
    if(WITH_NVML_MONITORING AND NVML_LIBRARY AND NVML_INCLUDE_DIR)
      message(STATUS "Enabling NVML for target cudabsgs.")
      target_include_directories(cudabsgs PRIVATE ${NVML_INCLUDE_DIR})
      target_link_libraries(cudabsgs PRIVATE ${NVML_LIBRARY})
      target_compile_definitions(cudabsgs PRIVATE WITH_NVML)
    endif()

else()
    message(STATUS "CUDA compilation DISABLED. Building C++ only version (cudabsgs).")
    add_executable(cudabsgs
        src/host.cpp
        src/checkpoint.cpp
    )
endif()

target_include_directories(cudabsgs PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src # If checkpoint.hpp is in src
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Project source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Target executable: cudabsgs")
