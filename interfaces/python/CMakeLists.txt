# This file is part of PIQP.
#
# Copyright (c) 2024 EPFL
# Copyright (c) 2022 INRIA
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 OLD)
endif()

# Find nanobind
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# detect arm64 cross compilation on macOS
if(DEFINED ENV{_PYTHON_HOST_PLATFORM})
    if($ENV{_PYTHON_HOST_PLATFORM} MATCHES "arm64")
        set(CMAKE_SYSTEM_PROCESSOR "arm64")
    endif()
endif()

# add instruction set detection on x86/amd64
nanobind_add_module(instruction_set src/instruction_set.cpp)

# only link cpu_features on x86/amd64
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    target_compile_definitions(instruction_set PRIVATE CPU_FEATURES_AVAILABLE=1)
    target_link_libraries(instruction_set PRIVATE cpu_features)
endif()

function(CREATE_PIQP_PYTHON_TARGET target_name library_target COMPILE_OPTIONS)
    nanobind_add_module(${target_name} src/piqp_python.cpp)
    target_compile_options(${target_name} PRIVATE ${COMPILE_OPTIONS})
    # Link against the shared library instead of header-only
    target_link_libraries(${target_name} PRIVATE ${library_target})
    target_compile_definitions(${target_name} PRIVATE PYTHON_MODULE_NAME=${target_name})
    # SKBUILD_PROJECT_VERSION_FULL is defined by scikit-build-core and passed into the C++ code as a define (VERSION_INFO) here.
    target_compile_definitions(${target_name} PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION_FULL})
    if(UNIX AND NOT APPLE)
        target_link_options(${target_name} PRIVATE "-Wl,-z,noexecstack")
    endif()
    
    # Set RPATH to find the shared library at runtime
    if(APPLE)
        set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "@loader_path/lib"
        )
    elseif(UNIX)
        set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "$ORIGIN/lib"
        )
    endif()
endfunction()

if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    set(AVX_COMPILE_OPTION "/arch:AVX")
    set(AVX2_COMPILE_OPTION "/arch:AVX2")
    set(FMA_COMPILE_OPTION "/fp:fast")
    set(AVX512_COMPILE_OPTION "/arch:AVX512")
else()
    set(AVX_COMPILE_OPTION "-mavx")
    set(AVX2_COMPILE_OPTION "-mavx2")
    set(FMA_COMPILE_OPTION "-mfma;-ffp-contract=on")
    set(AVX512_COMPILE_OPTION "-mavx512f")
endif()

create_piqp_python_target(piqp_python piqp_shared "")
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    create_piqp_python_target(piqp_python_avx2 piqp_shared_avx2 "${AVX2_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
    create_piqp_python_target(piqp_python_avx512 piqp_shared_avx512 "${AVX512_COMPILE_OPTION};${FMA_COMPILE_OPTION}")
endif()

# BLASFEO linking is now handled by the shared library targets (piqp_shared, piqp_shared_avx2, piqp_shared_avx512)
# which are linked by CREATE_PIQP_PYTHON_TARGET

install(TARGETS instruction_set LIBRARY DESTINATION piqp)
install(TARGETS piqp_python LIBRARY DESTINATION piqp)
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    install(TARGETS piqp_python_avx2 LIBRARY DESTINATION piqp)
    install(TARGETS piqp_python_avx512 LIBRARY DESTINATION piqp)
endif()

# Install shared libraries
install(TARGETS piqp_shared LIBRARY DESTINATION piqp/lib)
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    install(TARGETS piqp_shared_avx2 LIBRARY DESTINATION piqp/lib)
    install(TARGETS piqp_shared_avx512 LIBRARY DESTINATION piqp/lib)
endif()

# Install headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/piqp DESTINATION piqp/include FILES_MATCHING PATTERN "*.hpp" PATTERN "*.tpp")

# Install CMake config files
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/piqpConfigWheel.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/piqpConfig.cmake"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/piqpConfig.cmake" DESTINATION piqp/cmake)

# Also install the version file for compatibility
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/piqpConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/piqpConfigVersion.cmake" DESTINATION piqp/cmake)

# Generate type stubs for piqp_python module at install time
nanobind_add_stub(
    piqp_python_stub
    INSTALL_TIME
    MODULE piqp.piqp_python
    OUTPUT piqp/__init__.pyi
    MARKER_FILE piqp/py.typed
    COMPONENT python_stubs
)
