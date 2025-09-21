cmake_minimum_required(VERSION 3.22)

# ===== Build type flags =====
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE "Release")
    add_compile_definitions(RELEASE)
else ()
    add_compile_definitions(DEBUG)
endif ()

# ===== Project =====
project(visioncpp
        LANGUAGES C CXX
        VERSION 0.1.0
        DESCRIPTION "High-performance vision core for AR/Object detection"
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== Configure options (mirrors sample, renamed for our project) =====
option(VISIONCPP_BUILD_CLI            "Build CLI executable for desktop testing" OFF)
option(VISIONCPP_BUILD_WASM_SHIM      "Build WASM shim (Emscripten)"            OFF)
option(VISIONCPP_BUILD_TFLITE_FROM_SOURCE "Compile TensorFlow Lite from source" OFF)
option(VISIONCPP_BUILD_USE_XNNPACK    "Compile TensorFlow Lite with XNNPACK"    OFF)

if (VISIONCPP_BUILD_WASM_SHIM)
    option(VISIONCPP_WASM_ENABLE_THREADING "Compile WASM with -pthread" OFF)
    option(VISIONCPP_WASM_ENABLE_SIMD      "Compile WASM with -msimd128" OFF)
endif ()

# Auto-enable TFLite like the sample does
if (VISIONCPP_BUILD_CLI OR VISIONCPP_BUILD_WASM_SHIM)
    message(STATUS "Implicitly enabling TensorFlow Lite (CLI or WASM selected).")
    set(VISIONCPP_BUILD_TFLITE_FROM_SOURCE ON)
endif ()
if (VISIONCPP_BUILD_CLI)
    message(STATUS "Implicitly enabling XNNPACK for CLI.")
    set(VISIONCPP_BUILD_USE_XNNPACK ON)
endif ()
if (VISIONCPP_BUILD_CLI AND VISIONCPP_BUILD_WASM_SHIM)
    message(FATAL_ERROR "Only one of 'VISIONCPP_BUILD_WASM_SHIM' and 'VISIONCPP_BUILD_CLI' may be enabled.")
endif ()
if (VISIONCPP_BUILD_USE_XNNPACK)
    add_compile_definitions(USE_XNNPACK)
endif ()


include(${CMAKE_CURRENT_LIST_DIR}/cmake/opencv.cmake)
if (VISIONCPP_BUILD_WASM_SHIM)
    configure_opencv(
            ENABLE_THREADING ${VISIONCPP_WASM_ENABLE_THREADING}
            ENABLE_SIMD      ${VISIONCPP_WASM_ENABLE_SIMD}
            ENABLE_INTRINSICS OFF
            ENABLE_JPEG ON
            ENABLE_PNG  OFF
            ENABLE_FILE_SYSTEM_SUPPORT OFF
            ENABLE_LAPACK ON
    )
elseif (VISIONCPP_BUILD_CLI)
    configure_opencv(
            ENABLE_THREADING ON
            ENABLE_SIMD ON
            ENABLE_INTRINSICS ON
            ENABLE_JPEG ON
            ENABLE_PNG  ON
            ENABLE_FILE_SYSTEM_SUPPORT ON
            ENABLE_LAPACK OFF
    )
else ()
    configure_opencv(
            ENABLE_THREADING ON
            ENABLE_SIMD ON
            ENABLE_INTRINSICS ON
            ENABLE_JPEG ON
            ENABLE_PNG  OFF
            ENABLE_FILE_SYSTEM_SUPPORT OFF
            ENABLE_LAPACK ON
    )
endif ()

if (VISIONCPP_BUILD_TFLITE_FROM_SOURCE)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/tensorflowlite.cmake)
    configure_tensorflowlite(
            USE_XNNPACK ${VISIONCPP_BUILD_USE_XNNPACK}
            ENABLE_SIMD ${VISIONCPP_WASM_ENABLE_SIMD}
    )
endif ()

# ===== Sources
file(GLOB_RECURSE VISIONCPP_SRC CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
)
add_library(visioncpp ${VISIONCPP_SRC})

target_include_directories(visioncpp PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)

# ===== Embed Binary Resources (model + labels)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/embed_resources.cmake)
set(VISIONCPP_RESOURCES
        ${CMAKE_CURRENT_LIST_DIR}/resources/detect.tflite
        ${CMAKE_CURRENT_LIST_DIR}/resources/labelmap.txt
)
embed_resources(visioncpp ${VISIONCPP_RESOURCES})

# ===== Linkage
target_link_libraries(visioncpp PUBLIC
        opencv
)

# If we built TFLite here, also link it here (desktop/CLI). Otherwise, the parent (Android JNI)
# must link the provider (GMS client) into 'visioncpp' so that tflite_wrapper.cpp resolves.
if (VISIONCPP_BUILD_TFLITE_FROM_SOURCE)
    target_link_libraries(visioncpp PUBLIC tensorflow-lite)
endif ()

# ===== WASM / CLI targets (optional, like sample)
if (VISIONCPP_BUILD_WASM_SHIM)
    # include(platforms/wasm/configure.cmake)
    # configure wasm target here if you add it later
endif ()
if (VISIONCPP_BUILD_CLI)
    # include(platforms/cli/configure.cmake)
    # configure CLI target here if you add it later
endif ()

# ===== Android niceties (no GMS logic here)
if (ANDROID)
    target_compile_options(visioncpp PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        target_compile_options(visioncpp PRIVATE -march=armv8-a)
    elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
        target_compile_options(visioncpp PRIVATE -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16)
    endif()
endif ()
