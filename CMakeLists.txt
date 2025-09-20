cmake_minimum_required(VERSION 3.22)
project(visioncpp LANGUAGES C CXX)

option(VISIONCPP_BUILD_TFLITE_FROM_SOURCE "Build TensorFlow Lite from source" ON)
option(VISIONCPP_USE_XNNPACK              "Enable XNNPACK in TFLite"          ON)
option(VISIONCPP_ENABLE_JPEG              "Enable JPEG (imgcodecs)"           ON)
option(VISIONCPP_ENABLE_PNG               "Enable PNG (imgcodecs)"            OFF)
option(VISIONCPP_ENABLE_SIMD              "Enable OpenCV intrinsics"          ON)
option(VISIONCPP_USE_GMS_TFLITE           "Link gmscore TFLite instead"       OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions($<$<CONFIG:Release>:RELEASE> $<$<CONFIG:Debug>:DEBUG>)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/opencv.cmake)
configure_opencv(
        ENABLE_THREADING ON
        ENABLE_SIMD ${VISIONCPP_ENABLE_SIMD}
        ENABLE_INTRINSICS ${VISIONCPP_ENABLE_SIMD}
        ENABLE_JPEG ${VISIONCPP_ENABLE_JPEG}
        ENABLE_PNG  ${VISIONCPP_ENABLE_PNG}
        ENABLE_FILE_SYSTEM_SUPPORT OFF
)

if (VISIONCPP_USE_GMS_TFLITE)
    find_package(tensorflowlite_jni_gms_client REQUIRED CONFIG)
elseif (VISIONCPP_BUILD_TFLITE_FROM_SOURCE)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/tensorflowlite.cmake)
    configure_tensorflowlite(USE_XNNPACK ${VISIONCPP_USE_XNNPACK} ENABLE_SIMD OFF)
endif()

file(GLOB_RECURSE VISIONCPP_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_library(visioncpp STATIC ${VISIONCPP_SRC})
target_include_directories(visioncpp PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(visioncpp PUBLIC opencv)

if (VISIONCPP_USE_GMS_TFLITE)
    target_link_libraries(visioncpp PUBLIC tensorflowlite_jni_gms_client::tensorflowlite_jni_gms_client)
    target_compile_definitions(visioncpp PUBLIC TFLITE_GMSCORE=1)
elseif (VISIONCPP_BUILD_TFLITE_FROM_SOURCE)
    target_link_libraries(visioncpp PUBLIC tensorflow-lite)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/embed_resources.cmake)
set(VISIONCPP_RESOURCES
        ${CMAKE_CURRENT_LIST_DIR}/resources/detect.tflite
        ${CMAKE_CURRENT_LIST_DIR}/resources/labelmap.txt
)
embed_resources(visioncpp ${VISIONCPP_RESOURCES})

if (ANDROID)
    target_compile_options(visioncpp PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    if (${ANDROID_ABI} STREQUAL "arm64-v8a")
        target_compile_options(visioncpp PRIVATE -march=armv8-a)
    elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
        target_compile_options(visioncpp PRIVATE -march=armv7-a -mfloat-abi=softfp -mfpu=vfpv3-d16)
    endif()
endif()
