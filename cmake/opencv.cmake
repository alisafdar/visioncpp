## @brief Configures the opencv library target.
## @param ENABLE_THREADING Enables multithreading (ON/OFF).
## @param ENABLE_SIMD Enables SIMD (ON/OFF).
## @param ENABLE_INTRINSICS Enables intrinsic optimizations (ON/OFF).
## @param ENABLE_JPEG Enables compiling jpeg codec. (ON/OFF).
## @param ENABLE_PNG Enables compiling png codec. (ON/OFF).
## @param ENABLE_FILE_SYSTEM_SUPPORT Enables file system support. (ON/OFF).
## @param ENABLE_LAPACK enable LAPACK for speed optimization (disable for tests). (ON/OFF)
function(configure_opencv)
    cmake_parse_arguments(CONFIGURE
            ""
            "ENABLE_THREADING;ENABLE_SIMD;ENABLE_INTRINSICS;ENABLE_JPEG;ENABLE_PNG;ENABLE_FILE_SYSTEM_SUPPORT;ENABLE_LAPACK"
            ""
            ${ARGN})

    include(FetchContent)

    # Allowing git progress
    set(FETCHCONTENT_QUIET FALSE)

    FetchContent_Declare(
            opencv
            GIT_REPOSITORY https://github.com/opencv/opencv.git
            GIT_TAG 4.10.0
            GIT_PROGRESS TRUE
            EXCLUDE_FROM_ALL
    )

    set(WITH_V4L OFF)
    set(WITH_FFMPEG OFF)
    set(WITH_JPEG OFF)
    set(WITH_PNG OFF)
    set(WITH_TIFF OFF)
    set(WITH_GSTREAMER OFF)
    set(WITH_MSMF OFF)
    set(WITH_DSHOW OFF)
    set(WITH_AVFOUNDATION OFF)
    set(WITH_1394 OFF)
    set(WITH_MATLAB OFF)
    set(WITH_CUDA OFF)
    set(WITH_OPENGL OFF)
    set(WITH_CAROTENE OFF)
    set(WITH_WEBP OFF)
    set(WITH_OPENJPEG OFF)
    set(WITH_JASPER OFF)
    set(WITH_OPENEXR OFF)
    set(WITH_OpenCL OFF)
    set(WITH_OPENCLAMDFFT OFF)
    set(WITH_OPENCLAMDBLAS OFF)
    set(WITH_IPP OFF)
    set(WITH_OPENMP OFF)
    set(WITH_OBSENSOR OFF)
    set(WITH_FLATBUFFERS OFF)
    set(WITH_LAPACK OFF)

    set(BUILD_ITT OFF)
    set(BUILD_IPP_IW OFF)
    set(BUILD_opencv_calib3d OFF)
    set(BUILD_opencv_flann OFF)
    set(BUILD_opencv_features2d OFF)
    set(BUILD_opencv_dnn OFF)
    set(BUILD_PROTOBUF OFF)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_opencv_objdetect OFF)
    set(BUILD_opencv_gapi OFF)
    set(BUILD_opencv_world OFF)
    set(BUILD_opencv_video OFF)
    set(BUILD_opencv_videoio OFF)
    set(BUILD_opencv_highgui OFF)
    set(BUILD_opencv_ml OFF)
    set(BUILD_opencv_photo OFF)
    set(BUILD_opencv_stitching OFF)
    set(BUILD_opencv_superres OFF)
    set(BUILD_opencv_ts OFF)
    set(BUILD_opencv_apps OFF)
    set(BUILD_opencv_videostab OFF)
    set(BUILD_opencv_imgcodecs OFF)
    set(BUILD_opencv_objc_bindings_generator OFF)
    set(BUILD_opencv_js_bindings_generator OFF)
    set(BUILD_opencv_java_bindings_generator OFF)
    set(BUILD_opencv_java OFF)
    set(BUILD_opencv_js OFF)
    set(BUILD_opencv_objc OFF)
    set(BUILD_opencv_python OFF)
    set(BUILD_opencv_python3 OFF)
    set(BUILD_opencv_python2 OFF)
    set(BUILD_opencv_python_tests OFF)
    set(BUILD_TESTS OFF)
    set(BUILD_PERF_TESTS OFF)
    set(BUILD_DOCS OFF)
    set(BUILD_EXAMPLES OFF)
    set(BUILD_JPEG OFF)
    set(BUILD_PNG OFF)
    set(BUILD_TIFF OFF)

    set(VIDEOIO_ENABLED_PLUGINS OFF)
    set(OPENCV_DISABLE_FILESYSTEM_SUPPORT ON)
    set(CV_ENABLE_INTRINSICS OFF)

    set(BUILD_opencv_core ON)
    set(BUILD_opencv_imgproc ON)

    if (CONFIGURE_ENABLE_THREADING STREQUAL "ON")
        message(STATUS "OpenCV threading is enabled.")
        set(WITH_PTHREADS_PF ON)
    endif ()

    if (CONFIGURE_ENABLE_INTRINSICS STREQUAL "ON")
        message(STATUS "OpenCV intrinsics are enabled.")
        set(CV_ENABLE_INTRINSICS ON)
    endif ()

    if (CONFIGURE_ENABLE_JPEG STREQUAL "ON")
        message(STATUS "OpenCV jpeg codec is enabled.")
        set(BUILD_JPEG ON)
        set(WITH_JPEG ON)
        set(BUILD_opencv_imgcodecs ON)
        set(ENABLE_LIBJPEG_TURBO_SIMD OFF)

        # libjpeg-turbo SIMD support relies on x86 SSE NASM sources that cannot be translated by Emscripten.
        # Emscripten can only work with SSE intrinsics. Workarounds that utilize neon intrinsics did not
        # introduce any performance improvements.
        # Reference: https://github.com/libjpeg-turbo/libjpeg-turbo/issues/250
        if (CONFIGURE_ENABLE_SIMD STREQUAL "ON" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
            message(STATUS "OpenCV simd for jpeg-turbo is enabled.")
            set(ENABLE_LIBJPEG_TURBO_SIMD ON)
        endif ()
    endif ()

    if (CONFIGURE_ENABLE_PNG STREQUAL "ON")
        message(STATUS "OpenCV png codec is enabled.")
        set(BUILD_PNG ON)
        set(WITH_PNG ON)
        set(BUILD_opencv_imgcodecs ON)
    endif ()

    if (CONFIGURE_ENABLE_FILE_SYSTEM_SUPPORT STREQUAL "ON")
        message(STATUS "OpenCV file file system support is enabled.")
        set(OPENCV_DISABLE_FILESYSTEM_SUPPORT OFF)
    endif ()

    if (CONFIGURE_ENABLE_LAPACK STREQUAL "ON")
        message(STATUS "ENABLE_LAPACK=ON => forced WITH_LAPACK=ON in OpenCV.")
        set(WITH_LAPACK ON)
    endif ()

    FetchContent_MakeAvailable(opencv)

    add_library(opencv INTERFACE)

    set(OPENCV_LIBRARY_TARGETS
            opencv_core
            opencv_imgproc
    )
    set(OPENCV_LIBRARY_TARGETS_INCLUDE
            ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
            ${OPENCV_MODULE_opencv_core_LOCATION}/include
            ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include
    )

    if (BUILD_opencv_imgcodecs)
        list(APPEND OPENCV_LIBRARY_TARGETS opencv_imgcodecs)
        list(APPEND OPENCV_LIBRARY_TARGETS_INCLUDE ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include)
    endif ()

    target_include_directories(opencv INTERFACE ${OPENCV_LIBRARY_TARGETS_INCLUDE})
    target_link_libraries(opencv INTERFACE ${OPENCV_LIBRARY_TARGETS})
endfunction()
