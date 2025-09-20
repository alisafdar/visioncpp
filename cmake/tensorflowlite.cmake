function(configure_tensorflowlite)
    cmake_parse_arguments(CONFIGURE "" "USE_XNNPACK;ENABLE_SIMD" "" ${ARGN})
    include(FetchContent)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    set(FETCHCONTENT_QUIET FALSE)

    find_program(FLATC_EXECUTABLE_PATH flatc)
    if (NOT FLATC_EXECUTABLE_PATH)
        message(FATAL_ERROR "flatc not found. Install FlatBuffers compiler on host.")
    endif ()

    FetchContent_Declare(
            tensorflow
            GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
            GIT_TAG v2.19.0
            GIT_PROGRESS TRUE
            SOURCE_SUBDIR "tensorflow/lite"
            PATCH_COMMAND ${CMAKE_COMMAND} -E echo "No patches"
            EXCLUDE_FROM_ALL
    )

    if (CONFIGURE_USE_XNNPACK)
        set(TFLITE_ENABLE_XNNPACK ON)
    endif()

    set(ABSL_PROPAGATE_CXX_STD ON)
    FetchContent_MakeAvailable(tensorflow)

    target_compile_definitions(tensorflow-lite PRIVATE FLATBUFFERS_LOCALE_INDEPENDENT=0)
endfunction()
