function(embed_resources target)
    set(resource_files ${ARGN})
    set(EMBEDDED_RESOURCES_DIR ${CMAKE_BINARY_DIR}/${target}_embedded_resources)
    file(MAKE_DIRECTORY ${EMBEDDED_RESOURCES_DIR})
    set(EMBEDDED_RESOURCE_FILES)
    foreach (RESOURCE_FILE ${resource_files})
        get_filename_component(RESOURCE_FILE_NAME ${RESOURCE_FILE} NAME)
        get_filename_component(RESOURCE_FILE_NAME_WE ${RESOURCE_FILE} NAME_WE)
        set(EMBEDDED_RESOURCE_FILE "${EMBEDDED_RESOURCES_DIR}/${RESOURCE_FILE_NAME}.c")
        add_custom_command(
                OUTPUT ${EMBEDDED_RESOURCE_FILE}
                COMMAND xxd -n "${RESOURCE_FILE_NAME_WE}" -i ${RESOURCE_FILE} > ${EMBEDDED_RESOURCE_FILE}
                DEPENDS ${RESOURCE_FILE}
                COMMENT "Embedding ${RESOURCE_FILE_NAME}"
                VERBATIM
        )
        list(APPEND EMBEDDED_RESOURCE_FILES ${EMBEDDED_RESOURCE_FILE})
    endforeach ()
    target_sources(${target} PRIVATE ${EMBEDDED_RESOURCE_FILES})
    add_custom_target(${target}_resources ALL DEPENDS ${EMBEDDED_RESOURCE_FILES})
    add_dependencies(${target} ${target}_resources)
endfunction()
