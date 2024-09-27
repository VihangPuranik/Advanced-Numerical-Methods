find_package(Threads REQUIRED)

include(ExternalProject)
ExternalProject_Add(
        GTestProject
        URL https://github.com/google/googletest/archive/release-1.11.0.zip
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/gtest_source
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/gtest_binary
        CMAKE_ARGS "-Dgtest_force_shared_crt=ON"
        INSTALL_COMMAND  ""
)
add_library(GTest INTERFACE)
target_link_directories(
        GTest INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}/gtest_binary/lib
)

target_include_directories(GTest SYSTEM INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}/gtest_source/googletest/include
        )

# If you get linker error related to not finding
# -lgtestd
#  note -^
# Then you should:
# a) remove this target_link_libraries and uncomment the next one.
# b) write to us.
target_link_libraries(
        GTest INTERFACE
        gtest$<$<AND:$<PLATFORM_ID:Windows>,$<CONFIG:Debug>>:d>
)

## uncomment the next line if you have the problem above.
# target_link_libraries(GTest INTERFACE gtest)


target_link_libraries(
        GTest INTERFACE
        ${THREADS_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
)
add_dependencies(GTest GTestProject)
