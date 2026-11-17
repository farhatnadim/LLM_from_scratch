# Options and helpers for all C++ targets of the course.

# Python of the reference environment. The tests use it to make the golden
# files. A training run or an inference run does not use it (R-4.3).
set(COURSE_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH
    "Python of the reference environment (made by 'uv sync')")

# C++ standard, warnings and the source root. The standard is set for each
# target, never globally (see cmake/LibTorch.cmake).
add_library(course_options INTERFACE)
target_compile_features(course_options INTERFACE cxx_std_23)
target_compile_options(course_options INTERFACE -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(course_options INTERFACE COURSE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(course_options INTERFACE libtorch)

# One precompiled header for <torch/torch.h>. All course targets reuse it,
# so the large header compiles one time only.
add_library(course_pch OBJECT "${CMAKE_SOURCE_DIR}/cmake/pch.cpp")
target_link_libraries(course_pch PRIVATE course_options doctest nlohmann_json)
target_precompile_headers(course_pch PRIVATE <torch/torch.h>)
set_target_properties(course_pch PROPERTIES CXX_EXTENSIONS OFF)

# course_target(<target>): the options and the shared precompiled header.
function(course_target target)
    target_link_libraries(${target} PUBLIC course_options)
    target_link_libraries(${target} PRIVATE doctest nlohmann_json)
    target_precompile_headers(${target} REUSE_FROM course_pch)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
endfunction()

# course_test(<name> SOURCES <files> [LIBS <targets>] [LABELS <labels>] [FIXTURES <setup>])
# Makes one test program and one CTest test. The test needs the golden files.
function(course_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;LABELS;FIXTURES" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    course_target(${name})
    target_link_libraries(${name} PRIVATE course_testmain course ${ARG_LIBS})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        LABELS "${ARG_LABELS}"
        FIXTURES_REQUIRED "goldens;${ARG_FIXTURES}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    if(NOT ARG_FIXTURES)
        set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED goldens)
    endif()
endfunction()
