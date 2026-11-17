cmake_minimum_required(VERSION 3.28)
project(LLM_from_scratch LANGUAGES CXX)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "Use a preset: 'cmake --preset clang' (or gcc). It builds in build/<preset>/.")
endif()

# The course uses no C++ modules. Without the module scan, each file compiles faster.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include(cmake/LibTorch.cmake)
include(cmake/Dependencies.cmake)
include(cmake/Course.cmake)

enable_testing()

# Golden files: the Python reference writes them; the C++ tests read them.
# All tests need this setup step. It skips the work when nothing changed.
add_test(NAME goldens
    COMMAND "${COURSE_PYTHON}" "${CMAKE_SOURCE_DIR}/tools/make_goldens.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(goldens PROPERTIES FIXTURES_SETUP goldens LABELS "smoke")

add_subdirectory(course/cpp)
