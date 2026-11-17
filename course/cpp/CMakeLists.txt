# Shared C++ helpers of the course. No model logic here (docs/specification.md, S-2).
add_library(course STATIC
    course/golden.cpp
    course/compare.cpp
    course/args.cpp
    course/csv_log.cpp)
target_include_directories(course PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
course_target(course)

add_library(course_testmain OBJECT testmain.cpp)
course_target(course_testmain)

# Unit test of the helpers.
course_test(course_test SOURCES tests/course_test.cpp LABELS "smoke")

# Smoke test of milestone M0. One program, three CTest tests:
#   smoke_cpu: LibTorch on the CPU; C++ reads the file of Python and writes one for Python
#   smoke_py:  Python reads the file of C++
#   smoke_gpu: LibTorch on the graphics processor
add_executable(smoke_test tests/smoke_test.cpp)
course_target(smoke_test)
target_link_libraries(smoke_test PRIVATE course_testmain course)

add_test(NAME smoke_cpu COMMAND smoke_test --test-case=CPU*)
set_tests_properties(smoke_cpu PROPERTIES LABELS "smoke" FIXTURES_REQUIRED goldens FIXTURES_SETUP smoke_cpp_file)

# "-m" keeps course/py off sys.path[0], where io.py would hide the module io of Python.
add_test(NAME smoke_py COMMAND "${COURSE_PYTHON}" -m course.py.smoke
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(smoke_py PROPERTIES LABELS "smoke" FIXTURES_REQUIRED "goldens;smoke_cpp_file")

add_test(NAME smoke_gpu COMMAND smoke_test --test-case=GPU*)
set_tests_properties(smoke_gpu PROPERTIES LABELS "smoke;gpu" FIXTURES_REQUIRED goldens)
