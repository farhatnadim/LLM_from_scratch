#pragma once
// Test macros for the lesson tests. Include this file only in *_test.cpp files.

#include <doctest/doctest.h>

#include "course/compare.hpp"
#include "course/golden.hpp"
#include "course/todo.hpp"

/// Checks that two floating-point tensors agree to tol (largest absolute difference).
#define CHECK_CLOSE(actual, expected, tol)                                            \
    do {                                                                              \
        const ::course::Comparison course_cmp_ =                                      \
            ::course::compare_close((actual), (expected), (tol));                     \
        CHECK_MESSAGE(course_cmp_.ok, #actual " vs " #expected ": " << course_cmp_.report); \
    } while (false)

/// Same as CHECK_CLOSE, but stops the test case on failure.
#define REQUIRE_CLOSE(actual, expected, tol)                                          \
    do {                                                                              \
        const ::course::Comparison course_cmp_ =                                      \
            ::course::compare_close((actual), (expected), (tol));                     \
        REQUIRE_MESSAGE(course_cmp_.ok, #actual " vs " #expected ": " << course_cmp_.report); \
    } while (false)

/// Checks that two tensors have equal shape, dtype and values.
#define CHECK_TENSOR_EQUAL(actual, expected)                                          \
    do {                                                                              \
        const ::course::Comparison course_cmp_ =                                      \
            ::course::compare_equal((actual), (expected));                            \
        CHECK_MESSAGE(course_cmp_.ok, #actual " vs " #expected ": " << course_cmp_.report); \
    } while (false)

/// Same as CHECK_TENSOR_EQUAL, but stops the test case on failure.
#define REQUIRE_TENSOR_EQUAL(actual, expected)                                        \
    do {                                                                              \
        const ::course::Comparison course_cmp_ =                                      \
            ::course::compare_equal((actual), (expected));                            \
        REQUIRE_MESSAGE(course_cmp_.ok, #actual " vs " #expected ": " << course_cmp_.report); \
    } while (false)
