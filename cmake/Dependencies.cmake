# Header-only libraries, downloaded at a pinned version with a SHA-256 check
# (docs/specification.md, S-1). The files go into the build tree, not into git.

set(COURSE_THIRD_PARTY "${CMAKE_BINARY_DIR}/third_party/include")

function(course_download url file sha256)
    set(dest "${COURSE_THIRD_PARTY}/${file}")
    if(EXISTS "${dest}")
        file(SHA256 "${dest}" have)
        if(have STREQUAL sha256)
            return()
        endif()
    endif()
    message(STATUS "Download ${url}")
    file(DOWNLOAD "${url}" "${dest}" EXPECTED_HASH SHA256=${sha256} STATUS status)
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "Download failed: ${url}: ${status}")
    endif()
endfunction()

# doctest 2.5.3: test cases (MIT)
course_download(
    "https://raw.githubusercontent.com/doctest/doctest/v2.5.3/doctest/doctest.h"
    "doctest/doctest.h"
    "cfd518a3ef90f67e1f3ba514df23fb3627437de1a2feeba78cf5062a40021421")
add_library(doctest INTERFACE)
target_include_directories(doctest SYSTEM INTERFACE "${COURSE_THIRD_PARTY}")

# nlohmann/json 3.12.0: JSON and JSON Lines (MIT)
course_download(
    "https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp"
    "nlohmann/json.hpp"
    "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63")
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE "${COURSE_THIRD_PARTY}")
