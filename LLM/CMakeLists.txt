# Block 1: one library with the code of all lessons; one test program for each lesson.
# The lesson tests have the labels LLM and L01 .. L07: ctest --preset clang -L L03
add_library(llm STATIC
    L01_tokenizer/tokenizer.cpp
    L02_token_data/token_data.cpp
    L03_rmsnorm_swiglu/rmsnorm_swiglu.cpp
    L04_rope/rope.cpp
    L05_attention/attention.cpp
    L06_model/model.cpp
    L07_generation/generate.cpp)
target_include_directories(llm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
course_target(llm)
target_link_libraries(llm PUBLIC course nlohmann_json)

course_test(L01_tokenizer      SOURCES L01_tokenizer/tokenizer_test.cpp           LIBS llm LABELS "LLM;L01")
course_test(L02_token_data     SOURCES L02_token_data/token_data_test.cpp         LIBS llm LABELS "LLM;L02")
course_test(L03_rmsnorm_swiglu SOURCES L03_rmsnorm_swiglu/rmsnorm_swiglu_test.cpp LIBS llm LABELS "LLM;L03")
course_test(L04_rope           SOURCES L04_rope/rope_test.cpp                     LIBS llm LABELS "LLM;L04")
course_test(L05_attention      SOURCES L05_attention/attention_test.cpp           LIBS llm LABELS "LLM;L05")
course_test(L06_model          SOURCES L06_model/model_test.cpp                   LIBS llm LABELS "LLM;L06")
course_test(L07_generation     SOURCES L07_generation/generate_test.cpp           LIBS llm LABELS "LLM;L07")

# Weights from C++ to Python: the L06 test writes cpp_model.pt; Python checks it.
set_tests_properties(L06_model PROPERTIES FIXTURES_SETUP L06_cpp_checkpoint)
add_test(NAME L06_model_py COMMAND "${COURSE_PYTHON}" -m LLM.L06_model.golden WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(L06_model_py PROPERTIES LABELS "LLM;L06" FIXTURES_REQUIRED "goldens;L06_cpp_checkpoint")
