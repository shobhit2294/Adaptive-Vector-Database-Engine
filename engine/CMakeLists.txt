cmake_minimum_required(VERSION 3.20)
project(adaptive_vectordb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

add_library(vectordb STATIC
    src/distance.cpp
    src/brute_force.cpp
)

target_include_directories(vectordb PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(vectordb PUBLIC Threads::Threads)

enable_testing()

add_executable(test_distance_and_bf tests/test_distance_and_bf.cpp)
target_link_libraries(test_distance_and_bf PRIVATE vectordb GTest::gtest_main)
add_test(NAME test_distance_and_bf COMMAND test_distance_and_bf)
