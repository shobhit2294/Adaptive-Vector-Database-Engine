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
find_package(nlohmann_json REQUIRED)

add_library(vectordb STATIC
    src/distance.cpp
    src/brute_force.cpp
    src/hnsw.cpp
    src/wal.cpp
    src/mmap_storage.cpp
    src/pq.cpp
    src/metrics.cpp
    src/segment.cpp
    src/wahi_controller.cpp
    src/database.cpp
)

target_include_directories(vectordb PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(vectordb PUBLIC Threads::Threads nlohmann_json::nlohmann_json)

add_executable(vectordb_server src/server_main.cpp)
target_link_libraries(vectordb_server PRIVATE vectordb)

add_executable(vectordb_bench bench/engine_bench.cpp)
target_link_libraries(vectordb_bench PRIVATE vectordb)

enable_testing()

foreach(t IN ITEMS test_distance_and_bf test_hnsw test_wal_and_mmap test_pq test_wahi test_concurrency)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE vectordb GTest::gtest_main)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
