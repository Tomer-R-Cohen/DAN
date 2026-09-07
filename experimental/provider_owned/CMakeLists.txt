cmake_minimum_required(VERSION 3.20)
project(dan_provider_owned_v0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT LLAMA_SOURCE_DIR)
    message(FATAL_ERROR "Set LLAMA_SOURCE_DIR to the patched pinned llama.cpp checkout")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_SOURCE_DIR}" llama.cpp)

add_executable(dan-stage-worker stage_worker.cpp)
target_link_libraries(dan-stage-worker PRIVATE llama)
if(WIN32)
    target_link_libraries(dan-stage-worker PRIVATE ws2_32)
endif()
