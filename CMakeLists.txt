cmake_minimum_required(VERSION 3.20)

project(decentralized_ai_network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(coordinator src/coordinator.cpp)
add_executable(provider src/provider.cpp)
add_executable(managed_provider src/managed_provider.cpp)
add_executable(dan-provider src/gamer_provider.cpp)
add_executable(distributed_model_experiment src/distributed_model_experiment.cpp)
add_library(distributed_runtime STATIC src/distributed_runtime.cpp)
target_sources(distributed_runtime PRIVATE src/persistent_runtime.cpp)
target_include_directories(distributed_runtime PUBLIC include)
target_compile_options(distributed_runtime PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(coordinator PRIVATE distributed_runtime)
target_link_libraries(distributed_model_experiment PRIVATE distributed_runtime)

foreach(target coordinator provider managed_provider dan-provider distributed_model_experiment)
    target_include_directories(${target} PRIVATE include)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()
