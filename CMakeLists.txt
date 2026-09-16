cmake_minimum_required(VERSION 3.20)

project(decentralized_ai_network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(DAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR "" CACHE PATH
    "Patched pinned llama.cpp source used to build the provider-owned stage worker")

set(DAN_ENGINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/engine/include)
set(DAN_LEGACY_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/legacy/include)
set(DAN_PLATFORM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/platform/include)
set(DAN_UI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ui/include)

# The active provider-owned engine (dan-provider-owned-coordinator, dan-stage-worker,
# dan-provider's launcher, and the range-model lib they share) is deliberately never
# given legacy/include on its include path -- not just "doesn't happen to use it," but
# structurally unable to, so a stray #include "control_plane.hpp" et al. fails to
# compile instead of silently working. Legacy targets keep the broader set below.
set(DAN_ENGINE_INCLUDE_DIRS ${DAN_ENGINE_INCLUDE} ${DAN_PLATFORM_INCLUDE})
set(DAN_ENGINE_UI_INCLUDE_DIRS ${DAN_ENGINE_INCLUDE_DIRS} ${DAN_UI_INCLUDE})

# Legacy (the managed/RPC path) and the dashboard/UI code both still draw from all
# four include roots, matching how they were compiled before the engine/legacy/
# platform/ui split.
set(DAN_INCLUDE_DIRS ${DAN_ENGINE_INCLUDE} ${DAN_LEGACY_INCLUDE} ${DAN_PLATFORM_INCLUDE} ${DAN_UI_INCLUDE})

# Shared inference client: drives an already-resolved stage route (one token loop for
# both dan-client and the coordinator). It must never know how a route was discovered.
add_library(provider_owned_client STATIC engine/client.cpp engine/manifest.cpp)
target_include_directories(provider_owned_client PUBLIC ${DAN_ENGINE_INCLUDE})
if(WIN32)
    target_link_libraries(provider_owned_client PUBLIC ws2_32)
endif()

# Stages built from different llama.cpp patches may disagree on hidden states, so routes
# only combine workers reporting the same ABI string. Line endings are stripped so a
# Windows (CRLF) and a Linux (LF) checkout of the same patch agree.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/patches/llama-provider-owned.patch" DAN_PATCH_TEXT)
string(REPLACE "\r" "" DAN_PATCH_TEXT "${DAN_PATCH_TEXT}")
string(SHA256 DAN_PATCH_SHA256 "${DAN_PATCH_TEXT}")
string(SUBSTRING "${DAN_PATCH_SHA256}" 0 16 DAN_PATCH_ID)
target_compile_definitions(provider_owned_client PUBLIC
    DAN_RUNTIME_ABI="dan-stage-v1/f32le/${DAN_PATCH_ID}")

add_executable(dan-client engine/dan_client.cpp)

add_executable(dan-provider-owned-coordinator engine/coordinator.cpp)
target_include_directories(dan-provider-owned-coordinator PRIVATE ${DAN_ENGINE_INCLUDE_DIRS})
target_link_libraries(dan-provider-owned-coordinator PRIVATE provider_owned_client)
if(WIN32)
    target_link_libraries(dan-provider-owned-coordinator PRIVATE ws2_32)
endif()

if(DAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR)
    if(NOT EXISTS "${DAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "DAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR is not a llama.cpp checkout")
    endif()
    enable_language(C)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(CMAKE_CXX_STANDARD 17)
    add_subdirectory("${DAN_PROVIDER_OWNED_LLAMA_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/provider-owned-llama")
    set(CMAKE_CXX_STANDARD 23)
    add_executable(dan-stage-worker engine/stage_worker.cpp)
    target_include_directories(dan-stage-worker PRIVATE ${DAN_ENGINE_UI_INCLUDE_DIRS})
    target_link_libraries(dan-stage-worker PRIVATE llama)
    target_compile_definitions(dan-provider-owned-coordinator PRIVATE DAN_HAS_LLAMA)
    target_link_libraries(dan-provider-owned-coordinator PRIVATE llama)
    if(WIN32)
        target_link_libraries(dan-stage-worker PRIVATE ws2_32)
    endif()
endif()

if(WIN32)
    add_library(dan_platform STATIC platform/platform_windows.cpp)
    add_library(distributed_runtime STATIC
        legacy/distributed_runtime_windows.cpp legacy/persistent_runtime_windows.cpp)
    add_executable(dan-coordinator legacy/coordinator.cpp)
else()
    add_library(dan_platform STATIC platform/platform_posix.cpp)
endif()
# dan_platform is linked PUBLIC by targets on both sides of the split, so its own
# PUBLIC include set must stay minimal -- platform/include only -- or it would leak
# legacy/ui paths into the engine transitively regardless of the engine targets'
# own narrower target_include_directories() calls above.
target_include_directories(dan_platform PUBLIC ${DAN_PLATFORM_INCLUDE})

add_library(provider_owned_range_model STATIC engine/range_model.cpp)
target_include_directories(provider_owned_range_model PUBLIC ${DAN_ENGINE_INCLUDE_DIRS})
target_link_libraries(provider_owned_range_model PUBLIC dan_platform)
if(TARGET dan-stage-worker)
    target_link_libraries(dan-stage-worker PRIVATE provider_owned_range_model provider_ui)
endif()
target_link_libraries(dan-provider-owned-coordinator PRIVATE provider_owned_range_model)

# Stage placement (shared by the coordinator, the decentralized client and workers), and
# client-side dynamic placement: plan, reserve, assign.
add_library(provider_owned_planner STATIC engine/planner.cpp)
target_link_libraries(provider_owned_planner PUBLIC provider_owned_range_model)
add_library(provider_owned_placement STATIC engine/placement.cpp)
target_link_libraries(provider_owned_placement PUBLIC provider_owned_client provider_owned_planner)
target_link_libraries(dan-client PRIVATE provider_owned_placement)
target_link_libraries(dan-provider-owned-coordinator PRIVATE provider_owned_planner)
if(TARGET dan-stage-worker)
    target_link_libraries(dan-stage-worker PRIVATE provider_owned_planner provider_owned_client)
endif()

add_library(admin_dashboard STATIC ui/admin_dashboard.cpp)
target_include_directories(admin_dashboard PUBLIC ${DAN_INCLUDE_DIRS})
target_link_libraries(admin_dashboard PUBLIC dan_platform)
add_library(provider_ui STATIC ui/provider_ui.cpp)
# provider_ui is shared by legacy (managed_provider) and the engine (dan-stage-worker);
# like dan_platform, its PUBLIC includes stay narrow (ui + platform, no legacy) so
# linking it never reintroduces legacy/include into the engine.
target_include_directories(provider_ui PUBLIC ${DAN_UI_INCLUDE} ${DAN_PLATFORM_INCLUDE})
target_link_libraries(provider_ui PUBLIC dan_platform)

add_executable(managed_provider legacy/managed_provider.cpp)
add_executable(dan-provider engine/provider_launcher.cpp)
target_link_libraries(managed_provider PRIVATE provider_ui dan_platform)
target_include_directories(dan-provider PRIVATE ${DAN_PLATFORM_INCLUDE})
target_link_libraries(dan-provider PRIVATE dan_platform)
if(WIN32)
    target_link_libraries(dan-coordinator PRIVATE distributed_runtime admin_dashboard dan_platform)
endif()

if(WIN32)
    target_link_libraries(dan_platform PUBLIC ws2_32 bcrypt shell32)
    target_compile_definitions(dan_platform PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX)
else()
    add_executable(coordinator legacy/coordinator.cpp)
    add_executable(provider legacy/provider.cpp)
    add_executable(distributed_model_experiment legacy/distributed_model_experiment.cpp)
    add_library(distributed_runtime STATIC legacy/distributed_runtime.cpp legacy/persistent_runtime.cpp)
    target_include_directories(distributed_runtime PUBLIC ${DAN_INCLUDE_DIRS})
    target_link_libraries(coordinator PRIVATE distributed_runtime admin_dashboard dan_platform)
    target_link_libraries(provider PRIVATE dan_platform)
    target_link_libraries(distributed_model_experiment PRIVATE distributed_runtime)
endif()

# Targets whose include directories are already set precisely above (the engine,
# dan_platform, provider_ui) are excluded from this loop's broad include grant --
# only their compiler warning flags come from here. Everything else (legacy and the
# admin dashboard) still gets the full four-directory set.
set(dan_engine_scoped_targets dan-provider-owned-coordinator dan-provider dan_platform
    provider_owned_range_model provider_ui provider_owned_client dan-client
    provider_owned_planner provider_owned_placement)
if(TARGET dan-stage-worker)
    list(APPEND dan_engine_scoped_targets dan-stage-worker)
endif()

set(dan_targets dan_platform admin_dashboard provider_ui distributed_runtime managed_provider
    dan-provider dan-provider-owned-coordinator provider_owned_range_model
    provider_owned_client dan-client provider_owned_planner provider_owned_placement)
if(TARGET dan-stage-worker)
    list(APPEND dan_targets dan-stage-worker)
endif()
if(WIN32)
    list(APPEND dan_targets dan-coordinator)
endif()
if(NOT WIN32)
    list(APPEND dan_targets distributed_runtime coordinator provider distributed_model_experiment)
endif()
foreach(target IN LISTS dan_targets)
    if(NOT target IN_LIST dan_engine_scoped_targets)
        target_include_directories(${target} PRIVATE ${DAN_INCLUDE_DIRS})
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

include(CTest)
if(BUILD_TESTING)
    add_executable(provider_owned_protocol_test engine/tests/protocol_test.cpp)
    target_include_directories(provider_owned_protocol_test PRIVATE ${DAN_ENGINE_INCLUDE_DIRS})
    if(WIN32)
        target_link_libraries(provider_owned_protocol_test PRIVATE ws2_32)
    endif()
    add_test(NAME provider_owned_protocol_test COMMAND provider_owned_protocol_test)
    add_executable(provider_owned_range_model_test engine/tests/range_model_test.cpp)
    target_link_libraries(provider_owned_range_model_test PRIVATE provider_owned_range_model)
    add_test(NAME provider_owned_range_model_test COMMAND provider_owned_range_model_test)
    add_executable(provider_owned_formation_test engine/tests/formation_test.cpp)
    target_link_libraries(provider_owned_formation_test PRIVATE provider_owned_planner)
    add_test(NAME provider_owned_formation_test COMMAND provider_owned_formation_test)
    add_executable(provider_owned_client_test engine/tests/client_test.cpp)
    target_link_libraries(provider_owned_client_test PRIVATE provider_owned_client)
    add_test(NAME provider_owned_client_test COMMAND provider_owned_client_test)
    add_executable(provider_owned_route_test engine/tests/route_test.cpp)
    target_link_libraries(provider_owned_route_test PRIVATE provider_owned_client)
    add_test(NAME provider_owned_route_test COMMAND provider_owned_route_test)
    add_executable(provider_owned_lease_test engine/tests/lease_test.cpp)
    target_link_libraries(provider_owned_lease_test PRIVATE provider_owned_planner)
    target_include_directories(provider_owned_lease_test PRIVATE ${DAN_ENGINE_INCLUDE_DIRS})
    if(WIN32)
        target_link_libraries(provider_owned_lease_test PRIVATE ws2_32)
    endif()
    add_test(NAME provider_owned_lease_test COMMAND provider_owned_lease_test)
    add_executable(provider_owned_placement_test engine/tests/placement_test.cpp)
    target_link_libraries(provider_owned_placement_test PRIVATE provider_owned_placement)
    add_test(NAME provider_owned_placement_test COMMAND provider_owned_placement_test)
    add_executable(provider_owned_concurrency_client
        engine/tests/concurrency_client.cpp)
    target_include_directories(provider_owned_concurrency_client PRIVATE ${DAN_ENGINE_INCLUDE_DIRS})
    if(WIN32)
        target_link_libraries(provider_owned_concurrency_client PRIVATE ws2_32)
    endif()
    add_executable(provider_ui_test ui/tests/provider_ui_test.cpp)
    target_link_libraries(provider_ui_test PRIVATE provider_ui)
    add_test(NAME provider_ui_test COMMAND provider_ui_test)
    add_executable(admin_dashboard_test ui/tests/admin_dashboard_test.cpp)
    target_include_directories(admin_dashboard_test PRIVATE ${DAN_LEGACY_INCLUDE})
    target_link_libraries(admin_dashboard_test PRIVATE admin_dashboard)
    add_test(NAME admin_dashboard_test COMMAND admin_dashboard_test)
    add_executable(platform_test platform/tests/platform_test.cpp)
    target_include_directories(platform_test PRIVATE ${DAN_LEGACY_INCLUDE})
    target_link_libraries(platform_test PRIVATE dan_platform)
    add_test(NAME platform_test COMMAND platform_test)
    if(WIN32)
        add_executable(single_instance_test platform/tests/single_instance_test.cpp)
        target_link_libraries(single_instance_test PRIVATE dan_platform)
        add_test(NAME single_instance_test COMMAND single_instance_test)
        if(TARGET dan-stage-worker)
            add_test(NAME windows_provider_check COMMAND dan-provider
                --coordinator 127.0.0.1:50200 --advertise-host 127.0.0.1
                --stage-worker $<TARGET_FILE:dan-stage-worker>
                --nvidia-smi $<TARGET_FILE:platform_test>
                --state-dir ${CMAKE_CURRENT_BINARY_DIR}/windows-test-state
                --cache-dir ${CMAKE_CURRENT_BINARY_DIR}/windows-test-cache --check)
        endif()
        add_executable(windows_coordinator_test legacy/tests/windows_coordinator_test.cpp)
        target_include_directories(windows_coordinator_test PRIVATE ${DAN_LEGACY_INCLUDE})
        target_link_libraries(windows_coordinator_test PRIVATE dan_platform)
        target_compile_options(windows_coordinator_test PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/W4>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wpedantic>)
        add_test(NAME windows_coordinator_test COMMAND windows_coordinator_test
            $<TARGET_FILE:dan-coordinator>)
    endif()
endif()

if(WIN32)
    install(TARGETS dan-provider RUNTIME DESTINATION .)
    install(TARGETS managed_provider RUNTIME DESTINATION runtime)
    install(FILES scripts/setup_provider.ps1 DESTINATION runtime RENAME setup-provider.ps1)
    install(FILES docs/reference/operations/friends-testnet-windows.md DESTINATION . RENAME README.txt)
endif()
