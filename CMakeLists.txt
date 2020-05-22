project(ulms)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

file(GLOB ULMS_SOURCE ${PROJECT_SOURCE_DIR}/*.h ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HOOK_SOURCE ${PROJECT_SOURCE_DIR}/hook_api/*.h ${PROJECT_SOURCE_DIR}/hook_api/*.cpp)

list(REMOVE_ITEM ULMS_SOURCE ${PROJECT_SOURCE_DIR}/hook.cpp)

add_library(hook OBJECT ${PROJECT_SOURCE_DIR}/hook.cpp ${PROJECT_SOURCE_DIR}/hook.h)
add_dependencies(hook ${HOOK_SOURCE})

add_library(ulms SHARED $<TARGET_OBJECTS:hook> ${ULMS_SOURCE})
set_target_properties(ulms PROPERTIES PREFIX "")
