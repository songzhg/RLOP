﻿cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project ("lunar_lander")

include_directories(${CMAKE_SOURCE_DIR})

add_executable (lunar_lander "main.cc")

# libtorch gpu
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lunar_lander PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET lunar_lander PROPERTY CXX_STANDARD 17)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)

# pybind11
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(lunar_lander PRIVATE "-fvisibility=hidden")
endif()
target_link_libraries(lunar_lander PRIVATE pybind11::module pybind11::embed)


