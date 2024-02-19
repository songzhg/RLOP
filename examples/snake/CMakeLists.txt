cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project ("snake")

include_directories(${CMAKE_SOURCE_DIR})

add_executable (snake "main.cc")

# SFML
target_link_libraries(snake PRIVATE sfml-system sfml-window sfml-graphics)

# libtorch
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(snake PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET snake PROPERTY CXX_STANDARD 17)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)


