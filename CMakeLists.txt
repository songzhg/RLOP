cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project ("rlop")

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")


# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/pybind11)

add_subdirectory(examples/vrp)
# add_subdirectory(examples/snake)
# add_subdirectory(examples/multi_armed_bandit)
# add_subdirectory(examples/lunar_lander)
# add_subdirectory(examples/continuous_lunar_lander)
# add_subdirectory(examples/connect4)
